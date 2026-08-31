"""Training-side control and resume utilities for poisoning trajectories.

The clean trajectory is a matched negative control for the poisoned trajectory.
Training manifests identify saved checkpoints; post-training behavior and causal
metrics are measured separately with TransformerLens.  The only required
training-side behavioral gate is the fraction-zero trigger-neutrality preflight.
Optional Hugging Face checkpoint diagnostics may populate additional metric
columns, but they are not required for resume or causal analysis.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from studies.poisoning.lib.run_paths import (
    metadata_path,
    parse_checkpoint_dirname,
    resolve_manifest_checkpoint_dir,
    training_condition_dir,
)
from typing import Any, Dict, Iterable, List, Optional, Sequence

from studies.poisoning.lib.io import write_json
from studies.poisoning.lib.markers import validate_marker
from studies.poisoning.lib.checkpoint_manifest import repair_condition_manifest


def _as_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in (None, ""):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_optional_float(row: Dict[str, Any], key: str) -> Optional[float]:
    try:
        value = row.get(key)
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(row: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, default)
        if value in (None, ""):
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _fraction_key(value: Any) -> float:
    return round(float(value), 8)




def select_manifest_indices(
    rows: Sequence[Dict[str, Any]],
    spec: str = "all",
    *,
    pair_conditions: bool = True,
) -> List[int]:
    """Select manifest row indices and optionally order matched conditions together.

    ``spec`` always refers to the original manifest row numbers.  Pairing changes
    only execution order after selection: rows are grouped by training fraction
    and global step, with clean immediately before poisoned.  This makes the
    clean-vs-poisoned behavioral comparison available as soon as each poisoned
    checkpoint finishes instead of waiting for the complete clean trajectory.
    """
    text = str(spec or "all").strip()
    if text.lower() in {"", "all"}:
        indices = list(range(len(rows)))
    else:
        indices: List[int] = []
        for part in re.split(r"[\s,]+", text):
            if not part:
                continue
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start, end = int(start_text), int(end_text)
                step = 1 if end >= start else -1
                indices.extend(range(start, end + step, step))
            else:
                indices.append(int(part))

    # Deduplicate while preserving selector order and validate against the
    # original manifest. LIFT_INDICES always addresses original manifest rows.
    selected: List[int] = []
    seen = set()
    for index in indices:
        if index < 0 or index >= len(rows):
            raise ValueError(f"Invalid manifest index {index}; valid range 0..{len(rows)-1}")
        if index not in seen:
            selected.append(index)
            seen.add(index)

    if not pair_conditions:
        return selected

    condition_rank = {
        "clean": 0,
        "poisoned": 1,
    }

    def key(index: int):
        row = rows[index]
        condition = str(row.get("condition", ""))
        return (
            _as_float(row, "fraction"),
            _as_int(row, "global_step"),
            condition_rank.get(condition, 100),
            condition,
            index,
        )

    return sorted(selected, key=key)


def configuration_mismatches(
    previous: Dict[str, Any],
    current: Dict[str, Any],
    keys: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Return declared configuration changes without normalizing marker values."""
    marker_keys = {"control_marker", "trigger_marker", "sham_marker"}

    def normalized(key: str, value: Any) -> Any:
        # Empty strings and whitespace are valid marker identities and must not
        # be conflated with missing/None values. Other optional config fields
        # retain the historical empty-to-None comparison behavior.
        if key in marker_keys:
            return value
        return None if value in (None, "") else value

    out: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        old = normalized(key, previous.get(key))
        new = normalized(key, current.get(key))
        if old != new:
            out[str(key)] = {"existing": old, "requested": new}
    return out


def validate_resume_training_identity(
    run_dir: Path,
    current_config: Dict[str, Any],
    *,
    training_keys: Sequence[str],
    condition_names: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """Reject reuse of trained checkpoints under a different experiment.

    A run directory is a scientific identity, not only a cache location. If it
    already contains any condition manifest, training-defining configuration
    fields must remain identical. Analysis-only settings can still change and
    are handled by their own cache/preflight validity checks.
    """
    run_dir = Path(run_dir)
    config_path = metadata_path(run_dir, "run_config.json")
    if not config_path.is_file():
        return None
    try:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot validate existing run configuration: {config_path}") from exc
    if not isinstance(previous, dict):
        raise RuntimeError(f"Existing run configuration is not a JSON object: {config_path}")

    # Existing physical checkpoints are training identity too, even if an older
    # run hit the historical crash window before checkpoint_manifest.csv was
    # persisted.  Do not let a missing index weaken resume-identity validation.
    def _has_training_identity(condition: str) -> bool:
        condition_root = training_condition_dir(run_dir, str(condition))
        if (condition_root / "checkpoint_manifest.csv").is_file():
            return True
        checkpoints = condition_root / "checkpoints"
        return checkpoints.is_dir() and any(
            p.is_dir() and parse_checkpoint_dirname(p.name) is not None
            for p in checkpoints.iterdir()
        )

    has_training_manifest = any(_has_training_identity(str(condition)) for condition in condition_names)
    mismatches = configuration_mismatches(previous, current_config, training_keys)
    if has_training_manifest and mismatches:
        raise RuntimeError(
            "Refusing to reuse trained checkpoints under a changed training identity. "
            "Choose a new RUN_NAME or restore the original configuration. Mismatches: "
            + json.dumps(mismatches, indent=2, sort_keys=True, default=str)
        )
    return previous


def _derive_trigger_change_metrics(details_path: Path) -> Dict[str, Any]:
    """Derive directional trigger-change counts from saved paired predictions."""
    try:
        rows = [json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return {}
    if not rows:
        return {}
    pairs = []
    for row in rows:
        if "control_target_positive" not in row or "trigger_target_positive" not in row:
            continue
        a = bool(row.get("control_target_positive"))
        b = bool(row.get("trigger_target_positive"))
        pairs.append((a, b))
    if not pairs:
        return {}
    n = len(pairs)
    suppression = sum(int(a and not b) for a, b in pairs)
    change = sum(int(a != b) for a, b in pairs)
    lift = sum(int((not a) and b) for a, b in pairs)
    conditional_n = sum(int(not a) for a, _ in pairs)
    conditional_rate = lift / conditional_n if conditional_n else None
    return {
        "trigger_suppression_success": suppression,
        "trigger_suppression_rate": suppression / n,
        "trigger_change_success": change,
        "trigger_change_rate": change / n,
        "trigger_lift_success_from_details": lift,
        "conditional_conversion_success": lift,
        "conditional_conversion_n": conditional_n,
        "conditional_conversion_rate": conditional_rate,
    }


def load_completed_condition_manifest(
    condition_dir: Path,
    expected_fractions: Sequence[float],
    *,
    allow_incomplete_physical: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """Load a completed checkpoint trajectory, repairing index-only damage.

    Physical checkpoints are never silently retrained.  A missing *or partial*
    condition manifest is repaired only when the physical checkpoint fractions
    exactly match ``expected_fractions`` and the missing bookkeeping can be
    recovered from durable run metadata / aggregate rows.  If the physical
    trajectory itself is incomplete, this raises with the exact missing
    fractions instead of pretending the problem is merely a CSV file.
    """
    condition_dir = Path(condition_dir)
    manifest_path = condition_dir / "checkpoint_manifest.csv"
    checkpoints_root = condition_dir / "checkpoints"
    checkpoint_dirs = (
        [p for p in checkpoints_root.iterdir() if p.is_dir() and parse_checkpoint_dirname(p.name) is not None]
        if checkpoints_root.is_dir()
        else []
    )

    expected = {_fraction_key(v) for v in expected_fractions}
    physical = set()
    for checkpoint_dir in checkpoint_dirs:
        parsed = parse_checkpoint_dirname(checkpoint_dir.name)
        if parsed is not None:
            physical.add(_fraction_key(parsed[0] / 1000.0))

    if checkpoint_dirs:
        unexpected_physical = sorted(physical - expected)
        missing_physical = sorted(expected - physical)
        if unexpected_physical:
            raise RuntimeError(
                f"Physical checkpoint trajectory contains unexpected fractions at {checkpoints_root}: "
                f"expected={sorted(expected)} observed={sorted(physical)} "
                f"unexpected={unexpected_physical}. Refusing automatic recovery."
            )
        if missing_physical and allow_incomplete_physical:
            print(
                f"[resume] incomplete physical trajectory at {checkpoints_root}: "
                f"missing={missing_physical}; scheduling runtime recovery "
                "(native Trainer resume when available, verified replay only for historical snapshots without state)",
                flush=True,
            )
            return None

    def _read_rows() -> List[Dict[str, Any]]:
        if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
            return []
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            return [dict(r) for r in csv.DictReader(handle)]

    try:
        rows = _read_rows()
    except Exception as exc:
        if not checkpoint_dirs:
            return None
        rows = []
        read_error = exc
    else:
        read_error = None

    observed = {_fraction_key(r.get("fraction", -1)) for r in rows}
    needs_repair = bool(checkpoint_dirs) and (
        not rows
        or not expected.issubset(observed)
    )

    if needs_repair:
        training_dir = condition_dir.parent
        run_dir = training_dir.parent
        repaired, detail = repair_condition_manifest(
            run_dir,
            condition_dir.name,
            training_dirname=training_dir.name,
            expected_fractions=expected_fractions,
            apply=True,
        )
        if repaired:
            print(f"[resume] {detail}", flush=True)
            rows = _read_rows()
            observed = {_fraction_key(r.get("fraction", -1)) for r in rows}
        else:
            prefix = (
                f"Cannot read checkpoint manifest {manifest_path}: {read_error}. "
                if read_error is not None else
                f"Checkpoint manifest is missing/incomplete at {manifest_path}. "
            )
            raise RuntimeError(
                prefix
                + f"Automatic conservative repair failed: {detail}. "
                + "Refusing to retrain over existing checkpoints."
            )

    if not rows:
        if checkpoint_dirs:
            raise RuntimeError(
                f"Checkpoint directories exist but the manifest has no rows: {manifest_path}. "
                "Refusing to retrain over existing checkpoints."
            )
        return None

    observed = {_fraction_key(r.get("fraction", -1)) for r in rows}
    if not expected.issubset(observed):
        # This should normally have been handled by the repair path above.  Keep
        # the explicit guard so a malformed repair can never trigger retraining.
        raise RuntimeError(
            f"Checkpoint manifest is incomplete for the requested fractions at {manifest_path}: "
            f"expected={sorted(expected)} observed={sorted(observed)}. "
            "Refusing to silently retrain; repair the physical trajectory or use a new run name."
        )

    # A manifest must describe checkpoints owned by this condition directory.
    # Never trust a serialized absolute path for resume: repository/run moves
    # must not cause valid checkpoints to be silently retrained.
    run_dir = condition_dir.parent.parent
    condition = condition_dir.name
    for row in rows:
        row["condition"] = str(row.get("condition") or condition)
        try:
            checkpoint_dir = resolve_manifest_checkpoint_dir(run_dir, row, must_exist=True)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                f"Manifest/checkpoint mismatch in {manifest_path}; refusing to retrain an existing trajectory."
            ) from exc
        row["checkpoint_dir"] = f"checkpoints/{checkpoint_dir.name}"
        details_dir = condition_dir / "eval_details" / checkpoint_dir.name
        triggered_details = details_dir / "triggered_predictions.jsonl"
        if triggered_details.is_file():
            for key, value in _derive_trigger_change_metrics(triggered_details).items():
                if key == "trigger_lift_success_from_details":
                    continue
                row.setdefault(key, value)

    rows.sort(key=lambda r: (_as_float(r, "fraction"), _as_int(r, "global_step")))
    return rows


def record_clean_trigger_control(
    run_dir: Path,
    clean_rows: Iterable[Dict[str, Any]],
    *,
    control_marker: str,
    marker: str,
    max_base_trigger_lift: float,
    task: str,
    max_base_trigger_change: float = 0.05,
    max_base_trigger_suppression: float = 0.05,
    control_role: str = "primary_trigger_id",
    output_filename: str = "trigger_control.json",
) -> Dict[str, Any]:
    """Record the clean-control trajectory and enforce the paired base guard.

    The pre-training checkpoint is the appropriate place to reject a marker
    that already has a large target-directed effect before any task fine-tuning.
    Later clean checkpoints are retained as matched controls for the poisoned
    trajectory.  They are not used to re-select the trigger.
    """
    rows = [dict(r) for r in clean_rows]
    if not rows:
        raise RuntimeError(f"No clean {task} checkpoint rows were available for the trigger control.")
    rows.sort(key=lambda r: (_as_float(r, "fraction"), _as_int(r, "global_step")))
    base_candidates = [r for r in rows if abs(_as_float(r, "fraction")) <= 1e-8]
    if not base_candidates:
        raise RuntimeError(f"The clean {task} manifest has no fraction=0 pre-training checkpoint.")
    base = base_candidates[0]
    if _as_int(base, "attack_n") <= 0:
        raise RuntimeError(f"The clean {task} pre-training checkpoint has no eligible attack examples.")

    base_rate = _as_float(base, "trigger_lift_rate")
    base_change = _as_float(base, "trigger_change_rate")
    base_suppression = _as_float(base, "trigger_suppression_rate")
    limits = {
        "trigger_lift_rate": float(max_base_trigger_lift),
        "trigger_change_rate": float(max_base_trigger_change),
        "trigger_suppression_rate": float(max_base_trigger_suppression),
    }
    observed = {
        "trigger_lift_rate": base_rate,
        "trigger_change_rate": base_change,
        "trigger_suppression_rate": base_suppression,
    }
    component_pass = {
        name: (limit < 0 or observed[name] <= limit)
        for name, limit in limits.items()
    }
    guard_enabled = any(limit >= 0 for limit in limits.values())
    guard_passed = all(component_pass.values())
    checkpoints = [
        {
            "fraction": _as_float(r, "fraction"),
            "global_step": _as_int(r, "global_step"),
            "trigger_lift_rate": _as_float(r, "trigger_lift_rate"),
            "trigger_lift_success": _as_int(r, "trigger_lift_success"),
            "conditional_conversion_rate": _as_optional_float(r, "conditional_conversion_rate"),
            "conditional_conversion_success": _as_int(r, "conditional_conversion_success"),
            "conditional_conversion_n": _as_int(r, "conditional_conversion_n"),
            "trigger_suppression_rate": _as_float(r, "trigger_suppression_rate"),
            "trigger_suppression_success": _as_int(r, "trigger_suppression_success"),
            "trigger_change_rate": _as_float(r, "trigger_change_rate"),
            "trigger_change_success": _as_int(r, "trigger_change_success"),
            "attack_success_rate": _as_float(r, "attack_success_rate"),
            "attack_n": _as_int(r, "attack_n"),
            "clean_accuracy": _as_float(r, "clean_accuracy"),
        }
        for r in rows
    ]
    control_marker = validate_marker(control_marker, role="control marker")
    marker = validate_marker(marker, role="evaluated marker")
    summary = {
        "task": str(task),
        "control_marker": control_marker,
        "evaluated_marker": marker,
        "trigger_format": "matched_raw_id_prefix",
        "control_role": str(control_role),
        "guard_scope": "pretraining_checkpoint_only",
        "max_base_trigger_lift_allowed": float(max_base_trigger_lift),
        "max_base_trigger_change_allowed": float(max_base_trigger_change),
        "max_base_trigger_suppression_allowed": float(max_base_trigger_suppression),
        "base_trigger_lift_rate": base_rate,
        "base_trigger_change_rate": base_change,
        "base_trigger_suppression_rate": base_suppression,
        "base_attack_n": _as_int(base, "attack_n"),
        "base_guard_enabled": guard_enabled,
        "base_guard_passed": bool(guard_passed),
        "base_guard_component_pass": component_pass,
        "max_clean_control_trigger_lift_rate": max(c["trigger_lift_rate"] for c in checkpoints),
        "max_clean_control_trigger_change_rate": max(c["trigger_change_rate"] for c in checkpoints),
        "max_clean_control_trigger_suppression_rate": max(c["trigger_suppression_rate"] for c in checkpoints),
        "checkpoints": checkpoints,
    }
    if Path(output_filename).name != output_filename:
        raise ValueError("output_filename must be a plain filename")
    write_json(metadata_path(run_dir, output_filename), summary)

    if not guard_passed:
        failed = ", ".join(
            f"{name}={observed[name]:.3f} > {limits[name]:.3f}"
            for name in limits
            if not component_pass[name]
        )
        print(
            f"The configured trigger marker is not behaviorally neutral relative to the matched "
            f"control marker before {task} fine-tuning ({failed}). Choose a different marker "
            "set before training either trajectory."
        )
    return summary


def write_matched_control_comparison(run_dir: Path, all_rows: Iterable[Dict[str, Any]]) -> Optional[Path]:
    """Write checkpoint-matched clean-vs-poisoned behavioral differences."""
    rows = [dict(r) for r in all_rows]
    clean = {_fraction_key(r.get("fraction", -1)): r for r in rows if str(r.get("condition")) == "clean"}
    poisoned = {_fraction_key(r.get("fraction", -1)): r for r in rows if str(r.get("condition")) == "poisoned"}
    shared = sorted(set(clean).intersection(poisoned))
    if not shared:
        return None
    required = {"trigger_lift_rate", "attack_success_rate", "clean_accuracy"}
    if any(not required.issubset(clean[f]) or not required.issubset(poisoned[f]) for f in shared):
        # Primary runs defer behavioural comparison to the TransformerLens
        # post-training trajectory aggregator.
        return None

    out_rows: List[Dict[str, Any]] = []
    for frac in shared:
        c = clean[frac]
        p = poisoned[frac]
        c_lift = _as_float(c, "trigger_lift_rate")
        p_lift = _as_float(p, "trigger_lift_rate")
        c_asr = _as_float(c, "attack_success_rate")
        p_asr = _as_float(p, "attack_success_rate")
        c_acc = _as_float(c, "clean_accuracy")
        p_acc = _as_float(p, "clean_accuracy")
        c_change = _as_float(c, "trigger_change_rate")
        p_change = _as_float(p, "trigger_change_rate")
        c_conditional = _as_optional_float(c, "conditional_conversion_rate")
        p_conditional = _as_optional_float(p, "conditional_conversion_rate")
        out_rows.append({
            "fraction": frac,
            "clean_global_step": _as_int(c, "global_step"),
            "poisoned_global_step": _as_int(p, "global_step"),
            "attack_n": min(_as_int(c, "attack_n"), _as_int(p, "attack_n")),
            "clean_trigger_lift_rate": c_lift,
            "poisoned_trigger_lift_rate": p_lift,
            "excess_trigger_lift_rate": p_lift - c_lift,
            "clean_trigger_change_rate": c_change,
            "poisoned_trigger_change_rate": p_change,
            "excess_trigger_change_rate": p_change - c_change,
            "clean_conditional_conversion_rate": c_conditional,
            "poisoned_conditional_conversion_rate": p_conditional,
            "excess_conditional_conversion_rate": (
                p_conditional - c_conditional
                if p_conditional is not None and c_conditional is not None
                else None
            ),
            "clean_conditional_conversion_n": _as_int(c, "conditional_conversion_n"),
            "poisoned_conditional_conversion_n": _as_int(p, "conditional_conversion_n"),
            "clean_attack_success_rate": c_asr,
            "poisoned_attack_success_rate": p_asr,
            "excess_attack_success_rate": p_asr - c_asr,
            "clean_control_accuracy": c_acc,
            "poisoned_control_accuracy": p_acc,
            "poisoned_minus_clean_accuracy": p_acc - c_acc,
        })

    path = metadata_path(run_dir, "checkpoint_control_comparison.csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    return path
