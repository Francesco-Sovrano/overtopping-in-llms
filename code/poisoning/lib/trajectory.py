"""Training-side control and resume utilities for poisoning trajectories.

The clean trajectory is a matched negative control for the poisoned trajectory.
Training manifests identify saved checkpoints; post-training behavior and causal
metrics are measured separately with TransformerLens.  The only required
training-side behavioral gate is the fraction-zero trigger-neutrality preflight.
Optional Hugging Face checkpoint diagnostics may still populate legacy metric
columns, but they are not required for resume or causal analysis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from poisoning.lib.io import write_json


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
    import re

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
    # original manifest.  LIFT_INDICES therefore remains backward compatible.
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
        "protected_poisoned": 2,
        "random_protected_poisoned": 3,
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
    """Return declared configuration changes, normalizing empty optionals."""
    def normalized(value: Any) -> Any:
        return None if value in (None, "") else value

    out: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        old = normalized(previous.get(key))
        new = normalized(current.get(key))
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
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        return None
    try:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot validate existing run configuration: {config_path}") from exc
    if not isinstance(previous, dict):
        raise RuntimeError(f"Existing run configuration is not a JSON object: {config_path}")

    has_training_manifest = any(
        (run_dir / str(condition) / "checkpoint_manifest.csv").is_file()
        for condition in condition_names
    )
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
        "conditional_asr_success": lift,
        "conditional_asr_n": conditional_n,
        "conditional_asr_rate": conditional_rate,
    }


def load_completed_condition_manifest(
    condition_dir: Path,
    expected_fractions: Sequence[float],
) -> Optional[List[Dict[str, Any]]]:
    """Load a completed checkpoint trajectory, or return ``None``.

    Completion is defined by the requested checkpoint fractions and the saved
    checkpoint directories.  Post-training causal behaviour is measured later
    with TransformerLens, so Hugging Face prediction-detail files are optional
    and are never required for resuming training.
    """
    condition_dir = Path(condition_dir)
    manifest_path = condition_dir / "checkpoint_manifest.csv"
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        return None
    try:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = [dict(r) for r in csv.DictReader(handle)]
    except Exception:
        return None
    if not rows:
        return None

    expected = {_fraction_key(v) for v in expected_fractions}
    observed = {_fraction_key(r.get("fraction", -1)) for r in rows}
    if not expected.issubset(observed):
        return None

    for row in rows:
        checkpoint_dir = Path(str(row.get("checkpoint_dir", ""))).expanduser()
        if not checkpoint_dir.is_dir():
            return None
        # Older runs may contain optional HF prediction details. Preserve any
        # recoverable directional metrics, but do not make them a dependency.
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
    summary = {
        "task": str(task),
        "control_marker": str(control_marker),
        "evaluated_marker": str(marker),
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
    write_json(Path(run_dir) / output_filename, summary)

    if not guard_passed:
        failed = ", ".join(
            f"{name}={observed[name]:.3f} > {limits[name]:.3f}"
            for name in limits
            if not component_pass[name]
        )
        print(
            f"The configured marker ID is not behaviorally neutral relative to the matched "
            f"control ID before {task} fine-tuning ({failed}). Choose a different preregistered "
            "five-digit ID set before training either trajectory."
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

    path = Path(run_dir) / "checkpoint_control_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    return path
