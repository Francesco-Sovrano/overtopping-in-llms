"""Durable checkpoint-manifest I/O, repair, and aggregate-union helpers.

A physical checkpoint is authoritative evidence that model weights were saved.
Manifests are indexes over those checkpoints and must never be allowed to erase
or hide an otherwise complete trajectory.  Repairs are conservative: configured
checkpoint fractions and physical directories must agree exactly before any
missing/incomplete index is reconstructed.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

CONDITION_MANIFEST_NAME = "checkpoint_manifest.csv"
AGGREGATE_MANIFEST_NAME = "checkpoint_manifest_all.csv"
_CHECKPOINT_TAG_RE = re.compile(r"^frac_(\d{4})_step_(\d+)$")


def _stable_fieldnames(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    preferred = [
        "condition",
        "fraction",
        "global_step",
        "checkpoint_dir",
        "checkpoint_format",
        "resume_checkpoint_dir",
        "resume_checkpoint_format",
        "poison_schedule_mode",
        "planned_poison_examples",
        "planned_counterfactual_slots",
        "cumulative_poison_examples_seen",
        "cumulative_counterfactual_slots_seen",
    ]
    observed = {str(key) for row in rows for key in row}
    return [key for key in preferred if key in observed] + sorted(observed.difference(preferred))


def write_checkpoint_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically replace a checkpoint manifest."""
    path = Path(path)
    normalized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _stable_fieldnames(normalized)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if normalized:
                writer.writeheader()
                writer.writerows(normalized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def read_checkpoint_manifest(path: Path) -> List[Dict[str, str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def checkpoint_dirs(condition_dir: Path) -> List[Path]:
    root = Path(condition_dir) / "checkpoints"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and _CHECKPOINT_TAG_RE.match(p.name))


def _parse_tag(tag: str) -> tuple[int, int] | None:
    match = _CHECKPOINT_TAG_RE.match(str(tag))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _row_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    condition = str(row.get("condition", "")).strip()
    fraction = int(round(float(row.get("fraction", 0.0)) * 1000.0))
    step = int(float(row.get("global_step", 0)))
    return condition, fraction, step


def _read_json_object(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _configured_fraction_millis(run_config: Mapping[str, Any]) -> set[int]:
    raw = run_config.get("save_fracs")
    if raw is None:
        return set()
    values = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")
    return {
        int(round(float(str(value).strip()) * 1000.0))
        for value in values
        if str(value).strip()
    }


def _cumulative_slots_from_schedule(schedule: Mapping[str, Any], global_step: int) -> int:
    mode = str(schedule.get("mode", "trainer_random"))
    if mode != "uniform_optimizer_steps":
        # Matches the historical callback: random-sampler runs had no exact
        # paired-slots-seen function and therefore recorded zero here.
        return 0

    positions = [int(v) for v in schedule.get("counterfactual_stream_positions", [])]
    n_train = int(schedule.get("n_train", 0))
    batch = int(schedule.get("per_device_train_batch_size", 0))
    accum = int(schedule.get("gradient_accumulation_steps", 0))
    if n_train <= 0 or batch <= 0 or accum <= 0:
        raise ValueError("uniform poison_schedule.json is missing n_train/batch/accum metadata")

    micro_batches_per_epoch = int(math.ceil(n_train / batch))
    steps_per_epoch = int(math.ceil(micro_batches_per_epoch / accum))
    full_epochs, within_steps = divmod(max(0, int(global_step)), steps_per_epoch)
    consumed_within = min(n_train, within_steps * accum * batch)
    return int(full_epochs * len(positions) + sum(1 for pos in positions if pos < consumed_within))


def _reconstruct_rows_from_training_state(
    run_dir: Path,
    condition: str,
    *,
    training_dirname: str,
    physical: Sequence[Path],
) -> tuple[List[Dict[str, Any]] | None, str]:
    """Reconstruct base ledger rows from persisted training metadata."""
    training = Path(run_dir) / training_dirname
    condition_dir = training / condition
    config_path = training / "metadata" / "run_config.json"
    poison_meta_path = condition_dir / "poison_meta.json"
    schedule_path = condition_dir / "poison_schedule.json"
    missing = [str(p) for p in (config_path, poison_meta_path, schedule_path) if not p.is_file()]
    if missing:
        return None, "missing persisted reconstruction metadata: " + ", ".join(missing)

    try:
        run_config = _read_json_object(config_path)
        poison_meta = _read_json_object(poison_meta_path)
        schedule = _read_json_object(schedule_path)
    except Exception as exc:
        return None, f"cannot read persisted reconstruction metadata: {exc}"

    expected = _configured_fraction_millis(run_config)
    parsed: List[tuple[Path, int, int]] = []
    for path in physical:
        decoded = _parse_tag(path.name)
        if decoded is not None:
            parsed.append((path, decoded[0], decoded[1]))
    physical_fractions = {fraction for _, fraction, _ in parsed}
    if expected and physical_fractions != expected:
        return None, (
            "physical checkpoints do not match configured save_fracs; "
            f"missing_fractions={sorted(expected - physical_fractions)} "
            f"unexpected_fractions={sorted(physical_fractions - expected)}"
        )
    if len(parsed) != len(physical_fractions):
        return None, "multiple physical checkpoint directories map to the same configured fraction"

    planned = int(poison_meta.get("n_planned_poison_pairs", 0) or 0)
    schedule_mode = str(schedule.get("mode") or run_config.get("poison_schedule_mode") or "trainer_random")
    rows: List[Dict[str, Any]] = []
    for checkpoint_path, frac_millis, step in sorted(parsed, key=lambda item: (item[1], item[2])):
        try:
            paired_seen = _cumulative_slots_from_schedule(schedule, step)
        except Exception as exc:
            return None, f"cannot reconstruct exposure counts from {schedule_path}: {exc}"
        poison_seen = paired_seen if condition != "clean" else 0
        rows.append({
            "condition": condition,
            "fraction": frac_millis / 1000.0,
            "global_step": step,
            "checkpoint_dir": f"checkpoints/{checkpoint_path.name}",
            "checkpoint_format": (
                "peft_adapter" if (checkpoint_path / "adapter_config.json").is_file() else "hf_full_model"
            ),
            "poison_schedule_mode": schedule_mode,
            "planned_poison_examples": planned if condition != "clean" else 0,
            "planned_counterfactual_slots": planned,
            "cumulative_poison_examples_seen": poison_seen,
            "cumulative_counterfactual_slots_seen": paired_seen,
        })
    return rows, "persisted run_config.json + poison_meta.json + poison_schedule.json"


def repair_condition_manifest(
    run_dir: Path,
    condition: str,
    *,
    training_dirname: str,
    expected_fractions: Sequence[float] | None = None,
    apply: bool = False,
) -> tuple[bool, str]:
    """Repair a missing *or incomplete* condition manifest conservatively.

    Physical checkpoint directories must exactly match the requested/configured
    save fractions.  Existing local/aggregate rows are preserved when available;
    missing rows are reconstructed from persisted training metadata.  This fixes
    both historical "manifest written only at train end" runs and the later
    crash window where an incremental writer could leave only ``frac_0000``.
    """
    run_dir = Path(run_dir)
    training = run_dir / training_dirname
    condition_dir = training / condition
    manifest_path = condition_dir / CONDITION_MANIFEST_NAME
    physical = checkpoint_dirs(condition_dir)
    if not physical:
        return False, "no physical checkpoint directories"

    parsed_physical: Dict[str, tuple[int, int, Path]] = {}
    for path in physical:
        decoded = _parse_tag(path.name)
        if decoded is not None:
            parsed_physical[path.name] = (decoded[0], decoded[1], path)
    physical_fractions = {fraction for fraction, _, _ in parsed_physical.values()}

    if expected_fractions is not None:
        expected = {int(round(float(v) * 1000.0)) for v in expected_fractions}
    else:
        config_path = training / "metadata" / "run_config.json"
        expected = set()
        if config_path.is_file():
            try:
                expected = _configured_fraction_millis(_read_json_object(config_path))
            except Exception as exc:
                return False, f"cannot read configured save_fracs from {config_path}: {exc}"

    if expected and physical_fractions != expected:
        return False, (
            "physical checkpoints do not match requested/configured save_fracs; "
            f"missing_fractions={sorted(expected - physical_fractions)} "
            f"unexpected_fractions={sorted(physical_fractions - expected)}"
        )

    local_rows: List[Dict[str, Any]] = []
    local_read_error = ""
    if manifest_path.is_file() and manifest_path.stat().st_size > 0:
        try:
            local_rows = [dict(row) for row in read_checkpoint_manifest(manifest_path)]
        except Exception as exc:
            local_read_error = str(exc)

    aggregate_path = training / "metadata" / AGGREGATE_MANIFEST_NAME
    aggregate_rows: List[Dict[str, Any]] = []
    if aggregate_path.is_file() and aggregate_path.stat().st_size > 0:
        try:
            aggregate_rows = [
                dict(row) for row in read_checkpoint_manifest(aggregate_path)
                if str(row.get("condition", "")).strip() == condition
            ]
        except Exception:
            aggregate_rows = []

    def rows_by_tag(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for source in rows:
            tag = Path(str(source.get("checkpoint_dir", "")).strip()).name
            if tag in parsed_physical:
                out[tag] = dict(source)
        return out

    local_by_tag = rows_by_tag(local_rows)
    aggregate_by_tag = rows_by_tag(aggregate_rows)

    reconstructed, reconstruction_detail = _reconstruct_rows_from_training_state(
        run_dir,
        condition,
        training_dirname=training_dirname,
        physical=physical,
    )
    reconstructed_by_tag = rows_by_tag(reconstructed or [])

    # If reconstruction metadata is unavailable, existing durable indexes may
    # still be sufficient provided they cover every physical checkpoint.
    uncovered = set(parsed_physical) - (set(local_by_tag) | set(aggregate_by_tag) | set(reconstructed_by_tag))
    if uncovered:
        detail = reconstruction_detail
        if local_read_error:
            detail += f"; local manifest unreadable: {local_read_error}"
        return False, f"cannot construct rows for physical checkpoints {sorted(uncovered)}; {detail}"

    repaired_rows: List[Dict[str, Any]] = []
    for tag, (frac_millis, step, checkpoint_path) in sorted(
        parsed_physical.items(), key=lambda item: (item[1][0], item[1][1])
    ):
        # Reconstructed scientific bookkeeping is the base; aggregate diagnostic
        # columns and then local diagnostic columns are retained when present.
        row: Dict[str, Any] = {}
        row.update(reconstructed_by_tag.get(tag, {}))
        row.update(aggregate_by_tag.get(tag, {}))
        row.update(local_by_tag.get(tag, {}))
        # Physical identity always wins over serialized paths/rounding.
        row.update({
            "condition": condition,
            "fraction": frac_millis / 1000.0,
            "global_step": step,
            "checkpoint_dir": f"checkpoints/{tag}",
            "checkpoint_format": (
                "peft_adapter" if (checkpoint_path / "adapter_config.json").is_file() else "hf_full_model"
            ),
        })
        repaired_rows.append(row)

    current_keys = {
        (Path(str(row.get("checkpoint_dir", ""))).name, int(round(float(row.get("fraction", -1)) * 1000)), int(float(row.get("global_step", -1))))
        for row in local_rows
        if Path(str(row.get("checkpoint_dir", ""))).name
    }
    desired_keys = {
        (Path(str(row["checkpoint_dir"])).name, int(round(float(row["fraction"]) * 1000)), int(row["global_step"]))
        for row in repaired_rows
    }
    if current_keys == desired_keys and len(local_rows) == len(repaired_rows):
        return False, f"already complete: {manifest_path}"

    source_bits = []
    if reconstructed_by_tag:
        source_bits.append(reconstruction_detail)
    if aggregate_by_tag:
        source_bits.append(str(aggregate_path))
    if local_by_tag:
        source_bits.append("existing local rows")
    source_detail = " + ".join(source_bits) or "physical checkpoint identities"

    if apply:
        write_checkpoint_manifest(manifest_path, repaired_rows)
        return True, f"repaired {manifest_path} ({len(repaired_rows)} rows; source={source_detail})"
    return True, f"would repair {manifest_path} ({len(repaired_rows)} rows; source={source_detail})"



def write_aggregate_manifest_union(
    aggregate_path: Path,
    preferred_rows: Sequence[Mapping[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """Atomically write the aggregate as a union of *all* condition manifests.

    ``preferred_rows`` (typically the just-loaded/trained conditions after path
    annotation) override matching local rows.  Existing aggregate-only derived
    columns are retained.  Crucially, running ``--condition clean`` can no longer
    delete poisoned rows from ``checkpoint_manifest_all.csv``.
    """
    aggregate_path = Path(aggregate_path)
    training = aggregate_path.parent.parent

    existing: List[Dict[str, Any]] = []
    if aggregate_path.is_file() and aggregate_path.stat().st_size > 0:
        try:
            existing = [dict(row) for row in read_checkpoint_manifest(aggregate_path)]
        except Exception:
            existing = []

    merged: Dict[tuple[str, int, int], Dict[str, Any]] = {}
    for row in existing:
        try:
            merged[_row_key(row)] = dict(row)
        except Exception:
            continue

    # Local condition manifests are authoritative for trajectory membership.
    local_keys: set[tuple[str, int, int]] = set()
    for condition_dir in sorted(p for p in training.iterdir() if p.is_dir() and p.name != "metadata"):
        manifest = condition_dir / CONDITION_MANIFEST_NAME
        if not manifest.is_file() or manifest.stat().st_size == 0:
            continue
        try:
            rows = read_checkpoint_manifest(manifest)
        except Exception:
            continue
        for source in rows:
            row = {**source, "condition": condition_dir.name}
            tag = Path(str(row.get("checkpoint_dir", "")).strip()).name
            if tag:
                row["checkpoint_dir"] = f"checkpoints/{tag}"
            key = _row_key(row)
            local_keys.add(key)
            merged[key] = {**merged.get(key, {}), **row}

    # Drop stale aggregate rows for conditions that have a local manifest: the
    # local manifest defines membership for that condition.  Keep aggregate-only
    # rows only for legacy conditions that genuinely have no local manifest yet.
    conditions_with_local = {key[0] for key in local_keys}
    for key in list(merged):
        if key[0] in conditions_with_local and key not in local_keys:
            del merged[key]

    for source in preferred_rows:
        row = dict(source)
        key = _row_key(row)
        merged[key] = {**merged.get(key, {}), **row}

    rows = sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("condition", "")),
            float(row.get("fraction", 0.0)),
            int(float(row.get("global_step", 0))),
        ),
    )
    write_checkpoint_manifest(aggregate_path, rows)
    return rows
