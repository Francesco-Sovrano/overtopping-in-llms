"""Resolve Stage-5/Stage-6 discovery artifacts from evaluation-side stats paths.

The overtopping repository has used more than one on-disk layout for Stage-6
agonist bags. Reporting code must therefore resolve the configured bag label
against all supported layouts before deciding that discovery is missing.
"""
from __future__ import annotations

import json
from pathlib import Path

from studies.overtopping.analysis.primary_holdout_analysis import reference_stats_dir


def model_root_from_stats(stats_dir: Path) -> Path | None:
    """Return the model root containing ``rule_extraction_results``."""
    parts = list(Path(stats_dir).parts)
    try:
        index = parts.index("rule_extraction_results")
    except ValueError:
        return None
    return Path(*parts[:index])


def stage5_and_stage6_candidates(stats_dir: Path) -> tuple[Path | None, list[Path]]:
    """Return the Stage-5 path and every supported Stage-6 bag location.

    The first Stage-6 candidate is the current pipeline layout. Remaining
    candidates are legacy/export layouts retained for backward compatibility.
    """
    model_root = model_root_from_stats(stats_dir)
    if model_root is None:
        return None, []

    label = reference_stats_dir(Path(stats_dir)).name
    marker = "-agonist_neurons"
    if marker not in label:
        return None, []

    circuit_label, rest = label.split(marker, 1)
    bag_label = "agonist_neurons" + rest
    circuit_root = (
        model_root
        / "neural_circuit_discovery_results"
        / "eap_ig_inputs"
        / circuit_label
    )
    stage5_dir = circuit_root / "neural_circuits"
    stage6_candidates = [
        circuit_root / bag_label,
        circuit_root / "bag_of_rules" / bag_label,
        stage5_dir / bag_label,
    ]
    return stage5_dir, stage6_candidates


def resolve_stage6_dir(stats_dir: Path) -> Path | None:
    """Resolve the Stage-6 bag, preferring a location with bucket artifacts.

    If a supported directory exists but has no bucket file yet, return that
    directory so callers can distinguish ``stage6_buckets_missing`` from a
    wholly missing Stage-6 run.
    """
    _stage5_dir, candidates = stage5_and_stage6_candidates(stats_dir)
    if not candidates:
        return None

    existing = [path for path in candidates if path.exists()]
    with_buckets = [path for path in existing if any(path.rglob("neuron_buckets.json"))]
    for path in with_buckets:
        # Prefer the current layout when it is complete, but do not let a stale
        # partial export mask a verified complete legacy tree.
        _count, status = stage6_candidate_count(path)
        if status == "ok":
            return path
    if with_buckets:
        return with_buckets[0]
    if existing:
        return existing[0]
    # Preserve the current-layout inferred path for diagnostics.
    return candidates[0]





def stage6_directional_zero_status(stage6_dir: Path | None) -> dict[str, str]:
    """Return direction-specific status for a completed empty Stage-6 search.

    The Stage-6 availability manifest records the two source-state searches
    separately. ``negative`` is the B=0 source population used for 0->1
    discovery, while ``positive`` is the B=1 source population used for 1->0
    discovery. A completed baseline search with zero retained candidates is a
    measured zero for that direction. A skipped baseline (for example because
    the source population is empty or too small) is unavailable, not a zero.

    Returned values are ``"measured_zero"``, ``"unavailable"``, or
    ``"unknown"`` for keys ``"i2c"`` and ``"c2i"``.
    """
    out = {"i2c": "unknown", "c2i": "unknown"}
    if stage6_dir is None:
        return out
    manifest = Path(stage6_dir) / "stage6_downstream_availability.json"
    if not manifest.is_file():
        return out
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return out
    baselines = payload.get("baselines", []) if isinstance(payload, dict) else []
    if not isinstance(baselines, list):
        return out
    mapping = {"negative": "i2c", "positive": "c2i"}
    for row in baselines:
        if not isinstance(row, dict):
            continue
        direction = mapping.get(str(row.get("baseline_subset", "")).strip().lower())
        if direction is None:
            continue
        try:
            n_ok = int(row.get("n_ok", 0) or 0)
            n_skipped = int(row.get("n_skipped", 0) or 0)
            n_candidates = int(row.get("eligible_agonist_candidates", 0) or 0)
        except (TypeError, ValueError):
            continue
        if n_ok > 0 and n_skipped == 0 and n_candidates == 0:
            out[direction] = "measured_zero"
        elif n_skipped > 0 and n_ok == 0:
            out[direction] = "unavailable"
    return out

def stage5_completion_status(stage5_dir: Path | None) -> str:
    """Classify Stage-5 circuit-discovery completion without guessing from emptiness.

    Stage 5 creates the output directory before expensive work begins and writes
    ``manifest.json`` plus ``dataset_info.json`` only at the successful end of
    the stage.  Therefore an existing-but-empty ``neural_circuits`` directory
    is evidence that Stage 5 was started, not evidence of a completed empty
    candidate set.
    """
    if stage5_dir is None:
        return "stage5_unresolvable"
    if not stage5_dir.exists():
        return "stage5_missing"
    manifest = stage5_dir / "manifest.json"
    dataset_info = stage5_dir / "dataset_info.json"
    if manifest.is_file() and dataset_info.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            return "stage5_completion_unverified"
        if not isinstance(payload, list):
            return "stage5_completion_unverified"
        return "stage5_complete"
    if manifest.exists() or dataset_info.exists():
        return "stage5_completion_unverified"
    return "stage5_started_unfinished"


def feature_report_status(stats_dir: Path) -> str:
    """Report whether the model-level feature table needed before discovery exists."""
    model_root = model_root_from_stats(stats_dir)
    if model_root is None:
        return "feature_report_unresolvable"
    report = model_root / "feature_report"
    scores = report / "scores.csv"
    if scores.is_file():
        return "feature_report_complete"
    if report.exists():
        return "feature_report_partial"
    return "feature_report_missing"

def stage5_and_stage6_paths(stats_dir: Path) -> tuple[Path | None, Path | None]:
    """Return the inferred Stage-5 directory and resolved Stage-6 bag."""
    stage5_dir, candidates = stage5_and_stage6_candidates(stats_dir)
    if not candidates:
        return stage5_dir, None
    return stage5_dir, resolve_stage6_dir(stats_dir)


def stage6_candidate_count(stage6_dir: Path | None) -> tuple[int | None, str]:
    """Recover the exact retained-candidate count from completed Stage-6 output.

    A verified count of zero is a completed scientific result. Stage 7
    intentionally has no singleton candidates to evaluate in that case, so the
    absence of singleton metric files must not be treated as an unfinished run.
    """
    if stage6_dir is None or not stage6_dir.exists():
        return None, "stage6_missing"

    # A zero-candidate discovery may legitimately produce no bucket files.
    # The v2 availability manifest is then the completion record: at least one
    # requested source-state search completed successfully and the union of
    # eligible retained candidates is empty. A direction that was skipped is
    # tracked separately by ``stage6_directional_zero_status``.
    availability = stage6_dir / "stage6_downstream_availability.json"
    if availability.is_file():
        try:
            payload = json.loads(availability.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            try:
                eligible = int(payload.get("eligible_stage6_agonist_candidates"))
            except (TypeError, ValueError):
                eligible = None
            baselines = payload.get("baselines", [])
            completed = 0
            if isinstance(baselines, list):
                for row in baselines:
                    if not isinstance(row, dict):
                        continue
                    try:
                        completed += max(0, int(row.get("n_ok", 0) or 0))
                    except (TypeError, ValueError):
                        pass
            if eligible == 0 and completed > 0:
                return 0, "ok"

    bucket_files = sorted(stage6_dir.rglob("neuron_buckets.json"))
    if not bucket_files:
        return None, "stage6_buckets_missing"

    kept: set[str] = set()
    for bucket_path in bucket_files:
        knockout_path = bucket_path.with_name("rule_knockout.json")
        if not knockout_path.is_file():
            return None, "stage6_completion_unverified"
        try:
            knockout = json.loads(knockout_path.read_text(encoding="utf-8"))
            records = knockout if isinstance(knockout, list) else [knockout]
            if not records or not all(isinstance(record, dict) for record in records):
                return None, "stage6_completion_unverified"
            if any(str(record.get("status", "")).lower() != "ok" for record in records):
                return None, "stage6_completion_unverified"

            buckets = json.loads(bucket_path.read_text(encoding="utf-8"))
        except Exception:
            return None, "stage6_completion_unverified"

        nca = buckets.get("non_catastrophic_agonists", {}) if isinstance(buckets, dict) else {}
        keep = set(str(key) for key in nca.keys()) if isinstance(nca, dict) else set()

        catastrophic = buckets.get("catastrophic_zero", {}) if isinstance(buckets, dict) else {}
        if isinstance(catastrophic, dict):
            bad: set[str] = set()
            for name in ("confirmed", "not_always", "candidates"):
                block = catastrophic.get(name, {})
                if isinstance(block, dict):
                    bad.update(str(key) for key in block.keys())
            keep.difference_update(bad)
        kept.update(keep)

    return len(kept), "ok"
