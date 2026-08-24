"""Canonical filesystem names for poisoning runs and discovery caches.

Names are intentionally descriptive because these directories are inspected by
humans during long-running experiments. Production code never moves or rewrites
an existing run directory implicitly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

TRAINING_DIRNAME = "01_training_checkpoints"
COHORTS_DIRNAME = "02_evaluation_cohorts"
CAUSAL_DIRNAME = "03_checkpoint_causal_discovery"
COMPARISONS_DIRNAME = "04_condition_comparisons"
TRAJECTORIES_DIRNAME = "05_behavior_trajectories"
CIRCUITS_DIRNAME = "06_circuit_overlap_analysis"
METADATA_DIRNAME = "metadata"

# Internal CLI values remain stable; only their filesystem labels are clarified.
_PHASE_DIRNAMES = {
    "input_output": "prompt_and_generation",
    "output_only": "generation_only",
}


def phase_dirname(phase: str) -> str:
    phase = str(phase).strip()
    try:
        return _PHASE_DIRNAMES[phase]
    except KeyError as exc:
        raise ValueError(f"unknown poisoning phase: {phase!r}") from exc


def training_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / TRAINING_DIRNAME


def training_condition_dir(run_dir: str | Path, condition: str) -> Path:
    condition = str(condition).strip()
    if not condition or Path(condition).name != condition:
        raise ValueError("condition must be a plain directory name")
    return training_dir(run_dir) / condition


def metadata_dir(run_dir: str | Path) -> Path:
    return training_dir(run_dir) / METADATA_DIRNAME


def metadata_path(run_dir: str | Path, filename: str) -> Path:
    if Path(filename).name != filename:
        raise ValueError("metadata filename must be a plain filename")
    return metadata_dir(run_dir) / filename


def cohorts_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / COHORTS_DIRNAME


def cohort_path(run_dir: str | Path, filename: str) -> Path:
    if Path(filename).name != filename:
        raise ValueError("cohort filename must be a plain filename")
    return cohorts_dir(run_dir) / filename


def causal_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / CAUSAL_DIRNAME


def comparisons_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / COMPARISONS_DIRNAME


def trajectories_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / TRAJECTORIES_DIRNAME


def circuits_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / CIRCUITS_DIRNAME


def checkpoint_tag(row: Mapping[str, Any]) -> str:
    """Stable on-disk checkpoint identity used by training manifests."""
    frac = float(row.get("fraction", 0.0))
    step = int(float(row.get("global_step", 0)))
    return f"frac_{int(round(frac * 1000)):04d}_step_{step}"


def checkpoint_progress_label(row: Mapping[str, Any]) -> str:
    """Human-readable cache/result label for a checkpoint.

    The physical checkpoint tag is stable. Cache names use an
    ordinary percentage so ``frac_0100`` is no longer visually confused with
    100 percent.
    """
    frac = float(row.get("fraction", 0.0))
    step = int(float(row.get("global_step", 0)))
    pct = frac * 100.0
    if abs(pct - round(pct)) < 1e-9:
        pct_label = f"{int(round(pct)):03d}pct"
    else:
        pct_label = f"{pct:06.2f}".rstrip("0").rstrip(".").replace(".", "p") + "pct"
    return f"progress_{pct_label}__step_{step:04d}"


def checkpoint_cache_key(row: Mapping[str, Any]) -> str:
    condition = str(row.get("condition", "unknown")).strip() or "unknown"
    return f"{condition}__{checkpoint_progress_label(row)}"


def resolve_manifest_checkpoint_dir(
    run_dir: str | Path,
    row: Mapping[str, Any],
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve a manifest row to its checkpoint inside this run's training stage."""
    condition = str(row.get("condition", "")).strip()
    if not condition:
        raise ValueError("checkpoint manifest row is missing condition")

    raw = str(row.get("checkpoint_dir", "")).strip()
    name = Path(raw).name if raw else checkpoint_tag(row)
    if not name:
        name = checkpoint_tag(row)
    candidate = training_condition_dir(run_dir, condition) / "checkpoints" / name
    if must_exist and not candidate.is_dir():
        raise FileNotFoundError(
            f"Checkpoint declared by manifest is missing from the run-local training stage: {candidate}"
        )
    return candidate
