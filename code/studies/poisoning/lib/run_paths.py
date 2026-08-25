"""Canonical filesystem names for poisoning runs and discovery caches.

Names are intentionally descriptive because these directories are inspected by
humans during long-running experiments. Production code never moves or rewrites
an existing run directory implicitly.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

TRAINING_DIRNAME = "01_training_checkpoints"
COHORTS_DIRNAME = "02_evaluation_cohorts"
CAUSAL_DIRNAME = "03_checkpoint_causal_discovery"
COMPARISONS_DIRNAME = "04_condition_comparisons"
TRAJECTORIES_DIRNAME = "05_behavior_trajectories"
CIRCUITS_DIRNAME = "06_circuit_overlap_analysis"
METADATA_DIRNAME = "metadata"
DEFENCE_DIRNAME = "defence"

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


def defence_cache_dir(
    cache_root: str | Path,
    *,
    task: str,
    run_dir: str | Path,
    phase: str,
    fraction: float,
) -> Path:
    """Return the canonical cache directory for one inference-defence checkpoint.

    Defence caches belong to the same ``<task>/<run>`` namespace as the rest of
    a poisoning run. ``phase`` intentionally uses the analysis labels
    ``input_output`` and ``output_only`` rather than the presentation-oriented
    output-directory labels.
    """
    task = str(task).strip()
    if not task or Path(task).name != task:
        raise ValueError("task must be a plain directory name")
    run_name = Path(run_dir).name
    if not run_name:
        raise ValueError("run_dir must have a basename")
    phase = str(phase).strip()
    if phase not in {"input_output", "output_only"}:
        raise ValueError(f"unknown defence cache phase: {phase!r}")
    return (
        Path(cache_root).expanduser()
        / task
        / run_name
        / DEFENCE_DIRNAME
        / phase
        / f"fraction_{float(fraction):.6f}"
    )


_PROGRESS_CHECKPOINT_RE = re.compile(r"^progress_(\d{3})(?:p(\d))?pct__step_(\d+)$")
_LEGACY_CHECKPOINT_RE = re.compile(r"^frac_(\d{4})_step_(\d+)$")


def _fraction_millis(row: Mapping[str, Any]) -> int:
    return int(round(float(row.get("fraction", 0.0)) * 1000.0))


def checkpoint_progress_label(row: Mapping[str, Any]) -> str:
    """Canonical human-readable physical/result/cache label for a checkpoint.

    Training fractions are persisted at per-mille precision.  Whole percentage
    checkpoints use labels such as ``progress_010pct__step_0025``; fractional
    percentages use ``p`` as the decimal separator, e.g. ``progress_002p5pct``.
    """
    frac_millis = _fraction_millis(row)
    step = int(float(row.get("global_step", 0)))
    whole_pct, tenth_pct = divmod(frac_millis, 10)
    pct_label = f"{whole_pct:03d}pct" if tenth_pct == 0 else f"{whole_pct:03d}p{tenth_pct}pct"
    return f"progress_{pct_label}__step_{step:04d}"


def checkpoint_tag(row: Mapping[str, Any]) -> str:
    """Canonical checkpoint identity used by new training runs and analyses."""
    return checkpoint_progress_label(row)


def legacy_checkpoint_tag(row: Mapping[str, Any]) -> str:
    """Historical physical checkpoint name retained only for migration/resume."""
    step = int(float(row.get("global_step", 0)))
    return f"frac_{_fraction_millis(row):04d}_step_{step}"


def parse_checkpoint_dirname(name: str) -> tuple[int, int] | None:
    """Return ``(fraction_millis, global_step)`` for canonical or legacy names."""
    value = Path(str(name)).name
    match = _PROGRESS_CHECKPOINT_RE.match(value)
    if match:
        whole_pct = int(match.group(1))
        tenth_pct = int(match.group(2) or 0)
        return whole_pct * 10 + tenth_pct, int(match.group(3))
    match = _LEGACY_CHECKPOINT_RE.match(value)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


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
    root = training_condition_dir(run_dir, condition) / "checkpoints"
    names = []
    if raw:
        names.append(Path(raw).name)
    names.extend([checkpoint_tag(row), legacy_checkpoint_tag(row)])
    deduped = list(dict.fromkeys(name for name in names if name))
    for name in deduped:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    candidate = root / deduped[0]
    if must_exist:
        raise FileNotFoundError(
            "Checkpoint declared by manifest is missing from the run-local training stage; "
            f"tried: {[str(root / name) for name in deduped]}"
        )
    return candidate
