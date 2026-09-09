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
DETECTION_DIRNAME = "07_poisoning_example_detection"
METADATA_DIRNAME = "metadata"

# Endpoint directory names.
BACKDOOR_TRIGGER_TEST_DIRNAME = "backdoor_trigger_test"
NORMAL_TASK_BEHAVIOR_DIRNAME = "normal_task"
ATTACK_COHORT_CONTROL_CORRECTNESS_DIRNAME = "attack_cohort_control_correctness"
OBSERVED_TRAINING_MIXTURE_CORRECTNESS_DIRNAME = "observed_training_mixture_correctness"


def format_fraction_percent(value: float) -> str:
    """Format a [0,1] trajectory fraction as a compact percentage label."""
    return f"{100.0 * float(value):g}%"


def safe_component(value: object, *, fallback: str = "run") -> str:
    """Return a filesystem-safe run/path component."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return re.sub(r"_+", "_", text).strip("_") or fallback

def behavior_endpoint_dirname(kind: str) -> str:
    value = str(kind).strip().lower()
    if value == "backdoor_trigger_test":
        return BACKDOOR_TRIGGER_TEST_DIRNAME
    if value == "normal_task":
        return NORMAL_TASK_BEHAVIOR_DIRNAME
    raise ValueError(f"unknown poisoning behavior endpoint: {kind!r}")


def model_variant_label(condition: str) -> str:
    value = str(condition).strip().lower()
    return {
        "clean": "clean_trained_model",
        "poisoned": "poison_trained_model",
    }.get(value, value or "unknown_model_variant")
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


def detection_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / DETECTION_DIRNAME



_PROGRESS_CHECKPOINT_RE = re.compile(r"^progress_(\d{3})(?:p(\d))?pct__step_(\d+)$")


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
    """Canonical checkpoint identity used by training and analysis."""
    return checkpoint_progress_label(row)


def parse_checkpoint_dirname(name: str) -> tuple[int, int] | None:
    """Return ``(fraction_millis, global_step)`` for a canonical checkpoint name."""
    value = Path(str(name)).name
    match = _PROGRESS_CHECKPOINT_RE.match(value)
    if not match:
        return None
    whole_pct = int(match.group(1))
    tenth_pct = int(match.group(2) or 0)
    return whole_pct * 10 + tenth_pct, int(match.group(3))


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
    names.append(checkpoint_tag(row))
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
