"""Resolve the ordinary-model overtopping candidate set used as a Stage-06 reference.

The canonical membership source is ``frozen_candidate_ranking.csv`` produced by
the shared pipeline.  Using the ranking file keeps poisoning overlap analysis
consistent with the singleton channels actually evaluated and plotted by the
overtopping pipeline; raw ``neuron_buckets.json`` files are discovery internals
and are not treated as a second definition of the candidate set.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd


def _ranking_file(path: str | Path) -> Path:
    value = Path(path).expanduser().resolve()
    if value.is_file():
        if value.name != "frozen_candidate_ranking.csv":
            raise FileNotFoundError(f"Expected frozen_candidate_ranking.csv, got {value}")
        return value
    ranking = value / "frozen_candidate_ranking.csv"
    if not ranking.is_file():
        raise FileNotFoundError(f"Ordinary-model candidate source has no frozen_candidate_ranking.csv: {value}")
    return ranking


def _count(path: str | Path) -> int:
    try:
        return int(len(pd.read_csv(_ranking_file(path))))
    except Exception:
        return 0


def _source_phase(path: Path) -> str:
    return "output_only" if "decode_only" in str(path).replace("\\", "/") else "input_output"


def _configured_tau_label() -> str:
    raw = os.environ.get("CHA_TAU", "0.3").strip()
    try:
        return f"{float(raw):g}"
    except ValueError:
        return raw


def _candidate_score(path: Path, *, phase: str, intervention: str) -> tuple:
    text = str(path).replace("\\", "/")
    tau_label = _configured_tau_label()
    tau_match = False
    match = re.search(r"agonist_neurons-fast-random_anchor(?:-tau([^/]+))?", text)
    if match:
        actual_tau = match.group(1) or "0.2"
        try:
            actual_tau = f"{float(actual_tau):g}"
        except ValueError:
            pass
        tau_match = actual_tau == tau_label
    return (
        int(_source_phase(path) == phase),
        int(f"eval_{intervention}" in text),
        int("spectral_split-M200000" in text),
        int(tau_match),
        int("positive_baseline" in text),
        _count(path),
        -len(text),
    )


def normal_model_root(project_root: str | Path, *, task: str, task_data_dir: str, model_name: str) -> Path:
    if not str(task_data_dir).strip():
        raise ValueError("task_data_dir must be provided by the task definition")
    override = os.environ.get(f"POISONING_{task.upper()}_VIRGIN_MODEL_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path(project_root).resolve() / "data" / str(task_data_dir).strip() / Path(model_name)).resolve()


def validate_virgin_agonists_path(
    path: str | Path,
    *,
    project_root: str | Path,
    phase: str,
    expected_root: str | Path | None = None,
) -> Path:
    ranking = _ranking_file(path)
    poisoning_root = (Path(project_root).expanduser().resolve() / "data" / "poisoning").resolve()
    try:
        ranking.relative_to(poisoning_root)
    except ValueError:
        pass
    else:
        raise RuntimeError(
            "The ordinary-model overlap reference must not come from a poisoning trajectory; "
            f"got {ranking}"
        )
    if expected_root is not None:
        expected = Path(expected_root).expanduser().resolve()
        try:
            ranking.relative_to(expected)
        except ValueError as exc:
            raise RuntimeError(
                f"Ordinary-model reference must live under {expected}; got {ranking}. "
                "Set POISONING_<TASK>_VIRGIN_MODEL_ROOT if the ordinary analysis is stored elsewhere."
            ) from exc
    actual_phase = _source_phase(ranking)
    if actual_phase != phase:
        raise RuntimeError(
            f"Ordinary-model candidate ranking was produced for {actual_phase}, requested phase is {phase}."
        )
    return ranking.parent


def resolve_virgin_agonists_path(
    project_root: str | Path,
    *,
    task: str,
    model_name: str,
    task_data_dir: str,
    phase: str,
    intervention: str = "mean-donor",
    require_phase_match: bool = True,
) -> Path:
    explicit = os.environ.get(f"POISONING_{task.upper()}_VIRGIN_AGONISTS_PATH", "").strip()
    if not explicit:
        explicit = os.environ.get("POISONING_VIRGIN_AGONISTS_PATH", "").strip()
    root = normal_model_root(project_root, task=task, task_data_dir=task_data_dir, model_name=model_name)
    if explicit:
        if require_phase_match:
            return validate_virgin_agonists_path(
                explicit, project_root=project_root, phase=phase, expected_root=root
            )
        return _ranking_file(explicit).parent

    rankings = [
        p for p in root.rglob("frozen_candidate_ranking.csv")
        if "neural_circuit_discovery_results" in str(p)
        and "eap_ig_inputs" in str(p)
        and _count(p) > 0
    ]
    if require_phase_match:
        rankings = [p for p in rankings if _source_phase(p) == phase]
    if not rankings:
        raise FileNotFoundError(
            f"No {'phase-matched ' if require_phase_match else ''}ordinary-model frozen candidate ranking under {root}. "
            f"Set POISONING_{task.upper()}_VIRGIN_AGONISTS_PATH explicitly."
        )
    best = sorted(
        rankings,
        key=lambda p: (_candidate_score(p, phase=phase, intervention=intervention), str(p)),
        reverse=True,
    )[0]
    resolved = (
        validate_virgin_agonists_path(best, project_root=project_root, phase=phase, expected_root=root)
        if require_phase_match else best.parent
    )
    print(
        f"[virgin-agonists] source={resolved / 'frozen_candidate_ranking.csv'} "
        f"n={_count(resolved)} phase={_source_phase(resolved)}",
        flush=True,
    )
    return resolved


def read_agonist_coordinates(path: str | Path) -> list[tuple[str, int]]:
    ranking = pd.read_csv(_ranking_file(path))
    if "layer_label" not in ranking.columns or "neuron_id" not in ranking.columns:
        raise ValueError(f"frozen_candidate_ranking.csv lacks layer_label/neuron_id: {_ranking_file(path)}")
    coords = [
        (str(layer), int(neuron))
        for layer, neuron in zip(ranking["layer_label"], pd.to_numeric(ranking["neuron_id"], errors="raise"))
    ]
    return list(dict.fromkeys(coords))
