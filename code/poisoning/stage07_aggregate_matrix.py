#!/usr/bin/env python3
"""Aggregate poisoning trajectories with training seed as the replicate unit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from poisoning.lib.run_paths import metadata_path, phase_dirname, trajectories_dir
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import stats
from poisoning.tasks.registry import infer_task_from_run


DEFAULT_METRICS = (
    "ordinary_correctness_accuracy",
    "trigger_lift_success_rate",
    "conditional_conversion_rate",
    "trigger_excess_target_rate",
    "convertible_fraction",
    "primary_conditional_conversion_rate_on_sham_cohort",
    "sham_conditional_conversion_rate",
    "primary_minus_sham_conditional_conversion_rate",
    "trigger_target_positive_rate",
    "control_target_positive_rate",
    "lift_U(J)",
    "lift_Top",
    "lift_N.10",
    "ordinary_correctness_U(J)",
    "ordinary_correctness_Top",
)

EXPERIMENT_ID_COLUMNS = [
    "task",
    "model_name",
    "model_revision",
    "control_marker",
    "trigger_marker",
    "sham_marker",
    "sham_max_rows",
    "poison_rate",
    "poison_rate_basis",
    "poisoning_training_schema_version",
    "attacker_target",
]


def _ensure_identity_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in EXPERIMENT_ID_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame


def _csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _task_and_phase(run_dir: Path) -> tuple[str, str]:
    definition = infer_task_from_run(run_dir)
    return definition.name, definition.default_phase

def load_trajectory(run_dir: Path) -> pd.DataFrame:
    task, phase = _task_and_phase(run_dir)
    config = json.loads(metadata_path(run_dir, "run_config.json").read_text(encoding="utf-8"))
    path = trajectories_dir(run_dir) / phase_dirname(phase) / "backdoor_lift_overtopping_trajectory.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory: {path}")
    frame = pd.read_csv(path)
    if "ordinary_correctness_accuracy" not in frame.columns:
        _n_rows = pd.to_numeric(frame.get("ordinary_correctness_n_rows"), errors="coerce")
        _n_correct = pd.to_numeric(frame.get("ordinary_correctness_n_correct_total"), errors="coerce")
        if _n_rows is not None and _n_correct is not None:
            frame["ordinary_correctness_accuracy"] = _n_correct / _n_rows.where(_n_rows > 0)
    if "conditional_conversion_rate" not in frame.columns:
        n = pd.to_numeric(
            frame.get("attack_n", frame.get("lift_dataset_n")), errors="coerce"
        )
        lift = pd.to_numeric(frame.get("n_trigger_lift_success"), errors="coerce")
        no_target_rate = pd.to_numeric(
            frame.get(
                "control_target_rate_on_attack_cohort",
                frame.get("control_target_positive_rate"),
            ),
            errors="coerce",
        )
        conditional_n = (n * (1.0 - no_target_rate)).round()
        frame["conditional_conversion_n"] = conditional_n
        frame["conditional_conversion_success"] = lift
        frame["conditional_conversion_rate"] = lift / conditional_n.where(conditional_n > 0)
    frame.insert(0, "run_dir", str(run_dir.resolve()))
    frame.insert(1, "task", task)
    frame.insert(2, "model_name", str(config.get("model_name", "unknown")))
    frame.insert(3, "training_seed", int(config.get("seed", -1)))
    frame.insert(4, "model_revision", config.get("model_revision"))
    frame.insert(5, "control_marker", config.get("control_marker"))
    frame.insert(6, "trigger_marker", config.get("trigger_marker"))
    frame.insert(7, "sham_marker", config.get("sham_marker"))
    frame.insert(8, "sham_max_rows", config.get("sham_max_rows"))
    frame.insert(9, "poison_rate", config.get("poison_rate"))
    frame.insert(10, "poison_rate_basis", config.get("poison_rate_basis"))
    frame.insert(11, "poisoning_training_schema_version", config.get("poisoning_training_schema_version"))
    frame.insert(12, "attacker_target", config.get("target_label", config.get("target_answer")))
    return frame


def t_interval(values: Iterable[float], level: float) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan, math.nan
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / math.sqrt(len(values)))
    critical = float(stats.t.ppf(0.5 + level / 2.0, df=len(values) - 1))
    return mean - critical * sem, mean + critical * sem


def aggregate_seed_units(
    raw: pd.DataFrame,
    *,
    metrics: Iterable[str],
    level: float,
    min_seeds: int,
) -> pd.DataFrame:
    raw = _ensure_identity_columns(raw)
    records: List[Dict[str, Any]] = []
    group_cols = [*EXPERIMENT_ID_COLUMNS, "condition", "fraction"]
    for keys, group in raw.groupby(group_cols, dropna=False, sort=True):
        base = dict(zip(group_cols, keys))
        seeds = sorted(set(pd.to_numeric(group["training_seed"], errors="coerce").dropna().astype(int)))
        record: Dict[str, Any] = {
            **base,
            "n_training_seeds": len(seeds),
            "training_seeds": ",".join(str(seed) for seed in seeds),
            "developmental_claim_ready": len(seeds) >= int(min_seeds),
            "minimum_seeds_required": int(min_seeds),
        }
        for metric in metrics:
            if metric not in group.columns:
                continue
            # Exactly one checkpoint row per seed enters each cell. Duplicate
            # rows are an error rather than extra precision.
            per_seed = group[["training_seed", metric]].copy()
            per_seed[metric] = pd.to_numeric(per_seed[metric], errors="coerce")
            per_seed = per_seed.dropna(subset=[metric])
            if per_seed["training_seed"].duplicated().any():
                raise ValueError(f"Duplicate seed rows for {base} metric={metric}")
            values = per_seed[metric].to_numpy(dtype=float)
            lo, hi = t_interval(values, level)
            record[f"{metric}__n_seeds"] = int(len(values))
            record[f"{metric}__developmental_claim_ready"] = bool(
                len(values) >= int(min_seeds)
            )
            record[f"{metric}__mean"] = float(values.mean()) if len(values) else math.nan
            record[f"{metric}__sd"] = float(values.std(ddof=1)) if len(values) >= 2 else math.nan
            record[f"{metric}__ci_low"] = lo
            record[f"{metric}__ci_high"] = hi
        records.append(record)
    return pd.DataFrame(records)


def timing_by_seed(raw: pd.DataFrame, conversion_threshold: float) -> pd.DataFrame:
    raw = _ensure_identity_columns(raw)
    records: List[Dict[str, Any]] = []
    group_cols = [*EXPERIMENT_ID_COLUMNS, "training_seed", "condition", "run_dir"]
    for keys, group in raw.groupby(group_cols, dropna=False, sort=True):
        group = group.sort_values("fraction")
        conversion_source = group.get(
            "conditional_conversion_rate", pd.Series(np.nan, index=group.index)
        )
        conversion = pd.to_numeric(conversion_source, errors="coerce")
        fractions = pd.to_numeric(group["fraction"], errors="coerce")
        crossed = fractions[(conversion >= float(conversion_threshold)) & conversion.notna()]
        circuit = group.get("lift_circuit_defined", pd.Series(False, index=group.index)).astype(str).str.lower().isin({"1", "true", "yes"})
        circuit_fractions = fractions[circuit]
        records.append({
            **dict(zip(group_cols, keys)),
            "conditional_conversion_threshold": float(conversion_threshold),
            "first_fraction_at_conversion_threshold": float(crossed.min()) if len(crossed) else math.nan,
            "first_fraction_with_defined_trigger_lift_circuit": float(circuit_fractions.min()) if len(circuit_fractions) else math.nan,
            "n_checkpoint_rows": int(len(group)),
        })
    return pd.DataFrame(records)


def aggregate_timing_seed_units(
    timing: pd.DataFrame,
    *,
    level: float,
    min_seeds: int,
) -> pd.DataFrame:
    """Summarize checkpoint timing without treating missing crossings as zero.

    A seed that never crosses a threshold is right-censored at the final saved
    checkpoint. The table therefore reports both the reach rate across all
    seeds and a conditional mean timing among seeds that reached the event.
    """
    timing = _ensure_identity_columns(timing)
    records: List[Dict[str, Any]] = []
    group_cols = [*EXPERIMENT_ID_COLUMNS, "condition"]
    timing_metrics = (
        "first_fraction_at_conversion_threshold",
        "first_fraction_with_defined_trigger_lift_circuit",
    )
    for keys, group in timing.groupby(group_cols, dropna=False, sort=True):
        if group["training_seed"].duplicated().any():
            raise ValueError(
                "Duplicate developmental-timing rows for the same training seed and experiment identity"
            )
        seeds = sorted(set(pd.to_numeric(group["training_seed"], errors="coerce").dropna().astype(int)))
        record: Dict[str, Any] = {
            **dict(zip(group_cols, keys)),
            "n_training_seeds": len(seeds),
            "training_seeds": ",".join(str(seed) for seed in seeds),
            "developmental_claim_ready": len(seeds) >= int(min_seeds),
            "minimum_seeds_required": int(min_seeds),
            "conditional_conversion_threshold": group["conditional_conversion_threshold"].iloc[0],
        }
        for metric in timing_metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            observed = values.dropna().to_numpy(dtype=float)
            lo, hi = t_interval(observed, level)
            record[f"{metric}__n_reached"] = int(len(observed))
            record[f"{metric}__n_censored"] = int(len(seeds) - len(observed))
            record[f"{metric}__reach_rate"] = (
                float(len(observed) / len(seeds)) if seeds else math.nan
            )
            record[f"{metric}__all_seeds_reached"] = bool(
                seeds and len(observed) == len(seeds)
            )
            record[f"{metric}__conditional_mean"] = (
                float(observed.mean()) if len(observed) else math.nan
            )
            record[f"{metric}__conditional_ci_low"] = lo
            record[f"{metric}__conditional_ci_high"] = hi
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", required=True, help="Comma-separated completed poisoning run directories.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--confidence_level", type=float, default=0.95)
    parser.add_argument("--min_seeds", type=int, default=3)
    parser.add_argument("--conversion_threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("--confidence_level must be in (0, 1)")
    if args.min_seeds < 2:
        raise ValueError("--min_seeds must be at least 2")

    frames = [load_trajectory(Path(path).expanduser()) for path in _csv_list(args.run_dirs)]
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / "checkpoint_trajectories_all_seeds.csv", index=False)
    aggregate_seed_units(
        raw,
        metrics=_csv_list(args.metrics),
        level=args.confidence_level,
        min_seeds=args.min_seeds,
    ).to_csv(out_dir / "checkpoint_metrics_by_model_across_seeds.csv", index=False)
    timing = timing_by_seed(raw, args.conversion_threshold)
    timing.to_csv(out_dir / "developmental_timing_by_seed.csv", index=False)
    aggregate_timing_seed_units(
        timing,
        level=args.confidence_level,
        min_seeds=args.min_seeds,
    ).to_csv(out_dir / "developmental_timing_across_seeds.csv", index=False)
    (out_dir / "aggregation_config.json").write_text(
        json.dumps({
            "replicate_unit": "training_seed",
            "confidence_interval": "two-sided Student-t interval across seed-level estimates",
            "confidence_level": args.confidence_level,
            "minimum_seeds_for_developmental_claim": args.min_seeds,
            "conversion_threshold": args.conversion_threshold,
            "run_dirs": _csv_list(args.run_dirs),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote seed-level poisoning matrix summaries under {out_dir}")


if __name__ == "__main__":
    main()
