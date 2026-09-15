#!/usr/bin/env python3
"""Aggregate poisoning trajectories with training seed as the replicate unit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from studies.poisoning.lib.run_paths import checkpoint_progress_label, detection_dir, metadata_path, phase_dirname, trajectories_dir, training_condition_dir
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd
from scipy import stats
from studies.poisoning.tasks.registry import infer_task_from_run
from studies.poisoning.lib.scientific_config import (
    SCIENTIFIC_TRAINING_CONFIG_FIELDS,
    scientific_training_config_payload,
)


DEFAULT_METRICS = (
    "attack_cohort_control_correctness_accuracy",
    "poisoned_training_realized_poison_rate_overall",
    "poisoned_training_n_poisoned",
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
    "attack_cohort_control_correctness_U(J)",
    "attack_cohort_control_correctness_Top",
    "paired_control_U(J)",
    "paired_control_Top",
    "paired_attack_U(J)",
    "paired_attack_Top",
)



DETECTION_METRICS = (
    "roc_auc",
    "average_precision",
    "precision_at_expected_poison_count",
    "recall_at_expected_poison_count",
    "paired_poison_over_source_rate",
    "matched_random_roc_auc",
    "matched_random_average_precision",
    "matched_random_precision_at_expected_poison_count",
    "matched_random_recall_at_expected_poison_count",
    "matched_random_paired_poison_over_source_rate",
    "roc_auc_minus_matched_random",
    "average_precision_minus_matched_random",
    "precision_at_expected_poison_count_minus_matched_random",
    "recall_at_expected_poison_count_minus_matched_random",
    "paired_poison_over_source_rate_minus_matched_random",
    "poison_prevalence",
    "mean_score_gap",
    "sum_disruption_score",
    "max_disruption_score",
    "n_selected_disruptive_channels",
    "n_complete_u_j_channels",
    "n_mapped_causal_channels",
    "n_unique_mapped_parameter_rows",
    "n_gqa_or_other_collapsed_channel_mappings",
    "n_matched_control_realizations",
    "matched_control_roc_auc_mean", "matched_control_roc_auc_range_low", "matched_control_roc_auc_range_high",
    "roc_auc_candidate_percentile_vs_matched_controls",
    "matched_control_average_precision_mean", "matched_control_average_precision_range_low", "matched_control_average_precision_range_high",
    "average_precision_candidate_percentile_vs_matched_controls",
    "matched_control_precision_at_expected_poison_count_mean", "matched_control_precision_at_expected_poison_count_range_low", "matched_control_precision_at_expected_poison_count_range_high",
    "precision_at_expected_poison_count_candidate_percentile_vs_matched_controls",
    "matched_control_recall_at_expected_poison_count_mean", "matched_control_recall_at_expected_poison_count_range_low", "matched_control_recall_at_expected_poison_count_range_high",
    "recall_at_expected_poison_count_candidate_percentile_vs_matched_controls",
    "matched_control_paired_poison_over_source_rate_mean", "matched_control_paired_poison_over_source_rate_range_low", "matched_control_paired_poison_over_source_rate_range_high",
    "paired_poison_over_source_rate_candidate_percentile_vs_matched_controls",
    "poisoned_start_baseline_accuracy", "poisoned_end_baseline_accuracy",
    "clean_start_baseline_accuracy", "clean_end_baseline_accuracy",
    "poisoned_baseline_accuracy_change", "clean_baseline_accuracy_change",
    "poisoning_excess_baseline_accuracy_change",
    "selected_channel_mean_poisoning_excess_conditional_c2i_change",
)

DEFENSE_IDENTITY_COLUMNS = ("operating_benign_damage_budget",)

DEFENSE_METRICS = (
    "n_selected",
    "mean_clean_disruption",
    "mean_poisoned_disruption",
    "mean_poison_excess",
    "mean_attack_suppression",
    "mean_defense_leverage",
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
CONFIG_ID_COLUMNS = [f"config__{name}" for name in SCIENTIFIC_TRAINING_CONFIG_FIELDS]
EXPERIMENT_ID_COLUMNS.extend(CONFIG_ID_COLUMNS)

# Detection settings are part of the detector-family identity.  They are kept
# separate from the training experiment identity because a single training run
# may legitimately be re-analysed with multiple detector configurations.
DETECTOR_IDENTITY_COLUMNS = [
    "scoring_schema_version", "control_correctness_agonist_tau", "eval_intervention",
    "detector_max_channels", "detector_min_abs_delta_u", "detector_min_clean_null_z",
    "detector_bootstrap_draws", "detector_bootstrap_confidence_level", "detector_multiplicity_method",
    "detector_max_exposures_per_interval", "detector_sample_seed", "detector_matched_control_draws",
    "n_clean_null_trajectories", "u_j_definition",
]


def _poison_plan_metadata(run_dir: Path) -> dict[str, Any]:
    path = training_condition_dir(run_dir, "poisoned") / "poison_meta.json"
    if not path.is_file():
        return {
            "poisoned_training_realized_poison_rate_overall": None,
            "poisoned_training_n_poisoned": None,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "poisoned_training_realized_poison_rate_overall": payload.get("realized_poison_rate_overall"),
        "poisoned_training_n_poisoned": payload.get("n_poisoned"),
    }


def _explicit_config_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    """Expose every scientific config field directly as a groupable scalar."""
    payload = scientific_training_config_payload(config)
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (list, tuple, dict)):
            value = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
        out[f"config__{key}"] = value
    return out


def _attach_metadata(frame: pd.DataFrame, metadata: Mapping[str, Any]) -> pd.DataFrame:
    """Attach authoritative run metadata without creating duplicate columns.

    Stage-local CSVs may already carry columns such as ``task``.  Assigning
    metadata by column name deliberately overwrites those copies and guarantees
    a one-dimensional column index before groupby operations.
    """
    out = frame.copy()
    if out.columns.duplicated().any():
        duplicates = sorted(set(out.columns[out.columns.duplicated()].tolist()))
        raise ValueError(f"Input table already contains duplicate columns: {duplicates}")
    for key, value in metadata.items():
        out[key] = value
    if out.columns.duplicated().any():
        duplicates = sorted(set(out.columns[out.columns.duplicated()].tolist()))
        raise RuntimeError(f"Metadata attachment created duplicate columns: {duplicates}")
    return out

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


def _run_metadata(run_dir: Path) -> dict[str, Any]:
    """Return seed and scientific identity metadata shared by cross-seed tables."""
    task, _ = _task_and_phase(run_dir)
    config = json.loads(metadata_path(run_dir, "run_config.json").read_text(encoding="utf-8"))
    return {
        "run_dir": str(run_dir.resolve()),
        "task": task,
        "model_name": str(config.get("model_name", "unknown")),
        "training_seed": int(config.get("seed", -1)),
        "model_revision": config.get("model_revision"),
        "control_marker": config.get("control_marker"),
        "trigger_marker": config.get("trigger_marker"),
        "sham_marker": config.get("sham_marker"),
        "sham_max_rows": config.get("sham_max_rows"),
        "poison_rate": config.get("poison_rate"),
        "poison_rate_basis": config.get("poison_rate_basis"),
        "poisoning_training_schema_version": config.get("poisoning_training_schema_version"),
        "attacker_target": config.get("target_label", config.get("target_answer")),
        **_poison_plan_metadata(run_dir),
        **_explicit_config_metadata(config),
    }



def _stage07_endpoint_metrics(endpoint_dir: Path) -> dict[str, float]:
    out = {"U(J)": math.nan, "Top": math.nan}
    global_path = endpoint_dir / "flip_stats_global.json"
    flip_path = endpoint_dir / "flip_stats_by_neuron.csv"
    if global_path.is_file():
        try:
            payload = json.loads(global_path.read_text(encoding="utf-8"))
            out["U(J)"] = float(payload.get("union_c2i_unique_rate_conditional", math.nan))
        except Exception:
            pass
    if flip_path.is_file():
        try:
            flip = pd.read_csv(flip_path)
            values = pd.to_numeric(flip.get("c2i_rate"), errors="coerce")
            if values is not None and values.notna().any():
                out["Top"] = float(values.max())
        except Exception:
            pass
    return out


def _merge_stage07_paired_metrics(run_dir: Path, phase: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Attach post-discovery paired control/attack U(j) summaries when Stage 07 exists.

    This keeps cross-seed trajectory plots compatible while removing the old
    attack-cohort-control CHA.  The legacy ``attack_cohort_control_correctness``
    metric names now denote matched control *evaluation* of the defense-valid
    observed-mixture candidate union, not a localization endpoint.
    """
    out = frame.copy()
    manifest_path = metadata_path(run_dir, "checkpoint_manifest_all.csv")
    if not manifest_path.is_file():
        return out
    manifest = pd.read_csv(manifest_path)
    root = detection_dir(run_dir) / phase_dirname(phase) / "paired_u_j_materialization"
    if not root.is_dir():
        return out

    for column in (
        "paired_control_U(J)", "paired_control_Top",
        "paired_attack_U(J)", "paired_attack_Top",
    ):
        if column not in out.columns:
            out[column] = math.nan

    for rec in manifest.to_dict("records"):
        condition = str(rec.get("condition"))
        fraction = float(rec.get("fraction"))
        stage = checkpoint_progress_label(rec)
        state = root / condition / stage
        # Poisoned 0% reuses the identical clean pre-training materialization.
        if condition == "poisoned" and abs(fraction) <= 1e-12 and not state.is_dir():
            state = root / "clean" / stage
        control = _stage07_endpoint_metrics(state / "endpoint_stats" / "control")
        attack = (
            _stage07_endpoint_metrics(state / "endpoint_stats" / "attack")
            if condition == "poisoned" and abs(fraction) > 1e-12 else {"U(J)": math.nan, "Top": math.nan}
        )
        mask = out["condition"].astype(str).eq(condition) & np.isclose(
            pd.to_numeric(out["fraction"], errors="coerce"), fraction, atol=1e-12, rtol=0.0
        )
        if not bool(mask.any()):
            continue
        out.loc[mask, "paired_control_U(J)"] = control["U(J)"]
        out.loc[mask, "paired_control_Top"] = control["Top"]
        out.loc[mask, "paired_attack_U(J)"] = attack["U(J)"]
        out.loc[mask, "paired_attack_Top"] = attack["Top"]
        # Backward-compatible column names used by existing trajectory plots.
        if np.isfinite(control["U(J)"]):
            out.loc[mask, "attack_cohort_control_correctness_U(J)"] = control["U(J)"]
        if np.isfinite(control["Top"]):
            out.loc[mask, "attack_cohort_control_correctness_Top"] = control["Top"]
    return out

def load_trajectory(run_dir: Path) -> pd.DataFrame:
    task, phase = _task_and_phase(run_dir)
    path = trajectories_dir(run_dir) / phase_dirname(phase) / "backdoor_lift_overtopping_trajectory.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory: {path}")
    frame = pd.read_csv(path)
    frame = _merge_stage07_paired_metrics(run_dir, phase, frame)
    if "attack_cohort_control_correctness_accuracy" not in frame.columns:
        n_rows = pd.to_numeric(frame.get("attack_cohort_control_correctness_n_rows"), errors="coerce")
        n_correct = pd.to_numeric(frame.get("attack_cohort_control_correctness_n_correct_total"), errors="coerce")
        if n_rows is not None and n_correct is not None:
            frame["attack_cohort_control_correctness_accuracy"] = n_correct / n_rows.where(n_rows > 0)
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
    return _attach_metadata(frame, _run_metadata(run_dir))


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
    if raw.empty:
        return pd.DataFrame()
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
            "small_seed_count_caution": bool(len(seeds) < 5),
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
            record[f"{metric}__small_seed_count_caution"] = bool(len(values) < 5)
            record[f"{metric}__mean"] = float(values.mean()) if len(values) else math.nan
            record[f"{metric}__median"] = float(np.median(values)) if len(values) else math.nan
            record[f"{metric}__q25"] = float(np.quantile(values, 0.25)) if len(values) else math.nan
            record[f"{metric}__q75"] = float(np.quantile(values, 0.75)) if len(values) else math.nan
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
    if timing.empty:
        return pd.DataFrame()
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
            "small_seed_count_caution": bool(len(seeds) < 5),
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



def load_detection_metrics(run_dir: Path) -> pd.DataFrame:
    task, phase = _task_and_phase(run_dir)
    path = detection_dir(run_dir) / phase_dirname(phase) / "detection_metrics_by_interval.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if frame.empty:
        return frame
    return _attach_metadata(frame, _run_metadata(run_dir))


def aggregate_detection_seed_units(raw: pd.DataFrame, *, level: float, min_seeds: int) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    raw = _ensure_identity_columns(raw)
    detector_identity_cols = DETECTOR_IDENTITY_COLUMNS
    for column in detector_identity_cols:
        if column not in raw.columns:
            raw[column] = None
    group_cols = [*EXPERIMENT_ID_COLUMNS, *detector_identity_cols, "start_fraction", "end_fraction"]
    records: List[Dict[str, Any]] = []
    for keys, group in raw.groupby(group_cols, dropna=False, sort=True):
        base = dict(zip(group_cols, keys))
        seeds = sorted(set(pd.to_numeric(group["training_seed"], errors="coerce").dropna().astype(int)))
        rec: Dict[str, Any] = {
            **base,
            "n_training_seeds": len(seeds),
            "training_seeds": ",".join(str(seed) for seed in seeds),
            "developmental_claim_ready": len(seeds) >= int(min_seeds),
            "minimum_seeds_required": int(min_seeds),
            "small_seed_count_caution": bool(len(seeds) < 5),
        }
        for metric in DETECTION_METRICS:
            if metric not in group.columns:
                continue
            per_seed = group[["training_seed", metric]].copy()
            per_seed[metric] = pd.to_numeric(per_seed[metric], errors="coerce")
            per_seed = per_seed.dropna(subset=[metric])
            if per_seed["training_seed"].duplicated().any():
                raise ValueError(f"Duplicate poison-detection rows for {base} metric={metric}")
            values = per_seed[metric].to_numpy(dtype=float)
            lo, hi = t_interval(values, level)
            rec[f"{metric}__n_seeds"] = int(len(values))
            rec[f"{metric}__developmental_claim_ready"] = bool(len(values) >= int(min_seeds))
            rec[f"{metric}__small_seed_count_caution"] = bool(len(values) < 5)
            rec[f"{metric}__mean"] = float(values.mean()) if len(values) else math.nan
            rec[f"{metric}__median"] = float(np.median(values)) if len(values) else math.nan
            rec[f"{metric}__q25"] = float(np.quantile(values, 0.25)) if len(values) else math.nan
            rec[f"{metric}__q75"] = float(np.quantile(values, 0.75)) if len(values) else math.nan
            rec[f"{metric}__sd"] = float(values.std(ddof=1)) if len(values) >= 2 else math.nan
            rec[f"{metric}__ci_low"] = lo
            rec[f"{metric}__ci_high"] = hi
        records.append(rec)
    return pd.DataFrame(records)

def load_clean_reference_defense_tables(
    run_dir: Path,
    *,
    story_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load seed-level clean-reference defense summaries emitted by Stage 07.

    Stage 07 produces one summary per training seed. Cross-seed inference must
    therefore aggregate those seed-level checkpoint summaries, not selected
    channels pooled across runs.
    """
    task, phase = _task_and_phase(run_dir)
    story = story_root / task / run_dir.name / phase_dirname(phase) / "story"
    checkpoint_path = story / "clean_reference_benign_budget_checkpoint_summary.csv"
    curve_path = story / "clean_reference_benign_budget_curve.csv"
    screen_path = story / "clean_reference_benign_budget_screen.csv"
    if not checkpoint_path.is_file() or not curve_path.is_file():
        return pd.DataFrame(), pd.DataFrame()

    checkpoint = pd.read_csv(checkpoint_path)
    curve = pd.read_csv(curve_path)
    if checkpoint.empty or curve.empty:
        return pd.DataFrame(), pd.DataFrame()

    operating_budget = math.nan
    if screen_path.is_file() and screen_path.stat().st_size > 0:
        try:
            screen = pd.read_csv(screen_path)
            budgets = pd.to_numeric(
                screen.get("benign_damage_budget"), errors="coerce"
            ).dropna().unique()
            if len(budgets) == 1:
                operating_budget = float(budgets[0])
        except (pd.errors.EmptyDataError, OSError):
            pass
    if not np.isfinite(operating_budget):
        curve_budgets = pd.to_numeric(curve.get("benign_damage_budget"), errors="coerce")
        if np.isclose(
            curve_budgets.to_numpy(float), 0.30, atol=1e-12, rtol=0.0
        ).any():
            operating_budget = 0.30
    checkpoint["benign_damage_budget"] = operating_budget
    checkpoint["operating_benign_damage_budget"] = operating_budget
    curve["operating_benign_damage_budget"] = operating_budget

    metadata = _run_metadata(run_dir)
    return _attach_metadata(checkpoint, metadata), _attach_metadata(curve, metadata)


def aggregate_defense_seed_units(
    raw: pd.DataFrame,
    *,
    level: float,
    min_seeds: int,
) -> pd.DataFrame:
    """Aggregate clean-reference defense summaries with seed as replicate unit."""
    if raw.empty:
        return pd.DataFrame()
    raw = _ensure_identity_columns(raw)
    group_cols = [
        *EXPERIMENT_ID_COLUMNS,
        *DEFENSE_IDENTITY_COLUMNS,
        "target_fraction",
        "benign_damage_budget",
    ]
    records: List[Dict[str, Any]] = []
    for keys, group in raw.groupby(group_cols, dropna=False, sort=True):
        base = dict(zip(group_cols, keys))
        seeds = sorted(set(
            pd.to_numeric(group["training_seed"], errors="coerce").dropna().astype(int)
        ))
        rec: Dict[str, Any] = {
            **base,
            "n_training_seeds": len(seeds),
            "training_seeds": ",".join(str(seed) for seed in seeds),
            "developmental_claim_ready": len(seeds) >= int(min_seeds),
            "minimum_seeds_required": int(min_seeds),
            "small_seed_count_caution": bool(len(seeds) < 5),
        }
        for metric in DEFENSE_METRICS:
            if metric not in group.columns:
                continue
            per_seed = group[["training_seed", metric]].copy()
            per_seed[metric] = pd.to_numeric(per_seed[metric], errors="coerce")
            per_seed = per_seed.dropna(subset=[metric])
            if per_seed["training_seed"].duplicated().any():
                raise ValueError(
                    f"Duplicate clean-reference defense rows for {base} metric={metric}"
                )
            values = per_seed[metric].to_numpy(dtype=float)
            lo, hi = t_interval(values, level)
            rec[f"{metric}__n_seeds"] = int(len(values))
            rec[f"{metric}__mean"] = float(values.mean()) if len(values) else math.nan
            rec[f"{metric}__median"] = float(np.median(values)) if len(values) else math.nan
            rec[f"{metric}__q25"] = float(np.quantile(values, 0.25)) if len(values) else math.nan
            rec[f"{metric}__q75"] = float(np.quantile(values, 0.75)) if len(values) else math.nan
            rec[f"{metric}__sd"] = float(values.std(ddof=1)) if len(values) >= 2 else math.nan
            rec[f"{metric}__ci_low"] = lo
            rec[f"{metric}__ci_high"] = hi
        records.append(rec)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", required=True, help="Comma-separated completed poisoning run directories.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--story_root", default=None,
        help=(
            "Optional per_run_visualizations root containing Stage-07 story CSVs. "
            "When supplied, clean-reference defense summaries are aggregated across seeds."
        ),
    )
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--confidence_level", type=float, default=0.95)
    parser.add_argument("--min_seeds", type=int, default=3)
    parser.add_argument("--conversion_threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("--confidence_level must be in (0, 1)")
    if args.min_seeds < 2:
        raise ValueError("--min_seeds must be at least 2")

    run_dirs = [Path(path).expanduser() for path in _csv_list(args.run_dirs)]
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    defense_errors: list[dict[str, str]] = []
    if args.story_root:
        story_root = Path(args.story_root).expanduser()
        checkpoint_frames: list[pd.DataFrame] = []
        curve_frames: list[pd.DataFrame] = []
        for run_dir in run_dirs:
            try:
                checkpoint_frame, curve_frame = load_clean_reference_defense_tables(
                    run_dir, story_root=story_root
                )
                if not checkpoint_frame.empty:
                    checkpoint_frames.append(checkpoint_frame)
                if not curve_frame.empty:
                    curve_frames.append(curve_frame)
            except Exception as exc:
                defense_errors.append({
                    "run_dir": str(run_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                })
        defense_checkpoint_raw = (
            pd.concat(checkpoint_frames, ignore_index=True, sort=False)
            if checkpoint_frames else pd.DataFrame()
        )
        defense_curve_raw = (
            pd.concat(curve_frames, ignore_index=True, sort=False)
            if curve_frames else pd.DataFrame()
        )
        defense_checkpoint_raw.to_csv(
            out_dir / "clean_reference_defense_checkpoint_all_seeds.csv", index=False
        )
        aggregate_defense_seed_units(
            defense_checkpoint_raw,
            level=args.confidence_level,
            min_seeds=args.min_seeds,
        ).to_csv(
            out_dir / "clean_reference_defense_checkpoint_across_seeds.csv", index=False
        )
        defense_curve_raw.to_csv(
            out_dir / "clean_reference_defense_budget_curve_all_seeds.csv", index=False
        )
        aggregate_defense_seed_units(
            defense_curve_raw,
            level=args.confidence_level,
            min_seeds=args.min_seeds,
        ).to_csv(
            out_dir / "clean_reference_defense_budget_curve_across_seeds.csv", index=False
        )

    # Aggregate the detector first. This branch depends only on Stage 07 and is
    # intentionally independent of behavioral/backdoor-trajectory reporting.
    detection_frames = [load_detection_metrics(run_dir) for run_dir in run_dirs]
    detection_frames = [frame for frame in detection_frames if not frame.empty]
    detection_raw = pd.concat(detection_frames, ignore_index=True, sort=False) if detection_frames else pd.DataFrame()
    detection_raw.to_csv(out_dir / "poison_detection_metrics_all_seeds.csv", index=False)
    aggregate_detection_seed_units(
        detection_raw, level=args.confidence_level, min_seeds=args.min_seeds
    ).to_csv(out_dir / "poison_detection_metrics_across_seeds.csv", index=False)

    # Behavioral aggregation is optional. A missing or malformed Stage-05
    # artifact is recorded but cannot invalidate a successful Stage-07 detector
    # aggregation for the same run.
    frames = []
    behavior_trajectory_errors: list[dict[str, str]] = []
    for run_dir in run_dirs:
        try:
            frames.append(load_trajectory(run_dir))
        except Exception as exc:
            behavior_trajectory_errors.append({
                "run_dir": str(run_dir),
                "error": f"{type(exc).__name__}: {exc}",
            })
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    raw.to_csv(out_dir / "checkpoint_trajectories_all_seeds.csv", index=False)
    aggregate_seed_units(
        raw,
        metrics=_csv_list(args.metrics),
        level=args.confidence_level,
        min_seeds=args.min_seeds,
    ).to_csv(out_dir / "checkpoint_metrics_by_model_across_seeds.csv", index=False)
    timing = timing_by_seed(raw, args.conversion_threshold) if not raw.empty else pd.DataFrame()
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
            "developmental_claim_ready_interpretation": "configured minimum seed count met; this flag is not a power calculation or proof of confirmatory adequacy",
            "small_seed_count_caution_below": 5,
            "seed_interval_note": "Student-t intervals are untransformed and may extend beyond natural bounds for rate metrics; raw per-seed estimates are retained in the all-seeds tables",
            "default_plot_summary": "median with Q1-Q3 across training seeds",
            "available_plot_summaries": ["median_iqr", "mean_t_ci", "seed_traces"],
            "clean_reference_defense_replicate_unit": "training_seed; selected channels remain nested within each seed",
            "clean_reference_defense_story_root": str(Path(args.story_root).expanduser()) if args.story_root else None,
            "clean_reference_defense_errors": defense_errors,
            "conversion_threshold": args.conversion_threshold,
            "run_dirs": [str(p) for p in run_dirs],
            "behavior_trajectory_aggregation": "optional; detector aggregation runs first and does not require valid Stage-05 backdoor trajectories",
            "behavior_trajectory_errors": behavior_trajectory_errors,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote cross-seed poisoning summaries under {out_dir}")


if __name__ == "__main__":
    main()
