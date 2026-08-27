#!/usr/bin/env python3
"""Rank poisoned training examples from attack-cohort control-correctness disruption.

Stage 07 is trigger- and attacker-target-agnostic at channel selection and
example scoring.  It does not use trigger-lift circuit membership, trigger-lift
causal status, or the per-example poison label when constructing a score.  It
does require matched clean and suspicious/poisoned trajectories so that abnormal
control-correctness channel drift can be defined relative to normal fine-tuning.

For a singleton agonist channel j, U(j) is the probability, on one fixed
held-out attack-eligible non-target cohort, that intervening on j changes a baseline-correct
example to incorrect.  Candidate discovery still uses the attack-cohort control-correctness
CHA threshold tau, but U(j) is measured separately on the same row identities at
every matched checkpoint.  For a matched checkpoint
interval t0 -> t1 the descriptive excess drift is

    D_j = [U_p(j,t1) - U_p(j,t0)] - [U_c(j,t1) - U_c(j,t0)]

where p and c denote poisoned and clean training.  D_j is a clean-normalized
trajectory divergence statistic; it is not presented as a same-start causal
branch contrast.

Candidate membership is frozen by the ordinary-correctness CHA analysis at the
configured tau (0.3 by default).  Missing singleton evaluations are never
interpreted as U(j)=0.  Numerical disruption is computed only for channels with
materialized U(j) at all four matched states.  Membership changes are reported
separately as descriptive discovery events.

Training rows are ranked with a WANDA-style score.  For every selected channel,
its causal disruption magnitude weights the elementwise product between the
input activation to the corresponding trained projection and the absolute clean-normalized effective LoRA interval update, where the effective
adapter weight is scaling * (B @ A) and the scored update is
[(W_p,t1-W_p,t0)-(W_c,t1-W_c,t0)].  Grouped-query attention channels that map to the same
v-projection row are collapsed to one parameter row for scoring and are reported
explicitly in the mapping table.

The ground-truth ``is_poisoned`` flag is used only after scoring to evaluate the
ranking with ROC AUC, average precision, precision/recall at the true poison
count, and matched poison-versus-source comparisons.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from studies.poisoning.lib.checkpoint_manifest import read_checkpoint_manifest
from studies.poisoning.lib.completion_data import CausalCompletionDataset, CausalLMCollator
from studies.poisoning.lib.run_paths import (
    detection_dir,
    metadata_path,
    phase_dirname,
    checkpoint_progress_label,
    NORMAL_TASK_CORRECTNESS_DIRNAME,
    causal_dir,
    resolve_manifest_checkpoint_dir,
    training_condition_dir,
)
from studies.poisoning.lib.scientific_config import (
    SCIENTIFIC_TRAINING_CONFIG_FIELDS,
    scientific_training_config_payload,
)
from studies.poisoning.lib.scheduling import (
    build_paired_optimizer_exposure_order,
    build_uniform_exposure_order,
)
from studies.poisoning.tasks.registry import available_tasks, get_task_definition, infer_task_from_run
from studies.poisoning.stage07_plot_detection_implications import generate_implication_outputs

SCORING_SCHEMA_VERSION = 6
DEFAULT_REQUIRED_TAU = 0.3

_LORA_FACTOR_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.(?P<module>mlp\.down_proj|self_attn\.v_proj)"
    r"\.lora_(?P<factor>[AB])(?:\.[^.]+)?\.weight$"
)
_TARGET_MODULE_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.(?P<module>mlp\.down_proj|self_attn\.v_proj)$"
)
_MLP_RE = re.compile(r"^m(?P<layer>\d+)$")
_ATTN_RE = re.compile(r"^a(?P<layer>\d+)\.h(?P<head>\d+)$")
_TAU_RE = re.compile(r"-tau([0-9]+(?:\.[0-9]+)?)")


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _fraction_key(value: float) -> int:
    return int(round(float(value) * 1000.0))


def _interval_label(start_fraction: float, end_fraction: float) -> str:
    return f"interval_{100*float(start_fraction):05.1f}_to_{100*float(end_fraction):05.1f}pct".replace(".", "p")


def _tau_from_stats_dir(path: str | Path) -> float | None:
    stats = Path(path)
    for part in reversed(stats.parts):
        match = _TAU_RE.search(str(part))
        if match:
            return float(match.group(1))
    return None


def _safe_component(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return re.sub(r"_+", "_", text).strip("_") or "run"


def _ordinary_output_dir(
    run_dir: Path,
    row: Mapping[str, Any],
    phase: str,
    eval_intervention: str,
) -> Path:
    """Canonical Stage-03 ordinary-correctness output for one checkpoint."""
    return (
        causal_dir(run_dir)
        / str(row["condition"])
        / checkpoint_progress_label(row)
        / phase_dirname(phase)
        / NORMAL_TASK_CORRECTNESS_DIRNAME
        / f"eval_{_safe_component(eval_intervention)}"
    )


def _discover_ordinary_candidate_stats(
    run_dir: Path,
    row: Mapping[str, Any],
    phase: str,
    *,
    eval_intervention: str,
    required_tau: float,
) -> Path | None:
    """Find the ordinary CHA candidate ranking for a checkpoint.

    Stage 07 does not read the trigger-lift trajectory table.  Candidate
    membership is resolved directly from the ordinary-correctness Stage-03
    output.  A checkpoint whose ordinary CHA was legitimately undefined may
    return ``None``; its fixed-cohort U(j) can still be materialized for the
    union discovered at other checkpoints because ``feature_report/scores.csv``
    exists independently of CHA membership.
    """
    out = _ordinary_output_dir(run_dir, row, phase, eval_intervention)
    root = out / "rule_extraction_results" / "neuron_flip_rules" / "stats"
    if not root.is_dir():
        return None
    candidates: list[Path] = []
    for ranking in sorted(root.rglob("frozen_candidate_ranking.csv")):
        stats = ranking.parent
        tau = _tau_from_stats_dir(stats)
        if tau is not None and not math.isclose(tau, float(required_tau), rel_tol=0.0, abs_tol=1e-12):
            continue
        candidates.append(stats)
    if not candidates:
        return None

    # Accept only independent train-discovery/full-test ordinary artifacts.
    preferred: list[Path] = []
    for stats in candidates:
        scope = stats / "evaluation_scope.json"
        try:
            payload = json.loads(scope.read_text(encoding="utf-8")) if scope.is_file() else {}
        except Exception:
            payload = {}
        if (
            payload.get("candidate_discovery_split", "train") == "train"
            and payload.get("final_statistics_split") == "test"
            and payload.get("evaluation_baseline_subset", "all") == "all"
            and not bool(payload.get("exclude_discovery_rows_from_final_stats", False))
        ):
            preferred.append(stats)
    if not preferred:
        raise RuntimeError(
            "Ordinary candidate discovery must use train rows with final statistics on the full test cohort; "
            f"no qualifying artifact found under {root}"
        )
    pool = preferred
    if len(pool) > 1:
        identities = []
        for stats in pool:
            try:
                df = pd.read_csv(stats / "frozen_candidate_ranking.csv")
                ids = tuple(sorted(zip(df.get("layer_label", []), pd.to_numeric(df.get("neuron_id", []), errors="coerce"))))
            except Exception:
                ids = ()
            identities.append((stats, ids))
        unique = {ids for _, ids in identities}
        if len(unique) > 1:
            raise RuntimeError(
                "Multiple ordinary-correctness candidate rankings disagree for "
                f"{row['condition']} {checkpoint_progress_label(row)}: {[str(p) for p in pool]}"
            )
    return pool[0]


def _candidate_unit_set(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    ranking = Path(path) / "frozen_candidate_ranking.csv"
    if not ranking.is_file():
        return set()
    try:
        df = pd.read_csv(ranking)
    except pd.errors.EmptyDataError:
        return set()
    if "layer_label" not in df.columns and "layer_key" in df.columns:
        df["layer_label"] = df["layer_key"].astype(str)
    if "neuron_id" not in df.columns and "neuron" in df.columns:
        parts = df["neuron"].astype(str).str.rsplit(":", n=1, expand=True)
        if parts.shape[1] == 2:
            df["neuron_id"] = pd.to_numeric(parts[1], errors="coerce")
    if not {"layer_label", "neuron_id"}.issubset(df.columns):
        raise ValueError(f"Candidate ranking has no layer_label/neuron_id: {ranking}")
    df = df.dropna(subset=["layer_label", "neuron_id"])
    return {f"{str(a)}:{int(b)}" for a, b in zip(df["layer_label"], pd.to_numeric(df["neuron_id"], errors="raise"))}


def _build_candidate_union(candidate_stats: Mapping[tuple[str, int], Path | None]) -> pd.DataFrame:
    records: dict[str, dict[str, Any]] = {}
    for (condition, frac_key), stats in sorted(candidate_stats.items()):
        if stats is None:
            continue
        path = Path(stats) / "frozen_candidate_ranking.csv"
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if "layer_label" not in frame.columns and "layer_key" in frame.columns:
            frame["layer_label"] = frame["layer_key"].astype(str)
        if "neuron_id" not in frame.columns and "neuron" in frame.columns:
            parts = frame["neuron"].astype(str).str.rsplit(":", n=1, expand=True)
            if parts.shape[1] == 2:
                frame["neuron_id"] = pd.to_numeric(parts[1], errors="coerce")
        if not {"layer_label", "neuron_id"}.issubset(frame.columns):
            raise ValueError(f"Candidate ranking has no layer_label/neuron_id: {path}")
        for row in frame.dropna(subset=["layer_label", "neuron_id"]).to_dict("records"):
            layer = str(row["layer_label"]); neuron = int(row["neuron_id"])
            unit = f"{layer}:{neuron}"
            rec = records.setdefault(unit, {
                "layer_label": layer,
                "neuron_id": neuron,
                "unit_key": unit,
                "discovery_score": math.nan,
                "discovery_score_signed": math.nan,
                "discovery_baseline_subset": "positive",
                "candidate_seen_states": [],
            })
            score = pd.to_numeric(pd.Series([row.get("discovery_score")]), errors="coerce").iloc[0]
            old = pd.to_numeric(pd.Series([rec.get("discovery_score")]), errors="coerce").iloc[0]
            if pd.notna(score) and (pd.isna(old) or float(score) > float(old)):
                rec["discovery_score"] = float(score)
                signed = pd.to_numeric(pd.Series([row.get("discovery_score_signed")]), errors="coerce").iloc[0]
                rec["discovery_score_signed"] = float(signed) if pd.notna(signed) else math.nan
            rec["candidate_seen_states"].append(f"{condition}:{frac_key/1000.0:g}")
    rows = list(records.values())
    rows.sort(key=lambda r: (-float(r["discovery_score"]) if np.isfinite(r["discovery_score"]) else math.inf, r["unit_key"]))
    for rank, rec in enumerate(rows, start=1):
        rec["discovery_rank_global"] = rank
        rec["candidate_seen_states"] = ",".join(sorted(set(rec["candidate_seen_states"])))
        rec["ranking_source"] = "ordinary_candidate_union_across_matched_checkpoints"
    return pd.DataFrame(rows)


def _evaluation_cohort_ids_from_frame(
    frame: pd.DataFrame,
    *,
    source: str | Path,
) -> tuple[tuple[str, str], ...]:
    """Return ordered immutable (example ID, gold) identities from a score table."""
    gold_col = next(
        (c for c in ("label", "correct_answer_numeric", "correct_answer", "expected_label", "is_acceptable")
         if c in frame.columns and frame[c].notna().all()),
        None,
    )
    if gold_col is None:
        raise RuntimeError(
            "Fixed-cohort U(j) materialization requires a gold value; "
            f"columns in {source}: {list(frame.columns)}"
        )
    for id_col in ("source_row_id", "eval_example_id", "example_id"):
        if id_col not in frame.columns or not frame[id_col].notna().all():
            continue
        ids = frame[id_col].astype(str)
        if ids.duplicated().any():
            continue
        identities = tuple((str(i), str(g)) for i, g in zip(ids, frame[gold_col]))
        if len(set(identities)) == len(identities):
            return identities
    raise RuntimeError(
        "Fixed-cohort U(j) materialization requires unique immutable (ID, gold) identities using "
        f"source_row_id/eval_example_id/example_id; columns in {source}: {list(frame.columns)}"
    )


def _evaluation_cohort_ids(stats_dir: str | Path) -> tuple[tuple[str, str], ...]:
    """Return ordered immutable example identities plus gold values for U(j)."""
    scores = Path(stats_dir) / "scores.csv"
    if not scores.is_file():
        raise FileNotFoundError(f"Materialized U(j) directory has no scores.csv: {stats_dir}")
    frame = pd.read_csv(scores)
    return _evaluation_cohort_ids_from_frame(frame, source=scores)



def _safe_layer_key(layer_label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(layer_label)).strip("_")


def _baseline_correctness_array(stats_dir: str | Path, frame: pd.DataFrame) -> tuple[np.ndarray | None, str | None]:
    """Return the binary unablated-correctness vector used to define C->I flips.

    The column name is read from the materialized singleton statistics rather
    than guessed from task-specific names.  ``None`` is returned when the
    metadata are unavailable so the diagnostic decomposition cannot affect the
    primary detector.
    """
    stats = Path(stats_dir)
    global_path = stats / "flip_stats_global.json"
    if not global_path.is_file():
        return None, None
    try:
        payload = json.loads(global_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    col = payload.get("baseline_metric_col")
    if not col or str(col) not in frame.columns:
        return None, str(col) if col else None
    values = pd.to_numeric(frame[str(col)], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    positive = valid & np.isclose(values, 1.0, atol=1e-8, rtol=0.0)
    negative = valid & np.isclose(values, 0.0, atol=1e-8, rtol=0.0)
    if np.any(valid & ~(positive | negative)):
        bad = values[valid & ~(positive | negative)][:5].tolist()
        raise ValueError(f"Baseline predicate {col!r} is not binary in {stats}: examples={bad}")
    if not bool(valid.all()):
        return None, str(col)
    return positive.astype(bool), str(col)


def _joint_paired_disruption_inference(
    frame: pd.DataFrame,
    stats_by_state: Mapping[str, str | Path],
    *,
    draws: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    """Attach joint paired-bootstrap inference and baseline-drift diagnostics.

    All candidate channels are resampled together using the same fixed-cohort
    row multiplicities.  This both removes repeated CSV I/O and preserves the
    cross-channel dependence required for a max-statistic simultaneous band.
    The primary point estimate remains exactly

        D_j = (U_p1-U_p0) - (U_c1-U_c0).
    """
    out = frame.copy()
    if out.empty:
        return out
    for col, default in (
        ("disruption_ci_low", math.nan),
        ("disruption_ci_high", math.nan),
        ("disruption_ci_excludes_zero", False),
        ("disruption_simultaneous_ci_low", math.nan),
        ("disruption_simultaneous_ci_high", math.nan),
        ("disruption_simultaneous_ci_excludes_zero", False),
        ("disruption_fwer_p", math.nan),
        ("disruption_bootstrap_draws_valid", 0),
    ):
        out[col] = default
    out["disruption_multiplicity_method"] = "paired_row_bootstrap_max_abs_centered"

    if int(draws) <= 0:
        return out

    states = ("poisoned_start", "poisoned_end", "clean_start", "clean_end")
    state_frames: dict[str, pd.DataFrame] = {}
    identities: tuple[tuple[str, str], ...] | None = None
    baseline_arrays: dict[str, np.ndarray | None] = {}
    baseline_cols: dict[str, str | None] = {}
    for state in states:
        stats_dir = stats_by_state[state]
        state_frame = pd.read_csv(Path(stats_dir) / "scores.csv")
        state_ids = _evaluation_cohort_ids_from_frame(state_frame, source=Path(stats_dir) / "scores.csv")
        if identities is None:
            identities = state_ids
        elif state_ids != identities:
            raise RuntimeError(f"Joint paired bootstrap cohort mismatch: {state}")
        state_frames[state] = state_frame
        baseline_arrays[state], baseline_cols[state] = _baseline_correctness_array(stats_dir, state_frame)

    n = len(identities or ())
    if n <= 0:
        return out

    comparable_mask = out.get("complete_u_j_comparison", pd.Series(False, index=out.index)).map(_truthy).to_numpy(dtype=bool)
    comparable_indices = np.flatnonzero(comparable_mask)
    valid_indices: list[int] = []
    contrast_columns: list[np.ndarray] = []
    state_flip_by_unit: dict[int, dict[str, np.ndarray]] = {}
    for idx in comparable_indices.tolist():
        row = out.iloc[int(idx)]
        unit_key = str(row["unit_key"])
        layer_label, neuron_text = unit_key.rsplit(":", 1)
        flip_col = f"flip_c2i_{_safe_layer_key(layer_label)}_{int(neuron_text)}"
        flips: dict[str, np.ndarray] = {}
        if any(flip_col not in state_frames[state].columns for state in states):
            continue
        for state in states:
            flips[state] = state_frames[state][flip_col].fillna(False).map(_truthy).to_numpy(dtype=np.float64)
        contrast = flips["poisoned_end"] - flips["poisoned_start"] - flips["clean_end"] + flips["clean_start"]
        observed = float(contrast.mean())
        expected = float(pd.to_numeric(pd.Series([row.get("poisoning_excess_delta_u_j")]), errors="coerce").iloc[0])
        if not np.isfinite(expected) or not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"Joint bootstrap point estimate disagrees with materialized U(j) for {unit_key}: "
                f"row contrast={observed}, U(j) contrast={expected}"
            )
        valid_indices.append(int(idx))
        contrast_columns.append(contrast)
        state_flip_by_unit[int(idx)] = flips

    if not valid_indices:
        return out

    contrast_matrix = np.column_stack(contrast_columns).astype(np.float64, copy=False)  # [rows, channels]
    point = contrast_matrix.mean(axis=0)
    m = int(contrast_matrix.shape[1])
    rng = np.random.default_rng(int(seed))
    boot = np.empty((int(draws), m), dtype=np.float64)
    # Multinomial row counts are exactly equivalent to drawing n row indices
    # with replacement, but permit one BLAS matrix multiply for all channels.
    probs = np.full(n, 1.0 / float(n), dtype=np.float64)
    chunk = max(1, min(256, int(draws)))
    for start in range(0, int(draws), chunk):
        stop = min(int(draws), start + chunk)
        counts = rng.multinomial(n, probs, size=stop - start).astype(np.float64, copy=False)
        boot[start:stop] = (counts @ contrast_matrix) / float(n)

    alpha = (1.0 - float(confidence_level)) / 2.0
    pointwise_lo = np.quantile(boot, alpha, axis=0)
    pointwise_hi = np.quantile(boot, 1.0 - alpha, axis=0)
    centered_max = np.max(np.abs(boot - point[None, :]), axis=1)
    max_crit = float(np.quantile(centered_max, float(confidence_level)))
    sim_lo = point - max_crit
    sim_hi = point + max_crit

    for pos, idx in enumerate(valid_indices):
        out.at[idx, "disruption_ci_low"] = float(pointwise_lo[pos])
        out.at[idx, "disruption_ci_high"] = float(pointwise_hi[pos])
        out.at[idx, "disruption_ci_excludes_zero"] = bool(pointwise_lo[pos] > 0.0 or pointwise_hi[pos] < 0.0)
        out.at[idx, "disruption_simultaneous_ci_low"] = float(sim_lo[pos])
        out.at[idx, "disruption_simultaneous_ci_high"] = float(sim_hi[pos])
        out.at[idx, "disruption_simultaneous_ci_excludes_zero"] = bool(sim_lo[pos] > 0.0 or sim_hi[pos] < 0.0)
        # Family-wise adjusted max-statistic tail probability.  The +1 form
        # avoids reporting an exact zero with finite Monte Carlo draws.
        out.at[idx, "disruption_fwer_p"] = float((1.0 + np.sum(centered_max >= abs(point[pos]))) / (len(centered_max) + 1.0))
        out.at[idx, "disruption_bootstrap_draws_valid"] = int(draws)

    # Diagnostic decomposition: distinguish ordinary baseline-accuracy drift
    # from intervention susceptibility among rows that were baseline-correct.
    for state in states:
        baseline = baseline_arrays[state]
        out[f"{state}_baseline_metric_col"] = baseline_cols[state]
        out[f"{state}_baseline_accuracy"] = float(baseline.mean()) if baseline is not None else math.nan
        out[f"{state}_baseline_correct_n"] = int(baseline.sum()) if baseline is not None else math.nan
    if all(baseline_arrays[state] is not None for state in states):
        out["poisoned_baseline_accuracy_change"] = out["poisoned_end_baseline_accuracy"] - out["poisoned_start_baseline_accuracy"]
        out["clean_baseline_accuracy_change"] = out["clean_end_baseline_accuracy"] - out["clean_start_baseline_accuracy"]
        out["poisoning_excess_baseline_accuracy_change"] = out["poisoned_baseline_accuracy_change"] - out["clean_baseline_accuracy_change"]
        for idx in valid_indices:
            flips = state_flip_by_unit[idx]
            for state in states:
                baseline = baseline_arrays[state]
                assert baseline is not None
                denom = int(baseline.sum())
                susceptibility = float(flips[state].sum() / denom) if denom > 0 else math.nan
                out.at[idx, f"{state}_conditional_c2i_given_baseline_correct"] = susceptibility
            vals = {state: float(out.at[idx, f"{state}_conditional_c2i_given_baseline_correct"]) for state in states}
            out.at[idx, "poisoned_conditional_c2i_change"] = vals["poisoned_end"] - vals["poisoned_start"]
            out.at[idx, "clean_conditional_c2i_change"] = vals["clean_end"] - vals["clean_start"]
            out.at[idx, "poisoning_excess_conditional_c2i_change"] = (
                out.at[idx, "poisoned_conditional_c2i_change"] - out.at[idx, "clean_conditional_c2i_change"]
            )
    return out


def _materialization_complete(stats_dir: Path, union_units: set[str]) -> bool:
    required = [
        stats_dir / "frozen_candidate_ranking.csv",
        stats_dir / "flip_stats_by_neuron.csv",
        stats_dir / "flip_stats_global.json",
        stats_dir / "scores.csv",
        stats_dir / "evaluation_scope.json",
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        scope = json.loads((stats_dir / "evaluation_scope.json").read_text(encoding="utf-8"))
        if scope.get("candidate_discovery_split", "train") != "train":
            return False
        if scope.get("final_statistics_split") != "test":
            return False
        if scope.get("evaluation_baseline_subset", "all") != "all":
            return False
        if bool(scope.get("exclude_discovery_rows_from_final_stats", False)):
            return False
        if scope.get("sampling_max_points") not in (None, 0):
            return False
        observed = _candidate_unit_set(stats_dir)
        if observed != union_units:
            return False
        _evaluation_cohort_ids(stats_dir)
    except Exception:
        return False
    return True


def _ensure_fixed_u_j_materialization(
    *,
    run_dir: Path,
    row: Mapping[str, Any],
    task: str,
    phase: str,
    eval_intervention: str,
    candidate_union_csv: Path,
    output_root: Path,
    run_config: Mapping[str, Any],
    u_j_neuron_batch_size: int,
    overwrite: bool,
) -> Path:
    """Evaluate the fixed candidate union on one fixed held-out cohort."""
    state_root = output_root / str(row["condition"]) / checkpoint_progress_label(row)
    rules_dir = state_root / "neuron_flip_rules"
    stats_dir = rules_dir / "stats" / "fixed_test_all_candidates"
    # candidate_union_csv normally sits directly under Stage 07, not in a stats dir.
    union_frame = pd.read_csv(candidate_union_csv)
    union_units = {f"{str(a)}:{int(b)}" for a, b in zip(union_frame["layer_label"], pd.to_numeric(union_frame["neuron_id"], errors="raise"))}
    if not overwrite and _materialization_complete(stats_dir, union_units):
        return stats_dir
    if overwrite and state_root.exists():
        shutil.rmtree(state_root)

    ordinary_out = _ordinary_output_dir(run_dir, row, phase, eval_intervention)
    features = ordinary_out / "feature_report"
    if not (features / "scores.csv").is_file() or not (features / "features.json").is_file():
        raise FileNotFoundError(
            "Stage 07 requires the ordinary-correctness feature report for every matched checkpoint. "
            f"Missing under {features}; rerun checkpoint causal workflow with ordinary correctness enabled."
        )
    checkpoint_dir = resolve_manifest_checkpoint_dir(run_dir, row, must_exist=True)
    task_module = f"studies.poisoning.tasks.{task}:ORDINARY_TASK_SPEC"
    cmd = [
        sys.executable, "-m", "pipeline.stage07_refine_neuron_anchored_rules",
        "--task_module", task_module,
        "--ai_model", str(checkpoint_dir),
        "--rules_dir", str(rules_dir),
        "--features_scores_dir", str(features),
        "--circuit_agonists_path", str(ordinary_out),
        "--candidate_ranking_csv", str(candidate_union_csv),
        "--search_epsilon", "0",
        "--batch_size", "1",
        "--neuron_batch_size", str(int(u_j_neuron_batch_size)),
        "--sampling_max_points", "0",
        "--stats_dirname", "fixed_test_all_candidates",
        "--points_to_use_for_mean_ablation", "256",
        "--intervention", str(eval_intervention),
        "--evaluation_split", "test",
        "--evaluation_baseline_subset", "all",
        "--skip_agonist_metric_stats",
        "--no_tqdm_batches",
        "--seed", str(int(run_config.get("seed", 0))),
    ]
    if phase == "output_only":
        cmd.append("--decode_only")
    print("[u-j-materialization]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)
    if not _materialization_complete(stats_dir, union_units):
        raise RuntimeError(f"Fixed-cohort U(j) materialization is incomplete: {stats_dir}")
    return stats_dir


def _load_ordinary_singletons(path: str | Path | None) -> pd.DataFrame:
    """Load frozen ordinary agonists and their held-out singleton U(j)."""
    columns = [
        "layer_label", "neuron_id", "unit_key", "discovery_score",
        "discovery_score_signed", "discovery_rank_global", "u_j", "u_j_n_eval",
    ]
    if path in (None, "", "nan"):
        return pd.DataFrame(columns=columns)
    stats = Path(str(path))
    ranking_path = stats / "frozen_candidate_ranking.csv"
    flip_path = stats / "flip_stats_by_neuron.csv"
    if not ranking_path.is_file():
        raise FileNotFoundError(f"Ordinary singleton directory has no frozen_candidate_ranking.csv: {stats}")
    if not flip_path.is_file():
        raise RuntimeError(f"Ordinary-correctness candidate ranking has no held-out singleton table: {stats}")

    try:
        ranking = pd.read_csv(ranking_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)
    if "layer_label" not in ranking.columns and "layer_key" in ranking.columns:
        ranking["layer_label"] = ranking["layer_key"].astype(str)
    if "neuron_id" not in ranking.columns and "neuron" in ranking.columns:
        parts = ranking["neuron"].astype(str).str.rsplit(":", n=1, expand=True)
        if parts.shape[1] == 2:
            ranking["neuron_id"] = pd.to_numeric(parts[1], errors="coerce")
    if "layer_label" not in ranking.columns or "neuron_id" not in ranking.columns:
        raise ValueError(f"Ranking has no layer_label/neuron_id: {ranking_path}")
    ranking = ranking.dropna(subset=["layer_label", "neuron_id"]).copy()
    ranking["layer_label"] = ranking["layer_label"].astype(str)
    ranking["neuron_id"] = pd.to_numeric(ranking["neuron_id"], errors="raise").astype(int)
    ranking["unit_key"] = [f"{a}:{b}" for a, b in zip(ranking["layer_label"], ranking["neuron_id"])]

    try:
        flips = pd.read_csv(flip_path)
    except pd.errors.EmptyDataError:
        flips = pd.DataFrame(columns=["unit_key", "c2i_rate", "n_eval"])
    if "unit_key" not in flips.columns:
        if not {"layer_label", "neuron_id"}.issubset(flips.columns):
            raise ValueError(f"Singleton table has no unit identity: {flip_path}")
        flips["layer_label"] = flips["layer_label"].astype(str)
        flips["neuron_id"] = pd.to_numeric(flips["neuron_id"], errors="coerce")
        flips["unit_key"] = flips["layer_label"] + ":" + flips["neuron_id"].astype("Int64").astype(str)
    if "c2i_rate" not in flips.columns:
        raise ValueError(f"Singleton table has no c2i_rate, required for fixed-cohort U(j): {flip_path}")
    keep = [c for c in ("unit_key", "c2i_rate", "n_eval", "c2i_count") if c in flips.columns]
    flips = flips[keep].drop_duplicates("unit_key", keep="first")
    flips = flips.rename(columns={"c2i_rate": "u_j", "n_eval": "u_j_n_eval"})
    merged = ranking.merge(flips, on="unit_key", how="left", validate="one_to_one")
    merged["u_j"] = pd.to_numeric(merged["u_j"], errors="coerce")
    bad = merged["u_j"].notna() & ~merged["u_j"].between(0.0, 1.0, inclusive="both")
    if bad.any():
        raise ValueError(f"U(j) must be a probability in [0,1]: {stats}")
    return merged


def _singleton_map(path: str | Path | None) -> dict[str, dict[str, Any]]:
    df = _load_ordinary_singletons(path)
    out: dict[str, dict[str, Any]] = {}
    for row in df.to_dict("records"):
        key = str(row["unit_key"])
        if key in out:
            raise RuntimeError(f"Duplicate ordinary-correctness singleton {key} in {path}")
        out[key] = row
    return out


def compute_channel_disruption(
    *,
    start_fraction: float,
    end_fraction: float,
    poisoned_start_stats: str | Path,
    poisoned_end_stats: str | Path,
    clean_start_stats: str | Path,
    clean_end_stats: str | Path,
    poisoned_start_candidates: str | Path | None = None,
    poisoned_end_candidates: str | Path | None = None,
    clean_start_candidates: str | Path | None = None,
    clean_end_candidates: str | Path | None = None,
    bootstrap_draws: int = 2000,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_seed: int = 9173,
) -> pd.DataFrame:
    """Compare fixed-cohort ordinary singleton U(j) across matched states.

    ``*_stats`` are Stage-07 materializations of the same candidate union at
    all four states.  ``*_candidates`` are the original checkpoint-local CHA
    rankings and are used only to describe discovery membership changes.
    """
    p0 = _singleton_map(poisoned_start_stats)
    p1 = _singleton_map(poisoned_end_stats)
    c0 = _singleton_map(clean_start_stats)
    c1 = _singleton_map(clean_end_stats)
    units = sorted(set(p0) | set(p1) | set(c0) | set(c1))

    cohort = {
        "poisoned_start": _evaluation_cohort_ids(poisoned_start_stats),
        "poisoned_end": _evaluation_cohort_ids(poisoned_end_stats),
        "clean_start": _evaluation_cohort_ids(clean_start_stats),
        "clean_end": _evaluation_cohort_ids(clean_end_stats),
    }
    cohort_values = list(cohort.values())
    same_cohort = bool(cohort_values) and all(ids == cohort_values[0] for ids in cohort_values[1:])
    fixed_cohort_n = len(cohort_values[0]) if same_cohort else None

    presence = {
        "poisoned_start": _candidate_unit_set(poisoned_start_candidates) if poisoned_start_candidates is not None else set(p0),
        "poisoned_end": _candidate_unit_set(poisoned_end_candidates) if poisoned_end_candidates is not None else set(p1),
        "clean_start": _candidate_unit_set(clean_start_candidates) if clean_start_candidates is not None else set(c0),
        "clean_end": _candidate_unit_set(clean_end_candidates) if clean_end_candidates is not None else set(c1),
    }

    rows: list[dict[str, Any]] = []
    for unit_index, unit in enumerate(units):
        template = p1.get(unit) or p0.get(unit) or c1.get(unit) or c0.get(unit)
        if template is None:
            continue
        payloads = {
            "poisoned_start": p0.get(unit), "poisoned_end": p1.get(unit),
            "clean_start": c0.get(unit), "clean_end": c1.get(unit),
        }
        u_values: dict[str, float] = {}
        missing_labels: list[str] = []
        n_eval_values: dict[str, float] = {}
        for label, payload in payloads.items():
            if payload is None:
                u_values[label] = math.nan
                n_eval_values[label] = math.nan
                missing_labels.append(label)
                continue
            value = pd.to_numeric(pd.Series([payload.get("u_j")]), errors="coerce").iloc[0]
            n_value = pd.to_numeric(pd.Series([payload.get("u_j_n_eval")]), errors="coerce").iloc[0]
            u_values[label] = float(value) if pd.notna(value) else math.nan
            n_eval_values[label] = float(n_value) if pd.notna(n_value) else math.nan
            if pd.isna(value):
                missing_labels.append(label)

        complete = not missing_labels
        finite_n = [int(v) for v in n_eval_values.values() if np.isfinite(v)]
        all_eval_n_reported = len(finite_n) == 4
        same_eval_n = bool(all_eval_n_reported and len(set(finite_n)) == 1)
        eval_n_matches_cohort = bool(same_eval_n and fixed_cohort_n is not None and finite_n[0] == int(fixed_cohort_n))
        comparable = bool(complete and same_eval_n and eval_n_matches_cohort and same_cohort)
        p_change = u_values["poisoned_end"] - u_values["poisoned_start"] if comparable else math.nan
        c_change = u_values["clean_end"] - u_values["clean_start"] if comparable else math.nan
        excess = p_change - c_change if comparable else math.nan
        if not complete:
            comparison_status = "missing_u_j:" + ",".join(missing_labels)
        elif not same_cohort:
            comparison_status = "u_j_evaluation_cohort_mismatch"
        elif not all_eval_n_reported:
            comparison_status = "u_j_evaluation_n_missing"
        elif not same_eval_n:
            comparison_status = "u_j_evaluation_n_mismatch"
        elif not eval_n_matches_cohort:
            comparison_status = "u_j_evaluation_n_not_full_fixed_cohort"
        else:
            comparison_status = "complete"

        row = {
            "start_fraction": float(start_fraction),
            "end_fraction": float(end_fraction),
            "layer_label": str(template["layer_label"]),
            "neuron_id": int(template["neuron_id"]),
            "unit_key": unit,
            "u_j_definition": "fixed_test_cohort_correct_to_incorrect_rate",
            "poisoned_u_j_start": u_values["poisoned_start"],
            "poisoned_u_j_end": u_values["poisoned_end"],
            "clean_u_j_start": u_values["clean_start"],
            "clean_u_j_end": u_values["clean_end"],
            "poisoned_u_j_n_eval_start": n_eval_values["poisoned_start"],
            "poisoned_u_j_n_eval_end": n_eval_values["poisoned_end"],
            "clean_u_j_n_eval_start": n_eval_values["clean_start"],
            "clean_u_j_n_eval_end": n_eval_values["clean_end"],
            "fixed_cohort_n": fixed_cohort_n,
            "same_u_j_evaluation_cohort": bool(same_cohort),
            "all_u_j_evaluation_n_reported": bool(all_eval_n_reported),
            "same_u_j_evaluation_n": bool(same_eval_n),
            "u_j_evaluation_n_matches_full_fixed_cohort": bool(eval_n_matches_cohort),
            "poisoned_delta_u_j": p_change,
            "clean_delta_u_j": c_change,
            "poisoning_excess_delta_u_j": excess,
            "disruption_score": abs(excess) if comparable else math.nan,
            "complete_u_j_comparison": bool(comparable),
            "comparison_status": comparison_status,
            "present_poisoned_start": unit in presence["poisoned_start"],
            "present_poisoned_end": unit in presence["poisoned_end"],
            "present_clean_start": unit in presence["clean_start"],
            "present_clean_end": unit in presence["clean_end"],
            "candidate_membership_entered_poisoned": bool(unit not in presence["poisoned_start"] and unit in presence["poisoned_end"]),
            "candidate_membership_left_poisoned": bool(unit in presence["poisoned_start"] and unit not in presence["poisoned_end"]),
            "candidate_membership_entered_clean": bool(unit not in presence["clean_start"] and unit in presence["clean_end"]),
            "candidate_membership_left_clean": bool(unit in presence["clean_start"] and unit not in presence["clean_end"]),
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = _joint_paired_disruption_inference(
        out,
        {
            "poisoned_start": poisoned_start_stats, "poisoned_end": poisoned_end_stats,
            "clean_start": clean_start_stats, "clean_end": clean_end_stats,
        },
        draws=int(bootstrap_draws),
        confidence_level=float(bootstrap_confidence_level),
        seed=int(bootstrap_seed),
    )
    out["_sort"] = pd.to_numeric(out["disruption_score"], errors="coerce").fillna(-1.0)
    return out.sort_values(["_sort", "unit_key"], ascending=[False, True]).drop(columns="_sort").reset_index(drop=True)


def select_disruptive_channels(
    frame: pd.DataFrame,
    *,
    max_channels: int,
    min_abs_delta_u: float,
    min_clean_null_z: float | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = frame.get("complete_u_j_comparison", pd.Series(False, index=frame.index)).map(_truthy)
    score = pd.to_numeric(frame.get("disruption_score"), errors="coerce")
    mask &= score.notna() & (score >= float(min_abs_delta_u))
    ci_ok = frame.get("disruption_simultaneous_ci_excludes_zero", pd.Series(False, index=frame.index)).map(_truthy)
    mask &= ci_ok
    if min_clean_null_z is not None and "clean_null_abs_z" in frame.columns:
        z = pd.to_numeric(frame["clean_null_abs_z"], errors="coerce")
        mask &= z.notna() & (z >= float(min_clean_null_z))
    out = frame.loc[mask].copy().sort_values(["disruption_score", "unit_key"], ascending=[False, True])
    if int(max_channels) > 0:
        out = out.head(int(max_channels))
    out = out.reset_index(drop=True)
    out["disruption_rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def _load_manifest_by_condition(run_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    path = metadata_path(run_dir, "checkpoint_manifest_all.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Missing aggregate checkpoint manifest: {path}")
    rows = read_checkpoint_manifest(path)
    out: dict[str, dict[int, dict[str, Any]]] = {"clean": {}, "poisoned": {}}
    for row in rows:
        condition = str(row.get("condition", "")).strip()
        if condition not in out:
            continue
        row = dict(row)
        row["fraction"] = float(row["fraction"])
        row["global_step"] = int(float(row["global_step"]))
        out[condition][_fraction_key(row["fraction"])] = row
    if not out["clean"] or not out["poisoned"]:
        raise ValueError("Stage 07 requires both clean and poisoned checkpoint trajectories")
    return out


def _clean_delta_map(stats0: str | Path, stats1: str | Path) -> dict[str, float]:
    ids0 = _evaluation_cohort_ids(stats0)
    ids1 = _evaluation_cohort_ids(stats1)
    if ids0 != ids1:
        raise RuntimeError(f"Clean-null U(j) cohort mismatch: {stats0} vs {stats1}")
    a, b = _singleton_map(stats0), _singleton_map(stats1)
    out: dict[str, float] = {}
    for unit in set(a) & set(b):
        u0 = pd.to_numeric(pd.Series([a[unit].get("u_j")]), errors="coerce").iloc[0]
        u1 = pd.to_numeric(pd.Series([b[unit].get("u_j")]), errors="coerce").iloc[0]
        if pd.notna(u0) and pd.notna(u1):
            out[unit] = float(u1) - float(u0)
    return out


def annotate_clean_null(
    frame: pd.DataFrame,
    *,
    additional_clean_materializations: Sequence[tuple[Path, Path]],
) -> pd.DataFrame:
    """Attach an empirical normal-training delta-U(j) null.

    The matched clean trajectory in ``frame`` is one independent realization.
    Each tuple in ``additional_clean_materializations`` contributes one distinct
    clean training seed, already cross-evaluated on that run's fixed cohort.
    """
    if frame.empty:
        out = frame.copy()
        for name, dtype in (
            ("clean_null_n", int), ("clean_null_mean_delta_u_j", float),
            ("clean_null_sd_delta_u_j", float), ("clean_null_z", float),
            ("clean_null_abs_z", float),
        ):
            out[name] = pd.Series(dtype=dtype)
        return out

    samples: dict[str, list[float]] = {str(u): [] for u in frame["unit_key"].astype(str)}
    for row in frame.to_dict("records"):
        value = pd.to_numeric(pd.Series([row.get("clean_delta_u_j")]), errors="coerce").iloc[0]
        if pd.notna(value):
            samples[str(row["unit_key"])].append(float(value))

    for start_stats, end_stats in additional_clean_materializations:
        delta = _clean_delta_map(start_stats, end_stats)
        for unit in samples:
            if unit in delta:
                samples[unit].append(float(delta[unit]))

    out = frame.copy()
    n_list: list[int] = []
    means: list[float] = []
    sds: list[float] = []
    zs: list[float] = []
    for row in out.to_dict("records"):
        vals = np.asarray(samples.get(str(row["unit_key"]), []), dtype=float)
        vals = vals[np.isfinite(vals)]
        n = int(len(vals))
        mean = float(vals.mean()) if n else math.nan
        sd = float(vals.std(ddof=1)) if n >= 2 else math.nan
        pdelta = float(row.get("poisoned_delta_u_j", math.nan))
        z = (pdelta - mean) / sd if n >= 3 and np.isfinite(sd) and sd > 0 and np.isfinite(pdelta) else math.nan
        n_list.append(n); means.append(mean); sds.append(sd); zs.append(z)
    out["clean_null_n"] = n_list
    out["clean_null_mean_delta_u_j"] = means
    out["clean_null_sd_delta_u_j"] = sds
    out["clean_null_z"] = zs
    out["clean_null_abs_z"] = np.abs(pd.to_numeric(pd.Series(zs), errors="coerce"))
    return out


def reconstruct_training_order(run_dir: Path, n_train: int, run_config: Mapping[str, Any]) -> list[int]:
    condition_dir = training_condition_dir(run_dir, "poisoned")
    poison_meta = json.loads((condition_dir / "poison_meta.json").read_text(encoding="utf-8"))
    schedule = json.loads((condition_dir / "poison_schedule.json").read_text(encoding="utf-8"))
    mode = str(schedule.get("mode", run_config.get("poison_schedule_mode", "trainer_random")))
    if mode != "uniform_optimizer_steps":
        raise RuntimeError(
            "Per-datapoint checkpoint ranking requires poison_schedule_mode=uniform_optimizer_steps; "
            "the exact Trainer random-sampler order is not persisted."
        )
    persisted_order = schedule.get("training_exposure_order")
    if isinstance(persisted_order, list) and len(persisted_order) == int(n_train):
        order = [int(i) for i in persisted_order]
        if sorted(order) != list(range(int(n_train))):
            raise RuntimeError("Persisted training_exposure_order is not a permutation of the training rows")
        return order

    sources = [int(i) for i in poison_meta.get("paired_source_indices", [])]
    slots = [int(i) for i in poison_meta.get("paired_slot_indices", [])]
    seed = int(run_config["seed"])
    batch = int(run_config["per_device_train_batch_size"])
    accum = int(run_config["gradient_accumulation_steps"])
    if str(poison_meta.get("poison_training_mode")) == "paired_counterfactual" and slots:
        return build_paired_optimizer_exposure_order(
            n_train, sources, slots, seed=seed,
            per_device_batch_size=batch, gradient_accumulation_steps=accum,
        )
    return build_uniform_exposure_order(n_train, slots, seed=seed)


def exposures_between_steps(
    order: Sequence[int],
    *,
    start_step: int,
    end_step: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
) -> list[tuple[int, int]]:
    """Return ``(stream_position, training_slot)`` exposures in ``(start,end]``."""
    if end_step < start_step:
        raise ValueError("end_step must be >= start_step")
    n = len(order)
    if n == 0 or end_step == start_step:
        return []
    batch = int(per_device_batch_size)
    accum = int(gradient_accumulation_steps)
    if batch <= 0 or accum <= 0:
        raise ValueError("batch and accumulation must be positive")
    micro_batches_per_epoch = int(math.ceil(n / batch))
    steps_per_epoch = int(math.ceil(micro_batches_per_epoch / accum))

    def stream_until(step: int) -> list[int]:
        step = max(0, int(step))
        full_epochs, within_steps = divmod(step, steps_per_epoch)
        consumed_within = min(n, within_steps * accum * batch)
        return list(order) * full_epochs + list(order[:consumed_within])

    before = stream_until(start_step)
    after = stream_until(end_step)
    if len(after) < len(before) or after[: len(before)] != before:
        raise RuntimeError("Could not reconstruct a monotone training exposure stream")
    return [(pos, int(slot)) for pos, slot in enumerate(after[len(before):], start=len(before))]


def _adapter_tensor_file(checkpoint_dir: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = Path(checkpoint_dir) / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Stage 07 requires LoRA adapter checkpoint weights: {checkpoint_dir}")


def _load_adapter_tensors(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    path = _adapter_tensor_file(checkpoint_dir)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Loading PEFT safetensors requires safetensors") from exc
        raw = load_file(str(path), device="cpu")
    else:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    return {str(k): v.detach().cpu().float() for k, v in raw.items() if torch.is_tensor(v)}


def _adapter_scaling(checkpoint_dir: Path, *, module_name: str, rank: int) -> float:
    cfg_path = Path(checkpoint_dir) / "adapter_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing PEFT adapter_config.json: {checkpoint_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if bool(cfg.get("use_dora", False)):
        raise RuntimeError("Stage 07 WANDA scoring does not support DoRA adapters")
    if bool(cfg.get("fan_in_fan_out", False)):
        raise RuntimeError("Stage 07 WANDA scoring expects standard Linear LoRA weights (fan_in_fan_out=false)")

    def _matches(module: str, pattern: object) -> bool:
        p = str(pattern).strip()
        return bool(p) and (module == p or module.endswith("." + p) or module.endswith(p) or p.endswith("." + module) or p.endswith(module))

    alpha = float(cfg.get("lora_alpha", rank))
    for pattern, value in (cfg.get("alpha_pattern") or {}).items():
        if _matches(module_name, pattern):
            alpha = float(value)
            break
    effective_rank = int(rank)
    for pattern, value in (cfg.get("rank_pattern") or {}).items():
        if _matches(module_name, pattern):
            effective_rank = int(value)
            break
    if effective_rank != int(rank):
        raise RuntimeError(
            f"Adapter rank pattern declares r={effective_rank} but tensor shape implies r={rank} for {module_name}"
        )
    return alpha / (math.sqrt(rank) if bool(cfg.get("use_rslora", False)) else rank)


def _effective_lora_weight_map(checkpoint_dir: Path) -> dict[tuple[int, str], torch.Tensor]:
    factors: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    names: dict[tuple[int, str], str] = {}
    for name, tensor in _load_adapter_tensors(checkpoint_dir).items():
        match = _LORA_FACTOR_RE.search(name)
        if not match:
            continue
        key = (int(match.group("layer")), str(match.group("module")))
        factor = str(match.group("factor"))
        if factor in factors.setdefault(key, {}):
            raise RuntimeError(f"Duplicate LoRA factor {factor} for {key} in {checkpoint_dir}")
        factors[key][factor] = tensor
        names[key] = f"layers.{key[0]}.{key[1]}"
    out: dict[tuple[int, str], torch.Tensor] = {}
    for key, pair in factors.items():
        if set(pair) != {"A", "B"}:
            raise RuntimeError(f"Incomplete LoRA A/B factors for {key} in {checkpoint_dir}")
        A, B = pair["A"], pair["B"]
        if A.ndim != 2 or B.ndim != 2 or B.shape[1] != A.shape[0]:
            raise RuntimeError(f"Unexpected LoRA factor shapes A={tuple(A.shape)} B={tuple(B.shape)} for {key}")
        scaling = _adapter_scaling(checkpoint_dir, module_name=names[key], rank=int(A.shape[0]))
        out[key] = (B @ A) * float(scaling)
    if not out:
        raise RuntimeError(f"No supported LoRA A/B factors found in {checkpoint_dir}")
    return out


def _base_config(model: Any) -> Any:
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "hidden_size"):
        return cfg
    base = getattr(model, "base_model", None)
    cfg = getattr(base, "config", None)
    if cfg is not None:
        return cfg
    raise RuntimeError("Could not resolve base model configuration")


def coordinate_to_parameter_row(model: Any, layer_label: str, neuron_id: int) -> tuple[int, str, int, dict[str, Any]]:
    mlp = _MLP_RE.match(str(layer_label))
    if mlp:
        return int(mlp.group("layer")), "mlp.down_proj", int(neuron_id), {"mapping_kind": "mlp_residual_write"}
    attn = _ATTN_RE.match(str(layer_label))
    if not attn:
        raise ValueError(f"Unsupported overtopping channel label: {layer_label!r}")
    cfg = _base_config(model)
    layer = int(attn.group("layer")); head = int(attn.group("head"))
    n_heads = int(getattr(cfg, "num_attention_heads"))
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads) or n_heads)
    hidden = int(getattr(cfg, "hidden_size"))
    head_dim = int(getattr(cfg, "head_dim", hidden // n_heads) or (hidden // n_heads))
    if n_heads <= 0 or n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
        raise RuntimeError(f"Unsupported grouped-query attention: heads={n_heads} kv_heads={n_kv_heads}")
    if not (0 <= head < n_heads) or not (0 <= int(neuron_id) < head_dim):
        raise IndexError(f"Invalid attention coordinate {layer_label}:{neuron_id}")
    query_heads_per_kv = n_heads // n_kv_heads
    kv_head = head // query_heads_per_kv
    row = kv_head * head_dim + int(neuron_id)
    return layer, "self_attn.v_proj", row, {
        "mapping_kind": "attention_value_projection_proxy",
        "query_head": head,
        "kv_head": kv_head,
        "query_heads_per_kv_head": query_heads_per_kv,
    }


def _load_scoring_model(run_config: Mapping[str, Any], checkpoint_dir: Path):
    from studies.poisoning.lib.training import load_base_model, place_model_for_eval
    try:
        from peft import PeftModel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Stage 07 WANDA scoring requires peft") from exc
    args = argparse.Namespace(**dict(run_config))
    base = load_base_model(args)
    model = PeftModel.from_pretrained(base, str(checkpoint_dir), is_trainable=False)
    model = place_model_for_eval(model)
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    model.eval()
    return model


def _find_target_modules(model: Any) -> dict[tuple[int, str], tuple[str, torch.nn.Module]]:
    out: dict[tuple[int, str], tuple[str, torch.nn.Module]] = {}
    for name, module in model.named_modules():
        match = _TARGET_MODULE_RE.search(name)
        if not match:
            continue
        key = (int(match.group("layer")), str(match.group("module")))
        # PEFT can expose nested base_layer modules too; the exact wrapper name
        # wins because its input is the projection input used by B@A.
        current = out.get(key)
        if current is None or len(name) < len(current[0]):
            out[key] = (name, module)
    return out


def _map_channels_and_interval_updates(
    model: Any,
    selected_channels: pd.DataFrame,
    *,
    poisoned_start: Path,
    poisoned_end: Path,
    clean_start: Path,
    clean_end: Path,
    matched_random_seed: int,
    matched_control_draws: int,
    all_candidate_channels: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[tuple[int, str], dict[str, Any]]]:
    """Map causal channels to clean-normalized effective LoRA update rows.

    WANDA is scored against
        (W_p,end - W_p,start) - (W_c,end - W_c,start),
    where each W is the effective adapter matrix scaling*(B@A).  Matched
    non-candidate rows come from the same projection and are greedily matched on
    the L2 norm of this excess-update row, without replacement. Multiple
    independently sampled matched sets from a near-norm neighborhood are retained for control inference.
    """
    p0 = _effective_lora_weight_map(poisoned_start)
    p1 = _effective_lora_weight_map(poisoned_end)
    c0 = _effective_lora_weight_map(clean_start)
    c1 = _effective_lora_weight_map(clean_end)
    modules = _find_target_modules(model)
    records: list[dict[str, Any]] = []
    grouped: dict[tuple[int, str], dict[str, Any]] = {}

    for source in selected_channels.to_dict("records"):
        layer, module_name, row, extra = coordinate_to_parameter_row(
            model, str(source["layer_label"]), int(source["neuron_id"])
        )
        key = (layer, module_name)
        mappings = (("poisoned_start", p0), ("poisoned_end", p1), ("clean_start", c0), ("clean_end", c1), ("model", modules))
        missing = [label for label, mapping in mappings if key not in mapping]
        if missing:
            records.append({**source, "mapped": False, "mapping_error": f"missing {key} in {','.join(missing)}"})
            continue
        shapes = {tuple(mapping[key].shape) for _, mapping in mappings[:-1]}
        if len(shapes) != 1:
            raise RuntimeError(f"Effective LoRA shape differs across matched states for {key}: {shapes}")
        shape = next(iter(shapes))
        if not 0 <= row < shape[0]:
            raise IndexError(f"Mapped row {row} outside effective LoRA shape {shape} for {source['unit_key']}")
        parameter_row_key = f"L{layer}:{module_name}:row{row}"
        rec = {
            **source,
            "mapped": True,
            "lora_layer": layer,
            "lora_module": module_name,
            "effective_weight_row": row,
            "unique_parameter_row_key": parameter_row_key,
            **extra,
        }
        records.append(rec)
        group = grouped.setdefault(key, {"module_name": modules[key][0], "module": modules[key][1], "rows": {}})
        row_group = group["rows"].setdefault(row, {"channels": [], "disruption_scores": []})
        row_group["channels"].append(str(source["unit_key"]))
        row_group["disruption_scores"].append(float(source["disruption_score"]))

    excluded_rows_by_module: dict[tuple[int, str], set[int]] = {}
    exclusion_frame = all_candidate_channels if all_candidate_channels is not None else selected_channels
    for source in exclusion_frame.to_dict("records"):
        try:
            layer, module_name, row, _ = coordinate_to_parameter_row(
                model, str(source["layer_label"]), int(source["neuron_id"])
            )
        except (KeyError, TypeError, ValueError, IndexError, RuntimeError):
            continue
        excluded_rows_by_module.setdefault((layer, module_name), set()).add(int(row))

    total_weight = 0.0
    if int(matched_control_draws) <= 0:
        raise ValueError("matched_control_draws must be > 0")
    for key, group in grouped.items():
        p_delta = p1[key] - p0[key]
        c_delta = c1[key] - c0[key]
        excess = p_delta - c_delta
        selected_rows = sorted(int(row) for row in group["rows"])
        excluded_rows = excluded_rows_by_module.get(key, set(selected_rows)) | set(selected_rows)
        base_available = [row for row in range(int(excess.shape[0])) if row not in excluded_rows]
        if len(base_available) < len(selected_rows):
            raise RuntimeError(f"Not enough non-candidate rows for a cardinality-matched WANDA control in {key}")

        for row in selected_rows:
            info = group["rows"][row]
            selected_delta = excess[int(row)].contiguous()
            selected_norm = float(torch.linalg.vector_norm(selected_delta).item())
            info["effective_excess_interval_delta_row"] = selected_delta
            info["poisoned_interval_delta_row"] = p_delta[int(row)].contiguous()
            info["clean_interval_delta_row"] = c_delta[int(row)].contiguous()
            info["effective_excess_interval_delta_norm"] = selected_norm
            info["matched_control_rows"] = []
            info["matched_control_excess_interval_delta_rows"] = []
            info["matched_control_excess_interval_delta_norms"] = []
            info["matched_control_update_norm_abs_differences"] = []
            info["row_disruption_weight_raw"] = max(info["disruption_scores"]) if info["disruption_scores"] else 0.0
            total_weight += float(info["row_disruption_weight_raw"])

        available_norms = {int(row): float(torch.linalg.vector_norm(excess[int(row)]).item()) for row in base_available}
        for draw in range(int(matched_control_draws)):
            rng = np.random.default_rng(int(matched_random_seed) + draw * 104729)
            available = list(np.asarray(base_available, dtype=int)[rng.permutation(len(base_available))])
            for row in selected_rows:
                info = group["rows"][row]
                selected_norm = float(info["effective_excess_interval_delta_norm"])
                nearest = sorted(available, key=lambda candidate: abs(available_norms[int(candidate)] - selected_norm))[: min(8, len(available))]
                control_row = int(nearest[int(rng.integers(0, len(nearest)))])
                available.remove(control_row)
                control_delta = excess[int(control_row)].contiguous()
                control_norm = float(available_norms[int(control_row)])
                info["matched_control_rows"].append(int(control_row))
                info["matched_control_excess_interval_delta_rows"].append(control_delta)
                info["matched_control_excess_interval_delta_norms"].append(control_norm)
                info["matched_control_update_norm_abs_differences"].append(abs(control_norm - selected_norm))

        # Backward-compatible aliases point to the first matched realization.
        for row in selected_rows:
            info = group["rows"][row]
            info["matched_random_row"] = int(info["matched_control_rows"][0])
            info["matched_random_excess_interval_delta_row"] = info["matched_control_excess_interval_delta_rows"][0]
            info["matched_random_excess_interval_delta_norm"] = float(info["matched_control_excess_interval_delta_norms"][0])
            info["matched_random_update_norm_abs_difference"] = float(info["matched_control_update_norm_abs_differences"][0])
    if total_weight <= 0:
        total_weight = 1.0
    for group in grouped.values():
        for info in group["rows"].values():
            info["row_disruption_weight"] = float(info["row_disruption_weight_raw"]) / total_weight

        # Prepare immutable per-interval tensors once.  Per-example WANDA then
        # reduces to matrix products instead of rebuilding candidate/control
        # delta tensors for every training exposure.
        row_order = sorted(int(row) for row in group["rows"])
        group["row_order"] = row_order
        group["candidate_abs_delta_rows"] = torch.stack([
            torch.abs(group["rows"][row]["effective_excess_interval_delta_row"].to(torch.float32)).cpu()
            for row in row_order
        ], dim=0)
        group["row_weights"] = torch.tensor(
            [float(group["rows"][row]["row_disruption_weight"]) for row in row_order],
            dtype=torch.float32,
        )
        n_draws = min(len(group["rows"][row].get("matched_control_excess_interval_delta_rows", [])) for row in row_order)
        if n_draws <= 0:
            raise RuntimeError(f"Matched WANDA controls are missing for projection {key}")
        group["control_abs_delta_rows"] = torch.stack([
            torch.stack([
                torch.abs(group["rows"][row]["matched_control_excess_interval_delta_rows"][draw].to(torch.float32)).cpu()
                for row in row_order
            ], dim=0)
            for draw in range(n_draws)
        ], dim=0)
        group["candidate_update_sq"] = float(sum(
            torch.sum(group["rows"][row]["effective_excess_interval_delta_row"].to(torch.float32) ** 2).item()
            for row in row_order
        ))
        group["matched_random_update_sq"] = float(sum(
            torch.sum(group["rows"][row]["matched_control_excess_interval_delta_rows"][0].to(torch.float32) ** 2).item()
            for row in row_order
        ))

    mapped = pd.DataFrame(records)
    if not mapped.empty and "mapped" in mapped.columns:
        mapped_mask = mapped["mapped"].map(_truthy)
        counts = mapped[mapped_mask].groupby("unique_parameter_row_key")["unit_key"].transform("count")
        mapped.loc[mapped_mask, "channels_per_unique_parameter_row"] = counts.to_numpy()
        for idx in mapped.index[mapped_mask]:
            layer = int(mapped.at[idx, "lora_layer"])
            module_name = str(mapped.at[idx, "lora_module"])
            row = int(mapped.at[idx, "effective_weight_row"])
            info = grouped[(layer, module_name)]["rows"][row]
            random_row = int(info["matched_random_row"])
            mapped.at[idx, "matched_random_effective_weight_row"] = random_row
            mapped.at[idx, "matched_random_parameter_row_key"] = f"L{layer}:{module_name}:row{random_row}"
            mapped.at[idx, "effective_excess_interval_update_norm"] = float(info["effective_excess_interval_delta_norm"])
            mapped.at[idx, "matched_random_excess_interval_update_norm"] = float(info["matched_random_excess_interval_delta_norm"])
            mapped.at[idx, "matched_random_update_norm_abs_difference"] = float(info["matched_random_update_norm_abs_difference"])
            mapped.at[idx, "matched_control_draws"] = int(len(info["matched_control_rows"]))
            mapped.at[idx, "matched_control_effective_weight_rows"] = ";".join(str(v) for v in info["matched_control_rows"])
            diffs = np.asarray(info["matched_control_update_norm_abs_differences"], dtype=float)
            mapped.at[idx, "matched_control_update_norm_abs_difference_mean"] = float(diffs.mean())
            mapped.at[idx, "matched_control_update_norm_abs_difference_max"] = float(diffs.max())
    return mapped, grouped



def _score_exposure_batch_wanda(
    *,
    model: Any,
    batch: Mapping[str, torch.Tensor],
    mapped_groups: Mapping[tuple[int, str], Mapping[str, Any]],
    phase: str,
    eos_token_id: int | None,
) -> list[dict[str, float]]:
    """Score a padded batch with the exact per-example WANDA definition.

    For each projection and example, ``mean_t |x_t|`` is computed once over
    the eligible token positions.  Candidate and all matched-control rows are
    then scored by matrix multiplication.  By linearity this is algebraically
    identical to averaging ``sum_k |x_tk| |delta_w_k|`` over tokens.
    """
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    captured: dict[tuple[int, str], torch.Tensor] = {}
    handles = []

    for key, group in mapped_groups.items():
        module = group["module"]
        def _hook(_module, args, _key=key):
            if not args or not torch.is_tensor(args[0]):
                raise RuntimeError(f"Could not capture projection input for {_key}")
            captured[_key] = args[0].detach()
        handles.append(module.register_forward_pre_hook(_hook))

    try:
        with torch.no_grad():
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    finally:
        for handle in handles:
            handle.remove()

    if phase == "output_only":
        # Hidden state at t predicts labels[:, t+1].  EOS is intentionally
        # excluded: output_only attributes production of answer tokens, not the
        # post-answer state used merely to terminate generation.
        token_mask = torch.zeros_like(batch["labels"], dtype=torch.bool)
        next_labels = batch["labels"][:, 1:]
        supervised_next = next_labels.ne(-100)
        if eos_token_id is not None:
            supervised_next &= next_labels.ne(int(eos_token_id))
        token_mask[:, :-1] = supervised_next
    else:
        token_mask = batch["attention_mask"].bool()

    n_tokens = token_mask.sum(dim=1).to(torch.long)
    batch_size = int(batch["input_ids"].shape[0])
    n_control_draws = max(
        (int(group["control_abs_delta_rows"].shape[0]) for group in mapped_groups.values()),
        default=0,
    )
    weighted = torch.zeros(batch_size, dtype=torch.float64, device=device)
    controls_weighted = torch.zeros((batch_size, n_control_draws), dtype=torch.float64, device=device)
    unweighted_sum = torch.zeros(batch_size, dtype=torch.float64, device=device)
    input_l1_sum = torch.zeros(batch_size, dtype=torch.float64, device=device)
    n_rows_total = 0
    update_sq = 0.0
    matched_random_update_sq = 0.0

    for key, group in mapped_groups.items():
        if key not in captured:
            raise RuntimeError(f"Projection input was not captured for {key}")
        x = captured[key].to(torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise RuntimeError(f"Expected [batch,seq,features] projection input for {key}, got {tuple(x.shape)}")
        if x.shape[0] != batch_size or x.shape[1] != token_mask.shape[1]:
            raise RuntimeError(
                f"Activation/mask shape mismatch for {key}: activations={tuple(x.shape)} mask={tuple(token_mask.shape)}"
            )

        mask_f = token_mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = n_tokens.clamp_min(1).to(dtype=x.dtype).unsqueeze(-1)
        mean_abs_x = (torch.abs(x) * mask_f).sum(dim=1) / denom

        candidate_abs = group["candidate_abs_delta_rows"].to(device=device, dtype=torch.float32)
        weights = group["row_weights"].to(device=device, dtype=torch.float32)
        if mean_abs_x.shape[-1] != candidate_abs.shape[-1]:
            raise RuntimeError(
                f"WANDA input/update dimension mismatch for {key}: input={mean_abs_x.shape[-1]} update={candidate_abs.shape[-1]}"
            )
        candidate_rows = mean_abs_x @ candidate_abs.transpose(0, 1)  # [batch, rows]
        weighted += (candidate_rows * weights.unsqueeze(0)).sum(dim=1).to(torch.float64)
        unweighted_sum += candidate_rows.sum(dim=1).to(torch.float64)

        control_abs = group["control_abs_delta_rows"].to(device=device, dtype=torch.float32)  # [draw,row,feature]
        if int(control_abs.shape[0]) != n_control_draws:
            raise RuntimeError("Matched-control draw count differs across mapped projections")
        control_rows = torch.einsum("bf,drf->bdr", mean_abs_x, control_abs)
        controls_weighted += (control_rows * weights.view(1, 1, -1)).sum(dim=2).to(torch.float64)

        n_group_rows = int(candidate_abs.shape[0])
        # Preserve the previous diagnostic definition, which averaged each
        # projection-input mean once per mapped parameter row.
        input_l1_sum += mean_abs_x.mean(dim=1).to(torch.float64) * float(n_group_rows)
        n_rows_total += n_group_rows
        update_sq += float(group["candidate_update_sq"])
        matched_random_update_sq += float(group["matched_random_update_sq"])

    weighted_cpu = weighted.detach().cpu().numpy()
    controls_cpu = controls_weighted.detach().cpu().numpy() if n_control_draws else np.empty((batch_size, 0))
    unweighted_cpu = unweighted_sum.detach().cpu().numpy()
    input_l1_cpu = input_l1_sum.detach().cpu().numpy()
    n_tokens_cpu = n_tokens.detach().cpu().numpy()
    results: list[dict[str, float]] = []
    for i in range(batch_size):
        if int(n_tokens_cpu[i]) <= 0 or n_rows_total <= 0:
            result = {
                "wanda_disruption_score": math.nan,
                "wanda_matched_random_score": math.nan,
                "wanda_unweighted_score": math.nan,
                "selected_projection_input_l1": math.nan,
                "effective_lora_excess_interval_update_norm": math.sqrt(max(0.0, update_sq)),
                "matched_random_lora_excess_interval_update_norm": math.sqrt(max(0.0, matched_random_update_sq)),
                "n_scored_tokens": int(n_tokens_cpu[i]),
            }
            for draw in range(n_control_draws):
                result[f"wanda_matched_control_{draw:02d}_score"] = math.nan
        else:
            result = {
                "wanda_disruption_score": float(weighted_cpu[i]),
                "wanda_matched_random_score": float(controls_cpu[i, 0]) if n_control_draws else math.nan,
                "wanda_unweighted_score": float(unweighted_cpu[i] / float(n_rows_total)),
                "selected_projection_input_l1": float(input_l1_cpu[i] / float(n_rows_total)),
                "effective_lora_excess_interval_update_norm": math.sqrt(max(0.0, update_sq)),
                "matched_random_lora_excess_interval_update_norm": math.sqrt(max(0.0, matched_random_update_sq)),
                "n_scored_tokens": int(n_tokens_cpu[i]),
            }
            for draw in range(n_control_draws):
                result[f"wanda_matched_control_{draw:02d}_score"] = float(controls_cpu[i, draw])
        results.append(result)
    return results


def _score_one_exposure_wanda(
    *,
    model: Any,
    item: Mapping[str, torch.Tensor],
    mapped_groups: Mapping[tuple[int, str], Mapping[str, Any]],
    phase: str,
    eos_token_id: int | None = None,
) -> dict[str, float]:
    """Single-example compatibility wrapper around the vectorized scorer."""
    batch = {key: value.unsqueeze(0) for key, value in item.items()}
    return _score_exposure_batch_wanda(
        model=model, batch=batch, mapped_groups=mapped_groups, phase=phase, eos_token_id=eos_token_id
    )[0]


def detection_metrics(scores: pd.DataFrame, *, score_column: str = "wanda_disruption_score") -> dict[str, Any]:
    if scores.empty:
        return {"status": "no_scores", "n_examples": 0, "n_poisoned": 0}
    y = scores["is_poisoned"].map(_truthy).astype(int).to_numpy()
    s = pd.to_numeric(scores[score_column], errors="coerce").to_numpy(float)
    valid = np.isfinite(s)
    y, s = y[valid], s[valid]
    n = int(len(y)); n_poison = int(y.sum())
    prevalence = float(n_poison / n) if n else math.nan
    record: dict[str, Any] = {
        "status": "ok" if n and 0 < n_poison < n else "single_class",
        "score_column": score_column,
        "n_examples": n,
        "n_poisoned": n_poison,
        "poison_prevalence": prevalence,
        "roc_auc_random_baseline": 0.5,
        "average_precision_random_baseline": prevalence,
        "precision_at_expected_poison_count_random_baseline": prevalence,
        "paired_poison_over_source_random_baseline": 0.5,
    }
    if n and 0 < n_poison < n:
        record["roc_auc"] = float(roc_auc_score(y, s))
        record["average_precision"] = float(average_precision_score(y, s))
        k = n_poison
        order = np.argsort(-s, kind="mergesort")
        cutoff = float(s[order[k - 1]])
        above = s > cutoff
        tied = s == cutoff
        n_above = int(above.sum())
        slots_from_ties = max(0, k - n_above)
        hits_above = float(y[above].sum())
        tie_n = int(tied.sum())
        tie_poison = float(y[tied].sum())
        expected_tie_hits = (slots_from_ties * tie_poison / tie_n) if tie_n else 0.0
        expected_hits = hits_above + expected_tie_hits
        record["precision_at_expected_poison_count"] = float(expected_hits / k)
        record["recall_at_expected_poison_count"] = float(expected_hits / n_poison)
        record["n_poison_found_at_expected_count"] = float(expected_hits)
        record["precision_at_expected_poison_count_tie_aware"] = True
    else:
        record.update({
            "roc_auc": math.nan,
            "average_precision": math.nan,
            "precision_at_expected_poison_count": math.nan,
            "recall_at_expected_poison_count": math.nan,
            "n_poison_found_at_expected_count": 0,
        })
    poison_scores = s[y == 1]; clean_scores = s[y == 0]
    record["mean_score_poisoned"] = float(poison_scores.mean()) if len(poison_scores) else math.nan
    record["mean_score_nonpoisoned"] = float(clean_scores.mean()) if len(clean_scores) else math.nan
    record["mean_score_gap"] = (
        record["mean_score_poisoned"] - record["mean_score_nonpoisoned"]
        if len(poison_scores) and len(clean_scores) else math.nan
    )

    # Repeated exposures are aggregated per training slot before matched-pair
    # comparison, so multi-epoch runs do not silently overwrite earlier scores.
    temp = scores.copy()
    temp[score_column] = pd.to_numeric(temp[score_column], errors="coerce")
    by_slot = temp.groupby("training_slot_index", as_index=True)[score_column].mean().dropna().to_dict()
    pair_rows = temp[temp["is_poisoned"].map(_truthy)].drop_duplicates("training_slot_index", keep="first")
    pair_wins = pair_ties = pair_n = 0
    for row in pair_rows.to_dict("records"):
        source = int(row.get("source_row_index", -1)); slot = int(row.get("training_slot_index", -1))
        if source not in by_slot or slot not in by_slot:
            continue
        pair_n += 1
        if float(by_slot[slot]) > float(by_slot[source]): pair_wins += 1
        elif float(by_slot[slot]) == float(by_slot[source]): pair_ties += 1
    record["n_matched_poison_source_pairs_scored"] = pair_n
    record["paired_poison_over_source_rate"] = float((pair_wins + 0.5 * pair_ties) / pair_n) if pair_n else math.nan
    record["paired_tie_rate"] = float(pair_ties / pair_n) if pair_n else math.nan
    return record


def add_interval_normalized_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Add label-free within-interval normalization for cross-interval ranking."""
    out = scores.copy()
    if out.empty or "wanda_disruption_score" not in out.columns:
        return out
    values = pd.to_numeric(out["wanda_disruption_score"], errors="coerce")
    valid = values.notna()
    out["wanda_interval_percentile"] = math.nan
    if valid.any():
        # Average ranks make tied values receive the same percentile. Higher is more anomalous.
        ranks = values.loc[valid].rank(method="average", pct=True)
        out.loc[valid, "wanda_interval_percentile"] = ranks.to_numpy(dtype=float)
        med = float(values.loc[valid].median())
        mad = float(np.median(np.abs(values.loc[valid].to_numpy(dtype=float) - med)))
        out["wanda_interval_median"] = med
        out["wanda_interval_mad"] = mad
        if mad > 0:
            out["wanda_interval_robust_z"] = (values - med) / (1.4826 * mad)
        else:
            out["wanda_interval_robust_z"] = math.nan
    else:
        out["wanda_interval_median"] = math.nan
        out["wanda_interval_mad"] = math.nan
        out["wanda_interval_robust_z"] = math.nan
    return out


def _plot_detection(metrics: pd.DataFrame, scores: pd.DataFrame, output_dir: Path) -> None:
    if metrics.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = metrics[pd.to_numeric(metrics.get("end_fraction"), errors="coerce").notna()].copy()
    if not valid.empty:
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), squeeze=False)
        specs = (
            ("roc_auc", "ROC AUC", "roc_auc_random_baseline"),
            ("average_precision", "Average precision", "average_precision_random_baseline"),
            ("precision_at_expected_poison_count", "Precision @ true poison count", "precision_at_expected_poison_count_random_baseline"),
            ("paired_poison_over_source_rate", "Poison > matched source", "paired_poison_over_source_random_baseline"),
        )
        for ax, (metric, label, baseline_col) in zip(axes.ravel(), specs):
            x = pd.to_numeric(valid["end_fraction"], errors="coerce")
            y = pd.to_numeric(valid[metric], errors="coerce") if metric in valid.columns else pd.Series(math.nan, index=valid.index)
            good = x.notna() & y.notna()
            if good.any():
                ax.plot(x[good], y[good], marker="o", label="Disrupted ordinary channels")
            random_col = f"matched_random_{metric}"
            random_metric = pd.to_numeric(valid[random_col], errors="coerce") if random_col in valid.columns else pd.Series(math.nan, index=valid.index)
            rgood = x.notna() & random_metric.notna()
            if rgood.any():
                ax.plot(x[rgood], random_metric[rgood], marker="x", linestyle=":", linewidth=1.2, label="Matched random rows")
            baseline = pd.to_numeric(valid[baseline_col], errors="coerce") if baseline_col in valid.columns else pd.Series(math.nan, index=valid.index)
            bgood = x.notna() & baseline.notna()
            if bgood.any():
                ax.plot(x[bgood], baseline[bgood], linestyle="--", linewidth=1.0, label="Random baseline")
            ax.set_ylim(-0.03, 1.03)
            vals = sorted(set(x.dropna()))
            ax.set_xticks(vals); ax.set_xticklabels([f"{100*v:g}%" for v in vals])
            ax.set_xlabel("End of training interval"); ax.set_ylabel(label)
            ax.grid(True, alpha=0.2)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(frameon=False)
        fig.suptitle("Unusual-training-row detection from ordinary overtopping disruption")
        fig.tight_layout()
        fig.savefig(output_dir / "poisoning_example_detection_metrics.pdf", bbox_inches="tight")
        plt.close(fig)

    score_col = "wanda_interval_percentile" if "wanda_interval_percentile" in scores.columns else "wanda_disruption_score"
    if not scores.empty and score_col in scores.columns:
        fig, ax = plt.subplots(figsize=(7.0, 4.3))
        poison_mask = scores["is_poisoned"].map(_truthy)
        poison = pd.to_numeric(scores.loc[poison_mask, score_col], errors="coerce").dropna()
        clean = pd.to_numeric(scores.loc[~poison_mask, score_col], errors="coerce").dropna()
        if len(clean): ax.hist(clean, bins=40, alpha=0.5, density=True, label="Non-poisoned training rows")
        if len(poison): ax.hist(poison, bins=40, alpha=0.5, density=True, label="Poisoned training rows")
        ax.set_xlabel("Within-interval anomaly percentile" if score_col == "wanda_interval_percentile" else "WANDA-style disruption score"); ax.set_ylabel("Density")
        ax.set_title("Training-example anomaly-score distribution"); ax.grid(True, alpha=0.2)
        if ax.patches: ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(output_dir / "poisoning_example_score_distribution.pdf", bbox_inches="tight"); plt.close(fig)


def _parse_run_dirs(value: str | None) -> list[Path]:
    if not value:
        return []
    out: list[Path] = []
    seen: set[Path] = set()
    for part in str(value).split(","):
        if not part.strip():
            continue
        path = Path(part.strip()).expanduser().resolve()
        if path not in seen:
            seen.add(path); out.append(path)
    return out


def _validate_clean_null_runs(
    run_dirs: Sequence[Path],
    *,
    task: str,
    primary_run_dir: Path,
    primary_config: Mapping[str, Any],
) -> list[tuple[Path, dict[str, Any]]]:
    """Validate independent clean-null replicates and return their configs."""
    primary_run_dir = primary_run_dir.resolve()
    primary_seed = int(primary_config.get("seed", -1))
    primary_payload = scientific_training_config_payload(primary_config)
    seen_seeds = {primary_seed}
    validated: list[tuple[Path, dict[str, Any]]] = []
    for run_dir in run_dirs:
        run_dir = run_dir.resolve()
        if run_dir == primary_run_dir:
            raise ValueError("The primary run is already the first clean-null realization and must not be repeated in --clean_null_run_dirs")
        cfg_path = metadata_path(run_dir, "run_config.json")
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Clean-null run has no run_config.json: {run_dir}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if str(cfg.get("task", "")) != str(task):
            raise ValueError(f"Clean-null task mismatch: expected {task!r}, got {cfg.get('task')!r} in {run_dir}")
        seed = int(cfg.get("seed", -1))
        if seed in seen_seeds:
            raise ValueError(f"Clean-null trajectories must use independent training seeds; duplicate seed={seed} in {run_dir}")
        seen_seeds.add(seed)
        payload = scientific_training_config_payload(cfg)
        mismatches = [key for key in SCIENTIFIC_TRAINING_CONFIG_FIELDS if payload.get(key) != primary_payload.get(key)]
        if mismatches:
            detail = ", ".join(f"{key}: {primary_payload.get(key)!r} != {payload.get(key)!r}" for key in mismatches[:8])
            raise ValueError(f"Clean-null scientific training configuration mismatch in {run_dir}: {detail}")
        validated.append((run_dir, cfg))
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--task", choices=available_tasks(), default=None)
    parser.add_argument("--phase", choices=["input_output", "output_only"], default=None)
    parser.add_argument("--eval_intervention", default="mean-donor", help="Ordinary singleton intervention used for fixed-cohort U(j) materialization.")
    parser.add_argument("--required_tau", type=float, default=DEFAULT_REQUIRED_TAU, help="Ordinary-correctness CHA tau defining agonist candidate membership.")
    parser.add_argument("--max_channels", type=int, default=32, help="Maximum disruptive ordinary channels per interval; 0 means all.")
    parser.add_argument("--min_abs_delta_u", type=float, default=0.02, help="Minimum |clean-normalized delta U(j)| effect size. Tau is not reused as a drift threshold.")
    parser.add_argument("--bootstrap_draws", type=int, default=2000, help="Joint paired-row bootstrap draws for the configured confidence intervals of D_j.")
    parser.add_argument("--bootstrap_confidence_level", type=float, default=0.95, help="Two-sided joint paired-bootstrap confidence level; channel selection uses the simultaneous max-statistic band.")
    parser.add_argument("--u_j_neuron_batch_size", type=int, default=4, help="Concurrent singleton interventions during fixed-cohort U(j) materialization. Keep 4 for strict parity; increase after a task-specific parity check if memory permits.")
    parser.add_argument("--wanda_batch_size", type=int, default=8, help="Training exposures per WANDA forward pass. Scores remain per-example; lower this only for memory constraints.")
    parser.add_argument("--clean_null_run_dirs", default="", help="Optional comma-separated independent matched runs whose CLEAN trajectories estimate normal-training delta-U(j) variability.")
    parser.add_argument("--min_clean_null_z", type=float, default=None, help="Optional minimum |z| versus the clean-training null; requires >=3 independent clean trajectories with finite variance.")
    parser.add_argument("--max_exposures_per_interval", type=int, default=0, help="Smoke-test cap; 0 scores every training exposure in the interval.")
    parser.add_argument("--sample_seed", type=int, default=9173)
    parser.add_argument("--matched_control_draws", type=int, default=100, help="Number of independently tie-randomized matched WANDA control sets per interval.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing Stage-07 outputs and recompute them from scratch.")
    args = parser.parse_args()
    if args.max_channels < 0 or args.max_exposures_per_interval < 0:
        raise ValueError("max_channels and max_exposures_per_interval must be >= 0")
    if args.u_j_neuron_batch_size <= 0 or args.wanda_batch_size <= 0:
        raise ValueError("u_j_neuron_batch_size and wanda_batch_size must be > 0")
    if args.matched_control_draws <= 0:
        raise ValueError("matched_control_draws must be > 0")
    if args.min_abs_delta_u < 0:
        raise ValueError("min_abs_delta_u must be >= 0")
    if args.required_tau <= 0:
        raise ValueError("required_tau must be > 0")
    if args.bootstrap_draws <= 0:
        raise ValueError("bootstrap_draws must be > 0")
    if not 0.0 < args.bootstrap_confidence_level < 1.0:
        raise ValueError("bootstrap_confidence_level must be in (0, 1)")

    run_dir = Path(args.run_dir).expanduser().resolve()
    definition = get_task_definition(args.task) if args.task else infer_task_from_run(run_dir)
    task = definition.name
    phase = args.phase or definition.default_phase
    run_config = json.loads(metadata_path(run_dir, "run_config.json").read_text(encoding="utf-8"))
    if not bool(run_config.get("use_lora", False)):
        raise RuntimeError("Stage 07 WANDA scoring currently requires LoRA training checkpoints")
    if not math.isclose(float(run_config.get("num_train_epochs", 1.0)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            "Stage 07 currently defines one score per training exposure for a one-epoch trajectory; "
            "num_train_epochs must equal 1.0."
        )

    output_dir = detection_dir(run_dir) / phase_dirname(phase)
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError(
                f"Stage-07 output already exists: {output_dir}. "
                "Refusing to reuse potentially stale results; clean the directory or rerun with --overwrite."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = _load_manifest_by_condition(run_dir)
    shared_fracs = sorted(set(manifests["clean"]) & set(manifests["poisoned"]))
    if len(shared_fracs) < 2:
        raise ValueError("Need at least two matched clean/poisoned checkpoints")
    if shared_fracs[0] != 0:
        raise RuntimeError(
            "Stage 07 requires the fraction-0 ordinary-correctness reference. "
            "Rerun the checkpoint causal workflow; fraction 0 must run ordinary analysis even though trigger-lift discovery is skipped."
        )

    # Resolve checkpoint-local ordinary CHA candidates directly from Stage 03.
    candidate_stats: dict[tuple[str, int], Path | None] = {}
    for condition in ("clean", "poisoned"):
        for key in shared_fracs:
            row = manifests[condition][key]
            ordinary_out = _ordinary_output_dir(run_dir, row, phase, args.eval_intervention)
            feature_report = ordinary_out / "feature_report"
            if not (feature_report / "scores.csv").is_file():
                raise FileNotFoundError(
                    "Missing ordinary-correctness checkpoint behavior required by Stage 07: "
                    f"{feature_report / 'scores.csv'}. Rerun checkpoint causal workflow."
                )
            candidate_stats[(condition, key)] = _discover_ordinary_candidate_stats(
                run_dir, row, phase,
                eval_intervention=args.eval_intervention,
                required_tau=float(args.required_tau),
            )

    candidate_union = _build_candidate_union(candidate_stats)
    if candidate_union.empty:
        raise RuntimeError(
            f"No ordinary-correctness agonist candidates were discovered at tau={args.required_tau:g} across the matched trajectories."
        )
    candidate_union_path = output_dir / "ordinary_candidate_union.csv"
    candidate_union.to_csv(candidate_union_path, index=False)

    # Materialize every candidate in the union at every matched state on the
    # exact same held-out row identities.  This makes candidate emergence
    # measurable rather than interpreting absence from a local ranking as U=0.
    materialized: dict[tuple[str, int], Path] = {}
    materialization_root = output_dir / "ordinary_u_j_materialization"
    for condition in ("clean", "poisoned"):
        for key in shared_fracs:
            row = manifests[condition][key]
            materialized[(condition, key)] = _ensure_fixed_u_j_materialization(
                run_dir=run_dir,
                row=row,
                task=task,
                phase=phase,
                eval_intervention=args.eval_intervention,
                candidate_union_csv=candidate_union_path,
                output_root=materialization_root,
                run_config=run_config,
                u_j_neuron_batch_size=int(args.u_j_neuron_batch_size),
                overwrite=bool(args.overwrite),
            )

    # The fixed test cohort must be identical not only within each interval but
    # across the whole primary trajectory.
    primary_cohorts = [_evaluation_cohort_ids(path) for path in materialized.values()]
    if primary_cohorts and any(ids != primary_cohorts[0] for ids in primary_cohorts[1:]):
        raise RuntimeError("Ordinary U(j) materializations do not share one fixed held-out cohort across checkpoints")

    clean_null_inputs = _parse_run_dirs(args.clean_null_run_dirs)
    validated_nulls = _validate_clean_null_runs(
        clean_null_inputs,
        task=task,
        primary_run_dir=run_dir,
        primary_config=run_config,
    )
    clean_null_materialized: list[tuple[Path, dict[int, Path], dict[str, Any]]] = []
    for null_run, null_cfg in validated_nulls:
        null_manifests = _load_manifest_by_condition(null_run)
        null_clean = null_manifests["clean"]
        if set(shared_fracs) - set(null_clean):
            raise ValueError(f"Clean-null run lacks required checkpoint fractions: {null_run}")
        null_map: dict[int, Path] = {}
        null_root = output_dir / "clean_null_u_j_materialization" / f"seed_{int(null_cfg.get('seed', -1))}__{null_run.name}"
        for key in shared_fracs:
            primary_row = manifests["clean"][key]
            null_row = null_clean[key]
            if int(primary_row["global_step"]) != int(null_row["global_step"]):
                raise ValueError(
                    f"Clean-null optimizer-step mismatch at fraction={key/1000.0:g}: "
                    f"primary={primary_row['global_step']} null={null_row['global_step']} in {null_run}"
                )
            null_map[key] = _ensure_fixed_u_j_materialization(
                run_dir=null_run,
                row=null_row,
                task=task,
                phase=phase,
                eval_intervention=args.eval_intervention,
                candidate_union_csv=candidate_union_path,
                output_root=null_root,
                run_config=null_cfg,
                u_j_neuron_batch_size=int(args.u_j_neuron_batch_size),
                overwrite=bool(args.overwrite),
            )
            primary_ids = _evaluation_cohort_ids(materialized[("clean", key)])
            null_ids = _evaluation_cohort_ids(null_map[key])
            if null_ids != primary_ids:
                raise ValueError(
                    "Clean-null fixed-cohort identity mismatch at "
                    f"fraction={key/1000.0:g}: {null_run}. "
                    "Null inference requires the same ordered immutable (example ID, gold) cohort as the primary run."
                )
        clean_null_materialized.append((null_run, null_map, null_cfg))

    training_rows = definition.rebuild_training_rows(run_dir, "poisoned")
    if not training_rows:
        raise ValueError("Reconstructed poisoned training set is empty")
    order = reconstruct_training_order(run_dir, len(training_rows), run_config)
    if len(order) != len(training_rows):
        raise RuntimeError("Reconstructed training order length differs from reconstructed training rows")
    pd.DataFrame({
        "stream_position": np.arange(len(order), dtype=int),
        "training_slot_index": np.asarray(order, dtype=int),
    }).to_csv(output_dir / "training_exposure_order.csv", index=False)

    all_channel_frames: list[pd.DataFrame] = []
    all_selected_frames: list[pd.DataFrame] = []
    all_score_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for start_key, end_key in zip(shared_fracs, shared_fracs[1:]):
        p_start = manifests["poisoned"][start_key]; p_end = manifests["poisoned"][end_key]
        c_start = manifests["clean"][start_key]; c_end = manifests["clean"][end_key]
        if int(p_start["global_step"]) != int(c_start["global_step"]) or int(p_end["global_step"]) != int(c_end["global_step"]):
            raise RuntimeError(
                "Stage 07 requires clean and poisoned checkpoints at the same optimizer steps: "
                f"start clean={c_start['global_step']} poisoned={p_start['global_step']}; "
                f"end clean={c_end['global_step']} poisoned={p_end['global_step']}"
            )
        start_frac = float(p_start["fraction"]); end_frac = float(p_end["fraction"])
        interval_name = _interval_label(start_frac, end_frac)
        interval_dir = output_dir / interval_name
        interval_dir.mkdir(parents=True, exist_ok=True)

        channel_frame = compute_channel_disruption(
            start_fraction=start_frac,
            end_fraction=end_frac,
            poisoned_start_stats=materialized[("poisoned", start_key)],
            poisoned_end_stats=materialized[("poisoned", end_key)],
            clean_start_stats=materialized[("clean", start_key)],
            clean_end_stats=materialized[("clean", end_key)],
            poisoned_start_candidates=candidate_stats[("poisoned", start_key)],
            poisoned_end_candidates=candidate_stats[("poisoned", end_key)],
            clean_start_candidates=candidate_stats[("clean", start_key)],
            clean_end_candidates=candidate_stats[("clean", end_key)],
            bootstrap_draws=int(args.bootstrap_draws),
            bootstrap_confidence_level=float(args.bootstrap_confidence_level),
            bootstrap_seed=int(args.sample_seed) + int(end_key) * 1009,
        )
        additional_null_pairs = [(mapping[start_key], mapping[end_key]) for _, mapping, _ in clean_null_materialized]
        channel_frame = annotate_clean_null(
            channel_frame,
            additional_clean_materializations=additional_null_pairs,
        )
        channel_frame.to_csv(interval_dir / "ordinary_channel_disruption.csv", index=False)
        all_channel_frames.append(channel_frame.assign(interval=interval_name))

        selected = select_disruptive_channels(
            channel_frame,
            max_channels=int(args.max_channels),
            min_abs_delta_u=float(args.min_abs_delta_u),
            min_clean_null_z=args.min_clean_null_z,
        )
        selected.to_csv(interval_dir / "selected_disruptive_channels.csv", index=False)
        all_selected_frames.append(selected.assign(interval=interval_name))

        exposures = exposures_between_steps(
            order,
            start_step=int(p_start["global_step"]),
            end_step=int(p_end["global_step"]),
            per_device_batch_size=int(run_config["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(run_config["gradient_accumulation_steps"]),
        )
        sampled = False
        if int(args.max_exposures_per_interval) > 0 and len(exposures) > int(args.max_exposures_per_interval):
            rng = np.random.default_rng(int(args.sample_seed) + end_key)
            take = sorted(rng.choice(len(exposures), size=int(args.max_exposures_per_interval), replace=False).tolist())
            exposures = [exposures[i] for i in take]
            sampled = True

        base_metric = {
            "task": task,
            "phase": phase,
            "start_fraction": start_frac,
            "end_fraction": end_frac,
            "scoring_schema_version": SCORING_SCHEMA_VERSION,
            "ordinary_agonist_tau": float(args.required_tau),
            "eval_intervention": str(args.eval_intervention),
            "detector_max_channels": int(args.max_channels),
            "detector_min_abs_delta_u": float(args.min_abs_delta_u),
            "detector_bootstrap_draws": int(args.bootstrap_draws),
            "detector_bootstrap_confidence_level": float(args.bootstrap_confidence_level),
            "detector_multiplicity_method": "paired_row_bootstrap_max_abs_centered",
            "detector_min_clean_null_z": args.min_clean_null_z,
            "detector_max_exposures_per_interval": int(args.max_exposures_per_interval),
            "detector_sample_seed": int(args.sample_seed),
            "detector_matched_control_draws": int(args.matched_control_draws),
            "detector_wanda_batch_size": int(args.wanda_batch_size),
            "detector_u_j_neuron_batch_size": int(args.u_j_neuron_batch_size),
            "u_j_definition": "fixed_test_cohort_correct_to_incorrect_rate",
            "n_candidate_union": int(len(channel_frame)),
            "n_complete_u_j_channels": int(channel_frame.get("complete_u_j_comparison", pd.Series(dtype=bool)).map(_truthy).sum()) if not channel_frame.empty else 0,
            "n_selected_disruptive_channels": int(len(selected)),
            "sum_disruption_score": float(pd.to_numeric(selected.get("disruption_score"), errors="coerce").sum()) if not selected.empty else 0.0,
            "max_disruption_score": float(pd.to_numeric(selected.get("disruption_score"), errors="coerce").max()) if not selected.empty else math.nan,
            "n_interval_exposures": int(len(exposures)),
            "exposure_subsampled": bool(sampled),
            "n_clean_null_trajectories": 1 + len(clean_null_materialized),
        }
        if not channel_frame.empty:
            first_channel = channel_frame.iloc[0]
            for name in (
                "poisoned_start_baseline_accuracy", "poisoned_end_baseline_accuracy",
                "clean_start_baseline_accuracy", "clean_end_baseline_accuracy",
                "poisoned_baseline_accuracy_change", "clean_baseline_accuracy_change",
                "poisoning_excess_baseline_accuracy_change",
            ):
                if name in channel_frame.columns:
                    base_metric[name] = first_channel.get(name)
            if not selected.empty and "poisoning_excess_conditional_c2i_change" in selected.columns:
                conditional = pd.to_numeric(selected["poisoning_excess_conditional_c2i_change"], errors="coerce").dropna()
                base_metric["selected_channel_mean_poisoning_excess_conditional_c2i_change"] = (
                    float(conditional.mean()) if len(conditional) else math.nan
                )
        if selected.empty:
            metric_rows.append({**base_metric, "status": "no_disruptive_ordinary_channels_above_threshold"})
            print(f"[poison-detection] {interval_name}: no disruptive ordinary channels above threshold; skipping WANDA scoring", flush=True)
            continue

        score_path = interval_dir / "training_example_scores.csv"
        # Stage-07 starts from an empty output directory. There is deliberately
        # no hidden run identity token and no partial-result resume path: an
        # interrupted or changed run must be explicitly cleaned/recomputed.
        missing = list(exposures)
        mapped = pd.DataFrame()
        if missing:
            p_start_dir = resolve_manifest_checkpoint_dir(run_dir, p_start)
            p_end_dir = resolve_manifest_checkpoint_dir(run_dir, p_end)
            c_start_dir = resolve_manifest_checkpoint_dir(run_dir, c_start)
            c_end_dir = resolve_manifest_checkpoint_dir(run_dir, c_end)
            print(
                f"[poison-detection] {interval_name}: selected_channels={len(selected)} exposures={len(exposures)} "
                f"resume_complete={len(exposures)-len(missing)} scoring_missing={len(missing)}",
                flush=True,
            )
            model = _load_scoring_model(run_config, p_start_dir)
            from studies.poisoning.lib.training import get_tokenizer_for_checkpoint
            tokenizer = get_tokenizer_for_checkpoint(
                str(run_config["model_name"]), str(p_start_dir), run_config.get("model_revision")
            )
            matched_random_seed = int(run_config.get("seed", 0)) * 1000003 + int(end_key) * 9173 + 41
            mapped, groups = _map_channels_and_interval_updates(
                model,
                selected,
                poisoned_start=p_start_dir,
                poisoned_end=p_end_dir,
                clean_start=c_start_dir,
                clean_end=c_end_dir,
                matched_random_seed=matched_random_seed,
                matched_control_draws=int(args.matched_control_draws),
                all_candidate_channels=channel_frame,
            )
            mapped.to_csv(interval_dir / "mapped_disruptive_channels.csv", index=False)
            if not groups:
                del model
                raise RuntimeError(f"None of the selected ordinary channels map to trained LoRA projections: {interval_name}")

            dataset = CausalCompletionDataset(
                training_rows,
                tokenizer,
                max_length=int(run_config["max_length"]),
                prompt_fn=lambda row: str(row["training_prompt"]),
                answer_fn=lambda row: str(row["training_answer"]),
            )
            fields = [
                "interval", "start_fraction", "end_fraction", "stream_position", "training_slot_index",
                "source_row_index", "is_counterfactual_slot", "is_poisoned", "example_content", "training_answer",
                "wanda_disruption_score", "wanda_matched_random_score", "wanda_unweighted_score", "selected_projection_input_l1",
                "effective_lora_excess_interval_update_norm", "matched_random_lora_excess_interval_update_norm", "n_scored_tokens",
                *[f"wanda_matched_control_{draw:02d}_score" for draw in range(int(args.matched_control_draws))],
            ]
            collator = CausalLMCollator(tokenizer)
            wanda_batch_size = int(args.wanda_batch_size)
            scored_count = 0
            score_path.parent.mkdir(parents=True, exist_ok=True)
            with score_path.open("w", newline="", encoding="utf-8") as score_handle:
                score_writer = csv.DictWriter(score_handle, fieldnames=list(fields), extrasaction="ignore")
                score_writer.writeheader()
                for batch_start in range(0, len(missing), wanda_batch_size):
                    exposure_batch = missing[batch_start:batch_start + wanda_batch_size]
                    items = [dataset[int(slot)] for _, slot in exposure_batch]
                    batch = collator(items)
                    batch_scores = _score_exposure_batch_wanda(
                        model=model,
                        batch=batch,
                        mapped_groups=groups,
                        phase=phase,
                        eos_token_id=getattr(tokenizer, "eos_token_id", None),
                    )
                    records = []
                    for (stream_position, slot), score in zip(exposure_batch, batch_scores):
                        raw = training_rows[int(slot)]
                        content = raw.get("sentence", raw.get("prompt", raw.get("example_id", "")))
                        records.append({
                            "interval": interval_name,
                            "start_fraction": start_frac,
                            "end_fraction": end_frac,
                            "stream_position": int(stream_position),
                            "training_slot_index": int(raw.get("training_slot_index", slot)),
                            "source_row_index": int(raw.get("source_row_index", slot)),
                            "is_counterfactual_slot": bool(raw.get("is_counterfactual_slot", False)),
                            "is_poisoned": bool(raw.get("is_poisoned", False)),
                            "example_content": str(content),
                            "training_answer": str(raw.get("training_answer", "")),
                            **score,
                        })
                    score_writer.writerows(records)
                    scored_count += len(records)
                    if scored_count % 25 < len(records) or scored_count == len(missing):
                        print(f"[poison-detection] {interval_name}: scored {scored_count}/{len(missing)} missing exposures", flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

        scores = pd.read_csv(score_path) if score_path.is_file() else pd.DataFrame()
        if not scores.empty:
            scores = scores[pd.to_numeric(scores["stream_position"], errors="coerce").isin(expected_positions)].copy()
            scores = scores.sort_values("stream_position").drop_duplicates("stream_position", keep="last")
            scores = add_interval_normalized_scores(scores)
            scores.to_csv(score_path, index=False)
        if mapped.empty and (interval_dir / "mapped_disruptive_channels.csv").is_file():
            mapped = pd.read_csv(interval_dir / "mapped_disruptive_channels.csv")
        mapped_mask = mapped.get("mapped", pd.Series(False, index=mapped.index)).map(_truthy) if not mapped.empty else pd.Series(dtype=bool)
        n_unique_rows = int(mapped.loc[mapped_mask, "unique_parameter_row_key"].nunique()) if not mapped.empty else 0
        n_mapped_channels = int(mapped_mask.sum()) if not mapped.empty else 0
        metrics = {**base_metric, **detection_metrics(scores)}
        if "wanda_matched_random_score" in scores.columns:
            random_metrics = detection_metrics(scores, score_column="wanda_matched_random_score")
            for key, value in random_metrics.items():
                if key in {"n_examples", "n_poisoned", "poison_prevalence"}:
                    continue
                metrics[f"matched_random_{key}"] = value
            for metric_name in (
                "roc_auc", "average_precision", "precision_at_expected_poison_count",
                "recall_at_expected_poison_count", "paired_poison_over_source_rate",
            ):
                primary_value = metrics.get(metric_name)
                random_value = metrics.get(f"matched_random_{metric_name}")
                try:
                    metrics[f"{metric_name}_minus_matched_random"] = float(primary_value) - float(random_value)
                except (TypeError, ValueError):
                    metrics[f"{metric_name}_minus_matched_random"] = math.nan
        control_metric_values: dict[str, list[float]] = {}
        for draw in range(int(args.matched_control_draws)):
            col = f"wanda_matched_control_{draw:02d}_score"
            if col not in scores.columns:
                continue
            draw_metrics = detection_metrics(scores, score_column=col)
            for metric_name in ("roc_auc", "average_precision", "precision_at_expected_poison_count", "recall_at_expected_poison_count", "paired_poison_over_source_rate"):
                value = draw_metrics.get(metric_name)
                if value is not None and np.isfinite(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]):
                    control_metric_values.setdefault(metric_name, []).append(float(value))
        for metric_name, values in control_metric_values.items():
            arr = np.asarray(values, dtype=float)
            metrics[f"matched_control_{metric_name}_mean"] = float(arr.mean())
            metrics[f"matched_control_{metric_name}_range_low"] = float(np.quantile(arr, 0.025))
            metrics[f"matched_control_{metric_name}_range_high"] = float(np.quantile(arr, 0.975))
            primary = metrics.get(metric_name)
            if primary is not None and np.isfinite(float(primary)):
                less = float(np.sum(arr < float(primary)))
                ties = float(np.sum(arr == float(primary)))
                metrics[f"{metric_name}_candidate_percentile_vs_matched_controls"] = (less + 0.5 * ties) / float(len(arr))
        metrics["n_matched_control_realizations"] = int(max((len(v) for v in control_metric_values.values()), default=0))
        metrics["n_mapped_causal_channels"] = n_mapped_channels
        metrics["n_unique_mapped_parameter_rows"] = n_unique_rows
        metrics["n_gqa_or_other_collapsed_channel_mappings"] = max(0, n_mapped_channels - n_unique_rows)
        if not mapped.empty and "matched_random_update_norm_abs_difference" in mapped.columns:
            diffs = pd.to_numeric(mapped.loc[mapped_mask, "matched_random_update_norm_abs_difference"], errors="coerce").dropna()
            metrics["matched_random_update_norm_abs_difference_mean"] = float(diffs.mean()) if len(diffs) else math.nan
            metrics["matched_random_update_norm_abs_difference_max"] = float(diffs.max()) if len(diffs) else math.nan
        metrics["n_scored_exposures"] = int(len(scores))
        metrics["complete"] = bool(len(scores) == len(exposures))
        (interval_dir / "interval_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        metric_rows.append(metrics)
        if not scores.empty:
            all_score_frames.append(scores)
            scores.sort_values(["wanda_interval_percentile", "wanda_interval_robust_z", "wanda_disruption_score"], ascending=False).head(50).to_csv(
                interval_dir / "top_suspected_training_examples.csv", index=False
            )
        print(
            f"[poison-detection] {interval_name}: AUC={metrics.get('roc_auc')} AP={metrics.get('average_precision')} "
            f"matched_random_AUC={metrics.get('matched_random_roc_auc')} "
            f"precision@K={metrics.get('precision_at_expected_poison_count')} unique_rows={n_unique_rows}",
            flush=True,
        )

    channels_all = pd.concat(all_channel_frames, ignore_index=True, sort=False) if all_channel_frames else pd.DataFrame()
    selected_all = pd.concat(all_selected_frames, ignore_index=True, sort=False) if all_selected_frames else pd.DataFrame()
    scores_all = pd.concat(all_score_frames, ignore_index=True, sort=False) if all_score_frames else pd.DataFrame()
    metrics_all = pd.DataFrame(metric_rows)
    channels_all.to_csv(output_dir / "ordinary_channel_disruption_by_interval.csv", index=False)
    selected_all.to_csv(output_dir / "selected_disruptive_channels_by_interval.csv", index=False)
    scores_all.to_csv(output_dir / "training_example_scores_all_intervals.csv", index=False)
    metrics_all.to_csv(output_dir / "detection_metrics_by_interval.csv", index=False)
    if not scores_all.empty:
        scores_all.sort_values(
            ["wanda_interval_percentile", "wanda_interval_robust_z", "wanda_disruption_score"],
            ascending=False,
        ).head(200).to_csv(output_dir / "top_suspected_training_examples.csv", index=False)
    # Generate compact implication-first figures and align row rankability with
    # held-out backdoor efficacy at the end of each interval.  This is strictly
    # post-hoc: attack behavior never enters candidate selection or row scoring.
    visualization_result = generate_implication_outputs(
        run_dir=run_dir,
        phase=phase,
        eval_intervention=args.eval_intervention,
        output_dir=output_dir,
    )
    summary = {
        "task": task,
        "phase": phase,
        "run_dir": str(run_dir),
        "stage": "07_poisoning_example_detection",
        "scientific_question": "Can poisoned training rows be ranked from clean-normalized disruption of attack-cohort control-correctness channels?",
        "causal_endpoint": "attack_cohort_control_correctness",
        "causal_endpoint_legacy_name": "ordinary_correctness",
        "candidate_definition": f"attack-cohort control-correctness CHA agonists at tau={float(args.required_tau):g}; union across matched clean/poisoned checkpoints",
        "channel_strength": "U(j)=P(correct->incorrect under singleton intervention j) on one fixed held-out attack-eligible non-target cohort",
        "candidate_materialization": "every union candidate is explicitly evaluated at every matched checkpoint; discovery absence is never assigned U(j)=0",
        "score_definition": "WANDA-style |projection input|*|[(effective LoRA poisoned delta)-(effective LoRA clean delta)]|, weighted by |clean-normalized delta U(j)|; output_only shifts supervision one causal-LM token backward and excludes EOS prediction",
        "channel_selection": "requires |D_j| >= min_abs_delta_u and a joint paired-row max-statistic simultaneous bootstrap interval for D_j that excludes zero",
        "multiplicity_control": "family-wise simultaneous max-absolute centered paired-bootstrap band across the full comparable candidate union within each interval",
        "diagnostic_decomposition": "reports baseline ordinary-accuracy drift separately from C->I susceptibility conditional on baseline correctness; fixed-cohort D_j remains the primary endpoint",
        "attention_interpretation": "attention hook_z channels use a value-projection WANDA proxy; the score does not model attention-pattern routing",
        "specificity_control": f"{int(args.matched_control_draws)} same-projection non-candidate matched sets, each cardinality matched and greedily matched on clean-normalized effective-update row norm; metrics report the control distribution",
        "effective_lora_definition": "W=scaling*(B@A); scored interval update is (W_p,end-W_p,start)-(W_c,end-W_c,start)",
        "cross_interval_ranking": "global suspect tables use within-interval percentile/robust-z normalization; raw WANDA values are not compared directly across intervals",
        "ground_truth_policy": "is_poisoned is used only for post-hoc ranking evaluation, never for channel selection or WANDA scoring",
        "normal_training_control": "the matched clean trajectory is one null realization; optional additional clean runs must use distinct seeds and an identical scientific training configuration",
        "trigger_lift_dependency": "none",
        "eval_intervention": args.eval_intervention,
        "n_candidate_union": int(len(candidate_union)),
        "n_clean_null_trajectories": 1 + len(clean_null_materialized),
        "n_intervals": int(len(metrics_all)),
        "attack_behavior_comparison": {
            "primary_metric": "conversion_rate_among_convertible_examples",
            "interpretation": "Among attack-eligible examples not already at the target without the trigger, fraction converted to the target by the trigger.",
            "alignment": "Stage-07 detector metrics summarize rows inside each training interval; Stage-04 backdoor efficacy is measured at that interval's end checkpoint.",
            "primary_association_test": "two-sided permutation Spearman correlation between interval ROC AUC and the change in poisoned conditional conversion over the same interval",
            "association_result": visualization_result.get("primary_detection_attack_association"),
            "association_table": visualization_result.get("association_table"),
            "behavior_source": visualization_result.get("backdoor_behavior_trajectory"),
        },
        "visualization_schema_version": visualization_result.get("visualization_schema_version"),
        "output_dir": str(output_dir),
    }
    (output_dir / "detection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote poisoning-example detection outputs under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
