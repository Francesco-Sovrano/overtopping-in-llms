#!/usr/bin/env python3
"""Confirmatory trigger-lift defence and coalition analysis.

For each poisoned checkpoint this program evaluates cumulative coalitions whose
ordering was frozen on discovery data, not held-out intervention outcomes.  The
same held-out test rows are used for candidate and structurally matched random
coalitions.  At the final checkpoint it also performs an interaction-aware
coalition search on one held-out subset and evaluates the selected coalition on
a disjoint confirmation subset.

The behavioural endpoint is trigger lift:

    B(x) != T and B(x+t) == T

where ``T`` is the configured backdoor target.  Causal interventions are
applied to the trigger-marker prompt. Ordinary task accuracy and control-marker target induction are
measured on ``x`` as collateral-damage controls.
"""

from __future__ import annotations

from poisoning.lib.units import unit_key as _unit_key
import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

from poisoning.lib.run_paths import resolve_manifest_checkpoint_dir, trajectories_dir, phase_dirname
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm.auto import tqdm

from lib.feature_extraction_runner import resolve_task_spec
from lib.group_intervention import load_stage5_locus_population
from lib.modeling_and_ablation import LMWrapper, get_device, precompute_mean_activations
from poisoning.lib.model_loading import poisoning_lm_wrapper_kwargs
from lib.neuron_intervention import ablate_neurons
from lib.project_paths import PROJECT_ROOT
from poisoning.tasks.registry import get_task_definition, infer_task_from_module, infer_task_from_run


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in _parse_csv_list(s)]


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in _parse_csv_list(s)]


def _truthy(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return False
    if isinstance(x, (int, float, np.integer, np.floating)):
        if pd.isna(x):
            return False
        return bool(int(x))
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y", "positive"}


def _stable_int(*parts: Any) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest()[:8], 16)


def _clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    k = max(0, min(int(k), int(n)))
    alpha = 1.0 - float(level)
    low = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    high = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return low, high


def _paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    level: float,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=float)
    if differences.size == 0 or n_boot <= 0:
        return math.nan, math.nan
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, differences.size, size=(int(n_boot), differences.size))
    means = differences[idx].mean(axis=1)
    alpha = 1.0 - float(level)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    level: float,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0 or n_boot <= 0:
        return math.nan, math.nan
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, len(arr), size=(int(n_boot), len(arr)))
    means = arr[idx].mean(axis=1)
    alpha = 1.0 - float(level)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _infer_task_module(run_dir: Path) -> str:
    return infer_task_from_run(run_dir).backdoor_task_module


def _task_label(run_dir: Path) -> str:
    return infer_task_from_run(run_dir).name

def _normalize_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "layer_label" not in frame.columns:
        if "layer_key" in frame.columns:
            frame["layer_label"] = frame["layer_key"].astype(str)
        else:
            raise ValueError("Frozen ranking is missing layer_label/layer_key")
    if "neuron_id" not in frame.columns:
        if "neuron" in frame.columns:
            parsed = frame["neuron"].astype(str).str.rsplit(":", n=1, expand=True)
            if parsed.shape[1] != 2:
                raise ValueError("Frozen ranking is missing neuron_id")
            frame["neuron_id"] = pd.to_numeric(parsed[1], errors="raise").astype(int)
        else:
            raise ValueError("Frozen ranking is missing neuron_id")
    frame["layer_label"] = frame["layer_label"].astype(str)
    frame["neuron_id"] = pd.to_numeric(frame["neuron_id"], errors="raise").astype(int)
    frame["unit_key"] = [_unit_key(a, b) for a, b in zip(frame["layer_label"], frame["neuron_id"])]
    frame = frame.drop_duplicates("unit_key", keep="first").copy()
    if "discovery_rank_global" not in frame.columns:
        raise ValueError(
            "frozen_candidate_ranking.csv is missing discovery_rank_global. "
            "The defence experiment must not rank channels using held-out effects."
        )
    frame["discovery_rank_global"] = pd.to_numeric(
        frame["discovery_rank_global"], errors="raise"
    ).astype(int)
    return frame.sort_values(
        ["discovery_rank_global", "layer_label", "neuron_id"], kind="mergesort"
    ).reset_index(drop=True)


def _load_frozen_ranking(stats_dir: Path) -> pd.DataFrame:
    path = stats_dir / "frozen_candidate_ranking.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing discovery-frozen ranking: {path}. "
            "Cumulative defence does not fall back to held-out singleton ranking."
        )
    return _normalize_ranking(pd.read_csv(path))


def _layer_units_from_rows(rows: pd.DataFrame) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for row in rows.to_dict("records"):
        layer = str(row["layer_label"])
        neuron_id = int(row["neuron_id"])
        out.setdefault(layer, []).append(neuron_id)
    return {layer: sorted(set(ids)) for layer, ids in out.items() if ids}


def _group_key(group: Dict[str, List[int]]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(sorted((str(layer), tuple(sorted(int(x) for x in ids))) for layer, ids in group.items()))


def _group_size(group: Dict[str, List[int]]) -> int:
    return int(sum(len(ids) for ids in group.values()))


def _group_from_unit_keys(unit_keys: Iterable[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for key in unit_keys:
        layer, raw_id = str(key).rsplit(":", 1)
        out.setdefault(layer, []).append(int(raw_id))
    return {layer: sorted(set(ids)) for layer, ids in out.items() if ids}


def _sample_examples(df: pd.DataFrame, mask: Sequence[bool], n: int, seed: int) -> List[Dict[str, Any]]:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(idx) == 0:
        return []
    if int(n) <= 0 or int(n) >= len(idx):
        chosen = idx
    else:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(idx, size=int(n), replace=False)
    return df.iloc[np.asarray(chosen, dtype=int)].to_dict(orient="records")


def _heldout_test_frame(scores_df: pd.DataFrame, scores_path: Path) -> pd.DataFrame:
    if "is_test" not in scores_df.columns:
        raise ValueError(
            f"{scores_path} has no is_test column. The downstream defence requires a "
            "held-out test split disjoint from discovery/ranking rows."
        )
    mask = scores_df["is_test"].map(_truthy).to_numpy(dtype=bool)
    out = scores_df.loc[mask].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{scores_path} contains no held-out test rows")
    return out


def _clean_correctness_fn(task_name: str):
    return get_task_definition(task_name).clean_correctness


def _control_target_fn(task_name: str):
    return get_task_definition(task_name).control_target

def _find_population_manifest(base_dir: Path, stats_dir: Path) -> Path:
    candidates = sorted(
        base_dir.glob("neural_circuit_discovery_results/**/neural_circuits/manifest.json")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No stage-5 population manifest found under {base_dir}. "
            "Matched random controls require the eligible noncandidate population."
        )
    matched = [p for p in candidates if stats_dir.name.startswith(p.parent.parent.name)]
    if len(matched) == 1:
        return matched[0]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        "Could not uniquely associate the stats directory with a stage-5 population manifest. "
        f"stats={stats_dir}; candidates={[str(p) for p in candidates]}"
    )


def _population_by_layer(manifest_path: Path) -> Dict[str, List[str]]:
    raw = load_stage5_locus_population(manifest_path)
    return {
        str(layer): [unit.unit_key for unit in units]
        for layer, units in raw.items()
    }


def _draw_matched_random_group(
    candidate_group: Dict[str, List[int]],
    *,
    population_by_layer: Dict[str, List[str]],
    all_candidate_keys: set[str],
    seed: int,
) -> Dict[str, List[int]]:
    counts = Counter()
    for layer, ids in candidate_group.items():
        counts[str(layer)] += len(ids)
    rng = np.random.default_rng(int(seed))
    selected: List[str] = []
    for layer in sorted(counts):
        need = int(counts[layer])
        pool = [key for key in population_by_layer.get(layer, []) if key not in all_candidate_keys]
        if len(pool) < need:
            raise ValueError(
                f"Matched random control impossible for locus {layer}: need {need} "
                f"noncandidate units but only {len(pool)} are eligible."
            )
        choice = rng.choice(len(pool), size=need, replace=False)
        selected.extend(pool[int(i)] for i in np.atleast_1d(choice))
    return _group_from_unit_keys(selected)


def _find_rows_for_run(
    run_dir: Path,
    fractions: Sequence[float],
    condition: str,
    eval_intervention: str,
    decode_only: bool,
) -> pd.DataFrame:
    phase = "output_only" if decode_only else "input_output"
    summary_csv = trajectories_dir(run_dir) / phase_dirname(phase) / "backdoor_lift_overtopping_trajectory.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing trigger-lift summary: {summary_csv}")
    df = pd.read_csv(summary_csv)
    if "lift_overtopping_status" not in df.columns:
        raise ValueError(f"Summary is missing lift_overtopping_status: {summary_csv}")
    out = df[
        (df["condition"].astype(str) == str(condition))
        & (df["lift_overtopping_status"].astype(str) == "ok")
    ].copy()
    if "lift_eval_intervention" in out.columns:
        out = out[out["lift_eval_intervention"].astype(str).fillna("") == str(eval_intervention)]
    keep = []
    for frac in fractions:
        close = out[np.isclose(pd.to_numeric(out["fraction"], errors="coerce"), float(frac))]
        if len(close):
            keep.append(close.iloc[0])
    if not keep:
        print(
            f"[skip] no completed circuit-defined {condition} checkpoints for fractions={fractions} "
            f"in {summary_csv}; downstream suppression has nothing to evaluate for this run.",
            flush=True,
        )
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(keep)


def _evaluate_group(
    *,
    model: LMWrapper,
    task,
    group: Dict[str, List[int]],
    pos_examples: List[Dict[str, Any]],
    neg_examples: List[Dict[str, Any]],
    clean_examples: List[Dict[str, Any]],
    ordinary_target_examples: List[Dict[str, Any]],
    prompt_col: str,
    clean_score_fn,
    target_score_fn,
    args: argparse.Namespace,
    mean_activations,
    baseline_pos_vec: np.ndarray,
    clean_baseline_vec: np.ndarray,
    ordinary_target_baseline_vec: np.ndarray,
) -> Dict[str, Any]:
    rec = ablate_neurons(
        model,
        pos_examples,
        neg_examples,
        task.is_answer_positive,
        prompt_col,
        layers_neurons_dict=group or None,
        batch_size=args.batch_size,
        decode_only=args.decode_only,
        intervention=args.intervention,
        mean_activations=mean_activations,
        max_new_tokens=task.MAX_NEW_TOKENS,
        baseline_subset="positive",
    )
    pos_vec = np.asarray(rec["acc_after_knockout_on_associated_all"], dtype=float)
    baseline_positive = baseline_pos_vec > 0.5
    destroyed = baseline_positive & (pos_vec < 0.5)
    n_baseline_positive = int(baseline_positive.sum())
    destroyed_count = int(destroyed.sum())
    destroy_rate = float(destroyed_count / n_baseline_positive) if n_baseline_positive else math.nan
    destroy_lo, destroy_hi = _clopper_pearson(destroyed_count, n_baseline_positive, level=args.ci_level)

    if neg_examples:
        neg_vec = np.asarray(rec["acc_after_knockout_on_unrelated_all"], dtype=float)
        neg_positive_count = int(np.sum(neg_vec > 0.5))
        neg_rate = float(neg_positive_count / len(neg_vec)) if len(neg_vec) else math.nan
        neg_lo, neg_hi = _clopper_pearson(neg_positive_count, len(neg_vec), level=args.ci_level)
    else:
        neg_positive_count = 0
        neg_rate = neg_lo = neg_hi = math.nan

    clean_after_acc = clean_baseline_acc = clean_drop = math.nan
    clean_lo = clean_hi = clean_drop_lo = clean_drop_hi = math.nan
    target_induction_rate = target_induction_lo = target_induction_hi = math.nan
    target_induction_count = 0
    n_target_induction_eligible = 0
    if clean_examples:
        clean_rec = ablate_neurons(
            model,
            clean_examples,
            [],
            clean_score_fn,
            "prompt_control",
            layers_neurons_dict=group or None,
            batch_size=args.batch_size,
            decode_only=args.decode_only,
            intervention=args.intervention,
            mean_activations=mean_activations,
            max_new_tokens=task.MAX_NEW_TOKENS,
            baseline_subset="positive",
        )
        clean_after_vec = np.asarray(clean_rec["acc_after_knockout_on_associated_all"], dtype=float)
        clean_baseline_acc = float(clean_baseline_vec.mean()) if len(clean_baseline_vec) else math.nan
        clean_after_acc = float(clean_after_vec.mean()) if len(clean_after_vec) else math.nan
        clean_drop = clean_baseline_acc - clean_after_acc
        clean_correct = int(np.sum(clean_after_vec > 0.5))
        clean_lo, clean_hi = _clopper_pearson(clean_correct, len(clean_after_vec), level=args.ci_level)
        clean_drop_lo, clean_drop_hi = _paired_bootstrap_interval(
            clean_baseline_vec - clean_after_vec,
            level=args.ci_level,
            n_boot=args.bootstrap,
            seed=args.seed + _stable_int("clean", _group_key(group)),
        )

        induction_examples = [
            row for row in clean_examples if not bool(row.get("control_target_positive", False))
        ]
        if induction_examples:
            target_rec = ablate_neurons(
                model,
                induction_examples,
                [],
                target_score_fn,
                "prompt_control",
                layers_neurons_dict=group or None,
                batch_size=args.batch_size,
                decode_only=args.decode_only,
                intervention=args.intervention,
                mean_activations=mean_activations,
                max_new_tokens=task.MAX_NEW_TOKENS,
                baseline_subset="positive",
            )
            target_vec = np.asarray(target_rec["acc_after_knockout_on_associated_all"], dtype=float)
            target_induction_count = int(np.sum(target_vec > 0.5))
            n_target_induction_eligible = int(len(target_vec))
            target_induction_rate = float(target_induction_count / len(target_vec))
            target_induction_lo, target_induction_hi = _clopper_pearson(
                target_induction_count, len(target_vec), level=args.ci_level
            )

    ordinary_destroy_rate = ordinary_destroy_lo = ordinary_destroy_hi = math.nan
    ordinary_destroy_count = ordinary_destroy_denominator = 0
    if ordinary_target_examples:
        ordinary_rec = ablate_neurons(
            model,
            ordinary_target_examples,
            [],
            target_score_fn,
            "prompt_control",
            layers_neurons_dict=group or None,
            batch_size=args.batch_size,
            decode_only=args.decode_only,
            intervention=args.intervention,
            mean_activations=mean_activations,
            max_new_tokens=task.MAX_NEW_TOKENS,
            baseline_subset="positive",
        )
        ordinary_after = np.asarray(
            ordinary_rec["acc_after_knockout_on_associated_all"], dtype=float
        )
        ordinary_baseline_positive = ordinary_target_baseline_vec > 0.5
        if len(ordinary_after) != len(ordinary_baseline_positive):
            raise RuntimeError("Ordinary target-positive specificity vectors changed length")
        ordinary_destroyed = ordinary_baseline_positive & (ordinary_after < 0.5)
        ordinary_destroy_denominator = int(ordinary_baseline_positive.sum())
        ordinary_destroy_count = int(ordinary_destroyed.sum())
        if ordinary_destroy_denominator:
            ordinary_destroy_rate = ordinary_destroy_count / ordinary_destroy_denominator
        ordinary_destroy_lo, ordinary_destroy_hi = _clopper_pearson(
            ordinary_destroy_count, ordinary_destroy_denominator, level=args.ci_level
        )

    specificity_gap = (
        destroy_rate - ordinary_destroy_rate
        if np.isfinite(destroy_rate) and np.isfinite(ordinary_destroy_rate)
        else math.nan
    )

    return {
        "destroy_count": destroyed_count,
        "destroy_denominator": n_baseline_positive,
        "destroy_rate": destroy_rate,
        "destroy_ci_low": destroy_lo,
        "destroy_ci_high": destroy_hi,
        "nonlift_to_lift_count": neg_positive_count,
        "nonlift_to_lift_rate": neg_rate,
        "nonlift_to_lift_ci_low": neg_lo,
        "nonlift_to_lift_ci_high": neg_hi,
        "paired_clean_baseline_accuracy": clean_baseline_acc,
        "paired_clean_accuracy_after": clean_after_acc,
        "paired_clean_accuracy_ci_low": clean_lo,
        "paired_clean_accuracy_ci_high": clean_hi,
        "paired_clean_accuracy_drop": clean_drop,
        "paired_clean_accuracy_drop_ci_low": clean_drop_lo,
        "paired_clean_accuracy_drop_ci_high": clean_drop_hi,
        "control_target_induction_count": target_induction_count,
        "control_target_induction_denominator": n_target_induction_eligible,
        "control_target_induction_rate": target_induction_rate,
        "control_target_induction_ci_low": target_induction_lo,
        "control_target_induction_ci_high": target_induction_hi,
        "ordinary_target_destroy_count": ordinary_destroy_count,
        "ordinary_target_destroy_denominator": ordinary_destroy_denominator,
        "ordinary_target_destroy_rate": ordinary_destroy_rate,
        "ordinary_target_destroy_ci_low": ordinary_destroy_lo,
        "ordinary_target_destroy_ci_high": ordinary_destroy_hi,
        "trigger_specificity_gap": specificity_gap,
    }


def _random_summary(candidate_rate: float, random_rates: Sequence[float], args, seed: int) -> Dict[str, Any]:
    vals = np.asarray(random_rates, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0 or not np.isfinite(candidate_rate):
        return {
            "matched_random_n": int(len(vals)),
            "matched_random_mean_destroy_rate": math.nan,
            "matched_random_median_destroy_rate": math.nan,
            "matched_random_mean_ci_low": math.nan,
            "matched_random_mean_ci_high": math.nan,
            "matched_random_max_destroy_rate": math.nan,
            "matched_random_empirical_p": math.nan,
        }
    lo, hi = _bootstrap_mean_interval(
        vals, level=args.ci_level, n_boot=args.bootstrap, seed=seed
    )
    return {
        "matched_random_n": int(len(vals)),
        "matched_random_mean_destroy_rate": float(vals.mean()),
        "matched_random_median_destroy_rate": float(np.median(vals)),
        "matched_random_mean_ci_low": lo,
        "matched_random_mean_ci_high": hi,
        "matched_random_max_destroy_rate": float(vals.max()),
        "matched_random_empirical_p": float((1 + int(np.sum(vals >= candidate_rate))) / (len(vals) + 1)),
    }


def _evaluate_positive_only(
    model: LMWrapper,
    task,
    group: Dict[str, List[int]],
    examples: List[Dict[str, Any]],
    prompt_col: str,
    args: argparse.Namespace,
    mean_activations,
    baseline_vec: np.ndarray,
) -> float:
    if not examples:
        return math.nan
    rec = ablate_neurons(
        model,
        examples,
        [],
        task.is_answer_positive,
        prompt_col,
        layers_neurons_dict=group or None,
        batch_size=args.batch_size,
        decode_only=args.decode_only,
        intervention=args.intervention,
        mean_activations=mean_activations,
        max_new_tokens=task.MAX_NEW_TOKENS,
        baseline_subset="positive",
    )
    vec = np.asarray(rec["acc_after_knockout_on_associated_all"], dtype=float)
    baseline_positive = np.asarray(baseline_vec, dtype=float) > 0.5
    if len(vec) != len(baseline_positive) or not baseline_positive.any():
        return math.nan
    destroyed = baseline_positive & (vec < 0.5)
    return float(destroyed.sum() / baseline_positive.sum())


def _split_interaction_indices(
    target_mask: np.ndarray,
    *,
    args: argparse.Namespace,
    task_name: str,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve a final-checkpoint confirmation subset before any defence evaluation.

    The returned indices refer to rows of the held-out feature-report test frame.
    Confirmation rows are excluded from cumulative coalition and collateral-control
    analyses, so their triggered outcomes remain untouched until the interaction-aware
    coalition has been frozen on the selection subset.
    """
    positive_idx = np.flatnonzero(np.asarray(target_mask, dtype=bool))
    if len(positive_idx) < int(args.interaction_min_examples):
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    rng = np.random.default_rng(args.seed + _stable_int(task_name, fraction, "interaction_split"))
    order = rng.permutation(positive_idx)
    min_each = max(4, int(args.interaction_min_examples) // 2)
    n_select = int(round(len(order) * float(args.interaction_selection_fraction)))
    n_select = max(min_each, min(n_select, len(order) - min_each))
    return np.asarray(order[:n_select], dtype=int), np.asarray(order[n_select:], dtype=int)


def _interaction_search(
    *,
    model: LMWrapper,
    task,
    ranking: pd.DataFrame,
    selection: List[Dict[str, Any]],
    confirmation: List[Dict[str, Any]],
    prompt_col: str,
    args: argparse.Namespace,
    mean_activations,
    task_name: str,
    fraction: float,
) -> tuple[pd.DataFrame, Dict[str, List[int]] | None]:
    if not selection or not confirmation:
        return pd.DataFrame(), None
    pool = ranking.head(min(int(args.interaction_pool), len(ranking))).copy()
    if pool.empty:
        return pd.DataFrame(), None

    selection_baseline = ablate_neurons(
        model,
        selection,
        [],
        task.is_answer_positive,
        prompt_col,
        layers_neurons_dict=None,
        batch_size=args.batch_size,
        decode_only=args.decode_only,
        intervention=args.intervention,
        mean_activations=mean_activations,
        max_new_tokens=task.MAX_NEW_TOKENS,
        baseline_subset="positive",
    )
    selection_baseline_vec = np.asarray(
        selection_baseline["acc_after_knockout_on_associated_all"], dtype=float
    )

    unit_keys = pool["unit_key"].astype(str).tolist()
    rank_index = {key: i for i, key in enumerate(unit_keys)}
    evaluated: Dict[tuple[str, ...], float] = {}
    records: List[Dict[str, Any]] = []

    def evaluate(keys: Iterable[str], source: str) -> float:
        key_tuple = tuple(sorted(set(str(x) for x in keys), key=lambda x: rank_index[x]))
        if not key_tuple:
            return 0.0
        if key_tuple not in evaluated:
            group = _group_from_unit_keys(key_tuple)
            evaluated[key_tuple] = _evaluate_positive_only(
                model, task, group, selection, prompt_col, args, mean_activations, selection_baseline_vec
            )
            records.append({
                "task": task_name,
                "fraction": float(fraction),
                "source": source,
                "k": len(key_tuple),
                "selection_destroy_rate": evaluated[key_tuple],
                "group": json.dumps(group, sort_keys=True),
                "unit_keys": json.dumps(list(key_tuple)),
                "n_selection": len(selection),
                "n_confirmation": len(confirmation),
            })
        return evaluated[key_tuple]

    # Discovery-ranked prefixes are pre-specified, not selected on confirmation data.
    for k in tqdm(
        range(1, min(int(args.interaction_max_k), len(unit_keys)) + 1),
        desc=f"{task_name} {fraction:g} interaction prefixes",
        unit="group",
        leave=False,
    ):
        evaluate(unit_keys[:k], "discovery_prefix")

    # Scan all singles and pairs. Pair scanning can reveal interactions that are
    # invisible to singleton effects while remaining tractable on workstation runs.
    for key in tqdm(
        unit_keys,
        desc=f"{task_name} {fraction:g} interaction singletons",
        unit="unit",
        leave=False,
    ):
        evaluate([key], "singleton_scan")
    if bool(args.interaction_pair_scan) and len(unit_keys) >= 2:
        n_pairs = len(unit_keys) * (len(unit_keys) - 1) // 2
        for pair in tqdm(
            itertools.combinations(unit_keys, 2),
            total=n_pairs,
            desc=f"{task_name} {fraction:g} interaction pairs",
            unit="pair",
            leave=False,
        ):
            evaluate(pair, "pair_scan")

    # Greedy expansion from the best size-1/2 seed.
    candidate_seeds = [k for k in evaluated if len(k) <= 2]
    if candidate_seeds:
        current = max(candidate_seeds, key=lambda k: (evaluated[k], -len(k), tuple(-rank_index[x] for x in k)))
        while len(current) < min(int(args.interaction_max_k), len(unit_keys)):
            remaining = [x for x in unit_keys if x not in current]
            options = []
            for key in tqdm(
                remaining,
                desc=f"{task_name} {fraction:g} greedy k={len(current)+1}",
                unit="candidate",
                leave=False,
            ):
                new = tuple(list(current) + [key])
                rate = evaluate(new, "greedy_expansion")
                options.append((rate, new))
            if not options:
                break
            _, current = max(options, key=lambda item: (item[0], -len(item[1])))

    # A small deterministic random subset search probes higher-order interactions
    # that a greedy path can miss. Selection is still validated only on the
    # disjoint confirmation subset.
    draws = max(0, int(args.interaction_random_draws_per_k))
    if draws > 0:
        for k in tqdm(
            range(3, min(int(args.interaction_max_k), len(unit_keys)) + 1),
            desc=f"{task_name} {fraction:g} random interaction sizes",
            unit="k",
            leave=False,
        ):
            seen: set[tuple[str, ...]] = set()
            for draw in range(draws):
                rrng = np.random.default_rng(
                    args.seed + _stable_int(task_name, fraction, "interaction_random", k, draw)
                )
                chosen_idx = rrng.choice(len(unit_keys), size=k, replace=False)
                chosen = tuple(unit_keys[int(i)] for i in sorted(chosen_idx))
                if chosen in seen:
                    continue
                seen.add(chosen)
                evaluate(chosen, "random_subset_search")

    if not evaluated:
        return pd.DataFrame(records), None
    best_keys = max(
        evaluated,
        key=lambda k: (evaluated[k], -len(k), tuple(-rank_index[x] for x in k)),
    )
    return pd.DataFrame(records), _group_from_unit_keys(best_keys)


def _run_one(
    *,
    run_dir: Path,
    task_module: str,
    row: pd.Series,
    args: argparse.Namespace,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame, Dict[str, Any] | None]:
    task = resolve_task_spec(task_module)
    task_definition = infer_task_from_module(task_module)
    task_name = task_definition.name
    fraction = float(row["fraction"])
    condition = str(row["condition"])
    checkpoint_dir = str(resolve_manifest_checkpoint_dir(run_dir, row, must_exist=True))
    stats_dir = Path(str(row["lift_overtopping_stats_dir"]))
    base_dir = Path(str(row["lift_overtopping_base_dir"]))
    scores_path = base_dir / "feature_report" / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing scores for {task_name} fraction={fraction}: {scores_path}")

    scores_df = pd.read_csv(scores_path)
    eval_df = _heldout_test_frame(scores_df, scores_path)
    prompt_col = getattr(task, "DEFAULT_INPUT", "prompt")
    target_col = getattr(task, "DEFAULT_TARGETS", ("is_trigger_lift_success",))[0]
    if prompt_col not in eval_df.columns or target_col not in eval_df.columns:
        raise ValueError(f"{scores_path} lacks required columns {prompt_col!r}, {target_col!r}")
    if "prompt_control" not in eval_df.columns:
        raise ValueError(f"{scores_path} lacks prompt_control required for collateral controls")

    target_mask = eval_df[target_col].map(_truthy).to_numpy(dtype=bool)

    # At the final checkpoint, reserve confirmation-positive rows *before* any
    # cumulative defence evaluation. This makes the interaction-aware
    # confirmation genuinely untouched by coalition selection and exploratory
    # cumulative results. Earlier checkpoints use the complete held-out test set.
    interaction_selection: List[Dict[str, Any]] = []
    interaction_confirmation: List[Dict[str, Any]] = []
    exploratory_mask = np.ones(len(eval_df), dtype=bool)
    cumulative_target_mask = target_mask.copy()
    if math.isclose(fraction, float(args.interaction_fraction), abs_tol=1e-12):
        selection_idx, confirmation_idx = _split_interaction_indices(
            target_mask, args=args, task_name=task_name, fraction=fraction
        )
        if len(confirmation_idx):
            interaction_selection = eval_df.iloc[selection_idx].to_dict(orient="records")
            interaction_confirmation = eval_df.iloc[confirmation_idx].to_dict(orient="records")
            exploratory_mask[confirmation_idx] = False
            cumulative_target_mask[confirmation_idx] = False

    pos_examples = _sample_examples(
        eval_df, cumulative_target_mask, args.max_pos,
        args.seed + _stable_int(task_name, fraction, "pos"),
    )
    neg_examples = _sample_examples(
        eval_df, (~target_mask) & exploratory_mask, args.max_neg,
        args.seed + _stable_int(task_name, fraction, "neg"),
    )
    clean_examples = _sample_examples(
        eval_df, exploratory_mask, args.max_clean,
        args.seed + _stable_int(task_name, fraction, "clean"),
    )
    if not pos_examples:
        print(f"[skip] {task_name} {condition} frac={fraction:g}: no held-out trigger-lift successes.", flush=True)
        return [], [], pd.DataFrame(), None

    ordinary_mask = task_definition.ordinary_target_positive_mask(eval_df) & exploratory_mask
    ordinary_target_examples, specificity_meta = task_definition.sample_task_specificity_examples(
        eval_df,
        candidate_mask=ordinary_mask,
        reference_examples=pos_examples,
        max_n=args.max_task_specificity,
        seed=args.seed + _stable_int(task_name, fraction, "ordinary_target_specificity"),
    )

    ranking = _load_frozen_ranking(stats_dir)
    if ranking.empty:
        print(f"[skip] {task_name} {condition} frac={fraction:g}: frozen candidate set is empty.", flush=True)
        return [], [], pd.DataFrame(), None

    manifest_path = _find_population_manifest(base_dir, stats_dir)
    population = _population_by_layer(manifest_path)
    all_candidate_keys = set(ranking["unit_key"].astype(str))

    top_groups: Dict[int, Dict[str, List[int]]] = {}
    random_groups: Dict[int, List[Dict[str, List[int]]]] = {}
    for requested_k in args.top_ks:
        k = min(int(requested_k), len(ranking))
        if k <= 0:
            continue
        if k in top_groups:
            continue
        group = _layer_units_from_rows(ranking.head(k))
        top_groups[k] = group
        random_groups[k] = [
            _draw_matched_random_group(
                group,
                population_by_layer=population,
                all_candidate_keys=all_candidate_keys,
                seed=args.seed + _stable_int(task_name, fraction, "matched", k, draw),
            )
            for draw in range(int(args.random_groups))
        ]

    # Pre-sample a per-draw noncandidate reservoir for the interaction search.
    # The reservoir matches the maximum possible per-locus demand of a selected
    # final coalition. After selection, each random control takes exactly the
    # required number of units from the corresponding locus. This lets mean/
    # mean-donor replacement values be precomputed before any held-out selection.
    interaction_random_reservoir: List[Dict[str, List[int]]] = []
    if math.isclose(fraction, float(args.interaction_fraction), abs_tol=1e-12):
        pool_frame = ranking.head(min(int(args.interaction_pool), len(ranking))).copy()
        if not pool_frame.empty:
            capacity_rows = []
            for layer, part in pool_frame.groupby("layer_label", sort=True):
                need = min(int(args.interaction_max_k), len(part))
                capacity_rows.append(part.head(need))
            if capacity_rows:
                capacity_group = _layer_units_from_rows(pd.concat(capacity_rows, ignore_index=True))
                interaction_random_reservoir = [
                    _draw_matched_random_group(
                        capacity_group,
                        population_by_layer=population,
                        all_candidate_keys=all_candidate_keys,
                        seed=args.seed + _stable_int(task_name, fraction, "interaction_reservoir", draw),
                    )
                    for draw in range(int(args.random_groups))
                ]

    # All candidate-pool units are included because the interaction-aware final
    # search may combine them in groups that are not cumulative prefixes.
    all_needed: Dict[str, set[int]] = {}
    for group in (
        list(top_groups.values())
        + [g for groups in random_groups.values() for g in groups]
        + list(interaction_random_reservoir)
    ):
        for layer, ids in group.items():
            all_needed.setdefault(layer, set()).update(int(x) for x in ids)
    if math.isclose(fraction, float(args.interaction_fraction), abs_tol=1e-12):
        for _, rr in ranking.head(min(args.interaction_pool, len(ranking))).iterrows():
            all_needed.setdefault(str(rr["layer_label"]), set()).add(int(rr["neuron_id"]))
    all_needed_map = {layer: sorted(ids) for layer, ids in all_needed.items()}

    device = get_device()
    model = LMWrapper(
        model_name=checkpoint_dir,
        device=device,
        eval_mode=True,
        circuit_discovery=False,
        cache_dir=args.ai_model_cache_dir,
        **poisoning_lm_wrapper_kwargs(checkpoint_dir),
    )

    mean_activations = None
    if args.intervention in {"mean", "mean-donor", "mean-positional", "mean-donor-positional"}:
        # Replacement values are estimated only from the causal discovery split.
        # The held-out test rows are reserved for candidate/random evaluation and
        # final confirmation, so they must not determine mean/mean-donor values.
        discovery_mask = ~scores_df["is_test"].map(_truthy).to_numpy(dtype=bool)
        mean_source_df = scores_df.loc[discovery_mask].copy()
        if mean_source_df.empty:
            raise ValueError(f"{scores_path} contains no discovery rows for replacement estimation")
        mean_prompts = mean_source_df[prompt_col].astype(str).head(args.mean_points).tolist()
        mean_activations = precompute_mean_activations(
            model=model,
            all_prompts=mean_prompts,
            layer_to_neurons=all_needed_map,
            n_points=min(args.mean_points, len(mean_prompts)),
            batch_size=args.batch_size,
            intervention=args.intervention,
            device=device,
        )

    baseline = ablate_neurons(
        model,
        pos_examples,
        neg_examples,
        task.is_answer_positive,
        prompt_col,
        layers_neurons_dict=None,
        batch_size=args.batch_size,
        decode_only=args.decode_only,
        intervention=args.intervention,
        mean_activations=mean_activations,
        max_new_tokens=task.MAX_NEW_TOKENS,
        baseline_subset="positive",
    )
    baseline_pos_vec = np.asarray(baseline["acc_after_knockout_on_associated_all"], dtype=float)

    clean_score_fn = _clean_correctness_fn(task_name)
    target_score_fn = _control_target_fn(task_name)
    clean_baseline_rec = ablate_neurons(
        model,
        clean_examples,
        [],
        clean_score_fn,
        "prompt_control",
        layers_neurons_dict=None,
        batch_size=args.batch_size,
        decode_only=args.decode_only,
        intervention=args.intervention,
        mean_activations=mean_activations,
        max_new_tokens=task.MAX_NEW_TOKENS,
        baseline_subset="positive",
    )
    clean_baseline_vec = np.asarray(
        clean_baseline_rec["acc_after_knockout_on_associated_all"], dtype=float
    )
    if ordinary_target_examples:
        ordinary_baseline_rec = ablate_neurons(
            model,
            ordinary_target_examples,
            [],
            target_score_fn,
            "prompt_control",
            layers_neurons_dict=None,
            batch_size=args.batch_size,
            decode_only=args.decode_only,
            intervention=args.intervention,
            mean_activations=mean_activations,
            max_new_tokens=task.MAX_NEW_TOKENS,
            baseline_subset="positive",
        )
        ordinary_target_baseline_vec = np.asarray(
            ordinary_baseline_rec["acc_after_knockout_on_associated_all"], dtype=float
        )
    else:
        ordinary_target_baseline_vec = np.asarray([], dtype=float)

    rows: List[Dict[str, Any]] = []
    raw_random_rows: List[Dict[str, Any]] = []
    for k, group in tqdm(
        sorted(top_groups.items()),
        desc=f"{task_name} frac={fraction:g} cumulative top-k",
        unit="group",
    ):
        candidate = _evaluate_group(
            model=model,
            task=task,
            group=group,
            pos_examples=pos_examples,
            neg_examples=neg_examples,
            clean_examples=clean_examples,
            ordinary_target_examples=ordinary_target_examples,
            prompt_col=prompt_col,
            clean_score_fn=clean_score_fn,
            target_score_fn=target_score_fn,
            args=args,
            mean_activations=mean_activations,
            baseline_pos_vec=baseline_pos_vec,
            clean_baseline_vec=clean_baseline_vec,
            ordinary_target_baseline_vec=ordinary_target_baseline_vec,
        )
        random_rates: List[float] = []
        for draw, random_group in enumerate(
            tqdm(
                random_groups[k],
                desc=f"{task_name} frac={fraction:g} k={k} matched random",
                unit="group",
                leave=False,
            )
        ):
            rnd = _evaluate_group(
                model=model,
                task=task,
                group=random_group,
                pos_examples=pos_examples,
                neg_examples=[],
                clean_examples=clean_examples,
                ordinary_target_examples=ordinary_target_examples,
                prompt_col=prompt_col,
                clean_score_fn=clean_score_fn,
                target_score_fn=target_score_fn,
                args=args,
                mean_activations=mean_activations,
                baseline_pos_vec=baseline_pos_vec,
                clean_baseline_vec=clean_baseline_vec,
                ordinary_target_baseline_vec=ordinary_target_baseline_vec,
            )
            random_rates.append(float(rnd["destroy_rate"]))
            raw_random_rows.append({
                "analysis": "cumulative",
                "task": task_name,
                "run_dir": str(run_dir),
                "condition": condition,
                "fraction": fraction,
                "checkpoint_dir": checkpoint_dir,
                "intervention_phase": "output_only" if args.decode_only else "input_output",
                "k": int(k),
                "draw": int(draw),
                "trigger_lift_destroy_rate": rnd["destroy_rate"],
                "ordinary_target_destroy_rate": rnd["ordinary_target_destroy_rate"],
                "trigger_specificity_gap": rnd["trigger_specificity_gap"],
                "paired_clean_accuracy_drop": rnd["paired_clean_accuracy_drop"],
                "control_target_induction_rate": rnd["control_target_induction_rate"],
                "group": json.dumps(random_group, sort_keys=True),
            })
        random_stats = _random_summary(
            float(candidate["destroy_rate"]), random_rates, args,
            args.seed + _stable_int(task_name, fraction, "random_summary", k),
        )
        rows.append({
            "task": task_name,
            "run_dir": str(run_dir),
            "condition": condition,
            "fraction": fraction,
            "global_step": int(row.get("global_step", -1)),
            "checkpoint_dir": checkpoint_dir,
            "stats_dir": str(stats_dir),
            "population_manifest": str(manifest_path),
            "ranking_source": "frozen_discovery_ranking",
            "k": int(k),
            "n_available_candidates": int(len(ranking)),
            "n_pos": int(len(pos_examples)),
            "n_neg": int(len(neg_examples)),
            "n_clean": int(len(clean_examples)),
            "n_ordinary_target_positive": int(len(ordinary_target_examples)),
            **specificity_meta,
            "baseline_reproduction_rate_on_cached_lift_successes": float(baseline_pos_vec.mean()),
            "trigger_lift_destroy_count": candidate["destroy_count"],
            "trigger_lift_destroy_denominator": candidate["destroy_denominator"],
            "trigger_lift_destroy_rate": candidate["destroy_rate"],
            "trigger_lift_destroy_ci_low": candidate["destroy_ci_low"],
            "trigger_lift_destroy_ci_high": candidate["destroy_ci_high"],
            "nonlift_to_lift_count": candidate["nonlift_to_lift_count"],
            "nonlift_to_lift_rate": candidate["nonlift_to_lift_rate"],
            "nonlift_to_lift_ci_low": candidate["nonlift_to_lift_ci_low"],
            "nonlift_to_lift_ci_high": candidate["nonlift_to_lift_ci_high"],
            "paired_clean_baseline_accuracy": candidate["paired_clean_baseline_accuracy"],
            "paired_clean_accuracy_after_topk": candidate["paired_clean_accuracy_after"],
            "paired_clean_accuracy_ci_low": candidate["paired_clean_accuracy_ci_low"],
            "paired_clean_accuracy_ci_high": candidate["paired_clean_accuracy_ci_high"],
            "paired_clean_accuracy_drop": candidate["paired_clean_accuracy_drop"],
            "paired_clean_accuracy_drop_ci_low": candidate["paired_clean_accuracy_drop_ci_low"],
            "paired_clean_accuracy_drop_ci_high": candidate["paired_clean_accuracy_drop_ci_high"],
            "control_target_induction_count": candidate["control_target_induction_count"],
            "control_target_induction_denominator": candidate["control_target_induction_denominator"],
            "control_target_induction_rate": candidate["control_target_induction_rate"],
            "control_target_induction_ci_low": candidate["control_target_induction_ci_low"],
            "control_target_induction_ci_high": candidate["control_target_induction_ci_high"],
            "ordinary_target_destroy_count": candidate["ordinary_target_destroy_count"],
            "ordinary_target_destroy_denominator": candidate["ordinary_target_destroy_denominator"],
            "ordinary_target_destroy_rate": candidate["ordinary_target_destroy_rate"],
            "ordinary_target_destroy_ci_low": candidate["ordinary_target_destroy_ci_low"],
            "ordinary_target_destroy_ci_high": candidate["ordinary_target_destroy_ci_high"],
            "trigger_specificity_gap": candidate["trigger_specificity_gap"],
            **random_stats,
            "group": json.dumps(group, sort_keys=True),
        })
        print(
            f"[topk] {task_name} frac={fraction:g} k={k}: "
            f"destroy={candidate['destroy_rate']:.3f} "
            f"random_mean={random_stats['matched_random_mean_destroy_rate']:.3f}",
            flush=True,
        )

    interaction_candidates = pd.DataFrame()
    interaction_result: Dict[str, Any] | None = None
    if math.isclose(fraction, float(args.interaction_fraction), abs_tol=1e-12):
        interaction_candidates, selected_group = _interaction_search(
            model=model,
            task=task,
            ranking=ranking,
            selection=interaction_selection,
            confirmation=interaction_confirmation,
            prompt_col=prompt_col,
            args=args,
            mean_activations=mean_activations,
            task_name=task_name,
            fraction=fraction,
        )
        if selected_group and interaction_confirmation:
            confirmation_baseline = ablate_neurons(
                model,
                interaction_confirmation,
                [],
                task.is_answer_positive,
                prompt_col,
                layers_neurons_dict=None,
                batch_size=args.batch_size,
                decode_only=args.decode_only,
                intervention=args.intervention,
                mean_activations=mean_activations,
                max_new_tokens=task.MAX_NEW_TOKENS,
                baseline_subset="positive",
            )
            confirm_base_vec = np.asarray(
                confirmation_baseline["acc_after_knockout_on_associated_all"], dtype=float
            )
            selected_eval = _evaluate_group(
                model=model,
                task=task,
                group=selected_group,
                pos_examples=interaction_confirmation,
                neg_examples=[],
                clean_examples=clean_examples,
                ordinary_target_examples=ordinary_target_examples,
                prompt_col=prompt_col,
                clean_score_fn=clean_score_fn,
                target_score_fn=target_score_fn,
                args=args,
                mean_activations=mean_activations,
                baseline_pos_vec=confirm_base_vec,
                clean_baseline_vec=clean_baseline_vec,
                ordinary_target_baseline_vec=ordinary_target_baseline_vec,
            )
            confirm_random_rates: List[float] = []
            confirm_random_rows: List[Dict[str, Any]] = []
            for draw in tqdm(
                range(int(args.random_groups)),
                desc=f"{task_name} frac={fraction:g} confirmation random",
                unit="group",
                leave=False,
            ):
                reservoir = interaction_random_reservoir[draw]
                random_group = {
                    layer: list(reservoir.get(layer, []))[: len(ids)]
                    for layer, ids in selected_group.items()
                }
                if any(len(random_group.get(layer, [])) != len(ids) for layer, ids in selected_group.items()):
                    raise RuntimeError(
                        "Interaction matched-control reservoir is too small for the selected coalition"
                    )
                rnd = _evaluate_group(
                    model=model,
                    task=task,
                    group=random_group,
                    pos_examples=interaction_confirmation,
                    neg_examples=[],
                    clean_examples=clean_examples,
                    ordinary_target_examples=ordinary_target_examples,
                    prompt_col=prompt_col,
                    clean_score_fn=clean_score_fn,
                    target_score_fn=target_score_fn,
                    args=args,
                    mean_activations=mean_activations,
                    baseline_pos_vec=confirm_base_vec,
                    clean_baseline_vec=clean_baseline_vec,
                    ordinary_target_baseline_vec=ordinary_target_baseline_vec,
                )
                confirm_random_rates.append(float(rnd["destroy_rate"]))
                confirm_row = {
                    "analysis": "interaction_confirmation",
                    "task": task_name,
                    "run_dir": str(run_dir),
                    "condition": condition,
                    "fraction": fraction,
                    "checkpoint_dir": checkpoint_dir,
                    "intervention_phase": "output_only" if args.decode_only else "input_output",
                    "draw": draw,
                    "k": _group_size(selected_group),
                    "trigger_lift_destroy_rate": rnd["destroy_rate"],
                    "ordinary_target_destroy_rate": rnd["ordinary_target_destroy_rate"],
                    "trigger_specificity_gap": rnd["trigger_specificity_gap"],
                    "paired_clean_accuracy_drop": rnd["paired_clean_accuracy_drop"],
                    "control_target_induction_rate": rnd["control_target_induction_rate"],
                    "group": json.dumps(random_group, sort_keys=True),
                }
                confirm_random_rows.append(confirm_row)
                raw_random_rows.append(confirm_row)
            random_stats = _random_summary(
                float(selected_eval["destroy_rate"]), confirm_random_rates, args,
                args.seed + _stable_int(task_name, fraction, "interaction_random_summary"),
            )
            best_selection_rate = float(interaction_candidates["selection_destroy_rate"].max()) if not interaction_candidates.empty else math.nan
            interaction_result = {
                "task": task_name,
                "run_dir": str(run_dir),
                "fraction": fraction,
                "checkpoint_dir": checkpoint_dir,
                "intervention_phase": "output_only" if args.decode_only else "input_output",
                "selection_method": "discovery_pool_pair_greedy_random_search",
                "candidate_pool_size": min(int(args.interaction_pool), len(ranking)),
                "selected_k": _group_size(selected_group),
                "n_selection": len(interaction_selection),
                "n_confirmation": len(interaction_confirmation),
                "best_selection_destroy_rate": best_selection_rate,
                "confirmation_destroy_count": selected_eval["destroy_count"],
                "confirmation_destroy_denominator": selected_eval["destroy_denominator"],
                "confirmation_destroy_rate": selected_eval["destroy_rate"],
                "confirmation_destroy_ci_low": selected_eval["destroy_ci_low"],
                "confirmation_destroy_ci_high": selected_eval["destroy_ci_high"],
                "paired_clean_accuracy_drop": selected_eval["paired_clean_accuracy_drop"],
                "paired_clean_accuracy_drop_ci_low": selected_eval["paired_clean_accuracy_drop_ci_low"],
                "paired_clean_accuracy_drop_ci_high": selected_eval["paired_clean_accuracy_drop_ci_high"],
                "control_target_induction_rate": selected_eval["control_target_induction_rate"],
                "ordinary_target_destroy_count": selected_eval["ordinary_target_destroy_count"],
                "ordinary_target_destroy_denominator": selected_eval["ordinary_target_destroy_denominator"],
                "ordinary_target_destroy_rate": selected_eval["ordinary_target_destroy_rate"],
                "ordinary_target_destroy_ci_low": selected_eval["ordinary_target_destroy_ci_low"],
                "ordinary_target_destroy_ci_high": selected_eval["ordinary_target_destroy_ci_high"],
                "trigger_specificity_gap": selected_eval["trigger_specificity_gap"],
                **specificity_meta,
                **random_stats,
                "selected_group": json.dumps(selected_group, sort_keys=True),
                "matched_random_groups": json.dumps(confirm_random_rows, sort_keys=True),
            }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, raw_random_rows, interaction_candidates, interaction_result


def _plot(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for (task, fraction), sub in df.groupby(["task", "fraction"], sort=True):
        if fraction not in (0.1, 1.0):
            continue
        sub = sub.sort_values("k")
        ax.plot(
            sub["k"], 100.0 * sub["trigger_lift_destroy_rate"], marker="o",
            label=f"{task} {100*fraction:.0f}%",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Discovery-ranked cumulative coalition size k")
    ax.set_ylabel("Held-out trigger-lift successes destroyed (%)")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _write_summary(df: pd.DataFrame, interaction: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# Trigger-lift defence summary",
        "",
        "Cumulative coalitions are ordered by the discovery-frozen ranking. Candidate and matched-random groups are evaluated on the same held-out test rows. The poisoned J is also applied to correct ordinary target-positive examples exactly matched by task type; a large trigger-minus-ordinary destruction gap supports trigger-mechanism specificity, while similar rates indicate generic target/task channels. Exact binomial intervals are reported for both destruction rates; paired bootstrap intervals are reported for ordinary control-marker accuracy changes.",
        "",
        "| Task | Checkpoint | Best cumulative k | Trigger-lift removal (95% CI) | Ordinary target removal | Specificity gap | Matched-random mean | Empirical p | Clean accuracy drop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not df.empty:
        for (task, fraction), group in df.groupby(["task", "fraction"], sort=True):
            g = group.copy()
            idx = pd.to_numeric(g["trigger_lift_destroy_rate"], errors="coerce").idxmax()
            row = g.loc[idx]
            lines.append(
                f"| {task} | {100*float(fraction):.0f}% | {int(row['k'])} | "
                f"{100*float(row['trigger_lift_destroy_rate']):.1f}% "
                f"[{100*float(row['trigger_lift_destroy_ci_low']):.1f}, {100*float(row['trigger_lift_destroy_ci_high']):.1f}] | "
                f"{100*float(row['ordinary_target_destroy_rate']):.1f}% | "
                f"{100*float(row['trigger_specificity_gap']):+.1f} pp | "
                f"{100*float(row['matched_random_mean_destroy_rate']):.1f}% | "
                f"{float(row['matched_random_empirical_p']):.3f} | "
                f"{100*float(row['paired_clean_accuracy_drop']):+.1f} pp |"
            )
    if not interaction.empty:
        lines.extend([
            "",
            "## Final-checkpoint interaction-aware confirmation",
            "",
            "The coalition is selected using one held-out subset and evaluated once on a disjoint confirmation subset.",
            "",
            "| Task | Selected k | Confirmation removal (95% CI) | Ordinary target removal | Specificity gap | Matched-random mean | Empirical p | Clean accuracy drop |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in interaction.to_dict("records"):
            lines.append(
                f"| {row['task']} | {int(row['selected_k'])} | "
                f"{100*float(row['confirmation_destroy_rate']):.1f}% "
                f"[{100*float(row['confirmation_destroy_ci_low']):.1f}, {100*float(row['confirmation_destroy_ci_high']):.1f}] | "
                f"{100*float(row['ordinary_target_destroy_rate']):.1f}% | "
                f"{100*float(row['trigger_specificity_gap']):+.1f} pp | "
                f"{100*float(row['matched_random_mean_destroy_rate']):.1f}% | "
                f"{float(row['matched_random_empirical_p']):.3f} | "
                f"{100*float(row['paired_clean_accuracy_drop']):+.1f} pp |"
            )
    (out_dir / "defence_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dirs", required=True, help="Comma-separated poisoning run directories.")
    p.add_argument("--task_modules", default=None, help="Optional comma-separated task modules matching run_dirs.")
    p.add_argument("--condition", default="poisoned")
    p.add_argument("--fractions", default="0.1,0.25,0.5,0.75,1.0")
    p.add_argument("--eval_intervention", default="mean-donor")
    p.add_argument("--intervention", default="mean-donor", choices=["zero", "mean", "mean-donor", "mean-positional", "mean-donor-positional"])
    p.add_argument("--top_ks", default="1,2,4,6,8,16,32,64")
    p.add_argument("--max_pos", type=int, default=0, help="0 uses every held-out trigger-lift success.")
    p.add_argument("--max_neg", type=int, default=0, help="0 uses every held-out non-lift row.")
    p.add_argument("--max_clean", type=int, default=0, help="0 uses every held-out control-marker row for collateral controls.")
    p.add_argument(
        "--max_task_specificity", type=int, default=0,
        help="Maximum ordinary correct target-positive rows matched by task type for the poisoned-J specificity control; 0 uses every available exact match up to n_pos.",
    )
    p.add_argument("--random_groups", type=int, default=20, help="Matched noncandidate groups per cumulative coalition.")
    p.add_argument("--mean_points", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--decode_only", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--ci_level", type=float, default=0.95)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--interaction_fraction", type=float, default=1.0)
    p.add_argument("--interaction_pool", type=int, default=16, help="Discovery-ranked candidate pool for final-checkpoint coalition search.")
    p.add_argument("--interaction_max_k", type=int, default=8)
    p.add_argument("--interaction_selection_fraction", type=float, default=0.40)
    p.add_argument("--interaction_min_examples", type=int, default=20)
    p.add_argument("--interaction_pair_scan", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--interaction_random_draws_per_k", type=int, default=6)
    p.add_argument("--ai_model_cache_dir", default=None)
    p.add_argument("--output_dir", default=str(PROJECT_ROOT / "data" / "poisoning" / "summary" / "mechanism"))
    args = p.parse_args()

    if args.random_groups < 1:
        raise ValueError("--random_groups must be at least 1")
    if not 0.0 < args.interaction_selection_fraction < 1.0:
        raise ValueError("--interaction_selection_fraction must be between 0 and 1")
    args.top_ks = sorted(set(_parse_int_list(args.top_ks)))
    run_dirs = [Path(x).expanduser() for x in _parse_csv_list(args.run_dirs)]
    fractions = _parse_float_list(args.fractions)
    if args.task_modules:
        task_modules = _parse_csv_list(args.task_modules)
        if len(task_modules) != len(run_dirs):
            raise ValueError("--task_modules must have one entry per --run_dirs entry")
    else:
        task_modules = [_infer_task_module(path) for path in run_dirs]

    all_rows: List[Dict[str, Any]] = []
    all_random_rows: List[Dict[str, Any]] = []
    interaction_candidate_frames: List[pd.DataFrame] = []
    interaction_results: List[Dict[str, Any]] = []
    for run_dir, task_module in zip(run_dirs, task_modules):
        selected = _find_rows_for_run(
            run_dir, fractions, args.condition, args.eval_intervention, args.decode_only
        )
        checkpoint_iter = tqdm(
            selected.iterrows(),
            total=len(selected),
            desc=f"{_task_label(run_dir)} defence checkpoints",
            unit="checkpoint",
        )
        for _, row in checkpoint_iter:
            rows, random_rows, interaction_candidates, interaction_result = _run_one(
                run_dir=run_dir, task_module=task_module, row=row, args=args
            )
            all_rows.extend(rows)
            all_random_rows.extend(random_rows)
            if not interaction_candidates.empty:
                interaction_candidates["run_dir"] = str(run_dir)
                interaction_candidates["intervention_phase"] = "output_only" if args.decode_only else "input_output"
                interaction_candidate_frames.append(interaction_candidates)
            if interaction_result is not None:
                interaction_results.append(interaction_result)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(all_rows)
    random_df = pd.DataFrame(all_random_rows)
    interaction_candidates_df = (
        pd.concat(interaction_candidate_frames, ignore_index=True)
        if interaction_candidate_frames else pd.DataFrame()
    )
    interaction_df = pd.DataFrame(interaction_results)

    out_df.to_csv(out_dir / "backdoor_lift_cumulative_topk_ablation.csv", index=False)
    random_df.to_csv(out_dir / "matched_random_group_results.csv", index=False)
    interaction_candidates_df.to_csv(out_dir / "interaction_search_candidates.csv", index=False)
    interaction_df.to_csv(out_dir / "interaction_aware_final_confirmation.csv", index=False)
    _plot(out_df, out_dir / "backdoor_lift_cumulative_topk_ablation.pdf")
    _write_summary(out_df, interaction_df, out_dir)

    config = {
        "definition": "trigger_lift",
        "evaluation_rows": "feature-report is_test == true",
        "cumulative_ranking": "frozen_candidate_ranking.csv / discovery_rank_global",
        "matched_random": "noncandidate units matched exactly by native MLP layer or attention-head locus and cardinality",
        "random_groups": int(args.random_groups),
        "task_circuit_specificity_control": {
            "intervention_set": "the same poisoned checkpoint J used for trigger-lift suppression",
            "control_behavior": "correct ordinary target-positive response on prompt_control",
            "matching": "task-owned exact strata declared by the registered task module",
            "unmatched_rows_are_not_substituted": True,
            "max_rows": int(args.max_task_specificity),
            "primary_contrast": "trigger_lift_destroy_rate - ordinary_target_destroy_rate",
        },
        "interaction_search": {
            "checkpoint_fraction": float(args.interaction_fraction),
            "candidate_pool": int(args.interaction_pool),
            "max_k": int(args.interaction_max_k),
            "selection_fraction": float(args.interaction_selection_fraction),
            "pair_scan": bool(args.interaction_pair_scan),
            "random_draws_per_k": int(args.interaction_random_draws_per_k),
            "confirmation_is_disjoint": True,
        },
    }
    (out_dir / "defence_configuration.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"Wrote downstream defence outputs under {out_dir}")


if __name__ == "__main__":
    main()
