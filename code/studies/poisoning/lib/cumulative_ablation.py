#!/usr/bin/env python3
"""Confirmatory trigger-lift defence and coalition analysis.

For each poisoned checkpoint this program evaluates cumulative coalitions whose
ordering was frozen at a strictly earlier checkpoint.  The current checkpoint is
used only for evaluation; it never contributes channel identities or ranking.
The same held-out test rows are used for candidate and structurally matched random
coalitions.  This makes the defence prospective across checkpoints rather than a
post-hoc intervention selected from the checkpoint being defended.

The behavioural endpoint is trigger lift:

    B(x) != T and B(x+t) == T

where ``T`` is the configured backdoor target.  Causal interventions are
applied to the trigger-marker prompt. Ordinary task accuracy and control-marker target induction are
measured on ``x`` as collateral-damage controls.
"""

from __future__ import annotations

from studies.poisoning.lib.units import unit_key as _unit_key
import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from studies.poisoning.lib.run_paths import defence_cache_dir, resolve_manifest_checkpoint_dir, trajectories_dir, phase_dirname
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm.auto import tqdm

from core.feature_extraction_runner import resolve_task_spec
from core.group_intervention import load_stage5_locus_population
from core.modeling_and_ablation import LMWrapper, get_device, precompute_mean_activations
from studies.poisoning.lib.model_loading import poisoning_lm_wrapper_kwargs
from core.neuron_intervention import ablate_neurons
from core.project_paths import PROJECT_ROOT
from studies.poisoning.tasks.registry import get_task_definition, infer_task_from_module, infer_task_from_run



DEFENCE_CACHE_SCHEMA_VERSION = 1

def _file_signature(path: Path) -> dict[str, Any]:
    path = Path(path)
    st = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path.resolve()), "size": int(st.st_size), "sha256": h.hexdigest()}


def _checkpoint_signature(checkpoint_dir: str) -> dict[str, Any]:
    root = Path(checkpoint_dir)
    files = []
    for name in ("adapter_model.safetensors", "adapter_model.bin", "adapter_config.json", "config.json"):
        path = root / name
        if path.is_file():
            st = path.stat()
            files.append({"name": name, "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
    return {"path": str(root.resolve()), "files": files}

def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "run"

def _defence_cache_identity(*, run_dir: Path, task_name: str, fraction: float, checkpoint_dir: str,
                            source_fraction: float, source_stats_dir: Path, scores_path: Path,
                            manifest_path: Path, args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    payload = {
        "schema_version": DEFENCE_CACHE_SCHEMA_VERSION,
        "run_dir": str(run_dir.resolve()),
        "task": task_name,
        "fraction": float(fraction),
        "checkpoint": _checkpoint_signature(checkpoint_dir),
        "ranking_source_fraction": float(source_fraction),
        "ranking": _file_signature(source_stats_dir / "frozen_candidate_ranking.csv"),
        "scores": _file_signature(scores_path),
        "population_manifest": _file_signature(manifest_path),
        "intervention": args.intervention,
        "eval_intervention": args.eval_intervention,
        "decode_only": bool(args.decode_only),
        "top_ks": [int(x) for x in args.top_ks],
        "max_pos": int(args.max_pos),
        "max_neg": int(args.max_neg),
        "max_clean": int(args.max_clean),
        "max_task_specificity": int(args.max_task_specificity),
        "random_groups": int(args.random_groups),
        "mean_points": int(args.mean_points),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "ci_level": float(args.ci_level),
        "bootstrap": int(args.bootstrap),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload

def _cache_files(args: argparse.Namespace, run_dir: Path, task_name: str, fraction: float) -> tuple[Path, Path, Path]:
    phase = "output_only" if args.decode_only else "input_output"
    d = defence_cache_dir(
        args.cache_dir,
        task=task_name,
        run_dir=run_dir,
        phase=phase,
        fraction=fraction,
    )
    return d / "candidate_rows.csv", d / "matched_random_rows.csv", d / "cache_manifest.json"

def _load_defence_cache(candidate_path: Path, random_path: Path, manifest_path: Path, fingerprint: str):
    if not manifest_path.is_file() or not candidate_path.is_file() or not random_path.is_file():
        return [], []
    try:
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != fingerprint:
            return [], []
        cdf = pd.read_csv(candidate_path)
        rdf = pd.read_csv(random_path)
        return cdf.to_dict("records"), rdf.to_dict("records")
    except Exception as exc:
        print(f"[defence-cache] ignoring unreadable cache {manifest_path.parent}: {exc}", flush=True)
        return [], []

def _save_defence_cache(candidate_path: Path, random_path: Path, manifest_path: Path, *, fingerprint: str,
                        identity: dict[str, Any], rows: Sequence[dict[str, Any]], random_rows: Sequence[dict[str, Any]]) -> None:
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(candidate_path, index=False)
    pd.DataFrame(list(random_rows)).to_csv(random_path, index=False)
    manifest_path.write_text(json.dumps({"fingerprint": fingerprint, "identity": identity}, indent=2), encoding="utf-8")

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
    """Return evaluation checkpoints annotated with a strictly prior ranking source."""
    phase = "output_only" if decode_only else "input_output"
    summary_csv = trajectories_dir(run_dir) / phase_dirname(phase) / "backdoor_lift_overtopping_trajectory.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing trigger-lift summary: {summary_csv}")
    df = pd.read_csv(summary_csv)
    if "lift_overtopping_status" not in df.columns:
        raise ValueError(f"Summary is missing lift_overtopping_status: {summary_csv}")
    eligible = df[
        (df["condition"].astype(str) == str(condition))
        & (df["lift_overtopping_status"].astype(str) == "ok")
    ].copy()
    if "lift_eval_intervention" in eligible.columns:
        eligible = eligible[eligible["lift_eval_intervention"].astype(str).fillna("") == str(eval_intervention)]
    eligible["_fraction_numeric"] = pd.to_numeric(eligible["fraction"], errors="coerce")
    eligible = eligible[np.isfinite(eligible["_fraction_numeric"])].sort_values(
        ["_fraction_numeric", "global_step"], kind="mergesort"
    )

    keep = []
    for frac in fractions:
        close = eligible[np.isclose(eligible["_fraction_numeric"], float(frac))]
        if not len(close):
            continue
        current = close.iloc[0].copy()
        current_fraction = float(current["_fraction_numeric"])
        prior = eligible[eligible["_fraction_numeric"] < current_fraction - 1e-12]
        if prior.empty:
            current["defence_ranking_source_status"] = "no_prior_completed_ranking"
            current["defence_ranking_source_fraction"] = math.nan
            current["defence_ranking_source_stats_dir"] = ""
            current["defence_ranking_source_base_dir"] = ""
        else:
            source = prior.iloc[-1]
            current["defence_ranking_source_status"] = "ok"
            current["defence_ranking_source_fraction"] = float(source["_fraction_numeric"])
            current["defence_ranking_source_stats_dir"] = str(source["lift_overtopping_stats_dir"])
            current["defence_ranking_source_base_dir"] = str(source["lift_overtopping_base_dir"])
            current["defence_ranking_source_global_step"] = int(source["global_step"])
        keep.append(current)
    if not keep:
        print(
            f"[skip] no completed circuit-defined {condition} checkpoints for fractions={fractions} "
            f"in {summary_csv}; downstream suppression has nothing to evaluate for this run.",
            flush=True,
        )
        return pd.DataFrame(columns=list(df.columns) + [
            "defence_ranking_source_status", "defence_ranking_source_fraction",
            "defence_ranking_source_stats_dir", "defence_ranking_source_base_dir",
            "defence_ranking_source_global_step",
        ])
    return pd.DataFrame(keep).drop(columns=["_fraction_numeric"], errors="ignore")


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


def _run_one(
    *,
    run_dir: Path,
    task_module: str,
    row: pd.Series,
    args: argparse.Namespace,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    task = resolve_task_spec(task_module)
    task_definition = infer_task_from_module(task_module)
    task_name = task_definition.name
    fraction = float(row["fraction"])
    condition = str(row["condition"])
    checkpoint_dir = str(resolve_manifest_checkpoint_dir(run_dir, row, must_exist=True))
    stats_dir = Path(str(row["lift_overtopping_stats_dir"]))
    base_dir = Path(str(row["lift_overtopping_base_dir"]))
    source_status = str(row.get("defence_ranking_source_status", ""))
    if source_status != "ok":
        print(
            f"[skip-prospective-defence] {task_name} {condition} frac={fraction:g}: "
            "no strictly earlier checkpoint has a completed frozen ranking.",
            flush=True,
        )
        return [], []
    source_fraction = float(row["defence_ranking_source_fraction"])
    source_stats_dir = Path(str(row["defence_ranking_source_stats_dir"]))
    source_base_dir = Path(str(row["defence_ranking_source_base_dir"]))
    if not source_fraction < fraction - 1e-12:
        raise RuntimeError(
            f"Prospective defence leakage guard failed: source fraction {source_fraction:g} "
            f"is not strictly earlier than evaluation fraction {fraction:g}."
        )
    scores_path = base_dir / "feature_report" / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing scores for {task_name} fraction={fraction}: {scores_path}")

    manifest_path = _find_population_manifest(source_base_dir, source_stats_dir)
    cache_fingerprint, cache_identity = _defence_cache_identity(
        run_dir=run_dir, task_name=task_name, fraction=fraction, checkpoint_dir=checkpoint_dir,
        source_fraction=source_fraction, source_stats_dir=source_stats_dir, scores_path=scores_path,
        manifest_path=manifest_path, args=args,
    )
    cache_candidate_path, cache_random_path, cache_manifest_path = _cache_files(
        args, run_dir, task_name, fraction
    )
    cached_rows, cached_random_rows = _load_defence_cache(
        cache_candidate_path, cache_random_path, cache_manifest_path, cache_fingerprint
    )
    if cached_rows:
        cached_ks = {int(r["k"]) for r in cached_rows}
        expected_ks = {min(int(k), len(_load_frozen_ranking(source_stats_dir))) for k in args.top_ks if int(k) > 0}
        expected_ks.discard(0)
        random_counts = Counter(int(r["k"]) for r in cached_random_rows)
        complete = expected_ks.issubset(cached_ks) and all(random_counts[k] >= int(args.random_groups) for k in expected_ks)
        if complete:
            print(
                f"[defence-cache] hit {task_name} frac={fraction:g}: reused {len(cached_rows)} coalition rows and "
                f"{len(cached_random_rows)} matched-random rows from {cache_manifest_path.parent}",
                flush=True,
            )
            return cached_rows, cached_random_rows
        print(
            f"[defence-cache] partial hit {task_name} frac={fraction:g}: resuming missing coalition/random evaluations",
            flush=True,
        )

    scores_df = pd.read_csv(scores_path)
    eval_df = _heldout_test_frame(scores_df, scores_path)
    prompt_col = getattr(task, "DEFAULT_INPUT", "prompt")
    target_col = getattr(task, "DEFAULT_TARGETS", ("is_trigger_lift_success",))[0]
    if prompt_col not in eval_df.columns or target_col not in eval_df.columns:
        raise ValueError(f"{scores_path} lacks required columns {prompt_col!r}, {target_col!r}")
    if "prompt_control" not in eval_df.columns:
        raise ValueError(f"{scores_path} lacks prompt_control required for collateral controls")

    target_mask = eval_df[target_col].map(_truthy).to_numpy(dtype=bool)

    # The current checkpoint is evaluation-only.  No rows from it are used to
    # choose channel identities, ranking, or coalition structure.
    exploratory_mask = np.ones(len(eval_df), dtype=bool)
    cumulative_target_mask = target_mask.copy()

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
        return [], []

    ordinary_mask = task_definition.ordinary_target_positive_mask(eval_df) & exploratory_mask
    ordinary_target_examples, specificity_meta = task_definition.sample_task_specificity_examples(
        eval_df,
        candidate_mask=ordinary_mask,
        reference_examples=pos_examples,
        max_n=args.max_task_specificity,
        seed=args.seed + _stable_int(task_name, fraction, "ordinary_target_specificity"),
    )

    ranking = _load_frozen_ranking(source_stats_dir)
    if ranking.empty:
        print(
            f"[skip-prospective-defence] {task_name} {condition} frac={fraction:g}: "
            f"prior checkpoint frac={source_fraction:g} has an empty frozen candidate set.",
            flush=True,
        )
        return [], []

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

    all_needed: Dict[str, set[int]] = {}
    for group in list(top_groups.values()) + [g for groups in random_groups.values() for g in groups]:
        for layer, ids in group.items():
            all_needed.setdefault(layer, set()).update(int(x) for x in ids)
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

    rows: List[Dict[str, Any]] = list(cached_rows)
    raw_random_rows: List[Dict[str, Any]] = list(cached_random_rows)
    cached_candidate_ks = {int(r["k"]) for r in rows}
    cached_random_counts = Counter(int(r["k"]) for r in raw_random_rows)
    pending_top_groups = {
        k: group for k, group in top_groups.items()
        if k not in cached_candidate_ks or cached_random_counts[k] < int(args.random_groups)
    }
    for k, group in tqdm(
        sorted(pending_top_groups.items()),
        desc=f"{task_name} frac={fraction:g} cumulative top-k",
        unit="group",
    ):
        existing_candidate = next((r for r in rows if int(r["k"]) == int(k)), None)
        if existing_candidate is None:
            candidate = _evaluate_group(
                model=model, task=task, group=group, pos_examples=pos_examples, neg_examples=neg_examples,
                clean_examples=clean_examples, ordinary_target_examples=ordinary_target_examples,
                prompt_col=prompt_col, clean_score_fn=clean_score_fn, target_score_fn=target_score_fn,
                args=args, mean_activations=mean_activations, baseline_pos_vec=baseline_pos_vec,
                clean_baseline_vec=clean_baseline_vec, ordinary_target_baseline_vec=ordinary_target_baseline_vec,
            )
        else:
            candidate = {
                "destroy_rate": float(existing_candidate["trigger_lift_destroy_rate"]),
            }
        existing_random = [r for r in raw_random_rows if int(r["k"]) == int(k)]
        random_rates: List[float] = [float(r["trigger_lift_destroy_rate"]) for r in existing_random]
        start_draw = len(existing_random)
        for draw, random_group in enumerate(
            tqdm(
                random_groups[k][start_draw:],
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
            draw += start_draw
            random_rates.append(float(rnd["destroy_rate"]))
            raw_random_rows.append({
                "analysis": "cumulative",
                "task": task_name,
                "run_dir": str(run_dir),
                "condition": condition,
                "fraction": fraction,
                "checkpoint_dir": checkpoint_dir,
                "ranking_source_fraction": source_fraction,
                "ranking_source_stats_dir": str(source_stats_dir),
                "ranking_source_policy": "latest_strictly_prior_completed_checkpoint",
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
        if existing_candidate is not None:
            existing_candidate.update(random_stats)
        if existing_candidate is None:
            rows.append({
            "task": task_name,
            "run_dir": str(run_dir),
            "condition": condition,
            "fraction": fraction,
            "global_step": int(row.get("global_step", -1)),
            "checkpoint_dir": checkpoint_dir,
            "stats_dir": str(stats_dir),
            "population_manifest": str(manifest_path),
            "ranking_source": "prior_checkpoint_frozen_discovery_ranking",
            "ranking_source_fraction": source_fraction,
            "ranking_source_stats_dir": str(source_stats_dir),
            "ranking_source_policy": "latest_strictly_prior_completed_checkpoint",
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
        _save_defence_cache(
            cache_candidate_path, cache_random_path, cache_manifest_path, fingerprint=cache_fingerprint,
            identity=cache_identity, rows=rows, random_rows=raw_random_rows,
        )
        print(
            f"[topk] {task_name} frac={fraction:g} k={k}: "
            f"destroy={candidate['destroy_rate']:.3f} "
            f"random_mean={random_stats['matched_random_mean_destroy_rate']:.3f}",
            flush=True,
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, raw_random_rows


def _plot(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    plotted = False
    for (task, fraction), sub in df.groupby(["task", "fraction"], sort=True):
        sub = sub.copy()
        sub["k"] = pd.to_numeric(sub["k"], errors="coerce")
        sub["trigger_lift_destroy_rate"] = pd.to_numeric(
            sub["trigger_lift_destroy_rate"], errors="coerce"
        )
        sub = sub.loc[
            sub["k"].gt(0) & sub["trigger_lift_destroy_rate"].notna()
        ].sort_values("k")
        if sub.empty:
            continue

        source = pd.to_numeric(
            sub.get("ranking_source_fraction"), errors="coerce"
        ).dropna().unique()
        if len(source) == 1:
            label = f"{task} {100*fraction:.0f}% (rank {100*source[0]:.0f}%)"
        else:
            label = f"{task} {100*fraction:.0f}%"

        ax.plot(
            sub["k"],
            100.0 * sub["trigger_lift_destroy_rate"],
            marker="o",
            label=label,
        )
        plotted = True

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Prior-checkpoint-ranked cumulative coalition size k")
    ax.set_ylabel("Held-out trigger-lift successes destroyed (%)")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(frameon=False)
    else:
        ax.text(
            0.5,
            0.5,
            "No finite defence results to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# Trigger-lift defence summary",
        "",
        "For each checkpoint, cumulative coalitions are ordered by the latest frozen ranking from a strictly earlier completed checkpoint. Candidate and matched-random groups are evaluated on the current checkpoint held-out test rows. The poisoned J is also applied to correct ordinary target-positive examples exactly matched by task type; a large trigger-minus-ordinary destruction gap supports trigger-mechanism specificity, while similar rates indicate generic target/task channels. Exact binomial intervals are reported for both destruction rates; paired bootstrap intervals are reported for ordinary control-marker accuracy changes.",
        "",
        "| Task | Checkpoint | Ranking source | Best cumulative k | Trigger-lift removal (95% CI) | Ordinary target removal | Specificity gap | Matched-random mean | Empirical p | Clean accuracy drop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not df.empty:
        for (task, fraction), group in df.groupby(["task", "fraction"], sort=True):
            g = group.copy()
            idx = pd.to_numeric(g["trigger_lift_destroy_rate"], errors="coerce").idxmax()
            row = g.loc[idx]
            lines.append(
                f"| {task} | {100*float(fraction):.0f}% | {100*float(row['ranking_source_fraction']):.0f}% | {int(row['k'])} | "
                f"{100*float(row['trigger_lift_destroy_rate']):.1f}% "
                f"[{100*float(row['trigger_lift_destroy_ci_low']):.1f}, {100*float(row['trigger_lift_destroy_ci_high']):.1f}] | "
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
    p.add_argument("--ai_model_cache_dir", default=None)
    p.add_argument("--cache_dir", default=str(PROJECT_ROOT / "cache" / "poisoning"),
                   help="Base poisoning cache root. Defence entries are written under <task>/<run>/defence/<phase>/fraction_<fraction>.")
    p.add_argument("--output_dir", default=str(PROJECT_ROOT / "data" / "poisoning" / "final" / "manual" / "07_defence_evaluation" / "inference_time"))
    args = p.parse_args()

    if args.random_groups < 1:
        raise ValueError("--random_groups must be at least 1")
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
            rows, random_rows = _run_one(
                run_dir=run_dir, task_module=task_module, row=row, args=args
            )
            all_rows.extend(rows)
            all_random_rows.extend(random_rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(all_rows)
    random_df = pd.DataFrame(all_random_rows)
    out_df.to_csv(out_dir / "backdoor_lift_cumulative_topk_ablation.csv", index=False)
    random_df.to_csv(out_dir / "matched_random_group_results.csv", index=False)
    _plot(out_df, out_dir / "backdoor_lift_cumulative_topk_ablation.pdf")
    _write_summary(out_df, out_dir)

    config = {
        "definition": "trigger_lift",
        "evaluation_rows": "feature-report is_test == true",
        "cumulative_ranking": "latest strictly prior checkpoint frozen_candidate_ranking.csv / discovery_rank_global",
        "prospective_checkpoint_policy": "evaluation checkpoint never supplies channel identities, ranking, or adaptive coalition selection",
        "no_current_checkpoint_ranking_fallback": True,
        "adaptive_current_checkpoint_coalition_search": False,
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
    }
    (out_dir / "defence_configuration.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"Wrote downstream defence outputs under {out_dir}")


if __name__ == "__main__":
    main()
