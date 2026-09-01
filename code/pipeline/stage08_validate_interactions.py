#!/usr/bin/env python3
"""Validate frozen candidates with genuine simultaneous-set interventions.

Two complementary questions are evaluated on one fixed evaluation-example set:

1. ``E(J)``: the unconditional simultaneous effect of suppressing the complete
   frozen candidate set ``J``, compared with structurally matched noncandidate
   sets ``K_b``.
2. Conditional marginal contribution (CMC), unless ``--skip_cmc`` is set:
   for each draw ``b`` a noncandidate background ``S_b`` is sampled and held
   fixed while candidate and null are compared in the same perturbed context::

       M_b(J)   = E(S_b ∪ J)   - E(S_b)
       M_b(K_b) = E(S_b ∪ K_b) - E(S_b)
       D_b      = M_b(J)       - M_b(K_b)

``K_b`` and ``S_b`` are matched to the candidate topology by transformer
layer, computational locus, channel type and per-stratum cardinality.  Phase
and replacement baseline are fixed globally.  Candidate/null/background sets
are always evaluated by genuine simultaneous interventions; singleton unions
are never substituted.

"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.caching_and_prompting import set_deterministic
from core.feature_extraction_runner import resolve_task_spec
from core.group_intervention import (
    GroupSpec,
    dedupe_units,
    evaluate_groups,
    evaluation_frame,
    evaluation_row_records,
    group_units_by_matching_stratum,
    load_candidate_units,
    load_complete_group_batch_cache,
    load_dataset_info,
    load_stage5_locus_population,
    resolve_dataset_path,
    simultaneous_effect,
    unit_metadata,
)
from core.interaction_statistics import matched_null_summary, paired_conditional_summary
from core.high_n_singleton_eval import (
    UnitSpec,
    build_mean_prompt_pool,
    load_scores_for_baseline,
    precompute_replacements_for_units,
)
from core.modeling_and_ablation import LMWrapper, get_device


LOG_PREFIX = "[conditional-validation]"
SCHEMA = "conditional-marginal-validation-v1"
GROUP_CACHE_SCHEMA = "simultaneous-group-eval-v2"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_data_dir", required=True)
    parser.add_argument("--candidate_flip_stats_path", required=True)
    parser.add_argument("--singleton_scores_path", default=None)
    parser.add_argument("--frozen_ranking_path", default=None)
    parser.add_argument("--layer_population_manifest", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--task_module", default="core.tasks.arithmetic_task")
    parser.add_argument("--ai_model", default=None)
    parser.add_argument("--ai_model_cache_dir", default=None)
    parser.add_argument(
        "--intervention",
        choices=["zero", "mean", "mean-donor", "mean-positional", "mean-donor-positional"],
        default="mean-donor",
    )
    parser.add_argument("--decode_only", action="store_true")
    parser.add_argument(
        "--evaluation_split", choices=["test", "train", "all"], default="test",
        help="Rows used for every candidate, background and null intervention. Default: test.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--points_to_use_for_mean_ablation", type=int, default=2048)
    parser.add_argument(
        "--background_multipliers", default="1",
        help=(
            "Comma-separated nonnegative integer background loads. For multiplier q, "
            "S_b contains q times the candidate count in every exact layer/locus/type stratum. "
            "Default: 1."
        ),
    )
    parser.add_argument("--null_draws", type=int, default=100)
    parser.add_argument(
        "--skip_preemption",
        action="store_true",
        help="Skip P0.3 dominant-secondary pair interventions and example-level preemption outputs.",
    )
    parser.add_argument(
        "--preemption_min_singleton_rate",
        type=float,
        default=0.05,
        help="Minimum directional singleton effect for dominant/secondary units in P0.3. Default: 0.05.",
    )
    parser.add_argument(
        "--preemption_max_secondaries",
        type=int,
        default=8,
        help="Maximum secondary candidates evaluated per direction for P0.3. Default: 8; 0 means all eligible.",
    )
    parser.add_argument(
        "--threshold_diagnostics_dir",
        default=None,
        help=(
            "Optional Step-7b spiking-diagnostics directory. When present, P0.3 conditions "
            "secondary marginal effects on an internally held-out endogenous threshold event T_j*."
        ),
    )
    parser.add_argument("--preemption_threshold_holdout_fraction", type=float, default=0.25)
    parser.add_argument("--preemption_threshold_min_class", type=int, default=8)
    parser.add_argument(
        "--skip_cmc",
        action="store_true",
        help=(
            "Compute simultaneous E(J) and its matched-null distribution only. "
            "Skip conditional backgrounds and all CMC interventions/outputs."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for replacement-reference sampling and matched sets. Default: 42.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _parse_background_multipliers(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0:
            raise ValueError("--background_multipliers values must be nonnegative integers")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("--background_multipliers must contain at least one integer")
    return tuple(values)


def _load_ranking(path: Path, candidates: list[UnitSpec]) -> pd.DataFrame:
    """Validate that the candidate set is the same frozen discovery set.

    Conditional validation does not rank candidates, but retaining this check
    prevents a stale/mismatched candidate file from being evaluated silently.
    """
    frame = pd.read_csv(path)
    if "unit_key" not in frame.columns:
        if not {"layer_label", "neuron_id"}.issubset(frame.columns):
            raise ValueError("Frozen ranking must contain unit_key or layer_label and neuron_id")
        frame["unit_key"] = [
            f"{layer}:{int(neuron)}"
            for layer, neuron in zip(frame["layer_label"], frame["neuron_id"])
        ]
    candidate_keys = {unit.unit_key for unit in candidates}
    frame = frame.loc[frame["unit_key"].astype(str).isin(candidate_keys)].copy()
    observed = set(frame["unit_key"].astype(str))
    if observed != candidate_keys:
        missing = sorted(candidate_keys - observed)
        raise ValueError("Frozen ranking is incomplete: " + ", ".join(missing[:10]))
    if "discovery_rank_global" in frame.columns:
        frame["discovery_rank_global"] = pd.to_numeric(
            frame["discovery_rank_global"], errors="raise"
        ).astype(int)
        frame = frame.sort_values("discovery_rank_global", kind="mergesort")
    return frame.reset_index(drop=True)


def _group(units: Iterable[UnitSpec], label: str) -> GroupSpec:
    return GroupSpec(tuple(dedupe_units(units)), label=label)


def _effect_for_group(group: GroupSpec, *, baseline: np.ndarray, post_by_key: dict[str, np.ndarray]) -> dict:
    if group.size == 0:
        n = int(len(baseline))
        return {
            "count": 0,
            "denominator": n,
            "effect": 0.0 if n else math.nan,
            "status": "empty_intervention" if n else "undefined_zero_denominator",
            "effect_0to1": 0.0,
            "effect_1to0": 0.0,
        }
    return simultaneous_effect(baseline, post_by_key[group.key])


def _write_plot(path: Path, values: list[float], reference: float, title: str, xlabel: str) -> None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(finite) == 0 or not np.isfinite(reference):
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    bins = min(30, max(8, int(np.sqrt(len(finite)))))
    ax.hist(finite, bins=bins)
    ax.axvline(reference, linestyle="--", linewidth=1.5, label="candidate")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Matched draws")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _write_paired_plot(path: Path, candidate_values: list[float], null_values: list[float], title: str) -> None:
    cand = np.asarray(candidate_values, dtype=float)
    null = np.asarray(null_values, dtype=float)
    mask = np.isfinite(cand) & np.isfinite(null)
    if not mask.any():
        return
    delta = cand[mask] - null[mask]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    bins = min(30, max(8, int(np.sqrt(len(delta)))))
    ax.hist(delta, bins=bins)
    ax.axvline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"$M_b(J)-M_b(K_b)$")
    ax.set_ylabel("Paired backgrounds")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _latex_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Metric & Background & Candidate & Null median & $\Delta$ & $P$ & $p_{\mathrm{MC}}$ & $B$ \\",
        r"\midrule",
    ]
    for row in summary.to_dict("records"):
        def fmt(value):
            try:
                number = float(value)
            except Exception:
                return "--"
            return "--" if not np.isfinite(number) else f"{number:.3f}"
        background = "--" if pd.isna(row.get("background_multiplier")) else f"{int(row['background_multiplier'])}x"
        lines.append(
            f"{row['metric']} & {background} & {fmt(row.get('candidate'))} & "
            f"{fmt(row.get('median_null'))} & {fmt(row.get('Delta'))} & "
            f"{fmt(row.get('P'))} & {fmt(row.get('p_MC'))} & "
            f"{int(row.get('null_draws_requested', 0))} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _draw_matched_null_group(
    *,
    draw: int,
    candidate_strata: dict[tuple[int, str, str], list[UnitSpec]],
    population_strata: dict[tuple[int, str, str], list[UnitSpec]],
    candidate_keys: set[str],
    seed: int,
) -> GroupSpec:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 101, int(draw)]))
    units: list[UnitSpec] = []
    for stratum in sorted(candidate_strata):
        q = len(candidate_strata[stratum])
        pool = [unit for unit in population_strata[stratum] if unit.unit_key not in candidate_keys]
        if len(pool) < q:
            raise ValueError(
                f"Matched null impossible for stratum {stratum}: need {q} noncandidate units, found {len(pool)}"
            )
        chosen = rng.choice(len(pool), size=q, replace=False)
        units.extend(pool[int(index)] for index in np.atleast_1d(chosen))
    return _group(units, f"random_J_{draw}")


def _draw_background_group(
    *,
    draw: int,
    multiplier: int,
    candidate_strata: dict[tuple[int, str, str], list[UnitSpec]],
    population_strata: dict[tuple[int, str, str], list[UnitSpec]],
    candidate_keys: set[str],
    null_group: GroupSpec,
    seed: int,
) -> GroupSpec:
    if multiplier == 0:
        return _group([], f"background_{multiplier}x_{draw}")
    excluded = candidate_keys | {unit.unit_key for unit in null_group.units}
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 31337, int(multiplier), int(draw)])
    )
    units: list[UnitSpec] = []
    for stratum in sorted(candidate_strata):
        q = int(multiplier) * len(candidate_strata[stratum])
        pool = [unit for unit in population_strata[stratum] if unit.unit_key not in excluded]
        if len(pool) < q:
            raise ValueError(
                "Conditional background impossible for stratum "
                f"{stratum}: multiplier={multiplier} requires {q} additional noncandidate units "
                f"after excluding J and K_b, found {len(pool)}. "
                "Use a smaller --background_multipliers value; matching is never relaxed silently."
            )
        chosen = rng.choice(len(pool), size=q, replace=False)
        units.extend(pool[int(index)] for index in np.atleast_1d(chosen))
    return _group(units, f"background_{multiplier}x_{draw}")


def _union_group(*groups: GroupSpec, label: str) -> GroupSpec:
    return _group((unit for group in groups for unit in group.units), label)


def _candidate_stat_frame(path: Path, candidates: list[UnitSpec]) -> pd.DataFrame:
    """Load candidate singleton statistics with a stable unit_key column."""
    frame = pd.read_csv(path)
    if "unit_key" not in frame.columns:
        layer_col = "layer_label" if "layer_label" in frame.columns else "layer_key"
        if layer_col not in frame.columns or "neuron_id" not in frame.columns:
            return pd.DataFrame()
        frame["unit_key"] = [f"{layer}:{int(neuron)}" for layer, neuron in zip(frame[layer_col], frame["neuron_id"])]
    keys = {u.unit_key for u in candidates}
    return frame.loc[frame["unit_key"].astype(str).isin(keys)].copy()


def _preemption_pair_plan(
    *,
    candidates: list[UnitSpec],
    candidate_stats: pd.DataFrame,
    min_rate: float,
    max_secondaries: int,
) -> tuple[list[dict], dict[str, GroupSpec]]:
    """Choose a dominant unit and nontrivial secondaries independently by direction.

    P0.3 is directional.  For each source->target direction we freeze the
    strongest singleton as j* and pair it with the remaining candidates whose
    held-out directional singleton rate clears the prespecified threshold.
    """
    unit_by_key = {u.unit_key: u for u in candidates}
    plans: list[dict] = []
    groups: dict[str, GroupSpec] = {}
    for direction, rate_col in (("c2i", "c2i_rate"), ("i2c", "i2c_rate")):
        if rate_col not in candidate_stats.columns:
            continue
        work = candidate_stats[["unit_key", rate_col]].copy()
        work[rate_col] = pd.to_numeric(work[rate_col], errors="coerce")
        work = work.loc[work["unit_key"].astype(str).isin(unit_by_key)]
        work = work.loc[np.isfinite(work[rate_col]) & (work[rate_col] >= float(min_rate))]
        work = work.sort_values([rate_col, "unit_key"], ascending=[False, True], kind="mergesort")
        if len(work) < 2:
            continue
        dominant_key = str(work.iloc[0]["unit_key"])
        secondary_rows = work.iloc[1:]
        if int(max_secondaries) > 0:
            secondary_rows = secondary_rows.head(int(max_secondaries))
        for rank, row in enumerate(secondary_rows.itertuples(index=False), start=1):
            secondary_key = str(row.unit_key)
            pair = _group([unit_by_key[dominant_key], unit_by_key[secondary_key]], f"preemption_{direction}_{rank}")
            groups[pair.key] = pair
            plans.append({
                "direction": direction,
                "dominant_unit": dominant_key,
                "secondary_unit": secondary_key,
                "dominant_singleton_rate": float(work.iloc[0][rate_col]),
                "secondary_singleton_rate": float(getattr(row, rate_col)),
                "secondary_rank": int(rank),
                "pair_key": pair.key,
            })
    return plans, groups


def _directional_flip_column(unit: UnitSpec, direction: str) -> str:
    prefix = "flip_c2i" if direction == "c2i" else "flip_i2c"
    return f"{prefix}_{unit.layer_key}_{int(unit.neuron_id)}"


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else math.nan


def _mcc_binary(y: np.ndarray, pred: np.ndarray) -> float:
    y=np.asarray(y,dtype=bool); pred=np.asarray(pred,dtype=bool)
    tp=float(np.sum(y & pred)); fp=float(np.sum(~y & pred)); tn=float(np.sum(~y & ~pred)); fn=float(np.sum(y & ~pred))
    denom=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return ((tp*tn)-(fp*fn))/denom if denom>0 else math.nan


def _fit_preemption_threshold(x: np.ndarray, y: np.ndarray, min_class: int) -> dict | None:
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=bool)
    mask=np.isfinite(x); x=x[mask]; y=y[mask]
    if min(int(y.sum()),int((~y).sum())) < int(min_class): return None
    vals=np.unique(np.sort(x))
    if len(vals)<2: return None
    if len(vals)>256: vals=np.unique(np.quantile(vals,np.linspace(0,1,257)))
    thresholds=np.unique(np.concatenate([vals,(vals[:-1]+vals[1:])/2]))
    best=None
    for direction in (">=","<="):
        for threshold in thresholds:
            pred=x>=threshold if direction==">=" else x<=threshold
            mcc=_mcc_binary(y,pred)
            if not np.isfinite(mcc): continue
            candidate=(abs(float(mcc)),float(threshold),direction,float(mcc))
            if best is None or candidate[0]>best[0]: best=candidate
    if best is None: return None
    return {"abs_mcc":best[0],"threshold":best[1],"direction":best[2],"mcc":best[3]}


def _threshold_event_indicator(
    *, threshold_dir: Path | None, dominant_unit: str, direction: str, seed: int,
    holdout_fraction: float, min_class: int,
) -> tuple[dict[object,bool], dict]:
    """Fit T_j* on one internal train fold and return predictions only on its held-out fold."""
    if threshold_dir is None:
        return {}, {"status":"threshold_diagnostics_not_provided"}
    baseline_name="positive_baseline" if direction=="c2i" else "negative_baseline"
    base=Path(threshold_dir)/baseline_name
    raw_path=base/"threshold_activation_flip_rows.csv.gz"
    scores_path=base/"high_n_scores_with_flips.csv"
    tests_path=base/"threshold_unit_tests.csv"
    if not (raw_path.is_file() and scores_path.is_file() and tests_path.is_file()):
        return {}, {"status":"threshold_diagnostics_missing_files","baseline":baseline_name}
    try:
        raw=pd.read_csv(raw_path); high=pd.read_csv(scores_path); tests=pd.read_csv(tests_path)
    except Exception as exc:
        return {}, {"status":"threshold_diagnostics_read_error","error":str(exc)}
    target="flip_c2i" if direction=="c2i" else "flip_i2c"
    raw=raw.loc[raw.get("unit_key",pd.Series(index=raw.index,dtype=str)).astype(str)==str(dominant_unit)].copy()
    tests=tests.loc[(tests.get("unit_key",pd.Series(index=tests.index,dtype=str)).astype(str)==str(dominant_unit)) & (tests.get("target",pd.Series(index=tests.index,dtype=str)).astype(str)==target)].copy()
    if raw.empty or target not in raw.columns or tests.empty or "example_local_index" not in raw.columns:
        return {}, {"status":"threshold_unit_or_target_unavailable","target":target}
    features=[str(v) for v in tests.get("feature",pd.Series(dtype=str)).dropna().unique() if str(v) in raw.columns]
    if not features: return {}, {"status":"threshold_features_unavailable","target":target}
    y=pd.to_numeric(raw[target],errors="coerce").to_numpy(float); valid=np.isfinite(y)
    raw=raw.loc[valid].reset_index(drop=True); y=(y[valid]>.5)
    if min(int(y.sum()),int((~y).sum())) < 2*int(min_class):
        return {}, {"status":"insufficient_threshold_classes","n_pos":int(y.sum()),"n_neg":int((~y).sum())}
    rng=np.random.default_rng(np.random.SeedSequence([int(seed), 8675309, 1 if direction=="c2i" else 2]))
    train=[]; test=[]
    for cls in (False,True):
        idx=np.flatnonzero(y==cls); rng.shuffle(idx)
        n_test=max(int(min_class),min(len(idx)-int(min_class),int(round(float(holdout_fraction)*len(idx)))))
        test.extend(idx[:n_test].tolist()); train.extend(idx[n_test:].tolist())
    train=np.asarray(train,dtype=int); test=np.asarray(test,dtype=int)
    best=None
    for feature in features:
        x=pd.to_numeric(raw[feature],errors="coerce").to_numpy(float)
        finite=np.isfinite(x[train])
        fit=_fit_preemption_threshold(x[train][finite],y[train][finite],max(1,int(min_class)//2))
        if fit is None: continue
        candidate=(float(fit["abs_mcc"]),feature,fit,x)
        if best is None or candidate[0]>best[0] or (math.isclose(candidate[0],best[0]) and feature<best[1]): best=candidate
    if best is None: return {}, {"status":"threshold_fit_failed"}
    _,feature,fit,x=best
    finite=np.isfinite(x[test]); test=test[finite]
    pred=x[test]>=fit["threshold"] if fit["direction"]==">=" else x[test]<=fit["threshold"]
    ytest=y[test]
    local=pd.to_numeric(raw.iloc[test]["example_local_index"],errors="coerce").to_numpy()
    id_col="original_idx" if "original_idx" in high.columns else ("_orig_row" if "_orig_row" in high.columns else None)
    if id_col is None: return {}, {"status":"threshold_scores_missing_row_identity"}
    mapping={}
    for li,event in zip(local,pred):
        if not np.isfinite(li): continue
        pos=int(li)
        if 0<=pos<len(high): mapping[high.iloc[pos][id_col]]=bool(event)
    return mapping, {
        "status":"ok","feature":feature,"threshold":float(fit["threshold"]),"direction":str(fit["direction"]),
        "train_abs_mcc":float(fit["abs_mcc"]),"heldout_abs_mcc":abs(float(_mcc_binary(ytest,pred))),
        "n_threshold_holdout":int(len(test)),"n_threshold_event_present":int(np.sum(pred)),"target":target,"row_id_column":id_col,
    }


def _analyze_preemption(
    *,
    plans: list[dict],
    pair_groups: dict[str, GroupSpec],
    candidates: list[UnitSpec],
    candidate_full: GroupSpec,
    scores_df: pd.DataFrame,
    baseline: np.ndarray,
    post_by_key: dict[str, np.ndarray],
    out_dir: Path,
    threshold_events: dict[str, dict[object, bool]],
    threshold_meta: dict[str, dict],
) -> dict:
    """Write P0.3 example masks and endogenous-event conditional marginals.

    T_j* is predicted from a one-dimensional endogenous scalar using a disjoint
    internal threshold holdout created from Step-7b diagnostics.  The pair
    intervention is then evaluated on those same held-out examples, making
    Delta_k^+ versus Delta_k^- a non-tautological preemption test.
    """
    unit_by_key = {u.unit_key: u for u in candidates}
    summary_rows: list[dict] = []
    mask_rows: list[dict] = []
    if not plans:
        payload = {"status": "insufficient_nontrivial_candidates", "n_pairs": 0}
        pd.DataFrame(columns=["direction","dominant_unit","secondary_unit","n_source","n_threshold_event_defined","delta_secondary_given_dominant_present","delta_secondary_given_dominant_absent","preemption_index_absent_minus_present"]).to_csv(out_dir / "preemption_pair_summary.csv", index=False)
        (out_dir / "preemption_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    full_post = np.asarray(post_by_key[candidate_full.key], dtype=bool)
    full_flip = full_post != baseline
    if "original_idx" in scores_df.columns:
        orig = scores_df["original_idx"].to_numpy()
    elif "_orig_row" in scores_df.columns:
        orig = scores_df["_orig_row"].to_numpy()
    else:
        orig = np.arange(len(scores_df))

    for plan in plans:
        direction = str(plan["direction"])
        dominant = unit_by_key[str(plan["dominant_unit"])]
        secondary = unit_by_key[str(plan["secondary_unit"])]
        pair = pair_groups[str(plan["pair_key"])]
        dom_col = _directional_flip_column(dominant, direction)
        sec_col = _directional_flip_column(secondary, direction)
        if dom_col not in scores_df.columns or sec_col not in scores_df.columns:
            continue
        source = baseline if direction == "c2i" else ~baseline
        f_dom = scores_df[dom_col].fillna(False).astype(bool).to_numpy() & source
        f_sec = scores_df[sec_col].fillna(False).astype(bool).to_numpy() & source
        f_pair = (np.asarray(post_by_key[pair.key], dtype=bool) != baseline) & source
        f_full = full_flip & source
        reachable = f_dom | f_sec
        indicator = threshold_events.get(direction, {})
        t_defined = np.asarray([key in indicator for key in orig], dtype=bool) & source
        t_event = np.asarray([bool(indicator.get(key, False)) for key in orig], dtype=bool)
        dom_present = t_defined & t_event
        dom_absent = t_defined & ~t_event
        increment = f_pair.astype(float) - f_dom.astype(float)
        delta_plus = _mean_or_nan(increment[dom_present])
        delta_minus = _mean_or_nan(increment[dom_absent])
        preemption_index = delta_minus - delta_plus if np.isfinite(delta_plus) and np.isfinite(delta_minus) else math.nan
        n_source = int(source.sum())
        row = {
            **plan,
            "n_source": n_source,
            "n_threshold_event_defined": int(t_defined.sum()),
            "n_threshold_event_present": int(dom_present.sum()),
            "n_threshold_event_absent": int(dom_absent.sum()),
            "threshold_event_status": str(threshold_meta.get(direction, {}).get("status", "unavailable")),
            "threshold_event_feature": threshold_meta.get(direction, {}).get("feature"),
            "threshold_event_heldout_abs_mcc": threshold_meta.get(direction, {}).get("heldout_abs_mcc"),
            "dominant_rate_observed": _mean_or_nan(f_dom[source].astype(float)),
            "secondary_rate_observed": _mean_or_nan(f_sec[source].astype(float)),
            "pair_rate": _mean_or_nan(f_pair[source].astype(float)),
            "full_set_rate": _mean_or_nan(f_full[source].astype(float)),
            "delta_secondary_given_dominant_present": delta_plus,
            "delta_secondary_given_dominant_absent": delta_minus,
            "preemption_index_absent_minus_present": preemption_index,
            "singleton_reachable_fraction": _mean_or_nan(reachable[source].astype(float)),
            "singleton_and_joint_fraction": _mean_or_nan((reachable & f_pair)[source].astype(float)),
            "singleton_reachable_not_joint_fraction": _mean_or_nan((reachable & ~f_pair)[source].astype(float)),
            "joint_only_fraction": _mean_or_nan((~reachable & f_pair & source)[source].astype(float)),
            "multi_singleton_covered_fraction": _mean_or_nan((f_dom & f_sec)[source].astype(float)),
            "evaluation_population_n": int(len(scores_df)),
        }
        summary_rows.append(row)
        for i in np.flatnonzero(source):
            mask_rows.append({
                "evaluation_row": int(i),
                "_orig_row": orig[i],
                "direction": direction,
                "dominant_unit": dominant.unit_key,
                "secondary_unit": secondary.unit_key,
                "dominant_singleton_event": bool(f_dom[i]),
                "threshold_event_defined": bool(t_defined[i]),
                "dominant_threshold_event": bool(t_event[i]) if t_defined[i] else None,
                "secondary_singleton_event": bool(f_sec[i]),
                "pair_event": bool(f_pair[i]),
                "full_set_event": bool(f_full[i]),
                "singleton_reachable": bool(reachable[i]),
                "singleton_reachable_not_joint": bool(reachable[i] and not f_pair[i]),
                "joint_only": bool((not reachable[i]) and f_pair[i]),
                "multi_singleton_covered": bool(f_dom[i] and f_sec[i]),
            })

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=["direction","dominant_unit","secondary_unit","n_source","n_threshold_event_defined","delta_secondary_given_dominant_present","delta_secondary_given_dominant_absent","preemption_index_absent_minus_present"])
    summary_df.to_csv(out_dir / "preemption_pair_summary.csv", index=False)
    mask_df = pd.DataFrame(mask_rows)
    if mask_df.empty:
        mask_df = pd.DataFrame(columns=["evaluation_row","_orig_row","direction","dominant_unit","secondary_unit","threshold_event_defined","dominant_threshold_event","dominant_singleton_event","secondary_singleton_event","pair_event","full_set_event","singleton_reachable","singleton_reachable_not_joint","joint_only","multi_singleton_covered"])
    mask_df.to_csv(out_dir / "preemption_example_masks.csv.gz", index=False, compression="gzip")
    finite = pd.to_numeric(summary_df.get("preemption_index_absent_minus_present"), errors="coerce") if not summary_df.empty else pd.Series(dtype=float)
    finite = finite[np.isfinite(finite)]
    if summary_df.empty:
        preemption_status = "missing_singleton_flip_columns"
    elif len(finite):
        preemption_status = "ok"
    else:
        preemption_status = "pair_masks_ok_threshold_conditioning_unavailable"
    payload = {
        "status": preemption_status,
        "conditioning_event": "held-out endogenous one-dimensional threshold event T_j*(x)",
        "threshold_event_metadata": threshold_meta,
        "prediction": "delta_secondary_given_dominant_present < delta_secondary_given_dominant_absent",
        "n_pairs": int(len(summary_df)),
        "n_pairs_with_defined_preemption_index": int(len(finite)),
        "fraction_pairs_supporting_preemption": float((finite > 0).mean()) if len(finite) else math.nan,
        "median_preemption_index": float(finite.median()) if len(finite) else math.nan,
        "files": {
            "pair_summary": "preemption_pair_summary.csv",
            "example_masks": "preemption_example_masks.csv.gz",
        },
    }
    (out_dir / "preemption_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    if args.null_draws < 1:
        raise ValueError("--null_draws must be at least 1")
    compute_cmc = not bool(args.skip_cmc)
    compute_preemption = not bool(args.skip_preemption)
    if float(args.preemption_min_singleton_rate) < 0 or float(args.preemption_min_singleton_rate) > 1:
        raise ValueError("--preemption_min_singleton_rate must be in [0, 1]")
    if int(args.preemption_max_secondaries) < 0:
        raise ValueError("--preemption_max_secondaries must be nonnegative")
    if not (0.0 < float(args.preemption_threshold_holdout_fraction) < 1.0):
        raise ValueError("--preemption_threshold_holdout_fraction must be in (0, 1)")
    if int(args.preemption_threshold_min_class) < 1:
        raise ValueError("--preemption_threshold_min_class must be >= 1")
    background_multipliers = (
        _parse_background_multipliers(args.background_multipliers) if compute_cmc else ()
    )

    input_data_dir = Path(args.input_data_dir).expanduser().resolve()
    candidate_path = Path(args.candidate_flip_stats_path).expanduser().resolve()
    stats_dir = candidate_path.parent
    singleton_scores_path = (
        Path(args.singleton_scores_path).expanduser().resolve()
        if args.singleton_scores_path else stats_dir / "scores.csv"
    )
    ranking_path = (
        Path(args.frozen_ranking_path).expanduser().resolve()
        if args.frozen_ranking_path else stats_dir / "frozen_candidate_ranking.csv"
    )
    manifest_path = (
        Path(args.layer_population_manifest).expanduser().resolve()
        if args.layer_population_manifest else input_data_dir / "manifest.json"
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold_dir = Path(args.threshold_diagnostics_dir).expanduser().resolve() if args.threshold_diagnostics_dir else None
    global_path = stats_dir / "flip_stats_global.json"

    dataset_info_path = input_data_dir / "dataset_info.json"
    dataset_info = load_dataset_info(input_data_dir)
    task = resolve_task_spec(args.task_module)
    prompt_col = dataset_info.get("prompt_col") or task.DEFAULT_INPUT
    target_col = dataset_info.get("target_col") or task.DEFAULT_TARGETS[0]
    ai_model = args.ai_model or dataset_info.get("ai_model")
    if not ai_model:
        raise ValueError("Could not resolve model from --ai_model or dataset_info.json")
    train_scores_path = resolve_dataset_path(dataset_info["scores_path"], input_data_dir)

    candidates = load_candidate_units(candidate_path)
    candidate_keys_list = [unit.unit_key for unit in candidates]
    candidate_keys = set(candidate_keys_list)
    ranking = _load_ranking(ranking_path, candidates)
    candidate_stats = _candidate_stat_frame(candidate_path, candidates)

    locus_population = load_stage5_locus_population(manifest_path)
    all_population_units = dedupe_units(unit for units in locus_population.values() for unit in units)
    population_strata = group_units_by_matching_stratum(all_population_units)
    candidate_strata = group_units_by_matching_stratum(candidates)
    for stratum, stratum_candidates in candidate_strata.items():
        eligible = population_strata.get(stratum, [])
        eligible_keys = {unit.unit_key for unit in eligible}
        missing = {unit.unit_key for unit in stratum_candidates} - eligible_keys
        if missing:
            raise ValueError(
                f"Candidate stratum {stratum} is absent/incomplete in stage-5 population: "
                + ", ".join(sorted(missing))
            )

    scores_df = evaluation_frame(
        singleton_scores_path,
        prompt_col=prompt_col,
        target_col=target_col,
        evaluation_split=str(args.evaluation_split),
    )
    evaluation_rows = evaluation_row_records(
        scores_df,
        prompt_col=prompt_col,
        target_col=target_col,
    )
    baseline = pd.to_numeric(scores_df[target_col], errors="raise").to_numpy() > 0.5
    effective_seed = int(args.seed if args.seed is not None else 42)
    set_deterministic(effective_seed)

    train_scores = load_scores_for_baseline(
        scores_path=train_scores_path,
        target_col=target_col,
        baseline_subset="all",
        task_targets=task.DEFAULT_TARGETS,
        split="train",
    )
    if train_scores.empty and args.intervention.startswith("mean"):
        raise ValueError("No training rows are available for replacement estimation")
    replacement_reference_prompts = build_mean_prompt_pool(
        train_scores,
        prompt_col=prompt_col,
        target_col=target_col,
        n_points=int(args.points_to_use_for_mean_ablation),
        seed=effective_seed,
    )

    cache_identity = {
        "schema": SCHEMA,
        "task_module": str(args.task_module),
        "ai_model": ai_model,
        "intervention": str(args.intervention),
        "intervention_phase": "decode_only" if args.decode_only else "prefill_decode",
        "evaluation_split": str(args.evaluation_split),
        "prompt_col": str(prompt_col),
        "target_col": str(target_col),
        "evaluation_rows": evaluation_rows,
        "points_to_use_for_mean_ablation": int(args.points_to_use_for_mean_ablation),
        "replacement_reference_prompts": list(map(str, replacement_reference_prompts)),
        "seed": effective_seed,
        "max_new_tokens": int(task.MAX_NEW_TOKENS),
        "candidate_set": sorted(candidate_keys_list),
    }
    paths = {
        "input_data_dir": str(input_data_dir),
        "candidate_flip_stats_path": str(candidate_path),
        "singleton_scores_path": str(singleton_scores_path),
        "frozen_ranking_path": str(ranking_path),
        "layer_population_manifest": str(manifest_path),
        "replacement_reference_scores_path": str(train_scores_path),
        "threshold_diagnostics_dir": str(threshold_dir) if threshold_dir is not None else None,
    }
    configuration = {
        **cache_identity,
        "cache_identity": cache_identity,
        "paths": paths,
    }
    config_path = out_dir / "interaction_configuration.json"
    summary_path = out_dir / "interaction_validation_summary.json"
    # Build structurally matched K_b sets.
    random_full_groups: dict[int, GroupSpec] = {}
    for draw in range(int(args.null_draws)):
        random_full_groups[draw] = _draw_matched_null_group(
            draw=draw,
            candidate_strata=candidate_strata,
            population_strata=population_strata,
            candidate_keys=candidate_keys,
            seed=effective_seed,
        )

    # Backgrounds are matched by the same exact strata as J and are disjoint
    # from both J and the paired K_b.
    background_groups: dict[tuple[int, int], GroupSpec] = {}
    candidate_context_groups: dict[tuple[int, int], GroupSpec] = {}
    null_context_groups: dict[tuple[int, int], GroupSpec] = {}
    for multiplier in background_multipliers:
        for draw in range(int(args.null_draws)):
            background = _draw_background_group(
                draw=draw,
                multiplier=multiplier,
                candidate_strata=candidate_strata,
                population_strata=population_strata,
                candidate_keys=candidate_keys,
                null_group=random_full_groups[draw],
                seed=effective_seed,
            )
            background_groups[(multiplier, draw)] = background
            candidate_context_groups[(multiplier, draw)] = _union_group(
                background, _group(candidates, "candidate_J"),
                label=f"background_{multiplier}x_{draw}_plus_candidate_J",
            )
            null_context_groups[(multiplier, draw)] = _union_group(
                background, random_full_groups[draw],
                label=f"background_{multiplier}x_{draw}_plus_random_J_{draw}",
            )

    # Validate and write matching strata before model execution.
    stratum_rows: list[dict] = []
    for stratum in sorted(candidate_strata):
        transformer_layer, computational_locus, channel_type = stratum
        q = len(candidate_strata[stratum])
        eligible = len(population_strata[stratum])
        stratum_rows.append({
            "transformer_layer": int(transformer_layer),
            "computational_locus": computational_locus,
            "channel_type": channel_type,
            "candidate_count": q,
            "eligible_stage5_count": eligible,
            "noncandidate_pool_count": eligible - q,
            "intervention_phase": "decode_only" if args.decode_only else "prefill_decode",
            "replacement_baseline": args.intervention,
        })
    pd.DataFrame(stratum_rows).to_csv(out_dir / "matched_control_strata.csv", index=False)

    null_membership_rows: list[dict] = []
    background_membership_rows: list[dict] = []
    for draw, group in random_full_groups.items():
        for unit in group.units:
            meta = unit_metadata(unit)
            null_membership_rows.append({
                "draw": draw,
                "unit_key": unit.unit_key,
                "stage5_locus": meta["stage5_locus"],
                "neuron_id": int(unit.neuron_id),
                "transformer_layer": int(meta["transformer_layer"]),
                "computational_locus": meta["computational_locus"],
                "channel_type": meta["channel_type"],
                "intervention_phase": "decode_only" if args.decode_only else "prefill_decode",
                "replacement_baseline": args.intervention,
                "evaluation_population_n": int(len(scores_df)),
            })
    for (multiplier, draw), group in background_groups.items():
        for unit in group.units:
            meta = unit_metadata(unit)
            background_membership_rows.append({
                "background_multiplier": multiplier,
                "draw": draw,
                "unit_key": unit.unit_key,
                "stage5_locus": meta["stage5_locus"],
                "neuron_id": int(unit.neuron_id),
                "transformer_layer": int(meta["transformer_layer"]),
                "computational_locus": meta["computational_locus"],
                "channel_type": meta["channel_type"],
                "intervention_phase": "decode_only" if args.decode_only else "prefill_decode",
                "replacement_baseline": args.intervention,
                "evaluation_population_n": int(len(scores_df)),
            })
    pd.DataFrame(null_membership_rows).to_csv(out_dir / "matched_random_set_membership.csv", index=False)
    conditional_membership_path = out_dir / "conditional_background_membership.csv"
    if compute_cmc:
        pd.DataFrame(background_membership_rows).to_csv(conditional_membership_path, index=False)
    elif conditional_membership_path.exists():
        print(f"{LOG_PREFIX} preserving pre-existing optional CMC artifact: {conditional_membership_path}")
    ranking.to_csv(out_dir / "frozen_candidate_ranking.csv", index=False)

    preemption_plans, preemption_pair_groups = (
        _preemption_pair_plan(
            candidates=candidates,
            candidate_stats=candidate_stats,
            min_rate=float(args.preemption_min_singleton_rate),
            max_secondaries=int(args.preemption_max_secondaries),
        )
        if compute_preemption
        else ([], {})
    )
    pd.DataFrame(preemption_plans).to_csv(out_dir / "preemption_pair_plan.csv", index=False)
    threshold_events: dict[str, dict[object, bool]] = {}
    threshold_meta: dict[str, dict] = {}
    if compute_preemption:
        for direction in ("c2i", "i2c"):
            directional = [p for p in preemption_plans if p["direction"] == direction]
            if not directional:
                threshold_events[direction] = {}; threshold_meta[direction] = {"status": "no_eligible_pair"}; continue
            dominant_key = str(directional[0]["dominant_unit"])
            events, meta = _threshold_event_indicator(
                threshold_dir=threshold_dir, dominant_unit=dominant_key, direction=direction, seed=effective_seed,
                holdout_fraction=float(args.preemption_threshold_holdout_fraction), min_class=int(args.preemption_threshold_min_class),
            )
            threshold_events[direction] = events; threshold_meta[direction] = meta
    candidate_full = _group(candidates, "candidate_J")
    all_groups: list[GroupSpec] = [candidate_full, *random_full_groups.values(), *preemption_pair_groups.values()]
    for key in sorted(background_groups):
        all_groups.extend([
            background_groups[key], candidate_context_groups[key], null_context_groups[key]
        ])
    unique_groups = {group.key: group for group in all_groups if group.size > 0}
    groups_to_evaluate = list(unique_groups.values())
    all_units = dedupe_units(unit for group in groups_to_evaluate for unit in group.units)

    # Record the analysis configuration after the exact requested simultaneous
    # groups are known.  These fields describe reporting/analysis semantics; the
    # model-output cache below uses only the inputs that can change a group output.
    cache_identity.update({
        "compute_cmc": bool(compute_cmc),
        "compute_preemption": bool(compute_preemption),
        "preemption_min_singleton_rate": float(args.preemption_min_singleton_rate),
        "preemption_max_secondaries": int(args.preemption_max_secondaries),
        "preemption_threshold_holdout_fraction": float(args.preemption_threshold_holdout_fraction),
        "preemption_threshold_min_class": int(args.preemption_threshold_min_class),
        "background_multipliers": list(background_multipliers),
        "null_draws": int(args.null_draws),
        "requested_group_keys": sorted(unique_groups),
    })
    configuration.update(cache_identity)
    configuration["cache_identity"] = cache_identity

    # Group-intervention cache identity is explicit.  Rewriting Stage-7 or
    # Step-7b CSV/JSON files with the same rows/groups does not invalidate these
    # expensive model outputs, and batch size is deliberately absent because it
    # is operational rather than scientific.
    group_cache_context = {
        "schema": GROUP_CACHE_SCHEMA,
        "model": ai_model,
        "task_module": str(args.task_module),
        "evaluation_split": str(args.evaluation_split),
        "prompt_col": str(prompt_col),
        "target_col": str(target_col),
        "evaluation_rows": evaluation_rows,
        "decode_only": bool(args.decode_only),
        "intervention": str(args.intervention),
        "replacement_reference_prompts": list(map(str, replacement_reference_prompts)),
        "max_new_tokens": int(task.MAX_NEW_TOKENS),
    }
    print(
        f"{LOG_PREFIX} split={args.evaluation_split} rows={len(scores_df)} candidates={len(candidates)} "
        f"matching_strata={len(candidate_strata)} cmc={'on' if compute_cmc else 'off'} "
        f"backgrounds={background_multipliers if compute_cmc else 'disabled'} "
        f"preemption_pairs={len(preemption_plans) if compute_preemption else 'disabled'} "
        f"groups_to_evaluate={len(groups_to_evaluate)} null_draws={args.null_draws}"
    )

    post_by_key: dict[str, np.ndarray] = {}
    if groups_to_evaluate:
        group_cache_dir = out_dir / "group_eval_cache"
        if not args.force:
            cached_outputs = load_complete_group_batch_cache(
                cache_dir=group_cache_dir,
                groups=groups_to_evaluate,
                n_examples=len(scores_df),
                batch_size=int(args.batch_size),
                cache_context=group_cache_context,
            )
            if cached_outputs is not None:
                post_by_key = cached_outputs
                print(
                    f"{LOG_PREFIX} complete explicit cache covers all {len(groups_to_evaluate)} "
                    f"groups and {len(scores_df)} rows; skipping model load"
                )

        if not post_by_key:
            device = get_device()
            model = LMWrapper(
                model_name=ai_model,
                device=device,
                eval_mode=True,
                circuit_discovery=False,
                cache_dir=args.ai_model_cache_dir,
            )
            mean_activations = precompute_replacements_for_units(
                model=model,
                units=all_units,
                scores_df_for_mean=train_scores,
                prompt_col=prompt_col,
                target_col=target_col,
                intervention=args.intervention,
                points_to_use=int(args.points_to_use_for_mean_ablation),
                batch_size=int(args.batch_size),
                seed=effective_seed,
            )
            post_by_key = evaluate_groups(
                model=model,
                groups=groups_to_evaluate,
                scores_df=scores_df,
                prompt_col=prompt_col,
                is_answer_positive_fn=task.is_answer_positive,
                batch_size=int(args.batch_size),
                decode_only=bool(args.decode_only),
                intervention=args.intervention,
                mean_activations=mean_activations,
                max_new_tokens=int(task.MAX_NEW_TOKENS),
                cache_dir=group_cache_dir,
                cache_context=group_cache_context,
                force=bool(args.force),
            )

    candidate_effect = _effect_for_group(
        candidate_full, baseline=baseline, post_by_key=post_by_key
    )
    preemption_payload = (
        _analyze_preemption(
            plans=preemption_plans,
            pair_groups=preemption_pair_groups,
            candidates=candidates,
            candidate_full=candidate_full,
            scores_df=scores_df,
            baseline=baseline,
            post_by_key=post_by_key,
            out_dir=out_dir,
            threshold_events=threshold_events,
            threshold_meta=threshold_meta,
        )
        if compute_preemption
        else {"status": "disabled", "n_pairs": 0}
    )
    e_null_values: list[float] = []
    direct_null_rows: list[dict] = []
    for draw in range(int(args.null_draws)):
        effect = _effect_for_group(random_full_groups[draw], baseline=baseline, post_by_key=post_by_key)
        value = float(effect["effect"])
        status = str(effect["status"])
        e_null_values.append(value)
        direct_null_rows.append({
            "draw": draw,
            "metric": "E_J",
            "value": value,
            "status": status,
            "evaluation_population_n": int(len(scores_df)),
        })

    direct_summary = matched_null_summary(
        float(candidate_effect["effect"]), e_null_values, metric="E_J", m=None
    )

    conditional_draw_rows: list[dict] = []
    conditional_summaries: list[dict] = []
    for multiplier in background_multipliers:
        candidate_marginals: list[float] = []
        null_marginals: list[float] = []
        for draw in range(int(args.null_draws)):
            background = _effect_for_group(
                background_groups[(multiplier, draw)], baseline=baseline, post_by_key=post_by_key
            )
            candidate_context = _effect_for_group(
                candidate_context_groups[(multiplier, draw)], baseline=baseline, post_by_key=post_by_key
            )
            null_context = _effect_for_group(
                null_context_groups[(multiplier, draw)], baseline=baseline, post_by_key=post_by_key
            )
            candidate_marginal = float(candidate_context["effect"] - background["effect"])
            null_marginal = float(null_context["effect"] - background["effect"])
            paired_delta = float(candidate_marginal - null_marginal)
            candidate_marginals.append(candidate_marginal)
            null_marginals.append(null_marginal)
            conditional_draw_rows.append({
                "background_multiplier": multiplier,
                "draw": draw,
                "E_S": background["effect"],
                "E_S_plus_J": candidate_context["effect"],
                "E_S_plus_K": null_context["effect"],
                "candidate_marginal": candidate_marginal,
                "null_marginal": null_marginal,
                "paired_Delta": paired_delta,
                "background_status": background["status"],
                "candidate_context_status": candidate_context["status"],
                "null_context_status": null_context["status"],
                "evaluation_population_n": int(len(scores_df)),
            })
        summary = paired_conditional_summary(
            candidate_marginals,
            null_marginals,
            background_multiplier=multiplier,
        )
        conditional_summaries.append(summary)
        _write_paired_plot(
            out_dir / f"conditional_marginal_{multiplier}x_paired_delta",
            candidate_marginals,
            null_marginals,
            f"Conditional marginal contribution ({multiplier}x background)",
        )

    summary_rows = [direct_summary, *conditional_summaries]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "interaction_validation_summary.csv", index=False)
    pd.DataFrame(direct_null_rows).to_csv(out_dir / "matched_null_draws.csv", index=False)
    if compute_cmc:
        summary_df.loc[summary_df["metric"].astype(str) == "conditional_marginal"].to_csv(
            out_dir / "conditional_marginal_summary.csv", index=False
        )
        pd.DataFrame(conditional_draw_rows).to_csv(
            out_dir / "conditional_marginal_draws.csv", index=False
        )
    else:
        preserved = [
            out_dir / "conditional_marginal_summary.csv",
            out_dir / "conditional_marginal_draws.csv",
            *list(out_dir.glob("conditional_marginal_*_paired_delta.pdf")),
        ]
        for stale in preserved:
            if stale.exists():
                print(f"{LOG_PREFIX} preserving pre-existing optional CMC artifact: {stale}")
    (out_dir / "interaction_validation_table.tex").write_text(_latex_table(summary_df), encoding="utf-8")

    payload = {
        "definition_version": SCHEMA,
        "definitions": {
            "E_J": "P(B_J(x) != B(x)) under simultaneous suppression of the complete fixed candidate set J",
            "K_b": "noncandidate set exactly matched to J by transformer layer, computational locus, channel type and per-stratum cardinality",
            "S_b": "noncandidate background matched to the candidate topology, disjoint from J and its paired K_b",
            "candidate_marginal": "M_b(J)=E(S_b union J)-E(S_b)",
            "null_marginal": "M_b(K_b)=E(S_b union K_b)-E(S_b)",
            "paired_Delta": "D_b=M_b(J)-M_b(K_b)",
            "conditional_candidate": "mean_b M_b(J)",
            "conditional_Delta": "mean_b D_b",
            "P": "(1 + sum_b 1{D_b >= 0})/(B+1) for the paired conditional metric",
            "p_MC": "(1 + sum_b 1{D_b <= 0})/(B+1) for the paired conditional metric",
        },
        "candidate_set": candidate_keys_list,
        "candidate_set_size": len(candidates),
        "candidate_E_J": candidate_effect,
        "direct_E_J_null_summary": direct_summary,
        "cmc_enabled": bool(compute_cmc),
        "preemption_enabled": bool(compute_preemption),
        "preemption": preemption_payload,
        "conditional_marginal": conditional_summaries,
        "background_multipliers": list(background_multipliers),
        "null_draws": int(args.null_draws),
        "matching": [
            "transformer layer", "computational locus", "channel type",
            "intervention phase", "replacement baseline", "per-stratum cardinality",
        ],
        "intervention_phase": "decode_only" if args.decode_only else "prefill_decode",
        "replacement_baseline": args.intervention,
        "candidate_discovery_split": "train",
        "replacement_reference_split": "train",
        "evaluation_split": str(args.evaluation_split),
        "evaluation_population_n": int(len(scores_df)),
        "evaluation_row_identity_columns": list(evaluation_rows[0].keys()) if evaluation_rows else [],
        "same_evaluation_rows_for_candidate_background_and_all_null_draws": True,
        "layer_population_manifest": str(manifest_path),
        "frozen_ranking_path": str(ranking_path),
        "points_to_use_for_mean_ablation": int(args.points_to_use_for_mean_ablation),
        "seed": effective_seed,
        "notes": [
            "No effect is clipped.",
            "Every E value is obtained from a genuine simultaneous intervention on the named set.",
            "Singleton-union events are never used for E(J) or null controls.",
            *(
                [
                    "No marginal contribution is clipped.",
                    "Singleton-union events are never used for conditional marginals.",
                    "Candidate and paired null are compared in the identical S_b background and on identical evaluation examples.",
                ]
                if compute_cmc
                else ["CMC was disabled; no conditional-background interventions were evaluated."]
            ),
        ],
    }
    summary_path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    config_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")

    markdown = [
        (
            "# Simultaneous-set and conditional marginal validation"
            if compute_cmc else "# Simultaneous-set validation"
        ),
        "",
        f"- Candidate set size: {len(candidates)}",
        f"- Evaluation split: {args.evaluation_split}",
        f"- Evaluation examples: {len(scores_df)} (explicit row identities stored in interaction_configuration.json)",
        f"- Exact matched strata: {len(candidate_strata)}",
        f"- Matched random draws: {args.null_draws}",
        f"- CMC: {'enabled' if compute_cmc else 'disabled'}",
        f"- P0.3 preemption pairs: {preemption_payload.get('n_pairs', 0)} ({preemption_payload.get('status', 'unknown')})",
        f"- Background multipliers: {', '.join(map(str, background_multipliers)) if compute_cmc else 'not evaluated'}",
        f"- Intervention phase: {'decode only' if args.decode_only else 'input and output'}",
        f"- Replacement baseline: {args.intervention}",
        "- Every set effect is a genuine simultaneous intervention.",
        *(
            ["- Candidate and matched null use the same background S_b in each paired draw."]
            if compute_cmc else []
        ),
        "",
        "| Metric | Background | Candidate | Null median | Delta | P | p_MC | Status |",
        "|:--|:--|--:|--:|--:|--:|--:|:--|",
    ]
    for row in summary_rows:
        background = "--" if row.get("background_multiplier") is None else f"{row['background_multiplier']}x"
        def fmt(value):
            try:
                number = float(value)
            except Exception:
                return "NA"
            return "NA" if not np.isfinite(number) else f"{number:.4f}"
        markdown.append(
            f"| {row['metric']} | {background} | {fmt(row.get('candidate'))} | "
            f"{fmt(row.get('median_null'))} | {fmt(row.get('Delta'))} | "
            f"{fmt(row.get('P'))} | {fmt(row.get('p_MC'))} | {row.get('status', 'unknown')} |"
        )
    (out_dir / "interaction_validation_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    _write_plot(
        out_dir / "E_J_matched_null",
        e_null_values,
        float(candidate_effect["effect"]),
        "Simultaneous full-set effect",
        "E(J)",
    )

    if global_path.exists():
        global_payload = json.loads(global_path.read_text(encoding="utf-8"))
        global_payload.update({
            "E_J": candidate_effect["effect"],
            "E_J_count": candidate_effect.get("count"),
            "E_J_denominator": candidate_effect.get("denominator"),
            "E_J_status": candidate_effect.get("status"),
            "E_J_0to1": candidate_effect.get("effect_0to1"),
            "E_J_1to0": candidate_effect.get("effect_1to0"),
            "conditional_marginal": {
                str(row["background_multiplier"]): {
                    "value": row["candidate"],
                    "candidate_median": row["candidate_median"],
                    "null_mean": row["null_mean"],
                    "null_median": row["median_null"],
                    "paired_Delta_mean": row["Delta"],
                    "paired_Delta_median": row["Delta_median"],
                    "P": row["P"],
                    "p_MC": row["p_MC"],
                    "paired_win_rate": row["paired_win_rate"],
                    "status": row["status"],
                }
                for row in conditional_summaries
            },
            "interaction_validation_definition_version": SCHEMA,
            "interaction_validation_path": str(summary_path),
            "interaction_null_summary_path": str(out_dir / "interaction_validation_summary.csv"),
            "preemption_summary_path": str(out_dir / "preemption_pair_summary.csv") if compute_preemption else None,
            "preemption_median_index": preemption_payload.get("median_preemption_index"),
        })
        global_path.write_text(json.dumps(global_payload, indent=2, allow_nan=True), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"{LOG_PREFIX} wrote {out_dir}")


if __name__ == "__main__":
    main()
