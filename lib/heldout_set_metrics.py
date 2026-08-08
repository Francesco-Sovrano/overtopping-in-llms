"""Held-out set metrics derived from saved singleton flip events.

The functions in this module are model-free.  They operate on the per-example
singleton flip columns written by stage 7 and on a candidate ranking frozen on
discovery data.  No held-out outcome is used to choose ``H_m``.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd



def safe_layer_label(value: object) -> str:
    """Filesystem/column-safe layer label without importing model libraries."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


DEFAULT_THRESHOLDS = (0.01, 0.05, 0.10, 0.20, 0.30)


@dataclass(frozen=True)
class RatioResult:
    value: float
    status: str
    numerator: float
    denominator: float


def safe_ratio(numerator: float, denominator: float, *, epsilon: float) -> RatioResult:
    """Return an unclipped ratio and an explicit denominator status."""
    numerator = float(numerator)
    denominator = float(denominator)
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return RatioResult(math.nan, "undefined_nonfinite", numerator, denominator)
    if abs(denominator) <= float(epsilon):
        return RatioResult(math.nan, "undefined_near_zero_denominator", numerator, denominator)
    return RatioResult(numerator / denominator, "ok", numerator, denominator)


def unit_key(layer_label: object, neuron_id: object) -> str:
    return f"{str(layer_label)}:{int(neuron_id)}"


def flip_column(layer_label: object, neuron_id: object) -> str:
    return f"flip_{safe_layer_label(layer_label)}_{int(neuron_id)}"


def _candidate_frame(candidate_stats: pd.DataFrame) -> pd.DataFrame:
    layer_col = "layer_label" if "layer_label" in candidate_stats.columns else "layer_key"
    required = {layer_col, "neuron_id"}
    missing = required - set(candidate_stats.columns)
    if missing:
        raise ValueError(f"Candidate statistics are missing columns: {sorted(missing)}")
    out = candidate_stats.copy()
    out["layer_label"] = out[layer_col].astype(str)
    out["neuron_id"] = pd.to_numeric(out["neuron_id"], errors="raise").astype(int)
    out["unit_key"] = [unit_key(a, b) for a, b in zip(out["layer_label"], out["neuron_id"])]
    out = out.drop_duplicates("unit_key", keep="first").reset_index(drop=True)
    return out


def _ranked_candidates(
    candidates: pd.DataFrame,
    frozen_ranking: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str]:
    """Join a discovery-frozen ranking without falling back to held-out effects."""
    if frozen_ranking is None or frozen_ranking.empty:
        out = candidates.copy()
        out["discovery_rank_global"] = np.nan
        out["discovery_score"] = np.nan
        return out, "undefined_missing_discovery_ranking"

    rank = frozen_ranking.copy()
    if "unit_key" not in rank.columns:
        if not {"layer_label", "neuron_id"}.issubset(rank.columns):
            raise ValueError("Frozen ranking requires unit_key or layer_label and neuron_id")
        rank["unit_key"] = [unit_key(a, b) for a, b in zip(rank["layer_label"], rank["neuron_id"])]
    if "discovery_rank_global" not in rank.columns:
        raise ValueError("Frozen ranking is missing discovery_rank_global")
    keep = [c for c in [
        "unit_key", "discovery_rank_global", "discovery_rank_within_layer",
        "discovery_score", "discovery_score_signed", "ranking_source",
        "computational_locus", "channel_type", "transformer_layer",
    ] if c in rank.columns]
    merged = candidates.merge(rank[keep], on="unit_key", how="left", validate="one_to_one")
    if merged["discovery_rank_global"].isna().any():
        missing = merged.loc[merged["discovery_rank_global"].isna(), "unit_key"].tolist()
        raise ValueError(
            "Frozen discovery ranking is incomplete for candidate set J: "
            + ", ".join(missing[:10])
        )
    merged["discovery_rank_global"] = pd.to_numeric(
        merged["discovery_rank_global"], errors="raise"
    ).astype(int)
    merged = merged.sort_values(
        ["discovery_rank_global", "layer_label", "neuron_id"], kind="mergesort"
    ).reset_index(drop=True)
    return merged, "ok"


def _event_arrays(scores: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    events: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in candidates.itertuples(index=False):
        col = flip_column(row.layer_label, row.neuron_id)
        if col not in scores.columns:
            raise ValueError(f"Missing singleton flip column {col!r} for {row.unit_key}")
        series = scores[col]
        evaluated = series.notna().to_numpy(dtype=bool)
        flipped = series.eq(True).to_numpy(dtype=bool)
        events[str(row.unit_key)] = (flipped, evaluated)
    return events


def _union(events: Iterable[tuple[np.ndarray, np.ndarray]], n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    union = np.zeros(n_rows, dtype=bool)
    evaluated = np.zeros(n_rows, dtype=bool)
    for flipped, eval_mask in events:
        union |= np.asarray(flipped, dtype=bool)
        evaluated |= np.asarray(eval_mask, dtype=bool)
    return union, evaluated



def derive_legacy_aggregate_metrics(
    *,
    global_payload: dict,
    candidate_stats: pd.DataFrame,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    denominator_epsilon: float = 1e-12,
) -> dict:
    """Recover only metrics exactly identifiable from aggregate legacy outputs.

    This helper deliberately does *not* invent discovery-frozen TOC values,
    baseline-conditioned OCC values, or simultaneous-intervention quantities.
    Those require per-example events/discovery ranking or new model interventions.
    """
    if "flip_any_rate" not in candidate_stats.columns:
        raise ValueError("Legacy candidate statistics require flip_any_rate")
    rates = pd.to_numeric(candidate_stats["flip_any_rate"], errors="coerce").to_numpy(dtype=float)
    rates = rates[np.isfinite(rates)]
    j = int(global_payload.get("n_neurons", len(candidate_stats)))
    u_j = float(global_payload.get("union_flip_any_unique_rate", math.nan))
    s_1 = float(rates.max()) if len(rates) else math.nan
    sum_s = float(rates.sum()) if len(rates) else 0.0
    sum_s_sq = float(np.square(rates).sum()) if len(rates) else 0.0
    overlap = safe_ratio(u_j, sum_s, epsilon=denominator_epsilon)
    n_eff = safe_ratio(sum_s * sum_s, sum_s_sq, epsilon=denominator_epsilon)
    threshold_counts = {
        f"{float(t):g}": int(np.sum(rates >= float(t)))
        for t in sorted({float(v) for v in thresholds})
    }
    return {
        "definition_version": "legacy-aggregate-exact-v1",
        "J": j,
        "U_J": u_j,
        "s_1": s_1,
        "sum_s_j": sum_s,
        "sum_s_j_squared": sum_s_sq,
        "R_ov": (1.0 - overlap.value) if overlap.status == "ok" else math.nan,
        "R_ov_status": overlap.status,
        "N_eff": n_eff.value,
        "N_eff_status": n_eff.status,
        "N_t": threshold_counts,
        "TOC_m": {},
        "TOC_status": "unavailable_requires_discovery_frozen_ranking_and_per_example_flip_events",
        "OCC_0": math.nan,
        "OCC_1": math.nan,
        "OCC_0_status": "unavailable_requires_per_example_baseline_and_union_events",
        "OCC_1_status": "unavailable_requires_per_example_baseline_and_union_events",
        "E_J": math.nan,
        "E_J_status": "unavailable_requires_simultaneous_intervention",
        "GCCR_status": "unavailable_requires_simultaneous_interventions",
        "recoverable_from_legacy_aggregates": ["J", "U_J", "s_1", "R_ov", "N_eff", "N_t"],
        "not_recoverable_from_legacy_aggregates": ["TOC_m", "OCC_0", "OCC_1", "E_J", "GCCR_m", "matched_nulls"],
    }

def compute_singleton_set_metrics(
    *,
    scores: pd.DataFrame,
    candidate_stats: pd.DataFrame,
    baseline_col: str,
    frozen_ranking: pd.DataFrame | None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    denominator_epsilon: float = 1e-12,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute singleton-union, concentration, overlap and occupancy metrics.

    Returns ``(summary, candidates, topm, threshold_counts)``.  ``topm`` contains
    one row for every ``m=1,...,|J|`` when a discovery ranking is available.
    """
    if baseline_col not in scores.columns:
        raise ValueError(f"Baseline predicate column {baseline_col!r} is missing")
    candidates = _candidate_frame(candidate_stats)
    ranked, ranking_status = _ranked_candidates(candidates, frozen_ranking)
    events = _event_arrays(scores, candidates)
    n_rows = len(scores)
    if events:
        common_eval = np.logical_and.reduce(
            [np.asarray(evaluated, dtype=bool) for _, evaluated in events.values()]
        )
    else:
        # U(empty)=0 on the held-out universe. This branch also keeps occupancy
        # denominators meaningful when no candidate survives discovery.
        common_eval = np.ones(n_rows, dtype=bool)
    full_union = np.zeros(n_rows, dtype=bool)
    for flipped, _ in events.values():
        full_union |= np.asarray(flipped, dtype=bool)
    full_union &= common_eval
    n_eval = int(common_eval.sum())
    union_count = int(full_union.sum())
    union_rate = float(union_count / n_eval) if n_eval else math.nan

    singleton_rows: list[dict] = []
    for row in candidates.itertuples(index=False):
        flipped, _ = events[str(row.unit_key)]
        n_j = n_eval
        k_j = int((np.asarray(flipped, dtype=bool) & common_eval).sum())
        singleton_rows.append(
            {
                "unit_key": str(row.unit_key),
                "singleton_count": k_j,
                "singleton_denominator": n_j,
                "s_j": float(k_j / n_j) if n_j else math.nan,
            }
        )
    singleton_df = pd.DataFrame(singleton_rows)
    candidates = candidates.merge(singleton_df, on="unit_key", how="left", validate="one_to_one")
    if ranking_status == "ok":
        rank_cols = [c for c in ranked.columns if c not in candidates.columns or c == "unit_key"]
        candidates = candidates.merge(
            ranked[rank_cols], on="unit_key", how="left", validate="one_to_one"
        )

    finite_s = pd.to_numeric(candidates["s_j"], errors="coerce").to_numpy(dtype=float)
    finite_s = finite_s[np.isfinite(finite_s)]
    sum_s = float(finite_s.sum()) if len(finite_s) else 0.0
    sum_s_sq = float(np.square(finite_s).sum()) if len(finite_s) else 0.0
    s1 = float(finite_s.max()) if len(finite_s) else math.nan
    overlap_ratio = safe_ratio(union_rate, sum_s, epsilon=denominator_epsilon)
    r_ov = (
        1.0 - overlap_ratio.value
        if overlap_ratio.status == "ok"
        else math.nan
    )
    n_eff_ratio = safe_ratio(sum_s * sum_s, sum_s_sq, epsilon=denominator_epsilon)

    baseline_numeric = pd.to_numeric(scores[baseline_col], errors="coerce")
    valid_baseline = baseline_numeric.notna().to_numpy(dtype=bool)
    baseline = baseline_numeric.fillna(0).to_numpy(dtype=float) > 0.5
    occ: dict[str, dict] = {}
    for b in (0, 1):
        eligible = common_eval & valid_baseline & (baseline == bool(b))
        denominator = int(eligible.sum())
        numerator = int((full_union & eligible).sum())
        value = float(numerator / denominator) if denominator else math.nan
        occ[str(b)] = {
            "value": value,
            "count": numerator,
            "denominator": denominator,
            "status": "ok" if denominator else "undefined_zero_denominator",
        }

    topm_rows: list[dict] = []
    if ranking_status == "ok":
        ranked_keys = ranked.sort_values("discovery_rank_global")["unit_key"].astype(str).tolist()
        running_events: list[tuple[np.ndarray, np.ndarray]] = []
        for m, key in enumerate(ranked_keys, start=1):
            running_events.append(events[key])
            h_union = np.zeros(n_rows, dtype=bool)
            for flipped, _ in running_events:
                h_union |= np.asarray(flipped, dtype=bool)
            h_union &= common_eval
            h_n = n_eval
            h_k = int(h_union.sum())
            u_h = float(h_k / h_n) if h_n else math.nan
            toc = safe_ratio(u_h, union_rate, epsilon=denominator_epsilon)
            topm_rows.append(
                {
                    "m": int(m),
                    "H_m_unit_keys": json.dumps(ranked_keys[:m]),
                    "U_H_m_count": h_k,
                    "U_H_m_denominator": h_n,
                    "U_H_m": u_h,
                    "U_J": union_rate,
                    "TOC_m": toc.value,
                    "TOC_m_status": toc.status,
                    "ranking_status": ranking_status,
                    "ranking_source": (
                        str(ranked.iloc[0].get("ranking_source"))
                        if "ranking_source" in ranked.columns and len(ranked)
                        else "discovery_data"
                    ),
                }
            )
    topm_df = pd.DataFrame(topm_rows)

    threshold_rows = []
    for threshold in sorted({float(v) for v in thresholds}):
        threshold_rows.append(
            {
                "threshold": threshold,
                "N_t": int(np.sum(finite_s >= threshold)),
                "candidate_count": int(len(candidates)),
            }
        )
    threshold_df = pd.DataFrame(threshold_rows)

    summary = {
        "definition_version": "heldout-set-metrics-v2",
        "candidate_set_size": int(len(candidates)),
        "J": int(len(candidates)),
        "n_input_heldout_rows": int(n_rows),
        "n_evaluated_rows": n_eval,
        "n_rows_excluded_incomplete_singleton_evaluation": int(n_rows - n_eval),
        "evaluation_universe_status": (
            "complete" if n_eval == n_rows else "complete_case_due_to_missing_singleton_outputs"
        ),
        "U_J_count": union_count,
        "U_J": union_rate,
        "s_1": s1,
        "sum_s_j": sum_s,
        "sum_s_j_squared": sum_s_sq,
        "R_ov": r_ov,
        "R_ov_status": overlap_ratio.status,
        "N_eff": n_eff_ratio.value,
        "N_eff_status": n_eff_ratio.status,
        "OCC_0": occ["0"]["value"],
        "OCC_0_count": occ["0"]["count"],
        "OCC_0_denominator": occ["0"]["denominator"],
        "OCC_0_status": occ["0"]["status"],
        "OCC_1": occ["1"]["value"],
        "OCC_1_count": occ["1"]["count"],
        "OCC_1_denominator": occ["1"]["denominator"],
        "OCC_1_status": occ["1"]["status"],
        "frozen_ranking_status": ranking_status,
        "denominator_epsilon": float(denominator_epsilon),
        "N_t": {f"{row['threshold']:g}": int(row["N_t"]) for row in threshold_rows},
        "notes": [
            "All singleton, H_m, and J probabilities use one common complete-case held-out evaluation mask.",
            "H_m is ordered only by the supplied discovery-data ranking.",
            "R_ov and N_eff are not clipped.",
            "OCC_b conditions the singleton-union event on the unablated binary predicate B(x)=b.",
            "E(J) is a separate simultaneous-set intervention and is not defined by this model-free function.",
        ],
    }
    if not topm_df.empty:
        summary["TOC_m"] = {
            str(int(row.m)): {
                "value": float(row.TOC_m),
                "status": str(row.TOC_m_status),
                "U_H_m": float(row.U_H_m),
            }
            for row in topm_df.itertuples(index=False)
        }
    else:
        summary["TOC_m"] = {}

    return summary, candidates, topm_df, threshold_df
