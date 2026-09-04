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
from typing import Sequence

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
    # Candidate statistics may already carry discovery-ranking metadata (for
    # example after Stage 7 writes flip_stats_by_neuron.csv).  The explicit
    # frozen_ranking argument is authoritative.  Drop overlapping metadata
    # before the join so pandas cannot create *_x/*_y columns and make the
    # canonical unsuffixed ranking fields disappear.
    overlapping_rank_cols = [
        column for column in keep
        if column != "unit_key" and column in candidates.columns
    ]
    merge_base = candidates.drop(columns=overlapping_rank_cols)
    merged = merge_base.merge(rank[keep], on="unit_key", how="left", validate="one_to_one")
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



def compute_singleton_set_metrics(
    *,
    scores: pd.DataFrame,
    candidate_stats: pd.DataFrame,
    baseline_col: str,
    frozen_ranking: pd.DataFrame | None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    denominator_epsilon: float = 1e-12,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute held-out union, concentration, and direction-specific handle metrics.

    Direction is defined relative to the unablated binary endpoint ``baseline_col``:

    * ``i2c`` / ``0->1`` conditions on rows whose baseline endpoint is 0 and asks
      whether the intervention flips it to 1.
    * ``c2i`` / ``1->0`` conditions on rows whose baseline endpoint is 1 and asks
      whether the intervention flips it to 0.

    Directional singleton rates therefore use *direction-eligible denominators*.
    This is essential for threshold counts such as ``N_t_i2c``: an intervention
    cannot be penalized for failing to produce a 0->1 transition on a row whose
    baseline endpoint is already 1.

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
        common_eval = np.ones(n_rows, dtype=bool)

    baseline_numeric = pd.to_numeric(scores[baseline_col], errors="coerce")
    valid_baseline = baseline_numeric.notna().to_numpy(dtype=bool)
    baseline = baseline_numeric.fillna(0).to_numpy(dtype=float) > 0.5
    eligible_i2c = common_eval & valid_baseline & (~baseline)
    eligible_c2i = common_eval & valid_baseline & baseline
    n_i2c = int(eligible_i2c.sum())
    n_c2i = int(eligible_c2i.sum())

    full_union = np.zeros(n_rows, dtype=bool)
    for flipped, _ in events.values():
        full_union |= np.asarray(flipped, dtype=bool)
    full_union &= common_eval
    n_eval = int(common_eval.sum())
    union_count = int(full_union.sum())
    union_rate = float(union_count / n_eval) if n_eval else math.nan

    union_i2c_count = int((full_union & eligible_i2c).sum())
    union_c2i_count = int((full_union & eligible_c2i).sum())
    union_i2c = float(union_i2c_count / n_i2c) if n_i2c else math.nan
    union_c2i = float(union_c2i_count / n_c2i) if n_c2i else math.nan

    singleton_rows: list[dict] = []
    for row in candidates.itertuples(index=False):
        flipped, _ = events[str(row.unit_key)]
        flipped = np.asarray(flipped, dtype=bool)
        k_j = int((flipped & common_eval).sum())
        k_i2c = int((flipped & eligible_i2c).sum())
        k_c2i = int((flipped & eligible_c2i).sum())
        singleton_rows.append(
            {
                "unit_key": str(row.unit_key),
                "singleton_count": k_j,
                "singleton_denominator": n_eval,
                "s_j": float(k_j / n_eval) if n_eval else math.nan,
                "singleton_i2c_count": k_i2c,
                "singleton_i2c_denominator": n_i2c,
                "s_j_i2c": float(k_i2c / n_i2c) if n_i2c else math.nan,
                "singleton_c2i_count": k_c2i,
                "singleton_c2i_denominator": n_c2i,
                "s_j_c2i": float(k_c2i / n_c2i) if n_c2i else math.nan,
            }
        )
    if singleton_rows:
        singleton_df = pd.DataFrame(singleton_rows)
        candidates = candidates.merge(singleton_df, on="unit_key", how="left", validate="one_to_one")
    else:
        candidates = candidates.copy()
        for column, dtype in (
            ("singleton_count", "int64"),
            ("singleton_denominator", "int64"),
            ("s_j", "float64"),
            ("singleton_i2c_count", "int64"),
            ("singleton_i2c_denominator", "int64"),
            ("s_j_i2c", "float64"),
            ("singleton_c2i_count", "int64"),
            ("singleton_c2i_denominator", "int64"),
            ("s_j_c2i", "float64"),
        ):
            candidates[column] = pd.Series(dtype=dtype)
    if ranking_status == "ok":
        rank_cols = [c for c in ranked.columns if c not in candidates.columns or c == "unit_key"]
        candidates = candidates.merge(
            ranked[rank_cols], on="unit_key", how="left", validate="one_to_one"
        )

    def finite_metric(column: str) -> np.ndarray:
        values = pd.to_numeric(candidates[column], errors="coerce").to_numpy(dtype=float)
        return values[np.isfinite(values)]

    def support(values: np.ndarray) -> dict[str, float | str]:
        if len(values) == 0:
            return {
                "sum": 0.0,
                "sum_sq": 0.0,
                "max": math.nan,
                "N_eff": math.nan,
                "N_eff_status": "undefined_no_direction_eligible_rows",
            }
        total = float(values.sum())
        total_sq = float(np.square(values).sum())
        ratio = safe_ratio(total * total, total_sq, epsilon=denominator_epsilon)
        return {
            "sum": total,
            "sum_sq": total_sq,
            "max": float(values.max()),
            "N_eff": ratio.value,
            "N_eff_status": ratio.status,
        }

    finite_s = finite_metric("s_j")
    finite_i2c = finite_metric("s_j_i2c")
    finite_c2i = finite_metric("s_j_c2i")
    overall = support(finite_s)
    i2c = support(finite_i2c)
    c2i = support(finite_c2i)

    overlap_ratio = safe_ratio(union_rate, float(overall["sum"]), epsilon=denominator_epsilon)
    overlap_i2c = safe_ratio(union_i2c, float(i2c["sum"]), epsilon=denominator_epsilon)
    overlap_c2i = safe_ratio(union_c2i, float(c2i["sum"]), epsilon=denominator_epsilon)

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
            h_k = int(h_union.sum())
            u_h = float(h_k / n_eval) if n_eval else math.nan
            h_i2c_count = int((h_union & eligible_i2c).sum())
            h_c2i_count = int((h_union & eligible_c2i).sum())
            u_h_i2c = float(h_i2c_count / n_i2c) if n_i2c else math.nan
            u_h_c2i = float(h_c2i_count / n_c2i) if n_c2i else math.nan
            toc = safe_ratio(u_h, union_rate, epsilon=denominator_epsilon)
            toc_i2c = safe_ratio(u_h_i2c, union_i2c, epsilon=denominator_epsilon)
            toc_c2i = safe_ratio(u_h_c2i, union_c2i, epsilon=denominator_epsilon)
            topm_rows.append(
                {
                    "m": int(m),
                    "H_m_unit_keys": json.dumps(ranked_keys[:m]),
                    "U_H_m_count": h_k,
                    "U_H_m_denominator": n_eval,
                    "U_H_m": u_h,
                    "U_J": union_rate,
                    "TOC_m": toc.value,
                    "TOC_m_status": toc.status,
                    "U_H_m_i2c_count": h_i2c_count,
                    "U_H_m_i2c_denominator": n_i2c,
                    "U_H_m_i2c": u_h_i2c,
                    "U_J_i2c": union_i2c,
                    "TOC_m_i2c": toc_i2c.value,
                    "TOC_m_i2c_status": toc_i2c.status,
                    "U_H_m_c2i_count": h_c2i_count,
                    "U_H_m_c2i_denominator": n_c2i,
                    "U_H_m_c2i": u_h_c2i,
                    "U_J_c2i": union_c2i,
                    "TOC_m_c2i": toc_c2i.value,
                    "TOC_m_c2i_status": toc_c2i.status,
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
                "N_t_i2c": int(np.sum(finite_i2c >= threshold)),
                "N_t_c2i": int(np.sum(finite_c2i >= threshold)),
                "candidate_count": int(len(candidates)),
                "i2c_eligible_rows": n_i2c,
                "c2i_eligible_rows": n_c2i,
            }
        )
    threshold_df = pd.DataFrame(threshold_rows)

    summary = {
        "definition_version": "heldout-set-metrics-v3-directional",
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
        "U_J_i2c": union_i2c,
        "U_J_i2c_count": union_i2c_count,
        "U_J_i2c_denominator": n_i2c,
        "U_J_i2c_status": "ok" if n_i2c else "undefined_zero_denominator",
        "U_J_c2i": union_c2i,
        "U_J_c2i_count": union_c2i_count,
        "U_J_c2i_denominator": n_c2i,
        "U_J_c2i_status": "ok" if n_c2i else "undefined_zero_denominator",
        "s_1": overall["max"],
        "s_1_i2c": i2c["max"],
        "s_1_c2i": c2i["max"],
        "sum_s_j": overall["sum"],
        "sum_s_j_i2c": i2c["sum"],
        "sum_s_j_c2i": c2i["sum"],
        "sum_s_j_squared": overall["sum_sq"],
        "R_ov": 1.0 - overlap_ratio.value if overlap_ratio.status == "ok" else math.nan,
        "R_ov_status": overlap_ratio.status,
        "R_ov_i2c": 1.0 - overlap_i2c.value if overlap_i2c.status == "ok" else math.nan,
        "R_ov_i2c_status": overlap_i2c.status,
        "R_ov_c2i": 1.0 - overlap_c2i.value if overlap_c2i.status == "ok" else math.nan,
        "R_ov_c2i_status": overlap_c2i.status,
        "N_eff": overall["N_eff"],
        "N_eff_status": overall["N_eff_status"],
        "N_eff_i2c": i2c["N_eff"],
        "N_eff_i2c_status": i2c["N_eff_status"],
        "N_eff_c2i": c2i["N_eff"],
        "N_eff_c2i_status": c2i["N_eff_status"],
        "frozen_ranking_status": ranking_status,
        "denominator_epsilon": float(denominator_epsilon),
        "N_t": {f"{row['threshold']:g}": int(row["N_t"]) for row in threshold_rows},
        "N_t_i2c": {f"{row['threshold']:g}": int(row["N_t_i2c"]) for row in threshold_rows},
        "N_t_c2i": {f"{row['threshold']:g}": int(row["N_t_c2i"]) for row in threshold_rows},
        "notes": [
            "All singleton, H_m, and J probabilities use one common complete-case held-out evaluation mask.",
            "Directional rates condition on the unablated endpoint: i2c is 0->1 and c2i is 1->0.",
            "N_t_i2c and N_t_c2i threshold direction-conditioned singleton rates, not pooled s_j.",
            "H_m is ordered only by the supplied discovery-data ranking.",
            "R_ov and N_eff are not clipped.",
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
        summary["TOC_m_i2c"] = {
            str(int(row.m)): {
                "value": float(row.TOC_m_i2c),
                "status": str(row.TOC_m_i2c_status),
                "U_H_m": float(row.U_H_m_i2c),
            }
            for row in topm_df.itertuples(index=False)
        }
        summary["TOC_m_c2i"] = {
            str(int(row.m)): {
                "value": float(row.TOC_m_c2i),
                "status": str(row.TOC_m_c2i_status),
                "U_H_m": float(row.U_H_m_c2i),
            }
            for row in topm_df.itertuples(index=False)
        }
    else:
        summary["TOC_m"] = {}
        summary["TOC_m_i2c"] = {}
        summary["TOC_m_c2i"] = {}

    return summary, candidates, topm_df, threshold_df
