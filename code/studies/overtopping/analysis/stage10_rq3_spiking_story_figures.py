#!/usr/bin/env python3
"""Generate the population-level RQ3 spiking-like visual story with uncertainty.

The stage separates four empirical claims:

1. event localization: the largest continuous-margin change is preferentially
   localized at the persistent behavioral crossing;
2. causal-strength sharpening: margin-change concentration C increases across
   held-out singleton-causal-strength tertiles formed within each run/direction;
3. task-specific competence sharpening in Arithmetic output-only 1->0;
4. endpoint-preserving transient events that cannot occur under the simple
   one-dimensional affine-margin null.

Inference is condition-aware. Examples and channels are aggregated inside each
run/direction before population uncertainty is estimated. Bootstrap confidence
intervals resample run/direction conditions, not individual examples. The
Arithmetic n=6 Spearman p-value is computed by exact permutation. No paper
figure titles are emitted; captions should carry figure descriptions.
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import permutations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr, wilcoxon
except Exception:  # pragma: no cover
    spearmanr = None
    wilcoxon = None


PRIMARY_MARGIN = "divergence_token"
LOW = "Low"
MID = "Middle"
HIGH = "High"
BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260911


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graded-dir", required=True)
    p.add_argument("--spiking-dir", required=True)
    p.add_argument("--paper-figures-dir", default=None)
    return p.parse_args()


def _bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().eq("true")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _format_p(p: float, *, decimals: int = 4) -> str:
    """Format plotted p-values without ever printing a rounded false zero."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "p=n/a"
    if not np.isfinite(p):
        return "p=n/a"
    threshold = 10.0 ** (-decimals)
    if p < threshold:
        return f"p<{threshold:.{decimals}f}"
    return f"p={p:.{decimals}f}"


def _clean(ax, *, xgrid: bool = False) -> None:
    ax.grid(axis="x" if xgrid else "y", alpha=.18, linewidth=.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _optional_csv(path: Path) -> pd.DataFrame | None:
    """Read a story input when available without turning a partial build fatal."""
    if not path.is_file():
        return None
    return pd.read_csv(path, low_memory=False)


def _bootstrap_median_ci(values: np.ndarray, *, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
    boot = np.median(values[idx], axis=1)
    lo, hi = np.quantile(boot, [.025, .975])
    return float(lo), float(hi)


def _cluster_weighted_rate_ci(d: pd.DataFrame, *, seed: int) -> tuple[float, float]:
    """Cluster bootstrap over run/direction, retaining within-cluster counts."""
    c = d.groupby(["run_id", "direction"], as_index=False).agg(
        transient_count=("transient_count", "sum"),
        n_examples=("n_examples", "sum"),
    )
    if c.empty:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), size=(BOOTSTRAP_REPS, len(c)))
    counts = c["transient_count"].to_numpy(float)[idx].sum(axis=1)
    ns = c["n_examples"].to_numpy(float)[idx].sum(axis=1)
    rates = counts / ns
    lo, hi = np.quantile(rates, [.025, .975])
    return float(lo), float(hi)


def _exact_spearman_permutation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if spearmanr is None:
        return math.nan, math.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(x) > 8:
        r = spearmanr(x, y)
        return float(r.statistic), float(r.pvalue)
    observed = float(spearmanr(x, y).statistic)
    null = np.asarray([float(spearmanr(x, p).statistic) for p in permutations(y.tolist())])
    p = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    return observed, p


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    finite = [(name, float(p)) for name, p in p_values.items() if np.isfinite(p)]
    finite.sort(key=lambda x: x[1])
    m = len(finite)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(finite):
        raw = min(1.0, (m - i) * p)
        running = max(running, raw)
        adjusted[name] = running
    for name in p_values:
        adjusted.setdefault(name, math.nan)
    return adjusted


def _strict_margin_population(
    margin_examples: pd.DataFrame,
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    example_required = {
        "run_id", "direction", "unit_key", "margin_type", "support_kind",
        "n_doses", "n_margin_doses", "known_flip_endpoint_reproduced",
        "margin_max_step_concentration_ratio", "margin_max_step_end_dose",
        "observed_behavior_persistent_crossing_dose", "dose_grid_step",
        "behavior_single_persistent_crossing",
    }
    score_required = {"run_id", "rq3_direction", "unit_key", "population", "population_strength"}
    if not example_required.issubset(margin_examples.columns) or not score_required.issubset(scores.columns):
        return (
            pd.DataFrame(columns=sorted(example_required | {"population_strength"})),
            pd.DataFrame(columns=["run_id", "direction", "unit_key", "strength", "concentration", "strength_percentile", "strength_tertile"]),
        )

    d = margin_examples.loc[
        margin_examples["margin_type"].astype(str).eq(PRIMARY_MARGIN)
        & margin_examples["support_kind"].astype(str).eq("known_flip")
        & _num(margin_examples["n_doses"]).eq(11)
        & _num(margin_examples["n_margin_doses"]).eq(11)
    ].copy()
    d = d.loc[_bool(d["known_flip_endpoint_reproduced"])].copy()

    for col in [
        "margin_max_step_concentration_ratio",
        "margin_max_step_end_dose",
        "observed_behavior_persistent_crossing_dose",
        "dose_grid_step",
    ]:
        d[col] = _num(d[col])

    cand = scores.loc[
        scores["population"].astype(str).eq("flip_rule_candidate"),
        ["run_id", "rq3_direction", "unit_key", "population_strength"],
    ].rename(columns={"rq3_direction": "direction"})
    cand["population_strength"] = _num(cand["population_strength"])
    d = d.merge(cand, on=["run_id", "direction", "unit_key"], how="inner")

    unit = d.groupby(["run_id", "direction", "unit_key"], as_index=False).agg(
        strength=("population_strength", "first"),
        concentration=("margin_max_step_concentration_ratio", "median"),
    )
    unit = unit.loc[np.isfinite(unit["concentration"])].copy()
    unit["strength_percentile"] = unit.groupby(["run_id", "direction"])["strength"].rank(
        pct=True, method="average"
    )
    unit["strength_tertile"] = pd.cut(
        unit["strength_percentile"],
        [0, 1 / 3, 2 / 3, 1],
        labels=[LOW, MID, HIGH],
        include_lowest=True,
    )
    return d, unit


def _event_profile(examples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, int]:
    d = examples.loc[
        _bool(examples["behavior_single_persistent_crossing"])
        & examples["observed_behavior_persistent_crossing_dose"].notna()
        & examples["margin_max_step_end_dose"].notna()
        & examples["dose_grid_step"].notna()
    ].copy()
    d["offset"] = np.rint(
        (d["margin_max_step_end_dose"] - d["observed_behavior_persistent_crossing_dose"])
        / d["dose_grid_step"]
    ).astype("Int64")

    rows: list[dict] = []
    offsets = np.arange(-5, 6)
    for (run_id, direction, unit_key), g in d.groupby(
        ["run_id", "direction", "unit_key"], dropna=False
    ):
        vals = g["offset"].dropna().astype(int).to_numpy()
        if not len(vals):
            continue
        for offset in offsets:
            rows.append({
                "run_id": run_id,
                "direction": direction,
                "unit_key": unit_key,
                "offset": int(offset),
                "fraction": float(np.mean(vals == offset)),
            })
    unit_event = pd.DataFrame(rows)
    if unit_event.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, int(len(d))

    condition = unit_event.groupby(
        ["run_id", "direction", "offset"], as_index=False
    )["fraction"].mean()

    out_rows: list[dict] = []
    for offset, g in condition.groupby("offset"):
        vals = g["fraction"].to_numpy(float)
        ci_low, ci_high = _bootstrap_median_ci(vals, seed=BOOTSTRAP_SEED + int(offset) + 20)
        out_rows.append({
            "offset": int(offset),
            "median": float(np.median(vals)),
            "q25": float(np.quantile(vals, .25)),
            "q75": float(np.quantile(vals, .75)),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_conditions": int(len(vals)),
        })
    profile = pd.DataFrame(out_rows).sort_values("offset")

    wide = condition.pivot_table(index=["run_id", "direction"], columns="offset", values="fraction")
    pairs = wide.dropna(subset=[-1, 0, 1]).copy()
    pairs["adjacent_mean"] = (pairs[-1] + pairs[1]) / 2
    pairs["event_minus_adjacent"] = pairs[0] - pairs["adjacent_mean"]
    delta = pairs["event_minus_adjacent"].to_numpy(float)
    ci_low, ci_high = _bootstrap_median_ci(delta, seed=BOOTSTRAP_SEED + 111)
    p = math.nan
    if wilcoxon is not None and len(delta) and np.any(delta != 0):
        p = float(wilcoxon(delta, alternative="greater", zero_method="wilcox").pvalue)
    stats = {
        "n_conditions": int(len(delta)),
        "median_event_minus_adjacent": float(np.median(delta)) if len(delta) else math.nan,
        "median_event_minus_adjacent_ci95": [ci_low, ci_high],
        "wilcoxon_one_sided_p": p,
    }
    return profile, pairs.reset_index(), stats, int(len(d))


def _local_enrichment_profile(event_pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize each offset by that condition's adjacent-bin baseline."""
    if event_pairs.empty:
        return pd.DataFrame(), {
            "n_conditions": 0,
            "median_event_over_adjacent": math.nan,
            "median_event_over_adjacent_ci95": [math.nan, math.nan],
            "wilcoxon_one_sided_p": math.nan,
        }

    d = event_pairs.copy()
    d["adjacent_mean"] = _num(d["adjacent_mean"])
    d = d.loc[d["adjacent_mean"].gt(0)].copy()
    if d.empty:
        return pd.DataFrame(), {
            "n_conditions": 0,
            "median_event_over_adjacent": math.nan,
            "median_event_over_adjacent_ci95": [math.nan, math.nan],
            "wilcoxon_one_sided_p": math.nan,
        }

    def _offset_col(offset: int):
        if offset in d.columns:
            return offset
        key = str(offset)
        if key in d.columns:
            return key
        return None

    rows: list[dict] = []
    for offset in range(-5, 6):
        col = _offset_col(offset)
        if col is None:
            continue
        vals = _num(d[col]) / d["adjacent_mean"]
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        if not len(vals):
            continue
        ci_low, ci_high = _bootstrap_median_ci(vals, seed=BOOTSTRAP_SEED + 1600 + offset)
        rows.append({
            "offset": int(offset),
            "median": float(np.median(vals)),
            "q25": float(np.quantile(vals, .25)),
            "q75": float(np.quantile(vals, .75)),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_conditions": int(len(vals)),
        })
    profile = pd.DataFrame(rows).sort_values("offset") if rows else pd.DataFrame()

    zero_col = _offset_col(0)
    if zero_col is None:
        return profile, {
            "n_conditions": 0,
            "median_event_over_adjacent": math.nan,
            "median_event_over_adjacent_ci95": [math.nan, math.nan],
            "wilcoxon_one_sided_p": math.nan,
        }
    event_ratio = (_num(d[zero_col]) / d["adjacent_mean"]).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    ci_low, ci_high = _bootstrap_median_ci(event_ratio, seed=BOOTSTRAP_SEED + 1701) if len(event_ratio) else (math.nan, math.nan)
    p = math.nan
    if wilcoxon is not None and len(event_ratio) and np.any(np.abs(event_ratio - 1.0) > 1e-12):
        p = float(wilcoxon(event_ratio - 1.0, alternative="greater", zero_method="wilcox").pvalue)
    stats = {
        "n_conditions": int(len(event_ratio)),
        "median_event_over_adjacent": float(np.median(event_ratio)) if len(event_ratio) else math.nan,
        "median_event_over_adjacent_ci95": [ci_low, ci_high],
        "wilcoxon_one_sided_p": p,
    }
    return profile, stats


def _strength_tertile_summary(unit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if unit.empty or not {"run_id", "direction", "strength_tertile", "concentration"}.issubset(unit.columns):
        stats = {
            "n_candidate_channels": int(len(unit)),
            "n_run_direction_groups": 0,
            "n_paired_tertile_groups": 0,
            "n_pairs_high_greater_low": 0,
            "median_high_minus_low": math.nan,
            "median_high_minus_low_ci95": [math.nan, math.nan],
            "wilcoxon_one_sided_p": math.nan,
        }
        return pd.DataFrame(), pd.DataFrame(), stats

    condition = unit.groupby(
        ["run_id", "direction", "strength_tertile"], observed=True, as_index=False
    ).agg(concentration=("concentration", "median"))

    rows: list[dict] = []
    seeds = {LOW: 301, MID: 302, HIGH: 303}
    for tertile, g in condition.groupby("strength_tertile", observed=True):
        vals = g["concentration"].to_numpy(float)
        ci_low, ci_high = _bootstrap_median_ci(vals, seed=BOOTSTRAP_SEED + seeds[str(tertile)])
        rows.append({
            "strength_tertile": str(tertile),
            "median": float(np.median(vals)),
            "q25": float(np.quantile(vals, .25)),
            "q75": float(np.quantile(vals, .75)),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_conditions": int(len(vals)),
        })
    summary = pd.DataFrame(rows)
    summary["_order"] = summary["strength_tertile"].map({LOW: 0, MID: 1, HIGH: 2})
    summary = summary.sort_values("_order").drop(columns="_order")

    pairs: list[dict] = []
    for (run_id, direction), g in unit.groupby(["run_id", "direction"]):
        lo = g.loc[g["strength_tertile"].eq(LOW), "concentration"].dropna()
        hi = g.loc[g["strength_tertile"].eq(HIGH), "concentration"].dropna()
        if len(lo) and len(hi):
            low_c = float(lo.median())
            high_c = float(hi.median())
            pairs.append({
                "run_id": run_id,
                "direction": direction,
                "low_concentration": low_c,
                "high_concentration": high_c,
                "delta_concentration": high_c - low_c,
                "n_low_channels": int(len(lo)),
                "n_high_channels": int(len(hi)),
            })
    pairs_df = pd.DataFrame(pairs)
    delta = pairs_df["delta_concentration"].to_numpy(float) if len(pairs_df) else np.array([])
    ci_low, ci_high = _bootstrap_median_ci(delta, seed=BOOTSTRAP_SEED + 401) if len(delta) else (math.nan, math.nan)
    p = math.nan
    if wilcoxon is not None and len(delta) and np.any(delta != 0):
        p = float(wilcoxon(delta, alternative="greater", zero_method="wilcox").pvalue)

    stats = {
        "n_candidate_channels": int(len(unit)),
        "n_run_direction_groups": int(unit[["run_id", "direction"]].drop_duplicates().shape[0]),
        "n_paired_tertile_groups": int(len(pairs_df)),
        "n_pairs_high_greater_low": int((pairs_df.get("delta_concentration", pd.Series(dtype=float)) > 0).sum()),
        "median_high_minus_low": float(np.median(delta)) if len(delta) else math.nan,
        "median_high_minus_low_ci95": [ci_low, ci_high],
        "wilcoxon_one_sided_p": p,
    }
    return summary, pairs_df, stats


def _arithmetic_competence(margin_joined: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {
        "task", "phase", "direction", "margin_type", "n_doses", "competence",
        "median_max_step_concentration_ratio", "directional_causal_reach",
    }
    if margin_joined.empty or not required.issubset(margin_joined.columns):
        return pd.DataFrame(), {
            "n": 0,
            "spearman_rho": math.nan,
            "spearman_exact_two_sided_p": math.nan,
            "leave_one_out_rho_min": math.nan,
            "leave_one_out_rho_max": math.nan,
            "min_directional_reach": math.nan,
            "max_directional_reach": math.nan,
        }

    d = margin_joined.loc[
        margin_joined["task"].astype(str).eq("Arithmetic")
        & margin_joined["phase"].astype(str).eq("Out")
        & margin_joined["direction"].astype(str).eq("1to0")
        & margin_joined["margin_type"].astype(str).eq(PRIMARY_MARGIN)
        & _num(margin_joined["n_doses"]).eq(11)
    ].copy()
    d["competence"] = _num(d["competence"])
    d["concentration"] = _num(d["median_max_step_concentration_ratio"])
    d["directional_reach"] = _num(d["directional_causal_reach"])
    d = d.loc[d["competence"].gt(0) & d["concentration"].notna()].sort_values("competence")

    rho, p = _exact_spearman_permutation(
        d["competence"].to_numpy(float), d["concentration"].to_numpy(float)
    ) if len(d) >= 3 else (math.nan, math.nan)
    loo: list[float] = []
    if spearmanr is not None and len(d) >= 4:
        for idx in d.index:
            sub = d.drop(index=idx)
            loo.append(float(spearmanr(sub["competence"], sub["concentration"]).statistic))
    return d, {
        "n": int(len(d)),
        "spearman_rho": rho,
        "spearman_exact_two_sided_p": p,
        "leave_one_out_rho_min": float(np.nanmin(loo)) if loo else math.nan,
        "leave_one_out_rho_max": float(np.nanmax(loo)) if loo else math.nan,
        "min_directional_reach": float(d["directional_reach"].min()) if len(d) else math.nan,
        "max_directional_reach": float(d["directional_reach"].max()) if len(d) else math.nan,
    }


def _transient_affine_null(units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "run_id", "direction", "support_kind", "natural_state_reproduction_rate",
        "full_dose_support_reproduction_rate", "transient_flip_rate", "n_examples",
    }
    if units.empty or not required.issubset(units.columns):
        return pd.DataFrame(), pd.DataFrame()

    d = units.loc[units["support_kind"].astype(str).eq("same_agonist_nonflip")].copy()
    for col in [
        "natural_state_reproduction_rate",
        "full_dose_support_reproduction_rate",
        "transient_flip_rate",
        "n_examples",
    ]:
        d[col] = _num(d[col])
    d = d.loc[
        d["natural_state_reproduction_rate"].ge(1 - 1e-12)
        & d["full_dose_support_reproduction_rate"].ge(1 - 1e-12)
        & d["transient_flip_rate"].notna()
        & d["n_examples"].gt(0)
    ].copy()
    d["transient_count"] = d["transient_flip_rate"] * d["n_examples"]

    rows: list[dict] = []
    for i, direction in enumerate(["0to1", "1to0", "all"]):
        g = d if direction == "all" else d.loc[d["direction"].astype(str).eq(direction)]
        n = int(g["n_examples"].sum())
        k = int(round(g["transient_count"].sum()))
        ci_low, ci_high = _cluster_weighted_rate_ci(g, seed=BOOTSTRAP_SEED + 501 + i)
        rows.append({
            "group": {"0to1": "0→1", "1to0": "1→0", "all": "Overall"}[direction],
            "rate_percent": 100 * k / n if n else math.nan,
            "ci_low_percent": 100 * ci_low,
            "ci_high_percent": 100 * ci_high,
            "count": k,
            "n": n,
            "n_conditions": int(g[["run_id", "direction"]].drop_duplicates().shape[0]),
        })
    return pd.DataFrame(rows), d


def _plot_event_profile(profile: pd.DataFrame, stats: dict, path: Path) -> None:
    if profile.empty:
        path.unlink(missing_ok=True)
        return
    fig, ax = plt.subplots(figsize=(6.45, 4.1))
    x = profile["offset"].to_numpy(float)
    med = 100 * profile["median"].to_numpy(float)
    q25 = 100 * profile["q25"].to_numpy(float)
    q75 = 100 * profile["q75"].to_numpy(float)
    lo = 100 * profile["ci_low"].to_numpy(float)
    hi = 100 * profile["ci_high"].to_numpy(float)
    for xi, a, b, c, d in zip(x, q25, q75, lo, hi):
        ax.vlines(xi, a, b, linewidth=.8, alpha=.65)
        ax.vlines(xi, c, d, linewidth=3.0)
    ax.plot(x, med, marker="o", linewidth=1.8)
    zero = np.flatnonzero(profile["offset"].to_numpy() == 0)
    if len(zero):
        ax.scatter([0], [med[int(zero[0])]], s=90, zorder=5)
    ax.axvline(0, linestyle=":", linewidth=1.1)
    ax.text(.03, .97, "thin whisker = IQR\nthick whisker = 95% condition-bootstrap CI",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.3)
    ci = stats.get("median_event_minus_adjacent_ci95", [math.nan, math.nan])
    ax.text(.98, .97,
            f"EVENT − adjacent bins\nmedian +{100*stats.get('median_event_minus_adjacent', math.nan):.1f} pp "
            f"[{100*ci[0]:.1f}, {100*ci[1]:.1f}]\n"
            f"paired Wilcoxon {_format_p(stats.get('wilcoxon_one_sided_p', math.nan))}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.3)
    ax.set_xticks([-4, -2, 0, 2, 4], ["−4", "−2", "EVENT", "+2", "+4"])
    ax.set_xlabel("Largest margin-change interval relative to behavioral crossing")
    ax.set_ylabel("Condition-median trajectories (%)")
    ax.set_ylim(0, max(32.0, float(np.nanmax(q75)) * 1.18))
    _clean(ax)
    _save(fig, path)


def _plot_local_enrichment_profile(profile: pd.DataFrame, stats: dict, path: Path) -> None:
    if profile.empty:
        path.unlink(missing_ok=True)
        return
    fig, ax = plt.subplots(figsize=(6.45, 4.1))
    x = profile["offset"].to_numpy(float)
    med = profile["median"].to_numpy(float)
    q25 = profile["q25"].to_numpy(float)
    q75 = profile["q75"].to_numpy(float)
    lo = profile["ci_low"].to_numpy(float)
    hi = profile["ci_high"].to_numpy(float)
    for xi, a, b, c, d in zip(x, q25, q75, lo, hi):
        ax.vlines(xi, a, b, linewidth=.8, alpha=.65)
        ax.vlines(xi, c, d, linewidth=3.0)
    ax.plot(x, med, marker="o", linewidth=1.8)
    zero = np.flatnonzero(profile["offset"].to_numpy() == 0)
    if len(zero):
        ax.scatter([0], [med[int(zero[0])]], s=90, zorder=5)
    ax.axvline(0, linestyle=":", linewidth=1.1)
    ax.axhline(1.0, linestyle=":", linewidth=1.1)
    ax.text(.03, .97, "thin whisker = IQR\nthick whisker = 95% condition-bootstrap CI",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.3)
    ci = stats.get("median_event_over_adjacent_ci95", [math.nan, math.nan])
    ax.text(.98, .97,
            f"EVENT / adjacent bins\nmedian {stats.get('median_event_over_adjacent', math.nan):.2f}× "
            f"[{ci[0]:.2f}, {ci[1]:.2f}]\n"
            f"paired Wilcoxon {_format_p(stats.get('wilcoxon_one_sided_p', math.nan))}; n={stats.get('n_conditions', 0)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.3)
    ax.set_xticks([-4, -2, 0, 2, 4], ["−4", "−2", "EVENT", "+2", "+4"])
    ax.set_xlabel("Largest margin-change interval relative to behavioral crossing")
    ax.set_ylabel("Local enrichment relative to adjacent offsets\n(adjacent offsets −1/+1 mean = 1)")
    ax.set_ylim(0, max(3.2, float(np.nanmax(q75)) * 1.18))
    _clean(ax)
    _save(fig, path)

def _plot_strength_tertiles(summary: pd.DataFrame, stats: dict, path: Path) -> None:
    if summary.empty:
        path.unlink(missing_ok=True)
        return
    fig, ax = plt.subplots(figsize=(5.0, 4.1))
    x = np.arange(len(summary))
    med = summary["median"].to_numpy(float)
    q25 = summary["q25"].to_numpy(float)
    q75 = summary["q75"].to_numpy(float)
    lo = summary["ci_low"].to_numpy(float)
    hi = summary["ci_high"].to_numpy(float)
    for xi, a, b, c, d in zip(x, q25, q75, lo, hi):
        ax.vlines(xi, a, b, linewidth=.8, alpha=.65)
        ax.vlines(xi, c, d, linewidth=3.0)
    ax.plot(x, med, marker="o", linewidth=1.8)
    for xi, value in zip(x, med):
        ax.text(xi, value + .10, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, linestyle=":", linewidth=1.1)
    ax.text(.98, 1.03, "affine path C=1", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7.5)
    ci = stats.get("median_high_minus_low_ci95", [math.nan, math.nan])
    ax.text(.03, .97,
            f"High − Low: median ΔC={stats.get('median_high_minus_low', math.nan):+.2f}\n"
            f"95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]\n"
            f"{stats.get('n_pairs_high_greater_low', 0)}/{stats.get('n_paired_tertile_groups', 0)} > 0; "
            f"paired Wilcoxon {_format_p(stats.get('wilcoxon_one_sided_p', math.nan))}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.3)
    ax.text(.97, .08, "thin = IQR\nthick = 95% CI", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.1)
    ax.set_xticks(x, [str(v) for v in summary["strength_tertile"]])
    ax.set_xlabel("Held-out singleton causal strength\n(within-run/direction tertile)")
    ax.set_ylabel("Margin-change concentration (C)")
    ax.set_ylim(.8, max(3.6, float(np.nanmax(q75)) * 1.18))
    _clean(ax)
    _save(fig, path)


def _plot_strength_pairs(pairs: pd.DataFrame, margin_joined: pd.DataFrame, stats: dict, path: Path) -> pd.DataFrame:
    if pairs.empty:
        path.unlink(missing_ok=True)
        return pairs
    join_keys = ["run_id", "direction"]
    if set(join_keys).issubset(margin_joined.columns):
        meta_cols = join_keys + [c for c in ["task", "model", "phase"] if c in margin_joined.columns]
        meta = margin_joined[meta_cols].drop_duplicates(subset=join_keys)
        d = pairs.merge(meta, on=join_keys, how="left")
    else:
        d = pairs.copy()
    d["direction_label"] = d["direction"].map({"0to1": "0→1", "1to0": "1→0"}).fillna(d["direction"].astype(str))
    model = d["model"].astype(str) if "model" in d.columns else d["run_id"].astype(str)
    task = d["task"].astype(str) if "task" in d.columns else pd.Series(["run"] * len(d), index=d.index)
    d["label"] = model + " · " + task + " · " + d["direction_label"].astype(str)
    d = d.sort_values("delta_concentration", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.05))
    y = np.arange(len(d))
    for i, row in d.iterrows():
        ax.annotate("", xy=(row["high_concentration"], i), xytext=(row["low_concentration"], i),
                    arrowprops={"arrowstyle": "->", "linewidth": 1.35})
    ax.scatter(d["low_concentration"], y, marker="o", s=34, label="Low-strength tertile", zorder=3)
    ax.scatter(d["high_concentration"], y, marker=">", s=42, label="High-strength tertile", zorder=3)
    ax.axvline(1.0, linestyle=":", linewidth=1.1)
    ax.set_yticks(y, d["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Margin-change concentration (C; affine path = 1)")
    ci = stats.get("median_high_minus_low_ci95", [math.nan, math.nan])
    ax.text(.99, .03,
            f"median ΔC={stats.get('median_high_minus_low', math.nan):+.2f} "
            f"[95% CI {ci[0]:+.2f}, {ci[1]:+.2f}]\n"
            f"one-sided paired Wilcoxon {_format_p(stats.get('wilcoxon_one_sided_p', math.nan))}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.3)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _clean(ax, xgrid=True)
    _save(fig, path)
    return d


def _plot_arithmetic_competence(d: pd.DataFrame, stats: dict, path: Path) -> None:
    if d.empty:
        path.unlink(missing_ok=True)
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    x = 100 * d["competence"].to_numpy(float)
    y = d["concentration"].to_numpy(float)
    ax.plot(x, y, marker="o", linewidth=1.7)
    for _, row in d.iterrows():
        ax.annotate(str(row["model"]), (100 * float(row["competence"]), float(row["concentration"])),
                    xytext=(4, 5), textcoords="offset points", fontsize=7.5)
    ax.axhline(1.0, linestyle=":", linewidth=1.1)
    ax.set_xscale("log")
    ax.set_xlabel("Arithmetic competence (%)")
    ax.set_ylabel("Margin-change concentration (C)")
    ax.text(.03, .97,
            f"Spearman ρ={stats['spearman_rho']:.2f}; exact two-sided {_format_p(stats['spearman_exact_two_sided_p'])}; n={stats['n']}\n"
            f"leave-one-out ρ range [{stats['leave_one_out_rho_min']:.2f}, {stats['leave_one_out_rho_max']:.2f}]\n"
            f"directional reach {stats['min_directional_reach']:.2f}–{stats['max_directional_reach']:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.3)
    _clean(ax)
    _save(fig, path)


def _plot_transient_affine_null(d: pd.DataFrame, path: Path) -> None:
    if d.empty:
        path.unlink(missing_ok=True)
        return
    fig, ax = plt.subplots(figsize=(6.05, 3.9))
    y = np.arange(len(d))
    rate = d["rate_percent"].to_numpy(float)
    lo = d["ci_low_percent"].to_numpy(float)
    hi = d["ci_high_percent"].to_numpy(float)
    ax.errorbar(rate, y, xerr=np.vstack([rate - lo, hi - rate]), fmt="o", capsize=3.2, linewidth=1.7)
    for yi, row in zip(y, d.to_dict("records")):
        ax.text(row["ci_high_percent"] + .35, yi,
                f"{row['rate_percent']:.1f}% ({row['count']}/{row['n']})",
                va="center", fontsize=8)
    ax.axvline(0, linestyle=":", linewidth=1.1)
    ax.text(.01, .04, "affine endpoint-preserving null: 0%",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.3)
    ax.set_yticks(y, d["group"])
    ax.set_xlabel("Interior flip-and-return rate (%) with condition-cluster 95% CI")
    ax.set_ylabel("")
    ax.set_xlim(0, max(24.0, float(np.nanmax(hi)) * 1.28))
    _clean(ax, xgrid=True)
    _save(fig, path)


def _plot_main_pair(
    profile: pd.DataFrame,
    event_stats: dict,
    local_profile: pd.DataFrame,
    local_stats: dict,
    strength: pd.DataFrame,
    strength_stats: dict,
    path: Path,
) -> None:
    """Analysis diagnostic triptych: raw localization, local enrichment, and strength sharpening."""
    if profile.empty or local_profile.empty or strength.empty:
        path.unlink(missing_ok=True)
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.75))

    # Original/raw event localization.
    ax = axes[0]
    x = profile["offset"].to_numpy(float)
    med = 100 * profile["median"].to_numpy(float)
    q25 = 100 * profile["q25"].to_numpy(float)
    q75 = 100 * profile["q75"].to_numpy(float)
    lo = 100 * profile["ci_low"].to_numpy(float)
    hi = 100 * profile["ci_high"].to_numpy(float)
    for xi, a, b, c, d in zip(x, q25, q75, lo, hi):
        ax.vlines(xi, a, b, linewidth=.7, alpha=.65)
        ax.vlines(xi, c, d, linewidth=2.6)
    ax.plot(x, med, marker="o", linewidth=1.65)
    zero = np.flatnonzero(profile["offset"].to_numpy() == 0)
    if len(zero):
        ax.scatter([0], [med[int(zero[0])]], s=70, zorder=5)
    ax.axvline(0, linestyle=":", linewidth=1.0)
    ax.set_xticks([-4, -2, 0, 2, 4], ["−4", "−2", "EVENT", "+2", "+4"])
    ax.set_xlabel("Largest margin-change interval\nrelative to behavioral crossing")
    ax.set_ylabel("Condition-median trajectories (%)")
    ci = event_stats.get("median_event_minus_adjacent_ci95", [math.nan, math.nan])
    ax.text(.97, .96,
            f"EVENT−adjacent +{100*event_stats.get('median_event_minus_adjacent', math.nan):.1f} pp\n"
            f"95% CI [{100*ci[0]:.1f}, {100*ci[1]:.1f}]\n"
            f"paired Wilcoxon {_format_p(event_stats.get('wilcoxon_one_sided_p', math.nan))}; n={event_stats.get('n_conditions', 0)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.2)
    ax.set_ylim(0, max(32.0, float(np.nanmax(q75)) * 1.18))
    _clean(ax)

    # Same event localization normalized to the two adjacent offsets.
    ax = axes[1]
    x = local_profile["offset"].to_numpy(float)
    med = local_profile["median"].to_numpy(float)
    q25 = local_profile["q25"].to_numpy(float)
    q75 = local_profile["q75"].to_numpy(float)
    lo = local_profile["ci_low"].to_numpy(float)
    hi = local_profile["ci_high"].to_numpy(float)
    for xi, a, b, c, d in zip(x, q25, q75, lo, hi):
        ax.vlines(xi, a, b, linewidth=.7, alpha=.65)
        ax.vlines(xi, c, d, linewidth=2.6)
    ax.plot(x, med, marker="o", linewidth=1.65)
    zero = np.flatnonzero(local_profile["offset"].to_numpy() == 0)
    if len(zero):
        ax.scatter([0], [med[int(zero[0])]], s=70, zorder=5)
    ax.axvline(0, linestyle=":", linewidth=1.0)
    ax.axhline(1.0, linestyle=":", linewidth=1.0)
    ax.set_xticks([-4, -2, 0, 2, 4], ["−4", "−2", "EVENT", "+2", "+4"])
    ax.set_xlabel("Largest margin-change interval\nrelative to behavioral crossing")
    ax.set_ylabel("Local enrichment relative to adjacent offsets\n(adjacent offsets −1/+1 mean = 1)")
    ci = local_stats.get("median_event_over_adjacent_ci95", [math.nan, math.nan])
    ax.text(.97, .96,
            f"EVENT / adjacent={local_stats.get('median_event_over_adjacent', math.nan):.2f}×\n"
            f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}]\n"
            f"paired Wilcoxon {_format_p(local_stats.get('wilcoxon_one_sided_p', math.nan))}; n={local_stats.get('n_conditions', 0)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.2)
    ax.set_ylim(0, max(3.2, float(np.nanmax(q75)) * 1.18))
    _clean(ax)

    # Strength sharpening.
    ax = axes[2]
    xx = np.arange(len(strength))
    smed = strength["median"].to_numpy(float)
    sq25 = strength["q25"].to_numpy(float)
    sq75 = strength["q75"].to_numpy(float)
    slo = strength["ci_low"].to_numpy(float)
    shi = strength["ci_high"].to_numpy(float)
    for xi, a, b, c, d in zip(xx, sq25, sq75, slo, shi):
        ax.vlines(xi, a, b, linewidth=.7, alpha=.65)
        ax.vlines(xi, c, d, linewidth=2.6)
    ax.plot(xx, smed, marker="o", linewidth=1.65)
    ax.axhline(1.0, linestyle=":", linewidth=1.0)
    ax.set_xticks(xx, [str(v) for v in strength["strength_tertile"]])
    ax.set_xlabel("Held-out singleton causal strength\n(within-run/direction tertile)")
    ax.set_ylabel("Margin-change concentration (C)")
    ci = strength_stats.get("median_high_minus_low_ci95", [math.nan, math.nan])
    ax.text(.03, .96,
            f"High−Low ΔC={strength_stats.get('median_high_minus_low', math.nan):+.2f}\n"
            f"95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]\n"
            f"paired Wilcoxon {_format_p(strength_stats.get('wilcoxon_one_sided_p', math.nan))}; n={strength_stats.get('n_paired_tertile_groups', 0)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.2)
    ax.set_ylim(.8, max(3.6, float(np.nanmax(sq75)) * 1.18))
    _clean(ax)

    _save(fig, path)


def main() -> None:
    args = parse_args()
    graded = Path(args.graded_dir).expanduser().resolve()
    spiking = Path(args.spiking_dir).expanduser().resolve()
    paper = Path(args.paper_figures_dir).expanduser().resolve() if args.paper_figures_dir else None

    input_paths = {
        "margin_examples": graded / "graded_margin_all_examples.csv",
        "margin_joined": graded / "graded_margin_competence_reach_joined.csv",
        "units": graded / "graded_agonist_all_units.csv",
        "scores": spiking / "unit_primary_spiking_scores.csv",
    }
    loaded = {name: _optional_csv(path) for name, path in input_paths.items()}
    missing_inputs = [name for name, frame in loaded.items() if frame is None]

    margin_examples = loaded["margin_examples"] if loaded["margin_examples"] is not None else pd.DataFrame()
    margin_joined = loaded["margin_joined"] if loaded["margin_joined"] is not None else pd.DataFrame()
    units = loaded["units"] if loaded["units"] is not None else pd.DataFrame()
    scores = loaded["scores"] if loaded["scores"] is not None else pd.DataFrame()

    examples, unit = _strict_margin_population(margin_examples, scores)
    profile, event_pairs, event_stats, n_persistent_examples = _event_profile(examples)
    local_profile, local_stats = _local_enrichment_profile(event_pairs)
    strength, pairs, strength_stats = _strength_tertile_summary(unit)
    arithmetic, arithmetic_stats = _arithmetic_competence(margin_joined)
    transients, _ = _transient_affine_null(units)

    p_family = {
        "event_localization": event_stats.get("wilcoxon_one_sided_p", math.nan),
        "strength_sharpening": strength_stats.get("wilcoxon_one_sided_p", math.nan),
        "arithmetic_competence": arithmetic_stats.get("spearman_exact_two_sided_p", math.nan),
    }
    holm = _holm_adjust(p_family)

    profile.to_csv(graded / "population_event_localization.csv", index=False)
    local_profile.to_csv(graded / "population_event_local_enrichment.csv", index=False)
    event_pairs.to_csv(graded / "population_event_localization_condition_pairs.csv", index=False)
    strength.to_csv(graded / "strength_concentration_tertile_summary.csv", index=False)
    arithmetic.to_csv(graded / "arithmetic_1to0_competence_concentration.csv", index=False)
    transients.to_csv(graded / "endpoint_preserving_transient_events.csv", index=False)

    pairs_for_plot = _plot_strength_pairs(
        pairs, margin_joined, strength_stats, graded / "strength_concentration_paired.pdf"
    )
    pairs_for_plot.to_csv(graded / "strength_concentration_pairs.csv", index=False)

    _plot_event_profile(profile, event_stats, graded / "population_event_localization.pdf")
    _plot_local_enrichment_profile(local_profile, local_stats, graded / "population_event_local_enrichment.pdf")
    _plot_strength_tertiles(strength, strength_stats, graded / "strength_sharpens_concentration.pdf")
    _plot_arithmetic_competence(arithmetic, arithmetic_stats, graded / "arithmetic_1to0_competence_concentration.pdf")
    _plot_transient_affine_null(transients, graded / "endpoint_preserving_transient_events.pdf")
    _plot_main_pair(profile, event_stats, local_profile, local_stats, strength, strength_stats, graded / "population_spiking_story_main_pair.pdf")

    statistics = pd.DataFrame([
        {
            "analysis": "event_vs_adjacent_bins",
            "estimate": event_stats.get("median_event_minus_adjacent", math.nan),
            "ci_low": event_stats.get("median_event_minus_adjacent_ci95", [math.nan, math.nan])[0],
            "ci_high": event_stats.get("median_event_minus_adjacent_ci95", [math.nan, math.nan])[1],
            "p_value": p_family["event_localization"],
            "holm_adjusted_p": holm["event_localization"],
            "test": "one-sided paired Wilcoxon over run/direction conditions",
            "n": event_stats.get("n_conditions", 0),
        },
        {
            "analysis": "high_minus_low_strength_C",
            "estimate": strength_stats.get("median_high_minus_low", math.nan),
            "ci_low": strength_stats.get("median_high_minus_low_ci95", [math.nan, math.nan])[0],
            "ci_high": strength_stats.get("median_high_minus_low_ci95", [math.nan, math.nan])[1],
            "p_value": p_family["strength_sharpening"],
            "holm_adjusted_p": holm["strength_sharpening"],
            "test": "one-sided paired Wilcoxon over eligible run/direction groups",
            "n": strength_stats.get("n_paired_tertile_groups", 0),
        },
        {
            "analysis": "arithmetic_competence_spearman",
            "estimate": arithmetic_stats.get("spearman_rho", math.nan),
            "ci_low": math.nan,
            "ci_high": math.nan,
            "p_value": p_family["arithmetic_competence"],
            "holm_adjusted_p": holm["arithmetic_competence"],
            "test": "exact two-sided Spearman permutation test",
            "n": arithmetic_stats.get("n", 0),
        },
    ])
    statistics.to_csv(graded / "population_spiking_story_statistics.csv", index=False)

    summary = {
        "schema": "rq3-population-spiking-story-v2",
        "uncertainty": {
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_unit": "run/direction condition",
            "bootstrap_interval": "percentile 95%",
            "multiple_testing": "Holm adjustment across the three story-level inferential tests",
        },
        "input_availability": {
            name: {"available": frame is not None, "path": str(input_paths[name])}
            for name, frame in loaded.items()
        },
        "missing_inputs": missing_inputs,
        "partial_story": bool(missing_inputs),
        "continuous_margin_candidate_channels": int(len(unit)),
        "continuous_margin_run_direction_groups": int(unit[["run_id", "direction"]].drop_duplicates().shape[0]) if {"run_id", "direction"}.issubset(unit.columns) else 0,
        "persistent_endpoint_reproduced_examples": n_persistent_examples,
        "event_localization": event_stats,
        "event_local_enrichment": local_stats,
        "strength": strength_stats,
        "arithmetic_1to0": arithmetic_stats,
        "holm_adjusted_p": holm,
        "endpoint_preserving_transient_events": transients.to_dict(orient="records"),
        "interpretation": (
            "Event alignment is descriptive, while the event-vs-adjacent paired test quantifies localization without treating examples as independent. "
            "Held-out singleton causal-strength tertiles are formed within each run/direction before condition-level inference. "
            "The Arithmetic competence association is task/direction-specific and uses an exact permutation p-value at n=6. "
            "Endpoint-preserving transient interior events contradict the simple one-dimensional affine-margin null for those trajectories, but do not identify a unique discontinuous or biological mechanism."
        ),
    }
    (graded / "population_spiking_story_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )

    if paper is not None:
        paper.mkdir(parents=True, exist_ok=True)
        # Figure 4e is assembled by stage11 after cross-sweep temporal validation
        # is available: event localization, strength sharpening, and reciprocal validation.
        _plot_event_profile(profile, event_stats, paper / "fig4s10_population_event_localization.pdf")
        _plot_strength_pairs(pairs, margin_joined, strength_stats, paper / "fig4s11_strength_concentration_paired.pdf")
        _plot_arithmetic_competence(arithmetic, arithmetic_stats, paper / "fig4s12_arithmetic_competence_concentration.pdf")
        _plot_transient_affine_null(transients, paper / "fig4s13_affine_null_transient_events.pdf")

    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
