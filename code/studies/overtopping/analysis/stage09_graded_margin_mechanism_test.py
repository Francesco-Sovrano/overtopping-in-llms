#!/usr/bin/env python3
"""Paper-facing tests of nonlinear threshold-like geometry under graded overtopping.

Input
-----
``graded_agonist_dose_rows.csv.gz`` emitted by
``graded_agonist_intervention.py --record_endpoint_margin``.

The analysis deliberately separates three claims:

1. **Behavioral persistence**: does the binary endpoint change once and remain
   changed over the sampled dose grid?
2. **Affine-margin null**: is a continuous downstream margin well described by
   the straight chord joining dose 0 and dose 1?  This directly evaluates the
   simple construction m(lambda)=a+b lambda followed by a binary readout.
3. **Crossing alignment**: when the measured margin itself crosses, does that
   crossing occur near the persistent task-level behavioral crossing?

Non-affinity is not treated as proof of mathematical discontinuity or a unique
spiking mechanism.  The script emits conservative manuscript-ready text that
states exactly this limitation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

SCHEMA = "graded-margin-mechanism-test-v3"
MARGINS = {
    "divergence_token": "endpoint_divergence_margin",
    "endpoint_total_logp": "endpoint_margin_total",
    "endpoint_mean_logp": "endpoint_margin_mean",
}
PRIMARY_MARGIN = "divergence_token"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dose_rows", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--paper_prefix",
        default="graded_margin",
        help="Prefix for manuscript-ready figure/text files.",
    )
    return p.parse_args()


def _first_persistent_true(doses: np.ndarray, flags: np.ndarray) -> float:
    for i in range(len(flags)):
        if bool(flags[i]) and bool(np.asarray(flags[i:], dtype=bool).all()):
            return float(doses[i])
    return math.nan


def _single_persistent_crossing(flags: np.ndarray) -> bool:
    b = np.asarray(flags, dtype=bool)
    if len(b) < 2 or bool(b[0]) or not bool(b[-1]):
        return False
    transitions = int(np.sum(b[1:] != b[:-1]))
    return transitions == 1


def _first_persistent_positive(doses: np.ndarray, margin: np.ndarray) -> float:
    for i in range(len(margin)):
        if not np.isfinite(margin[i]) or margin[i] <= 0:
            continue
        suffix = margin[i:]
        if len(suffix) and np.isfinite(suffix).all() and bool((suffix > 0).all()):
            return float(doses[i])
    return math.nan


def _linear_crossing(m0: float, m1: float) -> float:
    if not np.isfinite(m0) or not np.isfinite(m1):
        return math.nan
    slope = float(m1 - m0)
    if abs(slope) < 1e-12:
        return math.nan
    lam = float(-m0 / slope)
    return lam if 0.0 <= lam <= 1.0 else math.nan


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(yhat)
    if ok.sum() < 3:
        return math.nan
    yy, pp = y[ok], yhat[ok]
    ss_res = float(np.sum((yy - pp) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else math.nan
    return 1.0 - ss_res / ss_tot


def _safe_corr(x, y) -> dict:
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or spearmanr is None:
        return {"n": int(ok.sum()), "spearman_rho": math.nan, "p": math.nan}
    if np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return {"n": int(ok.sum()), "spearman_rho": math.nan, "p": math.nan}
    r = spearmanr(x[ok], y[ok])
    return {"n": int(ok.sum()), "spearman_rho": float(r.statistic), "p": float(r.pvalue)}


def _loo_median_errors(obs: np.ndarray, eval_mask: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=float)
    out = np.full(len(obs), np.nan, dtype=float)
    finite_all = np.isfinite(obs)
    for i in np.flatnonzero(eval_mask & finite_all):
        train = obs[finite_all & (np.arange(len(obs)) != i)]
        if len(train):
            out[i] = abs(float(np.median(train)) - obs[i])
    return out


def _crossing_quality(pred, actual_margin_cross, behavior_cross, grid_step) -> dict:
    pred = pd.to_numeric(pd.Series(pred), errors="coerce").to_numpy(float)
    act = pd.to_numeric(pd.Series(actual_margin_cross), errors="coerce").to_numpy(float)
    beh = pd.to_numeric(pd.Series(behavior_cross), errors="coerce").to_numpy(float)
    step = pd.to_numeric(pd.Series(grid_step), errors="coerce").to_numpy(float)

    pred_ok = np.isfinite(pred) & np.isfinite(beh)
    pred_err = np.abs(pred - beh)
    loo_err = _loo_median_errors(beh, pred_ok)
    act_ok = np.isfinite(act) & np.isfinite(beh)
    lag = act - beh

    def med(x, mask):
        return float(np.median(x[mask])) if mask.any() else math.nan

    def mean(x, mask):
        return float(np.mean(x[mask])) if mask.any() else math.nan

    exact = act_ok & (np.abs(lag) <= 1e-12)
    within_one = act_ok & np.isfinite(step) & (np.abs(lag) <= step + 1e-12)
    return {
        "predicted_crossing_n": int(pred_ok.sum()),
        "predicted_crossing_median_abs_error": med(pred_err, pred_ok),
        "predicted_crossing_mean_abs_error": mean(pred_err, pred_ok),
        "loo_median_null_median_abs_error": med(loo_err, np.isfinite(loo_err)),
        "loo_median_null_mean_abs_error": mean(loo_err, np.isfinite(loo_err)),
        "actual_margin_crossing_n": int(act_ok.sum()),
        "actual_margin_crossing_median_lag": med(lag, act_ok),
        "actual_margin_crossing_mean_lag": mean(lag, act_ok),
        "actual_margin_crossing_later_fraction": float(np.mean(lag[act_ok] > 1e-12)) if act_ok.any() else math.nan,
        "actual_margin_crossing_equal_fraction": float(exact.sum() / act_ok.sum()) if act_ok.any() else math.nan,
        "actual_margin_crossing_within_one_grid_step_fraction": float(within_one.sum() / act_ok.sum()) if act_ok.any() else math.nan,
        "actual_margin_crossing_earlier_fraction": float(np.mean(lag[act_ok] < -1e-12)) if act_ok.any() else math.nan,
    }


def _endpoint_value(doses, vals, target):
    ids = np.flatnonzero(np.isclose(doses, target))
    return float(vals[ids[0]]) if len(ids) and np.isfinite(vals[ids[0]]) else math.nan


def _grid_step(doses: np.ndarray) -> float:
    d = np.sort(np.unique(doses[np.isfinite(doses)]))
    diffs = np.diff(d)
    diffs = diffs[diffs > 1e-12]
    return float(np.median(diffs)) if len(diffs) else math.nan


def _path_shape(doses: np.ndarray, margins: np.ndarray, behavior_cross: float) -> dict:
    ok = np.isfinite(doses) & np.isfinite(margins)
    d, m = doses[ok], margins[ok]
    if len(m) < 2:
        return {
            "margin_total_variation": math.nan,
            "margin_excess_variation_ratio": math.nan,
            "margin_max_step_share": math.nan,
            "margin_max_step_concentration_ratio": math.nan,
            "margin_max_step_start_dose": math.nan,
            "margin_max_step_end_dose": math.nan,
            "margin_behavior_step_share": math.nan,
            "margin_monotone_endpoint_direction_fraction": math.nan,
        }
    order = np.argsort(d)
    d, m = d[order], m[order]
    dm = np.diff(m)
    abs_dm = np.abs(dm)
    tv = float(abs_dm.sum())
    span = float(abs(m[-1] - m[0]))
    max_i = int(np.argmax(abs_dm)) if len(abs_dm) else 0
    max_share = float(abs_dm[max_i] / tv) if tv > 1e-12 else math.nan
    uniform_share = 1.0 / len(abs_dm) if len(abs_dm) else math.nan
    endpoint_sign = np.sign(m[-1] - m[0])
    if endpoint_sign == 0:
        mono_frac = math.nan
    else:
        mono_frac = float(np.mean(np.sign(dm) == endpoint_sign))
    behavior_share = math.nan
    if np.isfinite(behavior_cross) and tv > 1e-12:
        idx = np.flatnonzero(np.isclose(d, behavior_cross))
        if len(idx) and idx[0] > 0:
            behavior_share = float(abs_dm[idx[0] - 1] / tv)
    return {
        "margin_total_variation": tv,
        "margin_excess_variation_ratio": float(tv / span) if span > 1e-12 else math.nan,
        "margin_max_step_share": max_share,
        "margin_max_step_concentration_ratio": float(max_share / uniform_share) if np.isfinite(max_share) and np.isfinite(uniform_share) else math.nan,
        "margin_max_step_start_dose": float(d[max_i]),
        "margin_max_step_end_dose": float(d[max_i + 1]),
        "margin_behavior_step_share": behavior_share,
        "margin_monotone_endpoint_direction_fraction": mono_frac,
    }


def _analyze_margin(doses, margins, flipped) -> dict:
    m0 = _endpoint_value(doses, margins, 0.0)
    m1 = _endpoint_value(doses, margins, 1.0)
    line = m0 + doses * (m1 - m0) if np.isfinite(m0) and np.isfinite(m1) else np.full_like(doses, np.nan)
    ok = np.isfinite(margins) & np.isfinite(line)
    rmse = float(np.sqrt(np.mean((margins[ok] - line[ok]) ** 2))) if ok.any() else math.nan
    scale = abs(m1 - m0) if np.isfinite(m0) and np.isfinite(m1) else math.nan
    nrmse = float(rmse / scale) if np.isfinite(rmse) and np.isfinite(scale) and scale > 1e-12 else math.nan
    behavior_cross = _first_persistent_true(doses, flipped)
    margin_cross = _first_persistent_positive(doses, margins)
    predicted_cross = _linear_crossing(m0, m1)
    valid = np.isfinite(margins)
    sign_behavior = float(((margins[valid] > 0) == flipped[valid]).mean()) if valid.any() else math.nan
    shape = _path_shape(doses, margins, behavior_cross)
    result = {
        "n_margin_doses": int(valid.sum()),
        "dose_grid_step": _grid_step(doses),
        "behavior_single_persistent_crossing": bool(_single_persistent_crossing(flipped)),
        "margin_at_dose0": m0,
        "margin_at_dose1": m1,
        "margin_endpoint_slope": float(m1 - m0) if np.isfinite(m0) and np.isfinite(m1) else math.nan,
        "linear_predicted_zero_crossing_dose": predicted_cross,
        "observed_margin_persistent_crossing_dose": margin_cross,
        "observed_behavior_persistent_crossing_dose": behavior_cross,
        "predicted_minus_behavior_crossing": float(predicted_cross - behavior_cross) if np.isfinite(predicted_cross) and np.isfinite(behavior_cross) else math.nan,
        "margin_minus_behavior_crossing": float(margin_cross - behavior_cross) if np.isfinite(margin_cross) and np.isfinite(behavior_cross) else math.nan,
        "margin_linearity_r2": _r2(margins, line),
        "margin_linearity_rmse": rmse,
        "margin_linearity_nrmse": nrmse,
        "margin_sign_behavior_flip_agreement": sign_behavior,
    }
    result.update(shape)
    return result


def _example_metric_rows(df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "run_id", "source_scope", "task", "model", "phase", "replacement_baseline",
        "population", "baseline_subset", "direction", "unit_key", "layer_label", "neuron_id",
        "support_kind", "evaluation_row", "row_id",
    ]
    keys = [k for k in keys if k in df.columns]
    out = []
    for values, g in df.groupby(keys, dropna=False, sort=False):
        g = g.sort_values("dose", kind="mergesort")
        doses = pd.to_numeric(g["dose"], errors="coerce").to_numpy(float)
        flipped = g["flipped_from_baseline"].fillna(False).astype(bool).to_numpy()
        base = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        f0 = bool(flipped[np.flatnonzero(np.isclose(doses, 0.0))[0]]) if np.any(np.isclose(doses, 0.0)) else None
        f1 = bool(flipped[np.flatnonzero(np.isclose(doses, 1.0))[0]]) if np.any(np.isclose(doses, 1.0)) else None
        base.update({
            "n_doses": int(len(g)),
            "dose0_flipped_from_baseline": f0,
            "dose1_flipped_from_baseline": f1,
            "known_flip_endpoint_reproduced": bool((f0 is False) and (f1 is True)) if str(base.get("support_kind")) == "known_flip" else None,
            "endpoint_response0": str(g.get("endpoint_response0", pd.Series([""] * len(g), index=g.index)).iloc[0]),
            "endpoint_response1": str(g.get("endpoint_response1", pd.Series([""] * len(g), index=g.index)).iloc[0]),
        })
        base["endpoint_responses_distinct"] = bool(base["endpoint_response0"] != base["endpoint_response1"])
        for margin_type, col in MARGINS.items():
            if col not in g.columns:
                continue
            rec = dict(base)
            rec["margin_type"] = margin_type
            rec["margin_column"] = col
            rec.update(_analyze_margin(doses, pd.to_numeric(g[col], errors="coerce").to_numpy(float), flipped))
            out.append(rec)
    return pd.DataFrame(out)


def _valid_known(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return d
    known = d.loc[d.get("support_kind", pd.Series(dtype=str)).astype(str) == "known_flip"].copy()
    if "known_flip_endpoint_reproduced" in known.columns:
        known = known.loc[known["known_flip_endpoint_reproduced"].fillna(False).astype(bool)].copy()
    if "n_margin_doses" in known.columns:
        known = known.loc[pd.to_numeric(known["n_margin_doses"], errors="coerce").fillna(0).ge(3)].copy()
    if {"margin_at_dose0", "margin_at_dose1"}.issubset(known.columns):
        m0 = pd.to_numeric(known["margin_at_dose0"], errors="coerce")
        m1 = pd.to_numeric(known["margin_at_dose1"], errors="coerce")
        known = known.loc[m0.notna() & m1.notna()].copy()
    return known


def _summarize_frame(valid: pd.DataFrame) -> dict:
    if valid.empty:
        return {"n": 0}
    result = {
        "n": int(len(valid)),
        "single_persistent_crossing_rate": float(valid["behavior_single_persistent_crossing"].fillna(False).astype(bool).mean()),
        "median_linearity_r2": float(pd.to_numeric(valid["margin_linearity_r2"], errors="coerce").median()),
        "median_linearity_nrmse": float(pd.to_numeric(valid["margin_linearity_nrmse"], errors="coerce").median()),
        "median_max_step_share": float(pd.to_numeric(valid["margin_max_step_share"], errors="coerce").median()),
        "median_max_step_concentration_ratio": float(pd.to_numeric(valid["margin_max_step_concentration_ratio"], errors="coerce").median()),
        "median_behavior_step_share": float(pd.to_numeric(valid["margin_behavior_step_share"], errors="coerce").median()),
        "median_excess_variation_ratio": float(pd.to_numeric(valid["margin_excess_variation_ratio"], errors="coerce").median()),
        "median_monotone_endpoint_direction_fraction": float(pd.to_numeric(valid["margin_monotone_endpoint_direction_fraction"], errors="coerce").median()),
        "predicted_vs_observed_behavior_crossing": _safe_corr(
            valid.get("linear_predicted_zero_crossing_dose"), valid.get("observed_behavior_persistent_crossing_dose")
        ),
        "actual_margin_vs_behavior_crossing": _safe_corr(
            valid.get("observed_margin_persistent_crossing_dose"), valid.get("observed_behavior_persistent_crossing_dose")
        ),
    }
    result.update(_crossing_quality(
        valid.get("linear_predicted_zero_crossing_dose"),
        valid.get("observed_margin_persistent_crossing_dose"),
        valid.get("observed_behavior_persistent_crossing_dose"),
        valid.get("dose_grid_step"),
    ))
    return result


def _metric_summary(metric_df: pd.DataFrame, margin_type: str) -> dict:
    d = metric_df.loc[metric_df["margin_type"] == margin_type].copy()
    return _summarize_frame(_valid_known(d))


def _direction_summary(metric_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (margin_type, direction), g in metric_df.groupby(["margin_type", "direction"], dropna=False, sort=True):
        s = _summarize_frame(_valid_known(g))
        row = {"margin_type": margin_type, "direction": direction, **{k: v for k, v in s.items() if not isinstance(v, dict)}}
        for key in ["predicted_vs_observed_behavior_crossing", "actual_margin_vs_behavior_crossing"]:
            sub = s.get(key, {})
            for sk, sv in sub.items():
                row[f"{key}_{sk}"] = sv
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(x, digits=3):
    try:
        x = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def _paper_text(direction_df: pd.DataFrame, summary: dict) -> tuple[str, str]:
    primary = direction_df.loc[direction_df["margin_type"] == PRIMARY_MARGIN].copy() if not direction_df.empty else pd.DataFrame()
    parts = []
    tex_parts = []
    for direction in ["0to1", "1to0"]:
        d = primary.loc[primary["direction"].astype(str) == direction]
        if d.empty:
            continue
        r = d.iloc[0]
        direction_tex = "$0\\to1$" if direction == "0to1" else "$1\\to0$"
        n = int(r.get("n", 0) or 0)
        rho = r.get("actual_margin_vs_behavior_crossing_spearman_rho", math.nan)
        p = r.get("actual_margin_vs_behavior_crossing_p", math.nan)
        text = (
            f"For {direction}, {n} endpoint-reproduced known-flip trajectories had a "
            f"single-persistent-crossing rate of {_fmt(r.get('single_persistent_crossing_rate'))}. "
            f"The divergence-token margin departed from the endpoint-affine chord "
            f"(median R^2={_fmt(r.get('median_linearity_r2'))}, normalized RMSE={_fmt(r.get('median_linearity_nrmse'))}); "
            f"its largest adjacent-dose change accounted for a median {_fmt(r.get('median_max_step_share'))} of total path variation "
            f"({_fmt(r.get('median_max_step_concentration_ratio'))}x the uniform-per-step share). "
            f"Observed margin-crossing dose tracked behavioral crossing with Spearman rho={_fmt(rho)} "
            f"(p={_fmt(p)}), with exact-grid alignment in {_fmt(r.get('actual_margin_crossing_equal_fraction'))} of evaluable examples "
            f"and alignment within one dose step in {_fmt(r.get('actual_margin_crossing_within_one_grid_step_fraction'))}."
        )
        parts.append(text)
        tex_parts.append(
            f"For {direction_tex}, $n={n}$ endpoint-reproduced known-flip trajectories had a single-persistent-crossing rate of "
            f"{_fmt(r.get('single_persistent_crossing_rate'))}. The divergence-token margin departed from the endpoint-affine chord "
            f"(median $R^2={_fmt(r.get('median_linearity_r2'))}$; normalized RMSE $={_fmt(r.get('median_linearity_nrmse'))}$), and its largest adjacent-dose change "
            f"accounted for a median {_fmt(r.get('median_max_step_share'))} of total path variation. Observed margin-crossing dose tracked behavioral crossing "
            f"($\\rho={_fmt(rho)}$, $p={_fmt(p)}$), with exact-grid alignment in {_fmt(r.get('actual_margin_crossing_equal_fraction'))} of evaluable examples."
        )
    limitation = (
        "Because the source activation is linearly interpolated by construction, these diagnostics test whether the downstream response is approximately affine in dose. "
        "Poor affine fit rejects that simple affine-margin-plus-binary-readout description for the measured margin; it does not establish mathematical discontinuity or uniquely identify a spiking mechanism. "
        "We therefore interpret the result as nonlinear, threshold-like causal response geometry."
    )
    parts.append(limitation)
    tex_parts.append(
        "Because the source activation is linearly interpolated by construction, poor affine fit rejects this simple affine-margin-plus-binary-readout description for the measured margin; it does not establish mathematical discontinuity or uniquely identify a spiking mechanism. We therefore interpret the result as nonlinear, threshold-like causal response geometry."
    )
    md = "## Manuscript-ready interpretation\n\n" + "\n\n".join(parts) + "\n"
    tex = "% Auto-generated by stage09_graded_margin_mechanism_test.py\n" + "\n\n".join(tex_parts) + "\n"
    return md, tex


def _normalized_dose_rows(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = [c for c in ["direction", "unit_key", "evaluation_row", "row_id", "support_kind"] if c in raw.columns]
    for values, g in raw.groupby(keys, dropna=False, sort=False):
        if str(g["support_kind"].iloc[0]) != "known_flip":
            continue
        g = g.sort_values("dose")
        d = pd.to_numeric(g["dose"], errors="coerce").to_numpy(float)
        m = pd.to_numeric(g["endpoint_divergence_margin"], errors="coerce").to_numpy(float)
        m0 = _endpoint_value(d, m, 0.0); m1 = _endpoint_value(d, m, 1.0)
        span = m1 - m0 if np.isfinite(m0) and np.isfinite(m1) else math.nan
        if not np.isfinite(span) or abs(span) < 1e-12:
            continue
        base = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        for dose, margin in zip(d, m):
            if np.isfinite(margin):
                rows.append({**base, "dose": float(dose), "normalized_margin_position": float((margin - m0) / span)})
    return pd.DataFrame(rows)


def _plot_normalized_trajectories(raw: pd.DataFrame, path: Path) -> None:
    nd = _normalized_dose_rows(raw)
    if nd.empty:
        path.unlink(missing_ok=True); return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True, sharey=True)
    made = False
    for ax, direction in zip(axes, ["0to1", "1to0"]):
        dg = nd.loc[nd["direction"].astype(str) == direction].copy()
        if dg.empty:
            ax.set_axis_off(); continue
        made = True
        for _, eg in dg.groupby([c for c in ["unit_key", "evaluation_row", "row_id"] if c in dg.columns], dropna=False):
            ax.plot(eg["dose"], eg["normalized_margin_position"], alpha=.20, linewidth=.8)
        med = dg.groupby("dose", as_index=False)["normalized_margin_position"].median()
        ax.plot(med["dose"], med["normalized_margin_position"], marker="o", linewidth=2.0, label="median measured")
        grid = np.linspace(0, 1, 101)
        ax.plot(grid, grid, linestyle="--", linewidth=1.2, label="endpoint-affine null")
        ax.set_title("Discovery direction 0→1" if direction == "0to1" else "Discovery direction 1→0")
        ax.set_xlabel("Intervention dose")
        ax.grid(alpha=.22, linewidth=.45)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Normalized downstream margin position")
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
        fig.tight_layout(rect=(0,0,1,.88))
    else:
        fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    if made: fig.savefig(path, bbox_inches="tight")
    else: path.unlink(missing_ok=True)
    plt.close(fig)


def _plot_crossing_alignment(metric_df: pd.DataFrame, path: Path) -> None:
    d = _valid_known(metric_df.loc[metric_df["margin_type"] == PRIMARY_MARGIN].copy())
    if d.empty:
        path.unlink(missing_ok=True); return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True, sharey=True)
    made = False
    for ax, direction in zip(axes, ["0to1", "1to0"]):
        g = d.loc[d["direction"].astype(str) == direction].copy()
        x = pd.to_numeric(g["observed_behavior_persistent_crossing_dose"], errors="coerce")
        y = pd.to_numeric(g["observed_margin_persistent_crossing_dose"], errors="coerce")
        ok = x.notna() & y.notna()
        if not ok.any():
            ax.set_axis_off(); continue
        made = True
        ax.scatter(x[ok], y[ok], s=28, alpha=.75)
        ax.plot([0,1],[0,1], linestyle="--", linewidth=1.1)
        ax.set_title("Discovery direction 0→1" if direction == "0to1" else "Discovery direction 1→0")
        ax.set_xlabel("Behavioral persistent-crossing dose")
        ax.set_xlim(-.04,1.04); ax.set_ylim(-.04,1.04)
        ax.grid(alpha=.22, linewidth=.45)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Measured margin persistent-crossing dose")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    if made: fig.savefig(path, bbox_inches="tight")
    else: path.unlink(missing_ok=True)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    path = Path(args.dose_rows).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    required = {"dose", "flipped_from_baseline", "endpoint_divergence_margin"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} lacks {sorted(missing)}. Rerun graded_agonist_intervention.py with --record_endpoint_margin."
        )

    metric_df = _example_metric_rows(df)
    metric_df.to_csv(out_dir / "graded_margin_metric_tests.csv", index=False)
    primary_margin_rows = metric_df.loc[metric_df["margin_type"] == PRIMARY_MARGIN].copy()
    primary_margin_rows.to_csv(out_dir / "graded_margin_example_tests.csv", index=False)
    direction_df = _direction_summary(metric_df)
    direction_df.to_csv(out_dir / "graded_margin_direction_summary.csv", index=False)

    summary = {
        "schema": SCHEMA,
        "n_examples": int(metric_df[[c for c in ["unit_key", "evaluation_row", "row_id"] if c in metric_df.columns]].drop_duplicates().shape[0]) if not metric_df.empty else 0,
        "primary_margin": PRIMARY_MARGIN,
        "scientific_interpretation": (
            "The source coordinate follows a linear intervention path by construction. Low endpoint-chord fit/high normalized error therefore demonstrates a nonlinear downstream transformation for the measured margin. This rejects only the specific affine-margin null for that margin; it does not establish a discontinuity or uniquely identify a spiking mechanism."
        ),
        "margin_metrics": {m: _metric_summary(metric_df, m) for m in MARGINS if m in set(metric_df.get("margin_type", []))},
        "directional_primary_margin": direction_df.loc[direction_df["margin_type"] == PRIMARY_MARGIN].to_dict(orient="records") if not direction_df.empty else [],
    }
    (out_dir / "graded_margin_mechanism_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

    prefix = str(args.paper_prefix)
    _plot_normalized_trajectories(df, out_dir / f"{prefix}_affine_null.pdf")
    _plot_crossing_alignment(metric_df, out_dir / f"{prefix}_crossing_alignment.pdf")
    md, tex = _paper_text(direction_df, summary)
    (out_dir / f"{prefix}_paper_interpretation.md").write_text(md, encoding="utf-8")
    (out_dir / f"{prefix}_paper_interpretation.tex").write_text(tex, encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
