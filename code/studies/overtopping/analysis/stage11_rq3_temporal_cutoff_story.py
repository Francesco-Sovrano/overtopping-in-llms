#!/usr/bin/env python3
"""Aggregate the RQ3 temporal-cutoff experiment into paper-facing outputs.

Inference is performed at the run/direction-condition level. Channels and held-
out examples are aggregated before population statistics. The primary temporal
front-loading test is prespecified and selection-free:

    normalized capture at the halfway intervention horizon > 0.5.

This asks whether more than half of the full held-out effect is already captured
by the first half of the available autoregressive intervention window.

T50/T80 are *persistent* thresholds: the earliest tested horizon at which the
capture reaches the threshold and remains at or above it for every later tested
horizon. This is appropriate for an intervention-necessity claim because a
transient early crossing does not establish that later direct intervention is no
longer needed.

A max-step temporal-concentration statistic is retained only as a descriptive
quantity. It is not assigned a p-value against 1/K because max-share >= 1/K is a
mathematical consequence of taking a maximum and would not be a valid null test.

When complementary suffix-on artifacts are available, this stage also aggregates
them without changing the historical prefix outputs. Suffix increments are mapped
back to the absolute decode transition they newly add, which permits an independent
cross-sweep peak-agreement analysis.
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
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None

from studies.overtopping.analysis.rq3_population import expected_rq3_sources
from studies.overtopping.analysis.stage10_rq3_spiking_story_figures import (
    _bootstrap_median_ci,
    _clean,
    _format_p,
    _save,
)

BOOTSTRAP_SEED = 20260911
SCHEMA = "rq3-temporal-cutoff-story-v3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--primary-table", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--population-scope", choices=["primary", "primary+supplementary"], default="primary")
    p.add_argument("--evaluation-split", choices=["test", "train", "all"], default="test")
    p.add_argument("--paper-figures-dir", default=None)
    p.add_argument(
        "--graded-dir",
        default=None,
        help="Graded-agonist analysis directory containing the stage10 population story outputs. "
             "Defaults to <out>/../graded_agonist.",
    )
    return p.parse_args()


def _load(root: Path, args: argparse.Namespace, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    args.data_root = str(Path(args.data_root or root).expanduser().resolve())
    expected = expected_rq3_sources(args)
    rows_list: list[pd.DataFrame] = []
    audit: list[dict] = []

    for spec in expected:
        base = Path(spec["stats_dir"]) / "temporal_cutoff_intervention"
        manifest_path = base / "temporal_cutoff_intervention.json"
        rows_path = base / "temporal_cutoff_rows.csv.gz"
        payload: dict = {}
        status = "missing"
        n_known = 0
        n_selected = 0
        n_rows = 0

        if manifest_path.is_file():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                status = str(payload.get("status", "unknown"))
                n_known = int(payload.get("n_known_flip_examples_evaluated", 0) or 0)
                n_selected = int(payload.get("n_selected_agonist_directions", 0) or 0)
            except Exception:
                status = "manifest_read_error"

        if not bool(spec.get("decode_only", False)):
            audit.append({
                **spec,
                "status": "ineligible_non_decode_only",
                "manifest_status": status,
                "n_selected_agonist_directions": n_selected,
                "n_known_flip_examples_evaluated": n_known,
                "n_rows": 0,
            })
            continue

        df = pd.DataFrame()
        if status == "ok" and rows_path.is_file():
            try:
                df = pd.read_csv(rows_path)
            except Exception:
                status = "rows_read_error"
        elif status == "ok":
            status = "rows_missing"

        if not df.empty:
            n_rows = int(len(df))
            for key in ["run_id", "task", "model", "phase", "setting", "source_scope"]:
                df[key] = spec[key]
            rows_list.append(df)

        audit.append({
            **spec,
            "status": status,
            "manifest_status": str(payload.get("status", status)) if payload else status,
            "schema": str(payload.get("schema", "")) if payload else "",
            "n_selected_agonist_directions": n_selected,
            "n_known_flip_examples_evaluated": n_known,
            "n_rows": n_rows,
        })

    rows = pd.concat(rows_list, ignore_index=True) if rows_list else pd.DataFrame()
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(out / "temporal_cutoff_population_audit.csv", index=False)
    return rows, audit_df


def _load_suffix(root: Path, args: argparse.Namespace, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load complementary suffix-on artifacts, tolerating trees where they are absent."""
    args.data_root = str(Path(args.data_root or root).expanduser().resolve())
    expected = expected_rq3_sources(args)
    rows_list: list[pd.DataFrame] = []
    audit: list[dict] = []

    for spec in expected:
        base = Path(spec["stats_dir"]) / "temporal_suffix_intervention"
        manifest_path = base / "temporal_suffix_intervention.json"
        rows_path = base / "temporal_suffix_rows.csv.gz"
        payload: dict = {}
        status = "missing"
        n_known = 0
        n_selected = 0
        n_rows = 0

        if manifest_path.is_file():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                status = str(payload.get("status", "unknown"))
                n_known = int(payload.get("n_known_flip_examples_evaluated", 0) or 0)
                n_selected = int(payload.get("n_selected_agonist_directions", 0) or 0)
            except Exception:
                status = "manifest_read_error"

        if not bool(spec.get("decode_only", False)):
            audit.append({
                **spec,
                "status": "ineligible_non_decode_only",
                "manifest_status": status,
                "n_selected_agonist_directions": n_selected,
                "n_known_flip_examples_evaluated": n_known,
                "n_rows": 0,
            })
            continue

        df = pd.DataFrame()
        if status == "ok" and rows_path.is_file():
            try:
                df = pd.read_csv(rows_path)
            except Exception:
                status = "rows_read_error"
        elif status == "ok":
            status = "rows_missing"

        if not df.empty:
            n_rows = int(len(df))
            if "temporal_schedule" not in df.columns:
                df["temporal_schedule"] = "suffix"
            for key in ["run_id", "task", "model", "phase", "setting", "source_scope"]:
                df[key] = spec[key]
            rows_list.append(df)

        audit.append({
            **spec,
            "status": status,
            "manifest_status": str(payload.get("status", status)) if payload else status,
            "schema": str(payload.get("schema", "")) if payload else "",
            "n_selected_agonist_directions": n_selected,
            "n_known_flip_examples_evaluated": n_known,
            "n_rows": n_rows,
        })

    rows = pd.concat(rows_list, ignore_index=True) if rows_list else pd.DataFrame()
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(out / "temporal_suffix_population_audit.csv", index=False)
    return rows, audit_df


def _condition_curves(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"run_id", "direction", "unit_key", "active_decode_steps", "flipped_from_baseline", "support_kind"}
    if rows.empty or not required.issubset(rows.columns):
        return pd.DataFrame(), pd.DataFrame()

    d = rows.loc[rows["support_kind"].astype(str).eq("known_flip")].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    d["active_decode_steps"] = pd.to_numeric(d["active_decode_steps"], errors="coerce")
    d["flipped_from_baseline"] = d["flipped_from_baseline"].fillna(False).astype(bool).astype(float)
    d = d.loc[np.isfinite(d["active_decode_steps"])].copy()

    # Example -> unit -> run/direction condition. This prevents units with more
    # held-out support examples from dominating population inference.
    unit = (
        d.groupby(["run_id", "direction", "unit_key", "active_decode_steps"], as_index=False)["flipped_from_baseline"]
        .mean()
        .rename(columns={"flipped_from_baseline": "unit_rate"})
    )
    condition = (
        unit.groupby(["run_id", "direction", "active_decode_steps"], as_index=False)["unit_rate"]
        .mean()
        .rename(columns={"unit_rate": "condition_rate"})
        .sort_values(["run_id", "direction", "active_decode_steps"])
    )

    curve_rows: list[dict] = []
    inc_rows: list[dict] = []
    for (run_id, direction), g in condition.groupby(["run_id", "direction"], dropna=False, sort=False):
        g = g.sort_values("active_decode_steps")
        steps = g["active_decode_steps"].to_numpy(int)
        rates = g["condition_rate"].to_numpy(float)
        if not len(steps):
            continue
        baseline = float(rates[0])
        full = float(rates[-1])
        denom = full - baseline
        # A normalized 'capture of the full effect' is only meaningful when the
        # maximum horizon improves on horizon zero for this condition.
        capture = (rates - baseline) / denom if np.isfinite(denom) and denom > 1e-12 else np.full_like(rates, np.nan)

        previous = math.nan
        for step, raw_rate, normalized in zip(steps, rates, capture):
            curve_rows.append({
                "run_id": run_id,
                "direction": direction,
                "active_decode_steps": int(step),
                "condition_rate": float(raw_rate),
                "baseline_rate": baseline,
                "full_rate": full,
                "full_minus_baseline": denom,
                "normalized_capture": float(normalized) if np.isfinite(normalized) else math.nan,
            })
            if int(step) > 0:
                gain = float(normalized - previous) if np.isfinite(normalized) and np.isfinite(previous) else math.nan
                inc_rows.append({
                    "run_id": run_id,
                    "direction": direction,
                    "step": int(step),
                    "incremental_gain": gain,
                })
            previous = float(normalized) if np.isfinite(normalized) else math.nan

    return pd.DataFrame(curve_rows), pd.DataFrame(inc_rows)


def _suffix_increment_transitions(curves: pd.DataFrame, inc: pd.DataFrame) -> pd.DataFrame:
    """Map suffix active-count increments to the absolute transition newly added.

    If K is the full decode horizon, moving suffix active-count from s-1 to s adds
    transition K-s+1. The returned ``step`` column is therefore directly comparable
    with the prefix increment's absolute transition number.
    """
    if curves.empty or inc.empty:
        return pd.DataFrame(columns=["run_id", "direction", "active_decode_steps", "step", "incremental_gain"])
    maxima = (
        curves.groupby(["run_id", "direction"], as_index=False)["active_decode_steps"]
        .max()
        .rename(columns={"active_decode_steps": "full_decode_horizon"})
    )
    out = inc.rename(columns={"step": "active_decode_steps"}).merge(
        maxima, on=["run_id", "direction"], how="left", validate="many_to_one"
    )
    out["step"] = (
        pd.to_numeric(out["full_decode_horizon"], errors="coerce")
        - pd.to_numeric(out["active_decode_steps"], errors="coerce")
        + 1
    )
    cols = ["run_id", "direction", "active_decode_steps", "step", "incremental_gain", "full_decode_horizon"]
    return out[cols].sort_values(["run_id", "direction", "step"], kind="mergesort").reset_index(drop=True)


def _cross_sweep_peak_agreement(prefix_inc: pd.DataFrame, suffix_inc: pd.DataFrame) -> pd.DataFrame:
    """Compare peak transitions across complementary sweeps without self-alignment."""
    if prefix_inc.empty or suffix_inc.empty:
        return pd.DataFrame()

    def _peaks(df: pd.DataFrame, label: str) -> pd.DataFrame:
        rows: list[dict] = []
        for (run_id, direction), g in df.groupby(["run_id", "direction"], dropna=False, sort=False):
            gg = g.copy()
            gg["step"] = pd.to_numeric(gg["step"], errors="coerce")
            gg["incremental_gain"] = pd.to_numeric(gg["incremental_gain"], errors="coerce")
            gg = gg.loc[np.isfinite(gg["step"]) & np.isfinite(gg["incremental_gain"])].copy()
            if gg.empty:
                continue
            positive = np.clip(gg["incremental_gain"].to_numpy(float), 0.0, None)
            if float(np.sum(positive)) <= 1e-12:
                continue
            i = int(np.argmax(positive))
            rows.append({
                "run_id": run_id,
                "direction": direction,
                f"{label}_peak_transition": int(gg.iloc[i]["step"]),
                f"{label}_peak_gain": float(gg.iloc[i]["incremental_gain"]),
            })
        return pd.DataFrame(rows)

    p = _peaks(prefix_inc, "prefix")
    q = _peaks(suffix_inc, "suffix")
    if p.empty or q.empty:
        return pd.DataFrame()
    out = p.merge(q, on=["run_id", "direction"], how="inner", validate="one_to_one")
    out["prefix_peak_minus_suffix_event"] = (
        out["prefix_peak_transition"] - out["suffix_peak_transition"]
    )
    out["same_transition"] = out["prefix_peak_minus_suffix_event"].eq(0)
    out["within_one_transition"] = out["prefix_peak_minus_suffix_event"].abs().le(1)
    return out



def _cross_sweep_event_validation(
    discovery_inc: pd.DataFrame,
    validation_inc: pd.DataFrame,
    *,
    discovery_label: str,
    validation_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align one temporal sweep to an EVENT selected by the other sweep.

    EVENT is the absolute decode transition with the largest *positive* incremental
    gain in ``discovery_inc``. The profile plotted/tested comes entirely from
    ``validation_inc``. This deliberately avoids the self-alignment problem that
    occurs when a threshold/peak and its validating profile are derived from the
    same temporal trajectory.

    Returns
    -------
    aligned_rows
        One row per validation transition, with ``event_offset = step - EVENT``.
    condition_stats
        One row per run/direction with the held-out EVENT-vs-adjacent enrichment
        and cross-sweep peak agreement.
    """
    required = {"run_id", "direction", "step", "incremental_gain"}
    if (
        discovery_inc.empty
        or validation_inc.empty
        or not required.issubset(discovery_inc.columns)
        or not required.issubset(validation_inc.columns)
    ):
        return pd.DataFrame(), pd.DataFrame()

    def _positive_peaks(df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        for (run_id, direction), g in df.groupby(["run_id", "direction"], dropna=False, sort=False):
            gg = g[["step", "incremental_gain"]].copy()
            gg["step"] = pd.to_numeric(gg["step"], errors="coerce")
            gg["incremental_gain"] = pd.to_numeric(gg["incremental_gain"], errors="coerce")
            gg = gg.loc[np.isfinite(gg["step"]) & np.isfinite(gg["incremental_gain"])].sort_values("step")
            if gg.empty:
                continue
            positive = np.clip(gg["incremental_gain"].to_numpy(float), 0.0, None)
            if float(np.sum(positive)) <= 1e-12:
                continue
            i = int(np.argmax(positive))
            rows.append({
                "run_id": run_id,
                "direction": direction,
                "event_transition": int(gg.iloc[i]["step"]),
                "discovery_event_gain": float(gg.iloc[i]["incremental_gain"]),
            })
        return pd.DataFrame(rows)

    events = _positive_peaks(discovery_inc)
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    val = validation_inc.copy()
    val["step"] = pd.to_numeric(val["step"], errors="coerce")
    val["incremental_gain"] = pd.to_numeric(val["incremental_gain"], errors="coerce")
    val = val.loc[np.isfinite(val["step"]) & np.isfinite(val["incremental_gain"])].copy()
    aligned = val.merge(events, on=["run_id", "direction"], how="inner", validate="many_to_one")
    if aligned.empty:
        return pd.DataFrame(), pd.DataFrame()
    aligned["event_offset"] = aligned["step"] - aligned["event_transition"]
    aligned["discovery_sweep"] = str(discovery_label)
    aligned["validation_sweep"] = str(validation_label)

    rows: list[dict] = []
    for (run_id, direction), g in aligned.groupby(["run_id", "direction"], dropna=False, sort=False):
        gg = g.sort_values("step")
        center = pd.to_numeric(
            gg.loc[gg["event_offset"].eq(0), "incremental_gain"], errors="coerce"
        ).dropna().to_numpy(float)
        if not len(center):
            continue
        adjacent = pd.to_numeric(
            gg.loc[gg["event_offset"].isin([-1, 1]), "incremental_gain"], errors="coerce"
        ).dropna().to_numpy(float)
        event_gain = float(center[0])
        adjacent_mean = float(np.mean(adjacent)) if len(adjacent) else math.nan
        enrichment = event_gain - adjacent_mean if np.isfinite(adjacent_mean) else math.nan

        gains = pd.to_numeric(gg["incremental_gain"], errors="coerce").to_numpy(float)
        steps = pd.to_numeric(gg["step"], errors="coerce").to_numpy(float)
        positive = np.clip(gains, 0.0, None)
        validation_peak = math.nan
        if len(positive) and float(np.sum(positive)) > 1e-12:
            validation_peak = float(steps[int(np.argmax(positive))])

        event_transition = int(gg["event_transition"].iloc[0])
        rows.append({
            "run_id": run_id,
            "direction": direction,
            "event_transition": event_transition,
            "event_gain_in_validation_sweep": event_gain,
            "adjacent_mean_gain_in_validation_sweep": adjacent_mean,
            "event_minus_adjacent_gain": enrichment,
            "n_available_adjacent_transitions": int(len(adjacent)),
            "validation_peak_transition": validation_peak,
            "validation_peak_minus_event": (
                validation_peak - event_transition if np.isfinite(validation_peak) else math.nan
            ),
            "same_peak_transition": bool(validation_peak == event_transition) if np.isfinite(validation_peak) else False,
            "within_one_transition": bool(abs(validation_peak - event_transition) <= 1) if np.isfinite(validation_peak) else False,
            "discovery_sweep": str(discovery_label),
            "validation_sweep": str(validation_label),
        })
    return aligned.reset_index(drop=True), pd.DataFrame(rows)


def _cross_sweep_scope_stats(condition_stats: pd.DataFrame, *, direction: str | None = None) -> dict:
    if condition_stats.empty:
        return {}
    d = condition_stats if direction is None else condition_stats.loc[condition_stats["direction"].astype(str).eq(direction)]
    if d.empty:
        return {}
    vals = pd.to_numeric(d["event_minus_adjacent_gain"], errors="coerce").dropna().to_numpy(float)
    p_value = math.nan
    if wilcoxon is not None and len(vals) and np.any(np.abs(vals) > 1e-12):
        try:
            p_value = float(wilcoxon(vals, alternative="greater", zero_method="wilcox").pvalue)
        except Exception:
            pass
    ci = _bootstrap_median_ci(vals, seed=BOOTSTRAP_SEED + 1400 + len(d)) if len(vals) else (math.nan, math.nan)
    return {
        "n_conditions": int(len(d)),
        "n_event_enrichment_conditions": int(len(vals)),
        "median_event_minus_adjacent_gain": float(np.median(vals)) if len(vals) else math.nan,
        "median_event_minus_adjacent_gain_ci95": [float(ci[0]), float(ci[1])],
        "event_minus_adjacent_wilcoxon_one_sided_p": p_value,
        "same_peak_transition_rate": float(pd.to_numeric(d["same_peak_transition"], errors="coerce").mean()),
        "within_one_transition_rate": float(pd.to_numeric(d["within_one_transition"], errors="coerce").mean()),
    }


def _cross_sweep_aligned_profile(aligned: pd.DataFrame, *, direction: str | None = None, max_abs_offset: int = 4) -> pd.DataFrame:
    if aligned.empty:
        return pd.DataFrame()
    d = aligned if direction is None else aligned.loc[aligned["direction"].astype(str).eq(direction)]
    if d.empty:
        return pd.DataFrame()
    d = d.loc[pd.to_numeric(d["event_offset"], errors="coerce").abs().le(int(max_abs_offset))].copy()
    if d.empty:
        return pd.DataFrame()
    out: list[dict] = []
    for offset, g in d.groupby("event_offset", sort=True):
        vals = pd.to_numeric(g["incremental_gain"], errors="coerce").dropna().to_numpy(float)
        if not len(vals):
            continue
        ci_low, ci_high = _bootstrap_median_ci(vals, seed=BOOTSTRAP_SEED + 1500 + int(offset) + 20)
        out.append({
            "event_offset": int(offset),
            "median": float(np.median(vals)),
            "q25": float(np.quantile(vals, .25)),
            "q75": float(np.quantile(vals, .75)),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "n_conditions": int(len(vals)),
        })
    return pd.DataFrame(out).sort_values("event_offset") if out else pd.DataFrame()


def _plot_cross_sweep_event_validation(
    suffix_event_prefix_aligned: pd.DataFrame,
    suffix_event_prefix_stats: pd.DataFrame,
    prefix_event_suffix_aligned: pd.DataFrame,
    prefix_event_suffix_stats: pd.DataFrame,
    path: Path,
) -> None:
    """Paper-facing 2x3 cross-sweep EVENT validation figure.

    Top row: suffix-only sweep defines EVENT; prefix-only sweep is held-out validation.
    Bottom row: reciprocal sensitivity analysis. Because EVENT and the displayed
    profile come from different intervention schedules, a center spike is not
    guaranteed by construction.
    """
    if suffix_event_prefix_aligned.empty or prefix_event_suffix_aligned.empty:
        path.unlink(missing_ok=True)
        return

    fig, axes = plt.subplots(2, 3, figsize=(12.3, 6.6), sharex=True)
    scopes = [("1to0", "1→0"), ("0to1", "0→1"), (None, "All directions")]
    rows = [
        (
            suffix_event_prefix_aligned,
            suffix_event_prefix_stats,
            "Suffix-only sweep defines EVENT; prefix-only sweep validates",
        ),
        (
            prefix_event_suffix_aligned,
            prefix_event_suffix_stats,
            "Prefix-only sweep defines EVENT; suffix-only sweep validates",
        ),
    ]

    for row_idx, (aligned, cond_stats, row_title) in enumerate(rows):
        for col_idx, (direction, title) in enumerate(scopes):
            ax = axes[row_idx, col_idx]
            profile = _cross_sweep_aligned_profile(aligned, direction=direction, max_abs_offset=4)
            stats = _cross_sweep_scope_stats(cond_stats, direction=direction)
            if profile.empty:
                ax.text(.5, .5, "No paired cross-sweep conditions", transform=ax.transAxes, ha="center", va="center")
                _clean(ax)
                continue
            _whisker_curve(ax, profile, "event_offset")
            ax.axvline(0, linestyle=":", linewidth=1.0)
            ax.axhline(0, linestyle=":", linewidth=.8)
            offsets = profile["event_offset"].astype(int).tolist()
            ax.set_xticks(offsets, ["EVENT" if x == 0 else f"{x:+d}" for x in offsets])
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel("Incremental normalized causal effect\nin held-out sweep")
            if row_idx == 1:
                ax.set_xlabel("Decode transition relative to EVENT\ndefined by the other sweep")
            ci = stats.get("median_event_minus_adjacent_gain_ci95", [math.nan, math.nan])
            ax.text(
                .97, .97,
                f"EVENT−adjacent={stats.get('median_event_minus_adjacent_gain', math.nan):+.2f}\n"
                f"95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]\n"
                f"{_format_p(stats.get('event_minus_adjacent_wilcoxon_one_sided_p', math.nan))}; "
                f"n={stats.get('n_event_enrichment_conditions', 0)}\n"
                f"same peak={100*stats.get('same_peak_transition_rate', math.nan):.0f}%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.5,
            )
            _clean(ax)
        axes[row_idx, 1].text(
            .5, 1.13, row_title,
            transform=axes[row_idx, 1].transAxes,
            ha="center", va="bottom", fontsize=8.4, fontweight="bold",
        )

    fig.suptitle(
        "Cross-sweep temporal EVENT validation: a center spike is not guaranteed by construction",
        fontsize=10.2,
        y=1.01,
    )
    _save(fig, path)

def _profile(df: pd.DataFrame, *, xcol: str, ycol: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out: list[dict] = []
    for x, g in df.groupby(xcol):
        vals = pd.to_numeric(g[ycol], errors="coerce").dropna().to_numpy(float)
        if not len(vals):
            continue
        ci_low, ci_high = _bootstrap_median_ci(vals, seed=BOOTSTRAP_SEED + int(x) + 700)
        out.append({
            xcol: int(x),
            "median": float(np.median(vals)),
            "q25": float(np.quantile(vals, .25)),
            "q75": float(np.quantile(vals, .75)),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_conditions": int(len(vals)),
        })
    return pd.DataFrame(out).sort_values(xcol) if out else pd.DataFrame()


def _persistent_threshold_step(steps: np.ndarray, capture: np.ndarray, threshold: float) -> float:
    """Earliest tested horizon that reaches threshold and never drops below it later."""
    finite = np.isfinite(steps) & np.isfinite(capture)
    steps = np.asarray(steps)[finite]
    capture = np.asarray(capture, dtype=float)[finite]
    if not len(steps):
        return math.nan
    order = np.argsort(steps)
    steps = steps[order]
    capture = capture[order]
    for index in range(len(steps)):
        if capture[index] >= threshold and bool(np.all(capture[index:] >= threshold)):
            return float(steps[index])
    return math.nan


def _half_horizon_capture(steps: np.ndarray, capture: np.ndarray) -> tuple[float, float]:
    """Return capture at the latest tested horizon not exceeding half the full horizon."""
    finite = np.isfinite(steps) & np.isfinite(capture)
    steps = np.asarray(steps)[finite]
    capture = np.asarray(capture, dtype=float)[finite]
    if not len(steps):
        return math.nan, math.nan
    order = np.argsort(steps)
    steps = steps[order]
    capture = capture[order]
    full_horizon = float(np.max(steps))
    target = full_horizon / 2.0
    eligible = np.flatnonzero(steps <= target)
    if not len(eligible):
        return math.nan, math.nan
    idx = int(eligible[-1])
    return float(steps[idx]), float(capture[idx])


def _temporal_stats(curves: pd.DataFrame, inc: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    if curves.empty:
        return {}, pd.DataFrame()

    condition_rows: list[dict] = []
    for (run_id, direction), g in curves.groupby(["run_id", "direction"], dropna=False, sort=False):
        g = g.sort_values("active_decode_steps")
        steps = pd.to_numeric(g["active_decode_steps"], errors="coerce").to_numpy(float)
        capture = pd.to_numeric(g["normalized_capture"], errors="coerce").to_numpy(float)
        finite = np.isfinite(steps) & np.isfinite(capture)
        if not finite.any():
            continue

        steps_f = steps[finite]
        capture_f = capture[finite]
        max_step = int(np.max(steps_f))
        t50 = _persistent_threshold_step(steps_f, capture_f, 0.50)
        t80 = _persistent_threshold_step(steps_f, capture_f, 0.80)
        half_step, half_capture = _half_horizon_capture(steps_f, capture_f)

        inc_g = inc.loc[
            inc["run_id"].astype(str).eq(str(run_id))
            & inc["direction"].astype(str).eq(str(direction))
        ].copy()
        temporal_concentration = math.nan
        peak_step = math.nan
        early_mass = math.nan
        late_mass = math.nan
        if not inc_g.empty:
            inc_g["step"] = pd.to_numeric(inc_g["step"], errors="coerce")
            inc_g["incremental_gain"] = pd.to_numeric(inc_g["incremental_gain"], errors="coerce")
            inc_g = inc_g.loc[np.isfinite(inc_g["step"]) & np.isfinite(inc_g["incremental_gain"])].sort_values("step")
            if not inc_g.empty:
                half = max_step / 2.0
                early_mass = float(inc_g.loc[inc_g["step"] <= half, "incremental_gain"].sum())
                late_mass = float(inc_g.loc[inc_g["step"] > half, "incremental_gain"].sum())
                positive = np.clip(inc_g["incremental_gain"].to_numpy(float), 0.0, None)
                total_positive = float(np.sum(positive))
                if total_positive > 1e-12:
                    temporal_concentration = float(np.max(positive) / total_positive)
                    peak_step = float(inc_g.iloc[int(np.argmax(positive))]["step"])

        condition_rows.append({
            "run_id": run_id,
            "direction": direction,
            "max_step": max_step,
            "persistent_t50": t50,
            "persistent_t80": t80,
            "half_horizon_step": half_step,
            "half_horizon_capture": half_capture,
            "early_mass": early_mass,
            "late_mass": late_mass,
            "early_minus_late": early_mass - late_mass if np.isfinite(early_mass) and np.isfinite(late_mass) else math.nan,
            "temporal_concentration": temporal_concentration,
            "peak_step": peak_step,
        })

    cond = pd.DataFrame(condition_rows)
    if cond.empty:
        return {}, cond

    def _vals(column: str) -> np.ndarray:
        return pd.to_numeric(cond[column], errors="coerce").dropna().to_numpy(float)

    t50 = _vals("persistent_t50")
    t80 = _vals("persistent_t80")
    half_capture = _vals("half_horizon_capture")
    early_late = _vals("early_minus_late")
    concentration = _vals("temporal_concentration")

    half_p = math.nan
    if wilcoxon is not None and len(half_capture) and np.any(np.abs(half_capture - 0.5) > 1e-12):
        try:
            half_p = float(wilcoxon(half_capture - 0.5, alternative="greater", zero_method="wilcox").pvalue)
        except Exception:
            pass

    early_late_p = math.nan
    if wilcoxon is not None and len(early_late) and np.any(np.abs(early_late) > 1e-12):
        try:
            early_late_p = float(wilcoxon(early_late, alternative="greater", zero_method="wilcox").pvalue)
        except Exception:
            pass

    stats = {
        "n_conditions": int(len(cond)),
        "n_half_horizon_conditions": int(len(half_capture)),
        "median_persistent_t50": float(np.median(t50)) if len(t50) else math.nan,
        "median_persistent_t50_ci95": list(_bootstrap_median_ci(t50, seed=BOOTSTRAP_SEED + 900)) if len(t50) else [math.nan, math.nan],
        "median_persistent_t80": float(np.median(t80)) if len(t80) else math.nan,
        "median_persistent_t80_ci95": list(_bootstrap_median_ci(t80, seed=BOOTSTRAP_SEED + 901)) if len(t80) else [math.nan, math.nan],
        "median_half_horizon_capture": float(np.median(half_capture)) if len(half_capture) else math.nan,
        "median_half_horizon_capture_ci95": list(_bootstrap_median_ci(half_capture, seed=BOOTSTRAP_SEED + 902)) if len(half_capture) else [math.nan, math.nan],
        "half_horizon_capture_vs_half_wilcoxon_one_sided_p": half_p,
        "median_early_minus_late": float(np.median(early_late)) if len(early_late) else math.nan,
        "median_early_minus_late_ci95": list(_bootstrap_median_ci(early_late, seed=BOOTSTRAP_SEED + 903)) if len(early_late) else [math.nan, math.nan],
        "early_vs_late_wilcoxon_one_sided_p": early_late_p,
        # Descriptive only: no p-value because max share >= 1/K by construction.
        "median_temporal_concentration": float(np.median(concentration)) if len(concentration) else math.nan,
        "median_temporal_concentration_ci95": list(_bootstrap_median_ci(concentration, seed=BOOTSTRAP_SEED + 904)) if len(concentration) else [math.nan, math.nan],
        "n_temporal_concentration_conditions": int(len(concentration)),
    }
    return stats, cond


def _whisker_curve(ax, profile: pd.DataFrame, xcol: str) -> None:
    x = profile[xcol].to_numpy(float)
    med = profile["median"].to_numpy(float)
    q25 = profile["q25"].to_numpy(float)
    q75 = profile["q75"].to_numpy(float)
    lo = profile["ci_low"].to_numpy(float)
    hi = profile["ci_high"].to_numpy(float)
    for xi, a, b, c, d in zip(x, q25, q75, lo, hi):
        # Thin = IQR, thick = 95% bootstrap CI, same visual grammar as Fig. 4e.
        ax.vlines(xi, a, b, linewidth=.7, alpha=.65)
        ax.vlines(xi, c, d, linewidth=2.6)
    ax.plot(x, med, marker="o", linewidth=1.65)


def _plot_temporal_main(capture: pd.DataFrame, gain: pd.DataFrame, stats: dict, path: Path) -> None:
    if capture.empty or gain.empty:
        path.unlink(missing_ok=True)
        return
    # Slightly larger canvas and annotation fonts than the first draft.  Panel
    # labels are kept inside the axes so they cannot collide with rotated y labels.
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax = axes[0]
    _whisker_curve(ax, capture, "active_decode_steps")
    ax.axhline(0.5, linestyle=":", linewidth=1.0)
    ax.axhline(1.0, linestyle=":", linewidth=1.0)
    ax.set_xlabel(
        "Active autoregressive decode transitions\n(full intervention before cutoff; off after cutoff)",
        fontsize=10.5,
    )
    ax.set_ylabel("Normalized capture of full held-out flip effect", fontsize=10.5)
    ax.tick_params(labelsize=9.5)
    half_ci = stats.get("median_half_horizon_capture_ci95", [math.nan, math.nan])
    t50_ci = stats.get("median_persistent_t50_ci95", [math.nan, math.nan])
    t80_ci = stats.get("median_persistent_t80_ci95", [math.nan, math.nan])
    # Lower-right is intentionally empty because the capture curve is near one
    # after the first few transitions. Keeping the statistics there avoids
    # obscuring the curve/whiskers at the top of the panel.
    ax.text(
        .97, .07,
        f"Half-horizon capture={stats.get('median_half_horizon_capture', math.nan):.2f}\n"
        f"95% CI [{half_ci[0]:.2f}, {half_ci[1]:.2f}]\n"
        f"vs 0.5 {_format_p(stats.get('half_horizon_capture_vs_half_wilcoxon_one_sided_p', math.nan))}; "
        f"n={stats.get('n_half_horizon_conditions', 0)}\n"
        f"persistent T50={stats.get('median_persistent_t50', math.nan):.1f} [{t50_ci[0]:.1f}, {t50_ci[1]:.1f}]\n"
        f"persistent T80={stats.get('median_persistent_t80', math.nan):.1f} [{t80_ci[0]:.1f}, {t80_ci[1]:.1f}]",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.6,
        linespacing=1.18,
    )
    ax.set_ylim(min(-.08, float(np.nanmin(capture["q25"])) * 1.1), max(1.08, float(np.nanmax(capture["q75"])) * 1.1))
    _clean(ax)

    ax = axes[1]
    _whisker_curve(ax, gain, "step")
    ax.axhline(0.0, linestyle=":", linewidth=1.0)
    ax.set_xlabel("Decode transition receiving intervention", fontsize=10.5)
    ax.set_ylabel("Incremental normalized effect mass", fontsize=10.5)
    ax.tick_params(labelsize=9.5)
    early_ci = stats.get("median_early_minus_late_ci95", [math.nan, math.nan])
    conc_ci = stats.get("median_temporal_concentration_ci95", [math.nan, math.nan])
    ax.text(
        .97, .96,
        f"Early-late={stats.get('median_early_minus_late', math.nan):+.2f}\n"
        f"95% CI [{early_ci[0]:+.2f}, {early_ci[1]:+.2f}]\n"
        f"paired Wilcoxon {_format_p(stats.get('early_vs_late_wilcoxon_one_sided_p', math.nan))}\n"
        f"n={stats.get('n_conditions', 0)}; peak-share={stats.get('median_temporal_concentration', math.nan):.2f}\n"
        f"95% CI [{conc_ci[0]:.2f}, {conc_ci[1]:.2f}] (descriptive)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.6,
        linespacing=1.18,
    )
    _clean(ax)

    for label, ax in zip(["(a)", "(b)"], axes):
        ax.text(
            .01, .99, label, transform=ax.transAxes, fontsize=11.0,
            fontweight="bold", ha="left", va="top",
        )
    _save(fig, path)


def _plot_fig4e_event_strength_temporal(
    event_profile: pd.DataFrame,
    event_stats: dict,
    strength: pd.DataFrame,
    strength_stats: dict,
    capture: pd.DataFrame,
    temporal_stats: dict,
    path: Path,
) -> None:
    """Assemble Fig. 4e: event localization, strength sharpening, temporal capture."""
    if event_profile.empty or strength.empty or capture.empty:
        path.unlink(missing_ok=True)
        return

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.75))

    # (a) Population event localization.
    ax = axes[0]
    x = event_profile["offset"].to_numpy(float)
    med = 100 * event_profile["median"].to_numpy(float)
    q25 = 100 * event_profile["q25"].to_numpy(float)
    q75 = 100 * event_profile["q75"].to_numpy(float)
    lo = 100 * event_profile["ci_low"].to_numpy(float)
    hi = 100 * event_profile["ci_high"].to_numpy(float)
    for xi, a, b, c, d in zip(x, q25, q75, lo, hi):
        ax.vlines(xi, a, b, linewidth=.7, alpha=.65)
        ax.vlines(xi, c, d, linewidth=2.6)
    ax.plot(x, med, marker="o", linewidth=1.65)
    zero = np.flatnonzero(event_profile["offset"].to_numpy() == 0)
    if len(zero):
        ax.scatter([0], [med[int(zero[0])]], s=70, zorder=5)
    ax.axvline(0, linestyle=":", linewidth=1.0)
    ax.set_xticks([-4, -2, 0, 2, 4], ["−4", "−2", "EVENT", "+2", "+4"])
    ax.set_xlabel("Largest margin-change interval\nrelative to behavioral crossing")
    ax.set_ylabel("Condition-median trajectories (%)")
    ci = event_stats.get("median_event_minus_adjacent_ci95", [math.nan, math.nan])
    ax.text(
        .97, .96,
        f"EVENT−adjacent +{100*event_stats.get('median_event_minus_adjacent', math.nan):.1f} pp\n"
        f"95% CI [{100*ci[0]:.1f}, {100*ci[1]:.1f}]\n"
        f"paired Wilcoxon {_format_p(event_stats.get('wilcoxon_one_sided_p', math.nan))}; "
        f"n={event_stats.get('n_conditions', 0)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.9,
    )
    ax.set_ylim(0, max(32.0, float(np.nanmax(q75)) * 1.18))
    _clean(ax)

    # (b) Strength sharpening (the former Fig. 4e panel c).
    ax = axes[1]
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
    ax.text(
        .03, .96,
        f"High−Low ΔC={strength_stats.get('median_high_minus_low', math.nan):+.2f}\n"
        f"95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]\n"
        f"paired Wilcoxon {_format_p(strength_stats.get('wilcoxon_one_sided_p', math.nan))}; "
        f"n={strength_stats.get('n_paired_tertile_groups', 0)}",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.9,
    )
    ax.set_ylim(.8, max(3.6, float(np.nanmax(sq75)) * 1.18))
    _clean(ax)

    # (c) Temporal-cutoff capture profile, matching standalone Fig. S15.
    ax = axes[2]
    _whisker_curve(ax, capture, "active_decode_steps")
    ax.axhline(.5, linestyle=":", linewidth=1.0)
    ax.axhline(1.0, linestyle=":", linewidth=1.0)
    ci = temporal_stats.get("median_half_horizon_capture_ci95", [math.nan, math.nan])
    ax.text(
        .97, .07,
        f"Half-horizon capture={temporal_stats.get('median_half_horizon_capture', math.nan):.2f}\n"
        f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}]\n"
        f"vs 0.5 {_format_p(temporal_stats.get('half_horizon_capture_vs_half_wilcoxon_one_sided_p', math.nan))}; "
        f"n={temporal_stats.get('n_half_horizon_conditions', 0)}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, linespacing=1.15,
    )
    ax.set_xlabel("Active autoregressive decode transitions")
    ax.set_ylabel("Normalized capture of full held-out flip effect")
    ax.set_ylim(
        min(-.08, float(np.nanmin(capture["q25"])) * 1.1),
        max(1.08, float(np.nanmax(capture["q75"])) * 1.1),
    )
    _clean(ax)

    for label, ax in zip(["(a)", "(b)", "(c)"], axes):
        ax.text(-.12, 1.03, label, transform=ax.transAxes, fontweight="bold", ha="left", va="bottom")
    _save(fig, path)


def _load_stage10_fig4e_inputs(graded: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    """Load stage10 outputs required for the revised paper-facing Fig. 4e."""
    event_path = graded / "population_event_localization.csv"
    strength_path = graded / "strength_concentration_tertile_summary.csv"
    summary_path = graded / "population_spiking_story_summary.json"
    if not event_path.is_file() or not strength_path.is_file() or not summary_path.is_file():
        return pd.DataFrame(), {}, pd.DataFrame(), {}
    try:
        event_profile = pd.read_csv(event_path)
        strength = pd.read_csv(strength_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame(), {}, pd.DataFrame(), {}
    return (
        event_profile,
        dict(summary.get("event_localization", {})),
        strength,
        dict(summary.get("strength", {})),
    )


def _plot_capture(capture: pd.DataFrame, stats: dict, path: Path) -> None:
    if capture.empty:
        path.unlink(missing_ok=True)
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    _whisker_curve(ax, capture, "active_decode_steps")
    ax.axhline(.5, linestyle=":", linewidth=1.0)
    ax.axhline(1.0, linestyle=":", linewidth=1.0)
    ci = stats.get("median_half_horizon_capture_ci95", [math.nan, math.nan])
    # Put the annotation in the low-effect region rather than over the late
    # high-capture points at the top-right of the plot.
    ax.text(
        .97, .07,
        f"Half-horizon capture={stats.get('median_half_horizon_capture', math.nan):.2f}\n"
        f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}]\n"
        f"vs 0.5 {_format_p(stats.get('half_horizon_capture_vs_half_wilcoxon_one_sided_p', math.nan))}; "
        f"n={stats.get('n_half_horizon_conditions', 0)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.0,
        linespacing=1.18,
    )
    ax.set_xlabel("Active autoregressive decode transitions", fontsize=10.5)
    ax.set_ylabel("Normalized capture of full held-out flip effect", fontsize=10.5)
    ax.tick_params(labelsize=9.5)
    _clean(ax)
    _save(fig, path)


def _statistics_table(stats: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "analysis": "persistent_T50",
            "estimate": stats.get("median_persistent_t50", math.nan),
            "ci_low": stats.get("median_persistent_t50_ci95", [math.nan, math.nan])[0],
            "ci_high": stats.get("median_persistent_t50_ci95", [math.nan, math.nan])[1],
            "p_value": math.nan,
            "test": "condition-bootstrap median; persistent threshold",
            "n": stats.get("n_conditions", 0),
        },
        {
            "analysis": "persistent_T80",
            "estimate": stats.get("median_persistent_t80", math.nan),
            "ci_low": stats.get("median_persistent_t80_ci95", [math.nan, math.nan])[0],
            "ci_high": stats.get("median_persistent_t80_ci95", [math.nan, math.nan])[1],
            "p_value": math.nan,
            "test": "condition-bootstrap median; persistent threshold",
            "n": stats.get("n_conditions", 0),
        },
        {
            "analysis": "half_horizon_capture_vs_0.5",
            "estimate": stats.get("median_half_horizon_capture", math.nan),
            "ci_low": stats.get("median_half_horizon_capture_ci95", [math.nan, math.nan])[0],
            "ci_high": stats.get("median_half_horizon_capture_ci95", [math.nan, math.nan])[1],
            "p_value": stats.get("half_horizon_capture_vs_half_wilcoxon_one_sided_p", math.nan),
            "test": "one-sided paired Wilcoxon of condition half-horizon capture minus 0.5",
            "n": stats.get("n_half_horizon_conditions", 0),
        },
        {
            "analysis": "early_minus_late_effect_mass",
            "estimate": stats.get("median_early_minus_late", math.nan),
            "ci_low": stats.get("median_early_minus_late_ci95", [math.nan, math.nan])[0],
            "ci_high": stats.get("median_early_minus_late_ci95", [math.nan, math.nan])[1],
            "p_value": stats.get("early_vs_late_wilcoxon_one_sided_p", math.nan),
            "test": "one-sided paired Wilcoxon over run/direction conditions",
            "n": stats.get("n_conditions", 0),
        },
        {
            "analysis": "peak_share_temporal_concentration_descriptive",
            "estimate": stats.get("median_temporal_concentration", math.nan),
            "ci_low": stats.get("median_temporal_concentration_ci95", [math.nan, math.nan])[0],
            "ci_high": stats.get("median_temporal_concentration_ci95", [math.nan, math.nan])[1],
            "p_value": math.nan,
            "test": "descriptive only; no uniform-null p-value because max share >= 1/K by construction",
            "n": stats.get("n_temporal_concentration_conditions", 0),
        },
    ])


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    paper = Path(args.paper_figures_dir).expanduser().resolve() if args.paper_figures_dir else None
    graded = (
        Path(args.graded_dir).expanduser().resolve()
        if args.graded_dir
        else out.parent / "graded_agonist"
    )

    rows, audit = _load(root, args, out)
    curves, inc = _condition_curves(rows)
    capture_profile = _profile(curves, xcol="active_decode_steps", ycol="normalized_capture")
    gain_profile = _profile(inc, xcol="step", ycol="incremental_gain")
    stats, condition_stats = _temporal_stats(curves, inc)

    # Complementary suffix-on sweep is additive: absence never suppresses the
    # historical prefix analysis. Its increments are remapped onto absolute
    # transition numbers for direct cross-sweep comparison.
    suffix_rows, suffix_audit = _load_suffix(root, args, out)
    suffix_curves, suffix_inc_active_count = _condition_curves(suffix_rows)
    suffix_inc = _suffix_increment_transitions(suffix_curves, suffix_inc_active_count)
    cross_sweep = _cross_sweep_peak_agreement(inc, suffix_inc)

    # Cross-sweep EVENT validation.  EVENT is always selected by one temporal
    # schedule and evaluated using the other schedule, so the center is not
    # forced to be a peak by the alignment operation itself.
    suffix_event_prefix_aligned, suffix_event_prefix_stats = _cross_sweep_event_validation(
        suffix_inc, inc, discovery_label="suffix", validation_label="prefix"
    )
    prefix_event_suffix_aligned, prefix_event_suffix_stats = _cross_sweep_event_validation(
        inc, suffix_inc, discovery_label="prefix", validation_label="suffix"
    )

    rows.to_csv(out / "temporal_cutoff_population_rows.csv.gz", index=False, compression="gzip")
    curves.to_csv(out / "temporal_cutoff_condition_curves.csv", index=False)
    inc.to_csv(out / "temporal_cutoff_condition_incremental_gain.csv", index=False)
    capture_profile.to_csv(out / "temporal_cutoff_capture_profile.csv", index=False)
    gain_profile.to_csv(out / "temporal_cutoff_incremental_gain_profile.csv", index=False)
    condition_stats.to_csv(out / "temporal_cutoff_condition_statistics.csv", index=False)
    _statistics_table(stats).to_csv(out / "temporal_cutoff_statistics.csv", index=False)

    suffix_rows.to_csv(out / "temporal_suffix_population_rows.csv.gz", index=False, compression="gzip")
    suffix_curves.to_csv(out / "temporal_suffix_condition_curves.csv", index=False)
    suffix_inc.to_csv(out / "temporal_suffix_condition_incremental_gain.csv", index=False)
    cross_sweep.to_csv(out / "temporal_cross_sweep_peak_agreement.csv", index=False)
    suffix_event_prefix_aligned.to_csv(out / "temporal_cross_sweep_suffix_event_prefix_aligned.csv", index=False)
    suffix_event_prefix_stats.to_csv(out / "temporal_cross_sweep_suffix_event_prefix_condition_stats.csv", index=False)
    prefix_event_suffix_aligned.to_csv(out / "temporal_cross_sweep_prefix_event_suffix_aligned.csv", index=False)
    prefix_event_suffix_stats.to_csv(out / "temporal_cross_sweep_prefix_event_suffix_condition_stats.csv", index=False)

    _plot_temporal_main(capture_profile, gain_profile, stats, out / "temporal_cutoff_main_pair.pdf")
    _plot_cross_sweep_event_validation(
        suffix_event_prefix_aligned,
        suffix_event_prefix_stats,
        prefix_event_suffix_aligned,
        prefix_event_suffix_stats,
        out / "temporal_cross_sweep_event_validation.pdf",
    )
    _plot_capture(capture_profile, stats, out / "temporal_cutoff_capture_profile.pdf")

    summary = {
        "schema": SCHEMA,
        "input_rows": int(len(rows)),
        "n_audit_rows": int(len(audit)),
        "eligible_decode_only_conditions": int(rows[["run_id", "direction"]].drop_duplicates().shape[0]) if not rows.empty and {"run_id", "direction"}.issubset(rows.columns) else 0,
        "suffix_input_rows": int(len(suffix_rows)),
        "suffix_n_audit_rows": int(len(suffix_audit)),
        "suffix_eligible_decode_only_conditions": int(suffix_rows[["run_id", "direction"]].drop_duplicates().shape[0]) if not suffix_rows.empty and {"run_id", "direction"}.issubset(suffix_rows.columns) else 0,
        "cross_sweep_peak_agreement_conditions": int(len(cross_sweep)),
        "cross_sweep_same_transition_rate": float(pd.to_numeric(cross_sweep.get("same_transition"), errors="coerce").mean()) if not cross_sweep.empty else math.nan,
        "cross_sweep_within_one_transition_rate": float(pd.to_numeric(cross_sweep.get("within_one_transition"), errors="coerce").mean()) if not cross_sweep.empty else math.nan,
        "cross_sweep_suffix_event_prefix_validation": {
            "1to0": _cross_sweep_scope_stats(suffix_event_prefix_stats, direction="1to0"),
            "0to1": _cross_sweep_scope_stats(suffix_event_prefix_stats, direction="0to1"),
            "global": _cross_sweep_scope_stats(suffix_event_prefix_stats, direction=None),
        },
        "cross_sweep_prefix_event_suffix_validation": {
            "1to0": _cross_sweep_scope_stats(prefix_event_suffix_stats, direction="1to0"),
            "0to1": _cross_sweep_scope_stats(prefix_event_suffix_stats, direction="0to1"),
            "global": _cross_sweep_scope_stats(prefix_event_suffix_stats, direction=None),
        },
        "stats": stats,
        "interpretation": (
            "Cumulative prefix capture quantifies how much of the full held-out intervention effect is retained when direct intervention is removed after each autoregressive cutoff. "
            "The historical prefix-only half-horizon/T50/T80 summaries remain descriptive continuity outputs and are not used as independent EVENT definitions. "
            "Peak-share temporal concentration is descriptive only and is not tested against 1/K. "
            "When suffix-on artifacts are present, the inferential temporal-localization view is cross-sweep: one independently generated schedule defines EVENT by its peak transition and the complementary schedule supplies the aligned validation profile. "
            "No suffix values are reconstructed as full-minus-prefix."
        ),
    }
    (out / "temporal_cutoff_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

    if paper is not None:
        paper.mkdir(parents=True, exist_ok=True)
        event_profile, event_stats, strength, strength_stats = _load_stage10_fig4e_inputs(graded)
        _plot_fig4e_event_strength_temporal(
            event_profile, event_stats, strength, strength_stats, capture_profile, stats,
            paper / "fig4e_population_event_and_strength.pdf",
        )
        _plot_temporal_main(capture_profile, gain_profile, stats, paper / "fig4s14_temporal_cutoff_spiking.pdf")
        _plot_capture(capture_profile, stats, paper / "fig4s15_temporal_cutoff_capture_profile.pdf")
        _plot_cross_sweep_event_validation(
            suffix_event_prefix_aligned,
            suffix_event_prefix_stats,
            prefix_event_suffix_aligned,
            prefix_event_suffix_stats,
            paper / "fig4s16_temporal_cross_sweep_event_validation.pdf",
        )

    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
