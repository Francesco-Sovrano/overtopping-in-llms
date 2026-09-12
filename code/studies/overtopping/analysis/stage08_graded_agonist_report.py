#!/usr/bin/env python3
"""Aggregate graded-agonist RQ3 experiments over the exact manuscript population."""
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

from core.project_paths import PROJECT_ROOT
from studies.overtopping.analysis.rq3_population import expected_rq3_sources
from studies.overtopping.analysis.stage09_graded_margin_mechanism_test import (
    PRIMARY_MARGIN as MARGIN_PRIMARY,
    _example_metric_rows as _margin_example_metric_rows,
    _safe_corr as _margin_safe_corr,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--out", default=str(PROJECT_ROOT / "results/analysis/rq3_threshold_event/spiking_diagnostics/graded_agonist"))
    p.add_argument("--paper-figures-dir", default=None)
    p.add_argument("--primary-table", required=True)
    p.add_argument("--population-scope", choices=["primary", "primary+supplementary"], default="primary")
    p.add_argument("--evaluation-split", choices=["test", "train", "all"], default="test")
    p.add_argument("--sampling-max-points", type=int, default=10000)
    p.add_argument("--data-root", default=None)
    p.add_argument(
        "--require-negative-support",
        action="store_true",
        help="Require every expected completed run to have been executed with same-agonist negative support enabled.",
    )
    return p.parse_args()


def _load(root: Path, args: argparse.Namespace, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    args.data_root = str(Path(args.data_root or root).expanduser().resolve())
    expected = expected_rq3_sources(args)
    units: list[pd.DataFrame] = []
    doses: list[pd.DataFrame] = []
    audit: list[dict] = []
    for spec in expected:
        base = Path(spec["stats_dir"]) / "graded_agonist_intervention"
        plan_path = base / "graded_agonist_plan.csv"
        unit_path = base / "graded_agonist_unit_summary.csv"
        dose_path = base / "graded_agonist_dose_rows.csv.gz"
        manifest = base / "graded_agonist_intervention.json"

        status = "missing"
        payload: dict = {}
        negative_requested = False
        manifest_known_examples = 0
        manifest_negative_examples = 0
        plan_known_sampled = 0
        plan_negative_sampled = 0
        unit_n = 0
        dose_n = 0
        consistency_errors: list[str] = []

        if manifest.is_file():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                status = str(payload.get("status", "unknown"))
                negative_requested = bool(payload.get("same_agonist_negative_support", False))
                manifest_known_examples = int(payload.get("n_known_flip_examples_evaluated", 0) or 0)
                manifest_negative_examples = int(payload.get("n_same_agonist_negative_examples_evaluated", 0) or 0)
            except Exception:
                status = "manifest_read_error"
                payload = {}

        plan_df = pd.DataFrame()
        if plan_path.is_file():
            try:
                plan_df = pd.read_csv(plan_path)
                plan_known_sampled = int(pd.to_numeric(plan_df.get("n_known_flip_sampled"), errors="coerce").fillna(0).sum())
                plan_negative_sampled = int(pd.to_numeric(plan_df.get("n_same_agonist_negative_sampled"), errors="coerce").fillna(0).sum())
            except Exception:
                consistency_errors.append("plan_read_error")

        # Never ingest per-run science artifacts unless the manifest declares a
        # completed run. During recomputation Stage 7b writes status=running
        # before changing the plan, so an interrupted refresh cannot leak old
        # dose rows into new aggregate results.
        df_unit = pd.DataFrame()
        df_dose = pd.DataFrame()
        if status == "ok":
            if unit_path.is_file():
                try:
                    df_unit = pd.read_csv(unit_path)
                except Exception:
                    consistency_errors.append("unit_summary_read_error")
            else:
                consistency_errors.append("unit_summary_missing")
            if dose_path.is_file():
                try:
                    df_dose = pd.read_csv(dose_path)
                except Exception:
                    consistency_errors.append("dose_rows_read_error")
            else:
                consistency_errors.append("dose_rows_missing")

            cfg = payload.get("run_config") if isinstance(payload.get("run_config"), dict) else {}
            cfg_negative = cfg.get("same_agonist_negative_support")
            if cfg_negative is not None and bool(cfg_negative) != negative_requested:
                consistency_errors.append("manifest_negative_flag_mismatch")

            expected_doses = payload.get("doses") or cfg.get("doses") or []
            n_doses = len(expected_doses)
            if n_doses <= 0:
                consistency_errors.append("missing_dose_grid")
            elif not df_dose.empty:
                if "support_kind" not in df_dose.columns:
                    consistency_errors.append("dose_rows_missing_support_kind")
                else:
                    known_rows = int((df_dose["support_kind"].astype(str) == "known_flip").sum())
                    neg_rows = int((df_dose["support_kind"].astype(str) == "same_agonist_nonflip").sum())
                    if known_rows % n_doses != 0 or neg_rows % n_doses != 0:
                        consistency_errors.append("dose_row_count_not_divisible_by_doses")
                    actual_known = known_rows // n_doses
                    actual_negative = neg_rows // n_doses
                    if actual_known != manifest_known_examples:
                        consistency_errors.append("manifest_known_count_mismatch")
                    if actual_negative != manifest_negative_examples:
                        consistency_errors.append("manifest_negative_count_mismatch")

            # A completed run should have evaluated exactly the examples it
            # recorded as sampled in the plan. This catches hybrid directories
            # where a new plan was written over old known-flip-only outputs.
            if not plan_df.empty:
                if plan_known_sampled != manifest_known_examples:
                    consistency_errors.append("plan_known_count_mismatch")
                if plan_negative_sampled != manifest_negative_examples:
                    consistency_errors.append("plan_negative_count_mismatch")
                if (not negative_requested) and plan_negative_sampled > 0:
                    consistency_errors.append("negative_plan_with_disabled_manifest")

            if consistency_errors:
                status = "inconsistent_saved_outputs"
                df_unit = pd.DataFrame()
                df_dose = pd.DataFrame()

        if not df_unit.empty:
            df_unit["run_id"] = spec["run_id"]
            df_unit["source_scope"] = spec.get("source_scope", "primary")
            df_unit["task"] = spec["task"]
            df_unit["model"] = spec["model"]
            df_unit["phase"] = spec["phase"]
            df_unit["replacement_baseline"] = spec.get("replacement_baseline")
            df_unit["source_path"] = str(unit_path)
            units.append(df_unit)
            unit_n = len(df_unit)
        if not df_dose.empty:
            df_dose["run_id"] = spec["run_id"]
            df_dose["source_scope"] = spec.get("source_scope", "primary")
            df_dose["task"] = spec["task"]
            df_dose["model"] = spec["model"]
            df_dose["phase"] = spec["phase"]
            doses.append(df_dose)
            dose_n = len(df_dose)

        audit.append({
            "run_id": spec["run_id"],
            "source_scope": spec.get("source_scope", "primary"),
            "task": spec["task"],
            "model": spec["model"],
            "phase": spec["phase"],
            "graded_dir": str(base),
            "status": status,
            "consistency_errors": ";".join(consistency_errors),
            "unit_rows": unit_n,
            "dose_rows": dose_n,
            "same_agonist_negative_support_requested": negative_requested,
            "manifest_known_flip_examples": manifest_known_examples,
            "manifest_same_agonist_nonflip_examples": manifest_negative_examples,
            "plan_known_flip_examples_sampled": plan_known_sampled,
            "plan_same_agonist_nonflip_examples_sampled": plan_negative_sampled,
        })
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(out / "graded_agonist_population_audit.csv", index=False)
    unit_df = pd.concat(units, ignore_index=True) if units else pd.DataFrame()
    dose_df = pd.concat(doses, ignore_index=True) if doses else pd.DataFrame()
    return unit_df, dose_df, audit_df

def _condition_summary(unit_df: pd.DataFrame) -> pd.DataFrame:
    if unit_df.empty:
        return pd.DataFrame()
    d = unit_df.copy()
    cols = [
        "single_crossing_rate",
        "median_first_flip_dose",
        "median_first_persistent_crossing_dose",
        "natural_state_reproduction_rate",
        "full_dose_support_reproduction_rate",
        "full_dose_flip_rate",
    ]
    for col in cols:
        d[col] = pd.to_numeric(d[col], errors="coerce") if col in d.columns else np.nan
    keys = ["run_id", "source_scope", "task", "model", "phase", "direction", "support_kind"]
    return d.groupby(keys, dropna=False).agg(
        n_agonists=("unit_key", "nunique"),
        median_single_crossing_rate=("single_crossing_rate", "median"),
        median_first_flip_dose=("median_first_flip_dose", "median"),
        median_first_persistent_crossing_dose=("median_first_persistent_crossing_dose", "median"),
        median_natural_state_reproduction_rate=("natural_state_reproduction_rate", "median"),
        median_full_dose_support_reproduction_rate=("full_dose_support_reproduction_rate", "median"),
        median_full_dose_flip_rate=("full_dose_flip_rate", "median"),
    ).reset_index()


def _parse_flip_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "false": False, "0": False}
    parsed = normalized.map(mapping)
    if parsed.isna().any():
        bad = series.loc[parsed.isna()].head(5).tolist()
        raise ValueError(f"Could not parse flipped_from_baseline values: {bad}")
    return parsed.astype(bool)


def _dose_condition_summary(dose_df: pd.DataFrame) -> pd.DataFrame:
    if dose_df.empty:
        return pd.DataFrame()
    d = dose_df.copy()
    d["dose"] = pd.to_numeric(d["dose"], errors="coerce")
    d["flipped_from_baseline"] = _parse_flip_series(d["flipped_from_baseline"]).astype(float)
    unit = d.groupby(
        ["run_id", "source_scope", "task", "model", "phase", "direction", "support_kind", "unit_key", "dose"],
        dropna=False,
    )["flipped_from_baseline"].mean().reset_index(name="unit_flip_fraction")
    return unit.groupby(
        ["run_id", "source_scope", "task", "model", "phase", "direction", "support_kind", "dose"],
        dropna=False,
    )["unit_flip_fraction"].median().reset_index(name="condition_median_flip_fraction")


def _trajectory_condition_summary(dose_df: pd.DataFrame) -> pd.DataFrame:
    """Derive support-appropriate trajectory endpoints directly from dose rows.

    For known full-dose flips, single persistent crossing is informative.  For
    same-channel full-dose non-flips, the informative endpoint is stability at
    baseline across the entire dose sweep (and transient intermediate flips),
    not a trivially zero persistent-crossing rate.
    """
    if dose_df.empty:
        return pd.DataFrame()
    d = dose_df.copy()
    d["dose"] = pd.to_numeric(d["dose"], errors="coerce")
    d["_flip"] = _parse_flip_series(d["flipped_from_baseline"])
    example_keys = [
        "run_id", "source_scope", "task", "model", "phase", "direction",
        "support_kind", "unit_key", "evaluation_row", "row_id",
    ]
    rows: list[dict] = []
    for values, g in d.groupby(example_keys, dropna=False, sort=False):
        g = g.sort_values("dose", kind="mergesort")
        flips = g["_flip"].to_numpy(bool)
        endpoint = bool(flips[-1]) if len(flips) else False
        any_flip = bool(flips.any()) if len(flips) else False
        rows.append({
            **dict(zip(example_keys, values)),
            "ever_flipped_at_any_dose": any_flip,
            "ever_flipped_before_full_dose": bool(flips[:-1].any()) if len(flips) > 1 else False,
            "stable_at_baseline_all_doses": bool(not any_flip),
            "transient_flip": bool(any_flip and not endpoint),
            "full_dose_flipped_from_dose_rows": endpoint,
        })
    ex = pd.DataFrame(rows)
    if ex.empty:
        return pd.DataFrame()
    unit_keys = ["run_id", "source_scope", "task", "model", "phase", "direction", "support_kind", "unit_key"]
    unit = ex.groupby(unit_keys, dropna=False).agg(
        any_dose_flip_rate=("ever_flipped_at_any_dose", "mean"),
        intermediate_flip_rate=("ever_flipped_before_full_dose", "mean"),
        stable_all_doses_rate=("stable_at_baseline_all_doses", "mean"),
        transient_flip_rate=("transient_flip", "mean"),
        full_dose_flip_rate_from_dose_rows=("full_dose_flipped_from_dose_rows", "mean"),
    ).reset_index()
    condition_keys = ["run_id", "source_scope", "task", "model", "phase", "direction", "support_kind"]
    return unit.groupby(condition_keys, dropna=False).agg(
        median_any_dose_flip_rate=("any_dose_flip_rate", "median"),
        median_intermediate_flip_rate=("intermediate_flip_rate", "median"),
        median_stable_all_doses_rate=("stable_all_doses_rate", "median"),
        median_transient_flip_rate=("transient_flip_rate", "median"),
        median_full_dose_flip_rate_from_dose_rows=("full_dose_flip_rate_from_dose_rows", "median"),
    ).reset_index()


def _merge_trajectory(condition: pd.DataFrame, trajectory: pd.DataFrame) -> pd.DataFrame:
    if condition.empty:
        return trajectory.copy()
    if trajectory.empty:
        return condition.copy()
    keys = ["run_id", "source_scope", "task", "model", "phase", "direction", "support_kind"]
    return condition.merge(trajectory, on=keys, how="outer", validate="one_to_one")


def _support_contrast(condition: pd.DataFrame) -> pd.DataFrame:
    if condition.empty or "support_kind" not in condition.columns:
        return pd.DataFrame()
    keys = ["run_id", "source_scope", "task", "model", "phase", "direction"]
    known = condition.loc[condition["support_kind"].astype(str) == "known_flip"].drop(columns=["support_kind"])
    nonflip = condition.loc[condition["support_kind"].astype(str) == "same_agonist_nonflip"].drop(columns=["support_kind"])
    if known.empty and nonflip.empty:
        return pd.DataFrame()
    known = known.rename(columns={c: f"{c}_known_flip" for c in known.columns if c not in keys})
    nonflip = nonflip.rename(columns={c: f"{c}_same_agonist_nonflip" for c in nonflip.columns if c not in keys})
    out = known.merge(nonflip, on=keys, how="outer")
    known_cols = [c for c in out.columns if c.endswith("_known_flip")]
    nonflip_cols = [c for c in out.columns if c.endswith("_same_agonist_nonflip")]
    out["has_both_supports"] = out[known_cols].notna().any(axis=1) & out[nonflip_cols].notna().any(axis=1)
    if "median_single_crossing_rate_known_flip" in out and "median_stable_all_doses_rate_same_agonist_nonflip" in out:
        a = pd.to_numeric(out["median_single_crossing_rate_known_flip"], errors="coerce")
        b = pd.to_numeric(out["median_stable_all_doses_rate_same_agonist_nonflip"], errors="coerce")
        out["support_consistency_min"] = np.fmin(a, b)
    return out


def _dose_support_contrast(cond_dose: pd.DataFrame) -> pd.DataFrame:
    if cond_dose.empty:
        return pd.DataFrame()
    keys = ["run_id", "source_scope", "task", "model", "phase", "direction", "dose"]
    p = cond_dose.pivot_table(
        index=keys,
        columns="support_kind",
        values="condition_median_flip_fraction",
        aggfunc="first",
    ).reset_index()
    p.columns.name = None
    if "known_flip" in p.columns:
        p = p.rename(columns={"known_flip": "known_flip_fraction"})
    if "same_agonist_nonflip" in p.columns:
        p = p.rename(columns={"same_agonist_nonflip": "same_agonist_nonflip_fraction"})
    if "known_flip_fraction" in p.columns and "same_agonist_nonflip_fraction" in p.columns:
        p["flip_fraction_contrast"] = p["known_flip_fraction"] - p["same_agonist_nonflip_fraction"]
    return p


def _plot_dose(cond_dose: pd.DataFrame, path: Path) -> None:
    if cond_dose.empty:
        path.unlink(missing_ok=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), sharey=True)
    made = False
    for ax, direction in zip(axes, ["1to0", "0to1"]):
        d = cond_dose.loc[cond_dose["direction"].astype(str) == direction].copy()
        for support_kind, sg in d.groupby("support_kind", sort=False):
            rows = []
            for dose, dg in sg.groupby("dose", sort=True):
                vals = pd.to_numeric(dg["condition_median_flip_fraction"], errors="coerce").dropna().to_numpy(float)
                if len(vals):
                    rows.append((float(dose), float(np.median(vals)), float(np.quantile(vals, .25)), float(np.quantile(vals, .75))))
            if not rows:
                continue
            made = True
            arr = np.asarray(rows, dtype=float)
            display = {
                "known_flip": "known full-dose flip",
                "same_agonist_nonflip": "same-channel full-dose non-flip",
            }.get(str(support_kind), str(support_kind))
            ax.plot(arr[:, 0], arr[:, 1], marker="o", label=display)
            ax.fill_between(arr[:, 0], arr[:, 2], arr[:, 3], alpha=.15)
        ax.set_title("Discovery direction 1→0" if direction == "1to0" else "Discovery direction 0→1")
        ax.set_xlabel("Graded agonist intervention dose")
        ax.set_ylim(-.03, 1.03)
        ax.grid(axis="y", alpha=.25, linewidth=.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Behavior-flipped fraction")
    legend: dict[str, object] = {}
    for ax in axes:
        handles_i, labels_i = ax.get_legend_handles_labels()
        for handle, label in zip(handles_i, labels_i):
            legend.setdefault(label, handle)
    handles = list(legend.values())
    labels = list(legend.keys())
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0, 1, .90 if handles else 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    if made:
        fig.savefig(path, bbox_inches="tight")
    else:
        path.unlink(missing_ok=True)
    plt.close(fig)


def _plot_crossing(condition: pd.DataFrame, path: Path) -> None:
    if condition.empty:
        path.unlink(missing_ok=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), sharey=True)
    made = False
    groups = [
        ("known_flip", "median_single_crossing_rate", "known flip:\nsingle crossing"),
        ("same_agonist_nonflip", "median_stable_all_doses_rate", "same-channel non-flip:\nstable all doses"),
    ]
    for ax, direction in zip(axes, ["1to0", "0to1"]):
        d = condition.loc[condition["direction"].astype(str) == direction].copy()
        for xpos, (support_kind, metric, label) in enumerate(groups):
            if metric not in d.columns:
                continue
            vals = pd.to_numeric(
                d.loc[d["support_kind"].astype(str) == support_kind, metric], errors="coerce"
            ).dropna().to_numpy(float)
            if not len(vals):
                continue
            made = True
            ax.scatter(np.full(len(vals), xpos, dtype=float), vals, s=25, alpha=.75)
            ax.hlines(np.median(vals), xpos - .18, xpos + .18, linewidth=2)
        ax.set_xlim(-.45, 1.45)
        ax.set_xticks([0, 1], [groups[0][2], groups[1][2]], fontsize=7)
        ax.set_title("Discovery direction 1→0" if direction == "1to0" else "Discovery direction 0→1")
        ax.grid(axis="y", alpha=.25, linewidth=.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Support-consistent trajectory rate")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    if made:
        fig.savefig(path, bbox_inches="tight")
    else:
        path.unlink(missing_ok=True)
    plt.close(fig)



def _margin_condition_summary(dose_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute condition-level affine-null diagnostics without treating examples as independent conditions."""
    if dose_df.empty or "endpoint_divergence_margin" not in dose_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    metric = _margin_example_metric_rows(dose_df)
    if metric.empty:
        return metric, pd.DataFrame()
    known = metric.loc[
        metric.get("support_kind", pd.Series(dtype=str)).astype(str).eq("known_flip")
        & metric.get("known_flip_endpoint_reproduced", pd.Series(False, index=metric.index)).fillna(False).astype(bool)
    ].copy()
    # A concatenated population can contain legacy graded rows created before
    # endpoint-margin recording existed. Those rows have the margin columns only
    # because pandas aligned schemas across runs; their values are NaN and must
    # not count as margin-evaluable examples or conditions.
    if "n_margin_doses" in known.columns:
        known = known.loc[pd.to_numeric(known["n_margin_doses"], errors="coerce").fillna(0).ge(3)].copy()
    if {"margin_at_dose0", "margin_at_dose1"}.issubset(known.columns):
        m0 = pd.to_numeric(known["margin_at_dose0"], errors="coerce")
        m1 = pd.to_numeric(known["margin_at_dose1"], errors="coerce")
        known = known.loc[m0.notna() & m1.notna()].copy()
    if known.empty:
        return metric, pd.DataFrame()
    keys = [c for c in ["run_id", "source_scope", "task", "model", "phase", "replacement_baseline", "direction", "margin_type"] if c in known.columns]
    rows = []
    for values, g in known.groupby(keys, dropna=False, sort=False):
        base = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        behavior = pd.to_numeric(g["observed_behavior_persistent_crossing_dose"], errors="coerce")
        actual = pd.to_numeric(g["observed_margin_persistent_crossing_dose"], errors="coerce")
        predicted = pd.to_numeric(g["linear_predicted_zero_crossing_dose"], errors="coerce")
        step = pd.to_numeric(g["dose_grid_step"], errors="coerce")
        valid_actual = behavior.notna() & actual.notna()
        lag = actual - behavior
        exact = valid_actual & lag.abs().le(1e-12)
        within = valid_actual & step.notna() & lag.abs().le(step + 1e-12)
        valid_pred = behavior.notna() & predicted.notna()
        # LOO median crossing-dose null inside the condition.
        null_err = []
        for idx in g.index[valid_pred]:
            train = behavior.drop(index=idx, errors="ignore").dropna()
            if len(train):
                null_err.append(abs(float(np.median(train)) - float(behavior.loc[idx])))
        corr = _margin_safe_corr(actual, behavior)
        pred_corr = _margin_safe_corr(predicted, behavior)
        rows.append({
            **base,
            "n_examples": int(len(g)),
            "n_doses": int(round(float(pd.to_numeric(g.get("n_doses"), errors="coerce").median()))) if pd.to_numeric(g.get("n_doses"), errors="coerce").notna().any() else 0,
            "dose_grid_step": float(pd.to_numeric(g.get("dose_grid_step"), errors="coerce").median()) if pd.to_numeric(g.get("dose_grid_step"), errors="coerce").notna().any() else math.nan,
            "single_persistent_crossing_rate": float(g["behavior_single_persistent_crossing"].fillna(False).astype(bool).mean()),
            "median_linearity_r2": float(pd.to_numeric(g["margin_linearity_r2"], errors="coerce").median()),
            "median_linearity_nrmse": float(pd.to_numeric(g["margin_linearity_nrmse"], errors="coerce").median()),
            "median_max_step_share": float(pd.to_numeric(g["margin_max_step_share"], errors="coerce").median()),
            "median_max_step_concentration_ratio": float(pd.to_numeric(g["margin_max_step_concentration_ratio"], errors="coerce").median()),
            "median_behavior_step_share": float(pd.to_numeric(g["margin_behavior_step_share"], errors="coerce").median()),
            "median_excess_variation_ratio": float(pd.to_numeric(g["margin_excess_variation_ratio"], errors="coerce").median()),
            "median_monotone_endpoint_direction_fraction": float(pd.to_numeric(g["margin_monotone_endpoint_direction_fraction"], errors="coerce").median()),
            "actual_crossing_n": int(valid_actual.sum()),
            "actual_crossing_exact_fraction": float(exact.sum() / valid_actual.sum()) if valid_actual.any() else math.nan,
            "actual_crossing_within_one_step_fraction": float(within.sum() / valid_actual.sum()) if valid_actual.any() else math.nan,
            "median_actual_crossing_lag": float(lag[valid_actual].median()) if valid_actual.any() else math.nan,
            "actual_vs_behavior_spearman_rho": corr.get("spearman_rho", math.nan),
            "actual_vs_behavior_spearman_p": corr.get("p", math.nan),
            "predicted_vs_behavior_spearman_rho": pred_corr.get("spearman_rho", math.nan),
            "predicted_vs_behavior_spearman_p": pred_corr.get("p", math.nan),
            "affine_predicted_crossing_median_abs_error": float((predicted[valid_pred] - behavior[valid_pred]).abs().median()) if valid_pred.any() else math.nan,
            "loo_median_null_median_abs_error": float(np.median(null_err)) if null_err else math.nan,
        })
    return metric, pd.DataFrame(rows)


def _margin_direction_summary(condition: pd.DataFrame) -> pd.DataFrame:
    if condition.empty:
        return pd.DataFrame()
    primary = condition.loc[condition["margin_type"].astype(str).eq(MARGIN_PRIMARY)].copy()
    primary = primary.loc[pd.to_numeric(primary.get("median_linearity_r2"), errors="coerce").notna()].copy()
    # The manuscript protocol declares the 11-dose grid 0,.1,...,1. Quick
    # diagnostics used a five-dose grid in some runs; do not interleave those
    # different x-grids in a single paper trajectory or condition summary.
    primary["paper_standard_grid"] = pd.to_numeric(primary.get("n_doses"), errors="coerce").eq(11)
    rows = []
    for direction, all_g in primary.groupby("direction", dropna=False, sort=True):
        g = all_g.loc[all_g["paper_standard_grid"]].copy()
        if g.empty:
            continue
        def med(col):
            x = pd.to_numeric(g.get(col), errors="coerce").dropna()
            return float(x.median()) if len(x) else math.nan
        def q(col, p):
            x = pd.to_numeric(g.get(col), errors="coerce").dropna()
            return float(x.quantile(p)) if len(x) else math.nan
        pred = pd.to_numeric(g.get("affine_predicted_crossing_median_abs_error"), errors="coerce")
        null = pd.to_numeric(g.get("loo_median_null_median_abs_error"), errors="coerce")
        comparable = pred.notna() & null.notna()
        rows.append({
            "direction": direction,
            "n_margin_evaluable_conditions_all_grids": int(len(all_g)),
            "n_nonstandard_grid_conditions_excluded": int((~all_g["paper_standard_grid"]).sum()),
            "n_conditions": int(len(g)),
            "n_examples": int(pd.to_numeric(g.get("n_examples"), errors="coerce").fillna(0).sum()),
            "median_condition_single_persistent_crossing_rate": med("single_persistent_crossing_rate"),
            "median_condition_linearity_r2": med("median_linearity_r2"),
            "q25_condition_linearity_r2": q("median_linearity_r2", .25),
            "q75_condition_linearity_r2": q("median_linearity_r2", .75),
            "median_condition_linearity_nrmse": med("median_linearity_nrmse"),
            "median_condition_max_step_share": med("median_max_step_share"),
            "median_condition_max_step_concentration_ratio": med("median_max_step_concentration_ratio"),
            "median_condition_behavior_step_share": med("median_behavior_step_share"),
            "median_condition_exact_crossing_alignment": med("actual_crossing_exact_fraction"),
            "median_condition_within_one_step_crossing_alignment": med("actual_crossing_within_one_step_fraction"),
            "median_condition_actual_vs_behavior_rho": med("actual_vs_behavior_spearman_rho"),
            "n_conditions_with_defined_actual_vs_behavior_rho": int(pd.to_numeric(g.get("actual_vs_behavior_spearman_rho"), errors="coerce").notna().sum()),
            "fraction_conditions_affine_predictor_beats_loo_median_null": float((pred[comparable] < null[comparable]).mean()) if comparable.any() else math.nan,
            "n_conditions_affine_vs_null_comparable": int(comparable.sum()),
        })
    return pd.DataFrame(rows)


def _normalized_margin_condition_dose(dose_df: pd.DataFrame) -> pd.DataFrame:
    """Condition-median normalized divergence-token trajectories on the paper grid."""
    if dose_df.empty or "endpoint_divergence_margin" not in dose_df.columns:
        return pd.DataFrame()
    example_keys = [c for c in ["run_id", "source_scope", "task", "model", "phase", "direction", "unit_key", "evaluation_row", "row_id"] if c in dose_df.columns]
    rows = []
    for values, g in dose_df.loc[dose_df.get("support_kind", pd.Series(dtype=str)).astype(str).eq("known_flip")].groupby(example_keys, dropna=False, sort=False):
        g = g.sort_values("dose")
        d = pd.to_numeric(g["dose"], errors="coerce").to_numpy(float)
        unique_d = np.sort(np.unique(d[np.isfinite(d)]))
        if len(unique_d) != 11 or not np.allclose(unique_d, np.linspace(0.0, 1.0, 11), atol=1e-9, rtol=0.0):
            continue
        m = pd.to_numeric(g["endpoint_divergence_margin"], errors="coerce").to_numpy(float)
        i0 = np.flatnonzero(np.isclose(d, 0.0)); i1 = np.flatnonzero(np.isclose(d, 1.0))
        if not len(i0) or not len(i1) or not np.isfinite(m[i0[0]]) or not np.isfinite(m[i1[0]]):
            continue
        span = float(m[i1[0]] - m[i0[0]])
        if abs(span) < 1e-12:
            continue
        base = dict(zip(example_keys, values if isinstance(values, tuple) else (values,)))
        for dose, margin in zip(d, m):
            if np.isfinite(margin):
                rows.append({**base, "dose": float(dose), "normalized_margin_position": float((margin - m[i0[0]]) / span)})
    ex = pd.DataFrame(rows)
    if ex.empty:
        return ex
    cond_keys = [c for c in ["run_id", "source_scope", "task", "model", "phase", "direction", "dose"] if c in ex.columns]
    out = ex.groupby(cond_keys, dropna=False, as_index=False)["normalized_margin_position"].median()
    out["condition_label"] = out.apply(lambda r: f"{r.get('task','?')} | {r.get('model','?')} | {r.get('phase','?')} | {r.get('run_id','?')}", axis=1)
    return out


def _filtered_primary_margin_conditions(condition: pd.DataFrame) -> pd.DataFrame:
    if condition.empty:
        return pd.DataFrame()
    d = condition.loc[condition["margin_type"].astype(str).eq(MARGIN_PRIMARY)].copy()
    d = d.loc[pd.to_numeric(d.get("median_linearity_r2"), errors="coerce").notna()].copy()
    d = d.loc[pd.to_numeric(d.get("n_doses"), errors="coerce").eq(11)].copy()
    if d.empty:
        return d
    d["condition_label"] = d.apply(lambda r: f"{r.get('task','?')} | {r.get('model','?')} | {r.get('phase','?')} | {r.get('run_id','?')}", axis=1)
    return d


def _plot_margin_affine_null(dose_df: pd.DataFrame, path: Path) -> None:
    c = _normalized_margin_condition_dose(dose_df)
    if c.empty:
        path.unlink(missing_ok=True); return
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharex=True, sharey=True)
    made = False
    for ax, direction in zip(axes, ["0to1", "1to0"]):
        g = c.loc[c["direction"].astype(str).eq(direction)].copy()
        if g.empty:
            ax.set_axis_off(); continue
        made = True
        for _, cg in g.groupby("condition_label", sort=False):
            cg = cg.sort_values("dose")
            ax.plot(cg["dose"], cg["normalized_margin_position"], linewidth=.9, alpha=.20)
        rows=[]
        for dose, dg in g.groupby("dose", sort=True):
            vals=pd.to_numeric(dg["normalized_margin_position"], errors="coerce").dropna().to_numpy(float)
            if len(vals):
                rows.append((float(dose), float(np.median(vals)), float(np.quantile(vals,.25)), float(np.quantile(vals,.75)), len(vals)))
        arr=np.asarray(rows,float)
        ax.fill_between(arr[:,0],arr[:,2],arr[:,3],alpha=.15,label="condition IQR")
        ax.plot(arr[:,0],arr[:,1],marker="o",linewidth=2.2,label="condition median")
        grid=np.linspace(0,1,101); ax.plot(grid,grid,linestyle="--",linewidth=1.2,label="endpoint-affine null")
        n_cond = int(g["condition_label"].nunique())
        ax.text(0.02, 0.98, f"n conditions = {n_cond}", transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
        ax.set_title("Discovery direction 0→1" if direction=="0to1" else "Discovery direction 1→0")
        ax.set_xlabel("Intervention dose"); ax.grid(alpha=.22,linewidth=.45)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Normalized downstream margin position")
    handles,labels=axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles,labels,frameon=False,loc="upper center",ncol=3)
        fig.tight_layout(rect=(0,0,1,.88))
    else: fig.tight_layout()
    path.parent.mkdir(parents=True,exist_ok=True)
    if made: fig.savefig(path,bbox_inches="tight")
    else: path.unlink(missing_ok=True)
    plt.close(fig)


def _plot_margin_condition_summary(condition: pd.DataFrame, path: Path) -> None:
    d = _filtered_primary_margin_conditions(condition)
    if d.empty:
        path.unlink(missing_ok=True); return
    rng = np.random.default_rng(0)
    metrics = [
        ("median_linearity_r2", "Endpoint-chord $R^2$", (0.0, 1.0)),
        ("median_max_step_concentration_ratio", "Max-step concentration / uniform", (0.8, max(2.6, float(pd.to_numeric(d['median_max_step_concentration_ratio'], errors='coerce').max()) + 0.1))),
        ("actual_crossing_exact_fraction", "Exact crossing alignment", (0.0, 1.0)),
        ("actual_crossing_within_one_step_fraction", "Within-one-step alignment", (0.0, 1.0)),
    ]
    fig,axes=plt.subplots(1,4,figsize=(12.0,3.2))
    for ax,(metric,label,ylim) in zip(axes,metrics):
        for xpos,direction in enumerate(["0to1","1to0"]):
            vals=pd.to_numeric(d.loc[d["direction"].astype(str).eq(direction),metric],errors="coerce").dropna().to_numpy(float)
            if len(vals):
                jitter = rng.uniform(-0.08, 0.08, size=len(vals))
                ax.scatter(np.full(len(vals),xpos)+jitter, vals, s=24, alpha=.7)
                q1, med, q3 = np.quantile(vals,[.25,.5,.75])
                ax.vlines(xpos, q1, q3, linewidth=3, alpha=.6)
                ax.hlines(med, xpos-.18, xpos+.18, linewidth=2)
                ax.text(xpos, ylim[1] - 0.03*(ylim[1]-ylim[0]), f"n={len(vals)}", ha='center', va='top', fontsize=8)
        ax.set_xticks([0,1],["0→1","1→0"]); ax.set_ylabel(label); ax.grid(axis="y",alpha=.22,linewidth=.45)
        ax.set_ylim(*ylim)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,bbox_inches="tight"); plt.close(fig)


def _plot_margin_condition_heatmap(dose_df: pd.DataFrame, condition: pd.DataFrame, path: Path) -> None:
    c = _normalized_margin_condition_dose(dose_df)
    cond = _filtered_primary_margin_conditions(condition)
    if c.empty or cond.empty:
        path.unlink(missing_ok=True); return
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.8), sharex=True)
    made = False
    for ax, direction in zip(axes, ["0to1", "1to0"]):
        cg = c.loc[c["direction"].astype(str).eq(direction)].copy()
        dg = cond.loc[cond["direction"].astype(str).eq(direction)].copy()
        if cg.empty or dg.empty:
            ax.set_axis_off(); continue
        made = True
        order = dg.sort_values(["median_linearity_r2", "median_max_step_concentration_ratio"], ascending=[True, False])["condition_label"].tolist()
        pivot = cg.pivot_table(index="condition_label", columns="dose", values="normalized_margin_position", aggfunc="median")
        pivot = pivot.reindex(order)
        arr = pivot.to_numpy(float)
        im = ax.imshow(arr, aspect='auto', interpolation='nearest', vmin=0.0, vmax=1.0)
        cols = list(pivot.columns)
        ax.set_xticks(range(len(cols)), [f"{x:.1f}" for x in cols], rotation=0)
        ytick = np.arange(len(order))
        labels=[]
        for lab in order:
            row = dg.loc[dg['condition_label'].eq(lab)].iloc[0]
            labels.append(f"{row['task']} | {row['model']} | {row['phase']}")
        ax.set_yticks(ytick, labels, fontsize=7)
        ax.set_title("0→1 conditions, sorted by lower $R^2$ first" if direction=="0to1" else "1→0 conditions, sorted by lower $R^2$ first")
        ax.set_xlabel("Intervention dose")
    axes[0].set_ylabel("Condition")
    if made:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.92)
        cbar.set_label("Normalized downstream margin position")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches='tight')
    else:
        path.unlink(missing_ok=True)
    plt.close(fig)



def _attach_competence_reach(df: pd.DataFrame, primary_table_path: str | Path) -> pd.DataFrame:
    """Attach the manuscript competence score and direction-matched causal reach.

    Reach is U_{0->1}(J) for 0to1 rows and U_{1->0}(J) for 1to0 rows.
    R_ov is intentionally not used: it is an overlap/redundancy quantity, not reach.
    """
    if df.empty:
        return df.copy()
    table = pd.read_csv(Path(primary_table_path).expanduser().resolve())
    meta=[]
    for i,row in table.iterrows():
        meta.append({
            "run_id": f"primary_row_{i:02d}",
            "competence": pd.to_numeric(pd.Series([row.get("score")]), errors="coerce").iloc[0],
            "raw_task_score": pd.to_numeric(pd.Series([row.get("raw")]), errors="coerce").iloc[0],
            "chance_baseline": pd.to_numeric(pd.Series([row.get("chance")]), errors="coerce").iloc[0],
            "causal_reach_0to1": pd.to_numeric(pd.Series([row.get("U_J_i2c")]), errors="coerce").iloc[0],
            "causal_reach_1to0": pd.to_numeric(pd.Series([row.get("U_J_c2i")]), errors="coerce").iloc[0],
        })
    out=df.merge(pd.DataFrame(meta), on="run_id", how="left", validate="many_to_one")
    out["directional_causal_reach"] = np.where(
        out["direction"].astype(str).eq("0to1"),
        pd.to_numeric(out["causal_reach_0to1"], errors="coerce"),
        pd.to_numeric(out["causal_reach_1to0"], errors="coerce"),
    )
    return out


def _grid_size_by_condition(dose_df: pd.DataFrame) -> pd.DataFrame:
    if dose_df.empty:
        return pd.DataFrame()
    keys=[c for c in ["run_id","source_scope","direction","support_kind"] if c in dose_df.columns]
    if not keys or "dose" not in dose_df.columns:
        return pd.DataFrame()
    return dose_df.groupby(keys,dropna=False)["dose"].nunique().reset_index(name="n_doses")


def _behavior_competence_reach(condition: pd.DataFrame, dose_df: pd.DataFrame, primary_table_path: str | Path) -> pd.DataFrame:
    """One row per primary run/direction for the broad graded behavioral signature."""
    if condition.empty:
        return pd.DataFrame()
    d=condition.loc[condition.get("source_scope",pd.Series(dtype=str)).astype(str).eq("primary")].copy()
    grid=_grid_size_by_condition(dose_df)
    if not grid.empty:
        d=d.merge(grid,on=[c for c in ["run_id","source_scope","direction","support_kind"] if c in d.columns],how="left")
    # Keep the declared paper grid only, so bin summaries do not mix resolutions.
    d=d.loc[pd.to_numeric(d.get("n_doses"),errors="coerce").eq(11)].copy()
    known=d.loc[d.get("support_kind",pd.Series(dtype=str)).astype(str).eq("known_flip")].copy()
    if known.empty:
        return pd.DataFrame()
    keep=[c for c in ["run_id","source_scope","task","model","phase","direction","n_agonists","n_doses",
                      "median_single_crossing_rate","median_first_persistent_crossing_dose",
                      "median_natural_state_reproduction_rate","median_full_dose_support_reproduction_rate"] if c in known.columns]
    out=known[keep].copy()
    non=d.loc[d.get("support_kind",pd.Series(dtype=str)).astype(str).eq("same_agonist_nonflip")].copy()
    if not non.empty:
        nk=[c for c in ["run_id","source_scope","direction"] if c in non.columns]
        aux=non[nk+[c for c in ["median_stable_all_doses_rate","median_transient_flip_rate","median_full_dose_support_reproduction_rate"] if c in non.columns]].copy()
        aux=aux.rename(columns={
            "median_stable_all_doses_rate":"median_same_agonist_nonflip_stability",
            "median_transient_flip_rate":"median_same_agonist_nonflip_transient_rate",
            "median_full_dose_support_reproduction_rate":"median_nonflip_endpoint_reproduction",
        })
        out=out.merge(aux,on=nk,how="left",validate="one_to_one")
    out=out.rename(columns={
        "median_single_crossing_rate":"single_persistent_crossing_rate",
        "median_first_persistent_crossing_dose":"first_persistent_crossing_dose",
        "median_natural_state_reproduction_rate":"dose0_natural_state_reproduction",
        "median_full_dose_support_reproduction_rate":"known_flip_endpoint_reproduction",
    })
    return _attach_competence_reach(out,primary_table_path)


def _margin_competence_reach(margin_condition: pd.DataFrame, primary_table_path: str | Path) -> pd.DataFrame:
    """One row per primary run/direction for the continuous-margin signature."""
    d=_filtered_primary_margin_conditions(margin_condition)
    if d.empty:
        return d
    d=d.loc[d.get("source_scope",pd.Series(dtype=str)).astype(str).eq("primary")].copy()
    return _attach_competence_reach(d,primary_table_path)


def _fixed_third_bin(v):
    try: x=float(v)
    except Exception: return np.nan
    if not np.isfinite(x): return np.nan
    if x < 1.0/3.0: return "low"
    if x < 2.0/3.0: return "mid"
    return "high"


def _stratified_summary(df: pd.DataFrame, metrics: list[str], family: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    d=df.copy()
    d["competence_stratum"]=d["competence"].map(_fixed_third_bin)
    d["reach_stratum"]=d["directional_causal_reach"].map(_fixed_third_bin)
    rows=[]
    for (phase,direction),g in d.groupby(["phase","direction"],dropna=False,sort=True):
        for metric in metrics:
            vals=pd.to_numeric(g.get(metric),errors="coerce")
            tmp=g.assign(_metric=vals).dropna(subset=["competence_stratum","reach_stratum","_metric"])
            for (cb,rb),cell in tmp.groupby(["competence_stratum","reach_stratum"],dropna=False):
                x=cell["_metric"].to_numpy(float)
                rows.append({
                    "family":family,"phase":phase,"direction":direction,"metric":metric,
                    "competence_stratum":cb,"reach_stratum":rb,
                    "competence_lower":0.0 if cb=="low" else (1/3 if cb=="mid" else 2/3),
                    "reach_lower":0.0 if rb=="low" else (1/3 if rb=="mid" else 2/3),
                    "n_conditions":int(len(x)),"median":float(np.median(x)),
                    "q25":float(np.quantile(x,.25)),"q75":float(np.quantile(x,.75)),
                })
    return pd.DataFrame(rows)


def _partial_spearman(x, y, z) -> tuple[int,float]:
    frame=pd.DataFrame({"x":pd.to_numeric(x,errors="coerce"),"y":pd.to_numeric(y,errors="coerce"),"z":pd.to_numeric(z,errors="coerce")}).dropna()
    if len(frame)<4 or frame.x.nunique()<2 or frame.y.nunique()<2 or frame.z.nunique()<2:
        return len(frame),math.nan
    xr=frame.x.rank(method="average").to_numpy(float); yr=frame.y.rank(method="average").to_numpy(float); zr=frame.z.rank(method="average").to_numpy(float)
    X=np.column_stack([np.ones(len(zr)),zr])
    rx=xr-X@np.linalg.lstsq(X,xr,rcond=None)[0]; ry=yr-X@np.linalg.lstsq(X,yr,rcond=None)[0]
    if np.std(rx)<=1e-12 or np.std(ry)<=1e-12: return len(frame),math.nan
    return len(frame),float(np.corrcoef(rx,ry)[0,1])


def _association_summary(df: pd.DataFrame, metrics: list[str], family: str) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    rows=[]
    for (phase,direction),g in df.groupby(["phase","direction"],dropna=False,sort=True):
        for metric in metrics:
            y=pd.to_numeric(g.get(metric),errors="coerce")
            comp=_margin_safe_corr(g.get("competence"),y)
            reach=_margin_safe_corr(g.get("directional_causal_reach"),y)
            n_pc,r_pc=_partial_spearman(g.get("competence"),y,g.get("directional_causal_reach"))
            n_pr,r_pr=_partial_spearman(g.get("directional_causal_reach"),y,g.get("competence"))
            valid=g.assign(_y=y).dropna(subset=["competence","directional_causal_reach","_y"])
            hh=valid.loc[(valid.competence>=2/3)&(valid.directional_causal_reach>=2/3),"_y"]
            other=valid.loc[~((valid.competence>=2/3)&(valid.directional_causal_reach>=2/3)),"_y"]
            rows.append({
                "family":family,"phase":phase,"direction":direction,"metric":metric,
                "n_competence":comp.get("n",0),"spearman_competence":comp.get("spearman_rho",math.nan),"p_competence":comp.get("p",math.nan),
                "n_reach":reach.get("n",0),"spearman_directional_reach":reach.get("spearman_rho",math.nan),"p_directional_reach":reach.get("p",math.nan),
                "n_partial":min(n_pc,n_pr),"partial_spearman_competence_given_reach":r_pc,"partial_spearman_reach_given_competence":r_pr,
                "high_competence_high_reach_n":int(len(hh)),"high_competence_high_reach_median":float(hh.median()) if len(hh) else math.nan,
                "other_n":int(len(other)),"other_median":float(other.median()) if len(other) else math.nan,
                "high_high_minus_other_median":float(hh.median()-other.median()) if len(hh) and len(other) else math.nan,
            })
    return pd.DataFrame(rows)


def _plot_strata_heatmaps(strata: pd.DataFrame, metrics: list[tuple[str,str]], path: Path, title_prefix: str, phase: str) -> None:
    if strata.empty:
        path.unlink(missing_ok=True); return
    strata=strata.loc[strata.get("phase",pd.Series(dtype=str)).astype(str).eq(str(phase))].copy()
    if strata.empty:
        path.unlink(missing_ok=True); return
    dirs=[d for d in ["0to1","1to0"] if (strata.direction.astype(str)==d).any()]
    if not dirs:
        path.unlink(missing_ok=True); return
    nrows=len(metrics); ncols=len(dirs)
    fig,axes=plt.subplots(nrows,ncols,figsize=(4.3*ncols,3.25*nrows),squeeze=False)
    order=["low","mid","high"]
    made=False
    for r,(metric,label) in enumerate(metrics):
        md=strata.loc[strata.metric.eq(metric)]
        finite=pd.to_numeric(md.get("median"),errors="coerce").dropna()
        if metric in {"single_persistent_crossing_rate","median_same_agonist_nonflip_stability","actual_crossing_within_one_step_fraction"}:
            vmin,vmax=0.0,1.0
        else:
            vmin=float(finite.min()) if len(finite) else 0.0; vmax=float(finite.max()) if len(finite) else 1.0
            if abs(vmax-vmin)<1e-12: vmax=vmin+1.0
        for c,direction in enumerate(dirs):
            ax=axes[r,c]; g=md.loc[md.direction.astype(str).eq(direction)]
            arr=np.full((3,3),np.nan); ns=np.zeros((3,3),dtype=int)
            for _,row in g.iterrows():
                if row.competence_stratum in order and row.reach_stratum in order:
                    yi=order.index(row.reach_stratum); xi=order.index(row.competence_stratum)
                    arr[yi,xi]=float(row["median"]); ns[yi,xi]=int(row["n_conditions"])
            masked=np.ma.masked_invalid(arr)
            im=ax.imshow(masked,origin="lower",aspect="equal",vmin=vmin,vmax=vmax)
            for yi in range(3):
                for xi in range(3):
                    if np.isfinite(arr[yi,xi]): ax.text(xi,yi,f"{arr[yi,xi]:.2f}\nn={ns[yi,xi]}",ha="center",va="center",fontsize=8)
                    else: ax.text(xi,yi,"—",ha="center",va="center",fontsize=9)
            ax.set_xticks(range(3),["low","mid","high"]); ax.set_yticks(range(3),["low","mid","high"])
            ax.set_xlabel("Competence stratum"); ax.set_ylabel("Directional causal reach stratum")
            ax.set_title(f"{direction.replace('to','→')} | {label}")
            fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
            made=True
    fig.suptitle(f"{title_prefix} | phase={phase}\nFixed strata: low < 1/3, mid 1/3–2/3, high ≥ 2/3",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.96)); path.parent.mkdir(parents=True,exist_ok=True)
    if made: fig.savefig(path,bbox_inches="tight")
    else: path.unlink(missing_ok=True)
    plt.close(fig)


def _write_competence_reach_text(assoc: pd.DataFrame, out: Path) -> None:
    lines=["## Graded geometry stratified by competence and directional causal reach","",
           "Competence is the manuscript phase-specific competence score. Because I+O uses raw task success while Out uses the chance-normalized score, all competence/reach strata and continuous associations are reported separately by intervention phase. Reach is direction matched: U_{0→1}(J) for 0→1 graded trajectories and U_{1→0}(J) for 1→0 trajectories. R_ov is not used because it measures overlap/redundancy rather than causal reach.",""]
    if assoc.empty:
        lines.append("No evaluable primary conditions were available for this stratification.")
    else:
        for family in assoc.family.dropna().unique():
            lines.append(f"### {family}")
            for phase in ["I+O","Out"]:
                pg=assoc.loc[(assoc.family==family)&(assoc.phase.astype(str)==phase)]
                if pg.empty: continue
                lines.append(f"Phase {phase}:")
                for direction in ["0to1","1to0"]:
                    g=pg.loc[pg.direction.astype(str)==direction]
                    if g.empty: continue
                    lines.append(f"Direction {direction}:")
                    for _,r in g.iterrows():
                        lines.append(f"- {r.metric}: rho_comp={_fmt_margin(r.spearman_competence)}, rho_reach={_fmt_margin(r.spearman_directional_reach)}, partial rho_comp|reach={_fmt_margin(r.partial_spearman_competence_given_reach)}, partial rho_reach|comp={_fmt_margin(r.partial_spearman_reach_given_competence)}, high-high n={int(r.high_competence_high_reach_n)}, high-high minus other median={_fmt_margin(r.high_high_minus_other_median)}.")
    lines.extend(["", "Interpretation: these are descriptive cross-setting stratifications. A pattern concentrated at high competence and high directional reach would support enrichment of the spiking-like response geometry in capable, causally reachable regimes. Absence of such a pattern is evidence against that simple emergence hypothesis. Because tasks, model families, checkpoints, and replacement conditions are reused across settings, standard condition-level correlations are exploratory rather than independent-sample causal tests. Cross-sectional stratification also does not by itself establish temporal emergence; that stronger claim requires graded measurements across learning checkpoints."])
    (out/"graded_competence_reach_interpretation.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def _fmt_margin(x, digits=3):
    try: x=float(x)
    except Exception: return "NA"
    return f"{x:.{digits}f}" if np.isfinite(x) else "NA"


def _write_margin_paper_text(direction: pd.DataFrame, out: Path) -> None:
    if direction.empty:
        return
    md=[]; tex=[]
    for direction_name in ["0to1","1to0"]:
        g=direction.loc[direction["direction"].astype(str).eq(direction_name)]
        if g.empty: continue
        r=g.iloc[0]; n=int(r.get("n_conditions",0)); nex=int(r.get("n_examples",0))
        phrase=(
            f"Across {n} condition(s) in the {direction_name} discovery direction ({nex} endpoint-reproduced known-flip examples), "
            f"the median condition single-persistent-crossing rate was {_fmt_margin(r.get('median_condition_single_persistent_crossing_rate'))}. "
            f"The continuous divergence-token margin had median condition endpoint-chord R^2={_fmt_margin(r.get('median_condition_linearity_r2'))} "
            f"and normalized RMSE={_fmt_margin(r.get('median_condition_linearity_nrmse'))}; its largest adjacent-dose step accounted for "
            f"a median {_fmt_margin(r.get('median_condition_max_step_share'))} of total path variation "
            f"({_fmt_margin(r.get('median_condition_max_step_concentration_ratio'))}x the uniform-per-step share). "
            f"The median condition exact-grid margin/behavior crossing alignment was {_fmt_margin(r.get('median_condition_exact_crossing_alignment'))}, "
            f"and within-one-dose-step alignment was {_fmt_margin(r.get('median_condition_within_one_step_crossing_alignment'))}."
        )
        md.append(phrase)
        dtex="$0\\to1$" if direction_name=="0to1" else "$1\\to0$"
        tex.append(
            f"Across {n} condition(s) in the {dtex} discovery direction ({nex} endpoint-reproduced known-flip examples), the median condition single-persistent-crossing rate was {_fmt_margin(r.get('median_condition_single_persistent_crossing_rate'))}. The continuous divergence-token margin had median condition endpoint-chord $R^2={_fmt_margin(r.get('median_condition_linearity_r2'))}$ and normalized RMSE $={_fmt_margin(r.get('median_condition_linearity_nrmse'))}$; its largest adjacent-dose step accounted for a median {_fmt_margin(r.get('median_condition_max_step_share'))} of total path variation."
        )
    caveat=(
        "Interpretation: the intervened source coordinate is linear in dose by construction. These measurements therefore quantify downstream nonlinearity and transition concentration, not nonlinearity of the intervention itself. A poor affine fit directly challenges the simple affine-margin-plus-binary-readout construction for the measured margin, while remaining compatible with a smooth but nonlinear network. Accordingly, these results support the operational description 'spiking-like causal response geometry' but do not uniquely identify a discontinuous or biological spiking mechanism."
    )
    md.append(caveat); tex.append(caveat.replace("'spiking-like causal response geometry'","\\emph{spiking-like causal response geometry}"))
    (out/"graded_margin_paper_interpretation.md").write_text("## Nonlinear graded-response result\n\n"+"\n\n".join(md)+"\n",encoding="utf-8")
    (out/"graded_margin_paper_interpretation.tex").write_text("% Auto-generated; condition is the inferential unit.\n"+"\n\n".join(tex)+"\n",encoding="utf-8")

def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    unit_df, dose_df, audit_df = _load(root, args, out)
    unit_df.to_csv(out / "graded_agonist_all_units.csv", index=False)
    condition = _merge_trajectory(_condition_summary(unit_df), _trajectory_condition_summary(dose_df))
    condition.to_csv(out / "graded_agonist_by_condition.csv", index=False)
    cond_dose = _dose_condition_summary(dose_df)
    cond_dose.to_csv(out / "graded_agonist_dose_by_condition.csv", index=False)
    contrast = _support_contrast(condition)
    contrast.to_csv(out / "graded_agonist_support_contrast_by_condition.csv", index=False)
    dose_contrast = _dose_support_contrast(cond_dose)
    dose_contrast.to_csv(out / "graded_agonist_dose_support_contrast.csv", index=False)

    margin_examples, margin_condition = _margin_condition_summary(dose_df)
    margin_examples.to_csv(out / "graded_margin_all_examples.csv", index=False)
    margin_condition.to_csv(out / "graded_margin_by_condition.csv", index=False)
    margin_direction = _margin_direction_summary(margin_condition)
    margin_direction.to_csv(out / "graded_margin_by_direction.csv", index=False)
    _plot_margin_affine_null(dose_df, out / "graded_margin_affine_null.pdf")
    _plot_margin_condition_summary(margin_condition, out / "graded_margin_condition_diagnostics.pdf")
    _plot_margin_condition_heatmap(dose_df, margin_condition, out / "graded_margin_condition_heatmap.pdf")
    _write_margin_paper_text(margin_direction, out)

    # Competence/reach stratification. Keep broad behavioral geometry separate
    # from the smaller continuous-margin subset.
    behavior_cr = _behavior_competence_reach(condition, dose_df, args.primary_table)
    behavior_cr.to_csv(out / "graded_behavior_competence_reach_joined.csv", index=False)
    margin_cr = _margin_competence_reach(margin_condition, args.primary_table)
    margin_cr.to_csv(out / "graded_margin_competence_reach_joined.csv", index=False)
    behavior_metrics = ["single_persistent_crossing_rate", "median_same_agonist_nonflip_stability"]
    margin_metrics = ["median_linearity_nrmse", "median_max_step_concentration_ratio", "actual_crossing_within_one_step_fraction"]
    behavior_strata = _stratified_summary(behavior_cr, behavior_metrics, "behavioral_graded_geometry")
    margin_strata = _stratified_summary(margin_cr, margin_metrics, "continuous_margin_geometry")
    behavior_strata.to_csv(out / "graded_behavior_competence_reach_strata.csv", index=False)
    margin_strata.to_csv(out / "graded_margin_competence_reach_strata.csv", index=False)
    associations = pd.concat([
        _association_summary(behavior_cr, behavior_metrics, "behavioral_graded_geometry"),
        _association_summary(margin_cr, margin_metrics, "continuous_margin_geometry"),
    ], ignore_index=True)
    associations.to_csv(out / "graded_competence_reach_associations.csv", index=False)
    for phase, suffix in [("I+O", "io"), ("Out", "out")]:
        _plot_strata_heatmaps(
            behavior_strata,
            [("single_persistent_crossing_rate", "Single persistent crossing"),
             ("median_same_agonist_nonflip_stability", "Same-agonist non-flip stability")],
            out / f"graded_behavior_competence_reach_{suffix}.pdf",
            "Behavioral graded geometry by competence and reach",
            phase,
        )
        _plot_strata_heatmaps(
            margin_strata,
            [("median_linearity_nrmse", "Affine normalized RMSE"),
             ("median_max_step_concentration_ratio", "Transition concentration / uniform"),
             ("actual_crossing_within_one_step_fraction", "Margin/behavior alignment within one step")],
            out / f"graded_margin_competence_reach_{suffix}.pdf",
            "Continuous-margin geometry by competence and reach",
            phase,
        )
    _write_competence_reach_text(associations, out)

    _plot_dose(cond_dose, out / "graded_agonist_dose_response.pdf")
    _plot_crossing(condition, out / "graded_agonist_support_consistency.pdf")
    if args.paper_figures_dir:
        paper = Path(args.paper_figures_dir).expanduser().resolve()
        paper.mkdir(parents=True, exist_ok=True)
        for stale in [
            "fig4c_preemption.pdf",
            "fig4d_graded_agonist_dose_response.pdf",
            "fig4s6_graded_agonist_single_crossing.pdf",
            "fig4s7_graded_margin_condition_heatmap.pdf",
        ]:
            (paper / stale).unlink(missing_ok=True)
        _plot_dose(cond_dose, paper / "fig4c_graded_agonist_dose_response.pdf")
        _plot_crossing(condition, paper / "fig4s5_graded_agonist_single_crossing.pdf")
        _plot_margin_affine_null(dose_df, paper / "fig4d_graded_margin_affine_null.pdf")
        _plot_margin_condition_summary(margin_condition, paper / "fig4s6_graded_margin_condition_diagnostics.pdf")
        _plot_margin_condition_heatmap(dose_df, margin_condition, paper / "fig4s7_graded_margin_condition_heatmap.pdf")
        for phase, suffix in [("I+O", "io"), ("Out", "out")]:
            _plot_strata_heatmaps(
                behavior_strata,
                [("single_persistent_crossing_rate", "Single persistent crossing"),
                 ("median_same_agonist_nonflip_stability", "Same-agonist non-flip stability")],
                paper / f"fig4s8_graded_behavior_competence_reach_{suffix}.pdf",
                "Behavioral graded geometry by competence and reach",
                phase,
            )
            _plot_strata_heatmaps(
                margin_strata,
                [("median_linearity_nrmse", "Affine normalized RMSE"),
                 ("median_max_step_concentration_ratio", "Transition concentration / uniform"),
                 ("actual_crossing_within_one_step_fraction", "Margin/behavior alignment within one step")],
                paper / f"fig4s9_graded_margin_competence_reach_{suffix}.pdf",
                "Continuous-margin geometry by competence and reach",
                phase,
            )

    known = condition.loc[condition.get("support_kind", pd.Series(dtype=str)).astype(str) == "known_flip"] if not condition.empty else pd.DataFrame()
    nonflip = condition.loc[condition.get("support_kind", pd.Series(dtype=str)).astype(str) == "same_agonist_nonflip"] if not condition.empty else pd.DataFrame()
    rates = pd.to_numeric(known.get("median_single_crossing_rate"), errors="coerce").dropna().to_numpy(float) if not known.empty else np.asarray([])
    first = pd.to_numeric(known.get("median_first_flip_dose"), errors="coerce").dropna().to_numpy(float) if not known.empty else np.asarray([])
    persistent_first = pd.to_numeric(known.get("median_first_persistent_crossing_dose"), errors="coerce").dropna().to_numpy(float) if (not known.empty and "median_first_persistent_crossing_dose" in known.columns) else np.asarray([])
    stable_nonflip = pd.to_numeric(nonflip.get("median_stable_all_doses_rate"), errors="coerce").dropna().to_numpy(float) if not nonflip.empty else np.asarray([])
    transient_nonflip = pd.to_numeric(nonflip.get("median_transient_flip_rate"), errors="coerce").dropna().to_numpy(float) if not nonflip.empty else np.asarray([])
    requested_runs = int(audit_df.get("same_agonist_negative_support_requested", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not audit_df.empty else 0
    manifest_negative_examples = int(pd.to_numeric(audit_df.get("manifest_same_agonist_nonflip_examples", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not audit_df.empty else 0
    expected_runs = int(len(audit_df))
    completed_runs = int((audit_df.get("status", pd.Series(dtype=str)).astype(str) == "ok").sum()) if not audit_df.empty else 0
    inconsistent_runs = int((audit_df.get("status", pd.Series(dtype=str)).astype(str) == "inconsistent_saved_outputs").sum()) if not audit_df.empty else 0
    running_runs = int((audit_df.get("status", pd.Series(dtype=str)).astype(str) == "running").sum()) if not audit_df.empty else 0

    if inconsistent_runs:
        report_status = "inconsistent_saved_outputs"
    elif running_runs:
        report_status = "refresh_in_progress"
    elif args.require_negative_support and requested_runs < expected_runs:
        report_status = "negative_support_refresh_incomplete"
    elif not len(known):
        report_status = "not_available"
    elif requested_runs and manifest_negative_examples and not len(nonflip):
        report_status = "incomplete_negative_support_reporting"
    elif requested_runs and len(nonflip):
        report_status = "ok"
    else:
        report_status = "known_flip_only"

    status = {
        "status": report_status,
        "scientific_target": "graded causal trajectories on same-agonist held-out flip and non-flip support",
        "n_conditions_known_flip": int(len(known)),
        "n_conditions_same_agonist_nonflip": int(len(nonflip)),
        "n_conditions_with_both_supports": int(contrast.get("has_both_supports", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not contrast.empty else 0,
        "expected_runs": expected_runs,
        "completed_runs": completed_runs,
        "running_runs": running_runs,
        "inconsistent_saved_output_runs": inconsistent_runs,
        "negative_support_requested_runs": requested_runs,
        "manifest_same_agonist_nonflip_examples": manifest_negative_examples,
        "median_condition_single_crossing_rate_known_flip": float(np.median(rates)) if len(rates) else math.nan,
        "median_condition_first_flip_dose_known_flip": float(np.median(first)) if len(first) else math.nan,
        "median_condition_first_persistent_crossing_dose_known_flip": float(np.median(persistent_first)) if len(persistent_first) else math.nan,
        "median_condition_stable_all_doses_rate_same_agonist_nonflip": float(np.median(stable_nonflip)) if len(stable_nonflip) else math.nan,
        "median_condition_transient_flip_rate_same_agonist_nonflip": float(np.median(transient_nonflip)) if len(transient_nonflip) else math.nan,
        "graded_margin_n_evaluable_conditions_all_grids": int(len(margin_condition.loc[margin_condition.get("margin_type", pd.Series(dtype=str)).astype(str).eq(MARGIN_PRIMARY)])) if not margin_condition.empty else 0,
        "graded_margin_n_paper_standard_grid_conditions": int(pd.to_numeric(margin_direction.get("n_conditions"), errors="coerce").fillna(0).sum()) if not margin_direction.empty else 0,
        "graded_margin_direction_summary": margin_direction.to_dict(orient="records") if not margin_direction.empty else [],
        "graded_margin_interpretation": "source intervention is linear by construction; margin diagnostics quantify downstream non-affinity, crossing alignment, and transition concentration; they do not uniquely identify a discontinuous spiking mechanism",
        "graded_behavior_competence_reach_n_conditions": int(len(behavior_cr)),
        "graded_margin_competence_reach_n_conditions": int(len(margin_cr)),
        "graded_competence_reach_associations": associations.to_dict(orient="records") if not associations.empty else [],
        "graded_competence_reach_interpretation": "direction-matched causal reach uses U_J_i2c for 0to1 and U_J_c2i for 1to0; fixed low/mid/high strata use thirds of the [0,1] scale; results are descriptive cross-setting stratification rather than proof of temporal emergence",
        "inference_unit": "run/baseline/direction condition; agonists and examples are summarized within condition",
    }
    (out / "graded_agonist_report_status.json").write_text(json.dumps(status, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(status, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
