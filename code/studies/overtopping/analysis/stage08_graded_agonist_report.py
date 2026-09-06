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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--out", default=str(PROJECT_ROOT / "results/analysis/rq3_threshold_event/spiking_diagnostics/graded_agonist"))
    p.add_argument("--paper-figures-dir", default=None)
    p.add_argument("--primary-table", required=True)
    p.add_argument("--population-scope", choices=["primary", "primary+supplementary"], default="primary+supplementary")
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
        "natural_state_reproduction_rate",
        "full_dose_support_reproduction_rate",
        "full_dose_flip_rate",
    ]
    for col in cols:
        d[col] = pd.to_numeric(d.get(col), errors="coerce")
    keys = ["run_id", "source_scope", "task", "model", "phase", "direction", "support_kind"]
    return d.groupby(keys, dropna=False).agg(
        n_agonists=("unit_key", "nunique"),
        median_single_crossing_rate=("single_crossing_rate", "median"),
        median_first_flip_dose=("median_first_flip_dose", "median"),
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

    _plot_dose(cond_dose, out / "graded_agonist_dose_response.pdf")
    _plot_crossing(condition, out / "graded_agonist_support_consistency.pdf")
    # Retain the established analysis filename as a compatibility alias.
    _plot_crossing(condition, out / "graded_agonist_single_crossing.pdf")
    if args.paper_figures_dir:
        paper = Path(args.paper_figures_dir).expanduser().resolve()
        paper.mkdir(parents=True, exist_ok=True)
        for stale in [
            "fig4c_preemption.pdf",
            "fig4d_graded_agonist_dose_response.pdf",
            "fig4s6_graded_agonist_single_crossing.pdf",
        ]:
            (paper / stale).unlink(missing_ok=True)
        _plot_dose(cond_dose, paper / "fig4c_graded_agonist_dose_response.pdf")
        _plot_crossing(condition, paper / "fig4s5_graded_agonist_single_crossing.pdf")

    known = condition.loc[condition.get("support_kind", pd.Series(dtype=str)).astype(str) == "known_flip"] if not condition.empty else pd.DataFrame()
    nonflip = condition.loc[condition.get("support_kind", pd.Series(dtype=str)).astype(str) == "same_agonist_nonflip"] if not condition.empty else pd.DataFrame()
    rates = pd.to_numeric(known.get("median_single_crossing_rate"), errors="coerce").dropna().to_numpy(float) if not known.empty else np.asarray([])
    first = pd.to_numeric(known.get("median_first_flip_dose"), errors="coerce").dropna().to_numpy(float) if not known.empty else np.asarray([])
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
        "median_condition_stable_all_doses_rate_same_agonist_nonflip": float(np.median(stable_nonflip)) if len(stable_nonflip) else math.nan,
        "median_condition_transient_flip_rate_same_agonist_nonflip": float(np.median(transient_nonflip)) if len(transient_nonflip) else math.nan,
        "inference_unit": "run/baseline/direction condition; agonists and examples are summarized within condition",
    }
    (out / "graded_agonist_report_status.json").write_text(json.dumps(status, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(status, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
