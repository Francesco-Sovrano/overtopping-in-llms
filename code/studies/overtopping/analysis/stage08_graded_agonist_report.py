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
from studies.overtopping.analysis.stage07_overtopping_spiking_report import expected_rq3_sources


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--out", default=str(PROJECT_ROOT / "results/analysis/rq3_threshold_event/graded_agonist"))
    p.add_argument("--paper-figures-dir", default=None)
    p.add_argument("--primary-table", required=True)
    p.add_argument("--population-scope", choices=["primary", "primary+supplementary"], default="primary+supplementary")
    p.add_argument("--evaluation-split", choices=["test", "train", "all"], default="test")
    p.add_argument("--spiking-max-points", type=int, default=10000)
    p.add_argument("--data-root", default=None)
    return p.parse_args()


def _load(root: Path, args: argparse.Namespace, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    args.data_root = str(Path(args.data_root or root).expanduser().resolve())
    expected = expected_rq3_sources(args)
    units = []
    doses = []
    audit = []
    for spec in expected:
        base = Path(spec["stats_dir"]) / "graded_agonist_intervention"
        unit_path = base / "graded_agonist_unit_summary.csv"
        dose_path = base / "graded_agonist_dose_rows.csv.gz"
        manifest = base / "graded_agonist_intervention.json"
        status = "missing"
        unit_n = 0
        dose_n = 0
        if manifest.is_file():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                status = str(payload.get("status", "unknown"))
            except Exception:
                status = "manifest_read_error"
        if unit_path.is_file():
            try:
                df = pd.read_csv(unit_path)
            except Exception:
                df = pd.DataFrame()
            if not df.empty:
                df["run_id"] = spec["run_id"]
                df["source_scope"] = spec.get("source_scope", "primary")
                df["task"] = spec["task"]
                df["model"] = spec["model"]
                df["phase"] = spec["phase"]
                df["replacement_baseline"] = spec.get("replacement_baseline")
                df["source_path"] = str(unit_path)
                units.append(df)
                unit_n = len(df)
        if dose_path.is_file():
            try:
                df = pd.read_csv(dose_path)
            except Exception:
                df = pd.DataFrame()
            if not df.empty:
                df["run_id"] = spec["run_id"]
                df["source_scope"] = spec.get("source_scope", "primary")
                df["task"] = spec["task"]
                df["model"] = spec["model"]
                df["phase"] = spec["phase"]
                doses.append(df)
                dose_n = len(df)
        audit.append({
            "run_id": spec["run_id"], "source_scope": spec.get("source_scope", "primary"),
            "task": spec["task"], "model": spec["model"], "phase": spec["phase"],
            "graded_dir": str(base), "status": status, "unit_rows": unit_n, "dose_rows": dose_n,
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
    for col in ["single_crossing_rate", "median_first_flip_dose", "natural_state_reproduction_rate", "full_dose_support_reproduction_rate", "full_dose_flip_rate"]:
        d[col] = pd.to_numeric(d.get(col), errors="coerce")
    keys = [c for c in ["run_id", "source_scope", "task", "model", "phase", "direction", "support_kind"] if c in d.columns]
    return d.groupby(keys, dropna=False).agg(
        n_agonists=("unit_key", "nunique"),
        median_single_crossing_rate=("single_crossing_rate", "median"),
        median_first_flip_dose=("median_first_flip_dose", "median"),
        median_natural_state_reproduction_rate=("natural_state_reproduction_rate", "median"),
        median_full_dose_support_reproduction_rate=("full_dose_support_reproduction_rate", "median"),
        median_full_dose_flip_rate=("full_dose_flip_rate", "median"),
    ).reset_index()


def _dose_condition_summary(dose_df: pd.DataFrame) -> pd.DataFrame:
    if dose_df.empty:
        return pd.DataFrame()
    d = dose_df.copy()
    d["dose"] = pd.to_numeric(d["dose"], errors="coerce")
    flip_raw = d["flipped_from_baseline"]
    if flip_raw.dtype == bool:
        d["flipped_from_baseline"] = flip_raw.astype(float)
    else:
        normalized = flip_raw.astype(str).str.strip().str.lower()
        mapping = {"true": 1.0, "1": 1.0, "false": 0.0, "0": 0.0}
        parsed = normalized.map(mapping)
        if parsed.isna().any():
            bad = flip_raw.loc[parsed.isna()].head(5).tolist()
            raise ValueError(f"Could not parse flipped_from_baseline values: {bad}")
        d["flipped_from_baseline"] = parsed.astype(float)
    # First average within agonist, then within run/direction so each agonist and
    # each experimental condition have controlled weight.
    unit = d.groupby(["run_id", "source_scope", "task", "model", "phase", "direction", "support_kind", "unit_key", "dose"], dropna=False)["flipped_from_baseline"].mean().reset_index(name="unit_flip_fraction")
    cond = unit.groupby(["run_id", "source_scope", "task", "model", "phase", "direction", "support_kind", "dose"], dropna=False)["unit_flip_fraction"].median().reset_index(name="condition_median_flip_fraction")
    return cond


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
            ax.plot(arr[:,0], arr[:,1], marker="o", label=str(support_kind))
            ax.fill_between(arr[:,0], arr[:,2], arr[:,3], alpha=.15)
        ax.set_title("Discovery direction 1→0" if direction == "1to0" else "Discovery direction 0→1")
        ax.set_xlabel("Graded agonist intervention dose")
        ax.set_ylim(-.03, 1.03)
        ax.grid(axis="y", alpha=.25, linewidth=.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Behavior-flipped fraction")
    handles, labels = axes[0].get_legend_handles_labels()
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
    d = condition.loc[condition["support_kind"].astype(str) == "known_flip"].copy()
    if d.empty:
        path.unlink(missing_ok=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), sharey=True)
    made = False
    for ax, direction in zip(axes, ["1to0", "0to1"]):
        vals = pd.to_numeric(d.loc[d["direction"].astype(str) == direction, "median_single_crossing_rate"], errors="coerce").dropna().to_numpy(float)
        if len(vals):
            made = True
            ax.scatter(np.zeros(len(vals)), vals, s=25, alpha=.75)
            ax.hlines(np.median(vals), -.18, .18, linewidth=2)
            ax.set_xlim(-.35, .35)
            ax.set_xticks([])
            ax.text(.03,.96,f"conditions={len(vals)}\nmedian={np.median(vals):.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=7.5)
        ax.set_title("Discovery direction 1→0" if direction == "1to0" else "Discovery direction 0→1")
        ax.grid(axis="y", alpha=.25, linewidth=.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    axes[0].set_ylabel("Single persistent crossing rate\non known agonist flip support")
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
    unit_df, dose_df, audit = _load(root, args, out)
    unit_df.to_csv(out / "graded_agonist_all_units.csv", index=False)
    condition = _condition_summary(unit_df)
    condition.to_csv(out / "graded_agonist_by_condition.csv", index=False)
    cond_dose = _dose_condition_summary(dose_df)
    cond_dose.to_csv(out / "graded_agonist_dose_by_condition.csv", index=False)

    _plot_dose(cond_dose, out / "graded_agonist_dose_response.pdf")
    _plot_crossing(condition, out / "graded_agonist_single_crossing.pdf")
    if args.paper_figures_dir:
        paper = Path(args.paper_figures_dir).expanduser().resolve()
        _plot_dose(cond_dose, paper / "fig4d_graded_agonist_dose_response.pdf")
        _plot_crossing(condition, paper / "fig4s6_graded_agonist_single_crossing.pdf")

    known = condition.loc[condition.get("support_kind", pd.Series(dtype=str)).astype(str) == "known_flip"] if not condition.empty else pd.DataFrame()
    rates = pd.to_numeric(known.get("median_single_crossing_rate"), errors="coerce").dropna().to_numpy(float) if not known.empty else np.asarray([])
    first = pd.to_numeric(known.get("median_first_flip_dose"), errors="coerce").dropna().to_numpy(float) if not known.empty else np.asarray([])
    status = {
        "status": "ok" if len(known) else "not_available",
        "scientific_target": "graded causal crossing on the agonist's known held-out flip support",
        "n_conditions_known_flip": int(len(known)),
        "median_condition_single_crossing_rate": float(np.median(rates)) if len(rates) else math.nan,
        "median_condition_first_flip_dose": float(np.median(first)) if len(first) else math.nan,
        "same_agonist_negative_support_conditions": int((condition.get("support_kind", pd.Series(dtype=str)).astype(str) == "same_agonist_nonflip").sum()) if not condition.empty else 0,
        "inference_unit": "run/baseline/direction condition; agonists and examples are summarized within condition",
    }
    (out / "graded_agonist_report_status.json").write_text(json.dumps(status, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(status, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
