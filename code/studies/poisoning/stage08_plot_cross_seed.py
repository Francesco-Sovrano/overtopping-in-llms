#!/usr/bin/env python3
"""Render complete poisoning-study figures from cross-seed tables and defence outputs.

Behavior/circuit figures are generated from the cross-seed trajectory tables.  When
``--defence_root`` is supplied, the same command also renders manuscript-readable
summaries of both inference-time ablation and training-time protection.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy import stats

TASK_LABELS = {"arithmetic": "Arithmetic", "grammar": "Grammar"}

PRIMARY_METRICS = (
    ("ordinary_correctness_accuracy", "Task competence\nordinary accuracy"),
    ("trigger_lift_success_rate", "Backdoor acquisition\ntrigger-lift success"),
    ("lift_U(J)", "Backdoor causal reach\ntrigger-conditioned U(J)"),
    ("ordinary_correctness_U(J)", "Task causal reach\nordinary-correctness U(J)"),
)
SPECIFICITY_METRICS = (
    ("control_target_positive_rate", "Control behavior\ntarget rate"),
    ("trigger_excess_target_rate", "Trigger-specific shift\ntrigger excess"),
    ("conditional_conversion_rate", "Conditional attack success\nconversion rate"),
    ("primary_minus_sham_conditional_conversion_rate", "Marker specificity\nprimary - sham"),
)


def _model_short(name: str) -> str:
    return str(name or "unknown").rsplit("/", 1)[-1]


def _rate_axis(ax, *, signed: bool = False) -> None:
    if signed:
        ax.axhline(0.0, linewidth=0.8, alpha=0.5)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    else:
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(True, alpha=0.22)


def _fraction_ticks(ax, values: Iterable[float]) -> None:
    vals = sorted({float(x) for x in values if np.isfinite(float(x))})
    if vals:
        ax.set_xticks(vals)
        ax.set_xticklabels([f"{100*x:g}%" for x in vals])
    ax.set_xlabel("Fine-tuning progress")


def _family_rows(df: pd.DataFrame, model_col: str = "model_name") -> list[tuple[str, str]]:
    if df.empty or "task" not in df.columns or model_col not in df.columns:
        return []
    rows = [(str(t), str(m)) for t, m in df[["task", model_col]].drop_duplicates().itertuples(index=False, name=None)]
    order = {"arithmetic": 0, "grammar": 1}
    return sorted(rows, key=lambda x: (order.get(x[0], 99), x[0], x[1]))


def _plot_metric(ax, family: pd.DataFrame, metric: str) -> bool:
    mean_col, lo_col, hi_col = f"{metric}__mean", f"{metric}__ci_low", f"{metric}__ci_high"
    if mean_col not in family.columns:
        return False
    plotted = False
    labels = {"clean": "Clean fine-tune", "poisoned": "Poisoned fine-tune"}
    for condition, group in family.groupby("condition", sort=False):
        group = group.copy()
        group["fraction"] = pd.to_numeric(group["fraction"], errors="coerce")
        group[mean_col] = pd.to_numeric(group[mean_col], errors="coerce")
        group = group.dropna(subset=["fraction", mean_col]).sort_values("fraction")
        if group.empty:
            continue
        x, y = group["fraction"].to_numpy(float), group[mean_col].to_numpy(float)
        ax.plot(x, y, marker="o", linewidth=1.8, label=labels.get(str(condition), str(condition)))
        if lo_col in group.columns and hi_col in group.columns:
            lo = pd.to_numeric(group[lo_col], errors="coerce").to_numpy(float)
            hi = pd.to_numeric(group[hi_col], errors="coerce").to_numpy(float)
            good = np.isfinite(lo) & np.isfinite(hi)
            if good.any():
                ax.fill_between(x[good], lo[good], hi[good], alpha=0.15)
        plotted = True
    return plotted


def _plot_grid(df: pd.DataFrame, specs, out_path: Path, title: str, note: str | None = None) -> None:
    families = _family_rows(df)
    if not families:
        return
    fig, axes = plt.subplots(len(families), len(specs), figsize=(4.1 * len(specs), 3.25 * len(families) + 0.8), squeeze=False)
    all_fractions = pd.to_numeric(df.get("fraction"), errors="coerce").dropna().tolist()
    legend_handles = legend_labels = None
    for r, (task, model) in enumerate(families):
        family = df[(df["task"].astype(str) == task) & (df["model_name"].astype(str) == model)]
        row_label = f"{TASK_LABELS.get(task, task.title())}\n{_model_short(model)}"
        for c, (metric, metric_title) in enumerate(specs):
            ax = axes[r, c]
            plotted = _plot_metric(ax, family, metric)
            _rate_axis(ax, signed=(metric == "primary_minus_sham_conditional_conversion_rate"))
            _fraction_ticks(ax, all_fractions)
            if r == 0:
                ax.set_title(metric_title, fontsize=11)
            ax.set_ylabel(row_label if c == 0 else "", fontsize=10)
            if not plotted:
                ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
            if legend_handles is None and ax.lines:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
    fig.suptitle(title, fontsize=14, y=0.995)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=max(1, len(legend_labels)), frameon=False, bbox_to_anchor=(0.5, 0.965))
    if note:
        fig.text(0.5, 0.006, note, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.035 if note else 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _parse_defence_identity(path: Path, root: Path) -> dict[str, object] | None:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    parts = rel.parts
    # <task>/<model>/seed_<seed>/<phase>/<mechanism>/<file>
    if len(parts) < 6:
        return None
    task, model_slug, seed_part, phase = parts[:4]
    if not seed_part.startswith("seed_"):
        return None
    try:
        seed = int(seed_part[5:])
    except ValueError:
        return None
    return {"task": task, "model_slug": model_slug, "training_seed": seed, "phase": phase}


def _t_summary(frame: pd.DataFrame, metrics: Iterable[str], group_cols: list[str]) -> pd.DataFrame:
    records = []
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        for metric in metrics:
            vals = pd.to_numeric(group.get(metric), errors="coerce").dropna().to_numpy(float)
            rec[f"{metric}__mean"] = float(vals.mean()) if len(vals) else math.nan
            rec[f"{metric}__n"] = int(len(vals))
            if len(vals) >= 2:
                sem = float(vals.std(ddof=1) / math.sqrt(len(vals)))
                crit = float(stats.t.ppf(0.975, len(vals) - 1))
                rec[f"{metric}__ci_low"] = rec[f"{metric}__mean"] - crit * sem
                rec[f"{metric}__ci_high"] = rec[f"{metric}__mean"] + crit * sem
            else:
                rec[f"{metric}__ci_low"] = math.nan
                rec[f"{metric}__ci_high"] = math.nan
        records.append(rec)
    return pd.DataFrame(records)


def _load_inference_defence(roots: list[Path]) -> pd.DataFrame:
    frames = []
    for root in roots:
        for path in root.rglob("inference_time/backdoor_lift_cumulative_topk_ablation.csv"):
            ident = _parse_defence_identity(path, root)
            if ident is None:
                continue
            df = pd.read_csv(path)
            if df.empty or "fraction" not in df.columns:
                continue
            df = df.copy()
            for k, v in ident.items():
                df[k] = v
            df["trigger_lift_destroy_rate"] = pd.to_numeric(df.get("trigger_lift_destroy_rate"), errors="coerce")
            if df["trigger_lift_destroy_rate"].notna().any():
                frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _load_training_defence(roots: list[Path]) -> pd.DataFrame:
    frames = []
    for root in roots:
        for path in root.rglob("training_time/training_protection_key_metrics.csv"):
            ident = _parse_defence_identity(path, root)
            if ident is None:
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            for k, v in ident.items():
                df[k] = v
            frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _plot_defence_summary(agg: pd.DataFrame, specs, out_path: Path, title: str, series_col: str | None = None) -> None:
    families = _family_rows(agg, "model_slug")
    if not families:
        return
    fig, axes = plt.subplots(len(families), len(specs), figsize=(4.3 * len(specs), 3.3 * len(families) + 0.8), squeeze=False)
    legend_handles = legend_labels = None
    all_fractions = pd.to_numeric(agg.get("fraction"), errors="coerce").dropna().tolist()
    for r, (task, model) in enumerate(families):
        fam = agg[(agg["task"].astype(str) == task) & (agg["model_slug"].astype(str) == model)]
        row_label = f"{TASK_LABELS.get(task, task.title())}\n{_model_short(model)}"
        for c, (metric, metric_title, signed) in enumerate(specs):
            ax = axes[r, c]
            groups = fam.groupby(series_col, sort=False) if series_col and series_col in fam.columns else [(None, fam)]
            plotted = False
            for label, g in groups:
                mean = f"{metric}__mean"
                if mean not in g.columns:
                    continue
                g = g.copy()
                g["fraction"] = pd.to_numeric(g["fraction"], errors="coerce")
                g[mean] = pd.to_numeric(g[mean], errors="coerce")
                g = g.dropna(subset=["fraction", mean]).sort_values("fraction")
                if g.empty:
                    continue
                x, y = g["fraction"].to_numpy(float), g[mean].to_numpy(float)
                ax.plot(x, y, marker="o", linewidth=1.8, label=str(label).replace("_", " ") if label is not None else None)
                lo, hi = f"{metric}__ci_low", f"{metric}__ci_high"
                if lo in g.columns and hi in g.columns:
                    a, b = pd.to_numeric(g[lo], errors="coerce").to_numpy(float), pd.to_numeric(g[hi], errors="coerce").to_numpy(float)
                    good = np.isfinite(a) & np.isfinite(b)
                    if good.any():
                        ax.fill_between(x[good], a[good], b[good], alpha=0.15)
                plotted = True
            _rate_axis(ax, signed=signed)
            _fraction_ticks(ax, all_fractions)
            if r == 0:
                ax.set_title(metric_title, fontsize=11)
            ax.set_ylabel(row_label if c == 0 else "", fontsize=10)
            if not plotted:
                ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
            if legend_handles is None and ax.lines and series_col:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
    fig.suptitle(title, fontsize=14, y=0.995)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=max(1, len(legend_labels)), frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _plot_inference_dose_response(agg: pd.DataFrame, out_path: Path) -> None:
    families = _family_rows(agg, "model_slug")
    if not families:
        return
    specs = (
        ("trigger_lift_destroy_rate", "Trigger-lift removed", False),
        ("trigger_specificity_gap", "Trigger - ordinary removal", True),
        ("paired_clean_accuracy_drop", "Clean accuracy drop", True),
    )
    fig, axes = plt.subplots(len(families), len(specs), figsize=(4.4 * len(specs), 3.4 * len(families) + 0.9), squeeze=False)
    legend_handles = legend_labels = None
    for r, (task, model) in enumerate(families):
        fam = agg[(agg["task"].astype(str) == task) & (agg["model_slug"].astype(str) == model)]
        row_label = f"{TASK_LABELS.get(task, task.title())}\n{_model_short(model)}"
        for c, (metric, title, signed) in enumerate(specs):
            ax = axes[r, c]
            mean = f"{metric}__mean"
            plotted = False
            if mean in fam.columns:
                for fraction, g in fam.groupby("fraction", sort=True):
                    g = g.copy()
                    g["k"] = pd.to_numeric(g["k"], errors="coerce")
                    g[mean] = pd.to_numeric(g[mean], errors="coerce")
                    g = g.dropna(subset=["k", mean]).sort_values("k")
                    if g.empty:
                        continue
                    x, y = g["k"].to_numpy(float), g[mean].to_numpy(float)
                    ax.plot(x, y, marker="o", linewidth=1.6, label=f"{100*float(fraction):g}% checkpoint")
                    lo, hi = f"{metric}__ci_low", f"{metric}__ci_high"
                    if lo in g.columns and hi in g.columns:
                        a = pd.to_numeric(g[lo], errors="coerce").to_numpy(float)
                        b = pd.to_numeric(g[hi], errors="coerce").to_numpy(float)
                        good = np.isfinite(a) & np.isfinite(b)
                        if good.any():
                            ax.fill_between(x[good], a[good], b[good], alpha=0.12)
                    plotted = True
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Cumulative coalition size k")
            _rate_axis(ax, signed=signed)
            if r == 0:
                ax.set_title(title, fontsize=11)
            ax.set_ylabel(row_label if c == 0 else "", fontsize=10)
            if not plotted:
                ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
            if legend_handles is None and ax.lines:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
    fig.suptitle("Prospective inference-time defence: cumulative coalition dose-response", fontsize=14, y=0.995)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=min(5, max(1, len(legend_labels))), frameon=False, bbox_to_anchor=(0.5, 0.96))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def render_defence_figures(defence_roots: list[Path], output_dir: Path) -> None:
    inference = _load_inference_defence(defence_roots)
    if not inference.empty:
        metrics = [m for m in ("trigger_lift_destroy_rate", "trigger_specificity_gap", "paired_clean_accuracy_drop") if m in inference.columns]
        agg = _t_summary(inference, metrics, ["task", "model_slug", "phase", "fraction", "k"])
        _plot_inference_dose_response(agg, output_dir / "poisoning_inference_time_defence.pdf")

    training = _load_training_defence(defence_roots)
    if not training.empty and "training_intervention" in training.columns:
        metrics = [m for m in ("conditional_conversion_rate", "trigger_lift_success_rate", "ordinary_correctness_accuracy") if m in training.columns]
        agg = _t_summary(training, metrics, ["task", "model_slug", "phase", "training_intervention", "fraction"])
        attack_metric = "conditional_conversion_rate" if "conditional_conversion_rate" in training.columns else "trigger_lift_success_rate"
        specs = [(attack_metric, "Backdoor behavior\nconditional attack success" if attack_metric == "conditional_conversion_rate" else "Backdoor behavior\ntrigger-lift success", False)]
        if "ordinary_correctness_accuracy" in training.columns:
            specs.append(("ordinary_correctness_accuracy", "Task competence\nordinary accuracy", False))
        _plot_defence_summary(
            agg,
            tuple(specs),
            output_dir / "poisoning_training_time_defence.pdf",
            "Training-time poisoning protection",
            series_col="training_intervention",
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", required=True, help="Directory containing checkpoint_metrics_by_model_across_seeds.csv")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--defence_root", default=None, help="Optional comma-separated 07_defence_evaluation directories for both defence figures.")
    args = p.parse_args()

    input_dir, output_dir = Path(args.input_dir).expanduser(), Path(args.output_dir).expanduser()
    table_path = input_dir / "checkpoint_metrics_by_model_across_seeds.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing cross-seed poisoning table: {table_path}")
    df = pd.read_csv(table_path)
    if df.empty:
        raise ValueError(f"Cross-seed poisoning table is empty: {table_path}")

    _plot_grid(
        df, PRIMARY_METRICS, output_dir / "poisoning_primary_trajectories.pdf",
        "Poisoning trajectories: behavior and causal organization",
        note="Ordinary correctness and trigger lift are distinct endpoints. Gaps mean CHA was undefined/skipped. Shading is the across-seed Student-t interval when available.",
    )
    _plot_grid(
        df, SPECIFICITY_METRICS, output_dir / "poisoning_specificity_checks.pdf",
        "Poisoning specificity and control diagnostics",
        note="Control target rate diagnoses clean-path drift; trigger excess isolates marker-specific shift; conditional conversion conditions on control non-target rows; primary-sham contrast checks marker specificity.",
    )
    if args.defence_root:
        defence_roots = [Path(x.strip()).expanduser() for x in args.defence_root.split(",") if x.strip()]
        render_defence_figures(defence_roots, output_dir)
    print(f"Wrote poisoning figures under {output_dir}")


if __name__ == "__main__":
    main()
