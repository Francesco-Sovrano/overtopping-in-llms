#!/usr/bin/env python3
"""Render poisoning-study figures from cross-seed trajectory and detection tables.

The developmental plots summarize behavior/circuit trajectories.  When Stage 07
poisoning-example detection metrics are present in the same aggregate table
directory, this command also renders their across-seed performance.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from studies.poisoning.stage08_aggregate_cross_seed import EXPERIMENT_ID_COLUMNS

TASK_LABELS = {"arithmetic": "Arithmetic", "grammar": "Grammar"}

PRIMARY_METRICS = (
    ("attack_cohort_control_correctness_accuracy", "Attack-cohort control\naccuracy"),
    ("trigger_lift_success_rate", "Backdoor acquisition\ntrigger-lift success"),
    ("lift_U(J)", "Backdoor causal reach\ntrigger-conditioned U(J)"),
    ("attack_cohort_control_correctness_U(J)", "Attack-cohort causal reach\ncontrol-correctness U(J)"),
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


def _read_optional_csv(path: Path) -> pd.DataFrame:
    """Read an optional aggregate CSV, treating empty files as no data.

    Stage 08 intentionally permits detector/behavioral branches to be absent.
    ``DataFrame().to_csv`` produces a file with no parseable columns, which
    pandas reports as ``EmptyDataError``.  Optional reporting inputs should
    therefore map both a missing file and a zero-column/empty file to an empty
    frame rather than aborting the complete final-results build.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()



def _normalized_identity_value(value) -> str:
    if pd.isna(value):
        return "<NA>"
    return str(value)


def _validate_unambiguous_scientific_families(df: pd.DataFrame, *, model_col: str = "model_name") -> None:
    """Reject task/model families containing multiple scientific configurations."""
    if df.empty or "task" not in df.columns or model_col not in df.columns:
        return
    id_cols = [c for c in EXPERIMENT_ID_COLUMNS if c in df.columns and c not in {"task", "model_name"}]
    if not id_cols:
        return
    for (task, model), group in df.groupby(["task", model_col], dropna=False, sort=False):
        identities = {tuple(_normalized_identity_value(row[c]) for c in id_cols) for _, row in group[id_cols].iterrows()}
        if len(identities) > 1:
            raise ValueError(
                f"Ambiguous poisoning trajectory family task={task!r}, model={model!r}: "
                f"{len(identities)} scientific configurations are present. Plot one configuration at a time."
            )


def _available_specs(df: pd.DataFrame, specs):
    out = []
    for metric, title in specs:
        col = f"{metric}__mean"
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if np.isfinite(values.to_numpy(float)).any():
            out.append((metric, title))
    return tuple(out)

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
    _validate_unambiguous_scientific_families(df)
    specs = _available_specs(df, specs)
    families = _family_rows(df)
    if not families or not specs:
        out_path.unlink(missing_ok=True)
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


def _plot_poison_detection(input_dir: Path, output_dir: Path) -> None:
    path = input_dir / "poison_detection_metrics_across_seeds.csv"
    df = _read_optional_csv(path)
    if df.empty:
        return
    _validate_unambiguous_scientific_families(df)
    families = _family_rows(df)
    if not families:
        return
    specs = (
        ("roc_auc", "ROC AUC"),
        ("average_precision", "Average precision"),
        ("precision_at_expected_poison_count", "Precision @ true poison count"),
        ("paired_poison_over_source_rate", "Poison > matched source"),
    )
    fig, axes = plt.subplots(
        len(families), len(specs),
        figsize=(4.0 * len(specs), 3.2 * len(families) + 0.8),
        squeeze=False,
    )
    legend_handles = legend_labels = None
    all_end = pd.to_numeric(df.get("end_fraction"), errors="coerce").dropna().tolist()
    for r, (task, model) in enumerate(families):
        fam = df[(df["task"].astype(str) == task) & (df["model_name"].astype(str) == model)]
        row_label = f"{TASK_LABELS.get(task, task.title())}\n{_model_short(model)}"
        for c, (metric, title) in enumerate(specs):
            ax = axes[r, c]
            mean = f"{metric}__mean"
            plotted = False
            if mean in fam.columns:
                g = fam.copy()
                g["end_fraction"] = pd.to_numeric(g["end_fraction"], errors="coerce")
                g[mean] = pd.to_numeric(g[mean], errors="coerce")
                g = g.dropna(subset=["end_fraction", mean]).sort_values("end_fraction")
                if not g.empty:
                    x = g["end_fraction"].to_numpy(float)
                    y = g[mean].to_numpy(float)
                    ax.plot(x, y, marker="o", linewidth=1.8, label="Disrupted control-correctness channels")
                    random_mean = f"matched_random_{metric}__mean"
                    if random_mean in g.columns:
                        random_y = pd.to_numeric(g[random_mean], errors="coerce").to_numpy(float)
                        random_good = np.isfinite(random_y)
                        if random_good.any():
                            ax.plot(x[random_good], random_y[random_good], marker="x", linestyle=":", linewidth=1.3, label="Matched random rows")
                    lo, hi = f"{metric}__ci_low", f"{metric}__ci_high"
                    if lo in g.columns and hi in g.columns:
                        a = pd.to_numeric(g[lo], errors="coerce").to_numpy(float)
                        b = pd.to_numeric(g[hi], errors="coerce").to_numpy(float)
                        good = np.isfinite(a) & np.isfinite(b)
                        if good.any():
                            ax.fill_between(x[good], a[good], b[good], alpha=0.15)
                    plotted = True
            if metric in {"roc_auc", "paired_poison_over_source_rate"}:
                baseline = np.full_like(x, 0.5, dtype=float) if plotted else np.asarray([], dtype=float)
            else:
                prev_col = "poison_prevalence__mean"
                if plotted and prev_col in g.columns:
                    baseline = pd.to_numeric(g[prev_col], errors="coerce").to_numpy(float)
                else:
                    baseline = np.asarray([], dtype=float)
            if plotted and len(baseline) == len(x):
                good_base = np.isfinite(baseline)
                if good_base.any():
                    ax.plot(x[good_base], baseline[good_base], linestyle="--", linewidth=1.0, alpha=0.7, label="Random baseline")
            ax.set_ylim(-0.03, 1.03)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            _fraction_ticks(ax, all_end)
            if r == 0:
                ax.set_title(title, fontsize=11)
            ax.set_ylabel(row_label if c == 0 else "", fontsize=10)
            ax.grid(True, alpha=0.22)
            if r == 0 and c == 0 and plotted:
                ax.legend(frameon=False, fontsize=8)
            if not plotted:
                ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle(
        "Poisoned-example detection from clean-normalized overtopping-channel disruption",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "poisoning_example_detection.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", required=True, help="Directory containing checkpoint_metrics_by_model_across_seeds.csv")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    input_dir, output_dir = Path(args.input_dir).expanduser(), Path(args.output_dir).expanduser()
    table_path = input_dir / "checkpoint_metrics_by_model_across_seeds.csv"
    df = _read_optional_csv(table_path)
    if not df.empty:
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
    # Detector plotting is independent of the behavioral table and can still be
    # produced when every Stage-05 trajectory is unavailable.
    _plot_poison_detection(input_dir, output_dir)
    print(f"Wrote poisoning figures under {output_dir}")


if __name__ == "__main__":
    main()
