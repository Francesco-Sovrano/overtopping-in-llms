#!/usr/bin/env python3
"""Create centralized, paper-oriented poisoning figures from the seed-level matrix.

Only PDF figures are emitted. Per-run poisoning diagnostics live inside data/poisoning/<task>/<run>/;
only manuscript-facing visual summaries live under results/poisoning/figures/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


TASK_LABELS = {
    "arithmetic": "Arithmetic",
    "grammar": "Grammar",
}

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
    text = str(name or "unknown")
    return text.rsplit("/", 1)[-1]


def _rate_axis(ax) -> None:
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


def _family_rows(df: pd.DataFrame) -> list[tuple[str, str]]:
    rows = []
    for task, model in df[["task", "model_name"]].drop_duplicates().itertuples(index=False, name=None):
        rows.append((str(task), str(model)))
    order = {"arithmetic": 0, "grammar": 1}
    return sorted(rows, key=lambda x: (order.get(x[0], 99), x[0], x[1]))


def _plot_metric(ax, family: pd.DataFrame, metric: str) -> bool:
    mean_col = f"{metric}__mean"
    lo_col = f"{metric}__ci_low"
    hi_col = f"{metric}__ci_high"
    if mean_col not in family.columns:
        return False

    plotted = False
    condition_labels = {"clean": "Clean fine-tune", "poisoned": "Poisoned fine-tune"}
    for condition, group in family.groupby("condition", sort=False):
        group = group.copy()
        group["fraction"] = pd.to_numeric(group["fraction"], errors="coerce")
        group[mean_col] = pd.to_numeric(group[mean_col], errors="coerce")
        group = group.dropna(subset=["fraction", mean_col]).sort_values("fraction")
        if group.empty:
            continue
        x = group["fraction"].to_numpy(dtype=float)
        y = group[mean_col].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.8, label=condition_labels.get(str(condition), str(condition)))
        if lo_col in group.columns and hi_col in group.columns:
            lo = pd.to_numeric(group[lo_col], errors="coerce").to_numpy(dtype=float)
            hi = pd.to_numeric(group[hi_col], errors="coerce").to_numpy(dtype=float)
            good = np.isfinite(lo) & np.isfinite(hi)
            if good.any():
                ax.fill_between(x[good], lo[good], hi[good], alpha=0.15)
        plotted = True
    return plotted


def _plot_grid(df: pd.DataFrame, specs, out_path: Path, title: str, note: str | None = None) -> None:
    families = _family_rows(df)
    if not families:
        return
    nrows = len(families)
    ncols = len(specs)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.1 * ncols, 3.25 * nrows + 0.8),
        squeeze=False,
        sharey=False,
    )

    all_fractions = pd.to_numeric(df.get("fraction"), errors="coerce").dropna().tolist()
    legend_handles = legend_labels = None
    for r, (task, model) in enumerate(families):
        family = df[(df["task"].astype(str) == task) & (df["model_name"].astype(str) == model)]
        row_label = f"{TASK_LABELS.get(task, task.title())}\n{_model_short(model)}"
        for c, (metric, metric_title) in enumerate(specs):
            ax = axes[r, c]
            plotted = _plot_metric(ax, family, metric)
            _rate_axis(ax)
            _fraction_ticks(ax, all_fractions)
            if r == 0:
                ax.set_title(metric_title, fontsize=11)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=10)
            else:
                ax.set_ylabel("")
            if not plotted:
                ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
            if legend_handles is None and ax.lines:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.suptitle(title, fontsize=14, y=0.995)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=max(1, len(legend_labels)), frameon=False,
                   bbox_to_anchor=(0.5, 0.965))
    if note:
        fig.text(0.5, 0.006, note, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.035 if note else 0.0, 1.0, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    matrix_path = input_dir / "checkpoint_metrics_by_model_across_seeds.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing aggregated poisoning matrix: {matrix_path}")
    df = pd.read_csv(matrix_path)
    if df.empty:
        raise ValueError(f"Aggregated poisoning matrix is empty: {matrix_path}")

    _plot_grid(
        df,
        PRIMARY_METRICS,
        output_dir / "poisoning_primary_trajectories.pdf",
        "Poisoning trajectories: behavior and causal organization",
        note=(
            "Ordinary correctness and trigger lift are distinct endpoints. U(J) is held-out singleton-union reach for "
            "the endpoint named above each column. Gaps mean CHA was undefined/skipped, not U(J)=0. "
            "Shading is the across-seed Student-t interval when available."
        ),
    )
    _plot_grid(
        df,
        SPECIFICITY_METRICS,
        output_dir / "poisoning_specificity_checks.pdf",
        "Poisoning specificity and control diagnostics",
        note=(
            "Control target rate diagnoses clean-path drift; trigger excess isolates the marker-specific target shift; "
            "conditional conversion conditions on control non-target rows; primary-sham contrast checks marker specificity."
        ),
    )
    print(f"Wrote PDF poisoning figures under {output_dir}")


if __name__ == "__main__":
    main()
