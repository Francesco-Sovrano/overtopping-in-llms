#!/usr/bin/env python3
"""Compare normal poisoning with virgin-agonist and matched-random protection."""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from studies.poisoning.lib.run_paths import phase_dirname, trajectories_dir
import pandas as pd


def load(run_dir: Path, phase: str, label: str, expected_condition: str) -> pd.DataFrame:
    p = trajectories_dir(run_dir) / phase_dirname(phase) / "backdoor_lift_overtopping_trajectory.csv"
    if not p.is_file():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "condition" not in df.columns:
        raise ValueError(f"{p} has no condition column")
    df = df.loc[df["condition"].astype(str) == expected_condition].copy()
    if df.empty:
        raise ValueError(f"{p} contains no rows for condition={expected_condition!r}")
    df["training_intervention"] = label
    return df


ANALYSIS_CONFIG_COLUMNS = (
    "reference_cha_points_per_side",
    "cha_base_tau_at_reference_n",
    "low_data_policy",
    "causal_scan_max_rows",
    "causal_scan_sampling",
    "causal_scan_early_stop",
    "stage7_sampling_max_points",
)

def assert_matched_analysis_config(frames: list[pd.DataFrame]) -> None:
    """Require protection comparisons to use one poisoning-analysis policy."""
    labels = [str(df["training_intervention"].iloc[0]) if len(df) else f"run_{i}" for i, df in enumerate(frames)]
    problems: list[str] = []
    for col in ANALYSIS_CONFIG_COLUMNS:
        present = [col in df.columns for df in frames]
        if any(present) and not all(present):
            missing = [labels[i] for i, ok in enumerate(present) if not ok]
            problems.append(f"{col}: missing from {missing}")
            continue
        if not any(present):
            continue
        values = []
        for label, df in zip(labels, frames):
            series = df[col].dropna()
            uniq = sorted({str(v) for v in series.tolist()})
            values.append((label, uniq))
        nonempty = {tuple(v) for _, v in values if v}
        if len(nonempty) > 1:
            problems.append(f"{col}: " + "; ".join(f"{label}={vals}" for label, vals in values))
    if problems:
        raise ValueError(
            "Training-protection comparison requires matched post-training causal-analysis settings. "
            "Rerun discovery for the baseline/protected/random-protected trajectories with the same "
            "CHA_REFERENCE_N_PER_SIDE, CHA_TAU, CHA_LOW_DATA_POLICY, "
            "TRIGGER_LIFT_SCAN_MAX_ROWS, and REFINE_SAMPLING_MAX_POINTS. Mismatches: "
            + " | ".join(problems)
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_run", required=True)
    ap.add_argument("--protected_run", required=True)
    ap.add_argument("--random_protected_run", required=True)
    ap.add_argument("--phase", choices=["input_output", "output_only"], required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    frames = [
        load(Path(args.baseline_run), args.phase, "ordinary_poisoning", "poisoned"),
        load(Path(args.protected_run), args.phase, "virgin_agonist_direct_write_protection", "protected_poisoned"),
        load(Path(args.random_protected_run), args.phase, "matched_random_direct_write_protection", "random_protected_poisoned"),
    ]
    assert_matched_analysis_config(frames)
    df = pd.concat(frames, ignore_index=True, sort=False)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "training_protection_trajectory.csv", index=False)
    cols = [c for c in [
        "training_intervention", "condition", "fraction", "global_step",
        "reference_cha_points_per_side", "cha_base_tau_at_reference_n", "low_data_policy",
        "causal_scan_max_rows", "stage7_sampling_max_points",
        "ordinary_correctness_accuracy", "trigger_lift_success_rate", "n_trigger_lift_success",
        "conditional_conversion_rate", "conditional_conversion_n", "lift_U(J)", "lift_Top",
        "ordinary_correctness_U(J)", "ordinary_correctness_Top",
        "lift_n_overtopping_neurons", "lift_N.05", "lift_N.10", "lift_Neff", "lift_top1_mass",
    ] if c in df.columns]
    key_df = df[cols].copy()
    key_df.to_csv(out / "training_protection_key_metrics.csv", index=False)

    metric = next((c for c in ("conditional_conversion_rate", "trigger_lift_success_rate") if c in df.columns), None)
    if metric and "fraction" in df.columns:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for label, group in df.groupby("training_intervention", sort=False):
            plot_df = group[["fraction", metric]].copy()
            plot_df["fraction"] = pd.to_numeric(plot_df["fraction"], errors="coerce")
            plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
            plot_df = plot_df.dropna().sort_values("fraction")
            if not plot_df.empty:
                ax.plot(100.0 * plot_df["fraction"], 100.0 * plot_df[metric], marker="o", label=str(label))
        ax.set_xlabel("Training progress (%)")
        ax.set_ylabel(metric.replace("_", " ").title() + " (%)")
        ax.set_ylim(-2, 102)
        ax.grid(True, alpha=0.25)
        if ax.lines:
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "training_protection.pdf")
        plt.close(fig)

    summary = [
        "# Training-time protection summary",
        "",
        "The protected arm freezes direct writes to overtopping channels selected from the pre-poisoning virgin-model analysis. The matched-random arm protects the same number/type of channels but samples them randomly. Only the poisoned condition from each run is compared.",
        "",
    ]
    if metric:
        summary += [f"| Training arm | Final checkpoint {metric} |", "|---|---:|"]
        for label, group in df.groupby("training_intervention", sort=False):
            if "fraction" in group.columns:
                group = group.sort_values("fraction")
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            value = "n/a" if vals.empty else f"{100.0 * float(vals.iloc[-1]):.1f}%"
            summary.append(f"| {label} | {value} |")
    (out / "training_protection_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Wrote training-protection comparison under {out}")

if __name__ == "__main__":
    main()
