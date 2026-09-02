#!/usr/bin/env python3
"""Post-hoc overtopping/poisoning geometry analyses from existing results.

This script is designed to answer mechanistic questions about overtopping using
already-computed poisoning outputs and saved LoRA checkpoints. It does not rerun
training, CHA, singleton interventions, or model inference.

It produces a question-oriented interpretation report with mechanism, detection, update-geometry controls, and attack-behavior linkage:

1. Causal-disruption concentration with an integrated clean control:
   keep the original top-k |D_j| concentration view and overlay clean |delta U_j| drift on the same support.

2. Local geometry vs causal disruption:
   within-interval associations between |D_j| and clean/poison update geometry,
   plus candidate-vs-matched-control directional comparisons.

3. Geometry vs poisoning acquisition/detectability:
   interval-level associations between causal concentration/update geometry and
   attack acquisition / WANDA detector metrics.

4. Overtopping-support excess-update enrichment:
   exact poison-specific LoRA update energy for every output row, then the
   percentile/enrichment of the rows mapped from overtopping channels.

It also compares the causal-weighted Stage-07 WANDA score against the unweighted
activation×update score using the already-saved training-example score table.

Expected prerequisite:
    studies.poisoning.stage07_visualize_update_geometry

If its base geometry CSVs are absent, this script can generate them automatically
from the saved LoRA checkpoints and Stage-07 outputs.
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from studies.poisoning.lib.run_paths import detection_dir, phase_dirname
from studies.poisoning.lib.specificity import truthy
from studies.poisoning.stage07_detect_poisoning_examples import (
    _interval_label,
    _load_manifest_by_condition,
    detection_metrics,
)
from studies.poisoning.stage07_plot_detection_implications import (
    build_detection_vs_attack_table,
    generate_implication_outputs,
)
from studies.poisoning.stage07_visualize_update_geometry import (
    LowRankProduct,
    State,
    _load_all_states,
)
from studies.poisoning.stage04_compare_condition_behavior import (
    _score_path as _behavior_score_path,
    summarize_scores as _summarize_behavior_scores,
)
from studies.poisoning.tasks.registry import infer_task_from_run

EPS = 1e-30



def _figure_note(fig: plt.Figure, text: str) -> None:
    """Add an interpretation cue that stays with the exported figure."""
    fig.text(
        0.5,
        0.012,
        text,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="dimgray",
        wrap=True,
    )


def _finish_figure(fig: plt.Figure, output: Path, note: str | None = None) -> None:
    if note:
        _figure_note(fig, note)
        fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.965))
    else:
        fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _ensure_output_tree(root: Path) -> dict[str, Path]:
    """Create a small, question-oriented user-facing result tree."""
    dirs = {
        "root": root,
        "mechanism": root / "01_overtopping_mechanism",
        "detection": root / "02_poisoning_detection",
        "update_checks": root / "03_update_geometry_checks",
        "behavior": root / "04_link_to_attack_behavior",
        "tables": root / "supporting_tables",
        "intermediate": root / "intermediate",
    }
    for value in dirs.values():
        value.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_result_guide(root: Path) -> None:
    guide = """# Overtopping interpretation outputs

This directory contains analyses that add information beyond the compact checkpoint poisoning story. The compact aggregate trajectory, descriptive channel-role view, prospective defense-leverage analysis, and complete fixed-union checkpoint figure are generated separately by `stage07_plot_overtopping_poisoning_story.py`.

## 01_overtopping_mechanism

- `01_where_poisoning_specific_causal_control_moves.pdf`
  - Shows which channels and layers carry poisoning-specific causal change across training intervals.
- `02_does_the_same_causal_support_persist.pdf`
  - Quantifies support overlap with the preceding interval for all causal channels and the three strongest channels.
- `03_poisoning_specific_causal_concentration_vs_clean_drift.pdf`
  - Compares the rank concentration of poisoning-specific causal change with clean-training drift on the same support.

These three figures are complementary: location, persistence, and concentration.

## 02_poisoning_detection

- `01_does_overtopping_add_detection_power.pdf`
  - Detection gain from causal weighting relative to matched-random and unweighted activation-update baselines.
- `02_detector_components_vs_baselines.pdf`
  - Absolute detector performance for the causal-weighted score and its baselines.

## 03_update_geometry_checks

- `01_are_overtopping_rows_preferentially_updated.pdf`
  - Tests whether overtopping rows receive disproportionate poison-specific update energy.
- `02_how_extreme_are_overtopping_row_updates.pdf`
  - Places overtopping-row update magnitudes within their projection-specific update distributions.

## 04_link_to_attack_behavior

- `01_do_overtopping_signals_track_attack_growth.pdf`
  - Descriptive within-run association between mechanistic quantities and backdoor acquisition.

## supporting_tables

Machine-readable tables underlying the figures.

## intermediate

Derived update-geometry tables used by the interpretation analyses.
"""
    (root / "README.md").write_text(guide, encoding="utf-8")

def _finite(value: object) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if np.isfinite(x) else math.nan


def _spearman(x: Iterable[float], y: Iterable[float]) -> tuple[int, float]:
    a = pd.to_numeric(pd.Series(list(x)), errors="coerce")
    b = pd.to_numeric(pd.Series(list(y)), errors="coerce")
    ok = a.notna() & b.notna()
    if int(ok.sum()) < 3:
        return int(ok.sum()), math.nan
    aa = a[ok].rank(method="average").to_numpy(float)
    bb = b[ok].rank(method="average").to_numpy(float)
    if np.std(aa) <= 0 or np.std(bb) <= 0:
        return int(ok.sum()), math.nan
    return int(ok.sum()), float(np.corrcoef(aa, bb)[0, 1])


def _pearson(x: Iterable[float], y: Iterable[float]) -> tuple[int, float]:
    a = pd.to_numeric(pd.Series(list(x)), errors="coerce")
    b = pd.to_numeric(pd.Series(list(y)), errors="coerce")
    ok = a.notna() & b.notna()
    if int(ok.sum()) < 3:
        return int(ok.sum()), math.nan
    aa = a[ok].to_numpy(float)
    bb = b[ok].to_numpy(float)
    if np.std(aa) <= 0 or np.std(bb) <= 0:
        return int(ok.sum()), math.nan
    return int(ok.sum()), float(np.corrcoef(aa, bb)[0, 1])



def _ensure_base_geometry(
    *, run_dir: Path, phase: str, geometry_dir: Path, max_control_draws: int
) -> None:
    needed = [
        "update_geometry_by_interval.csv",
        "channel_update_geometry.csv",
        "matched_control_update_geometry.csv",
        "disruption_information_by_interval.csv",
    ]
    if all((geometry_dir / name).is_file() for name in needed):
        return
    cmd = [
        sys.executable,
        "-m",
        "studies.poisoning.stage07_visualize_update_geometry",
        "--run_dir",
        str(run_dir),
        "--phase",
        phase,
        "--output_dir",
        str(geometry_dir),
        "--max_control_draws",
        str(max_control_draws),
        "--tables_only",
    ]
    print("[overtopping-interpretation] deriving base geometry tables:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def _load_disruption_channels(stage07_dir: Path) -> pd.DataFrame:
    combined = stage07_dir / "control_correctness_channel_disruption_by_interval.csv"
    if combined.is_file():
        frame = pd.read_csv(combined, low_memory=False)
        if "interval" in frame.columns:
            return frame

    parts: list[pd.DataFrame] = []
    for interval_dir in sorted(stage07_dir.glob("interval_*")):
        path = interval_dir / "control_correctness_channel_disruption.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            continue
        if "interval" not in df.columns:
            df["interval"] = interval_dir.name
        parts.append(df)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _effective_support(values: np.ndarray) -> float:
    """Rényi-2 effective support for non-negative causal-change magnitudes."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    total = float(values.sum())
    if total <= 0:
        return math.nan
    p = values / total
    denom = float(np.sum(p**2))
    return float(1.0 / denom) if denom > 0 else math.nan


def causal_disruption_concentration(stage07_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare concentration of poisoning-specific causal change to clean-training drift.

    The control is |ΔU_clean(j)| evaluated on the exact same fixed channel support as
    the poisoning-specific excess |D_j| = |ΔU_poison(j) - ΔU_clean(j)|.  This is the
    appropriate cached control for a causal-concentration claim; matched LoRA rows
    control parameter geometry, not the concentration of causal effects.
    """
    frame = _load_disruption_channels(stage07_dir)
    required = {"clean_delta_u_j"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(), pd.DataFrame()

    records: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for interval, group in frame.groupby("interval", sort=False):
        if "complete_u_j_comparison" in group.columns:
            group = group[group["complete_u_j_comparison"].map(truthy)].copy()
        if group.empty:
            continue

        if "poisoning_excess_delta_u_j" in group.columns:
            excess = pd.to_numeric(group["poisoning_excess_delta_u_j"], errors="coerce").abs()
        elif "disruption_score" in group.columns:
            excess = pd.to_numeric(group["disruption_score"], errors="coerce").abs()
        else:
            continue
        clean = pd.to_numeric(group["clean_delta_u_j"], errors="coerce").abs()
        valid = excess.notna() & clean.notna()
        excess_values = excess[valid].to_numpy(dtype=float)
        clean_values = clean[valid].to_numpy(dtype=float)
        if len(excess_values) == 0:
            continue

        # Keep the same evaluated support in both distributions, including zeros.
        excess_sorted = np.sort(excess_values)[::-1]
        clean_sorted = np.sort(clean_values)[::-1]
        excess_total = float(excess_sorted.sum())
        clean_total = float(clean_sorted.sum())
        excess_cum = np.cumsum(excess_sorted) / excess_total if excess_total > 0 else np.full(len(excess_sorted), np.nan)
        clean_cum = np.cumsum(clean_sorted) / clean_total if clean_total > 0 else np.full(len(clean_sorted), np.nan)

        start = _finite(group.get("start_fraction", pd.Series([math.nan])).iloc[0])
        end = _finite(group.get("end_fraction", pd.Series([math.nan])).iloc[0])
        n = len(excess_sorted)
        for rank in range(1, n + 1):
            records.append({
                "interval": interval,
                "start_fraction": start,
                "end_fraction": end,
                "rank": rank,
                "poisoning_excess_abs_causal_change": float(excess_sorted[rank - 1]),
                "clean_control_abs_causal_change": float(clean_sorted[rank - 1]),
                "poisoning_excess_cumulative_share": float(excess_cum[rank - 1]) if np.isfinite(excess_cum[rank - 1]) else math.nan,
                "clean_control_cumulative_share": float(clean_cum[rank - 1]) if np.isfinite(clean_cum[rank - 1]) else math.nan,
            })

        row: dict[str, Any] = {
            "interval": interval,
            "start_fraction": start,
            "end_fraction": end,
            "n_compared_channels": int(n),
            "poisoning_excess_total_abs_causal_change": excess_total,
            "clean_control_total_abs_causal_change": clean_total,
            "poisoning_excess_effective_support": _effective_support(excess_values),
            "clean_control_effective_support": _effective_support(clean_values),
        }
        for k in (1, 2, 3, 5, 10):
            kk = min(k, n)
            pe = float(excess_cum[kk - 1]) if excess_total > 0 else math.nan
            cc = float(clean_cum[kk - 1]) if clean_total > 0 else math.nan
            row[f"poisoning_excess_top{k}_share"] = pe
            row[f"clean_control_top{k}_share"] = cc
            row[f"top{k}_share_excess_over_clean"] = pe - cc if np.isfinite(pe) and np.isfinite(cc) else math.nan
        ne_p = row["poisoning_excess_effective_support"]
        ne_c = row["clean_control_effective_support"]
        row["effective_support_ratio_poisoning_over_clean"] = (
            float(ne_p) / float(ne_c) if np.isfinite(ne_p) and np.isfinite(ne_c) and ne_c > 0 else math.nan
        )
        summary.append(row)

    return pd.DataFrame(records), pd.DataFrame(summary).sort_values("end_fraction") if summary else pd.DataFrame()


def _channels_to_reach(cumulative: pd.Series, threshold: float) -> int | None:
    values = pd.to_numeric(cumulative, errors="coerce").to_numpy(dtype=float)
    idx = np.flatnonzero(np.isfinite(values) & (values >= float(threshold)))
    return int(idx[0] + 1) if len(idx) else None


def _plot_cumulative_mass(frame: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    """Readable small multiples for poison-specific vs clean causal concentration."""
    if frame.empty:
        return

    intervals = (
        frame[["interval", "start_fraction", "end_fraction"]]
        .drop_duplicates()
        .assign(
            start_fraction=lambda x: pd.to_numeric(x["start_fraction"], errors="coerce"),
            end_fraction=lambda x: pd.to_numeric(x["end_fraction"], errors="coerce"),
        )
        .sort_values("end_fraction")
    )
    n = len(intervals)
    ncols = 3 if n > 3 else max(1, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.25 * ncols, 3.55 * nrows), sharey=True)
    axes = np.atleast_1d(axes).reshape(-1)
    poison_color = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])[0]
    clean_color = "0.42"

    for ax, rec in zip(axes, intervals.to_dict("records")):
        interval = str(rec["interval"])
        start_f = _finite(rec.get("start_fraction"))
        end_f = _finite(rec.get("end_fraction"))
        part = frame[frame["interval"].astype(str).eq(interval)].sort_values("rank")
        if part.empty:
            ax.axis("off")
            continue

        ranks = pd.to_numeric(part["rank"], errors="coerce")
        poison = pd.to_numeric(part["poisoning_excess_cumulative_share"], errors="coerce")
        clean = pd.to_numeric(part["clean_control_cumulative_share"], errors="coerce")
        ax.plot(ranks, poison, marker="o", markersize=3.4, linewidth=2.0,
                color=poison_color, label="Poison-specific |D_j|")
        ax.plot(ranks, clean, marker="o", markersize=2.8, linewidth=1.6,
                linestyle="--", color=clean_color, label="Clean drift |ΔU_clean|")
        ax.axhline(0.8, linestyle=":", linewidth=0.9, color="0.70")
        ax.axhline(0.9, linestyle=":", linewidth=0.9, color="0.80")
        ax.set_ylim(0.0, 1.03)
        ax.set_xlim(0.8, max(1.2, float(ranks.max()) + 0.2))
        ax.set_xticks(np.arange(1, int(ranks.max()) + 1))
        ax.grid(True, alpha=0.13)
        if np.isfinite(start_f) and np.isfinite(end_f):
            ax.set_title(f"{100*start_f:g}-{100*end_f:g}% of training", fontsize=10.5, fontweight="bold")
        elif np.isfinite(end_f):
            ax.set_title(f"to {100*end_f:g}% of training", fontsize=10.5, fontweight="bold")
        else:
            ax.set_title(interval, fontsize=10.5, fontweight="bold")

        # Numeric readout: no arbitrary verdict threshold; show the actual difference.
        k3 = min(3, len(part))
        p3 = float(poison.iloc[k3 - 1]) if k3 and np.isfinite(poison.iloc[k3 - 1]) else math.nan
        c3 = float(clean.iloc[k3 - 1]) if k3 and np.isfinite(clean.iloc[k3 - 1]) else math.nan
        kp80 = _channels_to_reach(poison, 0.8)
        kc80 = _channels_to_reach(clean, 0.8)
        diff = 100.0 * (p3 - c3) if np.isfinite(p3) and np.isfinite(c3) else math.nan
        lines = []
        if np.isfinite(p3) and np.isfinite(c3):
            lines.append(f"Top-{k3}: poison {100*p3:.0f}% | clean {100*c3:.0f}%")
            lines.append(f"Difference: {diff:+.0f} pp")
        if kp80 is not None and kc80 is not None:
            lines.append(f"Channels for 80%: {kp80} | {kc80}")
        if lines:
            ax.text(
                0.98, 0.06, "\n".join(lines), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.2,
                bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.92},
            )

    # Use any unused panel as the legend/explanation rather than crowding data panels.
    for ax in axes[n:]:
        ax.axis("off")
    if len(axes) > n:
        info = axes[n]
        info.axis("off")
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color=poison_color, linewidth=2.0, marker="o", label="Poison-specific excess |D_j|"),
            Line2D([0], [0], color=clean_color, linewidth=1.6, linestyle="--", marker="o", label="Clean-training drift"),
        ]
        info.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
        info.text(
            0.0, 0.62,
            "How to read\n\n"
            "At the same k:\n"
            "  solid above dashed -> poison-specific change\n"
            "  is more concentrated than clean drift.\n\n"
            "A smaller 'channels for 80%' value also means\n"
            "stronger concentration.\n\n"
            "This compares concentration, not channel identity.",
            transform=info.transAxes, ha="left", va="top", fontsize=9.2,
        )
    else:
        axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    for i, ax in enumerate(axes[:n]):
        if i % ncols == 0:
            ax.set_ylabel("Fraction of causal-change mass explained")
        ax.set_xlabel("Number of top channels kept")

    fig.suptitle("How many channels carry the causal change?", fontsize=14, fontweight="bold")
    _finish_figure(
        fig,
        output,
        "Each panel is one training interval. The poison-specific curve is the original |D_j| concentration result; "
        "the clean curve is a same-support reference. Compare solid vs dashed within a panel, especially for the first 1-3 channels.",
    )


def _parse_layer_from_unit_key(value: object) -> float:
    text = str(value).strip()
    match = re.match(r"^[ma](\d+)", text)
    return float(match.group(1)) if match else math.nan


def causal_support_dynamics(stage07_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Track whether the same overtopping support persists and how causal mass moves on it."""
    frame = _load_disruption_channels(stage07_dir)
    if frame.empty or "disruption_score" not in frame.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if "complete_u_j_comparison" in frame.columns:
        frame = frame[frame["complete_u_j_comparison"].map(truthy)].copy()
    frame["abs_D"] = pd.to_numeric(frame["disruption_score"], errors="coerce")
    frame = frame[np.isfinite(frame["abs_D"]) & (frame["abs_D"] > 0)].copy()
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    unit_col = "unit_key" if "unit_key" in frame.columns else (
        "channel_key" if "channel_key" in frame.columns else None
    )
    if unit_col is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    frame["unit_key"] = frame[unit_col].astype(str)
    frame["layer"] = frame["unit_key"].map(_parse_layer_from_unit_key)

    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []
    previous_units: set[str] | None = None
    previous_top3: set[str] | None = None

    interval_order = (
        frame[["interval", "start_fraction", "end_fraction"]]
        .drop_duplicates()
        .assign(end_fraction=lambda x: pd.to_numeric(x["end_fraction"], errors="coerce"))
        .sort_values("end_fraction")
    )
    for interval in interval_order["interval"].astype(str):
        group = frame[frame["interval"].astype(str).eq(interval)].copy()
        group = group.sort_values("abs_D", ascending=False)
        total = float(group["abs_D"].sum())
        if total <= 0:
            continue
        group["causal_mass_share"] = group["abs_D"] / total
        group["rank"] = np.arange(1, len(group) + 1)
        start = _finite(group.get("start_fraction", pd.Series([math.nan])).iloc[0])
        end = _finite(group.get("end_fraction", pd.Series([math.nan])).iloc[0])
        for rec in group.to_dict("records"):
            records.append({
                "interval": interval,
                "start_fraction": start,
                "end_fraction": end,
                "unit_key": str(rec["unit_key"]),
                "layer": _finite(rec.get("layer")),
                "abs_D": float(rec["abs_D"]),
                "causal_mass_share": float(rec["causal_mass_share"]),
                "rank": int(rec["rank"]),
            })

        units = set(group["unit_key"].astype(str))
        top3 = set(group.head(min(3, len(group)))["unit_key"].astype(str))
        if previous_units is None:
            jaccard = math.nan
            retained = math.nan
        else:
            union = units | previous_units
            jaccard = len(units & previous_units) / len(union) if union else math.nan
            retained = len(units & previous_units) / len(previous_units) if previous_units else math.nan
        if previous_top3 is None:
            top3_jaccard = math.nan
        else:
            union3 = top3 | previous_top3
            top3_jaccard = len(top3 & previous_top3) / len(union3) if union3 else math.nan
        summaries.append({
            "interval": interval,
            "start_fraction": start,
            "end_fraction": end,
            "n_positive_channels": int(len(units)),
            "support_jaccard_vs_previous_interval": jaccard,
            "previous_support_retained_fraction": retained,
            "top3_jaccard_vs_previous_interval": top3_jaccard,
            "dominant_unit_key": str(group.iloc[0]["unit_key"]),
            "dominant_unit_share": float(group.iloc[0]["causal_mass_share"]),
        })
        previous_units = units
        previous_top3 = top3

        if group["layer"].notna().any():
            for layer, sub in group.dropna(subset=["layer"]).groupby("layer"):
                layers.append({
                    "interval": interval,
                    "start_fraction": start,
                    "end_fraction": end,
                    "layer": int(layer),
                    "abs_D": float(sub["abs_D"].sum()),
                    "causal_mass_share": float(sub["abs_D"].sum() / total),
                    "n_channels": int(len(sub)),
                })

    return pd.DataFrame(records), pd.DataFrame(summaries), pd.DataFrame(layers)



def _plot_causal_handoff(
    channels: pd.DataFrame,
    summary: pd.DataFrame,
    layers: pd.DataFrame,
    output: Path,
) -> None:
    if channels.empty:
        return
    intervals = channels[["interval", "end_fraction"]].drop_duplicates().sort_values("end_fraction")
    interval_names = intervals["interval"].astype(str).tolist()
    labels = [f"{100*float(v):g}%" for v in intervals["end_fraction"]]
    unit_order = (
        channels.groupby("unit_key")["causal_mass_share"].max().sort_values(ascending=False).index.tolist()
    )
    matrix = (
        channels.pivot_table(index="unit_key", columns="interval", values="causal_mass_share", aggfunc="sum")
        .reindex(index=unit_order, columns=interval_names)
        .fillna(0.0)
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.6, max(7.2, 0.36 * len(unit_order) + 4.1)),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    image = axes[0].imshow(100.0 * matrix.to_numpy(float), aspect="auto", interpolation="nearest")
    axes[0].set_xticks(np.arange(len(labels)), labels)
    axes[0].set_yticks(np.arange(len(unit_order)), unit_order)
    axes[0].set_xlabel("End of training interval")
    axes[0].set_ylabel("Causal channel")
    axes[0].set_title("Which channels carry the poisoning-specific causal change?")
    cbar = fig.colorbar(image, ax=axes[0], fraction=0.025, pad=0.02)
    cbar.set_label("Share of causal change in that interval (%)")

    if not layers.empty:
        for layer, group in layers.groupby("layer"):
            group = group.sort_values("end_fraction")
            axes[1].plot(
                100 * group["end_fraction"].to_numpy(float),
                100 * group["causal_mass_share"],
                marker="o",
                label=f"layer {int(layer)}",
            )
        axes[1].set_ylabel("Share of causal change (%)")
        axes[1].set_xlabel("End of training interval")
        axes[1].set_ylim(0, 103)
        axes[1].legend(frameon=False, ncol=min(4, max(1, layers["layer"].nunique())))
        axes[1].set_title("How is the same causal change distributed across model depth?")
        axes[1].grid(True, alpha=0.2)
    else:
        axes[1].axis("off")

    fig.suptitle("Where does overtopping place causal control during poisoned training?")
    _finish_figure(
        fig,
        output,
        "Read each heatmap column as 100% of the poisoning-specific causal change in that interval. "
        "A few persistently dominant cells/layers indicate a stable causal corridor; moving dominance indicates migration.",
    )


def _plot_support_persistence(summary: pd.DataFrame, output: Path) -> None:
    if summary.empty:
        return
    frame = summary.sort_values("end_fraction")
    x = 100 * frame["end_fraction"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    ax.plot(x, frame["support_jaccard_vs_previous_interval"], marker="o", label="All causal channels")
    ax.plot(x, frame["top3_jaccard_vs_previous_interval"], marker="s", label="Three strongest channels")
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("End of training interval")
    ax.set_ylabel("Overlap with previous interval (0 = new, 1 = same)")
    ax.set_ylim(0, 1.04)
    ax.set_title("Does poisoning keep using the same overtopping channels?")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    _finish_figure(
        fig,
        output,
        "Values near 1 mean poisoning is reweighting a persistent causal support. "
        "Values near 0 mean the causal mechanism is moving to different channels.",
    )

def channel_geometry_associations(
    candidates: pd.DataFrame, controls: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_rows: list[dict[str, Any]] = []
    geometry_metrics = [
        "excess_update_norm",
        "poison_clean_cosine",
        "poison_orthogonal_fraction_to_clean",
        "excess_to_poison_norm_ratio",
        "excess_to_clean_norm_ratio",
    ]
    if not candidates.empty:
        for interval, group in candidates.groupby("interval"):
            for metric in geometry_metrics:
                if metric not in group.columns:
                    continue
                n, rho = _spearman(group["max_abs_disruption_score"], group[metric])
                corr_rows.append(
                    {
                        "interval": interval,
                        "end_fraction": _finite(group["end_fraction"].iloc[0]),
                        "geometry_metric": metric,
                        "n": n,
                        "spearman_rho_with_abs_D": rho,
                    }
                )

    paired_rows: list[dict[str, Any]] = []
    if not candidates.empty and not controls.empty:
        for metric in geometry_metrics:
            if metric not in candidates.columns or metric not in controls.columns:
                continue
            ctrl = (
                controls.groupby(["interval", "candidate_parameter_row_key"])[metric]
                .mean()
                .rename("control_mean")
                .reset_index()
            )
            cand = candidates[["interval", "parameter_row_key", "end_fraction", metric]].rename(
                columns={"parameter_row_key": "candidate_parameter_row_key", metric: "candidate_value"}
            )
            merged = cand.merge(ctrl, on=["interval", "candidate_parameter_row_key"], how="inner")
            if merged.empty:
                continue
            merged["candidate_minus_control"] = merged["candidate_value"] - merged["control_mean"]
            for interval, group in merged.groupby("interval"):
                diff = pd.to_numeric(group["candidate_minus_control"], errors="coerce").dropna().to_numpy(float)
                paired_rows.append(
                    {
                        "interval": interval,
                        "end_fraction": _finite(group["end_fraction"].iloc[0]),
                        "geometry_metric": metric,
                        "n_pairs": int(len(diff)),
                        "candidate_mean": float(group["candidate_value"].mean()),
                        "matched_control_mean": float(group["control_mean"].mean()),
                        "mean_candidate_minus_control": float(diff.mean()) if len(diff) else math.nan,
                        "median_candidate_minus_control": float(np.median(diff)) if len(diff) else math.nan,
                        "candidate_win_fraction": float(np.mean(diff > 0)) if len(diff) else math.nan,
                    }
                )
    return pd.DataFrame(corr_rows), pd.DataFrame(paired_rows)


def wanda_weighting_comparison(stage07_dir: Path) -> pd.DataFrame:
    path = stage07_dir / "training_example_scores_all_intervals.csv"
    if not path.is_file():
        return pd.DataFrame()
    scores = pd.read_csv(path, low_memory=False)
    if scores.empty or "interval" not in scores.columns:
        return pd.DataFrame()
    columns = [
        ("wanda_disruption_score", "causal_weighted_wanda"),
        ("wanda_unweighted_score", "unweighted_activation_update"),
        ("wanda_matched_random_score", "matched_random"),
    ]
    rows: list[dict[str, Any]] = []
    for interval, group in scores.groupby("interval"):
        start = _finite(group.get("start_fraction", pd.Series([math.nan])).iloc[0])
        end = _finite(group.get("end_fraction", pd.Series([math.nan])).iloc[0])
        for col, label in columns:
            if col not in group.columns:
                continue
            metrics = detection_metrics(group, score_column=col)
            rows.append(
                {
                    "interval": interval,
                    "start_fraction": start,
                    "end_fraction": end,
                    "score": label,
                    "roc_auc": metrics.get("roc_auc", math.nan),
                    "average_precision": metrics.get("average_precision", math.nan),
                    "precision_at_expected_poison_count": metrics.get(
                        "precision_at_expected_poison_count", math.nan
                    ),
                    "paired_poison_over_source_rate": metrics.get(
                        "paired_poison_over_source_rate", math.nan
                    ),
                    "poison_prevalence": metrics.get("poison_prevalence", math.nan),
                }
            )
    return pd.DataFrame(rows).sort_values(["end_fraction", "score"])



def _plot_wanda_comparison(frame: pd.DataFrame, output: Path) -> None:
    if frame.empty:
        return
    labels = {
        "causal_weighted_wanda": "Causal-weighted activation × update",
        "unweighted_activation_update": "Activation × update only",
        "matched_random": "Norm-matched random rows",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    for score, group in frame.groupby("score"):
        group = group.sort_values("end_fraction")
        x = 100 * group["end_fraction"].to_numpy(float)
        label = labels.get(str(score), str(score))
        axes[0].plot(x, group["roc_auc"], marker="o", label=label)
        axes[1].plot(x, group["paired_poison_over_source_rate"], marker="o", label=label)
    for ax in axes:
        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_xlabel("End of training interval")
        ax.set_ylim(0, 1.03)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Poison-example ranking quality (ROC AUC)")
    axes[0].set_title("Can the score rank poisoned examples?\n0.5 = random")
    axes[1].set_ylabel("Poison score > paired source score")
    axes[1].set_title("Does each poison outrank its source?\n0.5 = random")
    axes[1].legend(frameon=False)
    fig.suptitle("Does causal overtopping information improve poisoning detection?")
    _finish_figure(
        fig,
        output,
        "Useful outcome: the causal-weighted curve stays above both non-causal baselines and above the 0.5 random line. "
        "If it only matches the baselines, overtopping identity is not adding much detection information.",
    )

def wanda_specificity(frame: pd.DataFrame) -> pd.DataFrame:
    """Quantify what causal overtopping weighting adds beyond matched or unweighted scores."""
    if frame.empty:
        return pd.DataFrame()
    metrics = [
        "roc_auc",
        "average_precision",
        "precision_at_expected_poison_count",
        "paired_poison_over_source_rate",
    ]
    rows: list[dict[str, Any]] = []
    for interval, group in frame.groupby("interval"):
        by_score = {str(r["score"]): r for r in group.to_dict("records")}
        causal = by_score.get("causal_weighted_wanda")
        if causal is None:
            continue
        start = _finite(causal.get("start_fraction"))
        end = _finite(causal.get("end_fraction"))
        out: dict[str, Any] = {
            "interval": interval,
            "start_fraction": start,
            "end_fraction": end,
        }
        for metric in metrics:
            c = _finite(causal.get(metric))
            out[f"causal_{metric}"] = c
            for baseline in ("matched_random", "unweighted_activation_update"):
                rec = by_score.get(baseline)
                b = _finite(rec.get(metric)) if rec is not None else math.nan
                out[f"{baseline}_{metric}"] = b
                out[f"causal_minus_{baseline}_{metric}"] = c - b if np.isfinite(c) and np.isfinite(b) else math.nan
        rows.append(out)
    return pd.DataFrame(rows).sort_values("end_fraction") if rows else pd.DataFrame()



def _plot_wanda_specificity(frame: pd.DataFrame, output: Path) -> None:
    if frame.empty:
        return
    x = 100 * frame["end_fraction"].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    for baseline, label in (
        ("matched_random", "Gain over norm-matched random rows"),
        ("unweighted_activation_update", "Gain over activation × update only"),
    ):
        auc_col = f"causal_minus_{baseline}_roc_auc"
        ap_col = f"causal_minus_{baseline}_average_precision"
        if auc_col in frame:
            axes[0].plot(x, frame[auc_col], marker="o", label=label)
        if ap_col in frame:
            axes[1].plot(x, frame[ap_col], marker="o", label=label)
    for ax in axes:
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel("End of training interval")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Detection gain in ROC AUC")
    axes[0].set_title("Extra ranking power from causal weighting")
    axes[1].set_ylabel("Detection gain in average precision")
    axes[1].set_title("Extra recovery power from causal weighting")
    axes[1].legend(frameon=False)
    fig.suptitle("Does overtopping add detection power beyond simpler update signals?")
    _finish_figure(
        fig,
        output,
        "Positive values support practical value from overtopping-specific causal information. "
        "Zero means no added value; negative values mean the simpler baseline performed better.",
    )

def _combined_lowrank_row_energy(terms: list[tuple[float, LowRankProduct]]) -> torch.Tensor:
    """Exact per-output-row squared norm of sum_i coeff_i * scaling_i * B_i A_i."""
    if not terms:
        return torch.empty(0, dtype=torch.float64)
    out_dim = int(terms[0][1].B.shape[0])
    in_dim = int(terms[0][1].A.shape[1])
    B_parts: list[torch.Tensor] = []
    A_parts: list[torch.Tensor] = []
    for coeff, product in terms:
        if int(product.B.shape[0]) != out_dim or int(product.A.shape[1]) != in_dim:
            raise ValueError("Incompatible LoRA factors for exact row-energy calculation")
        B_parts.append(product.B.to(torch.float64) * (float(coeff) * float(product.scaling)))
        A_parts.append(product.A.to(torch.float64))
    B = torch.cat(B_parts, dim=1)
    A = torch.cat(A_parts, dim=0)
    gram = A @ A.transpose(0, 1)
    energies = torch.sum((B @ gram) * B, dim=1)
    return torch.clamp(energies, min=0.0)


def update_energy_enrichment(
    *,
    stage07_dir: Path,
    manifests: Mapping[str, Mapping[int, Mapping[str, Any]]],
    states: Mapping[tuple[str, int], State],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = sorted(set(manifests["clean"]).intersection(manifests["poisoned"]))
    interval_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for start_key, end_key in zip(shared, shared[1:]):
        start_fraction = float(manifests["clean"][start_key]["fraction"])
        end_fraction = float(manifests["clean"][end_key]["fraction"])
        interval = _interval_label(start_fraction, end_fraction)
        mapped_path = stage07_dir / interval / "mapped_disruptive_channels.csv"
        if not mapped_path.is_file():
            continue
        mapped = pd.read_csv(mapped_path, low_memory=False)
        if mapped.empty:
            continue
        if "mapped" in mapped.columns:
            mapped = mapped[mapped["mapped"].map(truthy)].copy()
        if mapped.empty:
            continue

        c0 = states[("clean", start_key)]
        c1 = states[("clean", end_key)]
        p0 = states[("poisoned", start_key)]
        p1 = states[("poisoned", end_key)]

        module_energy: dict[tuple[int, str], torch.Tensor] = {}
        total_energy = 0.0
        total_rows = 0
        common_modules = sorted(set(c0).intersection(c1).intersection(p0).intersection(p1))
        for key in common_modules:
            energy = _combined_lowrank_row_energy(
                [(+1.0, p1[key]), (-1.0, p0[key]), (-1.0, c1[key]), (+1.0, c0[key])]
            )
            module_energy[key] = energy
            total_energy += float(energy.sum().item())
            total_rows += int(len(energy))

        unique = mapped.drop_duplicates(["lora_layer", "lora_module", "effective_weight_row"])
        selected_energy = 0.0
        selected_rows = 0
        for rec in unique.to_dict("records"):
            key = (int(rec["lora_layer"]), str(rec["lora_module"]))
            row = int(rec["effective_weight_row"])
            if key not in module_energy or not (0 <= row < len(module_energy[key])):
                continue
            energies = module_energy[key].cpu().numpy().astype(float)
            value = float(energies[row])
            less = float(np.sum(energies < value))
            ties = float(np.sum(energies == value))
            percentile = (less + 0.5 * ties) / float(len(energies)) if len(energies) else math.nan
            selected_energy += value
            selected_rows += 1
            candidate_rows.append(
                {
                    "interval": interval,
                    "start_fraction": start_fraction,
                    "end_fraction": end_fraction,
                    "lora_layer": key[0],
                    "lora_module": key[1],
                    "effective_weight_row": row,
                    "excess_update_energy_sq": value,
                    "excess_update_norm": math.sqrt(max(0.0, value)),
                    "excess_energy_percentile_within_module": percentile,
                    "top_1pct_within_module": bool(percentile >= 0.99),
                    "top_5pct_within_module": bool(percentile >= 0.95),
                    "top_10pct_within_module": bool(percentile >= 0.90),
                }
            )

        energy_fraction = selected_energy / total_energy if total_energy > EPS else math.nan
        row_fraction = selected_rows / total_rows if total_rows else math.nan
        enrichment = energy_fraction / row_fraction if row_fraction and row_fraction > 0 else math.nan
        interval_rows.append(
            {
                "interval": interval,
                "start_fraction": start_fraction,
                "end_fraction": end_fraction,
                "total_excess_update_energy_sq": total_energy,
                "overtopping_excess_update_energy_sq": selected_energy,
                "overtopping_excess_update_energy_fraction": energy_fraction,
                "n_overtopping_parameter_rows": selected_rows,
                "n_all_lora_output_rows": total_rows,
                "overtopping_row_fraction": row_fraction,
                "overtopping_excess_energy_enrichment": enrichment,
            }
        )
    return pd.DataFrame(interval_rows), pd.DataFrame(candidate_rows)



def _plot_energy_enrichment(intervals: pd.DataFrame, channels: pd.DataFrame, output: Path) -> None:
    if intervals.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    x = 100 * intervals["end_fraction"].to_numpy(float)
    axes[0].plot(
        x,
        100 * intervals["overtopping_excess_update_energy_fraction"],
        marker="o",
        label="Poison-specific update energy on overtopping rows",
    )
    axes[0].plot(
        x,
        100 * intervals["overtopping_row_fraction"],
        marker="o",
        linestyle="--",
        label="Overtopping rows as share of all rows",
    )
    axes[0].set_ylabel("Share (%)")
    axes[0].set_title("Do overtopping rows capture more update energy\nthan their row count would predict?")
    axes[0].legend(frameon=False)

    axes[1].plot(x, intervals["overtopping_excess_energy_enrichment"], marker="o")
    axes[1].axhline(1.0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Poison-update energy enrichment")
    axes[1].set_title("Preferential writing onto overtopping rows\n1 = no enrichment")
    for ax in axes:
        ax.set_xlabel("End of training interval")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Are overtopping channels simply the parameters receiving unusually large poison updates?")
    _finish_figure(
        fig,
        output,
        "Enrichment > 1 supports preferential poison-specific writing onto overtopping rows. "
        "Enrichment near or below 1 argues that overtopping is a functional causal reorganization, not just large local updates.",
    )

def update_energy_percentile_summary(channels: pd.DataFrame) -> pd.DataFrame:
    if channels.empty or "excess_energy_percentile_within_module" not in channels.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for interval, group in channels.groupby("interval"):
        pct = pd.to_numeric(group["excess_energy_percentile_within_module"], errors="coerce").dropna()
        if pct.empty:
            continue
        rows.append({
            "interval": interval,
            "start_fraction": _finite(group.get("start_fraction", pd.Series([math.nan])).iloc[0]),
            "end_fraction": _finite(group.get("end_fraction", pd.Series([math.nan])).iloc[0]),
            "n_overtopping_rows": int(len(pct)),
            "mean_within_module_percentile": float(pct.mean()),
            "median_within_module_percentile": float(pct.median()),
            "top_1pct_fraction": float((pct >= 0.99).mean()),
            "top_5pct_fraction": float((pct >= 0.95).mean()),
            "top_10pct_fraction": float((pct >= 0.90).mean()),
        })
    return pd.DataFrame(rows).sort_values("end_fraction") if rows else pd.DataFrame()



def _plot_update_energy_percentiles(frame: pd.DataFrame, output: Path) -> None:
    if frame.empty:
        return
    x = 100 * frame["end_fraction"].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    axes[0].plot(x, 100 * frame["median_within_module_percentile"], marker="o", label="Median overtopping row")
    axes[0].plot(x, 100 * frame["mean_within_module_percentile"], marker="s", label="Mean overtopping row")
    axes[0].axhline(50.0, linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 103)
    axes[0].set_ylabel("Percentile among rows in the same projection")
    axes[0].set_title("How extreme are overtopping-row poison updates?\n50th percentile = non-overtopping row")
    axes[0].legend(frameon=False)

    axes[1].plot(x, 100 * frame["top_10pct_fraction"], marker="o", label="Observed overtopping rows in top 10%")
    axes[1].axhline(10.0, linestyle="--", linewidth=1, label="10% expected by chance")
    ymax = max(20.0, float(100 * frame["top_10pct_fraction"].max()) * 1.2 + 1.0)
    axes[1].set_ylim(0, ymax)
    axes[1].set_ylabel("Overtopping rows in top update decile (%)")
    axes[1].set_title("Are overtopping rows overrepresented\namong the largest poison-specific updates?")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xlabel("End of training interval")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Row-level check: does overtopping coincide with unusually large poison-specific updates?")
    _finish_figure(
        fig,
        output,
        "A strong local-update explanation would put percentiles near 100 and top-decile occupancy well above 10%. "
        "Otherwise, causal overtopping is not reducible to update magnitude.",
    )

def reconstruct_backdoor_behavior_from_cached_scores(
    *,
    run_dir: Path,
    phase: str,
    eval_intervention: str,
) -> pd.DataFrame:
    """Rebuild Stage-04 behavior trajectory directly from persisted scores.csv files."""
    manifests = _load_manifest_by_condition(run_dir)
    rows: list[dict[str, Any]] = []
    for condition in ("clean", "poisoned"):
        for _, row in sorted(manifests.get(condition, {}).items()):
            score_path = _behavior_score_path(
                run_dir,
                row,
                phase=phase,
                intervention=eval_intervention,
                kind="backdoor_trigger_test",
            )
            if not score_path.is_file():
                continue
            try:
                stats = _summarize_behavior_scores(score_path, "backdoor_trigger_test")
            except Exception as exc:
                print(f"[overtopping-geometry] behavior-score rebuild skipped {score_path}: {exc}", flush=True)
                continue
            rows.append({
                "condition": condition,
                "fraction": float(row.get("fraction", 0.0)),
                "global_step": int(float(row.get("global_step", 0))),
                **stats,
            })
    return pd.DataFrame(rows).sort_values(["condition", "fraction", "global_step"]) if rows else pd.DataFrame()


def complete_attack_detection_table(
    *,
    run_dir: Path,
    stage07_dir: Path,
    phase: str,
    eval_intervention: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prefer a direct reconstruction from cached behavior scores when it adds coverage."""
    combined_path = stage07_dir / "detection_vs_attack_success_by_interval.csv"
    if not combined_path.is_file():
        generate_implication_outputs(
            run_dir=run_dir,
            phase=phase,
            eval_intervention=eval_intervention,
            output_dir=stage07_dir,
        )
    existing = pd.read_csv(combined_path) if combined_path.is_file() else pd.DataFrame()
    metrics_path = stage07_dir / "detection_metrics_by_interval.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
    rebuilt_behavior = reconstruct_backdoor_behavior_from_cached_scores(
        run_dir=run_dir,
        phase=phase,
        eval_intervention=eval_intervention,
    )
    rebuilt = build_detection_vs_attack_table(metrics, rebuilt_behavior) if not metrics.empty and not rebuilt_behavior.empty else pd.DataFrame()

    def coverage(df: pd.DataFrame) -> int:
        if df.empty or "attack_behavior_available" not in df.columns:
            return 0
        return int(df["attack_behavior_available"].map(truthy).sum())

    chosen = rebuilt if coverage(rebuilt) > coverage(existing) else existing
    return chosen, rebuilt_behavior

def geometry_poisoning_associations(
    *,
    geometry: pd.DataFrame,
    disruption_info: pd.DataFrame,
    energy: pd.DataFrame,
    attack_detection: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = [df for df in (geometry, disruption_info, energy) if not df.empty]
    if not tables or attack_detection.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = tables[0].copy()
    for table in tables[1:]:
        keep = [c for c in table.columns if c not in merged.columns or c in {"start_fraction", "end_fraction", "interval"}]
        merged = merged.merge(table[keep], on=["start_fraction", "end_fraction"], how="outer", suffixes=("", "_dup"))
        dup = [c for c in merged.columns if c.endswith("_dup")]
        if dup:
            merged = merged.drop(columns=dup)
    merged = merged.merge(
        attack_detection,
        on=["start_fraction", "end_fraction"],
        how="left",
        suffixes=("", "_attack"),
    )

    predictors = [
        "total_abs_disruption_mass",
        "effective_support_renyi2",
        "top1_disruption_share",
        "poison_clean_cosine_fro",
        "poison_orthogonal_fraction_to_clean",
        "excess_to_poison_norm_ratio",
        "overtopping_excess_energy_enrichment",
        "overtopping_excess_update_energy_fraction",
    ]
    outcomes = [
        "roc_auc",
        "average_precision",
        "poisoned_conversion_change_over_interval",
        "conversion_change_gain_poisoned_vs_clean",
        "poisoned_conditional_conversion_rate",
    ]
    rows: list[dict[str, Any]] = []
    for pred in predictors:
        if pred not in merged.columns:
            continue
        for outcome in outcomes:
            if outcome not in merged.columns:
                continue
            n_s, rho = _spearman(merged[pred], merged[outcome])
            n_p, r = _pearson(merged[pred], merged[outcome])
            rows.append(
                {
                    "predictor": pred,
                    "outcome": outcome,
                    "n": min(n_s, n_p),
                    "spearman_rho": rho,
                    "pearson_r": r,
                    "note": "descriptive across training intervals; very small n, not an inferential test",
                }
            )
    return merged.sort_values("end_fraction"), pd.DataFrame(rows)




def _plot_geometry_vs_acquisition(merged: pd.DataFrame, output: Path) -> None:
    if merged.empty or "poisoned_conversion_change_over_interval" not in merged.columns:
        return

    y_all = pd.to_numeric(merged["poisoned_conversion_change_over_interval"], errors="coerce")
    end_all = pd.to_numeric(merged.get("end_fraction"), errors="coerce")
    available = y_all.notna() & end_all.notna()
    total_intervals = int(end_all.notna().sum())
    n_available = int(available.sum())

    if n_available < 3:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        if n_available:
            ax.plot(
                100 * end_all[available],
                100 * y_all[available],
                marker="o",
                linestyle="-",
            )
            for idx in merged.index[available]:
                ax.annotate(
                    f"to {100*float(end_all.loc[idx]):g}%",
                    (100 * float(end_all.loc[idx]), 100 * float(y_all.loc[idx])),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.set_xlabel("End of training interval")
        ax.grid(True, alpha=0.2)
        ax.set_title("Attack-link analysis is incomplete — correlation should not be interpreted")
        ax.text(
            0.5,
            0.60,
            f"Only {n_available} of {total_intervals} training intervals have aligned attack-behavior data.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.text(
            0.5,
            0.50,
            "The overtopping quantities are available, but the behavior side is missing at later checkpoints.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
        _finish_figure(
            fig,
            output,
            "No geometric/attack-growth relationship is claimed from fewer than 3 aligned intervals. "
            "Use this panel as a coverage warning, not evidence for or against overtopping.",
        )
        return

    candidates = [
        ("total_abs_disruption_mass", "Total poisoning-specific causal change"),
        ("effective_support_renyi2", "Effective number of causal channels"),
        ("poison_orthogonal_fraction_to_clean", "Poison update outside clean-update direction"),
        ("overtopping_excess_energy_enrichment", "Poison-update enrichment on overtopping rows"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6), sharey=True)
    for ax, (xcol, xlabel) in zip(axes.flat, candidates):
        if xcol not in merged.columns:
            ax.axis("off")
            continue
        x = pd.to_numeric(merged[xcol], errors="coerce")
        y = y_all
        ok = x.notna() & y.notna()
        ax.scatter(x[ok], 100 * y[ok], s=48)
        for idx in merged.index[ok]:
            end = _finite(merged.loc[idx, "end_fraction"])
            ax.annotate(
                f"to {100*end:g}%",
                (x.loc[idx], 100 * y.loc[idx]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.2)
    axes[0, 0].set_ylabel("Backdoor conversion gained\nin interval (percentage points)")
    axes[1, 0].set_ylabel("Backdoor conversion gained\nin interval (percentage points)")
    fig.suptitle("Do overtopping signals rise when the backdoor is being acquired?", y=0.995)
    _finish_figure(
        fig,
        output,
        "Useful pattern: intervals with stronger causal reorganization or stronger concentration coincide with larger backdoor growth. "
        "This is descriptive within one training trajectory; replication across seeds is needed for a robust claim.",
    )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--phase", choices=["input_output", "output_only"], default=None)
    parser.add_argument("--eval_intervention", default="mean-donor")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: <Stage07>/<phase>/overtopping_interpretation",
    )
    parser.add_argument("--max_control_draws", type=int, default=100)
    parser.add_argument(
        "--skip_update_energy",
        action="store_true",
        help="Skip all-row LoRA excess-update enrichment (still no inference; useful if checkpoints are unavailable).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    definition = infer_task_from_run(run_dir)
    phase = args.phase or definition.default_phase
    stage07_dir = detection_dir(run_dir) / phase_dirname(phase)
    if not stage07_dir.is_dir():
        raise FileNotFoundError(f"Missing Stage-07 directory: {stage07_dir}")

    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else stage07_dir / "overtopping_interpretation"
    )
    dirs = _ensure_output_tree(output_root)
    _write_result_guide(output_root)

    # Refresh user-facing outputs owned by this analysis. Intermediate geometry
    # tables are retained separately because they can be reused without inference.
    for key in ("mechanism", "detection", "update_checks", "behavior"):
        for path in dirs[key].glob("*.pdf"):
            path.unlink()
    for path in dirs["tables"].glob("*.csv"):
        path.unlink()

    geometry_dir = dirs["intermediate"] / "base_update_geometry"
    _ensure_base_geometry(
        run_dir=run_dir,
        phase=phase,
        geometry_dir=geometry_dir,
        max_control_draws=int(args.max_control_draws),
    )

    geometry = pd.read_csv(geometry_dir / "update_geometry_by_interval.csv")
    candidates = pd.read_csv(geometry_dir / "channel_update_geometry.csv")
    controls = pd.read_csv(geometry_dir / "matched_control_update_geometry.csv")
    disruption_info = pd.read_csv(geometry_dir / "disruption_information_by_interval.csv")

    # 1) Additional causal-mechanism stories not present in the compact
    # checkpoint story: concentration relative to clean drift, channel/layer
    # handoff, and support persistence.
    cumulative, cumulative_summary = causal_disruption_concentration(stage07_dir)
    cumulative.to_csv(dirs["tables"] / "causal_concentration_rank_curves_vs_clean_control.csv", index=False)
    cumulative_summary.to_csv(dirs["tables"] / "causal_concentration_vs_clean_control_by_interval.csv", index=False)
    _plot_cumulative_mass(
        cumulative,
        cumulative_summary,
        dirs["mechanism"] / "03_poisoning_specific_causal_concentration_vs_clean_drift.pdf",
    )

    support_channels, support_summary, layer_mass = causal_support_dynamics(stage07_dir)
    support_channels.to_csv(dirs["tables"] / "causal_channel_share_by_interval.csv", index=False)
    support_summary.to_csv(dirs["tables"] / "causal_support_persistence_by_interval.csv", index=False)
    layer_mass.to_csv(dirs["tables"] / "causal_layer_share_by_interval.csv", index=False)
    _plot_causal_handoff(
        support_channels,
        support_summary,
        layer_mass,
        dirs["mechanism"] / "01_where_poisoning_specific_causal_control_moves.pdf",
    )
    _plot_support_persistence(
        support_summary,
        dirs["mechanism"] / "02_does_the_same_causal_support_persist.pdf",
    )

    # 2) Does overtopping add poison-detection information beyond simpler signals?
    channel_corr, paired = channel_geometry_associations(candidates, controls)
    channel_corr.to_csv(dirs["tables"] / "channel_geometry_vs_causal_effect.csv", index=False)
    paired.to_csv(dirs["tables"] / "overtopping_vs_matched_row_geometry.csv", index=False)

    wanda = wanda_weighting_comparison(stage07_dir)
    wanda.to_csv(dirs["tables"] / "detector_components_by_interval.csv", index=False)
    _plot_wanda_comparison(
        wanda,
        dirs["detection"] / "02_detector_components_vs_baselines.pdf",
    )
    specificity = wanda_specificity(wanda)
    specificity.to_csv(dirs["tables"] / "overtopping_detection_gain_by_interval.csv", index=False)
    _plot_wanda_specificity(
        specificity,
        dirs["detection"] / "01_does_overtopping_add_detection_power.pdf",
    )

    # 3) Diagnostic control: is overtopping just where poisoning writes the largest updates?
    energy = pd.DataFrame()
    energy_channels = pd.DataFrame()
    if not args.skip_update_energy:
        manifests = _load_manifest_by_condition(run_dir)
        states, _ = _load_all_states(run_dir, manifests)
        energy, energy_channels = update_energy_enrichment(
            stage07_dir=stage07_dir,
            manifests=manifests,
            states=states,
        )
        energy.to_csv(dirs["tables"] / "overtopping_update_energy_by_interval.csv", index=False)
        energy_channels.to_csv(dirs["tables"] / "overtopping_row_update_energy_percentiles.csv", index=False)
        _plot_energy_enrichment(
            energy,
            energy_channels,
            dirs["update_checks"] / "01_are_overtopping_rows_preferentially_updated.pdf",
        )
        energy_pct = update_energy_percentile_summary(energy_channels)
        energy_pct.to_csv(dirs["tables"] / "overtopping_update_percentile_summary.csv", index=False)
        _plot_update_energy_percentiles(
            energy_pct,
            dirs["update_checks"] / "02_how_extreme_are_overtopping_row_updates.pdf",
        )

    # 4) Descriptive link to attack acquisition.
    attack_detection, rebuilt_behavior = complete_attack_detection_table(
        run_dir=run_dir,
        stage07_dir=stage07_dir,
        phase=phase,
        eval_intervention=args.eval_intervention,
    )
    if not rebuilt_behavior.empty:
        rebuilt_behavior.to_csv(
            dirs["tables"] / "reconstructed_backdoor_behavior_by_checkpoint.csv",
            index=False,
        )
    merged, assoc = geometry_poisoning_associations(
        geometry=geometry,
        disruption_info=disruption_info,
        energy=energy,
        attack_detection=attack_detection,
    )
    merged.to_csv(dirs["tables"] / "overtopping_signals_and_attack_growth_by_interval.csv", index=False)
    assoc.to_csv(dirs["tables"] / "descriptive_signal_attack_associations.csv", index=False)
    _plot_geometry_vs_acquisition(
        merged,
        dirs["behavior"] / "01_do_overtopping_signals_track_attack_growth.pdf",
    )

    print(f"[done] overtopping interpretation outputs: {output_root}", flush=True)
    print("  Start with README.md and the figures in 01_overtopping_mechanism/", flush=True)



if __name__ == "__main__":
    main()
