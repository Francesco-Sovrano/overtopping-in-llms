#!/usr/bin/env python3
"""Regenerate manuscript figures using the attached paper as visual reference.

The plotting choices here deliberately follow the established figure family in
that paper, including which labels, legends, statistics and panel marks are
shown.  The module reads materialized result tables only; it does not rerun
experiments or redefine populations/statistics.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from reporting.generate_manuscript_tables import generate_all as generate_manuscript_tables


# ---------------------------------------------------------------------------
# Manuscript plotting style
# ---------------------------------------------------------------------------
#
# Keep the visual contract in this file.  The attached manuscript figures are
# the source of truth, and the native sizes below preserve the figure-family
# aspect ratios and apparent font/marker/line sizes after LaTeX inclusion.

FIGSIZE = {
    "main_rq1": (545.77734375 / 72.0, 313.0 / 72.0),
    "main_rq2": (446.4 / 72.0, 154.8 / 72.0),
    "main_rq3": (942.6364135742188 / 72.0, 264.8153381347656 / 72.0),
    "main_rq4_traj": (373.474 / 72.0, 172.845 / 72.0),
    "main_rq4_defense": (882.882 / 72.0, 264.835 / 72.0),
    "supp_rq1_phase": (405.334 / 72.0, 230.68 / 72.0),
    "supp_rq1_size": (399.473 / 72.0, 233.487 / 72.0),
    "supp_rq2_decomp": (371.573 / 72.0, 175.933 / 72.0),
    "supp_rq2_matched": (407.871 / 72.0, 286.813 / 72.0),
    "supp_rq3_candidate": (634.513 / 72.0, 258.046 / 72.0),
    "supp_rq3_strength_matched": (402.074 / 72.0, 165.07 / 72.0),
    "supp_rq3_tecs": (401.793 / 72.0, 150.389 / 72.0),
    "supp_rq3_event": (456.636 / 72.0, 287.678 / 72.0),
    "supp_rq3_pair": (285.812 / 72.0, 214.741 / 72.0),
    "supp_rq3_arithmetic": (454.461 / 72.0, 287.951 / 72.0),
    "supp_rq3_transient": (327.828 / 72.0, 170.389 / 72.0),
    "supp_rq3_preemption": (430.546 / 72.0, 186.389 / 72.0),
    "supp_rq3_temporal": (560.193 / 72.0, 134.83 / 72.0),
    "supp_rq3_cross": (634.513 / 72.0, 258.046 / 72.0),
}

BASE_RC = {
    "font.family": "sans-serif",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


@contextmanager
def ref_rc(*, font=8.0, label=9.0, tick=8.0, legend=8.0, title=9.0):
    cfg = dict(BASE_RC)
    cfg.update({
        "font.size": font,
        "axes.labelsize": label,
        "axes.titlesize": title,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "legend.fontsize": legend,
    })
    with plt.rc_context(cfg):
        yield


def clean_axes(ax, *, grid="y", alpha=0.20, linewidth=0.45):
    if grid == "both":
        ax.grid(True, alpha=alpha, linewidth=linewidth)
    elif grid:
        ax.grid(axis=grid, alpha=alpha, linewidth=linewidth)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def save_exact(fig, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", required=True)
    p.add_argument("--main-dir", default=None)
    p.add_argument("--supp-dir", default=None)
    p.add_argument("--tables-dir", default=None)
    return p.parse_args()


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.is_file() else pd.DataFrame()


def _num(x):
    return pd.to_numeric(x, errors="coerce")


def _p_plain(value: object) -> str:
    try:
        p = float(value)
    except Exception:
        return "p=n/a"
    if not np.isfinite(p):
        return "p=n/a"
    if p < 1e-4:
        return "p<0.0001"
    if p < .001:
        return f"p={p:.1e}"
    return f"p={p:.5f}".rstrip("0")


def _copy(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _model_tag(model: object) -> str:
    s = str(model)
    low = s.lower()
    if "qwen2.5-1.5b" in low:
        return "Q2.5"
    if "qwen2-1.5b" in low:
        return "Q1.5"
    if "qwen2-7b" in low:
        return "Q7"
    if "pythia-6.9b" in low:
        return "P6.9"
    if "pythia-1b" in low:
        if "48000" in low or "48k" in low:
            return "P1-48k"
        if "96000" in low or "96k" in low:
            return "P1-96k"
        if "step0" in low or "@0" in low:
            return "P1-0"
        if "143000" in low or "143k" in low:
            return "P1-143k"
        return "P1"
    return s.replace("-Instruct", "")



# ---------------------------------------------------------------------------
# Main figures
# ---------------------------------------------------------------------------

def _rq1_clustered_labels(fig, ax, records, *, fontsize=6.2):
    """Place grouped model labels without text-text or text-point collisions."""
    if not records:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = ax.transData.transform([(x, y) for x, y, _ in records])
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    # Combine only points that are close enough that separate labels would
    # collide at manuscript scale.
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            if dx * dx + dy * dy <= 38.0 ** 2:
                union(i, j)

    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)

    axes_box = ax.get_window_extent(renderer)
    occupied = []
    # Reserve marker neighborhoods.
    for xdisp, ydisp in pts:
        occupied.append(Bbox.from_extents(xdisp - 14, ydisp - 12, xdisp + 14, ydisp + 12))
    # Reserve the statistics annotation and any other pre-existing text.
    for t in ax.texts:
        if t.get_text():
            occupied.append(t.get_window_extent(renderer).expanded(1.05, 1.08))

    directions = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
    candidates = []
    for radius in (14, 22, 32, 46, 62, 82, 104):
        for dx, dy in directions:
            candidates.append((dx * radius, dy * radius))
    candidates.extend([(120,0),(-120,0),(0,110),(0,-110),(92,44),(-92,44),(92,-44),(-92,-44)])

    # Dense low-value clusters are the hardest, so place them first.
    order = sorted(groups.values(), key=lambda ids: (min(records[i][1] for i in ids), -len(ids)))
    for ids in order:
        xs = [records[i][0] for i in ids]
        ys = [records[i][1] for i in ids]
        anchor = (float(np.mean(xs)), float(np.mean(ys)))
        labels = []
        for i in ids:
            lab = str(records[i][2])
            if lab not in labels:
                labels.append(lab)
        if len(labels) >= 4:
            split = (len(labels) + 1) // 2
            label = "/".join(labels[:split]) + "\n" + "/".join(labels[split:])
        elif len(labels) == 3:
            label = labels[0] + "/" + labels[1] + "\n" + labels[2]
        else:
            label = "/".join(labels)

        best_off = candidates[0]
        best_score = float("inf")
        for off in candidates:
            ann = ax.annotate(
                label, anchor, xytext=off, textcoords="offset points",
                fontsize=fontsize,
                ha="left" if off[0] > 0 else ("right" if off[0] < 0 else "center"),
                va="bottom" if off[1] > 0 else ("top" if off[1] < 0 else "center"),
                color="0.10", linespacing=.92, zorder=8,
                bbox={"boxstyle":"round,pad=.14", "facecolor":"white", "edgecolor":"0.70", "linewidth":.35, "alpha":0.72},
                arrowprops={"arrowstyle":"-", "lw":.36, "color":"0.38", "alpha":.72,
                            "shrinkA":1.5, "shrinkB":2.5},
            )
            bb = ann.get_window_extent(renderer).expanded(1.04, 1.10)
            ann.remove()
            outside = (
                max(0.0, axes_box.x0 - bb.x0) + max(0.0, bb.x1 - axes_box.x1) +
                max(0.0, axes_box.y0 - bb.y0) + max(0.0, bb.y1 - axes_box.y1)
            )
            n_overlap = sum(bb.overlaps(prev) for prev in occupied)
            score = 10000.0 * outside + 1000.0 * n_overlap + (abs(off[0]) + abs(off[1])) / 50.0
            if score < best_score:
                best_score = score
                best_off = off
            if outside == 0 and n_overlap == 0:
                break

        off = best_off
        ann = ax.annotate(
            label, anchor, xytext=off, textcoords="offset points",
            fontsize=fontsize,
            ha="left" if off[0] > 0 else ("right" if off[0] < 0 else "center"),
            va="bottom" if off[1] > 0 else ("top" if off[1] < 0 else "center"),
            color="0.10", linespacing=.92, zorder=8,
            bbox={"boxstyle":"round,pad=.18", "facecolor":"white", "edgecolor":"0.65", "linewidth":.4, "alpha":0.88},
            arrowprops={"arrowstyle":"-", "lw":.36, "color":"0.38", "alpha":.72,
                        "shrinkA":1.5, "shrinkB":2.5},
        )
        occupied.append(ann.get_window_extent(renderer).expanded(1.04, 1.10))

def main_rq1(root: Path, out: Path) -> None:
    """Current RQ1 data in the attached paper's 2x2 visual grammar."""
    base = root / "analysis/figure_data/02_rq1_prevalence"
    frames: dict[str, pd.DataFrame] = {}
    stats: dict[str, pd.DataFrame] = {}
    for direction, stem in [
        ("0to1", "fig2a_competence_vs_U_0to1"),
        ("1to0", "fig2b_competence_vs_U_1to0"),
    ]:
        frames[direction] = read(base / f"{stem}.csv")
        stats[direction] = read(base / f"{stem}_stats.csv")
    if any(v.empty for v in frames.values()):
        return

    task_order = ["arithmetic", "hans_nli", "grammar_acceptability", "random_fsm", "bon_jailbreaking"]
    task_labels = {
        "arithmetic": "Arithmetic",
        "hans_nli": "HANS NLI",
        "grammar_acceptability": "grammar",
        "random_fsm": "Random FSM",
        "bon_jailbreaking": "Jailbreak",
    }
    # Keep the paper's stable task-color identity. HANS is green and grammar is
    # orange; colors do not shift when a task/model condition is absent.
    colors = {
        "arithmetic": "#1f77b4",
        "hans_nli": "#2ca02c",
        "grammar_acceptability": "#ff7f0e",
        "random_fsm": "#d62728",
        "bon_jailbreaking": "#9467bd",
    }
    phases = ["input+output", "decode-only"]

    with plt.rc_context({
        "font.size": 8.2,
        "axes.labelsize": 10.2,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.4,
        "legend.fontsize": 7.4,
        "axes.linewidth": .75,
        "xtick.major.width": .7,
        "ytick.major.width": .7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["main_rq1"], sharex="col", sharey="row")
        for ri, direction in enumerate(["0to1", "1to0"]):
            d = frames[direction].copy()
            d["score"] = _num(d["score"])
            d["union_rate"] = _num(d["union_rate"])
            for ci, phase in enumerate(phases):
                ax = axes[ri, ci]
                g = d[d["phase"].astype(str).eq(phase)].copy()
                label_records=[]
                for _, r in g.iterrows():
                    task = str(r.get("task", ""))
                    if task not in colors or not np.isfinite(r["score"]) or not np.isfinite(r["union_rate"]):
                        continue
                    marker = "o" if str(r.get("baseline", "mean-donor")) == "mean-donor" else "s"
                    x=float(r["score"]); y=float(r["union_rate"])
                    ax.scatter(x, y, s=37, marker=marker, color=colors[task],
                               edgecolor="black", linewidth=.45, zorder=4)
                    label_records.append((x,y,_model_tag(r.get("model", ""))))

                fit = g[["score", "union_rate"]].dropna()
                if len(fit) >= 3 and fit["score"].nunique() > 1:
                    coef = np.polyfit(fit["score"].to_numpy(float), fit["union_rate"].to_numpy(float), 1)
                    xx = np.linspace(float(fit["score"].min()), float(fit["score"].max()), 100)
                    ax.plot(xx, coef[0] * xx + coef[1], "--", color="0.32", linewidth=1.12, zorder=2)

                ax.set_xlim(-.01, 1.02)
                ax.set_ylim(-.02, 1.03)
                ax.grid(True, alpha=.19, linewidth=.42)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                sr = stats[direction]
                sr = sr[sr["phase"].astype(str).eq(phase)] if not sr.empty else sr
                if not sr.empty:
                    q = sr.iloc[0]
                    pval=float(q["pearson_p"])
                    ptxt=f"{pval:.5f}" if pval >= 1e-4 else f"{pval:.5f}"
                    txt = f"$n={int(q['n'])},\\;r={float(q['pearson_r']):.2f}$\n$p={ptxt}$"
                    xy=(.98,.95) if (ri==0 and ci==1) else (.98,.07)
                    ax.text(*xy, txt, transform=ax.transAxes, ha="right",
                            va="top" if (ri==0 and ci==1) else "bottom", fontsize=7.2,
                            bbox={"boxstyle":"round,pad=.18","facecolor":"white",
                                  "edgecolor":"0.72","linewidth":.55,"alpha":.94}, zorder=9)
                _rq1_clustered_labels(fig, ax, label_records, fontsize=6.2)

        axes[0,0].set_ylabel(r"$0\!\to\!1$ reach $U^d(J)$")
        axes[1,0].set_ylabel(r"$1\!\to\!0$ reach $U^d(J)$")
        axes[1,0].set_xlabel("Raw task score")
        axes[1,1].set_xlabel(r"Chance-normalized score $\kappa$")
        handles=[Line2D([0],[0],marker="o",linestyle="none",markerfacecolor=colors[t],
                        markeredgecolor="black",markeredgewidth=.4,markersize=5.1,label=task_labels[t])
                 for t in task_order]
        fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.005),ncol=5,
                   frameon=False,handletextpad=.35,columnspacing=.85)
        fig.subplots_adjust(left=.095,right=.995,bottom=.16,top=.985,hspace=.10,wspace=.12)
        save_exact(fig,out)


def main_rq2(root: Path, out: Path) -> None:
    regime_path = root / "analysis/rq2_composition/regime_summary/rq2_settings_by_replacement_regime.csv"
    d = read(regime_path)
    value_col = "Delta_comp"
    if d.empty or "scope" not in d.columns:
        donor = read(root / "analysis/rq2_composition/interaction_decomposition/composition_decomposition_all_scopes.csv")
        mean = read(root / "analysis/rq2_composition/interaction_decomposition_mean/composition_decomposition_all_scopes.csv")
        frames = []
        if not donor.empty:
            donor = donor.copy()
            donor["replacement_regime"] = "mean-donor"
            frames.append(donor)
        if not mean.empty:
            mean = mean.copy()
            mean["replacement_regime"] = "mean"
            frames.append(mean)
        if not frames:
            return
        d = pd.concat(frames, ignore_index=True, sort=False)
        value_col = "Delta_comp_complete_case"
    elif "rq2_evaluable" in d.columns:
        d = d.loc[d["rq2_evaluable"].astype(bool)].copy()

    d[value_col] = _num(d[value_col])
    d = d[np.isfinite(d[value_col])].copy()
    if d.empty:
        return

    regimes = ["mean-donor", "mean"]
    regime_labels = {"mean-donor": "Mean-donor", "mean": "Direct mean"}
    regime_colors = {"mean-donor": "C0", "mean": "C1"}
    regime_markers = {"mean-donor": "o", "mean": "s"}
    scopes = ["0to1", "1to0", "overall"]
    scope_labels = [r"$0\rightarrow1$", r"$1\rightarrow0$", "All directions"]
    centers = np.array([1.0, 2.0, 3.0])
    offsets = {"mean-donor": -0.21, "mean": 0.21}
    width = 0.30

    with ref_rc(font=10.0, label=10.2, tick=10.0, legend=9.2):
        fig, ax = plt.subplots(figsize=(FIGSIZE["main_rq2"][0], FIGSIZE["main_rq2"][1] * 0.97))
        legend_handles = []
        for regime in regimes:
            color = regime_colors[regime]
            marker = regime_markers[regime]
            positions, box_vals = [], []
            for center, scope in zip(centers, scopes):
                vals = d.loc[
                    d["replacement_regime"].astype(str).eq(regime)
                    & d["scope"].astype(str).eq(scope),
                    value_col,
                ].to_numpy(float)
                if len(vals) == 0:
                    continue
                positions.append(center + offsets[regime])
                box_vals.append(vals)
            if not box_vals:
                continue

            bp = ax.boxplot(
                box_vals,
                positions=positions,
                widths=width,
                showfliers=False,
                patch_artist=True,
                medianprops={"linewidth": 1.35},
                whiskerprops={"linewidth": 1.0},
                capprops={"linewidth": 1.0},
            )
            for i, vals in enumerate(box_vals):
                bp["boxes"][i].set(facecolor=color, alpha=.10, edgecolor=color, linewidth=1.15)
                bp["medians"][i].set(color=color, linewidth=1.35)
                for art in bp["whiskers"][2*i:2*i+2] + bp["caps"][2*i:2*i+2]:
                    art.set(color=color, linewidth=1.0)
                # Spread observations horizontally so duplicates remain visible.
                jitter = np.linspace(-0.075, 0.075, len(vals)) if len(vals) > 1 else np.array([0.0])
                ax.scatter(
                    np.full(len(vals), positions[i]) + jitter,
                    np.sort(vals),
                    s=16,
                    marker=marker,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=.35,
                    alpha=.76,
                    zorder=3,
                )
            legend_handles.append(
                Line2D([0], [0], marker=marker, linestyle="none", markersize=7.0,
                       markerfacecolor=color, markeredgecolor=color, markeredgewidth=.8,
                       label=regime_labels[regime])
            )

        ax.axhline(0, linestyle="--", color="0.45", linewidth=.8, zorder=0)
        ax.set_xticks(centers, scope_labels)
        ax.set_xlabel("Intervention direction", labelpad=3)
        ax.set_ylabel(r"Composition gap $E(J)-U(J)$", labelpad=4)
        ax.set_xlim(.50, 3.50)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(axis="y", alpha=.13, linewidth=.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(.5, 1.005),
                ncol=2,
                handletextpad=.45,
                columnspacing=1.2,
                borderaxespad=0,
            )
        fig.subplots_adjust(left=.145, right=.995, bottom=.17, top=.84)
        save_exact(fig, out)


def main_rq3(root: Path, out: Path) -> None:
    graded = root / "analysis/rq3_threshold_event/spiking_diagnostics/graded_agonist"
    temporal = root / "analysis/rq3_threshold_event/spiking_diagnostics/temporal_cutoff"
    try:
        from studies.overtopping.analysis.stage11_rq3_temporal_cutoff_story import (
            _load_stage10_fig4e_inputs,
            _plot_fig4e_event_strength_temporal,
        )
        event_profile, event_stats, strength, strength_stats = _load_stage10_fig4e_inputs(graded)
        prefix_aligned = read(temporal / "temporal_cross_sweep_prefix_event_suffix_aligned.csv")
        prefix_stats = read(temporal / "temporal_cross_sweep_prefix_event_suffix_condition_stats.csv")
        if not event_profile.empty and not strength.empty and not prefix_aligned.empty and not prefix_stats.empty:
            _plot_fig4e_event_strength_temporal(
                event_profile, event_stats, strength, strength_stats,
                prefix_aligned, prefix_stats, out,
            )
            return
    except Exception:
        pass
    src = root / "paper/figures/04_rq3_spiking_cut/fig4e_population_event_and_strength.pdf"
    _copy(src, out)


def _checkpoint_rows(root: Path) -> pd.DataFrame:
    d = read(root / "analysis/reproducibility/primary_population_snapshot.csv")
    if d.empty:
        return d
    d = d[(d["phase"].astype(str).eq("I+O")) &
          (d["intervention"].astype(str).eq("mean-donor")) &
          d["task"].isin(["Grammar", "HANS NLI", "Random FSM"])].copy()

    def step(model: object):
        s = str(model).lower()
        if "step0" in s or "@0" in s or " 0k" in s:
            return 0
        if "48000" in s or "48k" in s:
            return 48000
        if "96000" in s or "96k" in s:
            return 96000
        if "pythia-1b" in s and "step" not in s and "@" not in s:
            return 143000
        return np.nan

    d["step"] = d["model"].map(step)
    return d[d["step"].notna()].copy()


def _annotate_checkpoint_values(ax, xs, ys, color, *, above=True):
    """Compact bordered value labels, following the reference trajectory."""
    finite = [(float(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(y)]
    by_x: dict[float, list[float]] = {}
    for x, y in finite:
        by_x.setdefault(x, []).append(y)
    for x, yvals in by_x.items():
        yvals = sorted(yvals)
        for j, y in enumerate(yvals):
            dx = 8 if x <= np.mean(xs) else -8
            dy = 6 + 8*j if above else -(8 + 8*j)
            ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                        ha="left" if dx > 0 else "right", va="bottom" if dy > 0 else "top",
                        fontsize=6.8, color="0.08",
                        bbox={"boxstyle": "round,pad=.08", "facecolor": "white",
                              "edgecolor": color, "linewidth": .55, "alpha": .92},
                        arrowprops={"arrowstyle": "-", "lw": .35, "color": color, "alpha": .60,
                                    "shrinkA": 0, "shrinkB": 0}, zorder=6)


def main_rq4_traj(root: Path, out: Path) -> None:
    d = _checkpoint_rows(root)
    if d.empty:
        return
    steps = np.asarray([0, 48000, 96000, 143000], dtype=float)
    step_labels = ["0", "48k", "96k", "143k"]
    specs = [("Grammar", "grammar"), ("HANS NLI", "HANS-NLI"), ("Random FSM", "Random FSM")]

    # Include only tasks with at least one measured value; missing measurements
    # stay missing and are never rendered as zero.
    series = []
    for task, label in specs:
        g = d[d["task"].astype(str).eq(task)]
        cov, comp = [], []
        for step in steps:
            h = g[np.isclose(_num(g["step"]).to_numpy(float), step, equal_nan=False)]
            if h.empty:
                cov.append(np.nan); comp.append(np.nan); continue
            r = h.iloc[-1]
            u = float(_num(pd.Series([r.get("U")])).iloc[0]) if np.isfinite(_num(pd.Series([r.get("U")])).iloc[0]) else np.nan
            sc = float(_num(pd.Series([r.get("score")])).iloc[0]) if np.isfinite(_num(pd.Series([r.get("score")])).iloc[0]) else np.nan
            status = str(r.get("availability_status", ""))
            if not np.isfinite(u) and "zero_candidates" in status:
                u = 0.0
            cov.append(u); comp.append(sc)
        if np.isfinite(cov).any() or np.isfinite(comp).any():
            series.append((task, label, cov, comp))
    if not series:
        return

    with plt.rc_context({
        "font.size": 9.2, "axes.labelsize": 9.6, "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.0, "legend.fontsize": 8.0, "axes.linewidth": .8,
        "xtick.major.width": .8, "ytick.major.width": .8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, (ax_cov, ax_comp) = plt.subplots(2, 1, figsize=FIGSIZE["main_rq4_traj"], sharex=True,
                                              gridspec_kw={"height_ratios": [1, 1], "hspace": .018})
        ax_cov.set_facecolor("#f5f8fc")
        ax_comp.set_facecolor("#fcf8f3")
        handles = []
        cov_ann = []
        comp_ann = []
        max_cov, max_comp = 0., 0.
        task_colors = {"Grammar": "#1f77b4", "HANS NLI": "#ff7f0e", "Random FSM": "#2ca02c"}
        for _task, label, cov, comp in series:
            color = task_colors.get(_task, "#1f77b4")
            line, = ax_cov.plot(steps, cov, marker="o", linewidth=1.35, markersize=4.2,
                                markeredgewidth=.80, color=color, label=label)
            ax_comp.plot(steps, comp, marker="s", linestyle="--", linewidth=1.25, markersize=4.0,
                         markerfacecolor="white", markeredgewidth=.85, color=color, label=label)
            max_cov = max(max_cov, *[v for v in cov if np.isfinite(v)], 0)
            max_comp = max(max_comp, *[v for v in comp if np.isfinite(v)], 0)
            cov_ann.append((cov, color)); comp_ann.append((comp, color))
            handles.append(Line2D([0], [0], color=color, lw=1.45, marker="o", markersize=4.0, label=label))

        cov_upper = max(.12, min(1.0, max_cov + .12))
        comp_upper = max(.12, min(1.0, max_comp + .10))
        for ax, upper in [(ax_cov, cov_upper), (ax_comp, comp_upper)]:
            ax.axvline(96000, linestyle=":", linewidth=.70, color="0.35", alpha=.75, zorder=1)
            ax.set_xlim(-4200, 147200)
            ax.set_ylim(-.035, upper)
            ax.grid(axis="y", alpha=.34, linewidth=.40)
            ax.tick_params(axis="y", pad=.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax_cov.text(96000, cov_upper * .74, "96k", rotation=90, va="center", ha="right", fontsize=6.9, color="0.20")
        ax_cov.set_ylabel(r"$U(J)$", labelpad=.8)
        ax_comp.set_ylabel("Competence", labelpad=.8)
        ax_comp.set_xlabel("Checkpoint", labelpad=.6)
        ax_comp.set_xticks(steps, step_labels)
        ax_cov.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        fig.subplots_adjust(left=.10, right=.995, bottom=.16, top=.975)
        fig.canvas.draw()
        for cov, color in cov_ann:
            _annotate_checkpoint_values(ax_cov, steps, cov, color, above=True)
        for comp, color in comp_ann:
            _annotate_checkpoint_values(ax_comp, steps, comp, color, above=True)
        ax_cov.legend(handles=handles, frameon=False, loc="upper left", ncol=max(1, len(handles)),
                      handlelength=1.0, columnspacing=.48, borderaxespad=.05, fontsize=8.0)
        save_exact(fig, out)


def _grammar_story(root: Path):
    """Legacy single-seed fallback for older result trees.

    Current result generation writes a cross-seed RQ4 defense figure directly.
    Never choose the first per-seed story when more than one training seed is
    present, because that would silently turn a replicate set into one seed.
    """
    base = root / "analysis/rq4_learning/poisoning/per_run_visualizations/grammar"
    screens = sorted(base.rglob("clean_reference_benign_budget_screen.csv"))
    if len(screens) != 1:
        return None, None, None
    story = screens[0].parent
    screen = read(story / "clean_reference_benign_budget_screen.csv")
    summary = read(story / "clean_reference_benign_budget_checkpoint_summary.csv")
    curve = read(story / "clean_reference_benign_budget_curve.csv")
    return screen, summary, curve


def main_rq4_defense(root: Path, out: Path) -> None:
    cross_root = root / "analysis/rq4_learning/poisoning/cross_seed_tables"
    checkpoint_across = read(cross_root / "clean_reference_defense_checkpoint_across_seeds.csv")
    checkpoint_all = read(cross_root / "clean_reference_defense_checkpoint_all_seeds.csv")
    budget_across = read(cross_root / "clean_reference_defense_budget_curve_across_seeds.csv")
    selected_channels = read(cross_root / "clean_reference_defense_selected_channels_all_seeds.csv")

    if checkpoint_across.empty or checkpoint_all.empty:
        # Fall back to the pre-rendered cross-seed figure only when the
        # summary tables are unavailable.
        cross_seed = root / "paper/figures/05_rq4_learning/rq4_grammar_clean_reference_defense.pdf"
        if cross_seed.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cross_seed, out)
        return

    # Keep the operating point used in the paper.
    checkpoint_across = checkpoint_across.copy()
    checkpoint_all = checkpoint_all.copy()
    budget_across = budget_across.copy()
    checkpoint_across = checkpoint_across[np.isclose(_num(checkpoint_across["operating_benign_damage_budget"]), 0.30, equal_nan=False)].copy()
    if checkpoint_across.empty:
        checkpoint_across = checkpoint_across[np.isclose(_num(checkpoint_across["benign_damage_budget"]), 0.30, equal_nan=False)].copy()
    checkpoint_all = checkpoint_all[np.isclose(_num(checkpoint_all["operating_benign_damage_budget"]), 0.30, equal_nan=False)].copy()
    if checkpoint_all.empty:
        checkpoint_all = checkpoint_all[np.isclose(_num(checkpoint_all["benign_damage_budget"]), 0.30, equal_nan=False)].copy()

    num_cols = [c for c in checkpoint_across.columns if any(k in c for k in ["fraction", "budget", "mean", "median", "q25", "q75", "sd", "ci_"])]
    num_cols += [c for c in checkpoint_all.columns if c not in num_cols and any(k in c for k in ["fraction", "budget", "mean", "median", "q25", "q75", "sd", "ci_"]) ]
    num_cols += [c for c in budget_across.columns if c not in num_cols and any(k in c for k in ["fraction", "budget", "mean", "median", "q25", "q75", "sd", "ci_"]) ]
    for frame in [checkpoint_across, checkpoint_all, budget_across]:
        for c in frame.columns:
            if c in num_cols:
                frame[c] = _num(frame[c])
    if not selected_channels.empty:
        for c in [
            "target_fraction", "benign_damage_budget", "selection_fraction",
            "target_benign_damage_rate", "target_attack_suppression_rate",
        ]:
            if c in selected_channels.columns:
                selected_channels[c] = _num(selected_channels[c])

    checkpoint_across = checkpoint_across.sort_values("target_fraction")
    checkpoint_all = checkpoint_all.sort_values(["target_fraction", "training_seed"])
    budget_across = budget_across.sort_values(["target_fraction", "benign_damage_budget"])

    with plt.rc_context({
        **BASE_RC,
        "font.size": 11.0,
        "axes.labelsize": 11.8,
        "xtick.labelsize": 10.8,
        "ytick.labelsize": 10.8,
        "legend.fontsize": 10.2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }):
        fig, (ax_line, ax_trade) = plt.subplots(1, 2, figsize=(FIGSIZE["main_rq4_defense"][0], FIGSIZE["main_rq4_defense"][1] * 0.94))
        fig.subplots_adjust(left=.075, right=.992, bottom=.16, top=.925, wspace=.28)
        fig.text(.012, .945, "(a)", fontsize=11.4, fontweight="bold", ha="left", va="top")
        fig.text(.515, .945, "(b)", fontsize=11.4, fontweight="bold", ha="left", va="top")

        x = 100 * checkpoint_across["target_fraction"].to_numpy(float)
        attack = 100 * checkpoint_across["mean_attack_suppression__median"].to_numpy(float)
        attack_lo = attack - 100 * checkpoint_across["mean_attack_suppression__q25"].to_numpy(float)
        attack_hi = 100 * checkpoint_across["mean_attack_suppression__q75"].to_numpy(float) - attack
        benign = 100 * checkpoint_across["mean_poisoned_disruption__median"].to_numpy(float)
        benign_lo = benign - 100 * checkpoint_across["mean_poisoned_disruption__q25"].to_numpy(float)
        benign_hi = 100 * checkpoint_across["mean_poisoned_disruption__q75"].to_numpy(float) - benign
        leverage = 100 * checkpoint_across["mean_defense_leverage__median"].to_numpy(float)

        ax_line.fill_between(x, benign, attack, where=attack >= benign, interpolate=True, color="#dcefdc", alpha=.85, linewidth=0, label=r"$\Delta_{\rm def}>0$")
        ax_line.fill_between(x, benign, attack, where=attack < benign, interpolate=True, color="#f4dede", alpha=.85, linewidth=0)
        ax_line.errorbar(x, attack, yerr=np.vstack([attack_lo, attack_hi]), marker="D", markersize=8.2, linewidth=2.35, capsize=3.4, label="Attack suppression")
        ax_line.errorbar(x, benign, yerr=np.vstack([benign_lo, benign_hi]), marker="o", markersize=8.0, linewidth=2.35, capsize=3.4, label="Benign damage")
        for xv, av, bv, lev in zip(x, attack, benign, leverage):
            ax_line.annotate(f"{lev:+.0f} pp", (xv, max(av, bv)), xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=10.0)
        ax_line.set_xticks(x)
        ax_line.set_xlim(max(5, float(np.nanmin(x)) - 3), min(100, float(np.nanmax(x)) + 3))
        ax_line.set_ylim(0, max(float(np.nanmax(attack + attack_hi)), float(np.nanmax(benign + benign_hi))) * 1.10)
        ax_line.set_xlabel("Checkpoint (%)", labelpad=3)
        ax_line.set_ylabel("Rate (%)", labelpad=4)
        ax_line.grid(True, alpha=.18)
        _handles, _labels = ax_line.get_legend_handles_labels()
        _by_label = dict(zip(_labels, _handles))
        _legend_order = ["Attack suppression", "Benign damage", r"$\Delta_{\rm def}>0$"]
        ax_line.legend(
            [_by_label[k] for k in _legend_order if k in _by_label],
            [k for k in _legend_order if k in _by_label],
            frameon=False, loc="upper left", fontsize=9.2,
        )
        ax_line.text(.985, .96, r"$\tau=30\%$", transform=ax_line.transAxes, ha="right", va="top", fontsize=10.0)

        if not budget_across.empty:
            inset = ax_line.inset_axes([.54, .07, .42, .36])
            checkpoints = sorted(budget_across["target_fraction"].dropna().unique())
            markers = ["o", "s", "^", "D", "P", "X"]
            for idx, frac in enumerate(checkpoints):
                cp = budget_across[np.isclose(budget_across["target_fraction"].to_numpy(float), float(frac), equal_nan=False)].sort_values("benign_damage_budget")
                y = 100 * cp["mean_defense_leverage__median"].to_numpy(float)
                q25 = 100 * cp["mean_defense_leverage__q25"].to_numpy(float)
                q75 = 100 * cp["mean_defense_leverage__q75"].to_numpy(float)
                n = _num(cp.get("mean_defense_leverage__n_seeds", pd.Series(index=cp.index, dtype=float))).fillna(0).to_numpy(int)
                y[n <= 0] = np.nan
                inset.plot(100 * cp["benign_damage_budget"].to_numpy(float), y, marker=markers[idx % len(markers)], linewidth=1.15, markersize=3.4)
                inset.fill_between(100 * cp["benign_damage_budget"].to_numpy(float), q25, q75, alpha=.08)
            inset.axhline(0, color="0.45", linestyle="--", linewidth=.7)
            inset.set_xlabel("Budget (%)", fontsize=7.8, labelpad=1.6)
            inset.set_ylabel(r"$\Delta_{\rm def}$ (pp)", fontsize=7.8, labelpad=1.2)
            inset.tick_params(axis="both", labelsize=7.2, pad=1)
            inset.grid(True, alpha=.12)

        # Panel (b): use the exact cross-seed selected-channel table used by
        # poisoning_clean_reference_defense, so the two renderings cannot
        # silently disagree about which channel-level observations are shown.
        channels = selected_channels.copy()
        required = {
            "target_fraction", "unit_key", "benign_damage_budget",
            "target_benign_damage_rate", "target_attack_suppression_rate",
        }
        if not channels.empty and required.issubset(channels.columns):
            if "selection_fraction" in channels.columns:
                channels = channels[np.isclose(
                    channels["selection_fraction"].to_numpy(float),
                    channels["target_fraction"].to_numpy(float),
                    equal_nan=False,
                )]
            channels = channels[np.isclose(
                channels["benign_damage_budget"].to_numpy(float), 0.30,
                atol=1e-12, rtol=0.0, equal_nan=False,
            )]
            channels = channels.dropna(subset=[
                "target_fraction", "unit_key",
                "target_benign_damage_rate", "target_attack_suppression_rate",
            ])
            dedup_cols = [c for c in ["training_seed", "target_fraction", "unit_key"] if c in channels.columns]
            if dedup_cols:
                channels = channels.drop_duplicates(dedup_cols, keep="last")
        else:
            channels = pd.DataFrame()

        if not channels.empty:
            xvals = 100.0 * channels["target_benign_damage_rate"].to_numpy(float)
            yvals = 100.0 * channels["target_attack_suppression_rate"].to_numpy(float)
            lim = 100.0
            trade_x = np.linspace(0, lim, 400)
            ax_trade.fill_between(
                trade_x, trade_x, lim, color="#dcefdc", alpha=.85, linewidth=0, zorder=0
            )
            ax_trade.fill_between(
                trade_x, 0, trade_x, color="#f4dede", alpha=.85, linewidth=0, zorder=0
            )
            ax_trade.plot([0, lim], [0, lim], "--", color="0.42", linewidth=1.0, zorder=1)
            ax_trade.axvline(30, color="0.50", linestyle=":", linewidth=1.0, zorder=1)
            ax_trade.text(.03*lim, .93*lim, "attack-selective channel", fontsize=11.0, fontweight="bold", color="0.18")
            ax_trade.text(.67*lim, .08*lim, "benign-costly channel", fontsize=11.0, fontweight="bold", color="0.18")

            markers = ["o", "s", "^", "D", "P", "X"]
            checkpoints = sorted(channels["target_fraction"].astype(float).unique())
            for idx, frac in enumerate(checkpoints):
                cp = channels[np.isclose(
                    channels["target_fraction"].to_numpy(float), frac, equal_nan=False
                )]
                ax_trade.scatter(
                    100.0 * cp["target_benign_damage_rate"],
                    100.0 * cp["target_attack_suppression_rate"],
                    s=58, marker=markers[idx % len(markers)],
                    alpha=.82, edgecolors="black", linewidths=.35,
                    label=f"{int(round(100*frac))}% checkpoint", zorder=3,
                )

            ax_trade.set_xlim(0, lim)
            ax_trade.set_ylim(0, lim)
            ax_trade.set_xlabel("Benign damage (%)", labelpad=3)
            ax_trade.set_ylabel("Attack suppression (%)", labelpad=4)
            ax_trade.grid(True, alpha=.18)
            ax_trade.legend(
                title="Checkpoint", loc="upper right", ncol=1,
                handletextpad=.45, borderaxespad=.2, labelspacing=.35,
            )
        else:
            ax_trade.text(
                .5, .5, "Channel-level defense data unavailable",
                transform=ax_trade.transAxes, ha="center", va="center",
            )
            ax_trade.set_axis_off()
        save_exact(fig, out)


# ---------------------------------------------------------------------------
# Appendix figures
# ---------------------------------------------------------------------------

def supp_rq1_phase(root: Path, out: Path) -> None:
    d = read(root / "analysis/figure_data/appendix_context/fig_phase_comparison.csv")
    if d.empty:
        return
    labels = list(dict.fromkeys(d["label"].astype(str)))
    labels = [s.replace("Gram.", "gram.") for s in labels]
    # map back after the editorial lower-case display change
    orig_labels = list(dict.fromkeys(d["label"].astype(str)))
    x = np.arange(len(orig_labels))
    with ref_rc(font=8.8, label=9.2, tick=7.8, legend=7.3):
        fig, ax = plt.subplots(figsize=FIGSIZE["supp_rq1_phase"])
        for phase, legend in [("input+output", "Input+output"), ("decode-only", "Output-only")]:
            g = d[d["requested_phase"].astype(str).eq(phase)].set_index("label")
            y = [float(_num(pd.Series([g.loc[z, "union_rate"]])).iloc[0]) if z in g.index else np.nan for z in orig_labels]
            ax.plot(x, y, marker="o", linewidth=1.35, markersize=3.3, label=legend)
        ax.set_ylabel(r"$U(J)$")
        ax.set_xticks(x, labels, rotation=65, ha="right")
        ax.set_ylim(0, .82)
        ax.grid(axis="y", alpha=.22, linewidth=.45)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="lower left")
        fig.subplots_adjust(left=.14, right=.99, bottom=.39, top=.98)
        save_exact(fig, out)


def supp_rq1_size(root: Path, out: Path) -> None:
    d = read(root / "analysis/figure_data/appendix_context/fig_size_comparison.csv")
    if d.empty:
        return
    x = np.arange(len(d))
    comp = _num(d["score"]).to_numpy(float)
    reach = _num(d["union_rate"]).to_numpy(float)
    labels = []
    for _, r in d.iterrows():
        model = str(r.get("model", "")) if pd.notna(r.get("model")) else ""
        task = str(r.get("task", "")) if pd.notna(r.get("task")) else ""
        if not model or not task:
            # Preserve the category identity from the supplied label, not the old
            # figure's accidental literal "nan" tick.
            labels.append(str(r.get("label", "")))
            continue
        m = model.replace("Qwen2-1.5B-Instruct", "Qwen2-1.5B-Instruct").replace("Qwen2-7B-Instruct", "Qwen2-7B-Instruct")
        task_name = {"arithmetic": "Arithmetic", "hans_nli": "HANS-NLI", "bon_jailbreaking": "Jailbreak"}.get(task, task)
        labels.append(f"{task_name}\n{m}")
    with ref_rc(font=8.8, label=9.2, tick=7.2, legend=8.0):
        fig, ax = plt.subplots(figsize=FIGSIZE["supp_rq1_size"])
        width = .36
        ax.bar(x - width/2, comp, width=width, label="Competence")
        ax.bar(x + width/2, reach, width=width, label=r"$U(J)$")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, max(0.98, float(np.nanmax(np.r_[comp, reach])) * 1.04))
        ax.set_xticks(x, labels, rotation=65, ha="right")
        ax.grid(axis="y", alpha=.18, linewidth=.45)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper left")
        fig.subplots_adjust(left=.12, right=.99, bottom=.39, top=.98)
        save_exact(fig, out)


def supp_rq2_decomp(root: Path, out: Path) -> None:
    d = read(root / "analysis/rq2_composition/interaction_decomposition/composition_decomposition_by_condition.csv")
    if d.empty:
        return
    d = d.copy()
    d["Delta_comp_complete_case"] = _num(d["Delta_comp_complete_case"])
    d["suppressed_rate_all"] = _num(d["suppressed_rate_all"])
    d["coalition_only_rate_all"] = _num(d["coalition_only_rate_all"])
    d = d.dropna(subset=["Delta_comp_complete_case", "suppressed_rate_all", "coalition_only_rate_all"]).sort_values("Delta_comp_complete_case").reset_index(drop=True)
    x = np.arange(len(d)); w = .36
    with ref_rc(font=8.2, label=9.0, tick=8.0, legend=7.2):
        fig, ax = plt.subplots(figsize=FIGSIZE["supp_rq2_decomp"])
        ax.bar(x - w/2, d["suppressed_rate_all"], width=w, label="Suppressed singleton effects")
        ax.bar(x + w/2, d["coalition_only_rate_all"], width=w, label="Coalition-only gains")
        ax.set_ylabel("Fraction of examples")
        ax.set_xlabel("Condition, ordered by composition gap")
        ax.set_xticks(np.arange(0, len(d), 5))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.15, linewidth=.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper right")
        fig.subplots_adjust(left=.14, right=.99, bottom=.22, top=.98)
        save_exact(fig, out)


def _display_task(value: object) -> str:
    s = str(value)
    if s.lower() == "grammar":
        return "grammar"
    return s


def supp_rq2_matched(root: Path, out: Path) -> None:
    d = read(root / "analysis/figure_data/03_rq2_composition/fig3c_matched_set_specificity.csv")
    if d.empty:
        return
    d = d.copy()
    d["value"] = _num(d["E_J_Delta"])
    d = d[np.isfinite(d["value"])].sort_values("value", ascending=False).reset_index(drop=True)
    labels = []
    for _, r in d.iterrows():
        task = _display_task(r.get("task", ""))
        model = str(r.get("model", ""))
        phase = str(r.get("phase", ""))
        labels.append(f"{task} | {model} | {phase}")
    y = np.arange(len(d))
    with ref_rc(font=7.6, label=9.0, tick=6.5, legend=7.0):
        fig, ax = plt.subplots(figsize=FIGSIZE["supp_rq2_matched"])
        ax.axvline(0, color="0.5", linewidth=.7)
        for yi, val in zip(y, d["value"].to_numpy(float)):
            ax.plot([0, val], [yi, yi], color="C0", alpha=.45, linewidth=.75)
        ax.scatter(d["value"], y, s=14, color="C0", zorder=3)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Discovered-set effect minus control median")
        ax.grid(axis="y", alpha=.16, linewidth=.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.subplots_adjust(left=.41, right=.99, bottom=.12, top=.99)
        save_exact(fig, out)


def _paired_condition(units: pd.DataFrame, direction: str, metric: str, how: str) -> pd.DataFrame:
    d = units[units["rq3_direction"].astype(str).eq(direction)].copy()
    if d.empty:
        return pd.DataFrame()
    if metric == "threshold_testable":
        d[metric] = d[metric].astype(str).str.lower().map({"true": 1.0, "false": 0.0}).fillna(_num(d[metric]))
    else:
        d[metric] = _num(d[metric])
    d["pop"] = d["population"].map({"random_nonagonist_control": "Control", "flip_rule_candidate": "Candidate"})
    d = d[d["pop"].notna()]
    if how == "mean":
        g = d.groupby(["run_id", "pop"])[metric].mean().unstack("pop")
    else:
        g = d.groupby(["run_id", "pop"])[metric].median().unstack("pop")
    if not {"Control", "Candidate"}.issubset(g.columns):
        return pd.DataFrame()
    return g[["Control", "Candidate"]].dropna(how="all")


def supp_candidate(root: Path, out: Path) -> None:
    u = read(root / "analysis/rq3_threshold_event/spiking_diagnostics/threshold_shape_validation/threshold_shape_unit_population.csv")
    if u.empty:
        return
    specs = [
        ("causal_strength", "Causal strength", "median"),
        ("threshold_testable", "Testable fraction", "mean"),
        ("nested_threshold_mcc", "Held-out |MCC|", "median"),
        ("nested_tecs_lower_bound", "TECS summary", "median"),
    ]
    with ref_rc(font=8.0, label=8.4, tick=7.6, legend=7.0):
        fig, axes = plt.subplots(2, 4, figsize=FIGSIZE["supp_rq3_candidate"])
        for ri, direction in enumerate(["1to0", "0to1"]):
            for ci, (metric, ylab, how) in enumerate(specs):
                ax = axes[ri, ci]
                p = _paired_condition(u, direction, metric, how)
                for _, r in p.dropna().iterrows():
                    ax.plot([0, 1], [r["Control"], r["Candidate"]], color="C0", alpha=.14, linewidth=.6)
                if not p.empty:
                    med = p.median()
                    ax.plot([0, 1], [med["Control"], med["Candidate"]], color="#1f4f85",
                            marker="o", markersize=3.0, linewidth=1.55, zorder=4)
                ax.set_xlim(-.06, 1.06)
                ax.set_xticks([0, 1], ["Control", "Candidate"])
                label = ylab
                if ci == 0:
                    label = ("1→0\n" if ri == 0 else "0→1\n") + ylab
                ax.set_ylabel(label)
                ax.grid(axis="y", alpha=.16, linewidth=.4)
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.subplots_adjust(left=.065, right=.965, bottom=.10, top=.99, wspace=.66, hspace=.34)
        save_exact(fig, out)


def supp_strength_matched(root: Path, out: Path) -> None:
    d = read(root / "analysis/rq3_threshold_event/spiking_diagnostics/threshold_shape_validation/threshold_strength_matched_pairs.csv")
    if d.empty:
        return
    with ref_rc(font=8.0, label=8.7, tick=7.8):
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["supp_rq3_strength_matched"], sharex=True, sharey=True)
        for ax, subset in zip(axes, ["positive", "negative"]):
            g = d[d["baseline_subset"].astype(str).str.lower().eq(subset)].copy()
            x = _num(g["control_nested_mcc"]); y = _num(g["candidate_nested_mcc"])
            ok = np.isfinite(x) & np.isfinite(y)
            x = x[ok].to_numpy(float); y = y[ok].to_numpy(float)
            if len(x):
                ax.scatter(x, y, s=14, alpha=.72)
                lo = min(0., float(x.min()), float(y.min())); hi = max(.1, float(x.max()), float(y.max()))
                ax.plot([lo, hi], [lo, hi], ":", color="0.45", linewidth=.8)
            ax.set_xlabel("Matched-control |MCC|")
            ax.grid(axis="y", alpha=.16, linewidth=.4)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        axes[0].set_ylabel("Candidate |MCC|")
        fig.subplots_adjust(left=.15, right=.99, bottom=.20, top=.98, wspace=.10)
        save_exact(fig, out)


def _ecdf(values):
    v = _num(pd.Series(values)).dropna().to_numpy(float)
    v = np.sort(v[np.isfinite(v)])
    return v, np.arange(1, len(v)+1) / len(v) if len(v) else v


def supp_tecs(root: Path, out: Path) -> None:
    u = read(root / "analysis/rq3_threshold_event/spiking_diagnostics/threshold_shape_validation/threshold_shape_unit_population.csv")
    if u.empty:
        return
    with ref_rc(font=8.0, label=8.7, tick=7.8, legend=7.0):
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["supp_rq3_tecs"], sharey=True)
        for ax, direction in zip(axes, ["1to0", "0to1"]):
            g = u[u["rq3_direction"].astype(str).eq(direction)]
            for pop, label in [("flip_rule_candidate", "Candidates"), ("random_nonagonist_control", "Controls")]:
                x, y = _ecdf(g.loc[g["population"].astype(str).eq(pop), "nested_tecs_lower_bound"])
                if len(x):
                    ax.plot(x, y, linewidth=1.3, label=label)
            ax.set_xlabel("TECS summary")
            ax.grid(axis="y", alpha=.16, linewidth=.4)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        axes[0].set_ylabel("Empirical CDF")
        axes[0].legend(frameon=False, loc="lower right")
        fig.subplots_adjust(left=.14, right=.99, bottom=.22, top=.98, wspace=.10)
        save_exact(fig, out)


def supp_event(root: Path, out: Path) -> None:
    src = root / "paper/figures/04_rq3_spiking_cut/fig4s10_population_event_localization.pdf"
    _copy(src, out)


def supp_pairs(root: Path, out: Path) -> None:
    d = read(root / "analysis/rq3_threshold_event/spiking_diagnostics/graded_agonist/strength_concentration_pairs.csv")
    if d.empty:
        return
    x = _num(d["low_concentration"]); y = _num(d["high_concentration"])
    ok = np.isfinite(x) & np.isfinite(y); x = x[ok].to_numpy(float); y = y[ok].to_numpy(float)
    if not len(x):
        return
    with ref_rc(font=8.2, label=8.8, tick=7.8):
        fig, ax = plt.subplots(figsize=FIGSIZE["supp_rq3_pair"])
        ax.scatter(x, y, s=17, alpha=.78)
        lo = min(float(x.min()), float(y.min())); hi = max(float(x.max()), float(y.max()))
        ax.plot([lo, hi], [lo, hi], ":", color="0.45", linewidth=.85)
        ax.set_xlabel("Low-strength concentration")
        ax.set_ylabel("High-strength concentration")
        ax.grid(axis="y", alpha=.16, linewidth=.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.subplots_adjust(left=.18, right=.99, bottom=.16, top=.98)
        save_exact(fig, out)


def supp_arithmetic(root: Path, out: Path) -> None:
    src = root / "paper/figures/04_rq3_spiking_cut/fig4s12_arithmetic_competence_concentration.pdf"
    _copy(src, out)


def supp_transient(root: Path, out: Path) -> None:
    d = read(root / "analysis/rq3_threshold_event/spiking_diagnostics/graded_agonist/endpoint_preserving_transient_events.csv")
    if d.empty:
        return
    d = d.set_index("group").reindex(["Overall", "1→0", "0→1"]).dropna(how="all").reset_index()
    y = np.arange(len(d))
    rate = _num(d["rate_percent"]).to_numpy(float)
    lo = _num(d["ci_low_percent"]).to_numpy(float)
    hi = _num(d["ci_high_percent"]).to_numpy(float)
    with ref_rc(font=8.2, label=8.8, tick=7.8):
        fig, ax = plt.subplots(figsize=FIGSIZE["supp_rq3_transient"])
        ax.errorbar(rate, y, xerr=np.vstack([rate-lo, hi-rate]), fmt="o", markersize=3.0,
                    capsize=2.4, linewidth=.9)
        ax.axvline(0, linestyle=":", color="0.45", linewidth=.8)
        ax.set_yticks(y, d["group"])
        ax.invert_yaxis()
        ax.set_xlabel("Interior flip-and-return rate (%)")
        ax.grid(axis="y", alpha=.16, linewidth=.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.subplots_adjust(left=.16, right=.99, bottom=.23, top=.97)
        save_exact(fig, out)


def supp_preemption(root: Path, out: Path) -> None:
    d = read(root / "analysis/rq3_threshold_event/spiking_diagnostics/preemption/preemption_by_condition.csv")
    if d.empty:
        return
    allv = pd.concat([_num(d["median_delta_absent"]), _num(d["median_delta_present"])]).dropna()
    if allv.empty:
        return
    lo = float(allv.min()); hi = float(allv.max()); pad = .05 * max(hi-lo, .1); lo -= pad; hi += pad
    with ref_rc(font=8.0, label=8.5, tick=7.6):
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["supp_rq3_preemption"], sharex=True, sharey=True)
        for ax, direction in zip(axes, ["c2i", "i2c"]):
            g = d[d["direction"].astype(str).eq(direction)]
            x = _num(g["median_delta_absent"]); y = _num(g["median_delta_present"])
            ok = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[ok], y[ok], s=15, alpha=.8)
            ax.plot([lo, hi], [lo, hi], ":", color="0.45", linewidth=.85)
            ax.set_xlabel("Secondary effect: dominant absent")
            ax.grid(axis="y", alpha=.16, linewidth=.4)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        axes[0].set_ylabel("Secondary effect: dominant present")
        axes[0].set_xlim(lo, hi); axes[0].set_ylim(lo, hi)
        fig.subplots_adjust(left=.17, right=.99, bottom=.22, top=.98, wspace=.10)
        save_exact(fig, out)


def supp_temporal(root: Path, out: Path) -> None:
    d = read(root / "analysis/rq3_threshold_event/spiking_diagnostics/temporal_cutoff/temporal_cutoff_condition_curves.csv")
    if d.empty:
        return
    d = d.copy()
    d["active_decode_steps"] = _num(d["active_decode_steps"])
    d["normalized_capture"] = _num(d["normalized_capture"])
    tm = d.groupby(["run_id", "direction"])["active_decode_steps"].max().rename("T").reset_index()
    d = d.merge(tm, on=["run_id", "direction"])
    Ts = sorted(int(x) for x in d["T"].dropna().unique())
    with ref_rc(font=7.8, label=8.2, tick=7.5):
        fig, axes = plt.subplots(1, len(Ts), figsize=FIGSIZE["supp_rq3_temporal"], sharey=True)
        axes = np.atleast_1d(axes)
        for ax, T in zip(axes, Ts):
            g = d[d["T"].eq(T)]
            for _, c in g.groupby(["run_id", "direction"]):
                c = c.sort_values("active_decode_steps")
                ax.plot(c["active_decode_steps"], c["normalized_capture"], color="C0", alpha=.13, linewidth=.55)
            med = g.groupby("active_decode_steps")["normalized_capture"].median().sort_index()
            ax.plot(med.index, med.values, color="#1f4f85", marker="o", markersize=2.5, linewidth=1.0)
            ticks = {2: [0,1,2], 3: [0,1,3], 5: [0,2,5], 9: [0,4,9]}.get(T, sorted(set([0, max(1, T//2), T])))
            ax.set_xticks(ticks)
            ax.set_xlabel(f"Decode transitions ($T = {T}$)")
            ax.set_ylim(-.02, 1.05)
            ax.grid(axis="y", alpha=.16, linewidth=.4)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        axes[0].set_ylabel("Normalized capture")
        fig.subplots_adjust(left=.075, right=.995, bottom=.25, top=.98, wspace=.15)
        save_exact(fig, out)


def _cross_profile(d: pd.DataFrame, direction: str | None) -> pd.DataFrame:
    g = d if direction is None else d[d["direction"].astype(str).eq(direction)]
    rows = []
    for off, h in g.groupby("event_offset"):
        v = _num(h["incremental_gain"]).dropna().to_numpy(float)
        if len(v):
            rows.append((int(off), float(np.median(v)), float(np.quantile(v, .25)), float(np.quantile(v, .75))))
    return pd.DataFrame(rows, columns=["offset", "median", "q25", "q75"]).sort_values("offset") if rows else pd.DataFrame()


def supp_cross(root: Path, out: Path) -> None:
    base = root / "analysis/rq3_threshold_event/spiking_diagnostics/temporal_cutoff"
    suffix_event = read(base / "temporal_cross_sweep_suffix_event_prefix_aligned.csv")
    prefix_event = read(base / "temporal_cross_sweep_prefix_event_suffix_aligned.csv")
    summary_path = base / "temporal_cutoff_summary.json"
    if suffix_event.empty or prefix_event.empty or not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text())
    row_specs = [
        (suffix_event, summary.get("cross_sweep_suffix_event_prefix_validation", {}), "Prefix-sweep effect"),
        (prefix_event, summary.get("cross_sweep_prefix_event_suffix_validation", {}), "Suffix-sweep effect"),
    ]
    col_specs = [("1to0", "1→0", "1to0"), ("0to1", "0→1", "0to1"), (None, "Pooled", "global")]

    with ref_rc(font=7.6, label=8.0, tick=7.2):
        fig, axes = plt.subplots(2, 3, figsize=FIGSIZE["supp_rq3_cross"], sharex=True)
        for ri, (data, stats, row_label) in enumerate(row_specs):
            for ci, (direction, direction_label, key) in enumerate(col_specs):
                ax = axes[ri, ci]
                p = _cross_profile(data, direction)
                if not p.empty:
                    for xi, q25, q75 in zip(p["offset"], p["q25"], p["q75"]):
                        ax.vlines(xi, q25, q75, linewidth=1.35, alpha=.55)
                    ax.plot(p["offset"], p["median"], marker="o", markersize=2.5, linewidth=1.0)
                ax.axvline(0, linestyle=":", color="0.45", linewidth=.8)
                ax.axhline(0, linestyle=":", color="0.55", linewidth=.7)
                ax.set_xticks([-4, -2, 0, 2, 4], ["−4", "−2", "EVENT", "+2", "+4"])
                if ci == 0:
                    ax.set_ylabel(row_label)
                if ri == 1:
                    # Direction is shown as part of the axis label rather than a
                    # subplot title, matching the title-free appendix convention.
                    ax.set_xlabel(f"{direction_label}\nTransition relative to EVENT")
                st = stats.get(key, {}) if isinstance(stats, dict) else {}
                ci95 = st.get("median_event_minus_adjacent_gain_ci95", [math.nan, math.nan])
                val = st.get("median_event_minus_adjacent_gain", math.nan)
                pval = st.get("event_minus_adjacent_wilcoxon_one_sided_p", math.nan)
                if np.isfinite(float(val)):
                    ax.text(.97, .96,
                            f"EVENT−adj. {float(val):+.2f}\n95% CI [{float(ci95[0]):+.2f}, {float(ci95[1]):+.2f}]\n{_p_plain(pval)}",
                            transform=ax.transAxes, ha="right", va="top", fontsize=5.8)
                ax.grid(axis="y", alpha=.16, linewidth=.4)
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.subplots_adjust(left=.09, right=.995, bottom=.18, top=.99, wspace=.34, hspace=.30)
        save_exact(fig, out)



def main_overview(_root: Path, out: Path) -> None:
    """Generate the paper overview figure directly, without cached reference PDFs."""
    import matplotlib as mpl
    from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch

    W, H = 1490, 348
    BLUE = '#1256F4'; RED = '#E92B3A'; GRAY = '#7B848C'; MID = '#A8ADB2'
    LIGHT = '#ECEFF2'; PALE_BLUE = '#EAF2FF'; PALE_RED = '#FDECEE'
    TEXT = '#22262A'; SUB = '#555D66'; WHITE = '#FFFFFF'
    with plt.rc_context({"pdf.fonttype":42, "ps.fonttype":42, "font.family":"DejaVu Sans", "mathtext.fontset":"dejavusans"}):
        fig = plt.figure(figsize=(13.2, 3.09), dpi=120, facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

        def txt(x, y, s, size=10.5, weight='normal', color=TEXT, ha='left', va='center', z=6):
            return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z)
        def box(x, y, w, h, ec='#4D5358', fc=WHITE, lw=.9, r=3.5, z=2):
            p = FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',edgecolor=ec,facecolor=fc,linewidth=lw,zorder=z); ax.add_patch(p); return p
        def arrow(x1,y1,x2,y2,color=TEXT,lw=1.1,ms=8.5,z=5):
            p=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=ms,color=color,linewidth=lw,zorder=z); ax.add_patch(p); return p
        def network(cx,cy,scale=.50):
            pts=np.array([[-32,-12],[-16,-32],[0,-42],[18,-28],[32,-9],[0,-10],[-24,10],[-3,10],[22,8],[-16,29],[3,38],[22,26]],float)*scale
            pts[:,0]+=cx; pts[:,1]+=cy
            edges=[(0,1),(1,2),(2,3),(3,4),(4,8),(8,11),(11,10),(10,9),(9,6),(6,0),(0,5),(1,5),(2,5),(3,5),(4,5),(5,7),(5,8),(6,7),(7,8),(7,9),(7,10),(8,10)]
            for i,j in edges: ax.add_line(Line2D([pts[i,0],pts[j,0]],[pts[i,1],pts[j,1]],color=TEXT,lw=.9,zorder=3))
            for x,y in pts: ax.add_patch(Circle((x,y),4.5*scale,ec=TEXT,fc='#EEF6FA',lw=.9,zorder=4))
        def small_axes(x,y,w,h): return fig.add_axes([x/W,(H-y-h)/H,w/W,h/H])

        for x in [258,662,1012]: ax.add_line(Line2D([x,x],[15,H-14],color='#D5DADF',lw=.8,zorder=1))
        SUBTITLE=9.2
        L,R=10,246; cx=(L+R)/2
        txt(L+2,20,'1. Setup',12.6,'bold'); txt(cx,42,'Frozen model, binary endpoint',SUBTITLE,color=SUB,ha='center')
        txt(cx,61,'Prompt $x$',9.4,'bold',ha='center'); box(cx-54,72,108,38,fc='#F7F7F7',ec='#C4C8CC'); txt(cx,91,r'$23+12=?$',12.2,ha='center')
        arrow(cx,112,cx,126,color=GRAY); box(cx-82,130,164,92,fc='#EAF4FA',ec='#B8C6D1'); txt(cx,147,'Frozen Transformer',9.7,'bold',ha='center'); txt(cx,166,r'$M(x)$',12.2,ha='center'); network(cx,194,.48); arrow(cx,224,cx,239,color=GRAY)
        ocx=cx-14; txt(ocx,249,'Observed outcomes',9.4,'bold',ha='center'); txt(ocx-42,266,'numeric',8.4,color=SUB,ha='center'); txt(ocx+61,266,'binary',8.4,color=SUB,ha='center')
        for yy,val,binary,col in [(287,'43','1 correct',BLUE),(320,'34','0 incorrect',RED)]:
            box(ocx-72,yy-14,54,28,ec=col,lw=1.25); txt(ocx-45,yy,val,13.0,'bold',col,ha='center'); arrow(ocx-12,yy,ocx+8,yy,lw=.95,ms=7); box(ocx+11,yy-14,105,28,ec=col,lw=1.25); txt(ocx+63.5,yy,binary,10.0,'bold',col,ha='center')

        L,R=278,646; cx=(L+R)/2
        txt(L+2,20,'2. Intervention',12.6,'bold'); txt(cx,42,'Replace one channel and re-evaluate',SUBTITLE,color=SUB,ha='center')
        for xx,fc,label in [(L+18,BLUE,'intact'),(L+96,GRAY,'replacement'),(L+206,WHITE,'untargeted')]:
            ax.add_patch(Rectangle((xx,61),14,14,ec=TEXT if fc==WHITE else fc,fc=fc,lw=.85,zorder=3)); txt(xx+19,68,label,8.7,color=SUB)
        box(L+5,94,92,35,ec='none',fc=PALE_BLUE,lw=0); txt(L+51,111,'Original',10.0,'bold',ha='center')
        xs=[L+126,L+154,L+182,L+210,L+238,L+266]; labs=['1','2','…',r'$j$','…','N']
        for x,lab in zip(xs,labs): txt(x,91,lab,9.0,ha='center')
        for x,lab in zip(xs,labs):
            if lab=='…': continue
            fc=BLUE if lab==r'$j$' else WHITE; ec=BLUE if lab==r'$j$' else TEXT; ax.add_patch(Rectangle((x-7,101),14,20,ec=ec,fc=fc,lw=.85,zorder=3))
        arrow(L+281,111,L+300,111,ms=7.5); txt(L+306,105,'43',12.4,'bold',BLUE); txt(L+306,118,'correct',8.5,color=BLUE)
        box(L+5,151,92,35,ec='none',fc=PALE_RED,lw=0); txt(L+51,168,'Replace $j$',10.0,'bold',ha='center')
        for x,lab in zip(xs,labs): txt(x,148,lab,9.0,ha='center')
        for x,lab in zip(xs,labs):
            if lab=='…': continue
            fc=GRAY if lab==r'$j$' else WHITE; ax.add_patch(Rectangle((x-7,158),14,20,ec=TEXT,fc=fc,lw=.85,zorder=3))
        xj=L+210; ax.add_line(Line2D([xj-6,xj+6],[150,162],color=RED,lw=1.8,zorder=5)); ax.add_line(Line2D([xj+6,xj-6],[150,162],color=RED,lw=1.8,zorder=5)); arrow(L+281,168,L+300,168,ms=7.5); txt(L+306,162,'34',12.4,'bold',RED); txt(L+306,175,'incorrect',8.5,color=RED); arrow(xj,180,xj,203,color=GRAY,lw=1.0,ms=7); txt(cx,208,'replacement value',8.7,color=SUB,ha='center')
        box(L+38,225,118,58,ec='#8A9096',fc='#FBFBFB'); txt(L+59,254,r'$\mu$',24,ha='center'); txt(L+109,242,'Mean',10.0,'bold',ha='center'); txt(L+109,258,'reference',8.0,ha='center')
        box(L+188,225,130,58,ec='#8A9096',fc='#FBFBFB')
        for px,py in [(L+210,248),(L+225,237),(L+240,249),(L+216,264),(L+237,264)]: ax.add_patch(Circle((px,py),3.5,ec=TEXT,fc=TEXT,lw=.7,zorder=4))
        for a,b in [((L+210,248),(L+225,237)),((L+225,237),(L+240,249)),((L+210,248),(L+216,264)),((L+240,249),(L+237,264)),((L+216,264),(L+237,264))]: ax.add_line(Line2D([a[0],b[0]],[a[1],b[1]],color=TEXT,lw=.8,zorder=3))
        txt(L+282,242,'Donor',10.0,'bold',ha='center'); txt(L+282,258,'example',8.0,ha='center'); txt(cx,310,'Singleton replacement measures causal leverage',8.6,color=SUB,ha='center')

        L,R=688,996; cx=(L+R)/2
        txt(L+2,20,'3. Overtopping vs. diffuse',12.5,'bold'); txt(cx,42,'Concentrated vs. distributed support',SUBTITLE,color=SUB,ha='center')
        box(L+4,61,142,23,ec='none',fc=PALE_BLUE,lw=0); txt(L+75,73,'OVERTOPPING',10.8,'bold',BLUE,ha='center'); box(L+158,61,142,23,ec='none',fc=PALE_RED,lw=0); txt(L+229,73,'DIFFUSE',10.8,'bold',RED,ha='center')
        txt(L+12,101,r'$U(J)\approx \mathrm{str}(j^*)$',10.5,color=BLUE); txt(L+162,101,r'$U(J)>\max_j \mathrm{str}(j)$',10.0,color=RED)
        p1=small_axes(L+20,135,118,165); v=[.12,.10,.09,1.0,.10,.12,.09,.11]; c=[MID]*len(v); c[3]=BLUE; p1.bar(range(len(v)),v,color=c,width=.74); p1.set_ylim(0,1.08); p1.set_yticks([0,.5,1]); p1.set_xticks([0,3,7],['1',r'$j^*$','N']); p1.set_xlabel('Channel $j$',fontsize=7.8); p1.set_ylabel('Effect',fontsize=7.8); p1.tick_params(labelsize=7); p1.spines[['top','right']].set_visible(False)
        p2=small_axes(L+170,135,118,165); v=[.36,.28,.33,.40,.31,.39,.43,.49,.38,.60]; cols=[MID,MID,MID]+['#F6B2BA','#F29AA6','#F48896','#F27687','#ED5E72','#F27687',RED]; p2.bar(range(len(v)),v,color=cols,width=.74); p2.set_ylim(0,1.08); p2.set_yticks([0,.5,1]); p2.set_xticks([0,4,9],['1',r'$j$','N']); p2.set_xlabel('Channel $j$',fontsize=7.8); p2.set_ylabel('Effect',fontsize=7.8); p2.tick_params(labelsize=7); p2.spines[['top','right']].set_visible(False)

        L,R=1037,1475; cx=(L+R)/2
        txt(L+2,20,'4. Four measurements',12.5,'bold'); txt(cx,42,'Four signatures characterize the regime',SUBTITLE,color=SUB,ha='center')
        labels=[('Reach',BLUE,PALE_BLUE,L+2,61),('Composition',TEXT,LIGHT,L+220,61),('Threshold',RED,PALE_RED,L+2,194),('Learning',TEXT,LIGHT,L+220,194)]
        for lab,col,fc,x,y in labels: box(x,y,208,23,ec='none',fc=fc,lw=0); txt(x+104,y+12,lab,10.5,'bold',col,ha='center')
        q1=small_axes(L+42,92,128,76); vals=[.90,.56,.28,.14,.08]; q1.bar(range(5),vals,color=['#316EE8','#5B89E9','#88A9EE','#AEC5F3','#D3DFF7']); q1.set_ylim(0,1.05); q1.set_xticks([0,2,4],['1','…','N']); q1.set_yticks([0,.5,1]); q1.tick_params(labelsize=6.8); q1.set_ylabel('Flip rate',fontsize=7.3); q1.spines[['top','right']].set_visible(False)
        q2=small_axes(L+260,92,128,76); stats=[dict(med=.37,q1=.25,q3=.52,whislo=.08,whishi=.78,fliers=[]),dict(med=.65,q1=.50,q3=.82,whislo=.27,whishi=1.0,fliers=[])]; bp=q2.bxp(stats,positions=[1,2],widths=.5,showfliers=False,patch_artist=True); bp['boxes'][0].set(facecolor='#DDE8FF',edgecolor=BLUE); bp['boxes'][1].set(facecolor='#FDE3E6',edgecolor=RED); q2.set_ylim(0,1.05); q2.set_xticks([1,2],['Single','Joint']); q2.set_yticks([0,.5,1]); q2.tick_params(labelsize=6.8); q2.set_ylabel('Effect',fontsize=7.3); q2.spines[['top','right']].set_visible(False)
        q3=small_axes(L+42,225,128,76); xx=np.logspace(-3,1,10); yy=np.array([.02,.03,.04,.08,.18,.48,.79,.93,.98,1]); q3.plot(xx,yy,'-o',color=RED,lw=1.4,ms=3); q3.axvline(1e-1,color=MID,ls='--',lw=.8); q3.set_xscale('log'); q3.set_ylim(0,1.05); q3.set_yticks([0,.5,1]); q3.tick_params(labelsize=6.2); q3.set_ylabel('Flip fraction',fontsize=7.0); q3.set_xlabel(r'Dose $\lambda$',fontsize=7.0); q3.spines[['top','right']].set_visible(False)
        q4=small_axes(L+260,225,128,76); xx=np.array([1,2,3,4,5,6]); yy=np.array([.15,.30,.43,.55,.66,.74]); cc=np.full(6,.11); q4.plot(xx,yy,'-o',color=RED,lw=1.4,ms=3); q4.plot(xx,cc,'-o',color=GRAY,lw=1.0,ms=2.7); q4.set_ylim(0,1.05); q4.set_xticks([1,3,6],['early','mid','late']); q4.set_yticks([0,.5,1]); q4.tick_params(labelsize=6.2); q4.set_ylabel('Flip rate',fontsize=7.0); q4.set_xlabel('Training',fontsize=7.0); q4.spines[['top','right']].set_visible(False)
        out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out,format='pdf',bbox_inches='tight',pad_inches=.015,facecolor='white'); plt.close(fig)


def _copy_tables(root: Path, tables_dir: Path) -> int:
    source = root / "paper" / "tables"
    if not source.is_dir():
        return 0
    tables_dir.mkdir(parents=True, exist_ok=True)
    n=0
    for src in sorted(source.glob("*.tex")):
        shutil.copy2(src, tables_dir / src.name)
        n+=1
    return n


def _build_manuscript_bundle(root: Path, main_dir: Path, supp_dir: Path, tables_dir: Path) -> dict[str,int]:
    """Create the upload bundle while preserving each paper figure family's style."""
    main_dir.mkdir(parents=True, exist_ok=True)
    supp_dir.mkdir(parents=True, exist_ok=True)
    paper = root / "paper" / "figures"
    counts={"copied_main":0,"generated_main":0,"copied_supp":0,"generated_supp":0}

    # Generate figures whose manuscript style differs from the standard pipeline
    # output. Copy only the few paper figures already emitted in the exact same
    # visual form by their owning reporting scripts.
    main_jobs=[
        (main_overview,"overview_overtopping_framework.pdf"),
        (main_rq1,"rq1_competence_causal_reach.pdf"),
        (main_rq2,"rq2_composition_boxplots.pdf"),
        (main_rq4_defense,"rq4_grammar_clean_reference_defense.pdf"),
    ]
    for fn,name in main_jobs:
        fn(root,main_dir/name)
        if (main_dir/name).is_file(): counts["generated_main"]+=1
    main_rq3(root, main_dir/"rq3_population_event_and_strength.pdf")
    if (main_dir/"rq3_population_event_and_strength.pdf").is_file():
        counts["generated_main"]+=1
    if _copy(paper/"05_rq4_learning"/"fig5a_pythia_checkpoint_trajectory.pdf", main_dir/"rq4_pythia_checkpoint_trajectory.pdf"):
        counts["copied_main"]+=1
    else:
        raise FileNotFoundError("Expected RQ4 trajectory from results/paper/figures/05_rq4_learning/fig5a_pythia_checkpoint_trajectory.pdf")

    # These two RQ1 context figures are already produced by the appendix-context
    # script in the same layout used by the paper. Do not redraw them.
    for src,name in [
        (paper/"appendix_context"/"fig_phase_comparison.pdf","rq1_phase_comparison.pdf"),
        (paper/"appendix_context"/"fig_size_comparison.pdf","rq1_model_size_comparison.pdf"),
    ]:
        if _copy(src,supp_dir/name): counts["copied_supp"]+=1

    supp_jobs=[
        (supp_rq2_decomp,"rq2_singleton_joint_decomposition.pdf"),
        (supp_rq2_matched,"rq2_matched_set_specificity.pdf"),
        (supp_candidate,"rq3_candidate_control_diagnostics.pdf"),
        (supp_strength_matched,"rq3_strength_matched_thresholdability.pdf"),
        (supp_tecs,"rq3_threshold_event_strength_ecdf.pdf"),
        (supp_pairs,"rq3_strength_concentration_paired.pdf"),
        (supp_transient,"rq3_affine_null_transient_events.pdf"),
        (supp_preemption,"rq3_preemption_diagnostic.pdf"),
        (supp_temporal,"rq3_temporal_duration.pdf"),
        (supp_cross,"rq3_temporal_cross_sweep_validation.pdf"),
    ]
    for fn,name in supp_jobs:
        fn(root,supp_dir/name)
        if (supp_dir/name).is_file(): counts["generated_supp"]+=1

    # These are already generated in the exact manuscript family style by the
    # RQ3 reporting script, but with current result values.
    for src,name in [
        (paper/"04_rq3_spiking_cut"/"fig4s10_population_event_localization.pdf","rq3_population_event_localization.pdf"),
        (paper/"04_rq3_spiking_cut"/"fig4s12_arithmetic_competence_concentration.pdf","rq3_arithmetic_competence_concentration.pdf"),
    ]:
        if _copy(src,supp_dir/name): counts["copied_supp"]+=1

    # Rebuild the paper-facing table sources from the same canonical analysis
    # CSVs before copying them into the manuscript bundle. This keeps generated
    # tables byte-for-byte synchronized with the current paper table layouts.
    generate_manuscript_tables(root)
    counts["tables"]=_copy_tables(root,tables_dir)
    return counts



def main() -> None:
    args = parse_args()
    root = Path(args.results_root).resolve()
    bundle = root / "manuscript_exact"
    main_dir = Path(args.main_dir).resolve() if args.main_dir else bundle / "figures" / "main"
    supp_dir = Path(args.supp_dir).resolve() if args.supp_dir else bundle / "figures" / "supplement"
    tables_dir = Path(args.tables_dir).resolve() if args.tables_dir else bundle / "tables"
    counts = _build_manuscript_bundle(root, main_dir, supp_dir, tables_dir)
    print(
        f"[manuscript-figures] main={main_dir} supplement={supp_dir} tables={tables_dir} "
        f"generated_main={counts['generated_main']} copied_main={counts['copied_main']} "
        f"generated_supp={counts['generated_supp']} copied_supp={counts['copied_supp']} "
        f"tables={counts['tables']}"
    )


if __name__ == "__main__":
    main()
