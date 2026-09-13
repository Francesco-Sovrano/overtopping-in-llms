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
import numpy as np
import pandas as pd


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
    p.add_argument("--main-dir", required=True)
    p.add_argument("--supp-dir", required=True)
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
            return "P48"
        if "96000" in low or "96k" in low:
            return "P96"
        if "step0" in low or "@0" in low:
            return "P0"
        return "P1"
    return s.replace("-Instruct", "")


# ---------------------------------------------------------------------------
# Main figures
# ---------------------------------------------------------------------------

def main_rq1(root: Path, out: Path) -> None:
    """2x2 competence/reach panel matching the attached paper."""
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
    phases = ["input+output", "decode-only"]

    with plt.rc_context({
        "font.size": 10.5,
        "axes.labelsize": 13.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 8.6,
        "axes.linewidth": .8,
        "xtick.major.width": .8,
        "ytick.major.width": .8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(7.68, 4.42), sharex="col", sharey="row")
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = {t: cycle[i] for i, t in enumerate(task_order)}

        for ri, direction in enumerate(["0to1", "1to0"]):
            d = frames[direction].copy()
            d["score"] = _num(d["score"])
            d["union_rate"] = _num(d["union_rate"])
            for ci, phase in enumerate(phases):
                ax = axes[ri, ci]
                g = d[d["phase"].astype(str).eq(phase)].copy()
                for _, r in g.iterrows():
                    task = str(r.get("task", ""))
                    if task not in colors or not np.isfinite(r["score"]) or not np.isfinite(r["union_rate"]):
                        continue
                    marker = "o" if str(r.get("baseline", "mean-donor")) == "mean-donor" else "s"
                    ax.scatter(
                        float(r["score"]), float(r["union_rate"]), s=55, marker=marker,
                        color=colors[task], edgecolor="black", linewidth=.65, zorder=4,
                    )
                    # The reference panel labels the plotted model at every point.
                    label = _model_tag(r.get("model", ""))
                    xoff = 4
                    yoff = 4
                    if float(r["union_rate"]) > .93:
                        yoff = -11
                    if float(r["score"]) > .88:
                        xoff = -24
                    ax.annotate(label, (float(r["score"]), float(r["union_rate"])),
                                xytext=(xoff, yoff), textcoords="offset points",
                                fontsize=7.7, ha="left", va="bottom", color="0.10", zorder=5)

                fit = g[["score", "union_rate"]].dropna()
                if len(fit) >= 3 and fit["score"].nunique() > 1:
                    coef = np.polyfit(fit["score"].to_numpy(float), fit["union_rate"].to_numpy(float), 1)
                    xx = np.linspace(float(fit["score"].min()), float(fit["score"].max()), 100)
                    ax.plot(xx, coef[0] * xx + coef[1], "--", color="0.32", linewidth=1.45, zorder=2)

                sr = stats[direction]
                sr = sr[sr["phase"].astype(str).eq(phase)] if not sr.empty else sr
                if not sr.empty:
                    q = sr.iloc[0]
                    txt = f"$n={int(q['n'])},\\ r={float(q['pearson_r']):.2f}$\n$p={float(q['pearson_p']):.5f}$"
                    # Match the reference placement: the dense top-left panel uses
                    # the lower-right corner, the top-right uses upper-right.
                    if ri == 0 and ci == 1:
                        xy, va = (.98, .97), "top"
                    else:
                        xy, va = (.98, .07), "bottom"
                    ax.text(*xy, txt, transform=ax.transAxes, ha="right", va=va, fontsize=8.2,
                            bbox={"boxstyle": "round,pad=.20", "facecolor": "white",
                                  "edgecolor": "0.72", "linewidth": .7, "alpha": .94}, zorder=8)

                ax.set_xlim(-.01, 1.02)
                ax.set_ylim(-.02, 1.03)
                ax.grid(True, alpha=.20, linewidth=.45)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

        axes[0, 0].set_ylabel(r"$0\!\to\!1$ reach $U^d(J)$")
        axes[1, 0].set_ylabel(r"$1\!\to\!0$ reach $U^d(J)$")
        axes[1, 0].set_xlabel("Raw task score")
        axes[1, 1].set_xlabel(r"Chance-normalized score $\kappa$")

        handles = [
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=colors[t],
                   markeredgecolor="black", markeredgewidth=.55, markersize=5.5, label=task_labels[t])
            for t in task_order
        ]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .005), ncol=5,
                   frameon=False, handletextpad=.35, columnspacing=.9)
        fig.subplots_adjust(left=.095, right=.995, bottom=.18, top=.995, hspace=.11, wspace=.12)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", pad_inches=.02)
        plt.close(fig)


def main_rq2(root: Path, out: Path) -> None:
    d = read(root / "analysis/rq2_composition/interaction_decomposition/composition_decomposition_all_scopes.csv")
    if d.empty:
        return
    d["Delta_comp_complete_case"] = _num(d["Delta_comp_complete_case"])
    d = d[np.isfinite(d["Delta_comp_complete_case"])].copy()
    order = ["0to1", "1to0", "overall"]
    labels = [r"$0\rightarrow1$", r"$1\rightarrow0$", "All directions"]
    colors = ["C0", "C1", "0.50"]
    vals = [d.loc[d["scope"].astype(str).eq(scope), "Delta_comp_complete_case"].to_numpy(float) for scope in order]

    with ref_rc(font=9, label=9, tick=9, legend=8):
        fig, ax = plt.subplots(figsize=FIGSIZE["main_rq2"])
        bp = ax.boxplot(vals, positions=[1, 2, 3], widths=.46, showfliers=False, patch_artist=True,
                        medianprops={"linewidth": 1.25}, whiskerprops={"linewidth": 1.0},
                        capprops={"linewidth": 1.0})
        rng = np.random.default_rng(0)
        for i, color in enumerate(colors):
            bp["boxes"][i].set(facecolor="white", edgecolor=color, linewidth=1.05)
            bp["medians"][i].set(color=color, linewidth=1.25)
            for art in bp["whiskers"][2*i:2*i+2] + bp["caps"][2*i:2*i+2]:
                art.set(color=color, linewidth=1.0)
            v = vals[i]
            jitter = rng.uniform(-.10, .10, size=len(v)) if len(v) else np.array([])
            ax.scatter((i + 1) + jitter, v, s=9, color=color, alpha=.58, linewidths=0, zorder=3)
        ax.axhline(0, linestyle="--", color="0.45", linewidth=.75)
        ax.set_xticks([1, 2, 3], labels)
        ax.set_xlabel("Intervention direction")
        ax.set_ylabel("Composition gap\n$E(J)-U(J)$")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(axis="y", alpha=.13, linewidth=.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.subplots_adjust(left=.15, right=.995, bottom=.25, top=.98)
        save_exact(fig, out)


def main_rq3(root: Path, out: Path) -> None:
    # This is already produced by the RQ3 story script in the exact established
    # style, so publish that canonical current-results figure unchanged.
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
        fig, (ax_cov, ax_comp) = plt.subplots(2, 1, figsize=(5.35, 2.45), sharex=True,
                                              gridspec_kw={"height_ratios": [1, 1], "hspace": .018})
        ax_cov.set_facecolor("#f5f8fc")
        ax_comp.set_facecolor("#fcf8f3")
        handles = []
        cov_ann = []
        comp_ann = []
        max_cov, max_comp = 0., 0.
        for _task, label, cov, comp in series:
            line, = ax_cov.plot(steps, cov, marker="o", linewidth=1.35, markersize=4.2,
                                markeredgewidth=.80, label=label)
            color = line.get_color()
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
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", pad_inches=.015)
        plt.close(fig)


def _grammar_story(root: Path):
    base = root / "analysis/rq4_learning/poisoning/per_run_visualizations/grammar"
    screens = sorted(base.rglob("clean_reference_benign_budget_screen.csv"))
    if not screens:
        return None, None, None
    story = screens[0].parent
    screen = read(story / "clean_reference_benign_budget_screen.csv")
    summary = read(story / "clean_reference_benign_budget_checkpoint_summary.csv")
    curve = read(story / "clean_reference_benign_budget_curve.csv")
    return screen, summary, curve


def main_rq4_defense(root: Path, out: Path) -> None:
    selection, operating, budget_summary = _grammar_story(root)
    if selection is None or selection.empty or operating.empty or budget_summary.empty:
        return
    selection = selection.copy()
    operating = operating.copy()
    budget_summary = budget_summary.copy()
    for frame in [selection, operating, budget_summary]:
        for c in frame.columns:
            if c in {
                "target_fraction", "target_attack_suppression_rate", "target_benign_damage_rate",
                "target_defense_leverage_proxy", "benign_damage_budget", "mean_attack_suppression",
                "mean_poisoned_disruption", "mean_defense_leverage", "n_selected",
            }:
                frame[c] = _num(frame[c])

    # Match stage07's established panel geometry and labels.
    with plt.rc_context({
        "font.size": 9.6, "axes.labelsize": 10.0, "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.6, "legend.fontsize": 8.8, "axes.linewidth": .8,
        "xtick.major.width": .8, "ytick.major.width": .8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, (ax_line, ax_trade) = plt.subplots(1, 2, figsize=(12.82, 4.09))
        fig.subplots_adjust(left=.075, right=.985, bottom=.20, top=.94, wspace=.28)
        x = 100 * operating["target_fraction"].to_numpy(float)
        attack = 100 * operating["mean_attack_suppression"].to_numpy(float)
        benign = 100 * operating["mean_poisoned_disruption"].to_numpy(float)
        leverage = 100 * operating["mean_defense_leverage"].to_numpy(float)
        ax_line.plot(x, attack, marker="D", linewidth=2.2, label="Attack suppression")
        ax_line.plot(x, benign, marker="o", linewidth=2.2, label="Benign damage")
        ax_line.fill_between(x, benign, attack, where=attack >= benign, interpolate=True,
                             color="#dcefdc", alpha=.85, linewidth=0, label=r"$\Delta_{\rm def}>0$")
        ax_line.fill_between(x, benign, attack, where=attack < benign, interpolate=True,
                             color="#f4dede", alpha=.85, linewidth=0)
        for xv, av, bv, lev in zip(x, attack, benign, leverage):
            ax_line.annotate(f"{lev:+.0f} pp", (xv, max(av, bv)), xytext=(0, 7), textcoords="offset points",
                             ha="center", va="bottom", fontsize=7.4)
        ax_line.set_xticks(x)
        ax_line.set_xlim(max(0, float(np.nanmin(x)) - 5), min(100, float(np.nanmax(x)) + 5))
        ax_line.set_ylim(bottom=0)
        ax_line.set_xlabel("Checkpoint (%)")
        ax_line.set_ylabel("Intervention rate (%)")
        ax_line.grid(True, alpha=.18)
        ax_line.legend(frameon=False, fontsize=7.6, loc="upper left")

        inset = ax_line.inset_axes([.54, .08, .43, .37])
        checkpoints = sorted(_num(budget_summary["target_fraction"]).dropna().unique())
        markers = ["o", "s", "^", "D", "P", "X"]
        for idx, frac in enumerate(checkpoints):
            cp = budget_summary[np.isclose(budget_summary["target_fraction"].to_numpy(float), float(frac), equal_nan=False)].sort_values("benign_damage_budget")
            y = 100 * cp["mean_defense_leverage"].to_numpy(float)
            n = _num(cp["n_selected"]).fillna(0).to_numpy(int)
            y[n <= 0] = np.nan
            inset.plot(100 * cp["benign_damage_budget"].to_numpy(float), y,
                       marker=markers[idx % len(markers)], linewidth=1.0, markersize=2.2)
        inset.axhline(0, color="0.45", linestyle="--", linewidth=.7)
        inset.axvline(30, color="0.50", linestyle=":", linewidth=.8)
        inset.set_xlabel(r"$\tau$ (%)", fontsize=6.5, labelpad=1)
        inset.set_ylabel(r"$\Delta_{\rm def}$ (pp)", fontsize=6.5, labelpad=1)
        inset.tick_params(axis="both", labelsize=6.0, pad=1)
        inset.grid(True, alpha=.12)

        single = selection[np.isclose(selection["benign_damage_budget"].to_numpy(float), .30, equal_nan=False)].copy()
        single = single.dropna(subset=["target_fraction", "unit_key", "target_attack_suppression_rate", "target_benign_damage_rate"])
        single = single.drop_duplicates(subset=["target_fraction", "unit_key"], keep="last")
        xvals = 100 * single["target_benign_damage_rate"].to_numpy(float)
        yvals = 100 * single["target_attack_suppression_rate"].to_numpy(float)
        lim = max(70., min(100., float(np.nanmax(np.r_[xvals, yvals])) * 1.10))
        trade_x = np.linspace(0, lim, 400)
        ax_trade.fill_between(trade_x, trade_x, lim, color="#dcefdc", alpha=.85, linewidth=0, zorder=0)
        ax_trade.fill_between(trade_x, 0, trade_x, color="#f4dede", alpha=.85, linewidth=0, zorder=0)
        ax_trade.plot([0, lim], [0, lim], "--", color="0.42", linewidth=1.0, zorder=1)
        ax_trade.axvline(30, color="0.50", linestyle=":", linewidth=1.1)
        ax_trade.text(.025*lim, .93*lim, "attack-selective", fontsize=8.0, fontweight="bold", color="0.18")
        ax_trade.text(.72*lim, .08*lim, "benign-costly", fontsize=8.0, fontweight="bold", color="0.18")
        for idx, frac in enumerate(sorted(single["target_fraction"].astype(float).unique())):
            cp = single[np.isclose(single["target_fraction"].to_numpy(float), frac, equal_nan=False)]
            ax_trade.scatter(100*cp["target_benign_damage_rate"], 100*cp["target_attack_suppression_rate"],
                             s=52, marker=markers[idx % len(markers)], zorder=3,
                             label=f"{int(round(100*frac))}% checkpoint")
        ax_trade.set_xlim(0, lim); ax_trade.set_ylim(0, lim)
        ax_trade.set_xlabel("Single-channel benign damage (%)")
        ax_trade.set_ylabel("Single-channel attack suppression (%)")
        ax_trade.grid(True, alpha=.18)
        ax_trade.legend(frameon=False, fontsize=6.8, loc="upper right")
        for label, ax in zip(["(a)", "(b)"], [ax_line, ax_trade]):
            ax.text(-.035, 1.025, label, transform=ax.transAxes, fontsize=10.5,
                    fontweight="bold", ha="left", va="bottom")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", pad_inches=.015)
        plt.close(fig)


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


def main() -> None:
    args = parse_args()
    root = Path(args.results_root).resolve()
    main_dir = Path(args.main_dir).resolve(); main_dir.mkdir(parents=True, exist_ok=True)
    supp_dir = Path(args.supp_dir).resolve(); supp_dir.mkdir(parents=True, exist_ok=True)

    jobs_main = [
        (main_rq1, "rq1_competence_causal_reach.pdf"),
        (main_rq2, "rq2_composition_boxplots.pdf"),
        (main_rq3, "rq3_population_event_and_strength.pdf"),
        (main_rq4_traj, "rq4_pythia_checkpoint_trajectory.pdf"),
        (main_rq4_defense, "rq4_grammar_clean_reference_defense.pdf"),
    ]
    jobs_supp = [
        (supp_rq1_phase, "rq1_phase_comparison.pdf"),
        (supp_rq1_size, "rq1_model_size_comparison.pdf"),
        (supp_rq2_decomp, "rq2_singleton_joint_decomposition.pdf"),
        (supp_rq2_matched, "rq2_matched_set_specificity.pdf"),
        (supp_candidate, "rq3_candidate_control_diagnostics.pdf"),
        (supp_strength_matched, "rq3_strength_matched_thresholdability.pdf"),
        (supp_tecs, "rq3_threshold_event_strength_ecdf.pdf"),
        (supp_event, "rq3_population_event_localization.pdf"),
        (supp_pairs, "rq3_strength_concentration_paired.pdf"),
        (supp_arithmetic, "rq3_arithmetic_competence_concentration.pdf"),
        (supp_transient, "rq3_affine_null_transient_events.pdf"),
        (supp_preemption, "rq3_preemption_diagnostic.pdf"),
        (supp_temporal, "rq3_temporal_duration.pdf"),
        (supp_cross, "rq3_temporal_cross_sweep_validation.pdf"),
    ]
    for fn, name in jobs_main:
        fn(root, main_dir / name)
    for fn, name in jobs_supp:
        fn(root, supp_dir / name)
    print(f"[manuscript-figures] main={main_dir} supplement={supp_dir}")


if __name__ == "__main__":
    main()
