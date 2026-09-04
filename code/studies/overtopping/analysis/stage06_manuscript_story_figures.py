#!/usr/bin/env python3
"""Manuscript-facing overtopping figures organized by research question.

The paper figures use compact phase panels, task-coded points, restrained
regression summaries, and analysis sidecars that identify every plotted setting.

Important display rules:
* the across-setting sample size is the number of settings with a defined metric;
* within-setting directional denominators are uncertainty metadata and do not silently remove settings from fits;
* directional reach shows exact binomial confidence intervals when available;
* dense panels label only a representative collision-free subset; the complete
  point identities are always exported in the figure-data CSV;
* extreme width-normalized densities are clipped only for display and explicitly
  annotated with their true value, so one outlier cannot flatten the full panel.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

from studies.overtopping.analysis.lib.directional_metrics import DIRECTIONAL_MIN_ELIGIBLE_N, direction_for_metric
from scipy import stats

from core.project_paths import PROJECT_ROOT


TASK_ORDER = ["Arithmetic", "Grammar", "NLI", "Random FSM", "Jailbreak"]
TASK_LABELS = {
    "Arithmetic": "Arithmetic",
    "Grammar": "Grammar",
    "NLI": "HANS NLI",
    "Random FSM": "Random FSM",
    "Jailbreak": "Jailbreak",
}
TASK_MARKERS = {
    "Arithmetic": "o",
    "Grammar": "s",
    "NLI": "^",
    "Random FSM": "D",
    "Jailbreak": "h",
}
PHASES = [("I+O", "Input+output"), ("Out", "Output-only")]


def paper_rc():
    return plt.rc_context({
        "font.size": 8.8,
        "axes.labelsize": 9.2,
        "axes.titlesize": 9.8,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.3,
        "axes.linewidth": 0.72,
        "xtick.major.width": 0.72,
        "ytick.major.width": 0.72,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
    })


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def p_text(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 1e-3:
        return f"{value:.1e}"
    return f"{value:.3f}"


def corr(x: Iterable[float], y: Iterable[float]) -> tuple[int, float, float, float, float]:
    a = pd.to_numeric(pd.Series(list(x)), errors="coerce")
    b = pd.to_numeric(pd.Series(list(y)), errors="coerce")
    ok = a.notna() & b.notna()
    a = a[ok].to_numpy(float)
    b = b[ok].to_numpy(float)
    if len(a) < 3 or np.std(a) <= 0 or np.std(b) <= 0:
        return int(len(a)), math.nan, math.nan, math.nan, math.nan
    pr = stats.pearsonr(a, b)
    sr = stats.spearmanr(a, b)
    return int(len(a)), float(pr.statistic), float(pr.pvalue), float(sr.statistic), float(sr.pvalue)


def task_colors(frame: pd.DataFrame) -> dict[str, object]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    tasks = [t for t in TASK_ORDER if t in set(frame["task"].astype(str))]
    tasks += [t for t in frame["task"].astype(str).unique() if t not in tasks]
    return {task: cycle[i % len(cycle)] if cycle else None for i, task in enumerate(tasks)}


def baseline_kind(row: pd.Series) -> str:
    path = str(row.get("stats_dir", "")).lower()
    return "donor" if ("mean-donor" in path or "eval_mean-donor" in path) else "mean"


def short_model(value: object) -> str:
    text = str(value).replace("-Instruct", "")
    text = text.replace("Pythia-1B ", "Pythia 1B@")
    text = text.replace("Pythia-1B", "Pythia 1B")
    text = text.replace("Pythia-6.9B", "Pythia 6.9B")
    return text


def compact_model(value: object) -> str:
    text = short_model(value)
    text = text.replace("Qwen2.5-1.5B", "Q2.5-1.5B")
    text = text.replace("Qwen2-1.5B", "Q2-1.5B")
    text = text.replace("Qwen2-7B", "Q2-7B")
    text = text.replace("Pythia 1B", "Py-1B")
    text = text.replace("Pythia 6.9B", "Py-6.9B")
    return text


def phase_x_label(phase: str) -> str:
    return "Raw task score" if phase == "I+O" else r"Chance-normalized score $\kappa$"


def adequate_mask(frame: pd.DataFrame, metric: str) -> pd.Series:
    direction = direction_for_metric(metric)
    if direction is None:
        return pd.Series(True, index=frame.index)
    ncol = f"U_J_{direction}_n"
    if ncol not in frame.columns:
        return pd.Series(False, index=frame.index)
    return numeric(frame, ncol) >= DIRECTIONAL_MIN_ELIGIBLE_N


def _display_cap(values: pd.Series) -> float | None:
    """Return a robust display cap only for an extreme isolated upper outlier."""
    x = pd.to_numeric(values, errors="coerce").dropna()
    x = x[np.isfinite(x) & (x >= 0)]
    if len(x) < 8:
        return None
    ordered = np.sort(x.to_numpy(float))
    top = float(ordered[-1])
    second = float(ordered[-2])
    q90 = float(np.quantile(ordered, 0.90))
    reference = max(second, q90, 1e-9)
    if top >= 4.0 * reference:
        return max(1.18 * second, 1.30 * q90)
    return None


def _bbox_overlap(a: Bbox, b: Bbox) -> float:
    if not a.overlaps(b):
        return 0.0
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))


def _representative_label_indices(part: pd.DataFrame, xcol: str, ycol: str, *, max_labels: int = 8) -> list[int]:
    """Select a diverse subset of points for labels; all points remain in the CSV."""
    work = part.copy()
    x = numeric(work, xcol)
    y = numeric(work, ycol)
    ok = x.notna() & y.notna() & adequate_mask(work, ycol)
    ids = list(work.index[ok])
    if len(ids) <= max_labels:
        return ids
    coords = np.column_stack([x.loc[ids].to_numpy(float), y.loc[ids].to_numpy(float)])
    scale = np.nanstd(coords, axis=0)
    scale[scale <= 0] = 1.0
    z = (coords - np.nanmean(coords, axis=0)) / scale
    # Seed with extremes, then farthest-point sample for coverage.
    seed_pos = {
        int(np.nanargmin(z[:, 0])), int(np.nanargmax(z[:, 0])),
        int(np.nanargmin(z[:, 1])), int(np.nanargmax(z[:, 1])),
    }
    selected = list(seed_pos)
    while len(selected) < min(max_labels, len(ids)):
        remaining = [i for i in range(len(ids)) if i not in selected]
        best = max(remaining, key=lambda i: min(float(np.sum((z[i] - z[j]) ** 2)) for j in selected))
        selected.append(best)
    return [ids[i] for i in selected]


def _place_labels(ax, part: pd.DataFrame, xcol: str, ycol: str, *, y_cap: float | None = None, max_labels: int = 8) -> None:
    """Collision-aware compact labels, replacing the broken fixed-offset scheme."""
    indices = _representative_label_indices(part, xcol, ycol, max_labels=max_labels)
    if not indices:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(renderer)
    safe = Bbox.from_extents(axbb.x0 + 2, axbb.y0 + 2, axbb.x1 - 2, axbb.y1 - 2)
    obstacles: list[Bbox] = list(getattr(ax, "_label_obstacles", []))
    offsets = []
    for r in (5, 8, 12, 17, 23, 31, 40):
        offsets.extend([(r, r, "left", "bottom"), (r, -r, "left", "top"), (-r, r, "right", "bottom"), (-r, -r, "right", "top"), (0, r, "center", "bottom"), (0, -r, "center", "top")])

    for idx in indices:
        row = part.loc[idx]
        x = float(row[xcol]); raw_y = float(row[ycol])
        y = min(raw_y, y_cap) if y_cap is not None else raw_y
        label = compact_model(row.get("model", ""))
        best = None; best_score = float("inf"); best_bb = None
        for dx, dy, ha, va in offsets:
            trial = ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                                fontsize=5.9, color="0.12", arrowprops={"arrowstyle": "-", "color": "0.72", "lw": 0.32})
            bb = trial.get_window_extent(renderer).expanded(1.06, 1.12)
            trial.remove()
            outside = max(0.0, safe.x0-bb.x0, bb.x1-safe.x1, safe.y0-bb.y0, bb.y1-safe.y1)
            if outside > 0:
                continue
            overlap = sum(_bbox_overlap(bb, ob) for ob in obstacles)
            score = 10000.0 * overlap + 0.03 * (abs(dx) + abs(dy))
            if score < best_score:
                best_score, best, best_bb = score, (dx, dy, ha, va), bb
            if overlap == 0:
                break
        if best is None:
            continue
        dx, dy, ha, va = best
        ann = ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                          fontsize=5.9, color="0.12", arrowprops={"arrowstyle": "-", "color": "0.72", "lw": 0.32},
                          annotation_clip=True, zorder=6)
        fig.canvas.draw()
        obstacles.append(ann.get_window_extent(renderer).expanded(1.06, 1.12) if best_bb is None else best_bb)


def _sidecar_path(out: Path, data_dir: Path | None) -> Path:
    target = (data_dir / out.name).with_suffix(".csv") if data_dir is not None else out.with_suffix(".csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def draw_phase_scatter(
    frame: pd.DataFrame,
    *,
    xcol: str,
    ycol: str,
    ylabel: str,
    out: Path,
    y_limits: tuple[float, float] | None = None,
    label_points: bool = True,
    xlabels: dict[str, str] | None = None,
    data_dir: Path | None = None,
    robust_y: bool = False,
    xscale: str | None = None,
    show_fit: bool = True,
) -> pd.DataFrame:
    out.parent.mkdir(parents=True, exist_ok=True)
    colors = task_colors(frame)
    stats_rows: list[dict] = []
    y_cap = _display_cap(numeric(frame, ycol)) if robust_y else None
    with paper_rc():
        fig, axes = plt.subplots(1, 2, figsize=(7.65, 2.65), sharey=True)
        for ax, (phase, title) in zip(axes, PHASES):
            part = frame[frame["phase"].astype(str).eq(phase)].copy()
            adequate = adequate_mask(part, ycol)
            for task in [t for t in TASK_ORDER if t in set(part["task"].astype(str))]:
                g = part[part["task"].astype(str).eq(task)].copy()
                for idx, row in g.iterrows():
                    xv = pd.to_numeric(pd.Series([row.get(xcol)]), errors="coerce").iloc[0]
                    yv = pd.to_numeric(pd.Series([row.get(ycol)]), errors="coerce").iloc[0]
                    if pd.isna(xv) or pd.isna(yv):
                        continue
                    good = bool(adequate.loc[idx])
                    donor = baseline_kind(row) == "donor"
                    plot_y = min(float(yv), y_cap) if y_cap is not None else float(yv)
                    alpha = 1.0 if good else 0.38
                    # Exact binomial interval for directional union reach.
                    direction = direction_for_metric(ycol)
                    if ycol in {"U_J_i2c", "U_J_c2i"} and direction is not None:
                        lo = pd.to_numeric(pd.Series([row.get(f"U_J_{direction}_ci_low")]), errors="coerce").iloc[0]
                        hi = pd.to_numeric(pd.Series([row.get(f"U_J_{direction}_ci_high")]), errors="coerce").iloc[0]
                        if pd.notna(lo) and pd.notna(hi):
                            ax.errorbar(float(xv), plot_y, yerr=[[max(0.0, plot_y-float(lo))], [max(0.0, float(hi)-plot_y)]],
                                        fmt="none", ecolor=colors[task], elinewidth=0.65, capsize=1.4, alpha=0.42 if good else 0.20, zorder=2)
                    marker = TASK_MARKERS.get(task, "o")
                    if y_cap is not None and float(yv) > y_cap:
                        marker = "^"
                    ax.scatter(float(xv), plot_y, s=31, marker=marker,
                               facecolor=(colors[task] if donor else "white"), edgecolor=colors[task],
                               linewidth=0.88, alpha=alpha, zorder=3)
                    if not good:
                        ax.scatter(float(xv), plot_y, s=18, marker="x", color="0.35", linewidth=0.65, alpha=0.6, zorder=4)
                    if y_cap is not None and float(yv) > y_cap:
                        ax.annotate(f"{float(yv):.0f}", (float(xv), plot_y), xytext=(0, 5), textcoords="offset points",
                                    ha="center", va="bottom", fontsize=6.2, color="0.15")

            x = numeric(part, xcol); y = numeric(part, ycol)
            fit_ok = x.notna() & y.notna() & adequate
            n_total = int((x.notna() & y.notna()).sum())
            n, r, p, rho, sp = corr(x[fit_ok], y[fit_ok])
            if show_fit and n >= 3 and x[fit_ok].nunique() > 1:
                coef = np.polyfit(x[fit_ok].to_numpy(float), y[fit_ok].to_numpy(float), 1)
                xx = np.linspace(float(x[fit_ok].min()), float(x[fit_ok].max()), 100)
                ax.plot(xx, coef[0] * xx + coef[1], linestyle="--", linewidth=1.0, color="0.25", alpha=0.78, zorder=1)
            sample_text = f"n={n}" if n == n_total else f"n={n}/{n_total}"
            stat_artist = ax.text(0.025, 0.975, f"{sample_text}, Pearson r={r:+.2f}\np={p_text(p)}",
                                  transform=ax.transAxes, ha="left", va="top", fontsize=6.9,
                                  bbox={"boxstyle": "square,pad=0.18", "facecolor": "white", "edgecolor": "0.72", "linewidth": 0.45, "alpha": 0.90}, zorder=8)
            fig.canvas.draw()
            setattr(ax, "_label_obstacles", [stat_artist.get_window_extent(fig.canvas.get_renderer()).expanded(1.06, 1.12)])
            if label_points:
                _place_labels(ax, part.loc[adequate], xcol, ycol, y_cap=y_cap, max_labels=8)
            ax.set_title(title)
            ax.set_xlabel((xlabels or {}).get(phase, phase_x_label(phase)))
            ax.grid(True, alpha=0.22, linewidth=0.45)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            if y_limits is not None:
                ax.set_ylim(*y_limits)
            elif y_cap is not None:
                ax.set_ylim(bottom=0, top=y_cap * 1.08)
            if xscale:
                ax.set_xscale(xscale)
            stats_rows.append({"phase": phase, "metric_x": xcol, "metric_y": ycol, "n_adequate": n, "n_total": n_total,
                               "pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": sp,
                               "directional_min_eligible_n": DIRECTIONAL_MIN_ELIGIBLE_N if direction_for_metric(ycol) else math.nan,
                               "display_y_cap": y_cap})
        axes[0].set_ylabel(ylabel, labelpad=5)
        handles = [Line2D([0], [0], marker=TASK_MARKERS.get(task, "o"), linestyle="None", markerfacecolor=colors[task],
                          markeredgecolor=colors[task], markersize=5.0, label=TASK_LABELS.get(task, task)) for task in colors]
        handles += [
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white", markeredgecolor="0.25", markersize=5.0, label="mean abl."),
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="0.45", markeredgecolor="0.25", markersize=5.0, label="donor abl."),
        ]
        if direction_for_metric(ycol):
            handles.append(Line2D([0], [0], marker="x", linestyle="None", color="0.35", markersize=5.0, label=f"eligible n<{DIRECTIONAL_MIN_ELIGIBLE_N}"))
        if show_fit:
            handles.append(Line2D([0], [0], linestyle="--", color="0.25", lw=1.0, label="adequate-sample fit"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=5, frameon=False,
                   handletextpad=0.35, columnspacing=0.8)
        fig.subplots_adjust(left=0.105, right=0.995, bottom=0.31, top=0.94, wspace=0.12)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    stats_df = pd.DataFrame(stats_rows)
    sidecar = _sidecar_path(out, data_dir)
    stats_path = sidecar.with_name(sidecar.stem + "_stats.csv")
    points_path = sidecar.with_name(sidecar.stem + "_points.csv")
    stats_df.to_csv(stats_path, index=False)
    points = frame.copy()
    points["adequate_for_directional_fit"] = adequate_mask(points, ycol)
    points["display_y"] = numeric(points, ycol).clip(upper=y_cap) if y_cap is not None else numeric(points, ycol)
    keep = [c for c in [
        "task", "model", "phase", "score", "stats_dir", xcol, ycol,
        "U_J_i2c_n", "U_J_c2i_n", "U_J_i2c_ci_low", "U_J_i2c_ci_high",
        "U_J_c2i_ci_low", "U_J_c2i_ci_high", "adequate_for_directional_fit", "display_y"
    ] if c in points.columns]
    points[keep].to_csv(points_path, index=False)
    return stats_df


def plot_rq1(frame: pd.DataFrame, out_dir: Path, data_dir: Path | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("U_J_i2c", r"Directional causal reach $U_{0\to1}(J)$", "fig2a_competence_vs_U_0to1.pdf", (0.0, 1.0), False),
        ("U_J_c2i", r"Directional causal reach $U_{1\to0}(J)$", "fig2b_competence_vs_U_1to0.pdf", (0.0, 1.0), False),
        ("N05_i2c_density", r"High-effect density $D^{0\to1}_{.05}=N_{.05}/d_{\rm layer}$", "fig2c_competence_vs_D05_0to1.pdf", (0.0, 1.0), False),
        ("N05_c2i_density", r"High-effect density $D^{1\to0}_{.05}=N_{.05}/d_{\rm layer}$", "fig2d_competence_vs_D05_1to0.pdf", (0.0, 1.0), False),
    ]
    for metric, ylabel, name, ylim, robust in specs:
        if metric in frame.columns and numeric(frame, metric).notna().any():
            draw_phase_scatter(frame, xcol="score", ycol=metric, ylabel=ylabel, out=out_dir/name, y_limits=ylim,
                               data_dir=(data_dir / "02_rq1_prevalence" if data_dir else None), robust_y=robust)
    for metric, ylabel, name in [
        ("N10_i2c_density", r"High-effect density $D^{0\to1}_{.10}=N_{.10}/d_{\rm layer}$", "fig2s1_competence_vs_D10_0to1.pdf"),
        ("N10_c2i_density", r"High-effect density $D^{1\to0}_{.10}=N_{.10}/d_{\rm layer}$", "fig2s2_competence_vs_D10_1to0.pdf"),
    ]:
        if metric in frame.columns and numeric(frame, metric).notna().any():
            draw_phase_scatter(frame, xcol="score", ycol=metric, ylabel=ylabel, out=out_dir/name,
                               data_dir=(data_dir / "02_rq1_prevalence" if data_dir else None), robust_y=True)
    for direction, ucol, neff, name in [
        ("0\\to1", "U_J_i2c", "N_eff_i2c_per_1k_layer", "fig2e_reach_vs_effective_support_0to1.pdf"),
        ("1\\to0", "U_J_c2i", "N_eff_c2i_per_1k_layer", "fig2s3_reach_vs_effective_support_1to0.pdf"),
    ]:
        if neff in frame.columns and numeric(frame, neff).notna().any():
            draw_phase_scatter(frame, xcol=neff, ycol=ucol, ylabel=rf"Directional reach $U_{{{direction}}}(J)$", out=out_dir/name,
                               y_limits=(0.0, 1.0), label_points=False,
                               xlabels={"I+O": r"Effective support / layer $\times10^3$", "Out": r"Effective support / layer $\times10^3$"},
                               data_dir=(data_dir / "02_rq1_prevalence" if data_dir else None), xscale="symlog", show_fit=False)


def compact_setting_label(row: pd.Series) -> str:
    task = {"Random FSM": "FSM", "Arithmetic": "Arith.", "Jailbreak": "Jailbreak", "Grammar": "Grammar", "NLI": "NLI"}.get(str(row.get("task", "")), str(row.get("task", "")))
    return f"{task} | {compact_model(row.get('model',''))} | {row.get('phase','')}"


def plot_horizontal_metric(frame: pd.DataFrame, value_col: str, xlabel: str, out: Path, *, zero_line: bool = True, data_dir: Path | None = None) -> None:
    work = frame.copy(); work[value_col] = numeric(work, value_col); work = work.dropna(subset=[value_col]).sort_values(value_col)
    if work.empty: return
    colors = task_colors(work); height = max(3.4, 0.235 * len(work) + 0.7)
    with paper_rc():
        fig, ax = plt.subplots(figsize=(6.55, height)); y = np.arange(len(work))
        if zero_line: ax.axvline(0.0, color="0.15", lw=0.8, zorder=1)
        for yi, (_, row) in zip(y, work.iterrows()):
            c = colors.get(str(row["task"]), "0.35"); val=float(row[value_col])
            ax.plot([0, val], [yi, yi], color=c, alpha=0.60, lw=1.05, zorder=2)
            ax.scatter(val, yi, s=26, marker=TASK_MARKERS.get(str(row["task"]), "o"), color=c, zorder=3)
        ax.set_yticks(y); ax.set_yticklabels([compact_setting_label(row) for _, row in work.iterrows()], fontsize=7.0)
        ax.set_xlabel(xlabel); ax.grid(axis="x", alpha=0.22, linewidth=0.45)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.spines["left"].set_visible(False)
        fig.subplots_adjust(left=0.31, right=0.985, bottom=0.07, top=0.99)
        out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, bbox_inches="tight", pad_inches=0.03); plt.close(fig)
    work.to_csv(_sidecar_path(out, data_dir), index=False)


def plot_superadditive_boundary(frame: pd.DataFrame, out: Path, data_dir: Path | None = None) -> pd.DataFrame:
    work = frame.copy(); work["Delta_comp"] = numeric(work, "E_J") - numeric(work, "U")
    work = work[np.isfinite(work["Delta_comp"]) & (work["Delta_comp"] > 0)].sort_values("Delta_comp", ascending=False)
    if work.empty: return work
    colors=task_colors(work)
    with paper_rc():
        fig, ax = plt.subplots(figsize=(6.35, max(2.55, 0.64*len(work)+1.05))); y=np.arange(len(work))
        u_values = numeric(work, "U")
        e_values = numeric(work, "E_J")
        delta_values = numeric(work, "Delta_comp")
        xmax = float(np.nanmax(np.concatenate([u_values.to_numpy(float), e_values.to_numpy(float), [0.0]])))
        label_pad = max(0.012, 0.028 * max(xmax, 0.22))
        delta_pad = max(0.018, 0.05 * max(xmax, 0.22))

        ax.barh(y-0.13, u_values, height=0.24, color="0.80", edgecolor="0.45", linewidth=0.5,
                label=r"Singleton-union baseline $U(J)$")
        # A numerically exact zero has no visible bar width. Mark the endpoint at
        # x=0 so boundary cases remain visible without changing the plotted value.
        zero_u = np.isfinite(u_values.to_numpy(float)) & np.isclose(u_values.to_numpy(float), 0.0)
        if zero_u.any():
            ax.scatter(np.zeros(int(zero_u.sum())), (y-0.13)[zero_u], marker="|", s=85, color="0.45", linewidths=1.2,
                       zorder=4, label=r"Valid zero $U(J)$")
        ax.barh(y+0.13, e_values, height=0.24, color="0.38", edgecolor="0.2", linewidth=0.5,
                label=r"Observed joint-set effect $E(J)$")
        for yi,(_,row) in zip(y,work.iterrows()):
            ej = float(row["E_J"])
            uj = float(row["U"])
            dj = float(row["Delta_comp"])
            ax.scatter(ej, yi+0.13, marker=TASK_MARKERS.get(str(row["task"]),"o"), s=20,
                       color=colors.get(str(row["task"]),"0.2"), zorder=4)
            ax.annotate(f"U={uj:.3f}", (uj, yi-0.13), xytext=(4, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=6.3, color="0.20",
                        bbox={"boxstyle":"round,pad=0.07","facecolor":"white","edgecolor":"0.75","linewidth":0.35,"alpha":0.90},
                        annotation_clip=True)
            ax.annotate(f"E={ej:.3f}", (ej, yi+0.13), xytext=(4, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=6.3, color="0.12",
                        bbox={"boxstyle":"round,pad=0.07","facecolor":"white","edgecolor":"0.45","linewidth":0.35,"alpha":0.92},
                        annotation_clip=True)
            ax.annotate(f"Δ={dj:+.3f}", (max(ej, uj), yi), xytext=(6, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=6.3, color="0.12",
                        bbox={"boxstyle":"round,pad=0.07","facecolor":"white","edgecolor":"0.55","linewidth":0.35,"alpha":0.88},
                        annotation_clip=True)
        ax.set_yticks(y); ax.set_yticklabels([compact_setting_label(row) for _,row in work.iterrows()], fontsize=7.2)
        ax.set_xlim(0.0, min(1.02, xmax + delta_pad + 0.18))
        ax.set_xlabel(r"Held-out behavioural effect (light bar: singleton-union baseline $U(J)$; dark bar: observed joint effect $E(J)$)")
        ax.set_title(r"Boundary cases where joint effect exceeds singleton-union reach: $E(J) > U(J)$", fontsize=9.3, pad=8)
        handles, labels = ax.get_legend_handles_labels()
        seen = set(); uniq_h=[]; uniq_l=[]
        for h,l in zip(handles, labels):
            if l in seen: continue
            seen.add(l); uniq_h.append(h); uniq_l.append(l)
        ax.legend(uniq_h, uniq_l, frameon=False, loc="lower right", ncol=1, fontsize=7.0)
        ax.grid(axis="x", alpha=0.22, linewidth=0.45); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.text(0.312, 0.055,
                 r"Interpretation: $U(J)$ is singleton-union reach—the fraction of held-out examples flipped by at least one singleton member of $J$; $E(J)$ is the held-out effect of intervening on the full set $J$ simultaneously. Positive $\Delta=E(J)-U(J)$ means the joint intervention exceeds singleton-union reach.",
                 ha="left", va="bottom", fontsize=6.6, color="0.18")
        fig.subplots_adjust(left=0.31,right=0.988,bottom=0.26,top=0.91)
        out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,bbox_inches="tight",pad_inches=0.03); plt.close(fig)
    work.to_csv(_sidecar_path(out, data_dir), index=False)
    return work


def plot_rq2(frame: pd.DataFrame, manuscript: pd.DataFrame | None, out_dir: Path, data_dir: Path | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True); work=frame.copy()
    if manuscript is not None and not manuscript.empty:
        join_cols=[c for c in ["task","model","phase"] if c in manuscript.columns and c in work.columns]
        extra=[c for c in ["E_J_Delta","E_J_null_median","E_J_p_MC","E_J_status","E_J_null_status"] if c in manuscript.columns]
        if join_cols and extra: work=work.merge(manuscript[join_cols+extra],on=join_cols,how="left",suffixes=("","_manuscript"))
    work["Delta_comp"]=numeric(work,"E_J")-numeric(work,"U")
    rqdata = data_dir / "03_rq2_composition" if data_dir else None
    if numeric(work,"Delta_comp").notna().any():
        plot_horizontal_metric(work,"Delta_comp",r"Composition gap $\Delta_{\rm comp}=E(J)-U(J)$",out_dir/"fig3a_composition_gap_all_settings.pdf",data_dir=rqdata)
        boundary=plot_superadditive_boundary(work,out_dir/"fig3b_superadditive_boundary_cases.pdf",data_dir=rqdata)
        if not boundary.empty:
            cols=[c for c in ["task","model","phase","score","U","E_J","Delta_comp","U_J_i2c","U_J_c2i","s_1_i2c","s_1_c2i","N05_i2c_density","N05_c2i_density"] if c in boundary.columns]
            target=(rqdata/"table3_superadditive_boundary_cases.csv") if rqdata else (out_dir/"table3_superadditive_boundary_cases.csv")
            target.parent.mkdir(parents=True,exist_ok=True); boundary[cols].to_csv(target,index=False)
    if "E_J_Delta" in work.columns and numeric(work,"E_J_Delta").notna().any():
        plot_horizontal_metric(work,"E_J_Delta",r"Candidate specificity $E(J)-\mathrm{median}\,E(K_b)$",out_dir/"fig3c_matched_set_specificity.pdf",data_dir=rqdata)


def checkpoint_step(model: object) -> tuple[int, str]:
    low=str(model).lower()
    if "48k" in low or "step48000" in low: return 48000,"48k"
    if "96k" in low or "step96000" in low: return 96000,"96k"
    if "pythia-1b" in low: return 143000,"143k"
    return -1,str(model)


def plot_rq4_pythia(frame: pd.DataFrame, out: Path, data_dir: Path | None = None) -> pd.DataFrame:
    work=frame[frame["model"].astype(str).str.contains("Pythia-1B",case=False,regex=False)].copy()
    if work.empty: return work
    work[["checkpoint_step","checkpoint_label"]]=work["model"].apply(lambda v: pd.Series(checkpoint_step(v)))
    work=work[(work["checkpoint_step"]>=0)&work["phase"].astype(str).eq("I+O")].copy()
    if work.empty: return work
    colors=task_colors(work); steps=sorted(work["checkpoint_step"].unique())

    def annotate(ax, xs, ys, color, *, valid=None):
        for i,(x,y) in enumerate(zip(xs,ys)):
            if not np.isfinite(float(y)): continue
            alpha=1.0 if valid is None or bool(valid[i]) else 0.45
            dy = -9 if float(y) >= 0.88 else 3
            va = "top" if dy < 0 else "bottom"
            ax.annotate(f"{float(y):.2f}",(float(x),float(y)),xytext=(3,dy),textcoords="offset points",fontsize=6.2,color="0.12",alpha=alpha,va=va,
                        bbox={"boxstyle":"round,pad=0.04","facecolor":"white","edgecolor":color,"linewidth":0.4,"alpha":0.86},annotation_clip=True)

    with paper_rc():
        fig,(ax_i2c,ax_c2i,ax_comp)=plt.subplots(3,1,figsize=(5.45,3.45),sharex=True,gridspec_kw={"height_ratios":[1,1,0.82],"hspace":0.045})
        for task in [t for t in TASK_ORDER if t in set(work["task"].astype(str))]:
            g=work[work["task"].astype(str).eq(task)].sort_values("checkpoint_step"); color=colors[task]
            xs=g["checkpoint_step"].to_numpy(float); i2c=numeric(g,"U_J_i2c").to_numpy(float); c2i=numeric(g,"U_J_c2i").to_numpy(float); score=numeric(g,"score").to_numpy(float)
            i2c_ok=(numeric(g,"U_J_i2c_n")>=DIRECTIONAL_MIN_ELIGIBLE_N).to_numpy(bool); c2i_ok=(numeric(g,"U_J_c2i_n")>=DIRECTIONAL_MIN_ELIGIBLE_N).to_numpy(bool)
            ax_i2c.plot(xs,i2c,marker="o",lw=1.35,ms=4.0,color=color,label=TASK_LABELS.get(task,task))
            # Break the c2i line across low-denominator checkpoints instead of implying precision.
            c2i_line=np.where(c2i_ok,c2i,np.nan); ax_c2i.plot(xs,c2i_line,marker="o",lw=1.25,ms=4.0,color=color)
            if (~c2i_ok & np.isfinite(c2i)).any():
                ax_c2i.scatter(xs[~c2i_ok],c2i[~c2i_ok],marker="x",s=25,color=color,alpha=0.45,zorder=4)
            ax_comp.plot(xs,score,marker="s",mfc="white",mec=color,linestyle="--",lw=1.15,ms=3.8,color=color)
            annotate(ax_i2c,xs,i2c,color,valid=i2c_ok); annotate(ax_c2i,xs,c2i,color,valid=c2i_ok); annotate(ax_comp,xs,score,color)
        ax_i2c.set_ylabel(r"$U_{0\to1}(J)$"); ax_c2i.set_ylabel(r"$U_{1\to0}(J)$"); ax_comp.set_ylabel("Competence")
        ax_i2c.set_ylim(-0.02,1.02); ax_c2i.set_ylim(-0.02,1.02); ax_comp.set_ylim(-0.02,max(0.12,min(1.0,float(numeric(work,"score").max())+0.08)))
        labels=[str(work.loc[work["checkpoint_step"].eq(step),"checkpoint_label"].iloc[0]) for step in steps]
        ax_comp.set_xticks(steps,labels); ax_comp.set_xlabel("Pythia-1B checkpoint")
        for ax in (ax_i2c,ax_c2i,ax_comp):
            ax.grid(True,alpha=0.20,linewidth=0.45); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        task_handles=[Line2D([0],[0],color=colors[t],marker="o",lw=1.25,ms=4.0,label=TASK_LABELS.get(t,t)) for t in colors]
        task_handles.append(Line2D([0],[0],color="0.35",marker="x",linestyle="None",ms=5.0,label=f"eligible n<{DIRECTIONAL_MIN_ELIGIBLE_N}"))
        ax_i2c.legend(handles=task_handles,frameon=False,ncol=min(4,len(task_handles)),loc="upper left",bbox_to_anchor=(0.0,1.05),columnspacing=0.8,handlelength=1.3)
        fig.subplots_adjust(left=0.13,right=0.995,bottom=0.13,top=0.96); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,bbox_inches="tight",pad_inches=0.03); plt.close(fig)
    work.to_csv(_sidecar_path(out, data_dir),index=False); return work


def write_rq_readmes(base: Path) -> None:
    texts={
        "02_rq1_prevalence": """# Figure 2 - RQ1: prevalence and competence

The primary directional reach/density figures use the explicit 39-setting primary+supplementary non-poisoning overtopping catalogue (17 input+output, 22 output-only) and phase-panel layout. Genuine zero-candidate settings remain explicit U(J)=0 points. Regression n is the number of plotted settings with a defined metric; within-setting directional denominators describe uncertainty and are not an across-setting sample-size filter.
`fig2e_reach_vs_effective_support_0to1.pdf` and the other structural companions remain primary-matrix context. Machine-readable sidecars live under `results/analysis/figure_data/02_rq1_prevalence/`.
""",
        "03_rq2_composition": """# Figure 3 - RQ2: composition and boundary conditions

`fig3a_composition_gap_all_settings.pdf` is primary. `fig3b_superadditive_boundary_cases.pdf` expands E(J)>U(J) cases. `fig3c_matched_set_specificity.pdf` is the matched-set specificity control. Machine-readable sidecars live under `results/analysis/figure_data/03_rq2_composition/`.
""",
        "05_rq4_learning": """# Figure 5 - RQ4: learning utility

`fig5a_pythia_checkpoint_trajectory.pdf` shows pooled U(J) and competence across Pythia checkpoints, including genuine U(J)=0 states. `fig5s1_pythia_checkpoint_U_0to1.pdf` and `fig5s2_pythia_checkpoint_U_1to0.pdf` are directional companions. Primary-matrix-only checkpoint diagnostics belong under analysis, not in the manuscript figure directory.
""",
    }
    for name,text in texts.items():
        path=base/name; path.mkdir(parents=True,exist_ok=True); (path/"README.md").write_text(text,encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary_table_augmented",required=True)
    p.add_argument("--manuscript_metrics",default=None)
    p.add_argument("--out_dir",default=str(PROJECT_ROOT/"results"/"paper"/"figures"))
    p.add_argument("--figure_data_dir",default=None,help="Optional analysis-only directory for CSV sidecars; keeps paper figure folders PDF-only.")
    p.add_argument("--skip_rq1", action="store_true", help="Do not emit primary-matrix RQ1 plots; the final-results orchestrator uses the explicit 39-setting primary+supplementary non-poisoning population instead.")
    p.add_argument("--skip_primary_rq4", action="store_true", help="Do not emit the primary-matrix-only Pythia trajectory into the paper tree.")
    return p.parse_args()


def main() -> None:
    args=parse_args(); frame=pd.read_csv(Path(args.primary_table_augmented).expanduser().resolve()); manuscript=None
    if args.manuscript_metrics:
        path=Path(args.manuscript_metrics).expanduser().resolve()
        if path.is_file(): manuscript=pd.read_csv(path)
    out=Path(args.out_dir).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    data_dir=Path(args.figure_data_dir).expanduser().resolve() if args.figure_data_dir else None
    if data_dir: data_dir.mkdir(parents=True,exist_ok=True)
    write_rq_readmes(out)
    if not args.skip_rq1:
        plot_rq1(frame,out/"02_rq1_prevalence",data_dir)
    plot_rq2(frame,manuscript,out/"03_rq2_composition",data_dir)
    if not args.skip_primary_rq4:
        plot_rq4_pythia(frame,out/"05_rq4_learning"/"fig5s3_primary_matrix_directional_checkpoint_trajectory.pdf",(data_dir/"05_rq4_learning" if data_dir else None))
    print(f"[done] manuscript story figures: {out}")


if __name__ == "__main__":
    main()
