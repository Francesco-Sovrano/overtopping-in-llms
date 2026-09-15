#!/usr/bin/env python3
"""RQ2 aggregate composition report over the configured overtopping study.

The report derives its population from completed configured settings and groups
replacement regimes explicitly.
RQ2 instead separates the replacement counterfactuals: mean-donor is the main
regime, while mean and mean-positional are collapsed into a separate ``mean``
sensitivity regime. The two replacement regimes are never pooled.

Intermediate Pythia checkpoints remain in the headline aggregate so RQ2 uses all
available evaluable settings. A checkpoint-free donor sensitivity is emitted for
the appendix.
"""
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
from scipy import stats

from core.project_paths import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--catalogue-csv",
        default=str(PROJECT_ROOT / "results/analysis/primary_matrix/tables/primary_table.csv"),
        help=(
            "Configured-setting table containing J, U(J), E(J), intervention, phase, task, and model. "
            "The reporting primary_table.csv is the canonical source; completed_experiments.csv is also accepted."
        ),
    )
    p.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "results/analysis/rq2_composition/regime_summary"),
    )
    p.add_argument("--paper-figures-dir", default=None)
    p.add_argument("--paper-tables-dir", default=None)
    return p.parse_args()


def replacement_regime(value: object) -> str:
    """Collapse positional mean into mean, while keeping mean-donor separate."""
    text = str(value).strip().lower()
    if text.startswith("mean-donor"):
        return "mean-donor"
    if text.startswith("mean"):
        return "mean"
    return text


def _short_task(value: object) -> str:
    return {
        "arithmetic": "Arithmetic",
        "grammar_acceptability": "Grammar",
        "hans_nli": "NLI",
        "random_fsm": "Random FSM",
        "bon_jailbreaking": "Jailbreak",
    }.get(str(value), str(value))


def _short_model(value: object) -> str:
    text = str(value).split("/")[-1].replace("-Instruct", "")
    text = text.replace("@step0", " 0k").replace("@step48000", " 48k").replace("@step96000", " 96k")
    return text.replace("pythia-", "Pythia-").replace("qwen", "Qwen")


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize RQ2 fields while retaining the complete configured manifest.

    ``rq2_evaluable`` is the analysis filter. Rows that are inapplicable or lack
    a required aggregate metric remain in the emitted population audit.
    """
    out = frame.copy()
    if "intervention" not in out.columns:
        if "stats_dir" not in out.columns:
            raise ValueError("Input table must contain intervention or stats_dir")
        # Older configured-study exports did not carry an intervention column.
        # Their persistent path convention is sufficient for RQ2's two reporting
        # regimes: donor paths contain eval_mean-donor; all other supported
        # replacement paths belong to the collapsed mean-family regime.
        stats_text = out["stats_dir"].fillna("").astype(str)
        out["intervention"] = np.where(
            stats_text.str.contains("eval_mean-donor", regex=False),
            "mean-donor",
            "mean",
        )
    out["replacement_regime"] = out["intervention"].map(replacement_regime)
    # stage02 calls the pooled singleton-union metric ``U``; the experiment
    # catalogue calls the same quantity ``U_J``. Accept both without rewriting
    # either stored artifact schema.
    u_source = "U_J" if "U_J" in out.columns else "U" if "U" in out.columns else None
    if u_source is None:
        raise ValueError("Input table must contain U_J or U")
    out["U_J"] = pd.to_numeric(out[u_source], errors="coerce")
    out["E_J"] = pd.to_numeric(out.get("E_J"), errors="coerce")
    has_j = "J" in out.columns
    if has_j:
        out["J"] = pd.to_numeric(out["J"], errors="coerce")
    out["Delta_comp"] = out["E_J"] - out["U_J"]
    if "study_component" in out.columns:
        out["is_intermediate_checkpoint"] = out["study_component"].astype(str).eq("intermediate_checkpoint")
    elif "model_id" in out.columns:
        out["is_intermediate_checkpoint"] = out["model_id"].astype(str).str.contains("@step", regex=False)
    else:
        model_text = out["model"].astype(str)
        out["is_intermediate_checkpoint"] = (
            model_text.str.contains("@step", regex=False)
            | model_text.str.contains(r"(?:^|\s)(?:48k|96k)(?:$|\s)", regex=True)
        )
    out["task_label"] = out["task"].map(_short_task)
    out["model_label"] = out["model"].map(_short_model)

    u_ok = np.isfinite(out["U_J"])
    e_ok = np.isfinite(out["E_J"])
    if has_j:
        j_known = np.isfinite(out["J"])
        applicable = j_known & out["J"].gt(0)
    else:
        # Compatibility for aggregate catalogues that predate an explicit J
        # column: finite U(J) and E(J) imply that composition was materialized.
        j_known = pd.Series(False, index=out.index, dtype=bool)
        applicable = u_ok & e_ok

    out["rq2_applicable"] = applicable
    out["rq2_metrics_available"] = u_ok & e_ok
    out["rq2_evaluable"] = applicable & u_ok & e_ok

    status = pd.Series("evaluable", index=out.index, dtype=object)
    if has_j:
        status.loc[~j_known] = "candidate_set_unavailable"
        status.loc[j_known & out["J"].le(0)] = "not_applicable_zero_candidate"
    status.loc[applicable & ~u_ok & ~e_ok] = "missing_U_J_and_E_J"
    status.loc[applicable & ~u_ok & e_ok] = "missing_U_J"
    status.loc[applicable & u_ok & ~e_ok] = "missing_E_J"
    if not has_j:
        status.loc[~out["rq2_evaluable"]] = "aggregate_metrics_unavailable"
    out["rq2_status"] = status
    return out.reset_index(drop=True)


def summarize(frame: pd.DataFrame, *, regime: str, checkpoint_free: bool) -> dict[str, object]:
    work = frame.loc[
        frame["replacement_regime"].eq(regime) & frame["rq2_evaluable"]
    ].copy()
    if checkpoint_free:
        work = work.loc[~work["is_intermediate_checkpoint"]].copy()
    x = work["U_J"].to_numpy(float)
    y = work["Delta_comp"].to_numpy(float)
    pr = stats.pearsonr(x, y) if len(work) >= 3 and np.std(x) > 0 and np.std(y) > 0 else None
    sr = stats.spearmanr(x, y) if len(work) >= 3 and np.std(x) > 0 and np.std(y) > 0 else None
    eps = 1e-12
    return {
        "replacement_regime": regime,
        "population": "checkpoint-free" if checkpoint_free else "all-evaluable",
        "n": int(len(work)),
        "n_negative_gap": int((work["Delta_comp"] < -eps).sum()),
        "n_positive_gap": int((work["Delta_comp"] > eps).sum()),
        "n_tie": int((work["Delta_comp"].abs() <= eps).sum()),
        "median_gap": float(work["Delta_comp"].median()) if len(work) else math.nan,
        "pearson_r": float(pr.statistic) if pr is not None else math.nan,
        "pearson_p": float(pr.pvalue) if pr is not None else math.nan,
        "spearman_rho": float(sr.statistic) if sr is not None else math.nan,
        "spearman_p": float(sr.pvalue) if sr is not None else math.nan,
    }



def _expand_scope_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Materialize per-scope composition gaps from each evaluable setting."""
    rows: list[pd.DataFrame] = []
    work = frame.loc[frame["rq2_evaluable"]].copy()
    for _, row in work.iterrows():
        stats_dir = row.get("stats_dir")
        if pd.isna(stats_dir):
            continue
        summary = Path(str(stats_dir)) / "interaction_validation" / "composition_decomposition_summary.csv"
        if not summary.exists():
            continue
        try:
            d = pd.read_csv(summary)
        except Exception:
            continue
        if "scope" not in d.columns or "Delta_comp_complete_case" not in d.columns:
            continue
        part = d[["scope", "Delta_comp_complete_case"]].copy()
        part["Delta_comp"] = pd.to_numeric(part["Delta_comp_complete_case"], errors="coerce")
        part = part[np.isfinite(part["Delta_comp"])].copy()
        if part.empty:
            continue
        part["replacement_regime"] = row["replacement_regime"]
        rows.append(part[["replacement_regime", "scope", "Delta_comp"]])
    if not rows:
        return pd.DataFrame(columns=["replacement_regime", "scope", "Delta_comp"])
    return pd.concat(rows, ignore_index=True)

def plot_compact_main(frame: pd.DataFrame, path: Path) -> None:
    """Main-paper RQ2 summary stratified by replacement regime and direction."""
    work = _expand_scope_rows(frame)
    if work.empty:
        # Conservative fallback when only aggregate setting-level rows are available.
        work = frame.loc[frame["rq2_evaluable"], ["replacement_regime", "Delta_comp"]].copy()
        work["scope"] = "overall"
    if work.empty:
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

    with plt.rc_context({
        "font.size": 10.0,
        "axes.labelsize": 10.2,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 9.2,
    }):
        fig, ax = plt.subplots(figsize=(5.4, 2.28))
        ax.axhline(0.0, linestyle="--", linewidth=0.8, alpha=0.7, color="0.45", zorder=0)
        handles = []
        for regime in regimes:
            color = regime_colors[regime]
            marker = regime_markers[regime]
            positions, box_vals = [], []
            for center, scope in zip(centers, scopes):
                vals = work.loc[
                    work["replacement_regime"].eq(regime) & work["scope"].eq(scope),
                    "Delta_comp",
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
                jitter = np.linspace(-0.075, 0.075, len(vals)) if len(vals) > 1 else np.array([0.0])
                ax.scatter(
                    np.full(len(vals), positions[i]) + jitter,
                    np.sort(vals),
                    s=17,
                    marker=marker,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=.35,
                    alpha=.76,
                    zorder=3,
                )
            handles.append(
                plt.Line2D([0], [0], marker=marker, linestyle="none", markersize=7.0,
                           markerfacecolor=color, markeredgecolor=color, markeredgewidth=.8,
                           label=regime_labels[regime])
            )

        ax.set_xticks(centers, scope_labels)
        ax.set_xlabel("Intervention direction", labelpad=3)
        ax.set_ylabel(r"Composition gap $E(J)-U(J)$", labelpad=4)
        ax.set_xlim(.50, 3.50)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if handles:
            ax.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(.5, 1.005),
                ncol=2,
                handletextpad=.45,
                columnspacing=1.2,
                borderaxespad=0,
                frameon=False,
            )
        fig.subplots_adjust(left=.145, right=.995, bottom=.17, top=.84)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def plot_gap(frame: pd.DataFrame, regime: str, path: Path) -> None:
    work = frame.loc[
        frame["replacement_regime"].eq(regime) & frame["rq2_evaluable"]
    ].copy()
    if work.empty:
        return
    work = work.sort_values("Delta_comp", kind="mergesort").reset_index(drop=True)
    labels = [f"{t} · {m} · {p}" for t, m, p in zip(work.task_label, work.model_label, work.phase)]
    y = np.arange(len(work))
    fig_h = max(3.2, 0.18 * len(work) + 1.0)
    fig, ax = plt.subplots(figsize=(6.8, fig_h))
    ax.axvline(0.0, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.scatter(work["Delta_comp"], y, s=18)
    ax.set_yticks(y, labels, fontsize=6.5)
    ax.set_xlabel(r"Composition gap $E(J)-U(J)$")
    ax.set_title("Mean-donor replacement" if regime == "mean-donor" else "Mean replacement (including mean-positional)")
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_latex_summary(summary: pd.DataFrame, path: Path) -> None:
    work = summary.loc[summary["population"].eq("all-evaluable")].copy()
    lines = [
        r"\begin{table}[htb]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Replacement & $n$ & $E<U$ & $E>U$ & Median $E-U$ & Pearson $r$ & $p$ \\", r"\midrule",
    ]
    for _, row in work.iterrows():
        label = "mean-donor" if row.replacement_regime == "mean-donor" else r"mean$^{\dagger}$"
        lines.append(
            f"{label} & {int(row.n)} & {int(row.n_negative_gap)} & {int(row.n_positive_gap)} & "
            f"{row.median_gap:+.3f} & {row.pearson_r:.3f} & {row.pearson_p:.3g} \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{RQ2 aggregate composition by replacement regime. The correlation is between singleton-union reach $U(J)$ and the composition gap $E(J)-U(J)$. $^{\dagger}$Mean includes mean-positional large-model runs. The regimes are reported separately and are not pooled.}",
        r"\label{tab:rq2-regime-summary}", r"\end{table}", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = Path(args.catalogue_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = prepare(pd.read_csv(source))
    frame.to_csv(out_dir / "rq2_settings_by_replacement_regime.csv", index=False)

    summaries = []
    for regime in ("mean-donor", "mean"):
        summaries.append(summarize(frame, regime=regime, checkpoint_free=False))
        summaries.append(summarize(frame, regime=regime, checkpoint_free=True))
    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "rq2_regime_summary.csv", index=False)
    (out_dir / "rq2_regime_summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=True), encoding="utf-8"
    )

    if args.paper_figures_dir:
        fig_dir = Path(args.paper_figures_dir).expanduser().resolve()
        plot_gap(frame, "mean-donor", fig_dir / "fig3a_composition_gap_all_settings.pdf")
        plot_compact_main(frame, fig_dir / "rq2_composition_boxplots.pdf")
        plot_gap(frame, "mean", fig_dir / "fig3s4_composition_gap_mean.pdf")
    if args.paper_tables_dir:
        write_latex_summary(summary, Path(args.paper_tables_dir).expanduser().resolve() / "rq2_regime_summary.tex")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
