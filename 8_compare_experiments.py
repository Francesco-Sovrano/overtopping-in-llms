import argparse
import math
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from math import ceil
from itertools import combinations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, ks_2samp, combine_pvalues

# ------------------------- run naming map -------------------------
NAME_MAP = {
    "rule_split-spectral_sample-decode_only-agonist_neurons-fast-spectral_anchor":
        "Rule split w/ Spectral anchoring plan",
    "rule_split-spectral_sample-decode_only-agonist_neurons-fast-random_anchor":
        "Rule split w/ Random anchoring plan",
    "rule_split-spectral_sample-decode_only-agonist_neurons-fast-spectral_anchor-fake_targets":
        "Fake rule split w/ Spectral anchoring plan",
    "spectral_split-decode_only-agonist_neurons-fast-random_anchor":
        "Spectral split",
    "rule_split-spectral_sample-decode_only-agonist_neurons-spectral_anchor": 
        "Rule split w/ Bruteforce search",

    "rule_split-spectral_sample-agonist_neurons-fast-spectral_anchor":
        "Rule split w/ Spectral anchoring plan",
    "rule_split-spectral_sample-agonist_neurons-fast-random_anchor":
        "Rule split w/ Random anchoring plan",
    "rule_split-spectral_sample-agonist_neurons-fast-spectral_anchor-fake_targets":
        "Fake rule split w/ Spectral anchoring plan",
    "spectral_split-agonist_neurons-fast-random_anchor":
        "Spectral split",
    "rule_split-spectral_sample-agonist_neurons-spectral_anchor": 
        "Rule split w/ Bruteforce search",
}
NAME_ORDER = list(NAME_MAP.values())
NAME_ORDER_INDEX = {v: i for i, v in enumerate(NAME_ORDER)}
ALLOWED_RUN_NAMES = set(NAME_MAP.keys())
ALLOWED_RUN_LABELS = set(NAME_MAP.values())

def filter_to_name_map_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose raw run name and mapped label are both canonical."""
    if df.empty:
        return df
    out = df.copy()
    if "stats_dirname" in out.columns:
        out = out[out["stats_dirname"].isin(ALLOWED_RUN_NAMES)]
    if "stats_label" in out.columns:
        out = out[out["stats_label"].isin(ALLOWED_RUN_LABELS)]
    return out.reset_index(drop=True)

# ------------------------- rule-metric distribution compare -------------------------

RULE_METRICS = [
    ("MCC", "MCC"),
    ("F1", "F1"),
    ("BalancedAcc", "Balanced accuracy"),
    ("balanced_accuracy_x_dataset_coverage", "Balanced accuracy × Dataset coverage"),
    # ("rule_len_atoms", "Rule length (atoms)"),
]

# ------------------------- helpers -------------------------

def metric_direction(col: str) -> int:
    # +1 means higher is better; -1 means lower is better
    if col in ("rule_len_atoms",):
        return -1
    return +1

_BOOLISH_REPLACEMENTS = {
    True: 1.0, False: 0.0,
    "True": 1.0, "False": 0.0,
    "true": 1.0, "false": 0.0,
    "1": 1.0, "0": 0.0,
}

def safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def norm_metric_name(name: str) -> str:
    n = str(name).strip().lower()
    if n == "mcc": return "MCC"
    if n in ("f1", "f1score"): return "F1"
    if n in ("balanced_accuracy", "balancedacc", "bal_acc"): return "BalancedAcc"
    if n == "tpr": return "TPR"
    if n == "tnr": return "TNR"
    return name

def make_thresholds(lo: float, hi: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("thr_step must be > 0")
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 10) for i in range(n)]

def resolve_stats_path(stats_path: str):
    """
    Accepts:
      - a stats/ directory
      - a parent directory containing stats/
      - a .zip containing a stats/ tree
    Returns: (Path, tmp_dir_to_cleanup_or_None)
    """
    stats_path = Path(stats_path).expanduser().resolve()
    if stats_path.is_file() and stats_path.suffix.lower() == ".zip":
        tmp = tempfile.mkdtemp(prefix="threshold_sweep_stats_")
        with zipfile.ZipFile(stats_path) as zf:
            zf.extractall(tmp)
        extracted = Path(tmp)
        if (extracted / "stats").is_dir():
            return (extracted / "stats"), tmp
        return extracted, tmp

    if stats_path.is_dir():
        if (stats_path / "stats").is_dir():
            return (stats_path / "stats"), None
        return stats_path, None

    raise FileNotFoundError(f"stats_path not found or unsupported: {stats_path}")

def discover_runs(stats_root: Path, run_name_regex: str | None = None) -> list[Path]:
    runs = []
    rx = re.compile(run_name_regex) if run_name_regex else None
    for p in stats_root.iterdir():
        if not p.is_dir(): 
            continue
        name = p.name
        if name.startswith(".") or name == "__MACOSX":
            continue
        if name not in NAME_MAP:
            continue
        if rx and (rx.search(name) is None):
            continue
        runs.append(p)
    return sorted(runs, key=lambda x: x.name)

def looks_like_stats_dir(p: Path) -> bool:
    # Heuristic: must contain at least one run dir that contains scores.csv or rule_combo_metrics_best_per_neuron.csv
    if not p.is_dir():
        return False
    for c in p.iterdir():
        if not c.is_dir() or c.name.startswith(".") or c.name == "__MACOSX":
            continue
        if (c / "scores.csv").exists() or (c / "rule_combo_metrics_best_per_neuron.csv").exists():
            return True
    return False

def discover_stats_roots(maybe_root: Path) -> list[Path]:
    """
    If maybe_root is a stats dir -> [maybe_root]
    Else recursively find subdirs named 'stats' that look like stats dirs.
    """
    if looks_like_stats_dir(maybe_root):
        return [maybe_root]

    found = []
    for p in maybe_root.rglob("stats"):
        if looks_like_stats_dir(p):
            found.append(p)
    # stable order
    found = sorted(set(found), key=lambda x: str(x))
    return found

def parse_task_llm_from_stats_root(stats_root: Path) -> tuple[str, str]:
    """
    Best-effort inference from common layout:
      .../data/<task>/<org>/<model>/rule_extraction_results/.../stats
    Returns (task, llm_label)
    """
    parts = list(stats_root.parts)
    task = "unknown_task"
    llm = "unknown_llm"
    # try to anchor on "data" or "data copy"
    for anchor in ("data", "data copy"):
        if anchor in parts:
            i = parts.index(anchor)
            if i + 3 < len(parts):
                task = parts[i + 1]
                org = parts[i + 2]
                model = parts[i + 3]
                llm = f"{org}/{model}"
            break
    return task, llm

# ------------------------- union flip coverage -------------------------

def union_flip_stats(scores_df: pd.DataFrame, cols: list[str]) -> dict:
    """
    Computes unique flipped datapoints (union across cols) and the evaluated-row denominator.
    """
    if not cols:
        return {"union_count": 0, "n_eval": 0, "pct": float("nan")}

    sub = scores_df.reindex(columns=cols)  # missing cols become all-NaN
    if sub.shape[1] == 0:
        return {"union_count": 0, "n_eval": 0, "pct": float("nan")}

    eval_mask = sub.notna().any(axis=1).to_numpy()
    n_eval = int(eval_mask.sum())
    if n_eval == 0:
        return {"union_count": 0, "n_eval": 0, "pct": float("nan")}

    sub_num = sub.replace(_BOOLISH_REPLACEMENTS).infer_objects()
    sub_num = sub_num.apply(pd.to_numeric, errors="coerce")
    mat = (sub_num.fillna(0.0).to_numpy() != 0.0)
    union_count = int(mat.any(axis=1).sum())

    pct = 100.0 * union_count / n_eval if n_eval else float("nan")
    return {"union_count": union_count, "n_eval": n_eval, "pct": pct}

# ------------------------- rule-metric collection -------------------------

def collect_rule_metrics_best(runs: list[Path]) -> pd.DataFrame:
    blocks = []
    for run_dir in runs:
        run_name = run_dir.name
        if run_name not in NAME_MAP:
            continue
        run_label = NAME_MAP[run_name]
        fp = run_dir / "rule_combo_metrics_best_per_neuron.csv"
        if not fp.exists():
            continue
        df = safe_read_csv(fp)
        df["stats_dirname"] = run_name
        df["stats_label"] = run_label
        blocks.append(df)
    if not blocks:
        return pd.DataFrame()
    return filter_to_name_map_runs(pd.concat(blocks, axis=0, ignore_index=True))

def add_derived_rule_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "dataset_coverage" not in df.columns or "BalancedAcc" not in df.columns:
        return df

    out = df.copy()
    cov = pd.to_numeric(out["dataset_coverage"], errors="coerce")
    bal_acc = pd.to_numeric(out["BalancedAcc"], errors="coerce")
    out["balanced_accuracy_x_dataset_coverage"] = (bal_acc * (np.maximum(cov, 1 - cov))).astype(float)
    return out

def write_rule_metrics_summary(df_all: pd.DataFrame, out_csv: Path, rule_len_cap: int = 30):
    rows = []
    for (col, _label) in RULE_METRICS:
        if col not in df_all.columns:
            continue
        for run in df_all["stats_label"].unique().tolist():
            v = pd.to_numeric(df_all.loc[df_all["stats_label"] == run, col], errors="coerce").to_numpy()
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            rows.append({
                "stats_label": run,
                "metric": col,
                "n": int(v.size),
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "q10": float(np.quantile(v, 0.10)),
                "q50": float(np.quantile(v, 0.50)),
                "q90": float(np.quantile(v, 0.90)),
                "q95": float(np.quantile(v, 0.95)),
                "p_ge_0.80": float(np.mean(v >= 0.80)),
                "p_ge_0.90": float(np.mean(v >= 0.90)),
                "tail_mean_ge_0.90": float(np.mean(v[v >= 0.90])) if np.any(v >= 0.90) else float("nan"),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["stats_order"] = out["stats_label"].map(NAME_ORDER_INDEX).fillna(10**9).astype(int)
        out = out.sort_values(["stats_order", "stats_label", "metric"]).drop(columns=["stats_order"])
    out.to_csv(out_csv, index=False)

# ------------------------- stats + corrections -------------------------

def holm_adjust(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for k, idx in enumerate(order):
        val = (m - k) * p[idx]
        val = max(val, prev)
        prev = val
        adj[idx] = min(val, 1.0)
    return adj.tolist()

def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "ns"

def choose_best_run(values_by_run: dict[str, np.ndarray], mode: str = "mean") -> str | None:
    best = None
    best_score = -np.inf
    for run, v in values_by_run.items():
        if v.size == 0:
            continue
        if mode == "median":
            score = float(np.median(v))
        elif mode.startswith("tail@"):
            t = float(mode.split("@", 1)[1])
            score = float(np.mean(v >= t))
        else:
            score = float(np.mean(v))
        if score > best_score:
            best_score = score
            best = run
    return best

# ------------------------- plotting -------------------------

def _ecdf_xy(values: np.ndarray):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None, None
    v.sort()
    y = (np.arange(1, v.size + 1, dtype=float) / float(v.size))
    return v, y

def plot_rule_metrics_ecdf_grid(df_all: pd.DataFrame, out_path: Path, rule_len_cap: int = 30,
                                annotate_sig: bool = True,
                                best_mode: str = "tail@0.9",  # "mean" | "median" | "tail@0.9"
                                test_kind: str = "mwu",        # "mwu" or "ks"
                                corr_within_subplot: bool = True,
                                title_suffix: str = ""):
    """ECDF grid for rule-metric distributions.

    Matches the original Script-8 reporting style:
    - each subplot has a compact legend showing significance vs the best run (using the SAME colored lines),
    - a global legend at the bottom maps colors to run names.
    """
    if df_all.empty:
        return

    try:
        from scipy.stats import mannwhitneyu, ks_2samp
    except Exception:
        mannwhitneyu = None
        ks_2samp = None

    ncols = 2
    nrows = int(ceil(len(RULE_METRICS) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.9 * ncols, 3.5 * nrows))
    axes = np.asarray(axes).reshape(-1)

    runs = [r for r in NAME_ORDER if r in df_all["stats_label"].unique().tolist()]
    runs += [r for r in df_all["stats_label"].unique().tolist() if r not in runs]

    for i, (col, label) in enumerate(RULE_METRICS):
        ax = axes[i]
        if col not in df_all.columns:
            ax.set_axis_off()
            continue

        # collect arrays + keep plotted line objects per run (so we can reuse their colors in legends)
        values_by_run: Dict[str, np.ndarray] = {}
        line_by_run: Dict[str, Any] = {}

        for run in runs:
            v = pd.to_numeric(df_all.loc[df_all["stats_label"] == run, col], errors="coerce").to_numpy()
            v = v[np.isfinite(v)]
            if col == "rule_len_atoms":
                v = np.clip(v, 0, float(rule_len_cap))
            values_by_run[run] = v

        # plot
        for run in runs:
            v = values_by_run.get(run, np.array([], dtype=float))
            x, y = _ecdf_xy(v)
            if x is None:
                continue
            ln, = ax.plot(x, y, linewidth=1.6, label=run)
            line_by_run[run] = ln

        ax.set_title(label + title_suffix, fontsize=10)
        ax.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.6)

        is_left = (i % ncols == 0)
        is_bottom = (i >= (nrows - 1) * ncols)
        if is_left:
            ax.set_ylabel("ECDF", fontsize=9)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        if is_bottom:
            xlab = label if col != "rule_len_atoms" else f"{label} (≤{rule_len_cap})"
            ax.set_xlabel(xlab, fontsize=9)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        ax.tick_params(labelsize=8)

        # ---------- compact significance legend (colors + stars) ----------
        if annotate_sig:
            if (test_kind == "mwu" and mannwhitneyu is None) or (test_kind == "ks" and ks_2samp is None):
                pass
            else:
                direction = metric_direction(col)
                vb = {r: (direction * v) for r, v in values_by_run.items()}

                best = choose_best_run(vb, mode=best_mode)
                if best is not None and best in line_by_run:
                    comps = [r for r in runs if r != best and r in line_by_run
                             and vb.get(r, np.array([])).size > 0 and vb.get(best, np.array([])).size > 0]

                    pvals = []
                    comp_runs = []
                    for r in comps:
                        a = vb[best]
                        b = vb[r]
                        if test_kind == "mwu":
                            _, p = mannwhitneyu(a, b, alternative="greater")
                        else:  # "ks"
                            # after direction transform, "better" means "smaller" in original => "greater" in vb
                            # ECDF test: alternative "less" means vb(best) tends to be larger
                            _, p = ks_2samp(a, b, alternative="less")
                        pvals.append(float(p))
                        comp_runs.append(r)

                    padj = holm_adjust(pvals) if (corr_within_subplot and pvals) else pvals

                    stars_map = {best: "best"}
                    for r, p in zip(comp_runs, padj):
                        stars_map[r] = p_to_stars(p)  # "***", "**", "*", "ns"

                    # legend with SAME colored lines, but labels are just stars/best
                    handles = [line_by_run[r] for r in runs if r in line_by_run]
                    labels_ = [stars_map.get(r, "") for r in runs if r in line_by_run]

                    ax.legend(
                        handles=handles,
                        labels=labels_,
                        title="sig vs best",
                        loc="lower right",
                        fontsize=7,
                        title_fontsize=7,
                        frameon=True,
                        handlelength=2.0,
                        borderpad=0.25,
                        labelspacing=0.15,
                        handletextpad=0.6,
                    )

    for j in range(len(RULE_METRICS), len(axes)):
        axes[j].set_axis_off()

    # global legend (run names) stays compact at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.01),
                   ncol=2, fontsize=8, frameon=True, handlelength=2.0, columnspacing=1.2)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_lines(df: pd.DataFrame, x_col: str, y_col: str, group_col: str, out_path: Path,
               title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(10.5, 6.0))
    groups = sorted(df[group_col].unique().tolist(), key=lambda g: (NAME_ORDER_INDEX.get(g, 10**9), str(g)))
    for g in groups:
        dfg = df[df[group_col] == g].sort_values(x_col)
        plt.plot(dfg[x_col].to_numpy(), dfg[y_col].to_numpy(), marker="o", linewidth=1.6, label=g)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

def plot_lines_mean_ci(df: pd.DataFrame, x_col: str, y_col: str, group_col: str,
                       out_path: Path, title: str, xlabel: str, ylabel: str,
                       ci: float = 1.96):
    """
    df must include columns: x_col, y_col, group_col, and 'llm' to aggregate over.
    Plots mean across LLMs with +/- ci * SEM shaded region.
    """
    if "llm" not in df.columns:
        raise ValueError("plot_lines_mean_ci requires a 'llm' column in df")

    plt.figure(figsize=(10.5, 6.0))
    groups = sorted(df[group_col].unique().tolist(), key=lambda g: (NAME_ORDER_INDEX.get(g, 10**9), str(g)))

    for g in groups:
        sub = df[df[group_col] == g]
        agg = sub.groupby([x_col]).agg(
            mean=(y_col, "mean"),
            std=(y_col, "std"),
            n=("llm", "nunique"),
        ).reset_index().sort_values(x_col)
        x = agg[x_col].to_numpy()
        mean = agg["mean"].to_numpy()
        n = np.maximum(agg["n"].to_numpy(), 1.0)
        sem = (agg["std"].to_numpy() / np.sqrt(n))
        (ln,) = plt.plot(x, mean, marker="o", linewidth=1.6, label=g)
        # Shade CI using the same color as the line (default cycle)
        plt.fill_between(x, mean - ci * sem, mean + ci * sem, alpha=0.18)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

# ------------------------- tests (single stats dir) -------------------------

def pairwise_metric_tests(df_all: pd.DataFrame, out_csv: Path, test_kind: str = "mwu"):
    """
    Pairwise tests across runs, metric-by-metric (pooled).
    """
    rows = []
    runs = df_all["stats_label"].unique().tolist()
    for (col, _label) in RULE_METRICS:
        if col not in df_all.columns:
            continue
        for a, b in combinations(runs, 2):
            va = pd.to_numeric(df_all.loc[df_all["stats_label"] == a, col], errors="coerce").to_numpy()
            vb = pd.to_numeric(df_all.loc[df_all["stats_label"] == b, col], errors="coerce").to_numpy()
            va = va[np.isfinite(va)]
            vb = vb[np.isfinite(vb)]
            if va.size == 0 or vb.size == 0:
                continue
            if test_kind == "ks":
                stat, p = ks_2samp(va, vb)
                test = "KS"
            else:
                stat, p = mannwhitneyu(va, vb, alternative="two-sided")
                test = "MWU"
            rows.append({
                "metric": col, "test": test,
                "A": a, "B": b,
                "n_A": int(va.size), "n_B": int(vb.size),
                "stat": float(stat), "p": float(p),
                "median_A": float(np.median(va)), "median_B": float(np.median(vb)),
                "median_diff": float(np.median(va) - np.median(vb)),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm_within_metric"] = np.nan
        for metric in out["metric"].unique():
            mask = out["metric"] == metric
            out.loc[mask, "p_holm_within_metric"] = holm_adjust(out.loc[mask, "p"].tolist())
        out["sig"] = out["p_holm_within_metric"].map(p_to_stars)
        out = out.sort_values(["metric", "p_holm_within_metric", "p"])
    out.to_csv(out_csv, index=False)

# ------------------------- tests (multi-llm, aggregated by task) -------------------------

def pairwise_metric_tests_by_task(df_all: pd.DataFrame, out_csv: Path, test_kind: str = "mwu"):
    """
    df_all must include columns: task, llm, stats_label, and metric columns.
    For each (task, metric, runA, runB):
      - compute a per-llm p-value
      - combine across llms with Fisher
      - Holm-adjust within (task, metric) across pairs
    """
    rows = []
    for task in sorted(df_all["task"].unique().tolist()):
        dft = df_all[df_all["task"] == task]
        runs = dft["stats_label"].unique().tolist()
        llms = dft["llm"].unique().tolist()
        for (col, _label) in RULE_METRICS:
            if col not in dft.columns:
                continue
            for a, b in combinations(runs, 2):
                pvals = []
                n_used = 0
                for llm in llms:
                    va = pd.to_numeric(dft[(dft["llm"] == llm) & (dft["stats_label"] == a)][col], errors="coerce").to_numpy()
                    vb = pd.to_numeric(dft[(dft["llm"] == llm) & (dft["stats_label"] == b)][col], errors="coerce").to_numpy()
                    va = va[np.isfinite(va)]
                    vb = vb[np.isfinite(vb)]
                    if va.size == 0 or vb.size == 0:
                        continue
                    n_used += 1
                    if test_kind == "ks":
                        _, p = ks_2samp(va, vb)
                        test = "KS"
                    else:
                        _, p = mannwhitneyu(va, vb, alternative="two-sided")
                        test = "MWU"
                    pvals.append(float(p))
                if not pvals:
                    continue
                _, p_comb = combine_pvalues(pvals, method="fisher")
                # pooled medians for interpretability
                va_all = pd.to_numeric(dft[dft["stats_label"] == a][col], errors="coerce").to_numpy()
                vb_all = pd.to_numeric(dft[dft["stats_label"] == b][col], errors="coerce").to_numpy()
                va_all = va_all[np.isfinite(va_all)]
                vb_all = vb_all[np.isfinite(vb_all)]
                rows.append({
                    "task": task,
                    "metric": col,
                    "test": test,
                    "A": a, "B": b,
                    "n_llm": int(n_used),
                    "p_fisher": float(p_comb),
                    "median_A_pooled": float(np.median(va_all)) if va_all.size else float("nan"),
                    "median_B_pooled": float(np.median(vb_all)) if vb_all.size else float("nan"),
                    "median_diff_pooled": float(np.median(va_all) - np.median(vb_all)) if (va_all.size and vb_all.size) else float("nan"),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm_within_task_metric"] = np.nan
        for task in out["task"].unique():
            for metric in out[out["task"] == task]["metric"].unique():
                mask = (out["task"] == task) & (out["metric"] == metric)
                out.loc[mask, "p_holm_within_task_metric"] = holm_adjust(out.loc[mask, "p_fisher"].tolist())
        out["sig"] = out["p_holm_within_task_metric"].map(p_to_stars)
        out = out.sort_values(["task", "metric", "p_holm_within_task_metric", "p_fisher"])
    out.to_csv(out_csv, index=False)

# ------------------------- per-stats-folder pipeline -------------------------

def threshold_sweep_for_stats_dir(stats_root: Path, out_dir: Path, args) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(stats_root, run_name_regex=args.run_name_regex)
    if not runs:
        return pd.DataFrame()

    metric_name = norm_metric_name(args.rule_quality_metric)
    mcol = metric_name

    thresholds = make_thresholds(args.thr_min, args.thr_max, args.thr_step)

    rows = []
    for run_dir in runs:
        run_name = run_dir.name
        if run_name not in NAME_MAP:
            continue
        run_label = NAME_MAP[run_name]

        best_path = run_dir / "rule_combo_metrics_best_per_neuron.csv"
        scores_path = run_dir / "scores.csv"
        if not best_path.exists() or not scores_path.exists():
            continue

        df_best = safe_read_csv(best_path)
        if mcol not in df_best.columns:
            mcol_local = "MCC"
        else:
            mcol_local = mcol

        df_best[mcol_local] = pd.to_numeric(df_best[mcol_local], errors="coerce")
        df_best = df_best.dropna(subset=["layer_key", "neuron_id"]).copy()
        df_best["layer_key"] = df_best["layer_key"].astype(str)
        df_best["neuron_id"] = df_best["neuron_id"].astype(int)

        scores_df = safe_read_csv(scores_path)

        for thr in thresholds:
            hq = df_best[df_best[mcol_local] >= float(thr)].drop_duplicates(subset=["layer_key", "neuron_id"])
            neurons = list(zip(hq["layer_key"].tolist(), hq["neuron_id"].tolist()))
            n_hq = int(len(neurons))

            cols_c2i = [f"flip_c2i_{lk}_{nid}" for lk, nid in neurons]
            cols_i2c = [f"flip_i2c_{lk}_{nid}" for lk, nid in neurons]

            st_c2i = union_flip_stats(scores_df, cols_c2i)
            st_i2c = union_flip_stats(scores_df, cols_i2c)

            rows.append({
                "stats_dirname": run_name,
                "stats_label": run_label,
                "threshold": float(thr),
                "rule_quality_metric": metric_name,
                "metric_col": mcol_local,
                "n_high_quality_neurons": n_hq,
                "union_c2i_pct": st_c2i["pct"],
                "union_i2c_pct": st_i2c["pct"],
                "union_c2i_count": st_c2i["union_count"],
                "union_i2c_count": st_i2c["union_count"],
                "eval_c2i_rows": st_c2i["n_eval"],
                "eval_i2c_rows": st_i2c["n_eval"],
                "scores_rows": int(len(scores_df)),
            })

    df_out = filter_to_name_map_runs(pd.DataFrame(rows))
    if df_out.empty:
        return df_out

    df_out["stats_order"] = df_out["stats_label"].map(NAME_ORDER_INDEX).fillna(10**9).astype(int)
    df_out = (
        df_out.sort_values(["stats_order", "stats_label", "threshold"])
              .drop(columns=["stats_order"])
              .reset_index(drop=True)
    )

    df_out.to_csv(out_dir / "threshold_sweep_summary.csv", index=False)

    plot_lines(
        df_out, "threshold", "n_high_quality_neurons", "stats_label",
        out_dir / "plot_high_quality_neurons.pdf",
        title=f"High-quality (>= t) neurons vs {metric_name} thresholds",
        xlabel=f"{metric_name} threshold (t)",
        ylabel="# high-quality (>= t) neurons",
    )
    plot_lines(
        df_out, "threshold", "union_c2i_pct", "stats_label",
        out_dir / "plot_union_flip_c2i_pct.pdf",
        title=f"Union flip coverage (correct→incorrect) vs {metric_name} thresholds",
        xlabel=f"{metric_name} threshold (t)",
        ylabel="Flipped datapoints (% of evaluated rows)",
    )
    plot_lines(
        df_out, "threshold", "union_i2c_pct", "stats_label",
        out_dir / "plot_union_flip_i2c_pct.pdf",
        title=f"Union flip coverage (incorrect→correct) vs {metric_name} thresholds",
        xlabel=f"{metric_name} threshold (t)",
        ylabel="Flipped datapoints (% of evaluated rows)",
    )

    return df_out

def rule_metrics_for_stats_dir(stats_root: Path, out_dir: Path, args) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(stats_root, run_name_regex=args.run_name_regex)
    df_rm = collect_rule_metrics_best(runs)
    df_rm = filter_to_name_map_runs(add_derived_rule_metrics(df_rm))
    if df_rm.empty:
        return df_rm

    out_csv = out_dir / "compare_rule_metrics_distributions_summary.csv"
    out_pdf = out_dir / "compare_rule_metrics_distributions_ecdf.pdf"
    write_rule_metrics_summary(df_rm, out_csv, rule_len_cap=int(args.rule_length_cap))
    plot_rule_metrics_ecdf_grid(
        df_rm, out_pdf,
        annotate_sig=True, best_mode=args.best_mode,
        test_kind="mwu", corr_within_subplot=True,
    )
    pairwise_metric_tests(df_rm, out_dir / "compare_rule_metrics_distributions_tests.csv", test_kind="mwu")
    return df_rm

# ------------------------- multi-llm aggregation (by task) -------------------------

def write_aggregates_by_task(thr_all: pd.DataFrame, rm_all: pd.DataFrame, out_dir: Path, args):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- threshold sweep aggregates (mean +/- CI across LLMs) ----
    if not thr_all.empty:
        thr_all = filter_to_name_map_runs(thr_all.copy())
        thr_all["net_acc_pp"] = thr_all["union_i2c_pct"] - thr_all["union_c2i_pct"]
        thr_all.to_csv(out_dir / "threshold_sweep_all_tasks_all_llms.csv", index=False)

        for task in sorted(thr_all["task"].unique().tolist()):
            dft = thr_all[thr_all["task"] == task].copy()
            task_dir = out_dir / task
            task_dir.mkdir(parents=True, exist_ok=True)

            dft.to_csv(task_dir / "threshold_sweep_all_llms.csv", index=False)

            metric_name = norm_metric_name(args.rule_quality_metric)

            plot_lines_mean_ci(
                dft, "threshold", "n_high_quality_neurons", "stats_label",
                task_dir / "plot_high_quality_neurons_by_task.pdf",
                title=f"[{task}] High-quality neurons vs {metric_name} thresholds (mean ± 95% CI over LLMs)",
                xlabel=f"{metric_name} threshold (t)",
                ylabel="# high-quality neurons",
            )
            plot_lines_mean_ci(
                dft, "threshold", "union_c2i_pct", "stats_label",
                task_dir / "plot_union_flip_c2i_pct_by_task.pdf",
                title=f"[{task}] Union flip (correct→incorrect) vs {metric_name} thresholds (mean ± 95% CI over LLMs)",
                xlabel=f"{metric_name} threshold (t)",
                ylabel="Flipped datapoints (% of evaluated rows)",
            )
            plot_lines_mean_ci(
                dft, "threshold", "union_i2c_pct", "stats_label",
                task_dir / "plot_union_flip_i2c_pct_by_task.pdf",
                title=f"[{task}] Union flip (incorrect→correct) vs {metric_name} thresholds (mean ± 95% CI over LLMs)",
                xlabel=f"{metric_name} threshold (t)",
                ylabel="Flipped datapoints (% of evaluated rows)",
            )
            plot_lines_mean_ci(
                dft, "threshold", "net_acc_pp", "stats_label",
                task_dir / "plot_net_accuracy_change_pp_by_task.pdf",
                title=f"[{task}] Net accuracy change (i2c − c2i) vs {metric_name} thresholds (mean ± 95% CI over LLMs)",
                xlabel=f"{metric_name} threshold (t)",
                ylabel="Net change (percentage points)",
            )

    # ---- rule-metric aggregates + tests by task ----
    if not rm_all.empty:
        rm_all = filter_to_name_map_runs(rm_all.copy())
        rm_all.to_csv(out_dir / "rule_metrics_best_per_neuron_all_tasks_all_llms.csv", index=False)

        # per-task ECDF plots (pooled across LLMs)
        for task in sorted(rm_all["task"].unique().tolist()):
            dft = rm_all[rm_all["task"] == task]
            task_dir = out_dir / task
            task_dir.mkdir(parents=True, exist_ok=True)

            write_rule_metrics_summary(dft, task_dir / "compare_rule_metrics_distributions_summary_by_task.csv")
            plot_rule_metrics_ecdf_grid(
                dft, task_dir / "compare_rule_metrics_distributions_ecdf_by_task.pdf",
                annotate_sig=True, best_mode=args.best_mode,
                test_kind="mwu", corr_within_subplot=True,
                title_suffix=f" [{task}]",
            )

        pairwise_metric_tests_by_task(rm_all, out_dir / "compare_rule_metrics_distributions_tests_combined_by_task.csv", test_kind="mwu")

def filter_complete_task_llm_sets(df: pd.DataFrame, run_col: str = "stats_label"):
    """
    Keep only (task, llm) experiment sets whose number of distinct runs
    matches the maximum for that task.
    """
    if df.empty:
        coverage = pd.DataFrame(columns=["task", "llm", "n_runs", "max_runs_for_task", "is_complete"])
        return df.copy(), coverage

    coverage = (
        df.groupby(["task", "llm"])[run_col]
          .nunique()
          .rename("n_runs")
          .reset_index()
    )
    coverage["max_runs_for_task"] = coverage.groupby("task")["n_runs"].transform("max")
    coverage["is_complete"] = coverage["n_runs"] == coverage["max_runs_for_task"]

    keep = coverage.loc[coverage["is_complete"], ["task", "llm"]].drop_duplicates()
    out = df.merge(keep, on=["task", "llm"], how="inner")

    return out, coverage.sort_values(["task", "llm"]).reset_index(drop=True)

# ------------------------- CLI -------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats_path", type=str, required=True,
                    help="Path to a stats/ directory, OR a parent directory containing many stats/ dirs, OR a .zip.")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Where to write CSVs and plots. If multiple stats folders are found, per-task aggregates are written here.")
    ap.add_argument("--rule_quality_metric", type=str, default="mcc",
                    choices=["mcc", "f1", "balanced_accuracy", "tpr", "tnr"])
    ap.add_argument("--thr_min", type=float, default=0.85)
    ap.add_argument("--thr_max", type=float, default=0.99)
    ap.add_argument("--thr_step", type=float, default=0.01)
    ap.add_argument("--best_mode", type=str, default="tail@0.9")
    ap.add_argument("--rule_length_cap", type=int, default=30)
    ap.add_argument("--run_name_regex", type=str, default=None,
                    help="Optional regex to filter which run folders to include.")
    return ap.parse_args()

def main():
    args = parse_args()

    stats_root_or_parent, tmp_to_cleanup = resolve_stats_path(args.stats_path)
    stats_roots = discover_stats_roots(stats_root_or_parent)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    multi = len(stats_roots) > 1

    thr_all = []
    rm_all = []

    for sr in stats_roots:
        task, llm = parse_task_llm_from_stats_root(sr)

        # Decide where per-stats-folder outputs go:
        # - if user points directly at a stats dir (single), preserve behavior (out_dir)
        # - else, write into out_dir/<task>/<llm>/leaf
        if (not multi) and (Path(args.out_dir).resolve() == out_dir):
            leaf_out = out_dir
        else:
            leaf_out = out_dir / task / llm / "leaf_stats_dir_outputs"

        df_thr = threshold_sweep_for_stats_dir(sr, leaf_out, args)
        if not df_thr.empty:
            df_thr = df_thr.copy()
            df_thr["task"] = task
            df_thr["llm"] = llm
            thr_all.append(df_thr)

        df_rm = rule_metrics_for_stats_dir(sr, leaf_out, args)
        if not df_rm.empty:
            df_rm = df_rm.copy()
            df_rm["task"] = task
            df_rm["llm"] = llm
            rm_all.append(df_rm)

    thr_all_df = filter_to_name_map_runs(pd.concat(thr_all, ignore_index=True)) if thr_all else pd.DataFrame()
    rm_all_df = filter_to_name_map_runs(pd.concat(rm_all, ignore_index=True)) if rm_all else pd.DataFrame()

    thr_all_df, thr_coverage = filter_complete_task_llm_sets(thr_all_df, run_col="stats_label")
    rm_all_df, rm_coverage = filter_complete_task_llm_sets(rm_all_df, run_col="stats_label")

    if multi:
        agg_dir = out_dir / "aggregated_by_task"
        agg_dir.mkdir(parents=True, exist_ok=True)
        thr_coverage.to_csv(agg_dir / "threshold_sweep_coverage.csv", index=False)
        rm_coverage.to_csv(agg_dir / "rule_metrics_coverage.csv", index=False)
        write_aggregates_by_task(thr_all_df, rm_all_df, agg_dir, args)
    
    if tmp_to_cleanup:
        shutil.rmtree(tmp_to_cleanup, ignore_errors=True)

if __name__ == "__main__":
    main()
