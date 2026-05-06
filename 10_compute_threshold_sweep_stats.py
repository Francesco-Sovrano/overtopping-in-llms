#!/usr/bin/env python3
"""
Compute the E1/E3 threshold-sweep table + statistical tests from:
  threshold_sweep_all_tasks_all_llms.csv

Target metric (default): n_high_quality_neurons (count of neurons with MCC >= threshold).

IMPORTANT: The table aggregates by SUM (not mean) over available (task,llm) runs.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


LABEL_MAP = {
    "Rule split w/ Spectral anchoring plan": "Rule split + Spectral plan",
    "Rule split w/ Random anchoring plan": "Rule split + Random plan",
    "Fake rule split w/ Spectral anchoring plan": "Fake rule split + Spectral plan",
    "Spectral split": "Spectral split (no rule)",
    "Rule split w/ Bruteforce search": "Rule split + Bruteforce search",
}

DEFAULT_THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.99]


def _fmt(x: float) -> str:
    if pd.isna(x):
        return ""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.1f}"


def load_pivot(csv_path: str, metric: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available columns: {list(df.columns)}")

    df["method"] = df["stats_label"].map(LABEL_MAP).fillna(df["stats_label"])

    # One row per (task, llm, threshold); one column per method.
    # Use mean here only to safely collapse accidental duplicates; aggregation happens later.
    pivot = df.pivot_table(
        index=["task", "llm", "threshold"],
        columns="method",
        values=metric,
        aggfunc="mean",
    ).sort_index()

    return pivot


def coverage_report(pivot: pd.DataFrame) -> pd.DataFrame:
    """How many (task,llm) runs exist per method (ignoring thresholds)."""
    out: Dict[str, int] = {}
    for m in pivot.columns:
        s = pivot[m].groupby(["task", "llm"]).mean()
        out[m] = int(s.dropna().shape[0])
    return (
        pd.DataFrame.from_dict(out, orient="index", columns=["n_runs"])
        .sort_values("n_runs", ascending=False)
    )


def make_table_sum(
    pivot: pd.DataFrame,
    thresholds: Iterable[float],
    methods: Optional[List[str]] = None,
    agg: str = "available",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      totals: methods x thresholds table of SUMS over runs
      ns:     methods x thresholds table of N (runs contributing to the sum)

    agg:
      - "available": each method sums over its available runs at that threshold
      - "intersection": restrict to runs where *all* listed methods have values (per threshold)
    """
    thresholds = list(thresholds)
    if methods is None:
        methods = list(pivot.columns)
    else:
        missing = [m for m in methods if m not in pivot.columns]
        if missing:
            raise ValueError(f"Methods not found in data: {missing}")

    total_rows = []
    n_rows = []

    for t in thresholds:
        try:
            sub = pivot.xs(t, level="threshold")[methods]
        except KeyError:
            continue

        if agg == "intersection":
            sub = sub.dropna(axis=0, how="any")

        for m in methods:
            col = sub[m].dropna() if agg == "available" else sub[m]
            total_rows.append(
                {"method": m, "threshold": t, "value": float(col.sum()) if len(col) else np.nan}
            )
            n_rows.append({"method": m, "threshold": t, "value": int(len(col))})

    totals = (
        pd.DataFrame(total_rows)
        .pivot(index="method", columns="threshold", values="value")
        .reindex(methods)
    )
    ns = (
        pd.DataFrame(n_rows)
        .pivot(index="method", columns="threshold", values="value")
        .reindex(methods)
    )
    return totals, ns


def to_latex_tabular(totals: pd.DataFrame) -> str:
    """Produces only the tabular (no table environment)."""
    thresholds = list(totals.columns)
    header = "Method & " + " & ".join([f"$t={t:.2f}$" for t in thresholds]) + r" \\"
    lines = [
        r"\begin{tabular}{l" + "c" * len(thresholds) + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for method in totals.index:
        vals = []
        for t in thresholds:
            v = totals.loc[method, t]
            vals.append("" if pd.isna(v) else _fmt(v))
        lines.append(method + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def wilcoxon_on_diffs(d: np.ndarray, alternative: str) -> Dict[str, float]:
    d = np.asarray(d, dtype=float)
    total = int(d.size)
    d_nz = d[d != 0]
    nz = int(d_nz.size)
    if nz == 0:
        return {"n_total": total, "n_nonzero": 0, "p": np.nan, "stat": np.nan}
    stat, p = stats.wilcoxon(
        d_nz, alternative=alternative, zero_method="wilcox", correction=False, mode="auto"
    )
    return {"n_total": total, "n_nonzero": nz, "p": float(p), "stat": float(stat)}


def paired_tests(
    pivot: pd.DataFrame, a: str, b: str, alternative: str = "greater"
) -> Dict[str, Dict[str, float]]:
    """
    Two paired tests:
      - run-level: each (task,llm) is a unit; compare SUM over thresholds per run
      - point-level: each (task,llm,threshold) is a unit (NOT independent; use with care)
    """
    if a not in pivot.columns or b not in pivot.columns:
        raise ValueError(f"Missing methods for test: {a} or {b}")

    # run-level: sum over thresholds per (task,llm)
    ra = pivot[a].groupby(["task", "llm"]).sum()
    rb = pivot[b].groupby(["task", "llm"]).sum()
    run = pd.concat([ra, rb], axis=1, keys=[a, b]).dropna()
    d_run = (run[a] - run[b]).to_numpy()
    run_res = wilcoxon_on_diffs(d_run, alternative)

    # point-level: (task,llm,threshold) pairs
    pts = pivot[[a, b]].dropna()
    d_pts = (pts[a] - pts[b]).to_numpy()
    pt_res = wilcoxon_on_diffs(d_pts, alternative)

    def summarize(d: np.ndarray) -> Dict[str, float]:
        d = np.asarray(d, dtype=float)
        d = d[~np.isnan(d)]
        if d.size == 0:
            return {"median_diff": np.nan, "mean_diff": np.nan}
        return {"median_diff": float(np.median(d)), "mean_diff": float(np.mean(d))}

    run_res.update(summarize(d_run))
    pt_res.update(summarize(d_pts))
    return {"run_level": run_res, "point_level": pt_res}


def per_threshold_tests(
    pivot: pd.DataFrame, a: str, b: str, alternative: str = "greater"
) -> pd.DataFrame:
    rows = []
    for t, sub in pivot.groupby(level="threshold"):
        s = sub[[a, b]].dropna()
        if s.shape[0] == 0:
            continue
        d = (s[a] - s[b]).to_numpy()
        res = wilcoxon_on_diffs(d, alternative)
        res.update({"threshold": float(t), "median_diff": float(np.median(d))})
        rows.append(res)
    return pd.DataFrame(rows).sort_values("threshold")

def filter_complete_task_llm_sets(pivot: pd.DataFrame):
    """
    Keep only (task, llm) rows whose number of available methods matches
    the maximum observed for that task.
    """
    if pivot.empty:
        coverage = pd.DataFrame(columns=["task", "llm", "n_methods", "max_methods_for_task", "is_complete"])
        return pivot.copy(), coverage

    coverage = (
        pivot.notna()
             .groupby(["task", "llm"])
             .any()
             .sum(axis=1)
             .rename("n_methods")
             .reset_index()
    )
    coverage["max_methods_for_task"] = coverage.groupby("task")["n_methods"].transform("max")
    coverage["is_complete"] = coverage["n_methods"] == coverage["max_methods_for_task"]

    keep = coverage.loc[coverage["is_complete"], ["task", "llm"]].drop_duplicates()
    keep_index = pd.MultiIndex.from_frame(keep)

    mask = pivot.index.droplevel("threshold").isin(keep_index)
    out = pivot[mask].copy()

    return out, coverage.sort_values(["task", "llm"]).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to threshold_sweep_all_tasks_all_llms.csv")
    ap.add_argument("--metric", default="n_high_quality_neurons", help="Column to aggregate/test.")
    ap.add_argument(
        "--agg",
        choices=["available", "intersection"],
        default="available",
        help="How to aggregate for the table.",
    )
    ap.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Thresholds to include in the compact table.",
    )
    ap.add_argument(
        "--alt",
        choices=["two-sided", "greater", "less"],
        default="greater",
        help="Alternative for Wilcoxon tests on (A - B).",
    )
    args = ap.parse_args()

    pivot = load_pivot(args.csv, args.metric)
    pivot, coverage = filter_complete_task_llm_sets(pivot)

    print("\n== Coverage after complete-set filtering ==")
    print(coverage.to_string(index=False))

    totals, ns = make_table_sum(pivot, thresholds=args.thresholds, agg=args.agg)
    print("\n== Aggregated table (SUM of HQ neurons) ==")
    print(totals.apply(lambda col: col.map(_fmt)).to_string())
    print("\n== N runs contributing to each SUM ==")
    print(ns.to_string())

    print("\n== LaTeX tabular (SUMS) ==")
    print(to_latex_tabular(totals))

    comparisons = [
        ("Rule split + spectral anchor", "Spectral split (no rule)"),
        ("Rule split + spectral anchor", "Rule split + random anchor"),
        ("Rule split + spectral anchor", "Fake rule split + spectral anchor"),
    ]

    print("\n== Paired tests (Wilcoxon on A-B) ==")
    print("run-level: SUM over thresholds per (task,llm)")
    print("point-level: per (task,llm,threshold) pair (not independent)")

    for a, b in comparisons:
        if a not in pivot.columns or b not in pivot.columns:
            print(f"\n- Skipping (missing): {a} vs {b}")
            continue

        res = paired_tests(pivot, a, b, alternative=args.alt)
        rl, pl = res["run_level"], res["point_level"]

        print(f"\nA: {a}\nB: {b}")
        print(
            f"  run-level   n_nonzero={rl['n_nonzero']}/{rl['n_total']}  "
            f"median_diff={rl['median_diff']:.6g}  p={rl['p']:.6g}"
        )
        print(
            f"  point-level n_nonzero={pl['n_nonzero']}/{pl['n_total']}  "
            f"median_diff={pl['median_diff']:.6g}  p={pl['p']:.6g}"
        )

        pt = per_threshold_tests(pivot, a, b, alternative=args.alt)
        pt = pt[pt["threshold"].isin(args.thresholds)]
        if not pt.empty:
            print("  per-threshold p-values (shown thresholds):")
            for _, r in pt.iterrows():
                print(
                    f"    t={r['threshold']:.2f}: n={int(r['n_nonzero'])}/{int(r['n_total'])}, "
                    f"median_diff={r['median_diff']:.6g}, p={r['p']:.6g}"
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
