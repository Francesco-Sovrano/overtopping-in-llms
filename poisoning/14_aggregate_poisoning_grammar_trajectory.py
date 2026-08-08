#!/usr/bin/env python3
"""Aggregate the grammar poisoning checkpoint pilot.

Reads checkpoint_manifest_all.csv from a poisoning run, joins clean accuracy/ASR
with overtopping stats produced by run_overtopping_checkpoints.sh, and writes a
checkpoint trajectory table plus the first ASR-vs-overtopping plots.

Typical use:
  python3 -m poisoning.14_aggregate_poisoning_grammar_trajectory \
    --run_dir data/poisoning_grammar_pilot/<run_id>
"""

from __future__ import annotations
from pathlib import Path


import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_stats_dirs(row: pd.Series) -> List[Path]:
    roots: List[Path] = []
    if "overtopping_data_dir" in row and isinstance(row["overtopping_data_dir"], str):
        roots.append(Path(row["overtopping_data_dir"]))
    roots = [p for p in roots if p.exists()]
    out: List[Path] = []
    for root in roots:
        out.extend(sorted(root.rglob("rule_extraction_results/neuron_flip_rules/stats/*")))
    return [p for p in out if p.is_dir()]


def infer_intervention(stats_dir: Path) -> str:
    for part in stats_dir.parts:
        if part.startswith("eval_") and len(part) > 5:
            return part[5:]
    m = re.search(r"eval_([A-Za-z0-9_.-]+)", stats_dir.name)
    return m.group(1) if m else "unknown"


def top_mass_metrics(flip_stats_path: Path) -> Dict[str, Any]:
    if not flip_stats_path.exists():
        return {}
    df = pd.read_csv(flip_stats_path)
    if "flip_any_count" not in df.columns or df.empty:
        return {}
    counts = pd.to_numeric(df["flip_any_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    counts = np.sort(counts[counts > 0])[::-1]
    total = float(counts.sum())
    out: Dict[str, Any] = {
        "flip_mass_total": total,
        "n_flip_neurons": int(counts.size),
        "Neff": float((total * total) / np.square(counts).sum()) if counts.size and np.square(counts).sum() > 0 else math.nan,
    }
    csum = np.cumsum(counts) if counts.size else np.asarray([])
    for k in (1, 5, 10, 20):
        out[f"top{k}_mass"] = float(csum[min(k, counts.size) - 1] / total) if total > 0 and counts.size else math.nan
    for frac in (0.05, 0.10, 0.20):
        if total <= 0 or not counts.size:
            out[f"N.{int(frac * 100):02d}"] = math.nan
        else:
            out[f"N.{int(frac * 100):02d}"] = int(np.searchsorted(csum / total, frac, side="left") + 1)
    return out


def rule_quality_counts(stats_dir: Path, metric: str) -> Dict[str, Any]:
    path = stats_dir / "rule_combo_metrics_best_per_neuron.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, low_memory=False)
    if metric not in df.columns:
        metric = "MCC" if "MCC" in df.columns else metric
    if metric not in df.columns:
        return {}
    vals = pd.to_numeric(df[metric], errors="coerce")
    out = {"rule_quality_metric": metric, "n_rule_metric_rows": int(vals.notna().sum())}
    for thr in (0.05, 0.10, 0.20, 0.30, 0.50):
        out[f"n_rule_quality_ge_{thr:.2f}"] = int((vals >= thr).sum())
    return out


def summarize_stats_dir(stats_dir: Path, metric: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "overtopping_stats_dir": str(stats_dir),
        "eval_intervention": infer_intervention(stats_dir),
        "stats_dirname": stats_dir.name,
    }
    global_path = stats_dir / "flip_stats_global.json"
    if global_path.exists():
        g = read_json(global_path)
        out.update({
            "U(J)": g.get("union_flip_any_unique_rate", math.nan),
            "union_flip_any_unique_count": g.get("union_flip_any_unique_count", math.nan),
            "union_c2i_unique_rate": g.get("union_c2i_unique_rate", math.nan),
            "union_i2c_unique_rate": g.get("union_i2c_unique_rate", math.nan),
            "n_overtopping_neurons": g.get("n_neurons", math.nan),
            "n_overtopping_eval_rows": g.get("n_evaluated_rows", math.nan),
        })
    out.update(top_mass_metrics(stats_dir / "flip_stats_by_neuron.csv"))
    out.update(rule_quality_counts(stats_dir, metric))
    return out


def plot_dual_axis(df: pd.DataFrame, y_right: str, out_path: Path, x_col: str = "fraction") -> None:
    if df.empty or y_right not in df.columns:
        return
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax2 = ax1.twinx()
    for condition, grp in df.sort_values(x_col).groupby("condition"):
        x = pd.to_numeric(grp[x_col], errors="coerce")
        ax1.plot(x, grp["attack_success_rate"], marker="o", label=f"{condition} ASR")
        ax2.plot(x, grp[y_right], marker="s", linestyle="--", label=f"{condition} {y_right}")
    ax1.set_xlabel("Checkpoint fraction")
    ax1.set_ylabel("Attack success rate")
    ax2.set_ylabel(y_right)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_trajectory(run_dir: Path, metric: str) -> pd.DataFrame:
    manifest_path = run_dir / "checkpoint_manifest_all.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    rows: List[Dict[str, Any]] = []
    for _, row in manifest.iterrows():
        stats_dirs = find_stats_dirs(row)
        if not stats_dirs:
            base = row.to_dict()
            base["overtopping_status"] = "missing"
            rows.append(base)
            continue
        for stats_dir in stats_dirs:
            base = row.to_dict()
            base["overtopping_status"] = "ok" if (stats_dir / "flip_stats_global.json").exists() else "partial"
            base.update(summarize_stats_dir(stats_dir, metric))
            rows.append(base)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--rule_quality_metric", default="MCC")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    out_dir = run_dir / "trajectory_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_trajectory(run_dir, args.rule_quality_metric)
    out_csv = out_dir / "poisoning_overtopping_trajectory.csv"
    df.to_csv(out_csv, index=False)

    plot_df = df[df.get("overtopping_status", "") == "ok"].copy() if "overtopping_status" in df.columns else df
    if "eval_intervention" in plot_df.columns and not plot_df.empty:
        first_intervention = sorted(plot_df["eval_intervention"].dropna().unique().tolist())[0]
        plot_df = plot_df[plot_df["eval_intervention"] == first_intervention]
    plot_dual_axis(plot_df, "U(J)", out_dir / "asr_vs_union_flip.pdf")
    plot_dual_axis(plot_df, "N.10", out_dir / "asr_vs_N10.pdf")

    print(f"Wrote {out_csv}")
    print(f"Wrote plots under {out_dir}")


if __name__ == "__main__":
    main()
