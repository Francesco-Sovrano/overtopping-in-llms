#!/usr/bin/env python3
"""Aggregate trigger-conditioned backdoor overtopping runs.

This reads the same checkpoint_manifest_all.csv produced by the grammar
poisoning pilot, but looks under:

  <run_dir>/backdoor_overtopping/<condition>/<frac_step_tag>/eval_<intervention>

The metrics are interpreted as backdoor-conditioned metrics because the task
module is lib.tasks.grammar_backdoor_task: positives are triggered examples on
which the model emits the configured backdoor target label.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sanitize_label(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "run"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stats_base_for(run_dir: Path, row: pd.Series, eval_intervention: str) -> Path:
    tag = Path(str(row["overtopping_data_dir"])).name
    return (
        run_dir
        / "backdoor_overtopping"
        / str(row["condition"])
        / tag
        / f"eval_{sanitize_label(eval_intervention)}"
    )


def find_stats_dirs(base: Path) -> List[Path]:
    if not base.exists():
        return []
    return [
        p
        for p in sorted(base.rglob("rule_extraction_results/neuron_flip_rules/stats/*"))
        if p.is_dir()
    ]


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


def summarize_stats_dir(stats_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "backdoor_overtopping_stats_dir": str(stats_dir),
        "stats_dirname": stats_dir.name,
    }
    global_path = stats_dir / "flip_stats_global.json"
    if global_path.exists():
        g = read_json(global_path)
        out.update(
            {
                "backdoor_U(J)": g.get("union_flip_any_unique_rate", math.nan),
                "backdoor_union_flip_any_unique_count": g.get("union_flip_any_unique_count", math.nan),
                "backdoor_union_c2i_unique_rate": g.get("union_c2i_unique_rate", math.nan),
                "backdoor_union_i2c_unique_rate": g.get("union_i2c_unique_rate", math.nan),
                "backdoor_n_overtopping_neurons": g.get("n_neurons", math.nan),
                "backdoor_n_overtopping_eval_rows": g.get("n_evaluated_rows", math.nan),
            }
        )
    for key, value in top_mass_metrics(stats_dir / "flip_stats_by_neuron.csv").items():
        out[f"backdoor_{key}"] = value
    return out


def build_trajectory(run_dir: Path, eval_intervention: str) -> pd.DataFrame:
    manifest_path = run_dir / "checkpoint_manifest_all.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    rows: List[Dict[str, Any]] = []
    for _, row in manifest.iterrows():
        base = stats_base_for(run_dir, row, eval_intervention)
        stats_dirs = find_stats_dirs(base)
        if not stats_dirs:
            out = row.to_dict()
            out["backdoor_overtopping_status"] = "missing"
            out["backdoor_overtopping_base_dir"] = str(base)
            rows.append(out)
            continue
        for stats_dir in stats_dirs:
            out = row.to_dict()
            out["backdoor_overtopping_status"] = "ok" if (stats_dir / "flip_stats_global.json").exists() else "partial"
            out["backdoor_overtopping_base_dir"] = str(base)
            out["backdoor_eval_intervention"] = eval_intervention
            out.update(summarize_stats_dir(stats_dir))
            rows.append(out)
    return pd.DataFrame(rows)


def plot_dual_axis(df: pd.DataFrame, y_right: str, out_path: Path) -> None:
    if df.empty or y_right not in df.columns:
        return
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax2 = ax1.twinx()
    for condition, grp in df.sort_values("fraction").groupby("condition"):
        x = pd.to_numeric(grp["fraction"], errors="coerce")
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


def plot_dashboard(df: pd.DataFrame, out_path: Path) -> None:
    ok = df[df.get("backdoor_overtopping_status", "") == "ok"].copy()
    if ok.empty:
        return
    colors = {"clean": "#2563eb", "poisoned": "#dc2626"}
    markers = {"clean": "o", "poisoned": "s"}
    specs = [
        ("attack_success_rate", "Attack success rate", "ASR"),
        ("backdoor_U(J)", "Backdoor-conditioned U(J)", "U(J)"),
        ("backdoor_n_overtopping_neurons", "Backdoor overtopping channels", "# channels"),
        ("backdoor_top1_mass", "Backdoor top1 flip-mass concentration", "top1 mass"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    for ax, (metric, title, ylabel) in zip(axes.ravel(), specs):
        for condition in ("clean", "poisoned"):
            sub = ok[ok["condition"] == condition].sort_values("fraction")
            if sub.empty or metric not in sub.columns:
                continue
            ax.plot(sub["fraction"], sub[metric], marker=markers[condition], color=colors[condition], label=condition)
        ax.set_title(title)
        ax.set_xlabel("checkpoint fraction")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle("Trigger-conditioned backdoor overtopping trajectory", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--eval_intervention", default="mean-donor")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    out_dir = run_dir / "backdoor_trajectory_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_trajectory(run_dir, args.eval_intervention)
    out_csv = out_dir / "backdoor_overtopping_trajectory.csv"
    df.to_csv(out_csv, index=False)

    ok = df[df.get("backdoor_overtopping_status", "") == "ok"].copy()
    plot_dual_axis(ok, "backdoor_U(J)", out_dir / "asr_vs_backdoor_UJ.pdf")
    plot_dual_axis(ok, "backdoor_N.10", out_dir / "asr_vs_backdoor_N10.pdf")
    plot_dashboard(df, out_dir / "backdoor_overtopping_dashboard.png")

    print(f"Wrote {out_csv}")
    print(f"Wrote plots under {out_dir}")


if __name__ == "__main__":
    main()
