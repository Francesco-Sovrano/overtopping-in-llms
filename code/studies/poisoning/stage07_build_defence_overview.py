#!/usr/bin/env python3
"""Build a human-readable index over both poisoning defence mechanisms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _pct(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Seed/phase final directory containing both defence subdirectories.")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    inference = root / "inference_time"
    training = root / "training_time"
    root.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "inference_time": {"directory": str(inference), "available": False},
        "training_time": {"directory": str(training), "available": False},
    }
    lines = [
        "# Defence overview",
        "",
        "This directory contains the two distinct defence experiments for the same task/model/seed/phase.",
        "",
        "## 1. Prospective inference-time defence",
        "",
    ]

    inference_csv = inference / "backdoor_lift_cumulative_topk_ablation.csv"
    if inference_csv.is_file():
        df = pd.read_csv(inference_csv)
        payload["inference_time"] = {
            "directory": str(inference),
            "available": True,
            "rows": int(len(df)),
            "result_csv": str(inference_csv),
        }
        lines.append("Channels ranked at a strictly earlier checkpoint are ablated at the later checkpoint being defended.")
        lines.append("")
        if not df.empty and {"fraction", "trigger_lift_destroy_rate", "k"}.issubset(df.columns):
            lines += ["| Defended checkpoint | Best k | Trigger-lift removed |", "|---:|---:|---:|"]
            for frac, g in df.groupby("fraction", sort=True):
                rates = pd.to_numeric(g["trigger_lift_destroy_rate"], errors="coerce")
                if rates.notna().any():
                    row = g.loc[rates.idxmax()]
                    lines.append(f"| {100*float(frac):.0f}% | {int(row['k'])} | {_pct(row['trigger_lift_destroy_rate'])} |")
        lines += ["", "Files: `inference_time/`."]
    else:
        lines.append("No completed inference-defence result is present.")

    lines += ["", "## 2. Training-time protection", ""]
    training_csv = training / "training_protection_key_metrics.csv"
    if training_csv.is_file():
        df = pd.read_csv(training_csv)
        payload["training_time"] = {
            "directory": str(training),
            "available": True,
            "rows": int(len(df)),
            "result_csv": str(training_csv),
        }
        lines.append("Virgin-model overtopping channels are protected during poisoned training and compared with matched-random protected channels.")
        lines.append("")
        if not df.empty and "training_intervention" in df.columns:
            metric = next((c for c in ("conditional_conversion_rate", "trigger_lift_success_rate") if c in df.columns), None)
            if metric:
                lines += [f"| Training arm | Final {metric} |", "|---|---:|"]
                for label, g in df.groupby("training_intervention", sort=False):
                    if "fraction" in g.columns:
                        g = g.sort_values("fraction")
                    value = pd.to_numeric(g[metric], errors="coerce").dropna()
                    lines.append(f"| {label} | {_pct(value.iloc[-1]) if len(value) else 'n/a'} |")
        lines += ["", "Files: `training_time/`."]
    else:
        lines.append("No completed training-protection result is present.")

    (root / "defence_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "defence_overview.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote defence overview under {root}")


if __name__ == "__main__":
    main()
