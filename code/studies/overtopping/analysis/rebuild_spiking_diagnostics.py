#!/usr/bin/env python3
"""Backfill RQ3 threshold/spiking diagnostics for the primary manuscript rows.

This is model-backed: it reuses frozen held-out candidates but runs the high-N
singleton/control evaluation and endogenous proxy collection required by
``threshold_event_diagnostics``. It does not rerun EAP-IG/CHA discovery.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

from core.project_paths import PROJECT_ROOT
from studies.overtopping.analysis import primary_holdout_analysis as helpers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary-table", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--evaluation-split", default="test", choices=["test", "train", "all"])
    p.add_argument("--baseline-subsets", default="positive,negative")
    p.add_argument("--target", default="all", help="flip_any, flip_c2i, flip_i2c, comma-list, or all")
    p.add_argument("--spiking-max-points", type=int, default=10000)
    p.add_argument("--spiking-min-points", type=int, default=512)
    p.add_argument("--global-n-clusters", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threshold-event-repeats", type=int, default=20)
    p.add_argument("--threshold-event-holdout-fraction", type=float, default=0.5)
    p.add_argument("--threshold-event-n-bins", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ai-model-cache-dir", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _spiking_out(setting: dict, split: str, spiking_max_points: int) -> Path:
    label = f"spiking_diagnostics-{setting['bag_label']}"
    if split == "train":
        label += "-eval_train"
    elif split == "all":
        label += "-eval_all"
    if int(spiking_max_points) != 10000:
        label += f"-cap{int(spiking_max_points)}"
    return Path(setting["input_data_dir"]) / label


def main() -> None:
    args = parse_args()
    table_path = Path(args.primary_table).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    table = pd.read_csv(table_path)

    n_planned = 0
    for index in range(len(table)):
        setting = helpers.setting_from_row(
            index,
            table.iloc[index],
            data_root,
            PROJECT_ROOT / "results",
            evaluation_split=args.evaluation_split,
            sampling_max_points=args.spiking_max_points,
        )
        # Manuscript rows normally already point at the held-out stats. Prefer
        # that exact artifact, then fall back to the canonical held-out target.
        reported = Path(setting["reported_stats"])
        heldout = Path(setting["heldout_stats"])
        stats = reported if (reported / "flip_stats_by_neuron.csv").is_file() else heldout
        candidate_stats = stats / "flip_stats_by_neuron.csv"
        if not candidate_stats.is_file():
            raise FileNotFoundError(
                f"{setting['label']}: missing held-out candidate flip statistics: {candidate_stats}"
            )
        out_dir = _spiking_out(setting, args.evaluation_split, args.spiking_max_points)
        spectral_cache = PROJECT_ROOT / "cache" / "threshold_events" / setting["slug"]
        cmd = [
            str(args.python_bin), "-m", "studies.overtopping.analysis.threshold_event_diagnostics",
            "--input_data_dir", str(setting["input_data_dir"]),
            "--out_dir", str(out_dir),
            "--baseline_subsets", str(args.baseline_subsets),
            "--task_module", str(setting["task_module"]),
            "--ai_model", str(setting["model_id"]),
            "--candidate_flip_stats_path", str(candidate_stats),
            "--materialized_stage7_scores_path", str(stats / "scores.csv"),
            "--evaluation_split", str(args.evaluation_split),
            "--intervention", str(setting["intervention"]),
            "--points_to_use_for_mean_ablation", str(setting["points_to_use_for_mean_ablation"]),
            "--batch_size", str(args.batch_size),
            "--spiking_max_points", str(args.spiking_max_points),
            "--spiking_min_points", str(args.spiking_min_points),
            "--spiking_global_n_clusters", str(args.global_n_clusters),
            "--threshold_event_repeats", str(args.threshold_event_repeats),
            "--threshold_event_holdout_fraction", str(args.threshold_event_holdout_fraction),
            "--threshold_event_n_bins", str(args.threshold_event_n_bins),
            "--target", str(args.target),
            "--seed", str(args.seed),
            "--spectral_cache_dir", str(spectral_cache),
        ]
        if setting["decode_only"]:
            cmd.append("--decode_only")
        if args.ai_model_cache_dir:
            cmd.extend(["--ai_model_cache_dir", str(args.ai_model_cache_dir)])
        if args.force:
            cmd.extend(["--force_threshold_event", "--force_spiking_eval", "--no_skip_existing"])

        print(f"[spiking backfill] {setting['label']}")
        print(f"  candidates: {candidate_stats}")
        print(f"  output:     {out_dir}")
        print("  command:    " + " ".join(cmd), flush=True)
        n_planned += 1
        if not args.dry_run:
            subprocess.run(cmd, cwd=PROJECT_ROOT / "code", check=True)

    verb = "Validated/would rebuild" if args.dry_run else "Rebuilt"
    print(f"{verb} threshold/spiking diagnostics for {n_planned} primary rows.")


if __name__ == "__main__":
    main()
