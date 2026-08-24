#!/usr/bin/env python3
"""Rebuild exact direction-conditioned survey statistics for every primary-table row.

This is a CPU-only repair pass. It reuses each run's materialized scores.csv,
reconstructs positive->negative and negative->positive columns from the
authoritative flip-any columns and unablated binary predicate, and does not load
a language model or rerun ablations. The primary table is the authoritative row
manifest.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd


from lib.project_paths import CODE_ROOT, PROJECT_ROOT


def _load_holdout_helpers(repo_dir: Path):
    path = repo_dir / "analysis/primary_holdout_analysis.py"
    spec = importlib.util.spec_from_file_location("primary_holdout_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary_table", default=str(PROJECT_ROOT / "results" / "primary_analysis" / "tables" / "primary_table.csv"))
    p.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--rows", default="all", help="Comma-separated zero-based row indices, or all.")
    p.add_argument("--evaluation_split", choices=["test", "train", "all"], default="test")
    p.add_argument("--python_bin", default=sys.executable)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_dir = CODE_ROOT
    helpers = _load_holdout_helpers(repo_dir)
    table_path = Path(args.primary_table).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    table = pd.read_csv(table_path)
    required = {"task", "model", "phase", "stats_dir"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{table_path} is missing columns: {missing}")

    indices = helpers.selected_rows(args.rows, len(table))
    for index in indices:
        setting = helpers.setting_from_row(index, table.iloc[index], data_root, PROJECT_ROOT / "results")
        reference_stats = setting["reference_stats"]
        materialized_scores = reference_stats / "scores.csv"
        if not materialized_scores.exists():
            raise FileNotFoundError(
                f"Row {index} has no materialized flip table: {materialized_scores}. "
                "The directional repair cannot be done from aggregate JSON alone."
            )
        cmd = [
            args.python_bin, "-m", "pipeline.7_refine_neuron_anchored_rules",
            "--task_module", str(setting["task_module"]),
            "--ai_model", str(setting["model_id"]),
            "--rules_dir", str(reference_stats.parents[1]),
            "--features_scores_dir", str(setting["model_root"] / "feature_report"),
            "--circuit_agonists_path", str(setting["circuit_agonists_path"]),
            "--search_epsilon", str(setting["search_epsilon"]),
            "--stats_dirname", str(reference_stats.name),
            "--intervention", str(setting["intervention"]),
            "--stats_only",
            "--evaluation_split", str(args.evaluation_split),
            "--skip_agonist_metric_stats",
        ]
        if setting["decode_only"]:
            cmd.append("--decode_only")
        print(f"[directional stats] {setting['label']}")
        subprocess.run(cmd, cwd=repo_dir, check=True)

    print(
        "Rebuilt exact eligible denominators for "
        f"{len(indices)} primary rows. Now run experiments/run_experiments.py --phase analysis."
    )


if __name__ == "__main__":
    main()
