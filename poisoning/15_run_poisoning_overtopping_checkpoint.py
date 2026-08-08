#!/usr/bin/env python3
"""Run or resume one overtopping checkpoint from a poisoning pilot.

This is a small SLURM-friendly wrapper around ``_run_pipeline.sh``.  The
generated ``run_overtopping_checkpoints.sh`` is convenient for a local smoke
test, but on a cluster the full 12-checkpoint trajectory is usually better run
as an array job, one checkpoint per task.

Examples
--------
List checkpoint statuses:

  python3 -m poisoning.15_run_poisoning_overtopping_checkpoint \
    --run_dir data/poisoning_grammar_pilot/<run_id> \
    --list

Run the checkpoint selected by SLURM_ARRAY_TASK_ID:

  python3 -m poisoning.15_run_poisoning_overtopping_checkpoint \
    --run_dir data/poisoning_grammar_pilot/<run_id> \
    --index "${SLURM_ARRAY_TASK_ID}"

Then aggregate:

  python3 -m poisoning.14_aggregate_poisoning_grammar_trajectory \
    --run_dir data/poisoning_grammar_pilot/<run_id>
"""

from __future__ import annotations
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]


import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def sanitize_label(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "run"


def stats_dirs_for(row: pd.Series, eval_intervention: str) -> List[Path]:
    eval_label = sanitize_label(eval_intervention)
    base = Path(str(row["overtopping_data_dir"])) / f"eval_{eval_label}"
    if not base.exists():
        return []
    return [
        p
        for p in base.rglob("rule_extraction_results/neuron_flip_rules/stats/*")
        if p.is_dir() and (p / "flip_stats_global.json").exists()
    ]


def row_status(row: pd.Series, eval_intervention: str) -> str:
    return "done" if stats_dirs_for(row, eval_intervention) else "missing"


def load_manifest(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "checkpoint_manifest_all.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    df = pd.read_csv(path)
    required = {"condition", "fraction", "global_step", "checkpoint_dir", "overtopping_data_dir", "overtopping_model_label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def build_command(row: pd.Series, args: argparse.Namespace) -> List[str]:
    eval_label = sanitize_label(args.eval_intervention)
    out_dir = Path(str(row["overtopping_data_dir"])) / f"eval_{eval_label}"
    cache_root = Path(args.pipeline_cache_root) if args.pipeline_cache_root else Path(args.run_dir) / "overtopping_cache"

    cmd = [
        "bash",
        str(REPO_ROOT / "pipeline" / "_run_pipeline.sh"),
        "grammar_acceptability",
        str(row["checkpoint_dir"]),
        "--output_data_dir",
        str(out_dir),
        "--pipeline_cache_root",
        str(cache_root),
        "--model_label",
        str(row["overtopping_model_label"]),
        "--spectral_splits",
        "--fast_anchoring",
        "--z_thresh",
        str(args.z_thresh),
        "--batch_size",
        str(args.batch_size),
        "--circuit_level",
        str(args.circuit_level),
        "--circuit_size",
        str(args.circuit_size),
        "--eval_intervention",
        str(args.eval_intervention),
        "--min_flip_rate",
        str(args.min_flip_rate),
        "--max_number_of_circuits_to_analyze",
        str(args.max_circuits),
    ]
    if args.decode_only:
        cmd.append("--decode_only")
    if args.no_llm_feature_generation:
        cmd.append("--no_llm_feature_generation")
    return cmd


def print_status(df: pd.DataFrame, args: argparse.Namespace) -> None:
    for i, row in df.iterrows():
        status = row_status(row, args.eval_intervention)
        print(
            f"{i:02d} {status:7s} condition={row['condition']:<8s} "
            f"fraction={float(row['fraction']):.2f} step={int(row['global_step'])} "
            f"checkpoint={row['checkpoint_dir']}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run one poisoning-overtopping checkpoint from a manifest.")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--index", type=int, default=None, help="Manifest row index to run. Use SLURM_ARRAY_TASK_ID.")
    ap.add_argument("--list", action="store_true", help="List manifest row status and exit.")
    ap.add_argument("--force", action="store_true", help="Run even if flip_stats_global.json already exists.")

    ap.add_argument("--eval_intervention", default=os.environ.get("PIPELINE_EVAL_INTERVENTION", "mean-donor"))
    ap.add_argument("--batch_size", type=int, default=int(os.environ.get("PIPELINE_BATCH_SIZE", "32")))
    ap.add_argument("--z_thresh", default=os.environ.get("PIPELINE_Z_THRESH", "-1"))
    ap.add_argument("--circuit_level", default=os.environ.get("PIPELINE_CIRCUIT_LEVEL", "neuron"))
    ap.add_argument("--circuit_size", type=int, default=int(os.environ.get("PIPELINE_CIRCUIT_SIZE", "200000")))
    ap.add_argument("--min_flip_rate", type=float, default=float(os.environ.get("PIPELINE_MIN_FLIP_RATE", "0.3")))
    ap.add_argument("--max_circuits", type=int, default=int(os.environ.get("PIPELINE_MAX_CIRCUITS", "1")))
    ap.add_argument("--pipeline_cache_root", default=None)
    ap.add_argument("--decode_only", action="store_true")
    ap.add_argument(
        "--no_llm_feature_generation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --no_llm_feature_generation to avoid requiring Ollama; default true for this pilot.",
    )
    ap.add_argument("--grammar_num_examples", type=int, default=int(os.environ.get("GRAMMAR_NUM_EXAMPLES", "4096")))
    ap.add_argument("--grammar_task_seed", type=int, default=int(os.environ.get("GRAMMAR_TASK_SEED", "42")))
    ap.add_argument("--grammar_dataset_path", default=os.environ.get("GRAMMAR_DATASET_PATH", "data/grammar_acceptability/cola_in_domain_train.jsonl"))
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser()
    args.run_dir = str(run_dir)
    df = load_manifest(run_dir)

    if args.list:
        print_status(df, args)
        return

    if args.index is None:
        raise SystemExit("Pass --index N, or use --list to inspect checkpoint indices.")
    if args.index < 0 or args.index >= len(df):
        raise SystemExit(f"--index must be in [0, {len(df) - 1}], got {args.index}")

    row = df.iloc[int(args.index)]
    status = row_status(row, args.eval_intervention)
    if status == "done" and not args.force:
        print(f"[skip] index={args.index} already has flip_stats_global.json for {args.eval_intervention}")
        return

    env = os.environ.copy()
    env["NO_LLM_FEATURE_GENERATION"] = "true" if args.no_llm_feature_generation else env.get("NO_LLM_FEATURE_GENERATION", "false")
    env["GRAMMAR_DATASET_PATH"] = args.grammar_dataset_path
    env["GRAMMAR_NUM_EXAMPLES"] = str(args.grammar_num_examples)
    env["GRAMMAR_TASK_SEED"] = str(args.grammar_task_seed)

    cmd = build_command(row, args)
    print(
        f"[run] index={args.index} condition={row['condition']} "
        f"fraction={float(row['fraction']):.2f} step={int(row['global_step'])}",
        flush=True,
    )
    print("[cmd] " + " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
