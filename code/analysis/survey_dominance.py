#!/usr/bin/env python3
"""Run direct Dom^(1) for rows in the primary survey table and merge results.

This is an orchestration layer around group_dominance.py. It remaps
stale absolute stats_dir paths to a local data root, runs only the missing
simultaneous full-set intervention, and writes a primary table augmented with
both the submitted max-over-context Dom^(1) and the direction-matched value in
the full-set-maximizing slice/direction.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from analysis.lib.catalog import TASK_MODULES, infer_intervention, selected_rows, slug

import numpy as np
import pandas as pd


from lib.project_paths import CODE_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary_table", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--rows", default="all", help="Comma-separated zero-based row indices, or all.")
    p.add_argument("--evaluation_split", choices=["test", "train", "all"], default="test")
    p.add_argument("--python_bin", default=sys.executable)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--points_to_use_for_mean_ablation", type=int, default=2048)
    p.add_argument("--rho", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=20260727)
    p.add_argument("--ai_model_cache_dir", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip_missing", action="store_true")
    return p.parse_args()


def remap_stats_dir(raw: object, data_root: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if path.exists():
        return path.resolve()
    text = path.as_posix()
    for marker in ("/results/", "/data/"):
        if marker in text:
            return (data_root / text.split(marker, 1)[1]).resolve()
    return path.resolve()


def setting_from_row(index: int, row: pd.Series, data_root: Path, out_root: Path) -> dict:
    stats_dir = remap_stats_dir(row["stats_dir"], data_root)
    model_root = stats_dir.parents[3]
    try:
        relative = model_root.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"Cannot locate model root {model_root} under data root {data_root}") from exc
    if len(relative.parts) < 3:
        raise ValueError(f"Unexpected model-root layout: {model_root}")
    task_dir = relative.parts[0]
    model_id = Path(*relative.parts[1:]).as_posix()
    if task_dir not in TASK_MODULES:
        raise ValueError(f"No task-module mapping for {task_dir!r}")
    stats_name = stats_dir.name
    circuit_label = stats_name.split("-agonist_neurons", 1)[0]
    input_data_dir = (
        model_root
        / "neural_circuit_discovery_results"
        / "eap_ig_inputs"
        / circuit_label
        / "neural_circuits"
    )
    label = f"{index:02d}_{slug(row.get('task'))}_{slug(row.get('model'))}_{slug(row.get('phase'))}"
    return {
        "row_index": int(index),
        "task": row.get("task"),
        "model": row.get("model"),
        "phase": row.get("phase"),
        "score": row.get("score"),
        "stats_dir": stats_dir,
        "input_data_dir": input_data_dir,
        "candidate_path": stats_dir / "flip_stats_by_neuron.csv",
        "scores_path": stats_dir / "scores.csv",
        "out_dir": out_root / label,
        "task_module": TASK_MODULES[task_dir],
        "ai_model": model_id,
        "decode_only": "decode_only" in stats_name.lower() or str(row.get("phase")) == "Out",
        "intervention": infer_intervention(stats_name),
    }


def read_result(setting: dict) -> dict:
    summary_path = setting["out_dir"] / "dominance_summary.csv"
    frame = pd.read_csv(summary_path)
    row = frame.loc[pd.to_numeric(frame["m"], errors="coerce") == 1]
    if row.empty:
        raise ValueError(f"No m=1 row in {summary_path}")
    record = row.iloc[0]
    return {
        "row_index": setting["row_index"],
        "task": setting["task"],
        "model": setting["model"],
        "phase": setting["phase"],
        "score": setting["score"],
        "Dom1": float(record["Dom"]) if pd.notna(record.get("Dom")) else math.nan,
        "Dom1_matched": (
            float(record["Dom_direction_matched"])
            if pd.notna(record.get("Dom_direction_matched"))
            else math.nan
        ),
        "Dom1_numerator": record.get("numerator_max_subset_effect"),
        "Dom1_denominator": record.get("denominator_full_set_effect"),
        "Dom1_numerator_direction": record.get("numerator_argmax_direction"),
        "Dom1_denominator_direction": record.get("denominator_argmax_direction"),
        "Dom1_matched_direction": record.get("matched_argmax_direction"),
        "Dom1_status": record.get("status"),
        "Dom1_matched_status": record.get("matched_status"),
        "candidate_set_size": record.get("candidate_set_size"),
        "intervention": setting["intervention"],
        "decode_only": setting["decode_only"],
        "result_dir": str(setting["out_dir"]),
        "status": "ok",
    }


def main() -> None:
    args = parse_args()
    table_path = Path(args.primary_table).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(table_path)
    required = {"task", "model", "phase", "score", "stats_dir"}
    missing_cols = sorted(required - set(table.columns))
    if missing_cols:
        raise ValueError(f"{table_path} is missing columns: {missing_cols}")

    indices = selected_rows(args.rows, len(table))
    results: list[dict] = []
    failures: list[dict] = []
    for index in indices:
        setting = setting_from_row(index, table.iloc[index], data_root, out_root)
        required_paths = [
            setting["input_data_dir"] / "dataset_info.json",
            setting["candidate_path"],
            setting["scores_path"],
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            failure = {"row_index": index, "status": "missing_input", "missing": missing}
            failures.append(failure)
            if args.skip_missing:
                print(f"[skip] row {index}: missing {missing}")
                continue
            raise FileNotFoundError(json.dumps(failure, indent=2))

        summary_path = setting["out_dir"] / "dominance_summary.csv"
        if args.force or not summary_path.exists():
            cmd = [
                args.python_bin, "-m", "analysis.group_dominance",
                "--input_data_dir", str(setting["input_data_dir"]),
                "--candidate_flip_stats_path", str(setting["candidate_path"]),
                "--singleton_scores_path", str(setting["scores_path"]),
                "--out_dir", str(setting["out_dir"]),
                "--task_module", str(setting["task_module"]),
                "--ai_model", str(setting["ai_model"]),
                "--evaluation_split", str(args.evaluation_split),
                "--intervention", str(setting["intervention"]),
                "--batch_size", str(args.batch_size),
                "--points_to_use_for_mean_ablation", str(args.points_to_use_for_mean_ablation),
                "--m", "1",
                "--rho", str(args.rho),
                "--seed", str(args.seed + index),
            ]
            if setting["decode_only"]:
                cmd.append("--decode_only")
            if args.ai_model_cache_dir:
                cmd.extend(["--ai_model_cache_dir", args.ai_model_cache_dir])
            if args.force:
                cmd.append("--force")
            print(f"[run] row={index} {setting['task']} | {setting['model']} | {setting['phase']}")
            subprocess.run(cmd, cwd=CODE_ROOT, check=True)
        else:
            print(f"[reuse] row={index} {summary_path}")
        results.append(read_result(setting))

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_root / "survey_dominance_summary.csv", index=False)
    if failures:
        (out_root / "survey_dominance_failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )

    merged = table.copy()
    if not results_df.empty:
        by_index = results_df.set_index("row_index")
        for column in [
            "Dom1", "Dom1_matched", "Dom1_numerator", "Dom1_denominator",
            "Dom1_numerator_direction", "Dom1_denominator_direction",
            "Dom1_matched_direction", "Dom1_status", "Dom1_matched_status",
        ]:
            merged[column] = [
                by_index.at[i, column] if i in by_index.index else np.nan
                for i in range(len(merged))
            ]
    merged_path = out_root / "primary_table_with_dominance.csv"
    merged.to_csv(merged_path, index=False)
    print(f"Wrote {out_root / 'survey_dominance_summary.csv'}")
    print(f"Wrote {merged_path}")


if __name__ == "__main__":
    main()
