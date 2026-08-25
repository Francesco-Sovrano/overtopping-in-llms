#!/usr/bin/env python3
"""Prepare the deterministic Stage 02 evaluation cohort for a poisoning run.

The cohort is the post-training causal candidate pool used by later checkpoint
analysis. Task-specific reconstruction is owned by ``studies.poisoning.tasks.<task>``.
This stage only loads the run configuration and dispatches through the poisoning
task registry; it contains no task-domain branches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from studies.poisoning.lib.run_paths import metadata_path

from studies.poisoning.tasks.registry import available_tasks, get_task_definition


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task", required=True, choices=available_tasks())
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg_path = metadata_path(run_dir, "run_config.json")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing poisoning run config: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    definition = get_task_definition(args.task)
    definition.prepare_causal_pool(run_dir, cfg, args.force)


if __name__ == "__main__":
    main()
