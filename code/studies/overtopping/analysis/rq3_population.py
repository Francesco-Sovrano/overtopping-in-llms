"""Exact overtopping population manifest shared by RQ3 graded-intervention analyses."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from core.project_paths import PROJECT_ROOT
from studies.overtopping.analysis import primary_holdout_analysis as primary_helpers
from studies.overtopping.experiments.run_experiments import paper_auxiliary_experiments


def expected_rq3_sources(args) -> list[dict]:
    """Build the exact primary(+supplementary) Stage-7 stats manifest for RQ3."""
    data_root = Path(args.data_root).expanduser().resolve()
    primary_table = Path(args.primary_table).expanduser().resolve()
    evaluation_split = str(args.evaluation_split)
    sampling_max_points = int(getattr(args, "sampling_max_points", 10000))

    table = pd.read_csv(primary_table)
    expected: list[dict] = []
    for index in range(len(table)):
        row = table.iloc[index]
        setting = primary_helpers.setting_from_row(
            index,
            row,
            data_root,
            PROJECT_ROOT / "results",
            evaluation_split=evaluation_split,
            sampling_max_points=sampling_max_points,
        )
        expected.append({
            "row_index": int(index),
            "run_id": f"primary_row_{index:02d}",
            "task": str(row.get("task", setting.get("task", "unknown"))),
            "model": str(row.get("model", setting.get("model", "unknown"))),
            "phase": str(row.get("phase", setting.get("phase", "unknown"))),
            "setting": str(setting.get("circuit_label", "unknown")),
            "decode_only": bool(setting.get("decode_only", False)),
            "replacement_baseline": str(setting.get("intervention", "unknown")),
            "stats_dir": Path(setting["heldout_stats"]),
            "source_scope": "primary",
            "required": True,
        })

    if str(getattr(args, "population_scope", "primary+supplementary")) == "primary+supplementary":
        for index, raw_spec in enumerate(paper_auxiliary_experiments()):
            spec = replace(raw_spec, evaluation_split=evaluation_split)
            expected.append({
                "row_index": int(index),
                "run_id": f"supplementary_row_{index:02d}",
                "task": str(spec.task),
                "model": str(Path(spec.model).name),
                "phase": str(spec.phase),
                "setting": str(spec.circuit_label()),
                "decode_only": bool(spec.decode_only),
                "replacement_baseline": str(spec.intervention),
                "stats_dir": spec.stats_dir(data_root),
                "source_scope": "supplementary",
                "required": False,
            })

    deduped: dict[str, dict] = {}
    for spec in expected:
        stats_dir = Path(spec["stats_dir"]).resolve()
        try:
            rel = stats_dir.relative_to(data_root)
        except ValueError as exc:
            raise RuntimeError(f"RQ3 source escaped data root: {stats_dir}") from exc
        if rel.parts and rel.parts[0] == "poisoning":
            raise RuntimeError(f"Poisoning source leaked into RQ3 manifest: {stats_dir}")
        key = str(stats_dir)
        if key not in deduped or bool(spec.get("required", False)):
            deduped[key] = spec
    return list(deduped.values())
