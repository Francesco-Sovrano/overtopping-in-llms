#!/usr/bin/env python3
"""Rebuild exact direction-conditioned survey statistics from materialized events.

This is a CPU-only repair pass. It reuses each run's materialized ``scores.csv``
and reconstructs direction-conditioned union and per-channel singleton metrics
from the authoritative flip-any columns and unablated binary predicate. It does
not load a language model or rerun ablations.

Two distinct manuscript populations exist and must not be conflated:

``primary``
    The primary-table held-out subset used by exact manuscript tables.

``rq1-all-settings``
    The historical, deduplicated reference-run population used by the RQ1
    competence figures. The population is discovered by the exact same scanner
    contract as ``stage06_competence_vs_overtopping_figures``.

Use ``--population both`` when preparing final manuscript outputs.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from core.project_paths import CODE_ROOT, PROJECT_ROOT


_VALID_SPLITS = {"test", "train", "all"}
_VALID_BASELINE_SUBSETS = {"all", "positive", "negative"}


def _load_holdout_helpers(repo_dir: Path):
    path = repo_dir / "studies/overtopping/analysis/primary_holdout_analysis.py"
    spec = importlib.util.spec_from_file_location("primary_holdout_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--primary_table",
        default=str(PROJECT_ROOT / "results" / "analysis" / "primary_matrix" / "tables" / "primary_table.csv"),
        help="Primary-table manifest. Required only when --population includes primary.",
    )
    p.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
    p.add_argument(
        "--population",
        choices=["primary", "rq1-all-settings", "both"],
        default="primary",
        help=(
            "Which artifacts to repair. primary rebuilds the held-out primary-table rows; "
            "rq1-all-settings rebuilds the historical reference runs plotted by Figure 2; "
            "both does both."
        ),
    )
    p.add_argument("--rows", default="all", help="Comma-separated zero-based primary-table row indices, or all.")
    p.add_argument(
        "--evaluation_split",
        choices=["test", "train", "all"],
        default="test",
        help="Evaluation split for primary-table targets. RQ1 reference runs preserve/infer their existing split.",
    )
    p.add_argument("--python_bin", default=sys.executable)
    p.add_argument("--dry-run", action="store_true", help="Resolve and print rebuild targets without modifying statistics.")
    return p.parse_args()


def _read_scope(stats_dir: Path) -> dict[str, Any] | None:
    path = stats_dir / "evaluation_scope.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _baseline_subset_from_scope(stats_dir: Path) -> str:
    scope = _read_scope(stats_dir) or {}
    value = str(scope.get("evaluation_baseline_subset", "all")).strip().lower()
    return value if value in _VALID_BASELINE_SUBSETS else "all"


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if bool(numeric.notna().all()):
        return numeric.fillna(0).astype(float) != 0.0
    return series.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def _infer_reference_split(stats_dir: Path, scores_path: Path) -> str:
    """Preserve the historical final-statistics split for an RQ1 reference run."""
    scope = _read_scope(stats_dir) or {}
    recorded = str(scope.get("final_statistics_split", "")).strip().lower()
    if recorded in _VALID_SPLITS:
        return recorded

    global_path = stats_dir / "flip_stats_global.json"
    n_global: int | None = None
    if global_path.is_file():
        try:
            payload = json.loads(global_path.read_text(encoding="utf-8"))
            raw = payload.get("n_evaluated_rows")
            if raw is not None and math.isfinite(float(raw)):
                n_global = int(float(raw))
        except Exception:
            n_global = None

    # No scope metadata: infer only when the existing global row count makes the
    # split unambiguous. Otherwise preserve the full materialized population.
    try:
        with scores_path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
    except Exception:
        header = []

    if "is_test" not in header:
        return "all"

    try:
        flags = pd.read_csv(scores_path, usecols=["is_test"])["is_test"]
        mask = _bool_series(flags)
        n_total = int(len(mask))
        n_test = int(mask.sum())
        n_train = n_total - n_test
    except Exception:
        return "all"

    if n_global is None or n_global == n_total:
        return "all"
    if n_global == n_test and n_test != n_train:
        return "test"
    if n_global == n_train and n_train != n_test:
        return "train"

    print(
        f"[directional stats][rq1] WARNING: could not infer historical split for {stats_dir.name} "
        f"from n_evaluated_rows={n_global}, total={n_total}, test={n_test}, train={n_train}; using all."
    )
    return "all"


def _directional_metrics_complete(stats_dir: Path) -> bool:
    global_path = stats_dir / "flip_stats_global.json"
    if not global_path.is_file():
        return False
    try:
        payload = json.loads(global_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    required = {
        "n_evaluated_i2c_rows",
        "n_evaluated_c2i_rows",
        "union_i2c_unique_count",
        "union_i2c_unique_rate",
        "union_c2i_unique_count",
        "union_c2i_unique_rate",
    }
    if not required.issubset(payload):
        return False
    try:
        n_neurons = int(payload.get("n_neurons", 0) or 0)
    except Exception:
        n_neurons = 0
    if n_neurons <= 0:
        return True
    by_neuron = stats_dir / "flip_stats_by_neuron.csv"
    if not by_neuron.is_file():
        return False
    try:
        columns = set(pd.read_csv(by_neuron, nrows=0).columns)
    except Exception:
        return False
    return {"i2c_count", "c2i_count"}.issubset(columns)


def _build_command(
    args: argparse.Namespace,
    setting: dict[str, Any],
    target_stats: Path,
    *,
    evaluation_split: str,
    evaluation_baseline_subset: str,
) -> list[str]:
    cmd = [
        args.python_bin,
        "-m",
        "pipeline.stage07_singleton_causal_evaluation",
        "--task_module",
        str(setting["task_module"]),
        "--ai_model",
        str(setting["model_id"]),
        "--rules_dir",
        str(target_stats.parents[1]),
        "--features_scores_dir",
        str(setting["model_root"] / "feature_report"),
        "--circuit_agonists_path",
        str(setting["circuit_agonists_path"]),
        "--search_epsilon",
        str(setting["search_epsilon"]),
        "--stats_dirname",
        str(target_stats.name),
        "--intervention",
        str(setting["intervention"]),
        "--stats_only",
        "--evaluation_split",
        str(evaluation_split),
        "--evaluation_baseline_subset",
        str(evaluation_baseline_subset),
        # scores.csv already contains the materialized singleton events for this
        # run. A stats-only refresh must never silently apply Stage-7's default
        # 512-row cap a second time.
        "--sampling_max_points",
        "0",
        "--skip_agonist_metric_stats",
    ]
    ranking = target_stats / "frozen_candidate_ranking.csv"
    if ranking.is_file():
        cmd.extend(["--candidate_ranking_csv", str(ranking)])
    if setting["decode_only"]:
        cmd.append("--decode_only")
    return cmd


def _run_target(
    args: argparse.Namespace,
    setting: dict[str, Any],
    target_stats: Path,
    *,
    population_label: str,
    evaluation_split: str,
) -> str:
    materialized_scores = target_stats / "scores.csv"
    if not materialized_scores.is_file():
        if _directional_metrics_complete(target_stats):
            print(
                f"[directional stats][{population_label}] already complete; no materialized scores needed: "
                f"{target_stats}"
            )
            return "already-complete"
        raise FileNotFoundError(
            f"{population_label} target has no materialized singleton-event table: {materialized_scores}. "
            "The directional metrics are also incomplete, so this run cannot be CPU-only rebuilt."
        )

    baseline_subset = _baseline_subset_from_scope(target_stats)
    cmd = _build_command(
        args,
        setting,
        target_stats,
        evaluation_split=evaluation_split,
        evaluation_baseline_subset=baseline_subset,
    )
    print(
        f"[directional stats][{population_label}] {setting['label']}\n"
        f"  reference: {setting['reference_stats']}\n"
        f"  rebuild target: {target_stats}\n"
        f"  preserved split: {evaluation_split}; baseline subset: {baseline_subset}"
    )
    if args.dry_run:
        print("  command:", " ".join(str(part) for part in cmd))
        return "dry-run"

    scope_path = target_stats / "evaluation_scope.json"
    scope_before = _read_scope(target_stats)

    # stage07 stats-only intentionally rewrites scores.csv to its selected
    # evaluation subset. For a repair pass that is undesirable: scores.csv is
    # the authoritative materialized event table and may be needed to rebuild a
    # different scope later. Restore it after aggregate statistics are written.
    with tempfile.TemporaryDirectory(prefix="directional_stats_") as tmp:
        backup = Path(tmp) / "scores.csv"
        shutil.copy2(materialized_scores, backup)
        subprocess.run(cmd, cwd=CODE_ROOT, check=True)
        shutil.copy2(backup, materialized_scores)

    if isinstance(scope_before, dict):
        scope_after = dict(scope_before)
        scope_after["directional_metrics_regenerated"] = True
        scope_after["directional_metrics_regeneration_mode"] = "stats_only_from_materialized_singleton_events"
        scope_after["directional_metrics_schema"] = "heldout-set-metrics-v3-directional"
        scope_after["directional_metrics_regeneration_population"] = population_label
        scope_path.write_text(json.dumps(scope_after, indent=2, ensure_ascii=False), encoding="utf-8")

    return "rebuilt"


def _primary_targets(args: argparse.Namespace, helpers: Any, data_root: Path) -> list[tuple[dict[str, Any], Path, str]]:
    table_path = Path(args.primary_table).expanduser().resolve()
    if not table_path.is_file():
        raise FileNotFoundError(f"Primary-table population requested but table does not exist: {table_path}")
    table = pd.read_csv(table_path)
    required = {"task", "model", "phase", "stats_dir"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{table_path} is missing columns: {missing}")

    out: list[tuple[dict[str, Any], Path, str]] = []
    for index in helpers.selected_rows(args.rows, len(table)):
        setting = helpers.setting_from_row(
            index,
            table.iloc[index],
            data_root,
            PROJECT_ROOT / "results",
            evaluation_split=args.evaluation_split,
        )
        # primary_table.csv is manuscript-facing and normally points at held-out
        # stats. Rebuild the reported held-out directory consumed by stages 02/04/05.
        reported_stats = setting["reported_stats"]
        reference_stats = setting["reference_stats"]
        target_stats = reported_stats if reported_stats != reference_stats else setting["heldout_stats"]
        out.append((setting, target_stats, str(args.evaluation_split)))
    return out


def _rq1_targets(args: argparse.Namespace, helpers: Any, data_root: Path) -> tuple[list[tuple[dict[str, Any], Path, str]], int]:
    from studies.overtopping.analysis.stage06_competence_vs_overtopping_figures import (
        discover_rq1_manuscript_points,
        locate_results_root,
    )

    root = locate_results_root(data_root)
    points = discover_rq1_manuscript_points(root)
    out: list[tuple[dict[str, Any], Path, str]] = []
    empty_count = 0
    for index, point in enumerate(points):
        run_dir = (root / point.source_path).resolve()
        if not (run_dir / "flip_stats_global.json").is_file():
            # This is a genuine empty candidate set selected by the RQ1 scanner.
            # The plotting contract defines U(empty)=N_t(empty)=0, so there are
            # no singleton events to regenerate.
            empty_count += 1
            print(
                f"[directional stats][rq1-all-settings] exact empty-candidate zero; no rebuild needed: "
                f"{point.task}/{point.model} {point.phase} {point.baseline} -> {run_dir.name}"
            )
            continue

        row = pd.Series(
            {
                "task": point.task,
                "model": point.model,
                "phase": point.phase,
                "stats_dir": str(run_dir),
            }
        )
        setting = helpers.setting_from_row(
            index,
            row,
            root,
            PROJECT_ROOT / "results",
            evaluation_split="all",  # only used to form an unused heldout path here
        )
        split = _infer_reference_split(run_dir, run_dir / "scores.csv")
        out.append((setting, run_dir, split))
    return out, empty_count


def main() -> None:
    args = parse_args()
    helpers = _load_holdout_helpers(CODE_ROOT)
    data_root = Path(args.data_root).expanduser().resolve()

    primary_targets: list[tuple[dict[str, Any], Path, str]] = []
    rq1_targets: list[tuple[dict[str, Any], Path, str]] = []
    rq1_empty = 0
    if args.population in {"primary", "both"}:
        primary_targets = _primary_targets(args, helpers, data_root)
    if args.population in {"rq1-all-settings", "both"}:
        rq1_targets, rq1_empty = _rq1_targets(args, helpers, data_root)

    results = {"rebuilt": 0, "already-complete": 0, "dry-run": 0}
    seen: set[Path] = set()
    for label, targets in (("primary", primary_targets), ("rq1-all-settings", rq1_targets)):
        for setting, target_stats, split in targets:
            target_stats = target_stats.resolve()
            # The two populations normally target different directories
            # (held-out vs reference), but keep the repair idempotent if a custom
            # primary table points directly at the same reference run.
            if target_stats in seen:
                print(f"[directional stats] duplicate target skipped: {target_stats}")
                continue
            seen.add(target_stats)
            status = _run_target(
                args,
                setting,
                target_stats,
                population_label=label,
                evaluation_split=split,
            )
            results[status] = results.get(status, 0) + 1

    action = "Validated" if args.dry_run else "Processed"
    print(
        f"{action} directional-stat targets: primary={len(primary_targets)}, "
        f"rq1_reference_nonempty={len(rq1_targets)}, rq1_empty_exact_zero={rq1_empty}, "
        f"rebuilt={results.get('rebuilt', 0)}, already_complete={results.get('already-complete', 0)}, "
        f"dry_run={results.get('dry-run', 0)}."
    )


if __name__ == "__main__":
    main()
