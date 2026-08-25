#!/usr/bin/env python3
"""Run strict held-out re-estimation for every row selected from primary_table.csv.

The primary table is the authoritative run manifest. No task/model/checkpoint run
list is embedded here. Each row's stats_dir determines the source candidate set,
model root, intervention, phase, CHA threshold, and circuit-output locations.

Stages
------
holdout   Re-evaluate the frozen CHA-selected candidate set on the selected evaluation split.
dominance Evaluate direction-matched and submitted Dom^(1) on that same split.
all       Run both stages; the paired manuscript audit is generated when the split is test.
plan      Validate/infer all paths and write the manifest without running models.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from studies.overtopping.analysis.lib.catalog import TASK_MODULES, infer_intervention, selected_rows, slug
from typing import Any

import pandas as pd


from core.project_paths import CODE_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary_table", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--rows", default="all", help="Comma-separated zero-based row indices, or all.")
    p.add_argument("--stage", choices=["holdout", "dominance", "all", "plan"], default="all")
    p.add_argument("--evaluation_split", choices=["test", "train", "all"], default="test")
    p.add_argument("--python_bin", default=sys.executable)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--neuron_batch_size", type=int, default=8)
    p.add_argument("--sampling_max_points", type=int, default=10000)
    p.add_argument("--global_n_clusters", type=int, default=64)
    p.add_argument("--points_to_use_for_mean_ablation", type=int, default=2048)
    p.add_argument("--rho", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--ai_model_cache_dir", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--allow_partial",
        action="store_true",
        help="Write a partial manifest instead of failing when a primary row cannot be run.",
    )
    return p.parse_args()


def remap_stats_dir(raw: object, data_root: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if path.exists():
        return path.resolve()
    text = path.as_posix()
    for marker in ("/results/", "/data/"):
        if marker in text:
            return (data_root / text.split(marker, 1)[1]).resolve()
    if not path.is_absolute():
        candidate = (data_root / path).resolve()
        if candidate.exists() or path.parts[:1] not in [("data",), ("results",)]:
            return candidate
    return path.resolve()


def infer_search_epsilon(stats_name: str) -> float:
    match = re.search(r"-tau([0-9]+(?:\.[0-9]+)?)", str(stats_name))
    return float(match.group(1)) if match else 0.2


def strict_scope_ok(
    stats_dir: Path,
    *,
    expected_sampling_max_points: int | None = None,
    expected_split: str = "test",
) -> bool:
    needed = [
        stats_dir / "flip_stats_global.json",
        stats_dir / "flip_stats_by_neuron.csv",
        stats_dir / "scores.csv",
        stats_dir / "evaluation_scope.json",
    ]
    if not all(path.exists() for path in needed):
        return False
    try:
        scope = json.loads((stats_dir / "evaluation_scope.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    expected_split = str(expected_split).strip().lower()
    if expected_split not in {"test", "train", "all"}:
        return False
    base_ok = (
        bool(scope.get("holdout_test_only")) == (expected_split == "test")
        and str(scope.get("final_statistics_split", "")).lower() == expected_split
        and str(scope.get("mean_replacement_reference_split", "")).lower() == "train"
        and str(scope.get("candidate_discovery_split", "train")).lower() == "train"
    )
    if not base_ok:
        return False
    if expected_sampling_max_points is None:
        return True
    try:
        n_rows_ablated = int(scope.get("n_rows_ablated"))
        recorded_cap = int(scope.get("sampling_max_points"))
    except (TypeError, ValueError):
        return False
    return (
        scope.get("use_spectral_sampling") is True
        and str(scope.get("sampling_pool_split", "")).lower() == expected_split
        and recorded_cap == int(expected_sampling_max_points)
        and n_rows_ablated <= int(expected_sampling_max_points)
    )


def _evaluation_stats_dir(
    reference_stats: Path,
    evaluation_split: str,
    *,
    sampling_max_points: int = 10000,
) -> Path:
    evaluation_split = str(evaluation_split).strip().lower()
    if evaluation_split == "test":
        suffix = "-heldout_test"
    elif evaluation_split == "train":
        suffix = "-eval_train"
    elif evaluation_split == "all":
        # This helper starts from an existing reference run, so use an explicit
        # side directory rather than overwriting that reference output.
        suffix = "-eval_all"
    else:
        raise ValueError(f"Unsupported evaluation split {evaluation_split!r}")
    # 10,000 is the historical implicit default. Preserve the established
    # dirname for that value; encode only explicit/non-default caps because
    # they change which evaluation rows are selected.
    cap = int(sampling_max_points)
    cap_suffix = "" if cap == 10000 else f"-cap{cap}"
    return reference_stats.parent / f"{reference_stats.name}{suffix}{cap_suffix}"


def setting_from_row(
    index: int,
    row: pd.Series,
    data_root: Path,
    out_root: Path,
    *,
    evaluation_split: str = "test",
    sampling_max_points: int = 10000,
) -> dict[str, Any]:
    reference_stats = remap_stats_dir(row["stats_dir"], data_root)
    if reference_stats.name.endswith("-heldout_test"):
        raise ValueError(f"Row {index} stats_dir must name the reported/reference run, not a held-out output")
    if "-agonist_neurons" not in reference_stats.name:
        raise ValueError(
            f"Row {index} stats_dir name does not encode circuit and agonist outputs: {reference_stats.name}"
        )
    model_root = reference_stats.parents[3]
    try:
        relative = model_root.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"Cannot locate model root {model_root} under data root {data_root}") from exc
    if len(relative.parts) < 3:
        raise ValueError(f"Unexpected model-root layout for row {index}: {model_root}")
    task_dir = relative.parts[0]
    model_id = Path(*relative.parts[1:]).as_posix()
    if task_dir not in TASK_MODULES:
        raise ValueError(f"No task-module mapping for primary-table task directory {task_dir!r}")

    circuit_label, bag_tail = reference_stats.name.split("-agonist_neurons", 1)
    bag_label = "agonist_neurons" + bag_tail
    heldout_stats = _evaluation_stats_dir(
        reference_stats,
        evaluation_split,
        sampling_max_points=sampling_max_points,
    )
    circuit_root = (
        model_root
        / "neural_circuit_discovery_results"
        / "eap_ig_inputs"
        / circuit_label
    )
    input_data_dir = circuit_root / "neural_circuits"

    # Script 6 writes the agonist outputs beside ``neural_circuits`` in the
    # current pipeline (DISCOVERY_OUT_DIR/BAG_LABEL).  Some older exports used
    # a bag_of_rules subdirectory, so resolve the exact bag label from the
    # filesystem rather than assuming one layout.  The primary-table stats_dir
    # still determines circuit_label and bag_label; this is not a run list.
    agonist_candidates = [
        circuit_root / bag_label,
        circuit_root / "bag_of_rules" / bag_label,
        input_data_dir / bag_label,
    ]
    existing_agonists = [path for path in agonist_candidates if path.exists()]
    if len(existing_agonists) == 1:
        circuit_agonists_path = existing_agonists[0]
    elif len(existing_agonists) > 1:
        # Prefer the current pipeline layout if duplicate exported trees exist.
        circuit_agonists_path = next(
            (path for path in agonist_candidates if path in existing_agonists),
            existing_agonists[0],
        )
    else:
        # Preserve the current-layout path in diagnostics so the missing-path
        # error tells the user exactly what was inferred from primary_table.csv.
        circuit_agonists_path = agonist_candidates[0]
    label = f"row {index:02d}: {row.get('task')} | {row.get('model')} | {row.get('phase')}"
    row_slug = f"{index:02d}_{slug(row.get('task'))}_{slug(row.get('model'))}_{slug(row.get('phase'))}"
    phase = str(row.get("phase", ""))
    intervention = infer_intervention(reference_stats.name)
    points = 2048 if "donor" in intervention else 256
    return {
        "row_index": index,
        "label": label,
        "slug": row_slug,
        "task": row.get("task"),
        "model": row.get("model"),
        "phase": phase,
        "score": row.get("score"),
        "task_dir": task_dir,
        "task_module": TASK_MODULES[task_dir],
        "model_id": model_id,
        "model_root": model_root,
        "reference_stats": reference_stats,
        "heldout_stats": heldout_stats,
        "evaluation_split": str(evaluation_split),
        "input_data_dir": input_data_dir,
        "circuit_agonists_path": circuit_agonists_path,
        "circuit_label": circuit_label,
        "bag_label": bag_label,
        "intervention": intervention,
        "search_epsilon": infer_search_epsilon(reference_stats.name),
        "decode_only": phase.strip().lower() in {"out", "output", "output-only"}
        or "decode_only" in reference_stats.name.lower(),
        "points_to_use_for_mean_ablation": points,
        "dominance_dir": out_root / "dominance" / row_slug,
    }


def required_source_paths(setting: dict[str, Any]) -> list[Path]:
    return [
        setting["reference_stats"] / "flip_stats_global.json",
        setting["reference_stats"] / "flip_stats_by_neuron.csv",
        setting["reference_stats"] / "scores.csv",
        setting["model_root"] / "feature_report" / "scores.csv",
        setting["input_data_dir"] / "dataset_info.json",
        setting["circuit_agonists_path"],
    ]


def run_holdout(setting: dict[str, Any], args: argparse.Namespace, repo_dir: Path) -> str:
    if not args.force and strict_scope_ok(
        setting["heldout_stats"],
        expected_sampling_max_points=args.sampling_max_points,
        expected_split=args.evaluation_split,
    ):
        print(f"[reuse heldout] {setting['label']} -> {setting['heldout_stats']}")
        return "reused"
    cmd = [
        args.python_bin, "-m", "pipeline.stage07_refine_neuron_anchored_rules",
        "--task_module", str(setting["task_module"]),
        "--ai_model", str(setting["model_id"]),
        "--rules_dir", str(setting["model_root"] / "rule_extraction_results" / "neuron_flip_rules"),
        "--features_scores_dir", str(setting["model_root"] / "feature_report"),
        "--circuit_agonists_path", str(setting["circuit_agonists_path"]),
        "--search_epsilon", str(setting["search_epsilon"]),
        "--batch_size", str(args.batch_size),
        "--neuron_batch_size", str(args.neuron_batch_size),
        "--stats_dirname", str(setting["heldout_stats"].name),
        "--use_spectral_sampling",
        "--sampling_max_points", str(args.sampling_max_points),
        "--global_n_clusters", str(args.global_n_clusters),
        "--spectral_cache_dir", str(repo_dir / "cache" / setting["task_dir"]),
        "--points_to_use_for_mean_ablation", str(
            max(args.points_to_use_for_mean_ablation, setting["points_to_use_for_mean_ablation"])
        ),
        "--intervention", str(setting["intervention"]),
        "--only_unique_datapoints_in_shap",
        "--exclude_discovery_rows_from_final_stats",
        "--evaluation_split", str(args.evaluation_split),
        "--skip_agonist_metric_stats",
        "--no_tqdm_batches",
        "--seed", str(args.seed + setting["row_index"]),
    ]
    if setting["decode_only"]:
        cmd.append("--decode_only")
    if args.ai_model_cache_dir:
        cmd.extend(["--ai_model_cache_dir", args.ai_model_cache_dir])
    print(f"[run heldout] {setting['label']}")
    subprocess.run(cmd, cwd=repo_dir, check=True)
    if not strict_scope_ok(
        setting["heldout_stats"],
        expected_sampling_max_points=args.sampling_max_points,
        expected_split=args.evaluation_split,
    ):
        raise RuntimeError(
            f"Evaluation run did not produce the required split={args.evaluation_split} scope: "
            f"{setting['heldout_stats']}"
        )
    return "ran"


def dominance_complete(setting: dict[str, Any]) -> bool:
    path = setting["dominance_dir"] / "dominance_summary.csv"
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
        return bool((pd.to_numeric(frame.get("m"), errors="coerce") == 1).any())
    except Exception:
        return False


def run_dominance(setting: dict[str, Any], args: argparse.Namespace, repo_dir: Path) -> str:
    if not strict_scope_ok(
        setting["heldout_stats"],
        expected_sampling_max_points=args.sampling_max_points,
        expected_split=args.evaluation_split,
    ):
        raise FileNotFoundError(
            f"Split-compatible statistics missing for {setting['label']} "
            f"(split={args.evaluation_split}): {setting['heldout_stats']}"
        )
    if not args.force and dominance_complete(setting):
        print(f"[reuse dominance] {setting['label']} -> {setting['dominance_dir']}")
        return "reused"
    cmd = [
        args.python_bin, "-m", "studies.overtopping.analysis.group_dominance",
        "--input_data_dir", str(setting["input_data_dir"]),
        "--candidate_flip_stats_path", str(setting["heldout_stats"] / "flip_stats_by_neuron.csv"),
        "--singleton_scores_path", str(setting["heldout_stats"] / "scores.csv"),
        "--out_dir", str(setting["dominance_dir"]),
        "--task_module", str(setting["task_module"]),
        "--ai_model", str(setting["model_id"]),
        "--evaluation_split", str(args.evaluation_split),
        "--intervention", str(setting["intervention"]),
        "--batch_size", str(args.batch_size),
        "--points_to_use_for_mean_ablation", str(
            max(args.points_to_use_for_mean_ablation, setting["points_to_use_for_mean_ablation"])
        ),
        "--m", "1",
        "--rho", str(args.rho),
        "--seed", str(args.seed + setting["row_index"]),
    ]
    if setting["decode_only"]:
        cmd.append("--decode_only")
    if args.ai_model_cache_dir:
        cmd.extend(["--ai_model_cache_dir", args.ai_model_cache_dir])
    print(f"[run dominance] {setting['label']}")
    subprocess.run(cmd, cwd=repo_dir, check=True)
    return "ran"


def write_manifest(settings: list[dict[str, Any]], statuses: dict[int, dict[str, str]], out_dir: Path) -> None:
    rows = []
    for s in settings:
        rows.append({
            "row_index": s["row_index"],
            "label": s["label"],
            "task": s["task"],
            "model": s["model"],
            "phase": s["phase"],
            "score": s["score"],
            "task_dir": s["task_dir"],
            "model_id": s["model_id"],
            "reference_stats_dir": str(s["reference_stats"]),
            "heldout_stats_dir": str(s["heldout_stats"]),
            "evaluation_split": s.get("evaluation_split", "test"),
            "circuit_agonists_path": str(s["circuit_agonists_path"]),
            "intervention": s["intervention"],
            "search_epsilon": s["search_epsilon"],
            "decode_only": s["decode_only"],
            "dominance_dir": str(s["dominance_dir"]),
            "holdout_status": statuses.get(s["row_index"], {}).get("holdout", "not_requested"),
            "dominance_status": statuses.get(s["row_index"], {}).get("dominance", "not_requested"),
        })
    pd.DataFrame(rows).to_csv(out_dir / "primary_holdout_manifest.csv", index=False)
    (out_dir / "primary_holdout_manifest.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )


def summarize(settings: list[dict[str, Any]], args: argparse.Namespace, repo_dir: Path, include_dom: bool) -> None:
    cmd = [args.python_bin, "-m", "studies.overtopping.analysis.stage04_analyze_primary_metrics", "holdout"]
    for s in settings:
        cmd.extend(["--run", f"{s['label']}={s['heldout_stats']}"])
        cmd.extend(["--reference", f"{s['label']}={s['reference_stats']}"])
        if include_dom:
            cmd.extend(["--dom", f"{s['label']}={s['dominance_dir']}"])
    cmd.extend(["--out_dir", str(Path(args.out_dir).expanduser().resolve())])
    subprocess.run(cmd, cwd=repo_dir, check=True)


def verify_complete(settings: list[dict[str, Any]], out_dir: Path, include_dom: bool) -> dict[str, Any]:
    summary_path = out_dir / "heldout_reestimation_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    frame = pd.read_csv(summary_path)
    expected = len(settings)
    problems: list[str] = []
    if len(frame) != expected:
        problems.append(f"summary rows={len(frame)} but expected={expected}")
    if "candidate_set_exact_match" in frame:
        bad = frame.loc[frame["candidate_set_exact_match"].astype(str).str.lower() != "true", "label"].tolist()
        if bad:
            problems.append(f"candidate-set mismatch: {bad}")
    if include_dom:
        if "Dom1" not in frame or "Dom1_matched" not in frame:
            problems.append("dominance columns are missing")
        else:
            missing_dom = frame.loc[frame["Dom1_status"].isna(), "label"].tolist() if "Dom1_status" in frame else []
            if missing_dom:
                problems.append(f"dominance missing: {missing_dom}")
    payload = {
        "n_primary_rows_selected": expected,
        "n_summary_rows": int(len(frame)),
        "all_rows_present": len(frame) == expected,
        "all_candidate_sets_exact_match": not any("candidate-set mismatch" in x for x in problems),
        "dominance_included": include_dom,
        "problems": problems,
    }
    (out_dir / "primary_holdout_coverage.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if problems:
        raise RuntimeError("Primary held-out coverage verification failed: " + "; ".join(problems))
    return payload


def main() -> None:
    args = parse_args()
    repo_dir = CODE_ROOT
    table_path = Path(args.primary_table).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(table_path)
    required = {"task", "model", "phase", "stats_dir"}
    missing_cols = sorted(required - set(table.columns))
    if missing_cols:
        raise ValueError(f"{table_path} is missing columns: {missing_cols}")
    indices = selected_rows(args.rows, len(table))
    settings: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index in indices:
        try:
            setting = setting_from_row(
                index,
                table.iloc[index],
                data_root,
                out_dir,
                evaluation_split=args.evaluation_split,
                sampling_max_points=args.sampling_max_points,
            )
            missing = [str(p) for p in required_source_paths(setting) if not p.exists()]
            if missing:
                raise FileNotFoundError(json.dumps({"row_index": index, "missing": missing}, indent=2))
            settings.append(setting)
        except Exception as exc:
            failures.append({"row_index": index, "error": str(exc)})
            if not args.allow_partial:
                raise
    if failures:
        (out_dir / "primary_holdout_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    if not settings:
        raise RuntimeError("No runnable primary-table rows")

    statuses: dict[int, dict[str, str]] = {}
    write_manifest(settings, statuses, out_dir)
    if args.stage == "plan":
        print(f"Validated {len(settings)} primary-table rows; plan written to {out_dir}")
        return

    for setting in settings:
        statuses.setdefault(setting["row_index"], {})
        if args.stage in {"holdout", "all"}:
            statuses[setting["row_index"]]["holdout"] = run_holdout(setting, args, repo_dir)
        if args.stage in {"dominance", "all"}:
            statuses[setting["row_index"]]["dominance"] = run_dominance(setting, args, repo_dir)
        write_manifest(settings, statuses, out_dir)

    include_dom = args.stage in {"dominance", "all"}
    if args.evaluation_split == "test":
        summarize(settings, args, repo_dir, include_dom=include_dom)
        coverage = verify_complete(settings, out_dir, include_dom=include_dom)
    else:
        coverage = {
            "n_primary_rows_selected": len(settings),
            "evaluation_split": args.evaluation_split,
            "paired_holdout_report_generated": False,
            "reason": "studies.overtopping.analysis.stage04_analyze_primary_metrics holdout is a test-split manuscript audit",
            "dominance_included": include_dom,
        }
        (out_dir / "primary_evaluation_coverage.json").write_text(
            json.dumps(coverage, indent=2), encoding="utf-8"
        )
    if failures and not args.allow_partial:
        raise RuntimeError(f"{len(failures)} primary rows failed")
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
