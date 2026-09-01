#!/usr/bin/env python3
"""One-off migration of persisted overtopping artifacts to canonical formats.

This script is intentionally external to the repository. It migrates persisted
artifact formats that current runtime code no longer accepts, then verifies that
no recognized legacy artifacts remain.

The script never loads an LLM. The only computational repair is the directional
singleton v2 -> v3 statistics refresh, which runs Stage 7 in --stats_only mode
against already materialized scores.csv and restores that scores.csv afterward.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LEGACY_GROUP_SCHEMA = "group-intervention-batch-v1"
CANONICAL_GROUP_SCHEMA = "group-intervention-batch-v2"
GROUP_CONTEXT_SCHEMA = "simultaneous-group-eval-v2"
CANONICAL_SINGLETON_SCHEMA = "heldout-set-metrics-v3-directional"
LEGACY_SINGLETON_SCHEMA = "heldout-set-metrics-v2"
STAGE07_RESUME_SCHEMA = 1

THRESHOLD_RUNTIME_KEYS = {
    "batch_size",
    "spiking_global_n_clusters",
    "spectral_cache_dir",
    "spectral_embedding_batch_size",
    "ai_model_cache_dir",
}


VERBOSE = False


def log_action(message: str) -> None:
    if VERBOSE:
        print(message)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", required=True, help="Repository root containing code/, data/, cache/.")
    p.add_argument("--apply", action="store_true", help="Apply changes. Default is a dry run.")
    # Unverifiable *cache* shards are disposable by definition and cannot be
    # accepted by the canonical runtime.  Purge them by default.  The legacy
    # flag remains accepted so previously suggested commands keep working.
    p.add_argument("--purge-unverifiable-caches", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--keep-unverifiable-caches", action="store_true",
        help="Debug only: leave unverifiable legacy cache shards in place (the canonical runtime will ignore them).",
    )
    p.add_argument("--verbose", action="store_true", help="Print every file-level migration action.")
    p.add_argument("--python-bin", default=sys.executable, help="Python used for CPU-only Stage-7 stats refresh.")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any], *, apply: bool) -> None:
    log_action(f"[migrate] write JSON: {path}")
    if not apply:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def remove_path(path: Path, *, apply: bool) -> None:
    if not path.exists():
        return
    log_action(f"[migrate] remove: {path}")
    if not apply:
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def copy_then_remove(src: Path, dst: Path, *, apply: bool) -> None:
    log_action(f"[migrate] move canonical: {src} -> {dst}")
    if not apply:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha256(src) != sha256(dst):
            raise RuntimeError(f"Conflicting canonical/legacy files: {src} vs {dst}")
    else:
        shutil.copy2(src, dst)
    src.unlink()


def prune_empty_dirs(start: Path, stop: Path, *, apply: bool) -> None:
    if not apply:
        return
    cur = start
    stop = stop.resolve()
    while cur.exists() and cur.resolve() != stop:
        try:
            cur.rmdir()
        except OSError:
            break
        cur = cur.parent


def migrate_rebuild_only_cache(project: Path, *, apply: bool) -> int:
    count = 0
    old = project / "cache" / "threshold_events"
    if old.exists():
        remove_path(old, apply=apply)
        count += 1
    for path in project.rglob("threshold_spiking_experiment.json.pre-canonical-migration.bak"):
        remove_path(path, apply=apply)
        count += 1
    return count


NORMAL_TASK_CANONICAL_BASE_FIELDS = {
    "normal_task_population_size",
    "normal_task_scan_max_rows",
    "normal_task_sampling_strategy",
    "normal_task_population_mode",
}

NORMAL_TASK_SAMPLED_ONLY_FIELDS = {
    "normal_task_candidate_order_seed",
    "normal_task_stratification",
    "normal_task_stratum",
}


@contextlib.contextmanager
def _temporary_environ(updates: dict[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        os.environ.update({key: str(value) for key, value in updates.items()})
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _normal_task_cache_needs_metadata_migration(obj: Any) -> bool:
    if not isinstance(obj, list) or not obj or not all(isinstance(row, dict) for row in obj):
        return False
    for row in obj:
        if not NORMAL_TASK_CANONICAL_BASE_FIELDS.issubset(row):
            return True
        strategy = row.get("normal_task_sampling_strategy")
        if strategy == "full_distribution":
            # Full-population evaluation is not stratified. Strip metadata
            # written by the earlier over-eager migration as well.
            if any(field in row for field in NORMAL_TASK_SAMPLED_ONLY_FIELDS):
                return True
        elif not NORMAL_TASK_SAMPLED_ONLY_FIELDS.issubset(row):
            return True
    return False


def _common_row_value(rows: list[dict[str, Any]], key: str) -> Any | None:
    values = {repr(row[key]): row[key] for row in rows if key in row}
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"legacy normal-task cache has inconsistent {key}: {sorted(values)}")
    return next(iter(values.values()))


def _normal_task_cache_coordinates(project: Path, path: Path) -> tuple[str, str, Path, Path]:
    root = (project / "cache" / "poisoning").resolve()
    try:
        rel = path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"normal-task cache is outside cache/poisoning: {path}") from exc
    if len(rel.parts) < 3:
        raise ValueError(f"cannot infer poisoning task/run from cache path: {path}")
    task, run_id = rel.parts[0], rel.parts[1]
    if task not in {"grammar", "arithmetic"}:
        raise ValueError(f"unsupported poisoning normal-task cache task {task!r}: {path}")
    run_dir = project / "data" / "poisoning" / task / run_id
    cohort = run_dir / "02_evaluation_cohorts" / f"{task}_causal_validation.jsonl"
    if not cohort.is_file():
        raise FileNotFoundError(f"normal-task source cohort is missing: {cohort}")
    return task, run_id, run_dir, cohort


def _normal_task_content_key(task: str, row: dict[str, Any]) -> tuple[Any, ...]:
    if task == "grammar":
        sentence = str(row.get("original_sentence", row.get("sentence", ""))).strip()
        gold = row.get("original_is_acceptable", row.get("is_acceptable"))
        return (sentence, bool(gold))
    prompt = str(row.get("original_prompt", row.get("prompt", ""))).strip()
    answer = row.get("correct_answer_numeric", row.get("correct_answer"))
    try:
        answer_key: Any = float(answer)
    except (TypeError, ValueError):
        answer_key = str(answer)
    return (prompt, answer_key)


def _normal_task_identity_mode(
    task: str,
    expected_rows: list[dict[str, Any]],
    legacy_rows: list[dict[str, Any]],
) -> tuple[str, Any]:
    """Choose a stable source identity that survives historical candidate reordering."""
    for field in ("source_row_id", "eval_example_id", "example_id"):
        if not all(row.get(field) is not None for row in expected_rows):
            continue
        if not all(row.get(field) is not None for row in legacy_rows):
            continue
        expected = [str(row[field]) for row in expected_rows]
        legacy = [str(row[field]) for row in legacy_rows]
        if len(set(expected)) == len(expected) and len(set(legacy)) == len(legacy):
            return field, lambda row, f=field: str(row[f])

    expected = [_normal_task_content_key(task, row) for row in expected_rows]
    legacy = [_normal_task_content_key(task, row) for row in legacy_rows]
    if len(set(expected)) == len(expected) and len(set(legacy)) == len(legacy):
        return "content", lambda row: _normal_task_content_key(task, row)
    raise ValueError("cannot establish a unique stable source identity for legacy normal-task rows")


def _normal_task_environment(
    *,
    task: str,
    cohort: Path,
    run_config: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, str], int, int, float]:
    seed_var = "GRAMMAR_BACKDOOR_TASK_SEED" if task == "grammar" else "ARITHMETIC_BACKDOOR_TASK_SEED"
    num_var = "GRAMMAR_BACKDOOR_NUM_EXAMPLES" if task == "grammar" else "ARITHMETIC_BACKDOOR_NUM_EXAMPLES"
    dataset_var = "GRAMMAR_BACKDOOR_DATASET_PATH" if task == "grammar" else "ARITHMETIC_BACKDOOR_DATASET_PATH"
    source_var = "GRAMMAR_BACKDOOR_SOURCE_FILTER" if task == "grammar" else "ARITHMETIC_BACKDOOR_SOURCE_FILTER"
    target_var = "GRAMMAR_BACKDOOR_TARGET_LABEL" if task == "grammar" else "ARITHMETIC_BACKDOOR_TARGET_ANSWER"

    # Match the canonical runtime defaults. Explicit caller overrides remain authoritative.
    candidate_seed = int(os.environ.get(seed_var, "42"))
    scan_max_rows = max(0, int(os.environ.get("NORMAL_TASK_SCAN_MAX_ROWS", "10000")))
    old_holdout_seed = _common_row_value(rows, "poisoning_holdout_seed")
    holdout_seed = int(os.environ.get("POISONING_HOLDOUT_SEED", run_config.get("seed", old_holdout_seed or 13)))
    old_fraction = _common_row_value(rows, "poisoning_holdout_test_fraction")
    holdout_fraction = float(
        os.environ.get("POISONING_HOLDOUT_TEST_FRACTION", old_fraction if old_fraction is not None else 1.0 / 3.0)
    )

    if task == "grammar":
        target = os.environ.get(target_var, str(run_config.get("target_label", "acceptable")))
    else:
        target = os.environ.get(target_var, str(run_config.get("target_answer", "0")))

    updates = {
        dataset_var: str(cohort),
        source_var: "all",
        seed_var: str(candidate_seed),
        num_var: os.environ.get(num_var, "0"),
        target_var: target,
        "NORMAL_TASK_SCAN_MAX_ROWS": str(scan_max_rows),
        "POISONING_HOLDOUT_SEED": str(holdout_seed),
        "POISONING_HOLDOUT_TEST_FRACTION": repr(holdout_fraction),
    }
    for env_name, config_key in (
        ("POISONING_CONTROL_MARKER", "control_marker"),
        ("POISONING_TRIGGER_MARKER", "trigger_marker"),
        ("POISONING_SHAM_MARKER", "sham_marker"),
    ):
        if config_key in run_config:
            updates[env_name] = str(run_config[config_key])
    return updates, candidate_seed, scan_max_rows, holdout_fraction


def _migrate_normal_task_pickle(project: Path, path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task, _run_id, run_dir, cohort = _normal_task_cache_coordinates(project, path)
    config_path = find_run_config(run_dir)
    run_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path is not None else {}
    env_updates, candidate_seed, scan_max_rows, holdout_fraction = _normal_task_environment(
        task=task,
        cohort=cohort,
        run_config=run_config,
        rows=rows,
    )
    holdout_seed = int(env_updates["POISONING_HOLDOUT_SEED"])

    code_dir = str(project / "code")
    added_path = code_dir not in sys.path
    if added_path:
        sys.path.insert(0, code_dir)
    try:
        with _temporary_environ(env_updates):
            module = importlib.import_module(f"studies.poisoning.tasks.{task}")
            runtime = importlib.import_module("studies.poisoning.lib.backdoor_runtime")
            trigger_lift = importlib.import_module("studies.poisoning.lib.trigger_lift")

            if task == "grammar":
                rows_full = list(module._load_heldout_rows())
                stratification = "dataset_x_five_word_length_bin_x_gold_label"
            else:
                rows_full = list(module._build_prompt_pool())
                stratification = "operator_group_x_gold_target_side"

            expected, sample_meta = runtime._select_normal_task_rows(
                rows_full,
                scan_max_rows=scan_max_rows,
                seed=candidate_seed,
                stratum_key=module.normal_task_stratum_key,
                stratification_name=stratification,
            )
            expected = [dict(row) for row in expected]
            identity_name, identity = _normal_task_identity_mode(task, expected, rows)
            legacy_by_identity = {identity(row): row for row in rows}
            missing = [identity(row) for row in expected if identity(row) not in legacy_by_identity]
            if missing:
                raise ValueError(
                    f"legacy cache lacks {len(missing)}/{len(expected)} rows required by the current "
                    f"normal-task sample using {identity_name} identity"
                )

            population_mode = (
                "full_heldout_distribution"
                if len(expected) == len(rows_full)
                else "deterministic_proportional_stratified_sample"
            )
            sampled = sample_meta["sampling_strategy"] != "full_distribution"
            migrated: list[dict[str, Any]] = []
            for expected_row in expected:
                row = dict(legacy_by_identity[identity(expected_row)])
                if "backdoor_example_id" in expected_row:
                    row["backdoor_example_id"] = expected_row["backdoor_example_id"]
                if "split" in expected_row:
                    row["split"] = expected_row["split"]

                # Sampling-only metadata is semantically invalid for an
                # exhaustive full-population cache. This also repairs caches
                # touched by the previous migration implementation.
                for field in NORMAL_TASK_SAMPLED_ONLY_FIELDS:
                    row.pop(field, None)

                row.update(
                    {
                        "behavior_reference": "transformerlens_checkpoint",
                        "behavior_endpoint": "normal_task_accuracy_without_trigger",
                        "poisoning_causal_cache_schema_version": trigger_lift.POISONING_CAUSAL_CACHE_SCHEMA_VERSION,
                        "normal_task_population_size": int(len(rows_full)),
                        "normal_task_scan_max_rows": int(scan_max_rows),
                        "normal_task_sampling_strategy": sample_meta["sampling_strategy"],
                        "normal_task_population_mode": population_mode,
                    }
                )
                if sampled:
                    row.update(
                        {
                            "normal_task_candidate_order_seed": int(candidate_seed),
                            "normal_task_stratification": sample_meta["stratification"],
                            "normal_task_stratum": runtime._format_stratum_key(
                                module.normal_task_stratum_key(expected_row)
                            ),
                        }
                    )
                migrated.append(row)

            migrated = trigger_lift.assign_stable_holdout(
                migrated,
                seed=holdout_seed,
                test_fraction=holdout_fraction,
            )
            if not module.NORMAL_TASK_SPEC.validate_generated_cache(migrated):
                raise ValueError("reconstructed normal-task cache fails the canonical runtime validator")
            return migrated
    finally:
        if added_path:
            try:
                sys.path.remove(code_dir)
            except ValueError:
                pass


def migrate_normal_task_cache(
    project: Path,
    *,
    apply: bool,
    purge_unverifiable: bool = True,
    unresolved: list[str] | None = None,
) -> int:
    """Rename legacy directories and upgrade normal-task row metadata without LLM inference.

    Historical poisoning normal-task caches can already contain all expensive
    ``prompt_control``/``raw_output_control`` results while lacking the sampling
    metadata now required by ``validate_control_only_behavior_cache``. Rebuild the
    deterministic cohort identity from the persisted evaluation cohort, reorder
    legacy rows when necessary, add canonical metadata, and re-run the current
    runtime validator before replacing the pickle.
    """
    unresolved = unresolved if unresolved is not None else []
    count = 0
    cache_root = project / "cache"
    if not cache_root.exists():
        return 0

    for legacy_dir in sorted(p for p in cache_root.rglob("normal_task_behavior") if p.is_dir()):
        canonical_dir = legacy_dir.with_name("normal_task")
        for src in sorted(p for p in legacy_dir.rglob("*") if p.is_file()):
            if src.name == ".DS_Store":
                remove_path(src, apply=apply)
                count += 1
                continue
            rel = src.relative_to(legacy_dir)
            dst = canonical_dir / rel
            copy_then_remove(src, dst, apply=apply)
            count += 1
        if apply:
            shutil.rmtree(legacy_dir, ignore_errors=True)
            prune_empty_dirs(legacy_dir.parent, cache_root, apply=True)

    poisoning_root = cache_root / "poisoning"
    if not poisoning_root.exists():
        return count
    candidates = sorted(poisoning_root.rglob("llm_io_data.pkl"))
    for path in candidates:
        rel_parts = path.relative_to(poisoning_root).parts
        if "normal_task" not in rel_parts and "normal_task_behavior" not in rel_parts:
            continue
        try:
            with path.open("rb") as f:
                obj = pickle.load(f)
        except Exception as exc:
            log_action(f"[migrate] skip unreadable normal-task cache {path}: {exc}")
            continue
        if not _normal_task_cache_needs_metadata_migration(obj):
            continue
        rows = [dict(row) for row in obj]
        try:
            migrated = _migrate_normal_task_pickle(project, path, rows)
        except Exception as exc:
            msg = f"unverifiable legacy normal-task cache {path}: {exc}"
            if purge_unverifiable:
                log_action(f"[migrate] purge {msg}")
                if apply:
                    path.unlink()
                count += 1
            else:
                unresolved.append(msg)
            continue

        strategy = migrated[0].get("normal_task_sampling_strategy", "unknown") if migrated else "unknown"
        log_action(
            f"[migrate] normal-task cache metadata -> canonical: {path} "
            f"({len(migrated)} rows, sampling={strategy})"
        )
        if apply:
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("wb") as f:
                pickle.dump(migrated, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
        count += 1
    return count


def _resolve_path(raw: str | Path, base: Path) -> Path:
    path = Path(str(raw)).expanduser()
    candidates = [path] if path.is_absolute() else [base / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def _resolve_task(project: Path, task_module: str):
    sys.path.insert(0, str(project / "code"))
    try:
        from core.feature_extraction_runner import resolve_task_spec
        return resolve_task_spec(task_module)
    finally:
        try:
            sys.path.remove(str(project / "code"))
        except ValueError:
            pass


def _evaluation_frame_light(path: Path, *, prompt_col: str, target_col: str, split: str) -> pd.DataFrame:
    frame = _read_table(path)
    missing = [c for c in (prompt_col, target_col) if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns {missing}")
    if "_evaluated" in frame.columns:
        frame = frame.loc[frame["_evaluated"].fillna(False).astype(bool)].copy()
    split = str(split).strip().lower()
    if split in {"test", "train"}:
        if "is_test" not in frame.columns:
            raise ValueError(f"split={split} requires is_test in {path}")
        is_test = frame["is_test"].fillna(False).astype(bool)
        frame = frame.loc[is_test if split == "test" else ~is_test].copy()
    elif split != "all":
        raise ValueError(f"unsupported evaluation split {split!r}")
    if frame.empty:
        raise ValueError(f"no rows remain for split={split}: {path}")
    return frame.reset_index(drop=True)


def _evaluation_row_records_light(frame: pd.DataFrame, *, prompt_col: str, target_col: str) -> list[dict[str, Any]]:
    columns: list[str] = []
    for column in ("_orig_row", "original_idx", "is_test", prompt_col, target_col):
        if column in frame.columns and column not in columns:
            columns.append(column)
    records: list[dict[str, Any]] = []
    for row in frame[columns].to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if pd.isna(value):
                clean[str(key)] = None
            elif isinstance(value, (np.bool_, bool)):
                clean[str(key)] = bool(value)
            elif isinstance(value, (np.integer, int)):
                clean[str(key)] = int(value)
            elif isinstance(value, (np.floating, float)):
                clean[str(key)] = float(value)
            else:
                clean[str(key)] = str(value)
        records.append(clean)
    return records


def _train_scores_light(path: Path, *, target_col: str) -> pd.DataFrame:
    frame = _read_table(path).copy()
    frame = frame.drop(columns=["original_idx"], errors="ignore")
    frame["original_idx"] = np.arange(len(frame))
    if "is_test" in frame.columns:
        frame = frame.loc[~frame["is_test"].fillna(False).astype(bool)].copy()
    if target_col not in frame.columns:
        raise ValueError(f"{path} lacks target column {target_col!r}")
    return frame.reset_index(drop=True)


def _build_mean_prompt_pool_light(
    frame: pd.DataFrame, *, prompt_col: str, target_col: str, n_points: int, seed: int
) -> list[str]:
    if len(frame) == 0 or int(n_points) <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    df = frame.copy()
    pos = df.loc[df[target_col] == True] if target_col in df.columns else df.iloc[0:0]
    neg = df.loc[df[target_col] == False] if target_col in df.columns else df.iloc[0:0]
    pieces: list[pd.DataFrame] = []
    half = max(1, int(n_points) // 2)
    if not pos.empty:
        pieces.append(pos.sample(n=min(half, len(pos)), random_state=int(rng.integers(0, 2**31 - 1))))
    if not neg.empty:
        pieces.append(neg.sample(
            n=min(int(n_points) - sum(len(piece) for piece in pieces), len(neg)),
            random_state=int(rng.integers(0, 2**31 - 1)),
        ))
    if pieces:
        pool = pd.concat(pieces, ignore_index=False)
        if len(pool) < min(int(n_points), len(df)):
            remaining = df.drop(index=pool.index, errors="ignore")
            if not remaining.empty:
                extra = remaining.sample(
                    n=min(int(n_points) - len(pool), len(remaining)),
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )
                pool = pd.concat([pool, extra], ignore_index=False)
        return pool[prompt_col].astype(str).tolist()
    return df.sample(n=min(int(n_points), len(df)), random_state=int(seed))[prompt_col].astype(str).tolist()


def _upgrade_group_config(config_path: Path, *, context: dict[str, Any], apply: bool) -> None:
    """Persist reconstructed exact row/replacement identity into an old Stage-8 config."""
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(cfg.get("cache_identity"), dict):
        identity = dict(cfg["cache_identity"])
    else:
        identity = {k: v for k, v in cfg.items() if k != "paths"}
    identity.update({
        "task_module": context["task_module"],
        "ai_model": context["model"],
        "intervention": context["intervention"],
        "intervention_phase": "decode_only" if context["decode_only"] else "prefill_decode",
        "evaluation_split": context["evaluation_split"],
        "prompt_col": context["prompt_col"],
        "target_col": context["target_col"],
        "evaluation_rows": context["evaluation_rows"],
        "replacement_reference_prompts": context["replacement_reference_prompts"],
        "max_new_tokens": context["max_new_tokens"],
    })
    # Preserve all legacy analysis/reporting fields while making exact cache identity explicit.
    cfg.update(identity)
    cfg["cache_identity"] = identity
    atomic_json(config_path, cfg, apply=apply)


def group_context_from_config(config_path: Path, project: Path) -> tuple[dict[str, Any], bool]:
    """Return canonical group-cache context, reconstructing exact legacy identity when possible.

    Old v1 batch files did not embed context.  Their sibling configuration did,
    however, pin the source paths, model/method, split, replacement sample size,
    and seed.  Replaying only the deterministic *selection* code (no model load)
    reconstructs the exact row identities and replacement-reference prompts.
    """
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    identity = cfg.get("cache_identity") if isinstance(cfg.get("cache_identity"), dict) else cfg
    exact_required = {
        "ai_model", "task_module", "evaluation_split", "prompt_col", "target_col",
        "evaluation_rows", "intervention_phase", "intervention",
        "replacement_reference_prompts", "max_new_tokens",
    }
    if exact_required.issubset(identity):
        return ({
            "schema": GROUP_CONTEXT_SCHEMA,
            "model": identity["ai_model"],
            "task_module": identity["task_module"],
            "evaluation_split": identity["evaluation_split"],
            "prompt_col": identity["prompt_col"],
            "target_col": identity["target_col"],
            "evaluation_rows": identity["evaluation_rows"],
            "decode_only": identity["intervention_phase"] == "decode_only",
            "intervention": identity["intervention"],
            "replacement_reference_prompts": identity["replacement_reference_prompts"],
            "max_new_tokens": int(identity["max_new_tokens"]),
        }, False)

    scalar_required = {
        "ai_model", "task_module", "evaluation_split", "intervention_phase", "intervention",
        "points_to_use_for_mean_ablation", "seed",
    }
    missing_scalar = sorted(key for key in scalar_required if key not in identity)
    paths = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
    if missing_scalar or not paths:
        raise ValueError(f"{config_path} lacks reconstructible v1 identity; missing={missing_scalar}, paths={bool(paths)}")

    input_data_dir = _resolve_path(paths.get("input_data_dir", ""), config_path.parent)
    dataset_info_path = input_data_dir / "dataset_info.json"
    if not dataset_info_path.is_file():
        raise ValueError(f"missing dataset_info.json for v1 lineage reconstruction: {dataset_info_path}")
    info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    task = _resolve_task(project, str(identity["task_module"]))
    prompt_col = str(info.get("prompt_col") or task.DEFAULT_INPUT)
    target_col = str(info.get("target_col") or task.DEFAULT_TARGETS[0])

    singleton_raw = paths.get("singleton_scores_path")
    if not singleton_raw:
        raise ValueError(f"{config_path} lacks singleton_scores_path")
    singleton_scores = _resolve_path(singleton_raw, config_path.parent)
    eval_frame = _evaluation_frame_light(
        singleton_scores, prompt_col=prompt_col, target_col=target_col,
        split=str(identity["evaluation_split"]),
    )
    evaluation_rows = _evaluation_row_records_light(eval_frame, prompt_col=prompt_col, target_col=target_col)

    replacement_raw = paths.get("replacement_reference_scores_path")
    if replacement_raw:
        replacement_scores = _resolve_path(replacement_raw, config_path.parent)
    else:
        scores_raw = info.get("scores_path")
        if not scores_raw:
            raise ValueError(f"{dataset_info_path} lacks scores_path")
        replacement_scores = _resolve_path(scores_raw, input_data_dir)
    train_scores = _train_scores_light(replacement_scores, target_col=target_col)
    replacement_prompts = _build_mean_prompt_pool_light(
        train_scores,
        prompt_col=prompt_col,
        target_col=target_col,
        n_points=int(identity["points_to_use_for_mean_ablation"]),
        seed=int(identity["seed"]),
    )
    if str(identity["intervention"]).startswith("mean") and not replacement_prompts:
        raise ValueError(f"could not reconstruct replacement prompt pool for {config_path}")

    return ({
        "schema": GROUP_CONTEXT_SCHEMA,
        "model": identity["ai_model"],
        "task_module": identity["task_module"],
        "evaluation_split": identity["evaluation_split"],
        "prompt_col": prompt_col,
        "target_col": target_col,
        "evaluation_rows": evaluation_rows,
        "decode_only": identity["intervention_phase"] == "decode_only",
        "intervention": identity["intervention"],
        "replacement_reference_prompts": list(map(str, replacement_prompts)),
        "max_new_tokens": int(task.MAX_NEW_TOKENS),
    }, True)

def migrate_group_caches(project: Path, *, apply: bool, purge_unverifiable: bool, unresolved: list[str]) -> int:
    count = 0
    for cache_dir in sorted(p for p in project.rglob("group_eval_cache") if p.is_dir()):
        config_path = cache_dir.parent / "interaction_configuration.json"
        context: dict[str, Any] | None = None
        context_error: Exception | None = None
        reconstructed = False
        try:
            context, reconstructed = group_context_from_config(config_path, project)
            if reconstructed:
                _upgrade_group_config(config_path, context=context, apply=apply)
        except Exception as exc:
            context_error = exc
        for path in sorted(cache_dir.glob("batch_*.pkl")):
            try:
                with path.open("rb") as f:
                    payload = pickle.load(f)
            except Exception:
                continue
            if not isinstance(payload, dict) or payload.get("schema") != LEGACY_GROUP_SCHEMA:
                continue
            if context is None:
                msg = f"unverifiable v1 group cache {path}: {context_error}"
                if purge_unverifiable:
                    log_action(f"[migrate] purge {msg}")
                    if apply:
                        path.unlink()
                    count += 1
                    continue
                unresolved.append(msg)
                continue
            start = int(payload.get("start", -1))
            end = int(payload.get("end", -1))
            outputs = payload.get("outputs")
            if start < 0 or end <= start or not isinstance(outputs, dict):
                unresolved.append(f"malformed v1 group cache: {path}")
                continue
            n = end - start
            clean_outputs: dict[str, np.ndarray] = {}
            for key, raw in outputs.items():
                arr = np.asarray(raw, dtype=bool)
                if arr.ndim != 1 or len(arr) != n:
                    raise RuntimeError(f"Invalid group array {key!r} in {path}")
                clean_outputs[str(key)] = arr
            new_payload = {
                "schema": CANONICAL_GROUP_SCHEMA,
                "context": context,
                "start": start,
                "end": end,
                "outputs": clean_outputs,
            }
            log_action(f"[migrate] group cache v1 -> v2: {path}")
            if apply:
                tmp = path.with_suffix(path.suffix + ".tmp")
                with tmp.open("wb") as f:
                    pickle.dump(new_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                tmp.replace(path)
            count += 1
    return count


def canonicalize_threshold_method(manifest_path: Path, payload: dict[str, Any]) -> bool:
    method = dict(payload.get("scientific_method") or {})
    if not method:
        return False
    original = dict(method)
    for key in THRESHOLD_RUNTIME_KEYS:
        method.pop(key, None)

    cand = method.get("candidate_flip_stats_path") or payload.get("candidate_flip_stats_path")
    mat = method.get("materialized_stage7_scores_path")
    if mat in (None, "") and cand:
        inferred = Path(str(cand)).expanduser().parent / "scores.csv"
        if inferred.is_file():
            mat = str(inferred)

    if "candidate_flip_stats_sha256" not in method:
        if cand and Path(str(cand)).expanduser().is_file():
            method["candidate_flip_stats_sha256"] = sha256(Path(str(cand)).expanduser())
        else:
            raise RuntimeError(f"Cannot canonicalize candidate source identity in {manifest_path}: {cand!r}")
    if "materialized_stage7_scores_sha256" not in method:
        if mat and Path(str(mat)).expanduser().is_file():
            method["materialized_stage7_scores_sha256"] = sha256(Path(str(mat)).expanduser())
        else:
            raise RuntimeError(f"Cannot canonicalize materialized Stage-7 identity in {manifest_path}: {mat!r}")

    method.pop("candidate_flip_stats_path", None)
    method.pop("materialized_stage7_scores_path", None)
    payload["scientific_method"] = method
    return method != original


def extract_threshold_csv_schemas(code_root: Path) -> dict[str, list[str]]:
    source = (code_root / "studies/overtopping/analysis/threshold_event_diagnostics.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {
        "THRESHOLD_UNIT_TEST_COLUMNS": "threshold_unit_tests.csv",
        "THRESHOLD_POPULATION_SUMMARY_COLUMNS": "threshold_population_summary.csv",
        "THRESHOLD_BINNED_CURVE_COLUMNS": "threshold_binned_flip_curves.csv",
    }
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    out[wanted[target.id]] = list(ast.literal_eval(value))
    missing = sorted(set(wanted.values()) - set(out))
    if missing:
        raise RuntimeError(f"Could not extract current threshold CSV schemas: {missing}")
    return out


def migrate_threshold_manifests_and_empty_csvs(project: Path, *, apply: bool) -> int:
    count = 0
    schemas = extract_threshold_csv_schemas(project / "code")
    for manifest in sorted(project.rglob("threshold_spiking_experiment.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("status") == "ok" and payload.get("scientific_method"):
            changed = canonicalize_threshold_method(manifest, payload)
            if changed:
                atomic_json(manifest, payload, apply=apply)
                count += 1
        if payload.get("status") != "ok":
            continue
        for filename, columns in schemas.items():
            path = manifest.parent / filename
            needs = False
            if not path.exists() or path.stat().st_size == 0:
                needs = True
            else:
                try:
                    pd.read_csv(path, nrows=1)
                except pd.errors.EmptyDataError:
                    needs = True
                except Exception:
                    pass
            if needs:
                log_action(f"[migrate] materialize schema-valid empty threshold CSV: {path}")
                if apply:
                    pd.DataFrame(columns=columns).to_csv(path, index=False)
                count += 1
    return count


def cache_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def candidate_old_batch_sizes(start: int, arr_len: int, declared_batch_size: int | None = None) -> list[int]:
    vals = {arr_len, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}
    if declared_batch_size is not None and int(declared_batch_size) > 0:
        vals.add(int(declared_batch_size))
    if start > 0:
        for d in range(1, int(math.isqrt(start)) + 1):
            if start % d == 0:
                vals.add(d)
                vals.add(start // d)
    return sorted(v for v in vals if v >= arr_len and v > 0 and (start == 0 or start % v == 0))

def load_task_max_new_tokens(project: Path, task_module: str) -> int:
    sys.path.insert(0, str(project / "code"))
    try:
        from core.feature_extraction_runner import resolve_task_spec
        task = resolve_task_spec(task_module)
        return int(task.MAX_NEW_TOKENS)
    finally:
        try:
            sys.path.remove(str(project / "code"))
        except ValueError:
            pass


def high_n_context(manifest: Path, payload: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], int | None]:
    """Reconstruct the exact runtime cache-key context for a completed high-N run.

    ``high_n_scores_with_flips.csv::_orig_row`` is the ORIGINAL source scores
    row identity after candidate/control results are merged.  The runtime cache
    key is instead built from ``eval_indices``: positions inside the
    split/baseline-filtered ``scores_df``.  Reconstruct that filtered frame and
    map the materialized source identities back to those positional indices.
    """
    method = payload.get("scientific_method") or {}
    materialized_scores_path = manifest.parent / "high_n_scores_with_flips.csv"
    if not materialized_scores_path.is_file():
        raise RuntimeError("missing high_n_scores_with_flips.csv")
    materialized = pd.read_csv(materialized_scores_path, usecols=lambda c: c == "_orig_row")
    if "_orig_row" not in materialized.columns:
        raise RuntimeError("high_n_scores_with_flips.csv lacks _orig_row")
    source_orig = pd.to_numeric(materialized["_orig_row"], errors="raise").astype(int)
    if source_orig.duplicated().any():
        raise RuntimeError("high_n_scores_with_flips.csv has duplicate _orig_row identities")

    input_data_dir = Path(str(method.get("input_data_dir", ""))).expanduser()
    dataset_info_path = input_data_dir / "dataset_info.json"
    if not dataset_info_path.is_file():
        raise RuntimeError(f"missing dataset_info.json: {dataset_info_path}")
    info = json.loads(dataset_info_path.read_text(encoding="utf-8"))

    task_module = str(method.get("task_module", ""))
    if not task_module:
        raise RuntimeError("scientific_method lacks task_module")
    task = _resolve_task(PROJECT_FOR_IMPORT, task_module)
    prompt_col = info.get("prompt_col") or task.DEFAULT_INPUT
    target_col = info.get("target_col") or task.DEFAULT_TARGETS[0]
    if not prompt_col or not target_col:
        raise RuntimeError(f"dataset_info/task spec lacks prompt_col/target_col: {dataset_info_path}")

    raw_scores_path = info.get("scores_path")
    if not raw_scores_path:
        raise RuntimeError(f"dataset_info lacks scores_path: {dataset_info_path}")
    raw_scores_path = Path(str(raw_scores_path)).expanduser()
    if not raw_scores_path.is_absolute():
        candidates = [input_data_dir / raw_scores_path, PROJECT_FOR_IMPORT / raw_scores_path]
        raw_scores_path = next((c for c in candidates if c.is_file()), raw_scores_path)
    if not raw_scores_path.is_file():
        raise RuntimeError(f"scores_path from dataset_info does not exist: {raw_scores_path}")

    baseline = str(method.get("baseline_subset", payload.get("baseline", "")))
    evaluation_split = str(method.get("evaluation_split", payload.get("evaluation_split", "test")))

    sys.path.insert(0, str(PROJECT_FOR_IMPORT / "code"))
    try:
        from core.high_n_singleton_eval import load_scores_for_baseline
        filtered_scores = load_scores_for_baseline(
            scores_path=raw_scores_path,
            target_col=str(target_col),
            baseline_subset=baseline,
            task_targets=task.DEFAULT_TARGETS,
            split=evaluation_split,
        )
    finally:
        try:
            sys.path.remove(str(PROJECT_FOR_IMPORT / "code"))
        except ValueError:
            pass

    if "original_idx" not in filtered_scores.columns:
        raise RuntimeError("filtered high-N score frame lacks original_idx")
    filtered_orig = pd.to_numeric(filtered_scores["original_idx"], errors="raise").astype(int)
    if filtered_orig.duplicated().any():
        raise RuntimeError("filtered high-N score frame has duplicate original_idx identities")
    pos_by_orig = {int(orig): int(pos) for pos, orig in enumerate(filtered_orig.tolist())}
    missing = [int(orig) for orig in source_orig.tolist() if int(orig) not in pos_by_orig]
    if missing:
        preview = ", ".join(map(str, missing[:12]))
        raise RuntimeError(
            f"{len(missing)} materialized high-N source row(s) are absent from the "
            f"current {evaluation_split}/{baseline} filtered frame: {preview}"
        )
    eval_indices = np.asarray([pos_by_orig[int(orig)] for orig in source_orig.tolist()], dtype=int)

    context = {
        "baseline_subset": baseline,
        "target_col": str(target_col),
        "prompt_col": str(prompt_col),
        "decode_only": bool(method.get("decode_only", False)),
        "intervention": str(method.get("intervention", "mean-donor")),
        "max_new_tokens": int(task.MAX_NEW_TOKENS),
    }
    declared_batch = method.get("batch_size")
    try:
        declared_batch = int(declared_batch) if declared_batch is not None else None
    except Exception:
        declared_batch = None
    return eval_indices, context, declared_batch


# Set once by main; kept global only to avoid threading project through dynamic task import helper.
PROJECT_FOR_IMPORT: Path


def migrate_high_n_cache_for_manifest(
    project: Path,
    manifest: Path,
    payload: dict[str, Any],
    *,
    apply: bool,
    purge_unverifiable: bool,
    unresolved: list[str],
) -> int:
    cache_dir = manifest.parent / "high_n_eval_cache" / "ablation_cache"
    if not cache_dir.is_dir():
        return 0
    paths = sorted(cache_dir.glob("*.pkl"))
    if not paths:
        return 0
    try:
        eval_indices, ctx, declared_batch_size = high_n_context(manifest, payload)
    except Exception as exc:
        if purge_unverifiable:
            for path in paths:
                log_action(f"[migrate] purge unverifiable high-N cache: {path} ({exc})")
                if apply:
                    path.unlink()
            return len(paths)
        unresolved.append(f"cannot verify high-N cache under {cache_dir}: {exc}")
        return 0

    count = 0
    rx = re.compile(r"^(?P<prefix>.+)_(?P<start>\d+)_(?P<key>[0-9a-f]{16})$")
    for path in paths:
        m = rx.match(path.stem)
        if not m:
            continue
        start = int(m.group("start"))
        file_key = m.group("key")
        try:
            with path.open("rb") as f:
                arr = np.asarray(pickle.load(f), dtype=bool)
        except Exception as exc:
            msg = f"unreadable high-N cache {path}: {exc}"
            if purge_unverifiable:
                log_action(f"[migrate] purge {msg}")
                if apply:
                    path.unlink()
                count += 1
            else:
                unresolved.append(msg)
            continue
        if arr.ndim != 1 or start + len(arr) > len(eval_indices):
            msg = f"invalid/stale high-N cache range {path}"
            if purge_unverifiable:
                log_action(f"[migrate] purge {msg}")
                if apply:
                    path.unlink()
                count += 1
            else:
                unresolved.append(msg)
            continue
        end = start + len(arr)
        base_payload = {
            "eval_indices": [int(i) for i in eval_indices[start:end].tolist()],
            **ctx,
        }
        canonical_key = cache_hash(base_payload)
        if file_key == canonical_key:
            continue
        matched_old = any(
            file_key == cache_hash(base_payload | {"batch_size": int(old_batch)})
            for old_batch in candidate_old_batch_sizes(start, len(arr), declared_batch_size)
        )
        if not matched_old:
            msg = f"unverifiable high-N cache key {path}"
            if purge_unverifiable:
                log_action(f"[migrate] purge {msg}")
                if apply:
                    path.unlink()
                count += 1
            else:
                unresolved.append(msg)
            continue
        new_path = path.with_name(f"{m.group('prefix')}_{start}_{canonical_key}.pkl")
        try:
            display_path = path.resolve().relative_to(project.resolve())
            display_new_path = new_path.resolve().relative_to(project.resolve())
        except ValueError:
            display_path = path
            display_new_path = new_path
        log_action(
            "[migrate] high-N cache key -> canonical: "
            f"{display_path} -> {display_new_path}"
        )
        if apply:
            if new_path.exists():
                with new_path.open("rb") as f:
                    existing = np.asarray(pickle.load(f), dtype=bool)
                if not np.array_equal(existing, arr):
                    raise RuntimeError(f"Conflicting canonical high-N cache: {new_path}")
            else:
                with new_path.open("wb") as f:
                    pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
            path.unlink()
        count += 1
    return count


def migrate_high_n_caches(project: Path, *, apply: bool, purge_unverifiable: bool, unresolved: list[str]) -> int:
    count = 0
    for manifest in sorted(project.rglob("threshold_spiking_experiment.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("status") != "ok" or not payload.get("scientific_method"):
            continue
        count += migrate_high_n_cache_for_manifest(
            project, manifest, payload, apply=apply,
            purge_unverifiable=purge_unverifiable, unresolved=unresolved,
        )
    return count


def read_scope(stats_dir: Path) -> dict[str, Any]:
    path = stats_dir / "evaluation_scope.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.fillna(0).astype(float) != 0.0
    return series.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def infer_split(stats_dir: Path) -> str:
    name = stats_dir.name
    if re.search(r"-heldout_test(?:-cap\d+)?$", name):
        return "test"
    if re.search(r"-eval_train(?:-cap\d+)?$", name):
        return "train"
    if re.search(r"-eval_all(?:-cap\d+)?$", name):
        return "all"
    scope = read_scope(stats_dir)
    for key in ("final_statistics_split", "evaluation_split"):
        val = str(scope.get(key, "")).lower()
        if val in {"test", "train", "all"}:
            return val
    scores = stats_dir / "scores.csv"
    glob = stats_dir / "flip_stats_global.json"
    if not scores.is_file() or not glob.is_file():
        return "all"
    try:
        n_global = int(json.loads(glob.read_text(encoding="utf-8")).get("n_evaluated_rows"))
        flags = pd.read_csv(scores, usecols=["is_test"])["is_test"]
        mask = bool_series(flags)
        counts = {"all": len(mask), "test": int(mask.sum()), "train": int((~mask).sum())}
        matches = [k for k, v in counts.items() if v == n_global]
        return matches[0] if len(matches) == 1 else "all"
    except Exception:
        return "all"


def baseline_subset(stats_dir: Path) -> str:
    value = str(read_scope(stats_dir).get("evaluation_baseline_subset", "all")).lower()
    return value if value in {"all", "positive", "negative"} else "all"


def poisoning_stats_setting(project: Path, data_root: Path, stats_dir: Path) -> dict[str, Any]:
    rel = stats_dir.relative_to(data_root)
    if len(rel.parts) < 3 or rel.parts[0] != "poisoning":
        raise ValueError("not a poisoning stats directory")
    task = str(rel.parts[1])
    if task not in {"arithmetic", "grammar"}:
        raise ValueError(f"unsupported poisoning task {task!r}")

    # .../<cohort>/eval_<intervention>/rule_extraction_results/neuron_flip_rules/stats/<name>
    eval_dir = stats_dir.parents[3]
    cohort = eval_dir.parent.name
    if cohort == "attack_cohort_control_correctness":
        task_module = f"studies.poisoning.tasks.{task}:ATTACK_COHORT_CONTROL_CORRECTNESS_SPEC"
    elif cohort == "backdoor_trigger_test":
        task_module = f"studies.poisoning.tasks.{task}:BACKDOOR_TASK_SPEC"
    else:
        raise ValueError(f"unsupported poisoning Stage-7 cohort {cohort!r}")

    features_scores_dir = eval_dir / "feature_report"
    info_candidates = [features_scores_dir / "dataset_info.json", eval_dir / "dataset_info.json"]
    info_path = next((p for p in info_candidates if p.is_file()), None)
    if info_path is None:
        # Stage 7 itself accepts dataset_info in parents of scores.csv; keep the
        # migration equally permissive while staying within this checkpoint tree.
        info_path = next(iter(eval_dir.rglob("dataset_info.json")), None)
    if info_path is None:
        raise ValueError(f"cannot locate dataset_info.json under {eval_dir}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    model_id = info.get("ai_model") or info.get("model_name") or info.get("model")
    if not model_id:
        raise ValueError(f"dataset_info has no model identity: {info_path}")

    scope = read_scope(stats_dir)
    intervention = str(scope.get("intervention") or eval_dir.name.removeprefix("eval_")).strip()
    phase = str(scope.get("intervention_phase", "")).strip().lower()
    decode_only = phase == "decode_only" or "generation_only" in rel.parts or "output_only" in rel.parts

    search_epsilon = None
    global_path = stats_dir / "flip_stats_global.json"
    if global_path.is_file():
        try:
            raw = json.loads(global_path.read_text(encoding="utf-8")).get("reference_cha_tau")
            if raw is not None and np.isfinite(float(raw)):
                search_epsilon = float(raw)
        except Exception:
            pass
    if search_epsilon is None:
        m = re.search(r"(?:^|-)tau(?P<tau>\d+(?:\.\d+)?)", stats_dir.name)
        if m:
            search_epsilon = float(m.group("tau"))
    if search_epsilon is None:
        # A frozen candidate ranking makes search_epsilon irrelevant to which
        # units are recomputed, but it still enters descriptive flip-stat metadata.
        # Do not invent it when it cannot be recovered.
        raise ValueError(f"cannot recover search_epsilon for {stats_dir}")

    return {
        "task_module": task_module,
        "model_id": str(model_id),
        "rules_dir": stats_dir.parents[1],
        "features_scores_dir": features_scores_dir,
        "circuit_agonists_path": eval_dir,
        "search_epsilon": float(search_epsilon),
        "intervention": intervention,
        "decode_only": bool(decode_only),
    }


def stats_setting(project: Path, data_root: Path, stats_dir: Path, index: int) -> dict[str, Any]:
    rel = stats_dir.relative_to(data_root)
    if rel.parts and rel.parts[0] == "poisoning":
        return poisoning_stats_setting(project, data_root, stats_dir)
    sys.path.insert(0, str(project / "code"))
    try:
        from studies.overtopping.analysis import primary_holdout_analysis as helpers
        ref = helpers.reference_stats_dir(stats_dir)
        phase = "Out" if "decode_only" in ref.name.lower() else "I+O"
        model_root = ref.parents[3]
        rel = model_root.relative_to(data_root)
        task = rel.parts[0]
        model = Path(*rel.parts[1:]).as_posix()
        row = pd.Series({"task": task, "model": model, "phase": phase, "stats_dir": str(stats_dir)})
        setting = helpers.setting_from_row(index, row, data_root, project / "results", evaluation_split=infer_split(stats_dir))
        setting = dict(setting)
        setting.setdefault("rules_dir", stats_dir.parents[1])
        setting.setdefault("features_scores_dir", setting["model_root"] / "feature_report")
        setting.setdefault("circuit_agonists_path", setting["circuit_agonists_path"])
        return setting
    finally:
        try:
            sys.path.remove(str(project / "code"))
        except ValueError:
            pass


def migrate_directional_singletons(project: Path, *, apply: bool, python_bin: str, unresolved: list[str]) -> int:
    data_root = project / "data"
    count = 0
    legacy_files: list[Path] = []
    for path in data_root.rglob("singleton_set_metrics.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("definition_version") == LEGACY_SINGLETON_SCHEMA:
            legacy_files.append(path)

    for index, path in enumerate(sorted(legacy_files)):
        stats_dir = path.parent
        scores = stats_dir / "scores.csv"
        if not scores.is_file():
            unresolved.append(f"cannot migrate directional singleton v2 without scores.csv: {stats_dir}")
            continue
        try:
            setting = stats_setting(project, data_root, stats_dir, index)
        except Exception as exc:
            unresolved.append(f"cannot resolve Stage-7 setting for {stats_dir}: {exc}")
            continue
        split = infer_split(stats_dir)
        cmd = [
            python_bin, "-m", "pipeline.stage07_refine_neuron_anchored_rules",
            "--task_module", str(setting["task_module"]),
            "--ai_model", str(setting["model_id"]),
            "--rules_dir", str(setting.get("rules_dir", stats_dir.parents[1])),
            "--features_scores_dir", str(setting.get("features_scores_dir", setting.get("model_root", stats_dir.parents[3]) / "feature_report")),
            "--circuit_agonists_path", str(setting["circuit_agonists_path"]),
            "--search_epsilon", str(setting["search_epsilon"]),
            "--stats_dirname", stats_dir.name,
            "--intervention", str(setting["intervention"]),
            "--stats_only",
            "--evaluation_split", split,
            "--evaluation_baseline_subset", baseline_subset(stats_dir),
            "--sampling_max_points", "0",
            "--skip_agonist_metric_stats",
        ]
        ranking = stats_dir / "frozen_candidate_ranking.csv"
        if ranking.is_file():
            cmd.extend(["--candidate_ranking_csv", str(ranking)])
        if setting["decode_only"]:
            cmd.append("--decode_only")
        log_action(f"[migrate] singleton v2 -> v3: {stats_dir}")
        if apply:
            scope_before = read_scope(stats_dir)
            with tempfile.TemporaryDirectory(prefix="singleton_v3_migration_") as tmpdir:
                backup = Path(tmpdir) / "scores.csv"
                shutil.copy2(scores, backup)
                subprocess.run(cmd, cwd=project / "code", check=True)
                shutil.copy2(backup, scores)
            new_payload = json.loads(path.read_text(encoding="utf-8"))
            if new_payload.get("definition_version") != CANONICAL_SINGLETON_SCHEMA:
                raise RuntimeError(f"Stage-7 stats-only migration did not produce v3: {path}")
            if scope_before:
                scope = dict(scope_before)
                scope["directional_metrics_regenerated"] = True
                scope["directional_metrics_regeneration_mode"] = "stats_only_from_materialized_singleton_events"
                scope["directional_metrics_schema"] = CANONICAL_SINGLETON_SCHEMA
                atomic_json(stats_dir / "evaluation_scope.json", scope, apply=True)
        count += 1
    return count


def find_run_config(run_dir: Path) -> Path | None:
    preferred = run_dir / "01_training_checkpoints" / "metadata" / "run_config.json"
    if preferred.is_file():
        return preferred
    candidates = list(run_dir.rglob("run_config.json"))
    return candidates[0] if len(candidates) == 1 else None


def migrate_poison_stage07_resume(project: Path, *, apply: bool, unresolved: list[str]) -> int:
    root = project / "data" / "poisoning"
    if not root.exists():
        return 0
    count = 0
    for summary_path in sorted(root.rglob("07_poisoning_example_detection/*/detection_summary.json")):
        phase_dir = summary_path.parent
        resume = phase_dir / "stage07_resume_configuration.json"
        if resume.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_dir = phase_dir.parents[1]
        run_config_path = find_run_config(run_dir)
        interval_metrics = sorted(phase_dir.rglob("interval_metrics.json"))
        if run_config_path is None or not interval_metrics:
            unresolved.append(f"cannot reconstruct Stage-07 resume identity: {phase_dir}")
            continue
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        metric = json.loads(interval_metrics[0].read_text(encoding="utf-8"))
        n_null = int(metric.get("n_clean_null_trajectories", 1) or 1)
        if n_null != 1:
            unresolved.append(
                f"Stage-07 output uses {n_null-1} additional clean-null run(s), but old output does not persist their paths: {phase_dir}"
            )
            continue
        required = [
            "control_correctness_agonist_tau", "detector_max_channels", "detector_min_abs_delta_u",
            "detector_bootstrap_draws", "detector_bootstrap_confidence_level",
            "detector_max_exposures_per_interval", "detector_sample_seed", "detector_matched_control_draws",
            "detector_wanda_batch_size", "detector_u_j_batch_size", "detector_u_j_neuron_batch_size",
        ]
        missing = [k for k in required if k not in metric]
        if missing:
            unresolved.append(f"Stage-07 interval metrics lack resume fields {missing}: {interval_metrics[0]}")
            continue
        payload = {
            "schema_version": STAGE07_RESUME_SCHEMA,
            "result_identity": {
                "run_dir": str(run_dir.resolve()),
                "task": str(summary.get("task")),
                "phase": str(summary.get("phase")),
                "scoring_schema_version": int(metric.get("scoring_schema_version", 0)),
                "eval_intervention": str(summary.get("eval_intervention", metric.get("eval_intervention"))),
                "required_tau": float(metric["control_correctness_agonist_tau"]),
                "max_channels": int(metric["detector_max_channels"]),
                "min_abs_delta_u": float(metric["detector_min_abs_delta_u"]),
                "bootstrap_draws": int(metric["detector_bootstrap_draws"]),
                "bootstrap_confidence_level": float(metric["detector_bootstrap_confidence_level"]),
                "clean_null_run_dirs": [],
                "min_clean_null_z": metric.get("detector_min_clean_null_z"),
                "max_exposures_per_interval": int(metric["detector_max_exposures_per_interval"]),
                "sample_seed": int(metric["detector_sample_seed"]),
                "matched_control_draws": int(metric["detector_matched_control_draws"]),
                "run_seed": int(run_config.get("seed", 0)),
                "model_name": str(run_config.get("model_name", "")),
                "model_revision": run_config.get("model_revision"),
            },
            "runtime": {
                "u_j_batch_size": int(metric["detector_u_j_batch_size"]),
                "u_j_neuron_batch_size": int(metric["detector_u_j_neuron_batch_size"]),
                "wanda_batch_size": int(metric["detector_wanda_batch_size"]),
            },
        }
        atomic_json(resume, payload, apply=apply)
        count += 1
    return count


def noncanonical_high_n_leftovers(project: Path) -> list[str]:
    leftovers: list[str] = []
    rx = re.compile(r"^(?P<prefix>.+)_(?P<start>\d+)_(?P<key>[0-9a-f]{16})$")
    for manifest in sorted(project.rglob("threshold_spiking_experiment.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("status") != "ok" or not payload.get("scientific_method"):
            continue
        cache_dir = manifest.parent / "high_n_eval_cache" / "ablation_cache"
        if not cache_dir.is_dir():
            continue
        try:
            eval_indices, ctx, _ = high_n_context(manifest, payload)
        except Exception:
            # A cache directory whose completed manifest can no longer establish
            # its identity is not canonical persisted state.
            leftovers.extend(str(p) for p in cache_dir.glob("*.pkl"))
            continue
        for path in cache_dir.glob("*.pkl"):
            m = rx.match(path.stem)
            if not m:
                leftovers.append(str(path)); continue
            try:
                start = int(m.group("start"))
                with path.open("rb") as f:
                    arr = np.asarray(pickle.load(f), dtype=bool)
            except Exception:
                leftovers.append(str(path)); continue
            if arr.ndim != 1 or start < 0 or start + len(arr) > len(eval_indices):
                leftovers.append(str(path)); continue
            end = start + len(arr)
            expected = cache_hash({
                "eval_indices": [int(i) for i in eval_indices[start:end].tolist()],
                **ctx,
            })
            if m.group("key") != expected:
                leftovers.append(str(path))
    return leftovers


def recognized_legacy_leftovers(project: Path) -> list[str]:
    leftovers: list[str] = []
    if (project / "cache" / "threshold_events").exists():
        leftovers.append(str(project / "cache" / "threshold_events"))
    for path in project.rglob("normal_task_behavior"):
        if path.is_dir():
            leftovers.append(str(path))
    poisoning_root = project / "cache" / "poisoning"
    if poisoning_root.exists():
        for path in poisoning_root.rglob("llm_io_data.pkl"):
            rel_parts = path.relative_to(poisoning_root).parts
            if "normal_task" not in rel_parts:
                continue
            try:
                with path.open("rb") as f:
                    payload = pickle.load(f)
                if _normal_task_cache_needs_metadata_migration(payload):
                    leftovers.append(str(path))
            except Exception:
                pass
    for path in project.rglob("singleton_set_metrics.json"):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("definition_version") == LEGACY_SINGLETON_SCHEMA:
                leftovers.append(str(path))
        except Exception:
            pass
    for path in project.rglob("group_eval_cache/batch_*.pkl"):
        try:
            with path.open("rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and payload.get("schema") == LEGACY_GROUP_SCHEMA:
                leftovers.append(str(path))
        except Exception:
            pass
    for path in project.rglob("threshold_spiking_experiment.json"):
        try:
            method = json.loads(path.read_text(encoding="utf-8")).get("scientific_method") or {}
        except Exception:
            continue
        if THRESHOLD_RUNTIME_KEYS.intersection(method) or "candidate_flip_stats_path" in method or "materialized_stage7_scores_path" in method:
            leftovers.append(str(path))
    for path in project.rglob("07_poisoning_example_detection/*/detection_summary.json"):
        if not (path.parent / "stage07_resume_configuration.json").is_file():
            leftovers.append(str(path.parent))
    leftovers.extend(noncanonical_high_n_leftovers(project))
    return sorted(set(leftovers))


def main() -> None:
    global PROJECT_FOR_IMPORT, VERBOSE
    args = parse_args()
    VERBOSE = bool(args.verbose)
    project = Path(args.project_root).expanduser().resolve()
    PROJECT_FOR_IMPORT = project
    if not (project / "code").is_dir() or not (project / "data").exists():
        raise SystemExit(f"Not an overtopping project root: {project}")

    # Canonical runtime has no legacy-cache reader.  Therefore a cache whose
    # scientific identity cannot be reconstructed is stale disposable state,
    # not a scientific-data migration failure.  Keep-mode exists only for inspection.
    purge_unverifiable = bool(args.purge_unverifiable_caches or not args.keep_unverifiable_caches)

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[migrate] {mode}: {project}")
    if purge_unverifiable:
        print("[migrate] unverifiable legacy caches: purge (scientific data is never purged)")
    else:
        print("[migrate] unverifiable legacy caches: KEEP (debug mode; canonical runtime will ignore them)")
    unresolved: list[str] = []
    counts: dict[str, int] = {}

    counts["rebuild_only_cache_removed"] = migrate_rebuild_only_cache(project, apply=args.apply)
    counts["normal_task_cache"] = migrate_normal_task_cache(
        project, apply=args.apply, purge_unverifiable=purge_unverifiable, unresolved=unresolved
    )
    counts["group_cache_v1"] = migrate_group_caches(
        project, apply=args.apply, purge_unverifiable=purge_unverifiable, unresolved=unresolved
    )
    # High-N keys must be migrated BEFORE threshold manifests are canonicalized,
    # because old manifests still contain the historical runtime batch_size that
    # participated in the legacy key hash.
    counts["high_n_cache"] = migrate_high_n_caches(
        project, apply=args.apply, purge_unverifiable=purge_unverifiable, unresolved=unresolved
    )
    counts["threshold_manifest_or_csv"] = migrate_threshold_manifests_and_empty_csvs(project, apply=args.apply)
    counts["directional_singleton_v2"] = migrate_directional_singletons(
        project, apply=args.apply, python_bin=args.python_bin, unresolved=unresolved
    )
    counts["poison_stage07_resume"] = migrate_poison_stage07_resume(project, apply=args.apply, unresolved=unresolved)

    print("[migrate] counts:")
    for key, value in counts.items():
        print(f"  {key}: {value}")

    if unresolved:
        print(f"[migrate] UNRESOLVED SCIENTIFIC DATA: {len(unresolved)} item(s)", file=sys.stderr)
        for item in unresolved[:25]:
            print(f"  - {item}", file=sys.stderr)
        if len(unresolved) > 25:
            print(f"  ... {len(unresolved)-25} more (rerun with --verbose only after fixing the first class)", file=sys.stderr)
        if args.apply:
            raise SystemExit(2)

    if args.apply:
        leftovers = recognized_legacy_leftovers(project)
        if leftovers:
            print(f"[migrate] recognized legacy artifacts remain: {len(leftovers)}", file=sys.stderr)
            for item in leftovers[:25]:
                print(f"  - {item}", file=sys.stderr)
            if len(leftovers) > 25:
                print(f"  ... {len(leftovers)-25} more", file=sys.stderr)
            raise SystemExit(3)
        print("[migrate] canonicalization complete; no recognized legacy persisted artifacts remain.")
    else:
        print("[migrate] dry run only; rerun with --apply to modify files.")


if __name__ == "__main__":
    main()
