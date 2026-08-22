"""Task-agnostic helpers for deterministic post-training causal pools."""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

CAUSAL_POOL_SCHEMA_VERSION = 2


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def metadata_current(
    meta_path: Path,
    causal_path: Path,
    *,
    require_full_rebuild_marker: bool = False,
) -> bool:
    if not meta_path.exists() or not causal_path.exists():
        return False
    try:
        meta = read_json(meta_path)
    except Exception:
        return False
    if int(meta.get("causal_pool_schema_version", -1)) != CAUSAL_POOL_SCHEMA_VERSION:
        return False
    if require_full_rebuild_marker and not bool(meta.get("causal_pool_rebuilt_full_for_posttraining", False)):
        return False
    expected = int(meta.get("causal_n_examples", -1))
    if expected <= 0:
        return False
    try:
        actual = count_jsonl(causal_path)
    except Exception:
        return False
    return actual == expected


def namespace_from_config(cfg: dict[str, Any]) -> Namespace:
    cfg = dict(cfg)
    cfg.setdefault("max_causal_eval", 0)
    return Namespace(**cfg)


def mark_rebuilt(meta_path: Path, cfg: dict[str, Any]) -> None:
    meta = read_json(meta_path)
    meta["causal_pool_schema_version"] = CAUSAL_POOL_SCHEMA_VERSION
    meta["causal_pool_rebuilt_full_for_posttraining"] = True
    meta["run_config_max_causal_eval_at_rebuild"] = int(cfg.get("max_causal_eval", 0) or 0)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def prepare_task_causal_pool(
    run_dir: Path,
    cfg: dict[str, Any],
    *,
    task_name: str,
    causal_filename: str,
    metadata_filename: str,
    rebuild,
    force: bool = False,
) -> Path:
    """Verify or rebuild a task-owned deterministic post-training causal pool.

    ``rebuild`` receives ``(run_dir, args)`` and must write both the causal JSONL
    and its metadata file. Task packages retain ownership of dataset semantics;
    this helper owns only pool validation, full-pool enforcement, and logging.
    """
    heldout = Path(run_dir) / "heldout"
    causal = heldout / causal_filename
    meta = heldout / metadata_filename
    configured_pool_cap = int(cfg.get("max_causal_eval", 0) or 0)
    if not force and metadata_current(
        meta, causal, require_full_rebuild_marker=(configured_pool_cap > 0)
    ):
        print(f"[causal-pool] {task_name} verified: {causal} ({count_jsonl(causal)} rows)", flush=True)
        return causal

    args = namespace_from_config(cfg)
    args.max_causal_eval = 0
    print(
        f"[causal-pool] rebuilding deterministic {task_name} causal candidate pool (no training)",
        flush=True,
    )
    detail = rebuild(Path(run_dir), args)
    mark_rebuilt(meta, cfg)
    suffix = f"; {detail}" if detail else ""
    print(
        f"[causal-pool] {task_name} ready: {causal} ({count_jsonl(causal)} rows{suffix})",
        flush=True,
    )
    return causal
