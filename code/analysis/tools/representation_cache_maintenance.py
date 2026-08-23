#!/usr/bin/env python3
"""Audit and maintain representation caches safely.

With no maintenance flags, this command performs a read-only audit. Maintenance
is dry-run by default. Supported cleanup operations include:

1. ``--merge-pythia-step-aliases`` repairs the short-lived cache-ID regression
   that changed ``@stepNNN`` into ``_stepNNN``. Missing keys are merged into the
   canonical ``@stepNNN`` file. Overlapping keys are considered redundant only
   when their vectors are numerically equivalent under explicit thresholds.
   Hard conflicts prevent automatic retirement of the alias file.

2. ``--prune-redundant-legacy`` removes only legacy cache entries that have a
   current-schema counterpart for the same prompt/model/base representation and
   whose vectors are numerically equivalent. Legacy-only prompts and materially
   different historical vectors are preserved.

``--prune-unreachable-legacy`` is a separate historical-data purge. It deletes
all old-schema entries whether or not their vectors match current values, and
therefore requires ``--allow-historical-data-loss`` when used with ``--apply``.

``--prune-pooling {last,mean}`` removes representation entries that use the
selected token-pooling method, regardless of cache schema. This is useful after
standardizing a repository on one pooling method. It is always dry-run unless
``--apply`` is supplied.

Numerical equivalence is evaluated in a scale-aware way. The default
``spectral`` profile is intended for these representation caches: cosine
similarity >= 0.99995 and symmetric relative L2 error <= 1e-2. A single
coordinate is not allowed to veto an otherwise near-identical high-dimensional
vector; max absolute difference is reported diagnostically and can be made a
hard bound explicitly with ``--max-abs-diff``. Use ``--equivalence-profile
strict`` for the older tighter cosine/L2 thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CURRENT_REQUIRED = {
    "spectral_space",
    "rep_pooling",
    "max_seq_len",
    "rep_hook_name",
    "model_backend",
}
BASE_FIELDS = ("spectral_space", "rep_pooling", "max_seq_len")
PYTHIA_BUGGY_ID_RE = re.compile(r"^(EleutherAI_pythia-[A-Za-z0-9.\-]+)_step([0-9]+)$")


@dataclass(frozen=True)
class Thresholds:
    min_cosine: float = 0.99995
    max_relative_l2: float = 1e-2
    max_abs_diff: float = math.inf


EQUIVALENCE_PROFILES = {
    # Practical near-duplicate criterion for high-dimensional hidden-state
    # representations used by the spectral pipeline.  The observed historical
    # drift in these caches is sub-1% in relative L2 with cosine > 0.99997.
    "spectral": Thresholds(0.99995, 1e-2, math.inf),
    # Tighter option for users who want only very small numerical drift.
    "strict": Thresholds(0.999999, 1e-3, math.inf),
}


@dataclass(frozen=True)
class PairMetric:
    exact: bool
    equivalent: bool
    cosine: float
    relative_l2: float
    mean_abs: float
    max_abs: float
    finite: bool
    shape_match: bool


@dataclass
class MergeStats:
    source_entries: int = 0
    rewritten_entries: int = 0
    added: int = 0
    exact_duplicates: int = 0
    numerical_duplicates: int = 0
    hard_conflicts: int = 0
    foreign_or_malformed: int = 0
    source_vector_bytes: int = 0
    worst_cosine: float = 1.0
    worst_relative_l2: float = 0.0
    worst_max_abs: float = 0.0

    @property
    def duplicate_overlaps(self) -> int:
        return self.exact_duplicates + self.numerical_duplicates


@dataclass
class LegacyPruneStats:
    legacy_entries: int = 0
    legacy_bytes: int = 0
    overlap_entries: int = 0
    exact_duplicates: int = 0
    numerical_duplicates: int = 0
    hard_differences: int = 0
    no_current_counterpart: int = 0
    removed_entries: int = 0
    removed_bytes: int = 0


@dataclass
class PoolingPruneStats:
    matched_entries: int = 0
    matched_bytes: int = 0
    unrecognized_entries: int = 0



def fmt_bytes(n: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    x = float(n)
    for unit in units:
        if x < 1024.0 or unit == units[-1]:
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{n} B"


def fmt_metric(x: float) -> str:
    if not math.isfinite(x):
        return str(x)
    if 1e-3 <= abs(x) < 1e4:
        return f"{x:.8f}"
    return f"{x:.3e}"


def value_nbytes(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, (int, float)):
        return int(nbytes)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    return 0


def parse_rep_key(key: Any):
    if not (isinstance(key, tuple) and len(key) == 3):
        return None
    prompt_key, model_id, cfg_raw = key
    if not isinstance(cfg_raw, str):
        return None
    try:
        cfg = json.loads(cfg_raw)
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None
    return prompt_key, str(model_id), cfg_raw, cfg


def is_current_cfg(cfg: dict[str, Any]) -> bool:
    return CURRENT_REQUIRED.issubset(cfg)


def base_signature(cfg: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(cfg.get(k, None) for k in BASE_FIELDS)


def cfg_label(cfg: dict[str, Any]) -> str:
    return ", ".join(
        [
            f"pool={cfg.get('rep_pooling', '<missing>')}",
            f"space={cfg.get('spectral_space', '<missing>')}",
            f"max_seq_len={cfg.get('max_seq_len', '<missing>')}",
            f"backend={cfg.get('model_backend', '<missing>')}",
            f"hook={cfg.get('rep_hook_name', '<missing>')}",
        ]
    )


def vector_metric(a: Any, b: Any, thresholds: Thresholds) -> PairMetric:
    try:
        aa0 = np.asarray(a)
        bb0 = np.asarray(b)
    except Exception:
        return PairMetric(False, False, math.nan, math.inf, math.inf, math.inf, False, False)
    if aa0.shape != bb0.shape:
        return PairMetric(False, False, math.nan, math.inf, math.inf, math.inf, False, False)

    exact = aa0.dtype == bb0.dtype and np.array_equal(aa0, bb0)
    try:
        aa = aa0.astype(np.float64, copy=False).ravel()
        bb = bb0.astype(np.float64, copy=False).ravel()
    except Exception:
        return PairMetric(exact, False, math.nan, math.inf, math.inf, math.inf, False, True)
    finite = bool(np.isfinite(aa).all() and np.isfinite(bb).all())
    if not finite:
        return PairMetric(exact, False, math.nan, math.inf, math.inf, math.inf, False, True)

    diff = aa - bb
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    nd = float(np.linalg.norm(diff))
    denom = max(na, nb, np.finfo(np.float64).tiny)
    relative_l2 = nd / denom
    if na == 0.0 and nb == 0.0:
        cosine = 1.0
    elif na == 0.0 or nb == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(aa, bb) / (na * nb))
        cosine = max(-1.0, min(1.0, cosine))
    absdiff = np.abs(diff)
    mean_abs = float(absdiff.mean()) if absdiff.size else 0.0
    max_abs = float(absdiff.max()) if absdiff.size else 0.0
    equivalent = (
        cosine >= thresholds.min_cosine
        and relative_l2 <= thresholds.max_relative_l2
        and max_abs <= thresholds.max_abs_diff
    )
    return PairMetric(exact, equivalent, cosine, relative_l2, mean_abs, max_abs, True, True)


def atomic_pickle_write(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        with tmp.open("rb") as f:
            check = pickle.load(f)
        if isinstance(obj, dict) and (not isinstance(check, dict) or len(check) != len(obj)):
            raise RuntimeError(f"verification failed for temporary cache {tmp}")
        del check
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_dict(path: Path) -> dict[Any, Any]:
    with path.open("rb") as f:
        value = pickle.load(f)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected dict, got {type(value).__name__}")
    return value


def backup_path(path: Path) -> Path:
    candidate = path.with_suffix(path.suffix + ".bak")
    if candidate.exists():
        raise FileExistsError(f"backup already exists: {candidate}")
    return candidate


def remove_or_backup(path: Path, *, backup: bool) -> None:
    if backup:
        shutil.move(path, backup_path(path))
    else:
        path.unlink()


def canonical_pythia_id(model_id: str) -> str | None:
    match = PYTHIA_BUGGY_ID_RE.fullmatch(model_id)
    if not match:
        return None
    return f"{match.group(1)}@step{match.group(2)}"


def buggy_pythia_files(paths: Iterable[Path]) -> list[tuple[Path, str, str]]:
    result: list[tuple[Path, str, str]] = []
    for path in paths:
        stem = path.name
        if not (stem.startswith("_") and stem.endswith("_reps_cache.pkl")):
            continue
        file_model_id = stem[1 : -len("_reps_cache.pkl")]
        canonical = canonical_pythia_id(file_model_id)
        if canonical is not None:
            result.append((path, file_model_id, canonical))
    return result


def merge_pythia_alias(
    source_path: Path,
    source_model_id: str,
    canonical_model_id: str,
    *,
    apply: bool,
    backup: bool,
    thresholds: Thresholds,
) -> MergeStats:
    target_path = source_path.with_name(f"_{canonical_model_id}_reps_cache.pkl")
    source = load_dict(source_path)
    target = load_dict(target_path) if target_path.exists() else {}
    stats = MergeStats(source_entries=len(source))
    additions: dict[Any, Any] = {}

    for key, value in source.items():
        stats.source_vector_bytes += value_nbytes(value)
        parsed = parse_rep_key(key)
        if parsed is None:
            stats.foreign_or_malformed += 1
            continue
        prompt_key, model_id, cfg_raw, _cfg = parsed
        if model_id != source_model_id:
            stats.foreign_or_malformed += 1
            continue

        stats.rewritten_entries += 1
        new_key = (prompt_key, canonical_model_id, cfg_raw)
        if new_key not in target:
            additions[new_key] = value
            stats.added += 1
            continue

        metric = vector_metric(value, target[new_key], thresholds)
        if metric.exact:
            stats.exact_duplicates += 1
        elif metric.equivalent:
            stats.numerical_duplicates += 1
        else:
            stats.hard_conflicts += 1
        if metric.shape_match and metric.finite:
            stats.worst_cosine = min(stats.worst_cosine, metric.cosine)
            stats.worst_relative_l2 = max(stats.worst_relative_l2, metric.relative_l2)
            stats.worst_max_abs = max(stats.worst_max_abs, metric.max_abs)

    print(f"\nPythia cache alias: {source_path}")
    print(f"  canonical target: {target_path}")
    print(f"  source entries: {stats.source_entries:,} ({fmt_bytes(stats.source_vector_bytes)} raw vectors)")
    print(f"  rewritable entries: {stats.rewritten_entries:,}")
    print(f"  new entries to add: {stats.added:,}")
    print(f"  exact duplicate overlaps: {stats.exact_duplicates:,}")
    print(f"  numerical duplicate overlaps: {stats.numerical_duplicates:,}")
    print(f"  materially different overlaps: {stats.hard_conflicts:,}")
    print(f"  foreign/malformed entries: {stats.foreign_or_malformed:,}")
    overlap_n = stats.duplicate_overlaps + stats.hard_conflicts
    if overlap_n:
        print(
            "  worst overlap metrics: "
            f"cosine={fmt_metric(stats.worst_cosine)}, "
            f"relative_L2={fmt_metric(stats.worst_relative_l2)}, "
            f"max_abs={fmt_metric(stats.worst_max_abs)}"
        )

    safe_to_retire_source = stats.hard_conflicts == 0 and stats.foreign_or_malformed == 0
    if not safe_to_retire_source:
        print("  ACTION: keep source; materially different or malformed overlaps prevent automatic retirement")
        return stats

    if not apply:
        if stats.added:
            print("  DRY RUN: would merge missing entries into canonical target, then retire source alias")
        else:
            print("  DRY RUN: alias is redundant under numerical-equivalence thresholds; would retire source")
        return stats

    if stats.added:
        merged = dict(target)
        merged.update(additions)
        if backup and target_path.exists():
            target_backup = backup_path(target_path)
            print(f"  backing up target -> {target_backup}")
            shutil.copy2(target_path, target_backup)
        print(f"  writing canonical target atomically ({len(merged):,} entries)")
        atomic_pickle_write(target_path, merged)
        del merged

    if backup:
        src_backup = backup_path(source_path)
        print(f"  retiring source -> {src_backup}")
        shutil.move(source_path, src_backup)
    else:
        print("  deleting redundant source alias")
        source_path.unlink()
    print("  ACTION: completed")
    return stats


def group_metadata(cache: dict[Any, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for key, value in cache.items():
        parsed = parse_rep_key(key)
        if parsed is None:
            continue
        _prompt_key, model_id, cfg_raw, cfg = parsed
        g = groups.setdefault(
            (model_id, cfg_raw),
            {"model": model_id, "cfg_raw": cfg_raw, "cfg": cfg, "count": 0, "nbytes": 0},
        )
        g["count"] += 1
        g["nbytes"] += value_nbytes(value)
    return groups


def prompt_value_lookup(cache: dict[Any, Any], model_id: str, cfg_raw: str) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key, value in cache.items():
        parsed = parse_rep_key(key)
        if parsed is None:
            continue
        prompt_key, mid, raw, _cfg = parsed
        if mid == model_id and raw == cfg_raw:
            result[prompt_key] = value
    return result


def redundant_legacy_plan(cache: dict[Any, Any], thresholds: Thresholds) -> tuple[set[Any], LegacyPruneStats]:
    groups = group_metadata(cache)
    current_by_base: dict[tuple[str, tuple[Any, ...]], list[dict[str, Any]]] = defaultdict(list)
    legacy_groups: list[dict[str, Any]] = []
    for g in groups.values():
        sig = (g["model"], base_signature(g["cfg"]))
        if is_current_cfg(g["cfg"]):
            current_by_base[sig].append(g)
        else:
            legacy_groups.append(g)

    removable_keys: set[Any] = set()
    stats = LegacyPruneStats()
    for lg in legacy_groups:
        stats.legacy_entries += int(lg["count"])
        stats.legacy_bytes += int(lg["nbytes"])
        legacy_lookup = prompt_value_lookup(cache, lg["model"], lg["cfg_raw"])
        candidates = current_by_base.get((lg["model"], base_signature(lg["cfg"])), [])
        if not candidates:
            stats.no_current_counterpart += len(legacy_lookup)
            continue

        candidate_lookups = [
            (cg, prompt_value_lookup(cache, cg["model"], cg["cfg_raw"]))
            for cg in candidates
        ]
        # Prefer the current namespace with the greatest prompt overlap.
        cg, modern_lookup = max(candidate_lookups, key=lambda pair: len(set(legacy_lookup) & set(pair[1])))
        overlap_prompts = set(legacy_lookup) & set(modern_lookup)
        stats.overlap_entries += len(overlap_prompts)
        stats.no_current_counterpart += len(legacy_lookup) - len(overlap_prompts)

        for prompt_key in overlap_prompts:
            metric = vector_metric(legacy_lookup[prompt_key], modern_lookup[prompt_key], thresholds)
            old_key = (prompt_key, lg["model"], lg["cfg_raw"])
            if metric.exact:
                stats.exact_duplicates += 1
                removable_keys.add(old_key)
                stats.removed_bytes += value_nbytes(legacy_lookup[prompt_key])
            elif metric.equivalent:
                stats.numerical_duplicates += 1
                removable_keys.add(old_key)
                stats.removed_bytes += value_nbytes(legacy_lookup[prompt_key])
            else:
                stats.hard_differences += 1

    stats.removed_entries = len(removable_keys)
    return removable_keys, stats


def prune_redundant_legacy(path: Path, *, apply: bool, backup: bool, thresholds: Thresholds) -> LegacyPruneStats:
    cache = load_dict(path)
    removable, stats = redundant_legacy_plan(cache, thresholds)
    if stats.legacy_entries == 0:
        return stats

    print(f"\nLegacy redundancy analysis: {path}")
    print(f"  legacy entries: {stats.legacy_entries:,} ({fmt_bytes(stats.legacy_bytes)})")
    print(f"  prompts with current counterpart: {stats.overlap_entries:,}")
    print(f"  exact duplicates: {stats.exact_duplicates:,}")
    print(f"  numerical duplicates: {stats.numerical_duplicates:,}")
    print(f"  materially different overlaps preserved: {stats.hard_differences:,}")
    print(f"  legacy-only prompts preserved: {stats.no_current_counterpart:,}")
    print(f"  removable as redundant: {stats.removed_entries:,} ({fmt_bytes(stats.removed_bytes)})")

    if not removable:
        print("  ACTION: nothing qualifies as redundant under the current thresholds")
        return stats
    if not apply:
        print("  DRY RUN: would remove only the exact/numerically-equivalent legacy entries above")
        return stats

    kept = {k: v for k, v in cache.items() if k not in removable}
    if backup:
        bak = backup_path(path)
        print(f"  backing up original -> {bak}")
        shutil.copy2(path, bak)
    print(f"  rewriting atomically: {len(cache):,} -> {len(kept):,} entries")
    atomic_pickle_write(path, kept)
    print("  ACTION: redundant legacy entries removed")
    return stats


def legacy_summary(cache: dict[Any, Any]) -> tuple[int, int, int]:
    legacy_entries = 0
    legacy_bytes = 0
    unrecognized = 0
    for key, value in cache.items():
        parsed = parse_rep_key(key)
        if parsed is None:
            unrecognized += 1
            continue
        _prompt_key, _model_id, _cfg_raw, cfg = parsed
        if not is_current_cfg(cfg):
            legacy_entries += 1
            legacy_bytes += value_nbytes(value)
    return legacy_entries, legacy_bytes, unrecognized


def prune_legacy(path: Path, *, apply: bool, backup: bool) -> tuple[int, int]:
    cache = load_dict(path)
    legacy_entries, legacy_bytes, unrecognized = legacy_summary(cache)
    if legacy_entries == 0:
        return 0, 0

    print(f"\nUnreachable legacy namespace(s): {path}")
    print(f"  legacy entries: {legacy_entries:,} ({fmt_bytes(legacy_bytes)} raw vectors)")
    if unrecognized:
        print(f"  unrecognized entries preserved: {unrecognized:,}")
    print("  WARNING: this operation does not test redundancy; materially different historical vectors may be deleted")

    if not apply:
        print("  DRY RUN: would purge all old-schema entries")
        return legacy_entries, legacy_bytes

    if legacy_entries == len(cache):
        print("  all entries are legacy; retiring entire cache file")
        remove_or_backup(path, backup=backup)
        return legacy_entries, legacy_bytes

    kept = {}
    for key, value in cache.items():
        parsed = parse_rep_key(key)
        if parsed is not None:
            _prompt_key, _model_id, _cfg_raw, cfg = parsed
            if not is_current_cfg(cfg):
                continue
        kept[key] = value

    if backup:
        bak = backup_path(path)
        print(f"  backing up original -> {bak}")
        shutil.copy2(path, bak)
    print(f"  rewriting atomically: {len(cache):,} -> {len(kept):,} entries")
    atomic_pickle_write(path, kept)
    return legacy_entries, legacy_bytes


def pooling_summary(cache: dict[Any, Any], pooling: str) -> tuple[set[Any], PoolingPruneStats]:
    removable: set[Any] = set()
    stats = PoolingPruneStats()
    for key, value in cache.items():
        parsed = parse_rep_key(key)
        if parsed is None:
            stats.unrecognized_entries += 1
            continue
        _prompt_key, _model_id, _cfg_raw, cfg = parsed
        if str(cfg.get("rep_pooling", "")) == str(pooling):
            removable.add(key)
            stats.matched_entries += 1
            stats.matched_bytes += value_nbytes(value)
    return removable, stats


def prune_pooling(path: Path, *, pooling: str, apply: bool, backup: bool) -> PoolingPruneStats:
    cache = load_dict(path)
    removable, stats = pooling_summary(cache, pooling)
    if not removable:
        return stats

    print(f"\nPooling cleanup: {path}")
    print(f"  pooling: {pooling}")
    print(f"  removable entries: {stats.matched_entries:,} ({fmt_bytes(stats.matched_bytes)})")
    if stats.unrecognized_entries:
        print(f"  unrecognized entries preserved: {stats.unrecognized_entries:,}")
    if not apply:
        print("  DRY RUN: would remove all representation entries using this pooling method")
        return stats

    if len(removable) == len(cache):
        print("  all recognized cache entries use this pooling method; retiring entire cache file")
        remove_or_backup(path, backup=backup)
        return stats

    kept = {k: v for k, v in cache.items() if k not in removable}
    if backup:
        bak = backup_path(path)
        print(f"  backing up original -> {bak}")
        shutil.copy2(path, bak)
    print(f"  rewriting atomically: {len(cache):,} -> {len(kept):,} entries")
    atomic_pickle_write(path, kept)
    print("  ACTION: pooling-specific entries removed")
    return stats


def discover(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("*_reps_cache.pkl"))




# ---- Read-only audit/reporting -------------------------------------------------

def prompt_digest(prompt_key: Any, model_id: str) -> bytes:
    b = pickle.dumps((prompt_key, model_id), protocol=4)
    return hashlib.blake2b(b, digest_size=16).digest()

def deterministic_sample(items: list[tuple[bytes, Any, Any]], limit: int) -> list[tuple[bytes, Any, Any]]:
    if limit <= 0 or len(items) <= limit:
        return items
    # Digest ordering makes the sample deterministic and independent of pickle insertion order.
    return sorted(items, key=lambda x: x[0])[:limit]

def summarize_metrics(metrics: list[PairMetric], *, total_overlap: int, thresholds: Thresholds) -> dict[str, Any]:
    n = len(metrics)
    exact = sum(m.exact for m in metrics)
    equivalent = sum(m.equivalent for m in metrics)
    shape_mismatch = sum(not m.shape_match for m in metrics)
    nonfinite = sum(m.shape_match and not m.finite for m in metrics)
    valid = [m for m in metrics if m.shape_match and m.finite]

    def summary(vals: Iterable[float], *, lower_is_better: bool) -> dict[str, float]:
        arr = np.asarray(list(vals), dtype=np.float64) if np is not None else []
        if len(arr) == 0:
            return {}
        qs = np.quantile(arr, [0.0, 0.01, 0.5, 0.95, 0.99, 1.0])
        return {
            "min": float(qs[0]),
            "p01": float(qs[1]),
            "median": float(qs[2]),
            "p95": float(qs[3]),
            "p99": float(qs[4]),
            "max": float(qs[5]),
            "mean": float(arr.mean()),
        }

    full_checked = n == total_overlap
    all_equivalent_checked = n > 0 and equivalent == n
    return {
        "checked": n,
        "total_overlap": total_overlap,
        "full_checked": full_checked,
        "exact": exact,
        "equivalent": equivalent,
        "shape_mismatch": shape_mismatch,
        "nonfinite": nonfinite,
        "all_equivalent_checked": all_equivalent_checked,
        "cosine": summary((m.cosine for m in valid), lower_is_better=False),
        "relative_l2": summary((m.relative_l2 for m in valid), lower_is_better=True),
        "mean_abs": summary((m.mean_abs for m in valid), lower_is_better=True),
        "max_abs": summary((m.max_abs for m in valid), lower_is_better=True),
        "thresholds": thresholds,
    }

def print_metric_summary(s: dict[str, Any]) -> None:
    checked = s["checked"]
    total = s["total_overlap"]
    suffix = "all overlaps" if s["full_checked"] else f"deterministic sample of {checked:,}/{total:,} overlaps"
    print(f"    numerical comparison: {suffix}")
    print(f"      exact bitwise vectors: {s['exact']:,}/{checked:,} ({s['exact']/max(1,checked):.1%})")
    print(f"      numerically equivalent: {s['equivalent']:,}/{checked:,} ({s['equivalent']/max(1,checked):.1%})")
    if s["shape_mismatch"]:
        print(f"      shape mismatches: {s['shape_mismatch']:,}")
    if s["nonfinite"]:
        print(f"      non-finite pairs: {s['nonfinite']:,}")
    c = s["cosine"]
    r = s["relative_l2"]
    ma = s["mean_abs"]
    xa = s["max_abs"]
    if c:
        print(
            "      cosine: "
            f"min={fmt_metric(c['min'])}, p01={fmt_metric(c['p01'])}, "
            f"median={fmt_metric(c['median'])}, mean={fmt_metric(c['mean'])}"
        )
        print(
            "      relative L2: "
            f"median={fmt_metric(r['median'])}, p95={fmt_metric(r['p95'])}, "
            f"p99={fmt_metric(r['p99'])}, max={fmt_metric(r['max'])}"
        )
        print(
            "      abs diff: "
            f"mean(|Δ|) median={fmt_metric(ma['median'])}, "
            f"max(|Δ|) p99={fmt_metric(xa['p99'])}, max={fmt_metric(xa['max'])}"
        )
    t: Thresholds = s["thresholds"]
    print(
        "      equivalence thresholds: "
        f"cosine >= {t.min_cosine:g}, relative_L2 <= {t.max_relative_l2:g}"
        + ("" if math.isinf(t.max_abs_diff) else f", max_abs <= {t.max_abs_diff:g}")
    )
    if s["full_checked"] and s["all_equivalent_checked"]:
        print("      classification: NUMERICAL NEAR-DUPLICATE over every overlapping prompt")
    elif s["all_equivalent_checked"]:
        print("      classification: sample is consistent with a numerical near-duplicate; use --verify-limit 0 before pruning")
    else:
        print("      classification: NOT numerically equivalent under these thresholds")

def build_group_prompt_values(cache: dict, model_id: str, cfg_raw: str) -> dict[bytes, Any]:
    out: dict[bytes, Any] = {}
    for k, v in cache.items():
        parsed = parse_rep_key(k)
        if parsed is None:
            continue
        prompt_key, mid, raw, _cfg = parsed
        if mid == model_id and raw == cfg_raw:
            out[prompt_digest(prompt_key, mid)] = v
    return out

def compare_groups(cache: dict, legacy: dict, modern: dict, *, verify_limit: int, thresholds: Thresholds) -> dict[str, Any]:
    legacy_values = build_group_prompt_values(cache, legacy["model"], legacy["cfg_raw"])
    modern_values = build_group_prompt_values(cache, modern["model"], modern["cfg_raw"])
    overlap = sorted(set(legacy_values) & set(modern_values))
    pairs = [(d, legacy_values[d], modern_values[d]) for d in overlap]
    selected = deterministic_sample(pairs, verify_limit)
    metrics = [vector_metric(a, b, thresholds) for _d, a, b in selected]
    return summarize_metrics(metrics, total_overlap=len(overlap), thresholds=thresholds)

def audit_file(path: Path, *, verify_legacy_values: bool, verify_limit: int, thresholds: Thresholds):
    print(f"\n=== {path} ===")
    print(f"file size: {fmt_bytes(path.stat().st_size)}")
    file_model_id = path.name[1 : -len("_reps_cache.pkl")] if path.name.startswith("_") and path.name.endswith("_reps_cache.pkl") else None
    if file_model_id:
        canonical = canonical_pythia_id(file_model_id)
        if canonical:
            print(f"WARNING: accidental Pythia cache alias; patched code expects @{canonical.split('@',1)[1]}: _{canonical}_reps_cache.pkl")

    with path.open("rb") as f:
        cache = pickle.load(f)
    if not isinstance(cache, dict):
        print(f"WARNING: expected dict, got {type(cache).__name__}; skipping namespace analysis")
        return cache, None
    if not cache:
        print("entries: 0 (empty cache; safe to remove if no process is using it)")
        return cache, {"groups": {}, "legacy_shadow_report": [], "malformed": 0}

    groups = {}
    malformed = 0
    for k, v in cache.items():
        parsed = parse_rep_key(k)
        if parsed is None:
            malformed += 1
            continue
        _prompt_key, model_id, cfg_raw, cfg = parsed
        gkey = (model_id, cfg_raw)
        if gkey not in groups:
            groups[gkey] = {"model": model_id, "cfg_raw": cfg_raw, "cfg": cfg, "count": 0, "nbytes": 0}
        g = groups[gkey]
        g["count"] += 1
        g["nbytes"] += value_nbytes(v)

    print(f"entries: {len(cache):,}")
    if malformed:
        print(f"unrecognized/non-representation keys: {malformed:,}")

    ordered = sorted(groups.values(), key=lambda g: (g["model"], not is_current_cfg(g["cfg"]), g["cfg_raw"]))
    for i, g in enumerate(ordered, 1):
        schema = "CURRENT-SCHEMA" if is_current_cfg(g["cfg"]) else "LEGACY/UNREACHABLE-BY-CURRENT-SCHEMA"
        if canonical_pythia_id(g["model"]):
            schema += "/BUGGY-PYTHIA-MODEL-ID"
        print(f"\n[{i}] {schema}")
        print(f"  model: {g['model']}")
        print(f"  entries: {g['count']:,}")
        if g["nbytes"]:
            print(f"  raw vector bytes: {fmt_bytes(g['nbytes'])}")
        print(f"  {cfg_label(g['cfg'])}")

    current_by_base = defaultdict(list)
    legacy_groups = []
    for g in groups.values():
        key = (g["model"], base_signature(g["cfg"]))
        if is_current_cfg(g["cfg"]):
            current_by_base[key].append(g)
        else:
            legacy_groups.append(g)

    legacy_shadow_report = []
    if legacy_groups:
        group_prompt_sets = defaultdict(set)
        for k in cache.keys():
            parsed = parse_rep_key(k)
            if parsed is None:
                continue
            prompt_key, model_id, cfg_raw, _cfg = parsed
            group_prompt_sets[(model_id, cfg_raw)].add(prompt_digest(prompt_key, model_id))

        print("\nLegacy shadowing checks:")
        for lg in legacy_groups:
            candidates = current_by_base.get((lg["model"], base_signature(lg["cfg"])), [])
            if not candidates:
                print(f"  legacy {cfg_label(lg['cfg'])}: no current-schema namespace with same base fields")
                legacy_shadow_report.append((lg, None, 0, False, None))
                continue
            lset = group_prompt_sets[(lg["model"], lg["cfg_raw"])]
            best = max(candidates, key=lambda cg: len(lset & group_prompt_sets[(cg["model"], cg["cfg_raw"])]))
            best_overlap = len(lset & group_prompt_sets[(best["model"], best["cfg_raw"])])
            full_shadow = best_overlap == len(lset)
            print(
                f"  legacy {cfg_label(lg['cfg'])}\n"
                f"    best current match: {cfg_label(best['cfg'])}\n"
                f"    prompt overlap: {best_overlap:,}/{len(lset):,} ({best_overlap / max(1,len(lset)):.1%})"
            )
            numerical = None
            if verify_legacy_values and best_overlap:
                numerical = compare_groups(cache, lg, best, verify_limit=verify_limit, thresholds=thresholds)
                print_metric_summary(numerical)
            legacy_shadow_report.append((lg, best, best_overlap, full_shadow, numerical))

    return cache, {"groups": groups, "legacy_shadow_report": legacy_shadow_report, "malformed": malformed}

def cross_file_overlap(paths: list[Path]):
    print("\n=== Cross-file exact cache-key overlap ===")
    seen = {}
    duplicate_count = 0
    pair_counts = defaultdict(int)
    for path in paths:
        with path.open("rb") as f:
            cache = pickle.load(f)
        if not isinstance(cache, dict):
            continue
        for k in cache.keys():
            parsed = parse_rep_key(k)
            if parsed is None:
                continue
            d = hashlib.blake2b(pickle.dumps(k, protocol=4), digest_size=16).digest()
            prev = seen.get(d)
            if prev is not None and prev != path:
                duplicate_count += 1
                pair_counts[tuple(sorted((str(prev), str(path))))] += 1
            else:
                seen[d] = path
        del cache
    print(f"duplicate full keys occurring in >1 cache file: {duplicate_count:,}")
    for (a, b), n in sorted(pair_counts.items(), key=lambda x: -x[1])[:30]:
        print(f"  {n:,}  {a}  <->  {b}")
    if duplicate_count:
        print("NOTE: these are duplicate cache keys across task files; this check does not claim their vector bytes are identical.")

def main() -> int:
    here = Path(__file__).resolve()
    # Installed location is usually <repo>/code/analysis/tools/...; a copied
    # standalone script is often placed directly in <repo>. Keep both usable.
    try:
        installed_repo_root = here.parents[3]
    except IndexError:
        installed_repo_root = here.parent
    repo_root = installed_repo_root if (installed_repo_root / "cache").exists() else Path.cwd()
    default_root = repo_root / "cache"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", default=str(default_root), help="cache root or one *_reps_cache.pkl file")

    audit_group = ap.add_argument_group("audit (read-only)")
    audit_group.add_argument("--audit", action="store_true", help="print the cache audit explicitly (this is the default when no maintenance operation is selected)")
    audit_group.add_argument("--verify-legacy-values", action="store_true", help="numerically compare overlapping legacy/current representations")
    audit_group.add_argument("--verify-limit", type=int, default=10_000, help="max overlaps to compare per namespace pair; 0 means all (default: 10000)")
    audit_group.add_argument("--cross-file-overlap", action="store_true", help="count identical full representation keys duplicated across cache files")

    maintenance_group = ap.add_argument_group("maintenance (dry-run unless --apply)")
    maintenance_group.add_argument("--merge-pythia-step-aliases", action="store_true", help="merge accidental _stepNNN cache IDs into canonical @stepNNN IDs")
    maintenance_group.add_argument("--prune-redundant-legacy", action="store_true", help="remove only legacy entries with a numerically-equivalent current counterpart (profile-controlled)")
    maintenance_group.add_argument("--prune-unreachable-legacy", action="store_true", help="purge all old-schema namespaces, even materially different historical values")
    maintenance_group.add_argument("--prune-pooling", choices=["last", "mean"], default=None, help="remove all representation entries using the selected token-pooling method")
    maintenance_group.add_argument("--allow-historical-data-loss", action="store_true", help="required with --apply --prune-unreachable-legacy")
    maintenance_group.add_argument("--apply", action="store_true", help="perform requested maintenance changes; otherwise maintenance is read-only")
    maintenance_group.add_argument("--backup", action="store_true", help="keep .bak copies when modifying caches (can require substantial disk space)")

    threshold_group = ap.add_argument_group("numerical-equivalence thresholds")
    threshold_group.add_argument(
        "--equivalence-profile",
        choices=sorted(EQUIVALENCE_PROFILES),
        default="spectral",
        help=(
            "threshold preset: 'spectral' treats sub-1%% relative-L2 / near-collinear "
            "representations as near-duplicates; 'strict' uses the older tighter "
            "cosine/L2 rule (default: spectral)"
        ),
    )
    threshold_group.add_argument("--min-cosine", type=float, default=None, help="override the profile minimum cosine similarity")
    threshold_group.add_argument("--max-relative-l2", type=float, default=None, help="override the profile maximum symmetric relative L2 error")
    threshold_group.add_argument(
        "--max-abs-diff",
        type=float,
        default=None,
        help=(
            "optional hard maximum coordinate error; disabled by default because a "
            "single coordinate is a poor equivalence test for high-dimensional vectors"
        ),
    )
    args = ap.parse_args()

    if args.verify_limit < 0:
        ap.error("--verify-limit must be >= 0")
    if args.apply and args.prune_unreachable_legacy and not args.allow_historical_data_loss:
        ap.error("--apply --prune-unreachable-legacy requires --allow-historical-data-loss")
    profile = EQUIVALENCE_PROFILES[args.equivalence_profile]
    thresholds = Thresholds(
        profile.min_cosine if args.min_cosine is None else args.min_cosine,
        profile.max_relative_l2 if args.max_relative_l2 is None else args.max_relative_l2,
        profile.max_abs_diff if args.max_abs_diff is None else args.max_abs_diff,
    )
    if not (-1.0 <= thresholds.min_cosine <= 1.0):
        ap.error("--min-cosine must be between -1 and 1")
    if thresholds.max_relative_l2 < 0 or thresholds.max_abs_diff < 0:
        ap.error("error thresholds must be non-negative")
    root = Path(args.root).expanduser().resolve()
    paths = discover(root)
    if not paths:
        print(f"No representation caches found under {root}", file=sys.stderr)
        return 2

    maintenance_requested = bool(
        args.merge_pythia_step_aliases
        or args.prune_redundant_legacy
        or args.prune_unreachable_legacy
        or args.prune_pooling is not None
    )
    audit_requested = bool(
        args.audit
        or args.verify_legacy_values
        or args.cross_file_overlap
        or not maintenance_requested
    )

    print(f"Cache files found: {len(paths)}")
    print(f"On-disk size: {fmt_bytes(sum(p.stat().st_size for p in paths))}")
    print(f"Equivalence profile: {args.equivalence_profile}")
    print(
        "Numerical-equivalence thresholds: "
        f"cosine >= {thresholds.min_cosine:g}, relative_L2 <= {thresholds.max_relative_l2:g}"
        + (", max_abs diagnostic-only" if math.isinf(thresholds.max_abs_diff) else f", max_abs <= {thresholds.max_abs_diff:g}")
    )

    if audit_requested:
        if args.verify_legacy_values:
            scope = "all overlaps" if args.verify_limit == 0 else f"up to {args.verify_limit:,} deterministic overlaps per namespace pair"
            print(f"Numerical verification: {scope}")
        for path in paths:
            cache, _report = audit_file(
                path,
                verify_legacy_values=args.verify_legacy_values,
                verify_limit=args.verify_limit,
                thresholds=thresholds,
            )
            del cache
        if args.cross_file_overlap and len(paths) > 1:
            cross_file_overlap(paths)

    if maintenance_requested:
        print(f"\nMaintenance mode: {'APPLY' if args.apply else 'DRY RUN'}")

    if args.merge_pythia_step_aliases:
        aliases = buggy_pythia_files(discover(root))
        if not aliases:
            print("\nNo accidental Pythia _stepNNN cache aliases found.")
        else:
            print(f"\nFound {len(aliases)} accidental Pythia _stepNNN alias file(s).")
            for source_path, source_id, canonical_id in aliases:
                merge_pythia_alias(
                    source_path,
                    source_id,
                    canonical_id,
                    apply=args.apply,
                    backup=args.backup,
                    thresholds=thresholds,
                )

    if args.prune_redundant_legacy:
        paths = discover(root)
        total_removed = 0
        total_bytes = 0
        total_hard = 0
        total_unique = 0
        for path in paths:
            stats = prune_redundant_legacy(path, apply=args.apply, backup=args.backup, thresholds=thresholds)
            total_removed += stats.removed_entries
            total_bytes += stats.removed_bytes
            total_hard += stats.hard_differences
            total_unique += stats.no_current_counterpart
        print(f"\nRedundant-legacy total: {total_removed:,} entries ({fmt_bytes(total_bytes)})")
        print(f"Preserved materially different overlaps: {total_hard:,}")
        print(f"Preserved legacy-only prompts: {total_unique:,}")
        if total_removed and not args.apply:
            print("Re-run with --apply to remove only these numerically redundant legacy entries.")

    if args.prune_pooling is not None:
        paths = discover(root)
        total_entries = 0
        total_bytes = 0
        for path in paths:
            stats = prune_pooling(path, pooling=args.prune_pooling, apply=args.apply, backup=args.backup)
            total_entries += stats.matched_entries
            total_bytes += stats.matched_bytes
        print(f"\nPooling-{args.prune_pooling} cleanup total: {total_entries:,} entries ({fmt_bytes(total_bytes)})")
        if total_entries and not args.apply:
            print(f"Re-run with --apply to remove all rep_pooling={args.prune_pooling!r} entries.")

    if args.prune_unreachable_legacy:
        paths = discover(root)
        total_entries = 0
        total_bytes = 0
        for path in paths:
            n, b = prune_legacy(path, apply=args.apply, backup=args.backup)
            total_entries += n
            total_bytes += b
        print(f"\nHistorical-purge total: {total_entries:,} entries ({fmt_bytes(total_bytes)} raw vectors)")
        if not args.apply and total_entries:
            print("Applying this purge additionally requires --allow-historical-data-loss.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
