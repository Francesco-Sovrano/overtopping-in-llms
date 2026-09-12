#!/usr/bin/env python3
"""Lightweight Stage-5 circuit availability checks used by pipeline guards.

This module deliberately does *not* run circuit discovery.  It only inspects
existing Stage-5 artifacts.  That makes it safe to use for a pre-Stage-5
"reuse-only" execution policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Iterable

NO_USABLE_CIRCUIT_EXIT = 3
MODE_MISMATCH_EXIT = 4


def _metadata_is_explicit_full_network(meta: dict) -> bool:
    return bool(meta.get("explicit_full_network_ablation"))


def _metadata_has_usable_units(meta: dict) -> bool:
    if bool(meta.get("fallback_full_network")) or _metadata_is_explicit_full_network(meta):
        return False
    selected_n = meta.get("selected_n")
    try:
        if selected_n is not None and int(selected_n) <= 0:
            return False
    except (TypeError, ValueError):
        pass
    level = str(meta.get("level") or "neuron")
    if level == "node":
        return bool(meta.get("node_label_score"))
    if level == "edge":
        return bool(meta.get("edge_label_score"))
    return bool(
        meta.get("neurons")
        or meta.get("mlp_neurons")
        or meta.get("attention_heads")
        or meta.get("attn_heads")
        or meta.get("neuron_label_score")
    )


def summarize_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Stage-5 manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Stage-5 manifest must be a JSON list: {path}")
    statuses = Counter(str(row.get("status", "missing")) for row in payload if isinstance(row, dict))
    usable = []
    explicit_full = []
    for row in payload:
        if not isinstance(row, dict) or row.get("status") != "ok":
            continue
        meta = row.get("metadata_topn") or {}
        if _metadata_is_explicit_full_network(meta):
            explicit_full.append(row)
            continue
        if not _metadata_has_usable_units(meta):
            continue
        usable.append(row)
    if explicit_full:
        status = "stage5_mode_mismatch_explicit_full_network_present"
    else:
        status = "stage5_circuit_available" if usable else "no_usable_existing_circuit"
    return {
        "schema": "pipeline-circuit-availability-v2",
        "inspection_mode": "manifest",
        "stage5_manifest": str(path),
        "n_manifest_entries": int(len(payload)),
        "n_usable_circuits": int(len(usable)),
        "n_explicit_full_network_candidates": int(len(explicit_full)),
        "manifest_status_counts": dict(sorted(statuses.items())),
        "status": status,
    }


def _iter_cached_circuit_dirs(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return []
    # A historical cache may predate manifest.json.  A reusable Stage-5 circuit
    # itself is represented by sibling eval.json + scores.json files.
    return sorted({p.parent for p in root.rglob("scores.json") if (p.parent / "eval.json").is_file()})


def summarize_root(root: Path) -> dict:
    """Inspect an existing Stage-5 ``neural_circuits`` directory cache-only.

    Prefer manifest.json when present.  If the manifest is absent, fall back to
    scanning cached per-circuit ``scores.json``/``eval.json`` pairs.  This lets
    old valid caches be reused without forcing Stage 5 to recreate a manifest.
    """
    manifest = root / "manifest.json"
    if manifest.is_file():
        result = summarize_manifest(manifest)
        result["stage5_root"] = str(root)
        return result

    dirs = list(_iter_cached_circuit_dirs(root))
    usable_dirs: list[str] = []
    explicit_full_dirs: list[str] = []
    malformed_dirs: list[str] = []
    legacy_fallback_dirs: list[str] = []
    for circuit_dir in dirs:
        try:
            meta = json.loads((circuit_dir / "scores.json").read_text(encoding="utf-8"))
        except Exception:
            malformed_dirs.append(str(circuit_dir))
            continue
        if bool(meta.get("fallback_full_network")):
            legacy_fallback_dirs.append(str(circuit_dir))
            continue
        if _metadata_is_explicit_full_network(meta):
            explicit_full_dirs.append(str(circuit_dir))
            continue
        if _metadata_has_usable_units(meta):
            usable_dirs.append(str(circuit_dir))

    if explicit_full_dirs:
        status = "stage5_mode_mismatch_explicit_full_network_present"
    else:
        status = "stage5_circuit_available" if usable_dirs else "no_usable_existing_circuit"
    return {
        "schema": "pipeline-circuit-availability-v2",
        "inspection_mode": "cache_scan",
        "stage5_root": str(root),
        "stage5_manifest": None,
        "n_manifest_entries": 0,
        "n_cached_circuit_dirs": int(len(dirs)),
        "n_usable_circuits": int(len(usable_dirs)),
        "n_explicit_full_network_candidates": int(len(explicit_full_dirs)),
        "usable_circuit_dirs": usable_dirs,
        "explicit_full_network_dirs": explicit_full_dirs,
        "legacy_full_network_fallback_dirs": legacy_fallback_dirs,
        "malformed_circuit_dirs": malformed_dirs,
        "manifest_status_counts": {},
        "status": status,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path)
    src.add_argument("--root", type=Path, help="Stage-5 neural_circuits directory to inspect cache-only.")
    ap.add_argument("--status-json", type=Path, default=None)
    args = ap.parse_args()

    source = args.manifest if args.manifest is not None else args.root
    resolved = source.expanduser().resolve()
    try:
        if args.manifest is not None:
            summary = summarize_manifest(resolved)
        else:
            summary = summarize_root(resolved)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        summary = {
            "schema": "pipeline-circuit-availability-v2",
            "inspection_mode": "manifest" if args.manifest is not None else "cache_scan",
            "status": "inspection_error",
            "n_usable_circuits": 0,
            "error": str(exc),
            "stage5_manifest": str(resolved) if args.manifest is not None else None,
            "stage5_root": str(resolved) if args.root is not None else None,
        }
        if args.status_json is not None:
            out = args.status_json.expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[circuit-availability] ERROR: {exc}", file=sys.stderr)
        return 2

    if args.status_json is not None:
        out = args.status_json.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[circuit-availability] usable={summary['n_usable_circuits']} "
        f"explicit_full={summary.get('n_explicit_full_network_candidates', 0)} "
        f"mode={summary['inspection_mode']} status={summary['status']}"
    )
    if summary.get("n_explicit_full_network_candidates", 0) > 0:
        print(
            "[circuit-availability] ERROR: explicit full-network ablation artifacts are present "
            "where a discovered Stage-5 circuit is required.",
            file=sys.stderr,
        )
        return MODE_MISMATCH_EXIT
    return 0 if summary["n_usable_circuits"] > 0 else NO_USABLE_CIRCUIT_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
