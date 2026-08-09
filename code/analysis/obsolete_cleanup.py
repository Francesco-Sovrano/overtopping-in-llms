"""Safe cleanup of obsolete GCCR-era artifacts.

The current interaction analysis uses paired conditional marginal contributions.
This module removes only artifacts that belong exclusively to the obsolete GCCR
analysis while preserving reusable simultaneous E(J), matched-null summaries,
matched-set memberships, singleton statistics, and current conditional caches.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import pandas as pd


@dataclass(frozen=True)
class CleanupAction:
    kind: str  # delete-file, delete-tree, rewrite-json, rewrite-csv
    path: Path
    description: str


def _json_without_gccr(path: Path) -> tuple[dict | None, bool]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, False
    changed = False
    if "GCCR_m" in payload:
        payload.pop("GCCR_m", None)
        changed = True
    if "interaction_validation_definition_version" in payload and path.name == "flip_stats_global.json":
        # This field described the GCCR-era interaction schema, not E(J) itself.
        payload.pop("interaction_validation_definition_version", None)
        changed = True
    notes = payload.get("notes")
    if isinstance(notes, list):
        filtered = [note for note in notes if "GCCR" not in str(note)]
        if filtered != notes:
            payload["notes"] = filtered
            changed = True
    return payload, changed


def _csv_without_gccr(path: Path) -> tuple[pd.DataFrame | None, bool]:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None, False
    if "metric" not in frame.columns:
        return frame, False
    keep = ~frame["metric"].astype(str).str.startswith("GCCR")
    if int(keep.sum()) == len(frame):
        return frame, False
    return frame.loc[keep].copy(), True


def collect_obsolete_gccr_actions(data_root: Path, results_root: Path) -> list[CleanupAction]:
    """Return a deterministic cleanup plan without changing the filesystem."""
    actions: list[CleanupAction] = []
    data_root = Path(data_root).expanduser().resolve()
    results_root = Path(results_root).expanduser().resolve()

    for interaction_dir in sorted(data_root.rglob("interaction_validation")):
        if not interaction_dir.is_dir():
            continue
        summary_json = interaction_dir / "interaction_validation_summary.json"
        legacy_v3 = False
        if summary_json.exists():
            try:
                payload = json.loads(summary_json.read_text(encoding="utf-8"))
                legacy_v3 = payload.get("definition_version") == "interaction-validation-v3"
            except Exception:
                legacy_v3 = False

        for name in [
            "candidate_support_layer_effects.json",
            "layer_interaction_effects.csv",
            "gccr_metrics.csv",
            "layer_populations.csv",
        ]:
            path = interaction_dir / name
            if path.exists():
                actions.append(CleanupAction("delete-file", path, "obsolete GCCR-only artifact"))
        for path in sorted(interaction_dir.glob("GCCR_*_matched_null.*")):
            actions.append(CleanupAction("delete-file", path, "obsolete GCCR matched-null artifact"))

        if legacy_v3:
            cache_dir = interaction_dir / "group_eval_cache"
            conditional_marker = interaction_dir / "conditional_background_membership.csv"
            # A partially migrated conditional run writes its background-membership
            # file before model evaluation. In that state the summary may still be
            # v3 while group_eval_cache already contains valuable new conditional
            # batches. Never delete that cache from the standalone cleanup.
            if cache_dir.exists() and not conditional_marker.exists():
                actions.append(CleanupAction(
                    "delete-tree", cache_dir,
                    "obsolete v3 group cache; E(J) summaries/memberships are preserved separately",
                ))

        if summary_json.exists():
            _, changed = _json_without_gccr(summary_json)
            if changed:
                actions.append(CleanupAction("rewrite-json", summary_json, "remove GCCR-only fields"))

        for csv_name in ["interaction_validation_summary.csv", "matched_null_draws.csv"]:
            path = interaction_dir / csv_name
            if path.exists():
                _, changed = _csv_without_gccr(path)
                if changed:
                    actions.append(CleanupAction("rewrite-csv", path, "remove GCCR-only rows"))

        global_path = interaction_dir.parent / "flip_stats_global.json"
        if global_path.exists():
            _, changed = _json_without_gccr(global_path)
            if changed:
                actions.append(CleanupAction("rewrite-json", global_path, "remove GCCR-only fields"))

    for root in [results_root / "manuscript", results_root / "catalogue"]:
        if root.exists():
            for path in sorted(root.glob("*GCCR*")):
                if path.is_file():
                    actions.append(CleanupAction("delete-file", path, "obsolete GCCR result"))

    # Deduplicate paths while preserving deterministic ordering.
    unique: dict[tuple[str, str], CleanupAction] = {}
    for action in actions:
        unique[(action.kind, str(action.path))] = action
    return sorted(unique.values(), key=lambda a: (str(a.path), a.kind))


def apply_obsolete_gccr_cleanup(actions: list[CleanupAction]) -> dict[str, int]:
    """Apply a plan produced by :func:`collect_obsolete_gccr_actions`."""
    removed_files = 0
    removed_trees = 0
    rewritten_files = 0
    for action in actions:
        path = action.path
        if action.kind == "delete-file":
            if path.exists() and path.is_file():
                path.unlink()
                removed_files += 1
        elif action.kind == "delete-tree":
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
                removed_trees += 1
        elif action.kind == "rewrite-json":
            if not path.exists():
                continue
            payload, changed = _json_without_gccr(path)
            if changed and payload is not None:
                path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
                rewritten_files += 1
        elif action.kind == "rewrite-csv":
            if not path.exists():
                continue
            frame, changed = _csv_without_gccr(path)
            if changed and frame is not None:
                frame.to_csv(path, index=False)
                rewritten_files += 1
        else:
            raise ValueError(f"Unknown cleanup action: {action.kind}")
    return {
        "removed_files": removed_files,
        "removed_trees": removed_trees,
        "rewritten_files": rewritten_files,
    }


def cleanup_obsolete_gccr(data_root: Path, results_root: Path) -> dict[str, int]:
    """Compatibility helper used by final-results generation."""
    actions = collect_obsolete_gccr_actions(data_root, results_root)
    return apply_obsolete_gccr_cleanup(actions)
