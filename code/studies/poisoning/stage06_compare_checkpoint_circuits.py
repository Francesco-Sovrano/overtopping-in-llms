#!/usr/bin/env python3
"""Compare the configured poisoning CHA candidate sets across checkpoints.

The candidate endpoint may be the attack-agnostic observed training mixture or
the fixed attack-cohort control-correctness view. Trigger/attack effects are
still evaluated later and never participate in Stage-06 set construction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.project_paths import PROJECT_ROOT
from studies.poisoning.lib.run_paths import (
    OBSERVED_TRAINING_MIXTURE_CORRECTNESS_DIRNAME,
    ATTACK_COHORT_CONTROL_CORRECTNESS_DIRNAME,
    causal_dir,
    checkpoint_progress_label,
    circuits_dir,
    metadata_path,
    phase_dirname,
    safe_component,
)
from studies.poisoning.lib.units import unit_key as _unit_key
from studies.poisoning.lib.virgin_agonists import read_agonist_coordinates, resolve_virgin_agonists_path
from studies.poisoning.tasks.registry import available_tasks, get_task_definition


def _load_set(ranking: Path) -> tuple[set[str], pd.DataFrame]:
    if not ranking.is_file():
        return set(), pd.DataFrame()
    df = pd.read_csv(ranking)
    if "layer_label" not in df.columns and "layer_key" in df.columns:
        df["layer_label"] = df["layer_key"].astype(str)
    if "neuron_id" not in df.columns and "neuron" in df.columns:
        parts = df["neuron"].astype(str).str.rsplit(":", n=1, expand=True)
        if parts.shape[1] == 2:
            df["neuron_id"] = pd.to_numeric(parts[1], errors="coerce")
    if "layer_label" not in df.columns or "neuron_id" not in df.columns:
        return set(), df
    df = df.dropna(subset=["layer_label", "neuron_id"]).copy()
    df["neuron_id"] = pd.to_numeric(df["neuron_id"], errors="coerce").astype(int)
    return {_unit_key(a, b) for a, b in zip(df["layer_label"], df["neuron_id"])}, df


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return float(len(a & b) / len(union)) if union else np.nan


def _overlap_record(a_name: str, a: set[str], b_name: str, b: set[str]) -> dict:
    inter = a & b
    union = a | b
    return {
        "set_a": a_name,
        "set_b": b_name,
        "n_a": len(a),
        "n_b": len(b),
        "n_intersection": len(inter),
        "n_union": len(union),
        "jaccard": _jaccard(a, b),
        "fraction_a_retained": float(len(inter) / len(a)) if a else np.nan,
        "fraction_b_explained": float(len(inter) / len(b)) if b else np.nan,
    }


def _parse_virgin(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {_unit_key(layer, neuron_id) for layer, neuron_id in read_agonist_coordinates(path)}


def _ranking_for_endpoint(endpoint: Path) -> tuple[Path | None, str]:
    """Prefer the no-heldout candidate freeze; accept one legacy ranking for reuse."""
    preferred = endpoint / "candidate_localization" / "frozen_candidate_ranking.csv"
    if preferred.is_file():
        return preferred, "stage6_candidate_localization"

    # Backward-compatible reuse: older observed-mixture runs wrote the same
    # Stage-6-derived candidate ranking while also doing an unnecessary held-out
    # singleton pass.  Reuse the ranking only; none of those singleton effects
    # participate in the streamlined analysis.
    legacy = sorted(endpoint.glob("rule_extraction_results/neuron_flip_rules/stats/*/frozen_candidate_ranking.csv"))
    if len(legacy) == 1:
        return legacy[0], "legacy_stage7_ranking_reused_without_effects"
    return None, "missing" if not legacy else "ambiguous_legacy_rankings"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--phase", choices=["input_output", "output_only"], required=True)
    ap.add_argument("--task", choices=available_tasks(), default=None)
    ap.add_argument("--eval_intervention", default="mean-donor")
    ap.add_argument(
        "--candidate_localization_endpoint",
        choices=[OBSERVED_TRAINING_MIXTURE_CORRECTNESS_DIRNAME, ATTACK_COHORT_CONTROL_CORRECTNESS_DIRNAME, "both"],
        default=OBSERVED_TRAINING_MIXTURE_CORRECTNESS_DIRNAME,
    )
    ap.add_argument(
        "--virgin_agonists", default=None,
        help="Optional unpoisoned-task positive_baseline directory or neuron_buckets.json for overlap reporting only.",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest_path = metadata_path(run_dir, "checkpoint_manifest_all.csv")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    summary_dir = circuits_dir(run_dir) / phase_dirname(args.phase)
    summary_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    sets: dict[tuple[str, float], set[str]] = {}
    selected_endpoints = (
        [OBSERVED_TRAINING_MIXTURE_CORRECTNESS_DIRNAME, ATTACK_COHORT_CONTROL_CORRECTNESS_DIRNAME]
        if args.candidate_localization_endpoint == "both"
        else [args.candidate_localization_endpoint]
    )
    for row in manifest.to_dict("records"):
        condition = str(row["condition"])
        fraction = float(row["fraction"])
        keys: set[str] = set()
        n_ranked_rows = 0
        source_records: list[str] = []
        paths: list[str] = []
        for endpoint_name in selected_endpoints:
            endpoint = (
                causal_dir(run_dir)
                / condition
                / checkpoint_progress_label(row)
                / phase_dirname(args.phase)
                / endpoint_name
                / f"eval_{safe_component(args.eval_intervention)}"
            )
            ranking_path, ranking_source = _ranking_for_endpoint(endpoint)
            if ranking_path is None:
                source_records.append(f"{endpoint_name}:missing")
                continue
            endpoint_keys, ranking = _load_set(ranking_path)
            keys.update(endpoint_keys)
            n_ranked_rows += int(len(ranking))
            source_records.append(f"{endpoint_name}:{ranking_source}")
            paths.append(str(ranking_path))
        if keys:
            sets[(condition, fraction)] = keys
        entries.append({
            "condition": condition,
            "fraction": fraction,
            "global_step": row.get("global_step"),
            "candidate_endpoint": args.candidate_localization_endpoint,
            "candidate_endpoints_in_union": ",".join(selected_endpoints),
            "candidate_localization_attack_agnostic": selected_endpoints == [OBSERVED_TRAINING_MIXTURE_CORRECTNESS_DIRNAME],
            "candidate_ranking_source": ";".join(source_records),
            "candidate_ranking_path": "|".join(paths) if paths else None,
            "circuit_defined": bool(keys),
            "n_discovered_channels": int(len(keys)) if keys else np.nan,
            "n_ranked_rows": int(n_ranked_rows) if paths else np.nan,
        })

    entries_df = pd.DataFrame(entries)
    if not entries_df.empty:
        entries_df = entries_df.sort_values(["condition", "fraction"])
    entries_df.to_csv(summary_dir / "checkpoint_circuit_sets.csv", index=False)

    pairwise: list[dict] = []
    for condition in sorted({c for c, _ in sets}):
        checkpoints = sorted((f, s) for (c, f), s in sets.items() if c == condition)
        for i, (fa, a) in enumerate(checkpoints):
            for fb, b in checkpoints[i + 1:]:
                rec = _overlap_record(f"{condition}:{fa:g}", a, f"{condition}:{fb:g}", b)
                rec.update({"comparison": "within_condition", "condition": condition, "fraction_a": fa, "fraction_b": fb})
                pairwise.append(rec)
    pd.DataFrame(pairwise).to_csv(summary_dir / "checkpoint_circuit_overlap_pairwise.csv", index=False)

    matched: list[dict] = []
    fractions = sorted({f for c, f in sets if c == "clean"} & {f for c, f in sets if c == "poisoned"})
    for frac in fractions:
        rec = _overlap_record(
            f"clean:{frac:g}", sets[("clean", frac)],
            f"poisoned:{frac:g}", sets[("poisoned", frac)],
        )
        rec.update({"comparison": "matched_clean_vs_poisoned", "fraction": frac})
        matched.append(rec)
    pd.DataFrame(matched).to_csv(summary_dir / "matched_clean_poisoned_circuit_overlap.csv", index=False)

    # This old artifact represented a now-removed comparison between two CHA
    # endpoints.  Remove it on rerun so stale files cannot be mistaken for a
    # current result.
    (summary_dir / "backdoor_trigger_vs_attack_cohort_control_correctness_overlap.csv").unlink(missing_ok=True)

    virgin_path = Path(args.virgin_agonists).expanduser().resolve() if args.virgin_agonists else None
    if virgin_path is None and args.task:
        try:
            cfg = json.loads(metadata_path(run_dir, "run_config.json").read_text(encoding="utf-8"))
            task_definition = get_task_definition(args.task)
            virgin_path = resolve_virgin_agonists_path(
                PROJECT_ROOT, task=args.task,
                task_data_dir=task_definition.task_data_dir,
                model_name=str(cfg.get("model_name", task_definition.default_model)),
                phase=args.phase, intervention="mean-donor", require_phase_match=True,
            )
        except Exception as exc:
            print(f"[virgin-overlap] optional source unavailable: {exc}")
            virgin_path = None
    if virgin_path is not None:
        virgin = _parse_virgin(virgin_path)
        rows = []
        for (condition, frac), current in sorted(sets.items()):
            rec = _overlap_record("virgin_task_agonists", virgin, f"{condition}:{frac:g}", current)
            rec.update({"condition": condition, "fraction": frac, "virgin_agonist_source": str(virgin_path)})
            rows.append(rec)
        pd.DataFrame(rows).to_csv(summary_dir / "virgin_agonist_overlap.csv", index=False)

    print(f"Wrote {args.candidate_localization_endpoint} checkpoint candidate comparisons under {summary_dir}")


if __name__ == "__main__":
    main()
