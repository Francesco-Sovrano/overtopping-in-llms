#!/usr/bin/env python3
"""Backfill RQ3 threshold/spiking diagnostics using the exact pipeline RunSpec.

The rebuild path intentionally mirrors ``run_overtopping_experiments.sh`` /
``run_pipeline.sh`` instead of maintaining independent defaults.  In
particular, the model batch size is taken from each matched experiment RunSpec
unless the caller explicitly supplies ``--batch-size``.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from core.project_paths import PROJECT_ROOT
from studies.overtopping.analysis import primary_holdout_analysis as helpers
from studies.overtopping.experiments.execution import RunSpec
from studies.overtopping.experiments.run_experiments import paper_study_experiments, paper_auxiliary_experiments


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else int(default)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary-table", required=True)
    p.add_argument(
        "--population-scope", choices=["primary", "primary+supplementary"], default="primary",
        help="primary uses the complete configured study table; primary+supplementary is an explicit extended scope.",
    )
    p.add_argument("--data-root", required=True)
    p.add_argument("--catalogue-json", default=None, help="configured_experiments.json written by run_overtopping_experiments.sh")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--evaluation-split", default="test", choices=["test", "train", "all"])
    p.add_argument("--baseline-subsets", default=os.environ.get("ANALYZE_BASELINE_SUBSETS", "positive,negative"))
    p.add_argument("--target", default=os.environ.get("THRESHOLD_EVENT_TARGET", "all"), help="flip_any, flip_c2i, flip_i2c, comma-list, or all")
    p.add_argument("--spiking-max-points", type=int, default=_env_int("THRESHOLD_EVENT_MAX_POINTS", 10000))
    p.add_argument("--spiking-min-points", type=int, default=_env_int("THRESHOLD_EVENT_MIN_POINTS", 512))
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Explicit override only. Default: exact per-run RunSpec.batch_size from the overtopping experiment catalogue.",
    )
    p.add_argument("--points-to-use-for-mean-ablation", type=int, default=_env_int("POINTS_TO_USE_FOR_MEAN_ABLATION", 256))
    p.add_argument("--threshold-event-min-examples", type=int, default=8)
    p.add_argument("--threshold-event-repeats", type=int, default=_env_int("THRESHOLD_EVENT_REPEATS", 20))
    p.add_argument("--threshold-event-holdout-fraction", type=float, default=float(os.environ.get("THRESHOLD_EVENT_HOLDOUT_FRACTION", "0.5")))
    p.add_argument("--threshold-event-n-bins", type=int, default=_env_int("THRESHOLD_EVENT_N_BINS", 10))
    p.add_argument("--seed", type=int, default=_env_int("THRESHOLD_EVENT_SEED", 42))
    p.add_argument("--ai-model-cache-dir", default=os.environ.get("HF_MODEL_CACHE_DIR") or None)


    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _spiking_out(spec: RunSpec, data_root: Path, split: str, spiking_max_points: int) -> Path:
    label = f"spiking_diagnostics-{spec.bag_label()}"
    if split == "train":
        label += "-eval_train"
    elif split == "all":
        label += "-eval_all"
    if int(spiking_max_points) != 10000:
        label += f"-cap{int(spiking_max_points)}"
    return spec.input_data_dir(data_root) / label


def _load_specs(catalogue_json: str | None, evaluation_split: str, population_scope: str) -> list[RunSpec]:
    specs: list[RunSpec] = []
    if catalogue_json:
        path = Path(catalogue_json).expanduser().resolve()
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            allowed = {f.name for f in fields(RunSpec)}
            for raw in payload:
                values = {k: v for k, v in dict(raw).items() if k in allowed}
                spec = RunSpec(**values)
                specs.append(replace(spec, evaluation_split=evaluation_split))
    # The configured-study registry is authoritative for manuscript coverage.
    # Merge it even when a catalogue JSON is supplied so a catalogue generated
    # from a filtered execution command cannot silently shrink the backfill
    # population. Catalogue entries still win when they provide an exact match.
    builtins = list(paper_study_experiments())
    if population_scope == "primary+supplementary":
        builtins.extend(paper_auxiliary_experiments())
    specs.extend(replace(spec, evaluation_split=evaluation_split) for spec in builtins)

    deduped: dict[tuple, RunSpec] = {}
    for spec in specs:
        key = (
            spec.task, spec.model, spec.intervention, spec.mode, spec.z_thresh,
            spec.batch_size, spec.circuit_level, spec.circuit_size,
            spec.min_flip_rate, spec.max_circuits, spec.mlp_neurons_only,
            spec.no_llm_feature_generation, spec.evaluation_split,
        )
        deduped.setdefault(key, spec)
    return list(deduped.values())


def _match_spec(setting: dict, specs: list[RunSpec], data_root: Path) -> RunSpec:
    reported = Path(setting["reported_stats"]).resolve()
    matches = [spec for spec in specs if spec.stats_dir(data_root).resolve() == reported]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"{setting['label']}: multiple catalogue RunSpecs map to {reported}")

    # Fallback for a primary table remapped from another absolute project root:
    # match the exact relative stats path under data_root.
    try:
        rel = reported.relative_to(data_root.resolve())
    except ValueError:
        rel = None
    if rel is not None:
        matches = []
        for spec in specs:
            try:
                if spec.stats_dir(data_root).resolve().relative_to(data_root.resolve()) == rel:
                    matches.append(spec)
            except ValueError:
                pass
        if len(matches) == 1:
            return matches[0]

    raise RuntimeError(
        f"{setting['label']}: no experiment-catalogue RunSpec matches Stage-7 stats {reported}. "
        "Re-run ./run_overtopping_experiments.sh --list / regenerate configured_experiments.json; "
        "the rebuild refuses to guess intervention or batch size."
    )


def _bucket_keep_keys(buckets: dict) -> set[str]:
    nc = buckets.get("non_catastrophic_agonists", {}) if isinstance(buckets, dict) else {}
    keep = set(nc.keys()) if isinstance(nc, dict) else set()
    cz = buckets.get("catastrophic_zero", {}) if isinstance(buckets, dict) else {}
    if isinstance(cz, dict):
        bad = set()
        for name in ("confirmed", "not_always"):
            blk = cz.get(name, {})
            if isinstance(blk, dict):
                bad |= set(blk.keys())
        keep -= bad
    return keep


def _backfill_candidate_direction_provenance(
    spec: RunSpec,
    data_root: Path,
    ranking_path: Path,
    flip_path: Path,
    *,
    write: bool = True,
) -> dict:
    """Recover discovery-direction membership from Stage-6 buckets.

    Existing complete provenance is reused without rewriting either CSV. When
    provenance must be reconstructed, ``write=False`` performs the full
    validation and reports whether files would change without modifying data.
    """
    ranking = pd.read_csv(ranking_path)
    if "unit_key" not in ranking.columns:
        if not {"layer_label", "neuron_id"}.issubset(ranking.columns):
            raise ValueError(f"Frozen ranking lacks unit identity: {ranking_path}")
        ranking["unit_key"] = [
            f"{layer}:{int(nid)}"
            for layer, nid in zip(ranking["layer_label"], ranking["neuron_id"])
        ]

    required_ranking = {
        "unit_key",
        "discovery_baseline_subset",
        "discovery_baseline_subsets",
        "n_discovery_baseline_subsets",
    }
    existing_complete = required_ranking.issubset(ranking.columns)
    if existing_complete:
        subsets = ranking["discovery_baseline_subsets"].fillna("").astype(str)
        strongest = ranking["discovery_baseline_subset"].fillna("").astype(str).str.lower()
        counts = pd.to_numeric(ranking["n_discovery_baseline_subsets"], errors="coerce")
        existing_complete = bool(
            subsets.str.len().gt(0).all()
            and strongest.isin({"positive", "negative"}).all()
            and counts.ge(1).all()
        )

    flips = pd.read_csv(flip_path)
    if "unit_key" not in flips.columns:
        layer_col = "layer_label" if "layer_label" in flips.columns else "layer_key"
        flips["unit_key"] = [
            f"{layer}:{int(nid)}"
            for layer, nid in zip(flips[layer_col], flips["neuron_id"])
        ]
    required_flip = {
        "discovery_baseline_subset",
        "discovery_baseline_subsets",
        "n_discovery_baseline_subsets",
    }
    flips_complete = required_flip.issubset(flips.columns)
    if flips_complete:
        flips_complete = bool(
            flips["discovery_baseline_subsets"].fillna("").astype(str).str.len().gt(0).all()
            and flips["discovery_baseline_subset"].fillna("").astype(str).str.lower().isin({"positive", "negative"}).all()
            and pd.to_numeric(flips["n_discovery_baseline_subsets"], errors="coerce").ge(1).all()
        )

    if existing_complete and flips_complete:
        subsets = ranking["discovery_baseline_subsets"].astype(str)
        counts = pd.to_numeric(ranking["n_discovery_baseline_subsets"], errors="coerce")
        return {
            "bag_dir": None,
            "bucket_files": 0,
            "n_candidates": int(len(ranking)),
            "n_positive": int(subsets.str.contains("positive").sum()),
            "n_negative": int(subsets.str.contains("negative").sum()),
            "n_both": int(counts.gt(1).sum()),
            "updated": False,
            "would_update": False,
        }

    candidate_keys = set(ranking["unit_key"].astype(str))
    memberships = {key: set() for key in candidate_keys}
    strongest_by_key = {}
    bag_dir = spec.input_data_dir(data_root).parent / spec.bag_label()
    if not bag_dir.is_dir():
        raise FileNotFoundError(f"Stage-6 agonist bag not found for direction provenance: {bag_dir}")
    n_files = 0
    for fp in sorted(bag_dir.rglob("neuron_buckets.json")):
        n_files += 1
        try:
            buckets = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        nca = buckets.get("non_catastrophic_agonists", {}) if isinstance(buckets, dict) else {}
        if not isinstance(nca, dict):
            continue
        for key in _bucket_keep_keys(buckets):
            entry = nca.get(key, {})
            if not isinstance(entry, dict):
                continue
            rec = entry.get("last_record") or {}
            try:
                score = float(rec.get("max_effect"))
            except Exception:
                continue
            if abs(score) < float(spec.min_flip_rate):
                continue
            if "layer_label" in rec and "neuron_id" in rec:
                unit_key = f"{rec['layer_label']}:{int(rec['neuron_id'])}"
            else:
                try:
                    layer, nid = str(key).rsplit(":", 1)
                    unit_key = f"{layer}:{int(nid)}"
                except Exception:
                    continue
            if unit_key not in candidate_keys:
                continue
            baseline = str(rec.get("baseline_subset", "")).strip().lower()
            if baseline not in {"positive", "negative"}:
                continue
            memberships[unit_key].add(baseline)
            prev = strongest_by_key.get(unit_key)
            if prev is None or abs(score) > abs(prev[0]):
                strongest_by_key[unit_key] = (score, baseline)

    missing = [key for key, dirs in memberships.items() if not dirs]
    if missing:
        raise RuntimeError(
            f"Could not recover discovery direction for {len(missing)} frozen candidate(s) from {bag_dir}: "
            + ", ".join(missing[:12])
        )

    ranking["discovery_baseline_subsets"] = [
        "|".join(sorted(memberships[str(key)])) for key in ranking["unit_key"].astype(str)
    ]
    ranking["n_discovery_baseline_subsets"] = [
        len(memberships[str(key)]) for key in ranking["unit_key"].astype(str)
    ]
    if "discovery_baseline_subset" not in ranking.columns:
        ranking["discovery_baseline_subset"] = [
            strongest_by_key[str(key)][1] for key in ranking["unit_key"].astype(str)
        ]
    else:
        repaired = []
        for key, current in zip(
            ranking["unit_key"].astype(str), ranking["discovery_baseline_subset"]
        ):
            cur = str(current).strip().lower()
            repaired.append(cur if cur in memberships[key] else strongest_by_key[key][1])
        ranking["discovery_baseline_subset"] = repaired

    prov = ranking[
        [
            c
            for c in [
                "unit_key",
                "discovery_baseline_subset",
                "discovery_baseline_subsets",
                "n_discovery_baseline_subsets",
                "discovery_score",
                "discovery_score_signed",
            ]
            if c in ranking.columns
        ]
    ].drop_duplicates("unit_key")
    flips = flips.drop(
        columns=[c for c in prov.columns if c != "unit_key" and c in flips.columns],
        errors="ignore",
    ).merge(prov, on="unit_key", how="left", validate="one_to_one")

    if write:
        ranking.to_csv(ranking_path, index=False)
        flips.to_csv(flip_path, index=False)

    subsets = ranking["discovery_baseline_subsets"].astype(str)
    counts = pd.to_numeric(ranking["n_discovery_baseline_subsets"], errors="coerce")
    return {
        "bag_dir": str(bag_dir),
        "bucket_files": int(n_files),
        "n_candidates": int(len(ranking)),
        "n_positive": int(subsets.str.contains("positive").sum()),
        "n_negative": int(subsets.str.contains("negative").sum()),
        "n_both": int(counts.gt(1).sum()),
        "updated": bool(write),
        "would_update": True,
    }


def main() -> None:
    args = parse_args()
    table_path = Path(args.primary_table).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    table = pd.read_csv(table_path)
    specs = _load_specs(args.catalogue_json, args.evaluation_split, args.population_scope)

    settings: list[tuple[str, dict]] = []
    primary_stats: set[str] = set()
    for index in range(len(table)):
        setting = helpers.setting_from_row(
            index, table.iloc[index], data_root, PROJECT_ROOT / "results",
            evaluation_split=args.evaluation_split, sampling_max_points=10000,
        )
        settings.append(("primary", setting))
        primary_stats.add(str(Path(setting["reported_stats"]).resolve()))
    if args.population_scope == "primary+supplementary":
        for j, raw_spec in enumerate(paper_auxiliary_experiments()):
            spec = replace(raw_spec, evaluation_split=args.evaluation_split)
            if str(spec.stats_dir(data_root).resolve()) in primary_stats:
                continue
            pseudo = pd.Series({
                "task": spec.task, "model": Path(spec.model).name, "phase": spec.phase,
                "stats_dir": str(spec.stats_dir(data_root)),
            })
            setting = helpers.setting_from_row(
                len(table)+j, pseudo, data_root, PROJECT_ROOT / "results",
                evaluation_split=args.evaluation_split, sampling_max_points=10000,
            )
            settings.append(("supplementary", setting))

    n_planned = 0
    n_skipped_supplementary = 0
    for source_scope, setting in settings:
        spec = _match_spec(setting, specs, data_root)

        stats = spec.stats_dir(data_root)
        candidate_stats = stats / "flip_stats_by_neuron.csv"
        candidate_ranking = stats / "frozen_candidate_ranking.csv"
        materialized_scores = stats / "scores.csv"
        if not candidate_stats.is_file() or not candidate_ranking.is_file() or not materialized_scores.is_file():
            if source_scope == "supplementary":
                print(f"[spiking backfill] skip supplementary without strict Stage-7 materialization: {setting['label']}")
                print(f"  required: {candidate_stats} / {candidate_ranking} / {materialized_scores}")
                n_skipped_supplementary += 1
                continue
            raise FileNotFoundError(
                f"{setting['label']}: missing exact pipeline Stage-7 materialization: "
                f"{candidate_stats} / {candidate_ranking} / {materialized_scores}"
            )

        provenance = _backfill_candidate_direction_provenance(
            spec, data_root, candidate_ranking, candidate_stats, write=not args.dry_run
        )
        print(
            f"[spiking provenance] {setting['label']}: candidates={provenance['n_candidates']} "
            f"positive={provenance['n_positive']} negative={provenance['n_negative']} both={provenance['n_both']} "
            f"updated={provenance['updated']} would_update={provenance['would_update']}"
        )

        out_dir = _spiking_out(spec, data_root, args.evaluation_split, args.spiking_max_points)
        batch_size = int(args.batch_size) if args.batch_size is not None else int(spec.batch_size)
        points_to_use = int(args.points_to_use_for_mean_ablation)
        if "donor" in spec.intervention and points_to_use < 2048:
            points_to_use = 2048
        cmd = [
            str(args.python_bin), "-m", "studies.overtopping.analysis.threshold_event_diagnostics",
            "--input_data_dir", str(spec.input_data_dir(data_root)),
            "--out_dir", str(out_dir),
            "--baseline_subsets", str(args.baseline_subsets),
            "--task_module", str(setting["task_module"]),
            "--ai_model", str(spec.model),
            "--candidate_flip_stats_path", str(candidate_stats),
            "--candidate_ranking_path", str(candidate_ranking),
            "--materialized_stage7_scores_path", str(materialized_scores),
            "--evaluation_split", str(args.evaluation_split),
            "--intervention", str(spec.intervention),
            "--points_to_use_for_mean_ablation", str(points_to_use),
            "--batch_size", str(batch_size),
            "--spiking_max_points", str(args.spiking_max_points),
            "--spiking_min_points", str(args.spiking_min_points),
            "--threshold_event_min_examples", str(args.threshold_event_min_examples),
            "--threshold_event_repeats", str(args.threshold_event_repeats),
            "--threshold_event_holdout_fraction", str(args.threshold_event_holdout_fraction),
            "--threshold_event_n_bins", str(args.threshold_event_n_bins),
            "--target", str(args.target),
            "--seed", str(args.seed),
        ]
        if spec.decode_only:
            cmd.append("--decode_only")
        if args.ai_model_cache_dir:
            cmd.extend(["--ai_model_cache_dir", str(args.ai_model_cache_dir)])
        if args.force:
            cmd.extend(["--force_threshold_event", "--force_spiking_eval", "--no_skip_existing"])

        print(f"[spiking backfill] {source_scope}: {setting['label']}")
        print(f"  RunSpec:     intervention={spec.intervention} decode_only={spec.decode_only} batch_size={batch_size}")
        print(f"  candidates:  {candidate_stats}")
        print(f"  output:      {out_dir}")
        print("  command:     " + " ".join(cmd), flush=True)
        n_planned += 1
        if not args.dry_run:
            subprocess.run(cmd, cwd=PROJECT_ROOT / "code", check=True)

    verb = "Validated/would rebuild" if args.dry_run else "Rebuilt"
    print(f"{verb} threshold/spiking diagnostics for {n_planned} exact RQ3 runs with pipeline-matched RunSpecs; skipped_supplementary={n_skipped_supplementary}.")


if __name__ == "__main__":
    main()
