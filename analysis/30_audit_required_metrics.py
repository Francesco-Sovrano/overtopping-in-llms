#!/usr/bin/env python3
"""Audit whether every requested manuscript metric is exact, backfillable, or unavailable.

The audit never substitutes singleton unions for simultaneous interventions and
never treats held-out-selected singleton rankings as discovery-frozen H_m.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis.primary_matrix import PRIMARY_PROFILE_CHOICES, normalize_primary_table

SINGLETON_SCHEMA = "heldout-set-metrics-v2"
INTERACTION_SCHEMA = "interaction-validation-v2"


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_stats_dir(raw: object, data_root: Path) -> Path:
    path = Path(str(raw)).expanduser()
    candidates = [path, data_root / path] if not path.is_absolute() else [path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def model_root_from_stats(stats_dir: Path) -> Path | None:
    parts = list(stats_dir.parts)
    try:
        index = parts.index("rule_extraction_results")
    except ValueError:
        return None
    return Path(*parts[:index])


def stage5_and_stage6_paths(stats_dir: Path) -> tuple[Path | None, Path | None]:
    model_root = model_root_from_stats(stats_dir)
    if model_root is None:
        return None, None
    label = stats_dir.name
    for suffix in ("-heldout_test", "-eval_train"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    marker = "-agonist_neurons"
    if marker not in label:
        return None, None
    circuit_label, rest = label.split(marker, 1)
    bag_label = "agonist_neurons" + rest
    discovery = model_root / "neural_circuit_discovery_results" / "eap_ig_inputs" / circuit_label
    return discovery / "neural_circuits", discovery / bag_label


def has_flip_columns(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        columns = pd.read_csv(path, nrows=0).columns
    except Exception:
        return False
    return any(str(column).startswith("flip_") for column in columns)


def audit_row(row: pd.Series, data_root: Path) -> dict:
    stats_dir = resolve_stats_dir(row["stats_dir"], data_root)
    global_path = stats_dir / "flip_stats_global.json"
    by_path = stats_dir / "flip_stats_by_neuron.csv"
    scores_path = stats_dir / "scores.csv"
    singleton_path = stats_dir / "singleton_set_metrics.json"
    ranking_path = stats_dir / "frozen_candidate_ranking.csv"
    interaction_dir = stats_dir / "interaction_validation"
    interaction_path = interaction_dir / "interaction_validation_summary.json"
    null_path = interaction_dir / "interaction_validation_summary.csv"
    global_payload = load_json(global_path)
    singleton = load_json(singleton_path)
    interaction = load_json(interaction_path)
    stage5_dir, stage6_dir = stage5_and_stage6_paths(stats_dir)

    aggregate_exact = bool(global_path.exists() and by_path.exists())
    singleton_exact = singleton.get("definition_version") == SINGLETON_SCHEMA
    ranking_exact = bool(singleton_exact and ranking_path.exists())
    interaction_exact = interaction.get("definition_version") == INTERACTION_SCHEMA
    null_exact = bool(interaction_exact and null_path.exists())

    materialized_events = has_flip_columns(scores_path)
    discovery_ranking_source = bool(
        stage6_dir is not None and stage6_dir.exists() and any(stage6_dir.rglob("neuron_buckets.json"))
    )
    singleton_backfill = bool(materialized_events and discovery_ranking_source)
    stage5_runtime = bool(
        stage5_dir is not None
        and (stage5_dir / "manifest.json").exists()
        and (stage5_dir / "dataset_info.json").exists()
    )
    interaction_backfill = bool(
        materialized_events
        and stage5_runtime
        and (ranking_path.exists() or singleton_backfill)
    )

    required = {
        "J": singleton_exact or aggregate_exact,
        "U_J": singleton_exact or aggregate_exact,
        "s_1": singleton_exact or aggregate_exact,
        "N_t": singleton_exact or aggregate_exact,
        "R_ov": singleton_exact or aggregate_exact,
        "N_eff": singleton_exact or aggregate_exact,
        "TOC_m": singleton_exact and ranking_exact,
        "OCC_0": singleton_exact,
        "OCC_1": singleton_exact,
        "E_J": interaction_exact,
        "GCCR_m": interaction_exact,
        "matched_null_E_J": null_exact,
        "matched_null_GCCR_m": null_exact,
    }
    all_exact = all(required.values())
    missing = [name for name, ok in required.items() if not ok]

    return {
        "task": row.get("task"),
        "model": row.get("model"),
        "phase": row.get("phase"),
        "stats_dir": str(stats_dir),
        "singleton_schema": singleton.get("definition_version", "missing"),
        "interaction_schema": interaction.get("definition_version", "missing"),
        "legacy_aggregate_exact_available": aggregate_exact,
        "materialized_singleton_events_available": materialized_events,
        "discovery_ranking_source_available": discovery_ranking_source,
        "stage5_interaction_runtime_available": stage5_runtime,
        "singleton_backfill_without_model_ablations": singleton_backfill,
        "interaction_backfill_requires_model": interaction_backfill,
        "all_required_metrics_exact": all_exact,
        "missing_required_metrics": ";".join(missing),
        **{f"exact_{name}": bool(ok) for name, ok in required.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary-table", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--primary-profile", required=True, choices=PRIMARY_PROFILE_CHOICES)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--require-complete", action="store_true")
    args = p.parse_args()

    source = Path(args.primary_table).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    table, excluded, profile_audit = normalize_primary_table(
        pd.read_csv(source), profile=args.primary_profile, source=source
    )
    rows = [audit_row(row, data_root) for _, row in table.iterrows()]
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "required_metrics_audit.csv", index=False)
    payload = {
        "primary_profile": args.primary_profile,
        "profile_audit": profile_audit,
        "setting_count": len(rows),
        "complete_setting_count": int(frame["all_required_metrics_exact"].sum()) if len(frame) else 0,
        "incomplete_setting_count": int((~frame["all_required_metrics_exact"]).sum()) if len(frame) else 0,
        "rows": rows,
        "interpretation": {
            "legacy_aggregate_exact_available": "J, U(J), s_(1), N_t, R_ov and N_eff can be recovered exactly from flip_stats_global.json + flip_stats_by_neuron.csv.",
            "singleton_backfill_without_model_ablations": "TOC_m and OCC_b can be regenerated from materialized singleton flip events plus the stage-6 discovery ranking, without repeating singleton model ablations.",
            "interaction_backfill_requires_model": "E(J), GCCR_m and matched nulls require genuine simultaneous interventions and model access; singleton unions are never substituted.",
        },
    }
    (out_dir / "required_metrics_audit.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True, default=str), encoding="utf-8"
    )
    print(
        f"[required-metrics] complete={payload['complete_setting_count']}/{payload['setting_count']} "
        f"audit={out_dir / 'required_metrics_audit.csv'}"
    )
    if args.require_complete and payload["incomplete_setting_count"]:
        raise SystemExit(
            "Required manuscript metrics are incomplete. Inspect required_metrics_audit.csv. "
            "Use the experiment pipeline on a full runtime cache to backfill exact singleton metrics "
            "and run simultaneous interactions; compact aggregate-only exports cannot reconstruct them."
        )


if __name__ == "__main__":
    main()
