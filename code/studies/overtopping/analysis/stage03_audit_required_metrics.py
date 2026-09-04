#!/usr/bin/env python3
"""Audit whether every requested manuscript metric is exact, backfillable, or unavailable.

The audit never substitutes singleton unions for simultaneous interventions and
never treats held-out-selected singleton rankings as discovery-frozen H_m.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from studies.overtopping.analysis.lib.files import load_json

import pandas as pd
from studies.overtopping.analysis.primary_holdout_analysis import reference_stats_dir

from studies.overtopping.analysis.lib.primary_matrix import PRIMARY_PROFILE_CHOICES, normalize_primary_table
from studies.overtopping.analysis.lib.interaction_schema import is_exact_interaction_schema

SINGLETON_SCHEMAS = {"heldout-set-metrics-v3-directional"}


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
    label = reference_stats_dir(stats_dir).name
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


def audit_row(row: pd.Series, data_root: Path, *, require_cmc: bool = True) -> dict:
    stats_dir = resolve_stats_dir(row["stats_dir"], data_root)
    scores_path = stats_dir / "scores.csv"
    singleton_path = stats_dir / "singleton_set_metrics.json"
    ranking_path = stats_dir / "frozen_candidate_ranking.csv"
    interaction_dir = stats_dir / "interaction_validation"
    interaction_path = interaction_dir / "interaction_validation_summary.json"
    null_path = interaction_dir / "interaction_validation_summary.csv"
    singleton = load_json(singleton_path)
    interaction = load_json(interaction_path)
    stage5_dir, stage6_dir = stage5_and_stage6_paths(stats_dir)

    singleton_schema = singleton.get("definition_version")
    singleton_exact = singleton_schema in SINGLETON_SCHEMAS
    directional_required_keys = {
        "U_J_i2c", "U_J_c2i", "N_t_i2c", "N_t_c2i",
        "N_eff_i2c", "N_eff_c2i", "s_1_i2c", "s_1_c2i",
    }
    directional_singleton_exact = (
        singleton_schema == "heldout-set-metrics-v3-directional"
        and directional_required_keys.issubset(singleton)
    )
    ranking_exact = bool(singleton_exact and ranking_path.exists())
    interaction_schema = interaction.get("definition_version")
    interaction_exact = is_exact_interaction_schema(interaction_schema)
    e_j_exact = interaction_exact
    null_exact = bool(null_path.exists() and e_j_exact)
    conditional_exact = bool(interaction_exact and interaction.get("conditional_marginal"))

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

    exact = {
        "J": singleton_exact,
        "U_J": singleton_exact,
        "s_1": singleton_exact,
        "N_t": singleton_exact,
        "R_ov": singleton_exact,
        "N_eff": singleton_exact,
        "TOC_m": singleton_exact and ranking_exact,
        "U_J_i2c": directional_singleton_exact,
        "U_J_c2i": directional_singleton_exact,
        "N_t_i2c": directional_singleton_exact,
        "N_t_c2i": directional_singleton_exact,
        "N_eff_i2c": directional_singleton_exact,
        "N_eff_c2i": directional_singleton_exact,
        "E_J": e_j_exact,
        "conditional_marginal": conditional_exact,
        "matched_null_E_J": null_exact,
        "paired_conditional_null": conditional_exact,
    }
    required_names = [
        "J", "U_J", "s_1", "N_t", "R_ov", "N_eff", "TOC_m",
        "U_J_i2c", "U_J_c2i", "N_t_i2c", "N_t_c2i", "N_eff_i2c", "N_eff_c2i",
        "E_J", "matched_null_E_J",
    ]
    if require_cmc:
        required_names.extend(["conditional_marginal", "paired_conditional_null"])
    all_exact = all(exact[name] for name in required_names)
    missing = [name for name in required_names if not exact[name]]

    return {
        "task": row.get("task"),
        "model": row.get("model"),
        "phase": row.get("phase"),
        "stats_dir": str(stats_dir),
        "singleton_schema": singleton.get("definition_version", "missing"),
        "interaction_schema": interaction.get("definition_version", "missing"),
        "materialized_singleton_events_available": materialized_events,
        "discovery_ranking_source_available": discovery_ranking_source,
        "stage5_interaction_runtime_available": stage5_runtime,
        "singleton_backfill_without_model_ablations": singleton_backfill,
        "interaction_backfill_requires_model": interaction_backfill,
        "cmc_required": bool(require_cmc),
        "all_required_metrics_exact": all_exact,
        "missing_required_metrics": ";".join(missing),
        **{f"exact_{name}": bool(ok) for name, ok in exact.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary-table", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--primary-profile", required=True, choices=PRIMARY_PROFILE_CHOICES)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--require-complete", action="store_true")
    p.add_argument(
        "--skip-cmc-requirement",
        action="store_true",
        help="Do not require CMC or its paired conditional null for completeness; E(J) and its matched null remain required.",
    )
    args = p.parse_args()

    source = Path(args.primary_table).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    table, _, profile_audit = normalize_primary_table(
        pd.read_csv(source), profile=args.primary_profile, source=source
    )
    require_cmc = not bool(args.skip_cmc_requirement)
    rows = [audit_row(row, data_root, require_cmc=require_cmc) for _, row in table.iterrows()]
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "required_metrics_audit.csv", index=False)
    payload = {
        "primary_profile": args.primary_profile,
        "profile_audit": profile_audit,
        "setting_count": len(rows),
        "cmc_required": bool(require_cmc),
        "complete_setting_count": int(frame["all_required_metrics_exact"].sum()) if len(frame) else 0,
        "incomplete_setting_count": int((~frame["all_required_metrics_exact"]).sum()) if len(frame) else 0,
        "rows": rows,
        "interpretation": {
            "singleton_backfill_without_model_ablations": "TOC_m plus directional U_J, N_t, and N_eff can be regenerated from materialized singleton flip events plus the discovery ranking, without repeating singleton model ablations.",
            "interaction_backfill_requires_model": (
                "E(J) and its matched controls require genuine simultaneous interventions and model access; "
                + ("CMC and paired conditional controls are also required for this audit. " if require_cmc else "CMC is not required for this audit. ")
                + "Singleton unions are never substituted for simultaneous interventions."
            ),
        },
    }
    (out_dir / "required_metrics_audit.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True, default=str), encoding="utf-8"
    )
    print(
        f"[required-metrics] complete={payload['complete_setting_count']}/{payload['setting_count']} "
        f"audit={out_dir / 'required_metrics_audit.csv'}"
    )
    if payload["incomplete_setting_count"]:
        missing_counts: dict[str, int] = {}
        for value in frame.loc[~frame["all_required_metrics_exact"], "missing_required_metrics"].fillna(""):
            for name in str(value).split(";"):
                if name:
                    missing_counts[name] = missing_counts.get(name, 0) + 1
        if missing_counts:
            summary = ", ".join(f"{name}={count}/{len(frame)}" for name, count in sorted(missing_counts.items()))
            print(f"[required-metrics] missing: {summary}")
    if args.require_complete and payload["incomplete_setting_count"]:
        raise SystemExit(
            "Required manuscript metrics are incomplete. Inspect required_metrics_audit.csv. "
            "The line above reports which metric families are missing. Exact directional singleton "
            "metrics can be rebuilt from cached singleton events; simultaneous E(J)/CMC validation "
            "requires its corresponding interaction artifacts."
        )


if __name__ == "__main__":
    main()
