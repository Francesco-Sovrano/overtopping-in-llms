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

from studies.overtopping.analysis.lib.primary_matrix import PRIMARY_PROFILE_CHOICES, normalize_primary_table
from studies.overtopping.analysis.lib.interaction_schema import is_exact_interaction_schema
from studies.overtopping.analysis.lib.discovery_artifacts import (
    feature_report_status,
    stage5_and_stage6_paths,
    stage5_completion_status,
    stage6_candidate_count,
)
from studies.overtopping.analysis.lib.stats_resolution import has_stats_artifacts, resolve_available_stats_dir
from studies.overtopping.analysis.primary_holdout_analysis import remap_stats_dir

SINGLETON_SCHEMAS = {"heldout-set-metrics-v3-directional"}


def resolve_stats_dir(raw: object, data_root: Path) -> tuple[Path, str]:
    """Resolve exact or usable partial held-out stats without calling data missing."""
    expected = remap_stats_dir(raw, data_root)
    return resolve_available_stats_dir(expected)



def has_flip_columns(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        columns = pd.read_csv(path, nrows=0).columns
    except Exception:
        return False
    return any(str(column).startswith("flip_") for column in columns)




def audit_row(row: pd.Series, data_root: Path, *, require_cmc: bool = True) -> dict:
    stats_dir, resolved_stats_status = resolve_stats_dir(row["stats_dir"], data_root)
    row_stats_status = str(row.get("stats_resolution", "") or "").strip()
    stats_resolution = row_stats_status if row_stats_status and row_stats_status != "nan" else resolved_stats_status
    scores_path = stats_dir / "scores.csv"
    singleton_path = stats_dir / "singleton_set_metrics.json"
    ranking_path = stats_dir / "frozen_candidate_ranking.csv"
    interaction_dir = stats_dir / "interaction_validation"
    interaction_path = interaction_dir / "interaction_validation_summary.json"
    null_path = interaction_dir / "interaction_validation_summary.csv"
    composition_path = interaction_dir / "composition_decomposition_summary.json"
    singleton = load_json(singleton_path)
    interaction = load_json(interaction_path)
    composition = load_json(composition_path)
    stage5_dir, stage6_dir = stage5_and_stage6_paths(stats_dir)
    feature_status = feature_report_status(stats_dir)
    stage5_status = stage5_completion_status(stage5_dir)
    discovered_candidate_count, discovery_candidate_status = stage6_candidate_count(stage6_dir)
    structural_zero_candidates = discovery_candidate_status == "ok" and discovered_candidate_count == 0

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
    composition_exact = bool(
        composition.get("schema") == "singleton-joint-composition-decomposition-v1"
        and composition.get("definition", {}).get("identity")
        and (interaction_dir / "composition_decomposition_summary.csv").is_file()
        and (interaction_dir / "composition_example_decomposition.csv").is_file()
    )

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
        "composition_decomposition": composition_exact,
        "paired_conditional_null": conditional_exact,
    }
    if structural_zero_candidates:
        # These quantities are resolved by the completed empty discovery set.
        # U(empty)=0 and threshold counts over an empty set are 0. Shape/
        # concentration metrics such as s_1, R_ov, N_eff and TOC_m are not
        # numerically defined for an empty candidate set, but that is a
        # structural non-applicability rather than missing computation.
        for name in (
            "J", "U_J", "N_t", "U_J_i2c", "U_J_c2i",
            "N_t_i2c", "N_t_c2i", "s_1", "R_ov", "N_eff", "TOC_m",
            "N_eff_i2c", "N_eff_c2i",
        ):
            exact[name] = True
    rq1_required_names = [
        "J", "U_J", "s_1", "N_t", "R_ov", "N_eff", "TOC_m",
        "U_J_i2c", "U_J_c2i", "N_t_i2c", "N_t_c2i", "N_eff_i2c", "N_eff_c2i",
    ]
    try:
        j_value = float(row.get("J"))
        rq2_applicable = bool(pd.notna(j_value) and j_value > 0)
    except (TypeError, ValueError):
        rq2_applicable = False
    if structural_zero_candidates:
        rq2_applicable = False

    rq2_required_names = ["E_J", "matched_null_E_J", "composition_decomposition"]
    if require_cmc:
        rq2_required_names.extend(["conditional_marginal", "paired_conditional_null"])
    required_names = rq1_required_names + (rq2_required_names if rq2_applicable else [])
    metric_artifacts_complete = all(exact[name] for name in required_names)
    missing = [name for name in required_names if not exact[name]]
    # A capped held-out evaluation contains real stats and is reportable in
    # best-effort mode, but it is not the full configured manuscript scope.
    evaluation_scope_complete = bool(structural_zero_candidates or stats_resolution == "exact")
    all_exact = bool(metric_artifacts_complete and evaluation_scope_complete)
    completion_status = (
        "complete_zero_candidates"
        if structural_zero_candidates and metric_artifacts_complete
        else "complete_metrics"
        if all_exact
        else "incomplete"
    )
    data_availability_status = (
        "partial_metrics"
        if metric_artifacts_complete and has_stats_artifacts(stats_dir)
        else "feature_stats_only"
        if feature_status in {"feature_report_complete", "feature_report_partial"} and not has_stats_artifacts(stats_dir)
        else "partial_overtopping_stats"
        if has_stats_artifacts(stats_dir)
        else "no_overtopping_stats"
    )
    if all_exact:
        pipeline_progress = completion_status
    elif data_availability_status == "partial_metrics":
        pipeline_progress = "partial_heldout_stats_available"
    elif discovery_candidate_status == "ok":
        pipeline_progress = "stage6_complete_downstream_metrics_missing"
    elif stage5_status == "stage5_complete":
        pipeline_progress = "stage5_complete_stage6_missing"
    elif stage5_status in {"stage5_started_unfinished", "stage5_completion_unverified"}:
        pipeline_progress = stage5_status
    else:
        pipeline_progress = feature_status

    return {
        "task": row.get("task"),
        "model": row.get("model"),
        "phase": row.get("phase"),
        "stats_dir": str(stats_dir),
        "stats_resolution": stats_resolution,
        "overtopping_stats_available": bool(has_stats_artifacts(stats_dir)),
        "metric_artifacts_complete": bool(metric_artifacts_complete),
        "evaluation_scope_complete": bool(evaluation_scope_complete),
        "stage6_dir": str(stage6_dir) if stage6_dir is not None else "",
        "completion_status": completion_status,
        "data_availability_status": data_availability_status,
        "pipeline_progress": pipeline_progress,
        "feature_report_status": feature_status,
        "stage5_completion_status": stage5_status,
        "singleton_schema": singleton.get("definition_version", "structural_zero_candidates" if structural_zero_candidates else "missing"),
        "interaction_schema": interaction.get("definition_version", "not_applicable_zero_candidates" if structural_zero_candidates else "missing"),
        "stage6_discovered_candidate_count": discovered_candidate_count,
        "stage6_candidate_count_status": discovery_candidate_status,
        "structural_zero_candidates": bool(structural_zero_candidates),
        "zero_candidate_undefined_metrics": (
            "s_1;R_ov;N_eff;TOC_m;N_eff_i2c;N_eff_c2i" if structural_zero_candidates else ""
        ),
        "materialized_singleton_events_available": materialized_events,
        "discovery_ranking_source_available": discovery_ranking_source,
        "stage5_interaction_runtime_available": stage5_runtime,
        "singleton_backfill_without_model_ablations": singleton_backfill,
        "interaction_backfill_requires_model": interaction_backfill,
        "cmc_required": bool(require_cmc),
        "rq2_interaction_applicable": rq2_applicable,
        "required_metric_names": ";".join(required_names),
        "all_required_metrics_exact": all_exact,
        "missing_required_metrics": ";".join(missing),
        **{f"exact_{name}": bool(ok) for name, ok in exact.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary-table", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument(
        "--primary-profile", default="configured", choices=PRIMARY_PROFILE_CHOICES,
        help="Validation profile for the configured study table. Default: configured (dynamic count).",
    )
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
        "structural_zero_setting_count": int(frame["structural_zero_candidates"].sum()) if len(frame) else 0,
        "incomplete_setting_count": int((~frame["all_required_metrics_exact"]).sum()) if len(frame) else 0,
        "rows": rows,
        "interpretation": {
            "singleton_backfill_without_model_ablations": "TOC_m plus directional U_J, N_t, and N_eff can be regenerated from materialized singleton flip events plus the discovery ranking, without repeating singleton model ablations.",
            "interaction_backfill_requires_model": (
                "For settings with a nonempty candidate set, E(J), its matched controls, and the singleton-union/full-set decomposition require the genuine simultaneous intervention output; "
                + ("CMC and paired conditional controls are also required for applicable settings. " if require_cmc else "CMC is not required by this audit. ")
                + "Singleton unions are never substituted for simultaneous interventions."
            ),
        },
    }
    (out_dir / "required_metrics_audit.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True, default=str), encoding="utf-8"
    )
    print(
        f"[required-metrics] complete={payload['complete_setting_count']}/{payload['setting_count']} "
        f"zero-candidate={payload['structural_zero_setting_count']} "
        f"audit={out_dir / 'required_metrics_audit.csv'}"
    )
    zero_rows = frame.loc[frame["structural_zero_candidates"]] if len(frame) else frame
    if len(zero_rows):
        print("[required-metrics] verified zero-candidate settings:")
        for _, item in zero_rows.iterrows():
            print(
                "  - "
                f"{item.get('task')} | {item.get('model')} | {item.get('phase')} "
                f"(Stage 6: {item.get('stage6_dir')})"
            )
    if payload["incomplete_setting_count"]:
        incomplete_rows = frame.loc[~frame["all_required_metrics_exact"]]
        print("[required-metrics] configured settings without full manuscript coverage:")
        for _, item in incomplete_rows.iterrows():
            missing_text = item.get("missing_required_metrics") or "none (stats exist; evaluation scope is partial)"
            print(
                "  - "
                f"{item.get('task')} | {item.get('model')} | {item.get('phase')} "
                f"[status={item.get('data_availability_status')}; stats={item.get('stats_resolution')}; "
                f"progress={item.get('pipeline_progress')}; missing_metrics={missing_text}]"
            )
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
            "The line above reports which metric families are missing. Directional singleton "
            "metrics can be rebuilt from materialized singleton events; simultaneous interaction "
            "metrics are required only where set composition is applicable and need their corresponding interaction artifacts."
        )


if __name__ == "__main__":
    main()
