#!/usr/bin/env python3
"""Interaction-aware validation for a frozen candidate set.

This script evaluates genuine simultaneous channel-set interventions.  It
reports the complete-set effect E(J), layer-population leave-subset effects,
GCCR_m, and exact-stratum matched random-set controls.  Singleton unions are
never used as substitutes for simultaneous interventions.
"""
from __future__ import annotations
from pathlib import Path


import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.caching_and_prompting import set_deterministic
from lib.feature_extraction_runner import resolve_task_spec
from lib.group_intervention import (
    GroupSpec,
    dedupe_units,
    evaluate_groups,
    evaluation_frame,
    load_candidate_units,
    load_dataset_info,
    load_layer_population,
    resolve_dataset_path,
    rows_fingerprint,
    simultaneous_effect,
    unit_metadata,
)
from lib.heldout_set_metrics import safe_ratio
from lib.interaction_statistics import matched_null_summary
from lib.high_n_singleton_eval import (
    UnitSpec,
    load_scores_for_baseline,
    precompute_replacements_for_units,
)
from lib.modeling_and_ablation import LMWrapper, get_device


LOG_PREFIX = "[interaction-validation]"



def _file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute simultaneous E(J), layer interaction GCCR_m, and matched "
            "random-set nulls. The evaluation split defaults to test."
        )
    )
    parser.add_argument("--input_data_dir", required=True)
    parser.add_argument("--candidate_flip_stats_path", required=True)
    parser.add_argument("--singleton_scores_path", default=None)
    parser.add_argument("--frozen_ranking_path", default=None)
    parser.add_argument("--layer_population_manifest", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--task_module", default="lib.tasks.arithmetic_task")
    parser.add_argument("--ai_model", default=None)
    parser.add_argument("--ai_model_cache_dir", default=None)
    parser.add_argument(
        "--intervention",
        choices=["zero", "mean", "mean-donor", "mean-positional", "mean-donor-positional"],
        default="mean-donor",
    )
    parser.add_argument("--decode_only", action="store_true")
    parser.add_argument(
        "--evaluation_split", choices=["test", "train", "all"], default="test",
        help="Rows used for simultaneous interventions. Default: test.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--points_to_use_for_mean_ablation", type=int, default=2048)
    parser.add_argument(
        "--m_values",
        default="1,2,3,5",
        help="Comma-separated m values. J_{l,m} contains the top min(m,|J∩C_l|) discovery-ranked channels in each layer population.",
    )
    parser.add_argument("--null_draws", type=int, default=100)
    parser.add_argument("--denominator_epsilon", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _parse_m_values(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in str(raw).split(",") if item.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("--m_values must contain positive integers")
    return values


def _load_ranking(path: Path, candidates: list[UnitSpec]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "unit_key" not in frame.columns:
        if not {"layer_label", "neuron_id"}.issubset(frame.columns):
            raise ValueError("Frozen ranking must contain unit_key or layer_label and neuron_id")
        frame["unit_key"] = [
            f"{layer}:{int(neuron)}"
            for layer, neuron in zip(frame["layer_label"], frame["neuron_id"])
        ]
    if "discovery_rank_global" not in frame.columns:
        raise ValueError("Frozen ranking is missing discovery_rank_global")
    candidate_keys = {unit.unit_key for unit in candidates}
    frame = frame.loc[frame["unit_key"].astype(str).isin(candidate_keys)].copy()
    if set(frame["unit_key"].astype(str)) != candidate_keys:
        missing = sorted(candidate_keys - set(frame["unit_key"].astype(str)))
        raise ValueError("Frozen ranking is incomplete: " + ", ".join(missing[:10]))
    frame["discovery_rank_global"] = pd.to_numeric(
        frame["discovery_rank_global"], errors="raise"
    ).astype(int)
    if "discovery_rank_within_layer" not in frame.columns:
        frame = frame.sort_values("discovery_rank_global", kind="mergesort")
        frame["discovery_rank_within_layer"] = (
            frame.groupby("layer_label", sort=False).cumcount() + 1
        )
    frame["discovery_rank_within_layer"] = pd.to_numeric(
        frame["discovery_rank_within_layer"], errors="raise"
    ).astype(int)
    return frame.sort_values("discovery_rank_global", kind="mergesort").reset_index(drop=True)


def _unit_lookup(units: Iterable[UnitSpec]) -> dict[str, UnitSpec]:
    return {unit.unit_key: unit for unit in units}


def _group(units: Iterable[UnitSpec], label: str) -> GroupSpec:
    return GroupSpec(tuple(dedupe_units(units)), label=label)


def _complement(population: list[UnitSpec], removed: Iterable[UnitSpec], label: str) -> GroupSpec:
    removed_keys = {unit.unit_key for unit in removed}
    return _group([unit for unit in population if unit.unit_key not in removed_keys], label)


def _validated_stratum_metadata(
    layer_label: str,
    population: list[UnitSpec],
    candidates: list[UnitSpec],
) -> dict:
    """Verify that an exact layer label is one auditable matching stratum."""
    population_meta = [unit_metadata(unit) for unit in population]
    candidate_meta = [unit_metadata(unit) for unit in candidates]
    identity_fields = ("transformer_layer", "computational_locus", "channel_type", "layer_population")
    identities = {
        tuple(meta.get(field) for field in identity_fields)
        for meta in population_meta + candidate_meta
    }
    if len(identities) != 1:
        raise ValueError(
            f"Layer population {layer_label!r} mixes computational strata: {sorted(identities, key=str)}"
        )
    identity = dict(zip(identity_fields, next(iter(identities))))
    if (
        identity["transformer_layer"] is None
        or identity["computational_locus"] == "unknown"
        or identity["channel_type"] == "unknown"
    ):
        raise ValueError(
            f"Cannot verify exact matched-control metadata for layer population {layer_label!r}: {identity}"
        )
    return {"layer_label": str(layer_label), **identity}


def _effect_for_group(
    group: GroupSpec,
    *,
    baseline: np.ndarray,
    post_by_key: dict[str, np.ndarray],
) -> dict:
    if group.size == 0:
        return {
            "count": 0,
            "denominator": int(len(baseline)),
            "effect": 0.0 if len(baseline) else math.nan,
            "status": "empty_intervention" if len(baseline) else "undefined_zero_denominator",
            "effect_B0": 0.0,
            "effect_B1": 0.0,
        }
    return simultaneous_effect(baseline, post_by_key[group.key])


def _write_plot(path: Path, values: list[float], candidate: float, title: str, xlabel: str) -> None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(finite) == 0 or not np.isfinite(candidate):
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    bins = min(30, max(8, int(np.sqrt(len(finite)))))
    ax.hist(finite, bins=bins)
    ax.axvline(candidate, linestyle="--", linewidth=1.5, label="candidate")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Matched random sets")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def _latex_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Metric & $m$ & Candidate & Null median & $\Delta$ & $P$ & $p_{\mathrm{MC}}$ & $B$ \\",
        r"\midrule",
    ]
    for row in summary.to_dict("records"):
        m = "--" if pd.isna(row.get("m")) else str(int(row["m"]))
        def fmt(value):
            return "--" if value is None or not np.isfinite(float(value)) else f"{float(value):.3f}"
        lines.append(
            f"{row['metric']} & {m} & {fmt(row['candidate'])} & {fmt(row['median_null'])} & "
            f"{fmt(row['Delta'])} & {fmt(row['P'])} & {fmt(row['p_MC'])} & "
            f"{int(row['null_draws_requested'])} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.null_draws < 1:
        raise ValueError("--null_draws must be at least 1")
    set_deterministic(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    m_values = _parse_m_values(args.m_values)

    input_data_dir = Path(args.input_data_dir).expanduser().resolve()
    candidate_path = Path(args.candidate_flip_stats_path).expanduser().resolve()
    stats_dir = candidate_path.parent
    singleton_scores_path = (
        Path(args.singleton_scores_path).expanduser().resolve()
        if args.singleton_scores_path
        else stats_dir / "scores.csv"
    )
    ranking_path = (
        Path(args.frozen_ranking_path).expanduser().resolve()
        if args.frozen_ranking_path
        else stats_dir / "frozen_candidate_ranking.csv"
    )
    manifest_path = (
        Path(args.layer_population_manifest).expanduser().resolve()
        if args.layer_population_manifest
        else input_data_dir / "manifest.json"
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_info_path = input_data_dir / "dataset_info.json"
    dataset_info = load_dataset_info(input_data_dir)
    task = resolve_task_spec(args.task_module)
    prompt_col = dataset_info.get("prompt_col") or task.DEFAULT_INPUT
    target_col = dataset_info.get("target_col") or task.DEFAULT_TARGETS[0]
    ai_model = args.ai_model or dataset_info.get("ai_model")
    if not ai_model:
        raise ValueError("Could not resolve model from --ai_model or dataset_info.json")
    train_scores_path = resolve_dataset_path(dataset_info["scores_path"], input_data_dir)

    configuration = {
        "schema": "interaction-validation-v2",
        "input_data_dir": str(input_data_dir),
        "candidate_flip_stats_path": str(candidate_path),
        "singleton_scores_path": str(singleton_scores_path),
        "frozen_ranking_path": str(ranking_path),
        "layer_population_manifest": str(manifest_path),
        "task_module": str(args.task_module),
        "ai_model": ai_model,
        "intervention": str(args.intervention),
        "intervention_phase": "decode_only" if args.decode_only else "prefill_decode",
        "m_values": m_values,
        "null_draws": int(args.null_draws),
        "denominator_epsilon": float(args.denominator_epsilon),
        "seed": int(args.seed),
        "input_fingerprints": {
            "candidate": _file_fingerprint(candidate_path),
            "singleton_scores": _file_fingerprint(singleton_scores_path),
            "frozen_ranking": _file_fingerprint(ranking_path),
            "layer_population_manifest": _file_fingerprint(manifest_path),
            "dataset_info": _file_fingerprint(dataset_info_path),
            "train_scores": _file_fingerprint(train_scores_path),
        },
    }
    config_path = out_dir / "interaction_configuration.json"
    summary_path = out_dir / "interaction_validation_summary.json"
    if not args.force and config_path.exists() and summary_path.exists():
        try:
            if json.loads(config_path.read_text(encoding="utf-8")) == configuration:
                print(f"{LOG_PREFIX} compatible cached validation found at {summary_path}; skipping model load")
                return
        except Exception:
            pass
    config_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")

    candidates = load_candidate_units(candidate_path)
    candidate_by_key = _unit_lookup(candidates)
    ranking = _load_ranking(ranking_path, candidates)
    layer_population = load_layer_population(manifest_path)

    candidates_by_layer: dict[str, list[UnitSpec]] = {}
    for unit in candidates:
        candidates_by_layer.setdefault(str(unit.layer_label), []).append(unit)
    candidate_layers = sorted(candidates_by_layer)
    population_layers = sorted(layer_population)
    missing_layers = [layer for layer in candidate_layers if layer not in layer_population]
    if missing_layers:
        raise ValueError(
            "Candidate layers are absent from the prespecified stage-5 population: "
            + ", ".join(missing_layers)
        )
    stratum_metadata: dict[str, dict] = {}
    for layer in population_layers:
        layer_candidates = candidates_by_layer.get(layer, [])
        population_keys = {unit.unit_key for unit in layer_population[layer]}
        candidate_keys = {unit.unit_key for unit in layer_candidates}
        missing = candidate_keys - population_keys
        if missing:
            raise ValueError(
                f"J is not a subset of C_l for {layer}: " + ", ".join(sorted(missing))
            )
        stratum_metadata[layer] = _validated_stratum_metadata(
            layer, layer_population[layer], layer_candidates
        )

    scores_df = evaluation_frame(
        singleton_scores_path,
        prompt_col=prompt_col,
        target_col=target_col,
        evaluation_split=str(args.evaluation_split),
    )
    baseline = pd.to_numeric(scores_df[target_col], errors="raise").to_numpy() > 0.5

    ranked_by_layer: dict[str, list[UnitSpec]] = {}
    for layer, part in ranking.groupby("layer_label", sort=False):
        ordered = part.sort_values(
            ["discovery_rank_within_layer", "discovery_rank_global"], kind="mergesort"
        )
        ranked_by_layer[str(layer)] = [candidate_by_key[key] for key in ordered["unit_key"].astype(str)]

    candidate_full = _group(candidates, "candidate_J")
    population_groups = {
        layer: _group(layer_population[layer], f"C_{layer}") for layer in population_layers
    }
    candidate_subsets: dict[tuple[str, int], list[UnitSpec]] = {}
    candidate_complements: dict[tuple[str, int], GroupSpec] = {}
    for layer in population_layers:
        for m in m_values:
            ranked_layer = ranked_by_layer.get(layer, [])
            subset = ranked_layer[: min(int(m), len(ranked_layer))]
            candidate_subsets[(layer, m)] = subset
            candidate_complements[(layer, m)] = _complement(
                layer_population[layer], subset, f"C_{layer}_minus_J_{m}"
            )

    # Draw exact-stratum controls. Same layer label fixes transformer layer,
    # computational locus, and channel type; run arguments fix phase/baseline.
    random_full_groups: dict[int, GroupSpec] = {}
    random_subsets: dict[tuple[int, str, int], list[UnitSpec]] = {}
    random_complements: dict[tuple[int, str, int], GroupSpec] = {}
    null_membership_rows: list[dict] = []
    for draw in range(int(args.null_draws)):
        full_units: list[UnitSpec] = []
        for layer in population_layers:
            layer_candidates = candidates_by_layer.get(layer, [])
            candidate_keys = {unit.unit_key for unit in layer_candidates}
            pool = [unit for unit in layer_population[layer] if unit.unit_key not in candidate_keys]
            q = len(layer_candidates)
            if len(pool) < q:
                raise ValueError(
                    f"Matched null impossible for {layer}: need {q} noncandidate units, found {len(pool)}"
                )
            if q:
                chosen_indices = rng.choice(len(pool), size=q, replace=False)
                chosen = [pool[int(index)] for index in np.atleast_1d(chosen_indices)]
                rng.shuffle(chosen)
            else:
                chosen = []
            full_units.extend(chosen)
            for unit in chosen:
                meta = stratum_metadata[layer]
                null_membership_rows.append(
                    {
                        "draw": draw,
                        "unit_key": unit.unit_key,
                        "layer_label": unit.layer_label,
                        "neuron_id": int(unit.neuron_id),
                        **meta,
                        "intervention_phase": "decode_only" if args.decode_only else "prefill_decode",
                        "replacement_baseline": args.intervention,
                    }
                )
            for m in m_values:
                q_m = len(candidate_subsets[(layer, m)])
                subset = chosen[:q_m]
                random_subsets[(draw, layer, m)] = subset
                random_complements[(draw, layer, m)] = _complement(
                    layer_population[layer], subset, f"C_{layer}_minus_R_{draw}_{m}"
                )
        random_full_groups[draw] = _group(full_units, f"random_J_{draw}")

    all_groups = [candidate_full]
    all_groups.extend(population_groups.values())
    all_groups.extend(candidate_complements.values())
    all_groups.extend(random_full_groups.values())
    all_groups.extend(random_complements.values())
    unique_groups = {group.key: group for group in all_groups if group.size > 0}
    groups_to_evaluate = list(unique_groups.values())
    all_units = dedupe_units(
        unit for group in groups_to_evaluate for unit in group.units
    )

    train_scores = load_scores_for_baseline(
        scores_path=train_scores_path,
        target_col=target_col,
        baseline_subset="all",
        task_targets=task.DEFAULT_TARGETS,
        split="train",
    )
    if train_scores.empty and args.intervention.startswith("mean"):
        raise ValueError("No training rows are available for replacement estimation")

    print(
        f"{LOG_PREFIX} split={args.evaluation_split} rows={len(scores_df)} candidates={len(candidates)} "
        f"layer_populations={len(population_layers)} candidate_layers={len(candidate_layers)} "
        f"groups={len(groups_to_evaluate)} null_draws={args.null_draws}"
    )
    device = get_device()
    model = LMWrapper(
        model_name=ai_model,
        device=device,
        eval_mode=True,
        circuit_discovery=False,
        cache_dir=args.ai_model_cache_dir,
    )
    mean_activations = precompute_replacements_for_units(
        model=model,
        units=all_units,
        scores_df_for_mean=train_scores,
        prompt_col=prompt_col,
        target_col=target_col,
        intervention=args.intervention,
        points_to_use=int(args.points_to_use_for_mean_ablation),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )
    cache_context = {
        "rows": rows_fingerprint(scores_df, prompt_col),
        "model": ai_model,
        "task_module": args.task_module,
        "evaluation_split": str(args.evaluation_split),
        "prompt_col": prompt_col,
        "target_col": target_col,
        "decode_only": bool(args.decode_only),
        "intervention": args.intervention,
        "max_new_tokens": int(task.MAX_NEW_TOKENS),
    }
    post_by_key = evaluate_groups(
        model=model,
        groups=groups_to_evaluate,
        scores_df=scores_df,
        prompt_col=prompt_col,
        is_answer_positive_fn=task.is_answer_positive,
        batch_size=int(args.batch_size),
        decode_only=bool(args.decode_only),
        intervention=args.intervention,
        mean_activations=mean_activations,
        max_new_tokens=int(task.MAX_NEW_TOKENS),
        cache_dir=out_dir / "group_eval_cache",
        cache_context=cache_context,
        force=bool(args.force),
    )

    candidate_effect = _effect_for_group(candidate_full, baseline=baseline, post_by_key=post_by_key)
    layer_effect_rows: list[dict] = []
    gccr_rows: list[dict] = []
    population_effects: dict[str, dict] = {}
    denominator = 0.0
    for layer in population_layers:
        effect = _effect_for_group(population_groups[layer], baseline=baseline, post_by_key=post_by_key)
        population_effects[layer] = effect
        denominator += float(effect["effect"])

    for m in m_values:
        numerator = 0.0
        for layer in population_layers:
            full_effect = population_effects[layer]
            complement = candidate_complements[(layer, m)]
            complement_effect = _effect_for_group(complement, baseline=baseline, post_by_key=post_by_key)
            delta = float(full_effect["effect"] - complement_effect["effect"])
            numerator += delta
            meta = stratum_metadata[layer]
            layer_effect_rows.append(
                {
                    "m": m,
                    "layer_population": layer,
                    "population_size": len(layer_population[layer]),
                    "candidate_layer_size": len(candidates_by_layer.get(layer, [])),
                    "J_l_m_size": len(candidate_subsets[(layer, m)]),
                    "J_l_m_unit_keys": json.dumps([unit.unit_key for unit in candidate_subsets[(layer, m)]]),
                    "E_l_C_l": full_effect["effect"],
                    "E_l_C_l_status": full_effect["status"],
                    "E_l_C_l_minus_J_l_m": complement_effect["effect"],
                    "E_l_C_l_minus_J_l_m_status": complement_effect["status"],
                    "Delta_l": delta,
                    **meta,
                }
            )
        ratio = safe_ratio(numerator, denominator, epsilon=float(args.denominator_epsilon))
        gccr_rows.append(
            {
                "m": m,
                "numerator_sum_Delta_l": numerator,
                "denominator_sum_E_l_C_l": denominator,
                "GCCR_m": ratio.value,
                "status": ratio.status,
                "denominator_epsilon": float(args.denominator_epsilon),
            }
        )

    null_rows: list[dict] = []
    e_null_values: list[float] = []
    gccr_null_values: dict[int, list[float]] = {m: [] for m in m_values}
    for draw in range(int(args.null_draws)):
        random_effect = _effect_for_group(random_full_groups[draw], baseline=baseline, post_by_key=post_by_key)
        e_null_values.append(float(random_effect["effect"]))
        null_rows.append(
            {
                "draw": draw,
                "metric": "E_J",
                "m": None,
                "value": random_effect["effect"],
                "status": random_effect["status"],
            }
        )
        for m in m_values:
            numerator = 0.0
            for layer in population_layers:
                complement_effect = _effect_for_group(
                    random_complements[(draw, layer, m)], baseline=baseline, post_by_key=post_by_key
                )
                numerator += float(population_effects[layer]["effect"] - complement_effect["effect"])
            ratio = safe_ratio(numerator, denominator, epsilon=float(args.denominator_epsilon))
            gccr_null_values[m].append(ratio.value)
            null_rows.append(
                {
                    "draw": draw,
                    "metric": "GCCR_m",
                    "m": m,
                    "value": ratio.value,
                    "status": ratio.status,
                    "numerator_sum_Delta_l": numerator,
                    "denominator_sum_E_l_C_l": denominator,
                }
            )

    gccr_by_m = {int(row["m"]): row for row in gccr_rows}
    summary_rows = [matched_null_summary(candidate_effect["effect"], e_null_values, metric="E_J", m=None)]
    for m in m_values:
        summary_rows.append(
            matched_null_summary(
                float(gccr_by_m[m]["GCCR_m"]),
                gccr_null_values[m],
                metric="GCCR_m",
                m=m,
            )
        )

    layer_population_rows = []
    for layer in population_layers:
        candidate_keys = {unit.unit_key for unit in candidates_by_layer.get(layer, [])}
        for unit in layer_population[layer]:
            layer_population_rows.append(
                {
                    "layer_population": layer,
                    "unit_key": unit.unit_key,
                    "layer_label": unit.layer_label,
                    "neuron_id": int(unit.neuron_id),
                    "is_candidate": unit.unit_key in candidate_keys,
                    **stratum_metadata[layer],
                }
            )

    pd.DataFrame([stratum_metadata[layer] for layer in population_layers]).to_csv(
        out_dir / "matched_control_strata.csv", index=False
    )
    pd.DataFrame(layer_population_rows).to_csv(out_dir / "layer_populations.csv", index=False)
    ranking.to_csv(out_dir / "frozen_candidate_ranking.csv", index=False)
    pd.DataFrame(layer_effect_rows).to_csv(out_dir / "layer_interaction_effects.csv", index=False)
    pd.DataFrame(gccr_rows).to_csv(out_dir / "gccr_metrics.csv", index=False)
    pd.DataFrame(null_membership_rows).to_csv(out_dir / "matched_random_set_membership.csv", index=False)
    pd.DataFrame(null_rows).to_csv(out_dir / "matched_null_draws.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "interaction_validation_summary.csv", index=False)
    (out_dir / "interaction_validation_table.tex").write_text(
        _latex_table(summary_df), encoding="utf-8"
    )

    payload = {
        "definition_version": "interaction-validation-v2",
        "definitions": {
            "E_J": "P(B_J(x) != B(x)) under simultaneous suppression of the complete fixed set J",
            "Delta_l": "E_l(C_l)-E_l(C_l\\J_lm)",
            "GCCR_m": "sum_l Delta_l / sum_l E_l(C_l)",
            "Delta_null": "candidate-median(null)",
            "P": "(1 + sum_b 1{null_b <= candidate})/(B+1)",
            "p_MC": "(1 + sum_b 1{null_b >= candidate})/(B+1)",
        },
        "candidate_set": [unit.unit_key for unit in candidates],
        "candidate_set_size": len(candidates),
        "population_layer_count": len(population_layers),
        "candidate_layer_count": len(candidate_layers),
        "population_scope": "all layer populations parsed from the prespecified stage-5 manifest",
        "candidate_E_J": candidate_effect,
        "m_values": m_values,
        "GCCR_m": gccr_rows,
        "null_summary": summary_rows,
        "null_draws": int(args.null_draws),
        "matching": [
            "exact layer population label",
            "transformer layer",
            "computational locus",
            "channel type",
            "intervention phase",
            "cardinality",
            "replacement baseline",
        ],
        "intervention_phase": "decode_only" if args.decode_only else "prefill_decode",
        "replacement_baseline": args.intervention,
        "candidate_discovery_split": "train",
        "replacement_reference_split": "train",
        "evaluation_split": str(args.evaluation_split),
        "layer_population_manifest": str(manifest_path),
        "matched_control_strata_path": str(out_dir / "matched_control_strata.csv"),
        "frozen_ranking_path": str(ranking_path),
        "denominator_epsilon": float(args.denominator_epsilon),
        "notes": [
            "No value is clipped to [0,1].",
            "Near-zero GCCR denominators are reported as undefined.",
            "The GCCR denominator sums E_l(C_l) over every prespecified stage-5 layer population; layers with J_lm empty contribute Delta_l=0.",
            "Every E value is obtained from a simultaneous intervention on the named set.",
            "Singleton-union events are not used for E(J), GCCR, or null controls.",
            "Random controls exclude candidate channels and match exact layer populations.",
        ],
    }
    (out_dir / "interaction_validation_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )

    markdown = [
        "# Interaction-aware global validation",
        "",
        f"- Candidate set size: {len(candidates)}",
        f"- Evaluation split: {args.evaluation_split}",
        f"- Evaluation examples: {len(scores_df)}",
        f"- Prespecified layer populations: {len(population_layers)}",
        f"- Candidate-containing layer populations: {len(candidate_layers)}",
        f"- Matched random draws: {args.null_draws}",
        f"- Intervention phase: {'decode only' if args.decode_only else 'input and output'}",
        f"- Replacement baseline: {args.intervention}",
        "- All set effects use simultaneous interventions; values are not clipped.",
        "",
        "| Metric | m | Candidate | Null median | Delta | P | p_MC | Status |",
        "|:--|--:|--:|--:|--:|--:|--:|:--|",
    ]
    for row in summary_rows:
        m_text = "--" if row["m"] is None else str(row["m"])
        def fmt(value):
            return "NA" if value is None or not np.isfinite(float(value)) else f"{float(value):.4f}"
        markdown.append(
            f"| {row['metric']} | {m_text} | {fmt(row['candidate'])} | {fmt(row['median_null'])} | "
            f"{fmt(row['Delta'])} | {fmt(row['P'])} | {fmt(row['p_MC'])} | {row['status']} |"
        )
    (out_dir / "interaction_validation_summary.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    _write_plot(
        out_dir / "E_J_matched_null",
        e_null_values,
        float(candidate_effect["effect"]),
        "Simultaneous full-set effect",
        "E(J)",
    )
    for m in m_values:
        _write_plot(
            out_dir / f"GCCR_{m}_matched_null",
            gccr_null_values[m],
            float(gccr_by_m[m]["GCCR_m"]),
            f"Interaction-aware global coverage, m={m}",
            f"GCCR_{m}",
        )

    global_path = stats_dir / "flip_stats_global.json"
    if global_path.exists():
        global_payload = json.loads(global_path.read_text(encoding="utf-8"))
        global_payload.update(
            {
                "E_J": candidate_effect["effect"],
                "E_J_count": candidate_effect["count"],
                "E_J_denominator": candidate_effect["denominator"],
                "E_J_status": candidate_effect["status"],
                "E_J_OCC_0": candidate_effect.get("effect_B0"),
                "E_J_OCC_1": candidate_effect.get("effect_B1"),
                "GCCR_m": {
                    str(row["m"]): {
                        "value": row["GCCR_m"],
                        "status": row["status"],
                        "numerator": row["numerator_sum_Delta_l"],
                        "denominator": row["denominator_sum_E_l_C_l"],
                    }
                    for row in gccr_rows
                },
                "interaction_validation_path": str(out_dir / "interaction_validation_summary.json"),
                "interaction_null_summary_path": str(out_dir / "interaction_validation_summary.csv"),
            }
        )
        global_path.write_text(
            json.dumps(global_payload, indent=2, allow_nan=True), encoding="utf-8"
        )

    print(summary_df.to_string(index=False))
    print(f"{LOG_PREFIX} wrote {out_dir}")


if __name__ == "__main__":
    main()
