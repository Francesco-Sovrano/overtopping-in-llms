#!/usr/bin/env python3
"""Interaction-aware validation for a frozen candidate set.

GCCR is defined over transformer-layer populations.  For every transformer
layer that contains at least one frozen candidate, ``C_l`` is the complete
eligible stage-5 channel population in that transformer layer, aggregated
across MLP and attention loci.  ``J ∩ C_l`` contains the frozen candidates in
that transformer layer.  Only ``m=1`` and ``m=all`` are evaluated.

Every reported E value comes from a genuine simultaneous-set intervention.
Singleton-union events are never substituted for E(J), layer effects, GCCR, or
matched-null controls.
"""
from __future__ import annotations

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
    group_units_by_matching_stratum,
    load_candidate_units,
    load_dataset_info,
    load_transformer_layer_population,
    matching_stratum_key,
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
SCHEMA = "interaction-validation-v3"
M_VALUES = ("1", "all")


def _file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute simultaneous E(J), candidate-support-transformer-layer "
            "GCCR for m=1 and m=all, and exactly matched random-set nulls. "
            "The evaluation split defaults to test."
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
        help="Rows used for every candidate and null simultaneous intervention. Default: test.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--points_to_use_for_mean_ablation", type=int, default=2048)
    parser.add_argument(
        "--m_values",
        default="1,all",
        help=(
            "Compatibility option. GCCR is defined only for m=1 and m=all; "
            "the only accepted value is '1,all' (order/whitespace may vary)."
        ),
    )
    parser.add_argument("--null_draws", type=int, default=100)
    parser.add_argument("--denominator_epsilon", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _validate_m_values(raw: str) -> tuple[str, str]:
    tokens = {item.strip().lower() for item in str(raw).split(",") if item.strip()}
    if tokens != {"1", "all"}:
        raise ValueError("GCCR is computed only for m=1 and m=all; use --m_values 1,all")
    return M_VALUES


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
    observed = set(frame["unit_key"].astype(str))
    if observed != candidate_keys:
        missing = sorted(candidate_keys - observed)
        raise ValueError("Frozen ranking is incomplete: " + ", ".join(missing[:10]))
    frame["discovery_rank_global"] = pd.to_numeric(
        frame["discovery_rank_global"], errors="raise"
    ).astype(int)
    return frame.sort_values("discovery_rank_global", kind="mergesort").reset_index(drop=True)


def _unit_lookup(units: Iterable[UnitSpec]) -> dict[str, UnitSpec]:
    return {unit.unit_key: unit for unit in units}


def _group(units: Iterable[UnitSpec], label: str) -> GroupSpec:
    return GroupSpec(tuple(dedupe_units(units)), label=label)


def _complement(population: list[UnitSpec], removed: Iterable[UnitSpec], label: str) -> GroupSpec:
    removed_keys = {unit.unit_key for unit in removed}
    return _group([unit for unit in population if unit.unit_key not in removed_keys], label)


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


def _m_text(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "--"
    text = str(raw).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def _latex_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Metric & $m$ & Candidate & Null median & $\Delta$ & $P$ & $p_{\mathrm{MC}}$ & $B$ \\",
        r"\midrule",
    ]
    for row in summary.to_dict("records"):
        m = _m_text(row.get("m"))

        def fmt(value):
            return "--" if value is None or not np.isfinite(float(value)) else f"{float(value):.3f}"

        lines.append(
            f"{row['metric']} & {m} & {fmt(row['candidate'])} & {fmt(row['median_null'])} & "
            f"{fmt(row['Delta'])} & {fmt(row['P'])} & {fmt(row['p_MC'])} & "
            f"{int(row['null_draws_requested'])} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _support_layer_candidates(
    candidates: list[UnitSpec],
) -> dict[int, list[UnitSpec]]:
    output: dict[int, list[UnitSpec]] = {}
    for unit in candidates:
        layer, _locus, _channel_type = matching_stratum_key(unit)
        output.setdefault(int(layer), []).append(unit)
    return {layer: dedupe_units(units) for layer, units in sorted(output.items())}


def _ranked_candidates_by_support_layer(
    ranking: pd.DataFrame,
    candidate_by_key: dict[str, UnitSpec],
) -> dict[int, list[UnitSpec]]:
    output: dict[int, list[UnitSpec]] = {}
    for row in ranking.sort_values("discovery_rank_global", kind="mergesort").to_dict("records"):
        unit = candidate_by_key[str(row["unit_key"])]
        layer, _locus, _channel_type = matching_stratum_key(unit)
        output.setdefault(int(layer), []).append(unit)
    return output


def _stratum_row(key: tuple[int, str, str], candidate_count: int, eligible_count: int) -> dict:
    transformer_layer, computational_locus, channel_type = key
    return {
        "transformer_layer": int(transformer_layer),
        "computational_locus": str(computational_locus),
        "channel_type": str(channel_type),
        "candidate_count": int(candidate_count),
        "eligible_stage5_count": int(eligible_count),
        "noncandidate_pool_count": int(eligible_count - candidate_count),
    }


def main() -> None:
    args = parse_args()
    if args.null_draws < 1:
        raise ValueError("--null_draws must be at least 1")
    m_values = _validate_m_values(args.m_values)
    set_deterministic(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

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
        "schema": SCHEMA,
        "input_data_dir": str(input_data_dir),
        "candidate_flip_stats_path": str(candidate_path),
        "singleton_scores_path": str(singleton_scores_path),
        "frozen_ranking_path": str(ranking_path),
        "layer_population_manifest": str(manifest_path),
        "task_module": str(args.task_module),
        "ai_model": ai_model,
        "intervention": str(args.intervention),
        "intervention_phase": "decode_only" if args.decode_only else "prefill_decode",
        "evaluation_split": str(args.evaluation_split),
        "m_values": list(m_values),
        "gccr_population_scope": "candidate-support transformer layers",
        "null_matching": [
            "transformer layer",
            "computational locus",
            "channel type",
            "intervention phase",
            "replacement baseline",
            "per-layer cardinality",
        ],
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
    transformer_population = load_transformer_layer_population(manifest_path)
    candidates_by_transformer_layer = _support_layer_candidates(candidates)
    candidate_support_layers = sorted(candidates_by_transformer_layer)

    missing_support_layers = [
        layer for layer in candidate_support_layers if layer not in transformer_population
    ]
    if missing_support_layers:
        raise ValueError(
            "Candidate transformer layers are absent from the stage-5 population: "
            + ", ".join(map(str, missing_support_layers))
        )

    # GCCR uses only transformer layers containing at least one frozen candidate.
    support_population = {
        layer: transformer_population[layer]
        for layer in candidate_support_layers
    }
    for layer in candidate_support_layers:
        population_keys = {unit.unit_key for unit in support_population[layer]}
        candidate_keys = {unit.unit_key for unit in candidates_by_transformer_layer[layer]}
        missing = candidate_keys - population_keys
        if missing:
            raise ValueError(
                f"J is not a subset of C_l for transformer layer {layer}: "
                + ", ".join(sorted(missing))
            )

    ranked_by_transformer_layer = _ranked_candidates_by_support_layer(ranking, candidate_by_key)
    for layer in candidate_support_layers:
        ranked_keys = {unit.unit_key for unit in ranked_by_transformer_layer.get(layer, [])}
        candidate_keys = {unit.unit_key for unit in candidates_by_transformer_layer[layer]}
        if ranked_keys != candidate_keys:
            raise ValueError(f"Frozen discovery ranking is incomplete in transformer layer {layer}")

    population_strata = group_units_by_matching_stratum(
        unit for layer in candidate_support_layers for unit in support_population[layer]
    )
    candidate_strata = group_units_by_matching_stratum(candidates)
    all_candidate_keys = {unit.unit_key for unit in candidates}
    for stratum, stratum_candidates in candidate_strata.items():
        eligible = population_strata.get(stratum, [])
        eligible_keys = {unit.unit_key for unit in eligible}
        missing = {unit.unit_key for unit in stratum_candidates} - eligible_keys
        if missing:
            raise ValueError(
                f"Candidate matching stratum {stratum} is absent/incomplete in stage-5 population: "
                + ", ".join(sorted(missing))
            )
        noncandidate_count = sum(unit.unit_key not in all_candidate_keys for unit in eligible)
        if noncandidate_count < len(stratum_candidates):
            raise ValueError(
                f"Matched null impossible for stratum {stratum}: need {len(stratum_candidates)} "
                f"noncandidate units, found {noncandidate_count}"
            )

    scores_df = evaluation_frame(
        singleton_scores_path,
        prompt_col=prompt_col,
        target_col=target_col,
        evaluation_split=str(args.evaluation_split),
    )
    evaluation_rows_fingerprint = rows_fingerprint(scores_df, prompt_col)
    baseline = pd.to_numeric(scores_df[target_col], errors="raise").to_numpy() > 0.5

    candidate_full = _group(candidates, "candidate_J")
    population_groups = {
        layer: _group(support_population[layer], f"C_transformer_layer_{layer}")
        for layer in candidate_support_layers
    }
    candidate_subsets: dict[tuple[int, str], list[UnitSpec]] = {}
    candidate_complements: dict[tuple[int, str], GroupSpec] = {}
    for layer in candidate_support_layers:
        ranked_layer = ranked_by_transformer_layer[layer]
        candidate_subsets[(layer, "1")] = ranked_layer[:1]
        candidate_subsets[(layer, "all")] = list(ranked_layer)
        for m in m_values:
            candidate_complements[(layer, m)] = _complement(
                support_population[layer],
                candidate_subsets[(layer, m)],
                f"C_transformer_layer_{layer}_minus_J_{m}",
            )

    population_effect_cache_path = out_dir / "candidate_support_layer_effects.json"
    population_effect_cache_identity = {
        "schema": SCHEMA,
        "evaluation_rows_fingerprint": evaluation_rows_fingerprint,
        "evaluation_split": str(args.evaluation_split),
        "intervention_phase": "decode_only" if args.decode_only else "prefill_decode",
        "replacement_baseline": args.intervention,
        "ai_model": ai_model,
        "task_module": str(args.task_module),
        "population_group_keys": {
            str(layer): population_groups[layer].key
            for layer in candidate_support_layers
        },
    }
    cached_population_effects: dict[int, dict] | None = None
    if population_effect_cache_path.exists() and not args.force:
        try:
            cached_payload = json.loads(population_effect_cache_path.read_text(encoding="utf-8"))
            cached_identity = {key: cached_payload.get(key) for key in population_effect_cache_identity}
            cached_effects = cached_payload.get("effects") or {}
            if (
                cached_identity == population_effect_cache_identity
                and set(cached_effects) == {str(layer) for layer in candidate_support_layers}
            ):
                cached_population_effects = {
                    int(layer): dict(cached_effects[str(layer)])
                    for layer in candidate_support_layers
                }
        except Exception:
            cached_population_effects = None

    # Draw exact matched controls.  Full-set E(J) matches every candidate
    # stratum exactly.  GCCR m=1 removes one control from the exact stratum of
    # the discovery-top candidate in that transformer layer; m=all removes the
    # complete per-layer matched control set.
    random_full_groups: dict[int, GroupSpec] = {}
    random_subsets: dict[tuple[int, int, str], list[UnitSpec]] = {}
    random_complements: dict[tuple[int, int, str], GroupSpec] = {}
    null_membership_rows: list[dict] = []
    for draw in range(int(args.null_draws)):
        chosen_by_stratum: dict[tuple[int, str, str], list[UnitSpec]] = {}
        full_units: list[UnitSpec] = []
        for stratum, stratum_candidates in candidate_strata.items():
            eligible = population_strata[stratum]
            pool = [unit for unit in eligible if unit.unit_key not in all_candidate_keys]
            q = len(stratum_candidates)
            chosen_indices = rng.choice(len(pool), size=q, replace=False)
            chosen = [pool[int(index)] for index in np.atleast_1d(chosen_indices)]
            rng.shuffle(chosen)
            chosen_by_stratum[stratum] = chosen
            full_units.extend(chosen)
            for unit in chosen:
                transformer_layer, computational_locus, channel_type = stratum
                null_membership_rows.append(
                    {
                        "draw": draw,
                        "unit_key": unit.unit_key,
                        "stage5_locus": unit.layer_label,
                        "neuron_id": int(unit.neuron_id),
                        "transformer_layer": int(transformer_layer),
                        "computational_locus": computational_locus,
                        "channel_type": channel_type,
                        "candidate_stratum_cardinality": q,
                        "intervention_phase": "decode_only" if args.decode_only else "prefill_decode",
                        "replacement_baseline": args.intervention,
                        "evaluation_rows_fingerprint": evaluation_rows_fingerprint,
                    }
                )

        random_full_groups[draw] = _group(full_units, f"random_J_{draw}")
        for layer in candidate_support_layers:
            chosen_layer = [
                unit
                for stratum, units in chosen_by_stratum.items()
                if stratum[0] == layer
                for unit in units
            ]
            if len(chosen_layer) != len(candidates_by_transformer_layer[layer]):
                raise AssertionError(
                    f"Per-layer null cardinality mismatch for transformer layer {layer}"
                )
            top_candidate = candidate_subsets[(layer, "1")][0]
            top_stratum = matching_stratum_key(top_candidate)
            random_subsets[(draw, layer, "1")] = [chosen_by_stratum[top_stratum][0]]
            random_subsets[(draw, layer, "all")] = dedupe_units(chosen_layer)
            for m in m_values:
                random_complements[(draw, layer, m)] = _complement(
                    support_population[layer],
                    random_subsets[(draw, layer, m)],
                    f"C_transformer_layer_{layer}_minus_R_{draw}_{m}",
                )

    all_groups = [candidate_full]
    if cached_population_effects is None:
        all_groups.extend(population_groups.values())
    all_groups.extend(candidate_complements.values())
    all_groups.extend(random_full_groups.values())
    all_groups.extend(random_complements.values())
    unique_groups = {group.key: group for group in all_groups if group.size > 0}
    groups_to_evaluate = list(unique_groups.values())
    all_units = dedupe_units(unit for group in groups_to_evaluate for unit in group.units)

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
        f"candidate_support_transformer_layers={len(candidate_support_layers)} "
        f"matching_strata={len(candidate_strata)} groups={len(groups_to_evaluate)} "
        f"layer_effect_cache={'hit' if cached_population_effects is not None else 'miss'} "
        f"null_draws={args.null_draws}"
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
        "schema": SCHEMA,
        "rows": evaluation_rows_fingerprint,
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

    # Each E_l(C_l) is evaluated once per candidate-support transformer layer,
    # then reused for m=1, m=all, and every null draw.  The simultaneous group
    # output itself is persisted by evaluate_groups/group_eval_cache.
    if cached_population_effects is not None:
        population_effects = cached_population_effects
    else:
        population_effects: dict[int, dict] = {}
        for layer in candidate_support_layers:
            population_effects[layer] = _effect_for_group(
                population_groups[layer], baseline=baseline, post_by_key=post_by_key
            )
    population_effect_cache = {
        **population_effect_cache_identity,
        "effects": {
            str(layer): {
                "transformer_layer": int(layer),
                "C_l_size": len(support_population[layer]),
                **population_effects[layer],
            }
            for layer in candidate_support_layers
        },
    }
    population_effect_cache_path.write_text(
        json.dumps(population_effect_cache, indent=2, allow_nan=True), encoding="utf-8"
    )

    denominator = float(sum(float(population_effects[layer]["effect"]) for layer in candidate_support_layers))
    layer_effect_rows: list[dict] = []
    gccr_rows: list[dict] = []
    for m in m_values:
        numerator = 0.0
        for layer in candidate_support_layers:
            full_effect = population_effects[layer]
            complement = candidate_complements[(layer, m)]
            complement_effect = _effect_for_group(complement, baseline=baseline, post_by_key=post_by_key)
            delta = float(full_effect["effect"] - complement_effect["effect"])
            numerator += delta
            selected = candidate_subsets[(layer, m)]
            layer_effect_rows.append(
                {
                    "m": m,
                    "transformer_layer": int(layer),
                    "C_l_size": len(support_population[layer]),
                    "J_intersect_C_l_size": len(candidates_by_transformer_layer[layer]),
                    "J_l_m_size": len(selected),
                    "J_l_m_unit_keys": json.dumps([unit.unit_key for unit in selected]),
                    "E_l_C_l": full_effect["effect"],
                    "E_l_C_l_status": full_effect["status"],
                    "E_l_C_l_minus_J_l_m": complement_effect["effect"],
                    "E_l_C_l_minus_J_l_m_status": complement_effect["status"],
                    "Delta_l": delta,
                    "evaluation_rows_fingerprint": evaluation_rows_fingerprint,
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
                "candidate_support_transformer_layer_count": len(candidate_support_layers),
            }
        )

    null_rows: list[dict] = []
    e_null_values: list[float] = []
    gccr_null_values: dict[str, list[float]] = {m: [] for m in m_values}
    for draw in range(int(args.null_draws)):
        random_effect = _effect_for_group(
            random_full_groups[draw], baseline=baseline, post_by_key=post_by_key
        )
        e_null_values.append(float(random_effect["effect"]))
        null_rows.append(
            {
                "draw": draw,
                "metric": "E_J",
                "m": None,
                "value": random_effect["effect"],
                "status": random_effect["status"],
                "evaluation_rows_fingerprint": evaluation_rows_fingerprint,
            }
        )
        for m in m_values:
            numerator = 0.0
            for layer in candidate_support_layers:
                complement_effect = _effect_for_group(
                    random_complements[(draw, layer, m)],
                    baseline=baseline,
                    post_by_key=post_by_key,
                )
                numerator += float(
                    population_effects[layer]["effect"] - complement_effect["effect"]
                )
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
                    "evaluation_rows_fingerprint": evaluation_rows_fingerprint,
                }
            )

    gccr_by_m = {str(row["m"]): row for row in gccr_rows}
    summary_rows = [
        matched_null_summary(candidate_effect["effect"], e_null_values, metric="E_J", m=None)
    ]
    for m in m_values:
        summary_rows.append(
            matched_null_summary(
                float(gccr_by_m[m]["GCCR_m"]),
                gccr_null_values[m],
                metric="GCCR_m",
                m=m,
            )
        )

    stratum_rows = []
    for stratum, stratum_candidates in candidate_strata.items():
        stratum_rows.append(
            _stratum_row(
                stratum,
                len(stratum_candidates),
                len(population_strata[stratum]),
            )
        )
    pd.DataFrame(stratum_rows).to_csv(out_dir / "matched_control_strata.csv", index=False)

    population_rows = []
    for layer in candidate_support_layers:
        candidate_keys = {unit.unit_key for unit in candidates_by_transformer_layer[layer]}
        for unit in support_population[layer]:
            meta = unit_metadata(unit)
            population_rows.append(
                {
                    "transformer_layer": int(layer),
                    "unit_key": unit.unit_key,
                    "stage5_locus": meta["stage5_locus"],
                    "computational_locus": meta["computational_locus"],
                    "channel_type": meta["channel_type"],
                    "neuron_id": int(unit.neuron_id),
                    "is_candidate": unit.unit_key in candidate_keys,
                }
            )
    # Keep the historical filename for downstream compatibility; its columns
    # now state explicitly that C_l is transformer-layer based.
    pd.DataFrame(population_rows).to_csv(out_dir / "layer_populations.csv", index=False)
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
        "definition_version": SCHEMA,
        "definitions": {
            "C_l": (
                "full eligible stage-5 channel population of transformer layer l, "
                "restricted to transformer layers containing at least one frozen candidate"
            ),
            "J_intersect_C_l": "all frozen candidates in candidate-support transformer layer l",
            "J_l_1": "discovery-top frozen candidate within candidate-support transformer layer l",
            "J_l_all": "all frozen candidates in candidate-support transformer layer l",
            "E_J": "P(B_J(x) != B(x)) under simultaneous suppression of the complete fixed set J",
            "Delta_l": "E_l(C_l)-E_l(C_l\\J_lm)",
            "GCCR_m": "sum_l Delta_l / sum_l E_l(C_l), over candidate-support transformer layers only",
            "Delta_null": "candidate-median(null)",
            "P": "(1 + sum_b 1{null_b <= candidate})/(B+1)",
            "p_MC": "(1 + sum_b 1{null_b >= candidate})/(B+1)",
        },
        "candidate_set": [unit.unit_key for unit in candidates],
        "candidate_set_size": len(candidates),
        "candidate_support_transformer_layers": candidate_support_layers,
        "candidate_support_transformer_layer_count": len(candidate_support_layers),
        "population_scope": "candidate-support transformer layers only",
        "candidate_E_J": candidate_effect,
        "m_values": list(m_values),
        "GCCR_m": gccr_rows,
        "null_summary": summary_rows,
        "null_draws": int(args.null_draws),
        "matching": [
            "transformer layer",
            "computational locus",
            "channel type",
            "intervention phase",
            "replacement baseline",
            "per-layer cardinality",
        ],
        "intervention_phase": "decode_only" if args.decode_only else "prefill_decode",
        "replacement_baseline": args.intervention,
        "candidate_discovery_split": "train",
        "replacement_reference_split": "train",
        "evaluation_split": str(args.evaluation_split),
        "evaluation_rows_fingerprint": evaluation_rows_fingerprint,
        "same_evaluation_rows_for_candidate_and_all_null_draws": True,
        "layer_population_manifest": str(manifest_path),
        "matched_control_strata_path": str(out_dir / "matched_control_strata.csv"),
        "candidate_support_layer_effects_cache_path": str(out_dir / "candidate_support_layer_effects.json"),
        "frozen_ranking_path": str(ranking_path),
        "denominator_epsilon": float(args.denominator_epsilon),
        "notes": [
            "No value is clipped to [0,1].",
            "Near-zero GCCR denominators are reported as undefined.",
            "Candidate-free transformer layers are excluded from GCCR.",
            "Each E_l(C_l) is computed once and reused for m=1, m=all, and every null draw.",
            "Every E value is obtained from a simultaneous intervention on the named set.",
            "Singleton-union events are not used for E(J), GCCR, or null controls.",
            "Random controls exclude candidate channels and exactly match layer/locus/type counts.",
            "Candidate and every null draw use the identical fixed evaluation-example subset.",
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
        f"- Evaluation-row fingerprint: `{evaluation_rows_fingerprint}`",
        f"- Candidate-support transformer layers: {len(candidate_support_layers)}",
        f"- Exact matched strata: {len(candidate_strata)}",
        f"- Matched random draws: {args.null_draws}",
        f"- Intervention phase: {'decode only' if args.decode_only else 'input and output'}",
        f"- Replacement baseline: {args.intervention}",
        "- GCCR is reported only for m=1 and m=all.",
        "- All set effects use simultaneous interventions; values are not clipped.",
        "",
        "| Metric | m | Candidate | Null median | Delta | P | p_MC | Status |",
        "|:--|:--|--:|--:|--:|--:|--:|:--|",
    ]
    for row in summary_rows:
        m_text = _m_text(row["m"])

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
                "interaction_validation_definition_version": SCHEMA,
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
