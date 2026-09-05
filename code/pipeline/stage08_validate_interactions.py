#!/usr/bin/env python3
"""Validate frozen candidates with genuine simultaneous-set interventions.

Two complementary questions are evaluated on one fixed evaluation-example set:

1. ``E(J)``: the unconditional simultaneous effect of suppressing the complete
   frozen candidate set ``J``, compared with structurally matched noncandidate
   sets ``K_b``.
2. Conditional marginal contribution (CMC), unless ``--skip_cmc`` is set:
   for each draw ``b`` a noncandidate background ``S_b`` is sampled and held
   fixed while candidate and null are compared in the same perturbed context::

       M_b(J)   = E(S_b ∪ J)   - E(S_b)
       M_b(K_b) = E(S_b ∪ K_b) - E(S_b)
       D_b      = M_b(J)       - M_b(K_b)

``K_b`` and ``S_b`` are matched to the candidate topology by transformer
layer, computational locus, channel type and per-stratum cardinality.  Phase
and replacement baseline are fixed globally.  Candidate/null/background sets
are always evaluated by genuine simultaneous interventions; singleton unions
are never substituted.

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

from core.caching_and_prompting import set_deterministic
from core.feature_extraction_runner import resolve_task_spec
from core.group_intervention import (
    GroupSpec,
    dedupe_units,
    evaluate_groups,
    evaluation_frame,
    evaluation_row_records,
    group_units_by_matching_stratum,
    load_candidate_units,
    load_complete_group_batch_cache,
    load_dataset_info,
    load_stage5_locus_population,
    resolve_dataset_path,
    simultaneous_effect,
    unit_metadata,
)
from core.interaction_statistics import matched_null_summary, paired_conditional_summary
from core.high_n_singleton_eval import (
    UnitSpec,
    build_mean_prompt_pool,
    load_scores_for_baseline,
    precompute_replacements_for_units,
)
from core.modeling_and_ablation import LMWrapper, get_device
from studies.overtopping.analysis.lib.interaction_schema import CURRENT_INTERACTION_SCHEMA


LOG_PREFIX = "[conditional-validation]"
SCHEMA = CURRENT_INTERACTION_SCHEMA
GROUP_CACHE_SCHEMA = "simultaneous-group-eval-v2"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_data_dir", required=True)
    parser.add_argument("--candidate_flip_stats_path", required=True)
    parser.add_argument("--singleton_scores_path", default=None)
    parser.add_argument("--frozen_ranking_path", default=None)
    parser.add_argument("--layer_population_manifest", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--task_module", default="core.tasks.arithmetic_task")
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
        help="Rows used for every candidate, background and null intervention. Default: test.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--points_to_use_for_mean_ablation", type=int, default=2048)
    parser.add_argument(
        "--background_multipliers", default="1",
        help=(
            "Comma-separated nonnegative integer background loads. For multiplier q, "
            "S_b contains q times the candidate count in every exact layer/locus/type stratum. "
            "Default: 1."
        ),
    )
    parser.add_argument("--null_draws", type=int, default=100)
    parser.add_argument(
        "--skip_cmc",
        action="store_true",
        help=(
            "Compute simultaneous E(J) and its matched-null distribution only. "
            "Skip conditional backgrounds and all CMC interventions/outputs."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for replacement-reference sampling and matched sets. Default: 42.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _parse_background_multipliers(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0:
            raise ValueError("--background_multipliers values must be nonnegative integers")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("--background_multipliers must contain at least one integer")
    return tuple(values)


def _load_ranking(path: Path, candidates: list[UnitSpec]) -> pd.DataFrame:
    """Validate that the candidate set is the same frozen discovery set.

    Conditional validation does not rank candidates, but retaining this check
    prevents a stale/mismatched candidate file from being evaluated silently.
    """
    frame = pd.read_csv(path)
    if "unit_key" not in frame.columns:
        if not {"layer_label", "neuron_id"}.issubset(frame.columns):
            raise ValueError("Frozen ranking must contain unit_key or layer_label and neuron_id")
        frame["unit_key"] = [
            f"{layer}:{int(neuron)}"
            for layer, neuron in zip(frame["layer_label"], frame["neuron_id"])
        ]
    candidate_keys = {unit.unit_key for unit in candidates}
    frame = frame.loc[frame["unit_key"].astype(str).isin(candidate_keys)].copy()
    observed = set(frame["unit_key"].astype(str))
    if observed != candidate_keys:
        missing = sorted(candidate_keys - observed)
        raise ValueError("Frozen ranking is incomplete: " + ", ".join(missing[:10]))
    if "discovery_rank_global" in frame.columns:
        frame["discovery_rank_global"] = pd.to_numeric(
            frame["discovery_rank_global"], errors="raise"
        ).astype(int)
        frame = frame.sort_values("discovery_rank_global", kind="mergesort")
    return frame.reset_index(drop=True)


def _group(units: Iterable[UnitSpec], label: str) -> GroupSpec:
    return GroupSpec(tuple(dedupe_units(units)), label=label)


def _effect_for_group(group: GroupSpec, *, baseline: np.ndarray, post_by_key: dict[str, np.ndarray]) -> dict:
    if group.size == 0:
        n = int(len(baseline))
        return {
            "count": 0,
            "denominator": n,
            "effect": 0.0 if n else math.nan,
            "status": "empty_intervention" if n else "undefined_zero_denominator",
            "effect_0to1": 0.0,
            "effect_1to0": 0.0,
        }
    return simultaneous_effect(baseline, post_by_key[group.key])


def _write_plot(path: Path, values: list[float], reference: float, title: str, xlabel: str) -> None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(finite) == 0 or not np.isfinite(reference):
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    bins = min(30, max(8, int(np.sqrt(len(finite)))))
    ax.hist(finite, bins=bins)
    ax.axvline(reference, linestyle="--", linewidth=1.5, label="candidate")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Matched draws")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _write_paired_plot(path: Path, candidate_values: list[float], null_values: list[float], title: str) -> None:
    cand = np.asarray(candidate_values, dtype=float)
    null = np.asarray(null_values, dtype=float)
    mask = np.isfinite(cand) & np.isfinite(null)
    if not mask.any():
        return
    delta = cand[mask] - null[mask]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    bins = min(30, max(8, int(np.sqrt(len(delta)))))
    ax.hist(delta, bins=bins)
    ax.axvline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"$M_b(J)-M_b(K_b)$")
    ax.set_ylabel("Paired backgrounds")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _latex_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Metric & Background & Candidate & Null median & $\Delta$ & $P$ & $p_{\mathrm{MC}}$ & $B$ \\",
        r"\midrule",
    ]
    for row in summary.to_dict("records"):
        def fmt(value):
            try:
                number = float(value)
            except Exception:
                return "--"
            return "--" if not np.isfinite(number) else f"{number:.3f}"
        background = "--" if pd.isna(row.get("background_multiplier")) else f"{int(row['background_multiplier'])}x"
        lines.append(
            f"{row['metric']} & {background} & {fmt(row.get('candidate'))} & "
            f"{fmt(row.get('median_null'))} & {fmt(row.get('Delta'))} & "
            f"{fmt(row.get('P'))} & {fmt(row.get('p_MC'))} & "
            f"{int(row.get('null_draws_requested', 0))} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _draw_matched_null_group(
    *,
    draw: int,
    candidate_strata: dict[tuple[int, str, str], list[UnitSpec]],
    population_strata: dict[tuple[int, str, str], list[UnitSpec]],
    candidate_keys: set[str],
    seed: int,
) -> GroupSpec:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 101, int(draw)]))
    units: list[UnitSpec] = []
    for stratum in sorted(candidate_strata):
        q = len(candidate_strata[stratum])
        pool = [unit for unit in population_strata[stratum] if unit.unit_key not in candidate_keys]
        if len(pool) < q:
            raise ValueError(
                f"Matched null impossible for stratum {stratum}: need {q} noncandidate units, found {len(pool)}"
            )
        chosen = rng.choice(len(pool), size=q, replace=False)
        units.extend(pool[int(index)] for index in np.atleast_1d(chosen))
    return _group(units, f"random_J_{draw}")


def _draw_background_group(
    *,
    draw: int,
    multiplier: int,
    candidate_strata: dict[tuple[int, str, str], list[UnitSpec]],
    population_strata: dict[tuple[int, str, str], list[UnitSpec]],
    candidate_keys: set[str],
    null_group: GroupSpec,
    seed: int,
) -> GroupSpec:
    if multiplier == 0:
        return _group([], f"background_{multiplier}x_{draw}")
    excluded = candidate_keys | {unit.unit_key for unit in null_group.units}
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 31337, int(multiplier), int(draw)])
    )
    units: list[UnitSpec] = []
    for stratum in sorted(candidate_strata):
        q = int(multiplier) * len(candidate_strata[stratum])
        pool = [unit for unit in population_strata[stratum] if unit.unit_key not in excluded]
        if len(pool) < q:
            raise ValueError(
                "Conditional background impossible for stratum "
                f"{stratum}: multiplier={multiplier} requires {q} additional noncandidate units "
                f"after excluding J and K_b, found {len(pool)}. "
                "Use a smaller --background_multipliers value; matching is never relaxed silently."
            )
        chosen = rng.choice(len(pool), size=q, replace=False)
        units.extend(pool[int(index)] for index in np.atleast_1d(chosen))
    return _group(units, f"background_{multiplier}x_{draw}")


def _union_group(*groups: GroupSpec, label: str) -> GroupSpec:
    return _group((unit for group in groups for unit in group.units), label)



def main() -> None:
    args = parse_args()
    if args.null_draws < 1:
        raise ValueError("--null_draws must be at least 1")
    compute_cmc = not bool(args.skip_cmc)
    background_multipliers = (
        _parse_background_multipliers(args.background_multipliers) if compute_cmc else ()
    )

    input_data_dir = Path(args.input_data_dir).expanduser().resolve()
    candidate_path = Path(args.candidate_flip_stats_path).expanduser().resolve()
    stats_dir = candidate_path.parent
    singleton_scores_path = (
        Path(args.singleton_scores_path).expanduser().resolve()
        if args.singleton_scores_path else stats_dir / "scores.csv"
    )
    ranking_path = (
        Path(args.frozen_ranking_path).expanduser().resolve()
        if args.frozen_ranking_path else stats_dir / "frozen_candidate_ranking.csv"
    )
    manifest_path = (
        Path(args.layer_population_manifest).expanduser().resolve()
        if args.layer_population_manifest else input_data_dir / "manifest.json"
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    global_path = stats_dir / "flip_stats_global.json"

    dataset_info = load_dataset_info(input_data_dir)
    task = resolve_task_spec(args.task_module)
    prompt_col = dataset_info.get("prompt_col") or task.DEFAULT_INPUT
    target_col = dataset_info.get("target_col") or task.DEFAULT_TARGETS[0]
    ai_model = args.ai_model or dataset_info.get("ai_model")
    if not ai_model:
        raise ValueError("Could not resolve model from --ai_model or dataset_info.json")
    train_scores_path = resolve_dataset_path(dataset_info["scores_path"], input_data_dir)

    candidates = load_candidate_units(candidate_path)
    candidate_keys_list = [unit.unit_key for unit in candidates]
    candidate_keys = set(candidate_keys_list)
    ranking = _load_ranking(ranking_path, candidates)

    locus_population = load_stage5_locus_population(manifest_path)
    all_population_units = dedupe_units(unit for units in locus_population.values() for unit in units)
    population_strata = group_units_by_matching_stratum(all_population_units)
    candidate_strata = group_units_by_matching_stratum(candidates)
    for stratum, stratum_candidates in candidate_strata.items():
        eligible = population_strata.get(stratum, [])
        eligible_keys = {unit.unit_key for unit in eligible}
        missing = {unit.unit_key for unit in stratum_candidates} - eligible_keys
        if missing:
            raise ValueError(
                f"Candidate stratum {stratum} is absent/incomplete in stage-5 population: "
                + ", ".join(sorted(missing))
            )

    scores_df = evaluation_frame(
        singleton_scores_path,
        prompt_col=prompt_col,
        target_col=target_col,
        evaluation_split=str(args.evaluation_split),
    )
    evaluation_rows = evaluation_row_records(
        scores_df,
        prompt_col=prompt_col,
        target_col=target_col,
    )
    baseline = pd.to_numeric(scores_df[target_col], errors="raise").to_numpy() > 0.5
    effective_seed = int(args.seed if args.seed is not None else 42)
    set_deterministic(effective_seed)

    train_scores = load_scores_for_baseline(
        scores_path=train_scores_path,
        target_col=target_col,
        baseline_subset="all",
        task_targets=task.DEFAULT_TARGETS,
        split="train",
    )
    if train_scores.empty and args.intervention.startswith("mean"):
        raise ValueError("No training rows are available for replacement estimation")
    replacement_reference_prompts = build_mean_prompt_pool(
        train_scores,
        prompt_col=prompt_col,
        target_col=target_col,
        n_points=int(args.points_to_use_for_mean_ablation),
        seed=effective_seed,
    )

    cache_identity = {
        "schema": SCHEMA,
        "task_module": str(args.task_module),
        "ai_model": ai_model,
        "intervention": str(args.intervention),
        "intervention_phase": "decode_only" if args.decode_only else "prefill_decode",
        "evaluation_split": str(args.evaluation_split),
        "prompt_col": str(prompt_col),
        "target_col": str(target_col),
        "evaluation_rows": evaluation_rows,
        "points_to_use_for_mean_ablation": int(args.points_to_use_for_mean_ablation),
        "replacement_reference_prompts": list(map(str, replacement_reference_prompts)),
        "seed": effective_seed,
        "max_new_tokens": int(task.MAX_NEW_TOKENS),
        "candidate_set": sorted(candidate_keys_list),
    }
    paths = {
        "input_data_dir": str(input_data_dir),
        "candidate_flip_stats_path": str(candidate_path),
        "singleton_scores_path": str(singleton_scores_path),
        "frozen_ranking_path": str(ranking_path),
        "layer_population_manifest": str(manifest_path),
        "replacement_reference_scores_path": str(train_scores_path),
    }
    configuration = {
        **cache_identity,
        "cache_identity": cache_identity,
        "paths": paths,
    }
    config_path = out_dir / "interaction_configuration.json"
    summary_path = out_dir / "interaction_validation_summary.json"
    # Build structurally matched K_b sets.
    random_full_groups: dict[int, GroupSpec] = {}
    for draw in range(int(args.null_draws)):
        random_full_groups[draw] = _draw_matched_null_group(
            draw=draw,
            candidate_strata=candidate_strata,
            population_strata=population_strata,
            candidate_keys=candidate_keys,
            seed=effective_seed,
        )

    # Backgrounds are matched by the same exact strata as J and are disjoint
    # from both J and the paired K_b.
    background_groups: dict[tuple[int, int], GroupSpec] = {}
    candidate_context_groups: dict[tuple[int, int], GroupSpec] = {}
    null_context_groups: dict[tuple[int, int], GroupSpec] = {}
    for multiplier in background_multipliers:
        for draw in range(int(args.null_draws)):
            background = _draw_background_group(
                draw=draw,
                multiplier=multiplier,
                candidate_strata=candidate_strata,
                population_strata=population_strata,
                candidate_keys=candidate_keys,
                null_group=random_full_groups[draw],
                seed=effective_seed,
            )
            background_groups[(multiplier, draw)] = background
            candidate_context_groups[(multiplier, draw)] = _union_group(
                background, _group(candidates, "candidate_J"),
                label=f"background_{multiplier}x_{draw}_plus_candidate_J",
            )
            null_context_groups[(multiplier, draw)] = _union_group(
                background, random_full_groups[draw],
                label=f"background_{multiplier}x_{draw}_plus_random_J_{draw}",
            )

    # Validate and write matching strata before model execution.
    stratum_rows: list[dict] = []
    for stratum in sorted(candidate_strata):
        transformer_layer, computational_locus, channel_type = stratum
        q = len(candidate_strata[stratum])
        eligible = len(population_strata[stratum])
        stratum_rows.append({
            "transformer_layer": int(transformer_layer),
            "computational_locus": computational_locus,
            "channel_type": channel_type,
            "candidate_count": q,
            "eligible_stage5_count": eligible,
            "noncandidate_pool_count": eligible - q,
            "intervention_phase": "decode_only" if args.decode_only else "prefill_decode",
            "replacement_baseline": args.intervention,
        })
    pd.DataFrame(stratum_rows).to_csv(out_dir / "matched_control_strata.csv", index=False)

    null_membership_rows: list[dict] = []
    background_membership_rows: list[dict] = []
    for draw, group in random_full_groups.items():
        for unit in group.units:
            meta = unit_metadata(unit)
            null_membership_rows.append({
                "draw": draw,
                "unit_key": unit.unit_key,
                "stage5_locus": meta["stage5_locus"],
                "neuron_id": int(unit.neuron_id),
                "transformer_layer": int(meta["transformer_layer"]),
                "computational_locus": meta["computational_locus"],
                "channel_type": meta["channel_type"],
                "intervention_phase": "decode_only" if args.decode_only else "prefill_decode",
                "replacement_baseline": args.intervention,
                "evaluation_population_n": int(len(scores_df)),
            })
    for (multiplier, draw), group in background_groups.items():
        for unit in group.units:
            meta = unit_metadata(unit)
            background_membership_rows.append({
                "background_multiplier": multiplier,
                "draw": draw,
                "unit_key": unit.unit_key,
                "stage5_locus": meta["stage5_locus"],
                "neuron_id": int(unit.neuron_id),
                "transformer_layer": int(meta["transformer_layer"]),
                "computational_locus": meta["computational_locus"],
                "channel_type": meta["channel_type"],
                "intervention_phase": "decode_only" if args.decode_only else "prefill_decode",
                "replacement_baseline": args.intervention,
                "evaluation_population_n": int(len(scores_df)),
            })
    pd.DataFrame(null_membership_rows).to_csv(out_dir / "matched_random_set_membership.csv", index=False)
    conditional_membership_path = out_dir / "conditional_background_membership.csv"
    if compute_cmc:
        pd.DataFrame(background_membership_rows).to_csv(conditional_membership_path, index=False)
    elif conditional_membership_path.exists():
        print(f"{LOG_PREFIX} preserving pre-existing optional CMC artifact: {conditional_membership_path}")
    ranking.to_csv(out_dir / "frozen_candidate_ranking.csv", index=False)

    candidate_full = _group(candidates, "candidate_J")
    all_groups: list[GroupSpec] = [candidate_full, *random_full_groups.values()]
    for key in sorted(background_groups):
        all_groups.extend([
            background_groups[key], candidate_context_groups[key], null_context_groups[key]
        ])
    unique_groups = {group.key: group for group in all_groups if group.size > 0}
    groups_to_evaluate = list(unique_groups.values())
    all_units = dedupe_units(unit for group in groups_to_evaluate for unit in group.units)

    # Record the analysis configuration after the exact requested simultaneous
    # groups are known.  These fields describe reporting/analysis semantics; the
    # model-output cache below uses only the inputs that can change a group output.
    cache_identity.update({
        "compute_cmc": bool(compute_cmc),
        "background_multipliers": list(background_multipliers),
        "null_draws": int(args.null_draws),
        "requested_group_keys": sorted(unique_groups),
    })
    configuration.update(cache_identity)
    configuration["cache_identity"] = cache_identity

    # Group-intervention cache identity is explicit.  Rewriting Stage-7 or
    # Step-7b CSV/JSON files with the same rows/groups does not invalidate these
    # expensive model outputs, and batch size is deliberately absent because it
    # is operational rather than scientific.
    group_cache_context = {
        "schema": GROUP_CACHE_SCHEMA,
        "model": ai_model,
        "task_module": str(args.task_module),
        "evaluation_split": str(args.evaluation_split),
        "prompt_col": str(prompt_col),
        "target_col": str(target_col),
        "evaluation_rows": evaluation_rows,
        "decode_only": bool(args.decode_only),
        "intervention": str(args.intervention),
        "replacement_reference_prompts": list(map(str, replacement_reference_prompts)),
        "max_new_tokens": int(task.MAX_NEW_TOKENS),
    }
    print(
        f"{LOG_PREFIX} split={args.evaluation_split} rows={len(scores_df)} candidates={len(candidates)} "
        f"matching_strata={len(candidate_strata)} cmc={'on' if compute_cmc else 'off'} "
        f"backgrounds={background_multipliers if compute_cmc else 'disabled'} "
        f"groups_to_evaluate={len(groups_to_evaluate)} null_draws={args.null_draws}"
    )

    post_by_key: dict[str, np.ndarray] = {}
    if groups_to_evaluate:
        group_cache_dir = out_dir / "group_eval_cache"
        if not args.force:
            cached_outputs = load_complete_group_batch_cache(
                cache_dir=group_cache_dir,
                groups=groups_to_evaluate,
                n_examples=len(scores_df),
                batch_size=int(args.batch_size),
                cache_context=group_cache_context,
            )
            if cached_outputs is not None:
                post_by_key = cached_outputs
                print(
                    f"{LOG_PREFIX} complete explicit cache covers all {len(groups_to_evaluate)} "
                    f"groups and {len(scores_df)} rows; skipping model load"
                )

        if not post_by_key:
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
                seed=effective_seed,
            )
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
                cache_dir=group_cache_dir,
                cache_context=group_cache_context,
                force=bool(args.force),
            )

    candidate_effect = _effect_for_group(
        candidate_full, baseline=baseline, post_by_key=post_by_key
    )
    e_null_values: list[float] = []
    direct_null_rows: list[dict] = []
    for draw in range(int(args.null_draws)):
        effect = _effect_for_group(random_full_groups[draw], baseline=baseline, post_by_key=post_by_key)
        value = float(effect["effect"])
        status = str(effect["status"])
        e_null_values.append(value)
        direct_null_rows.append({
            "draw": draw,
            "metric": "E_J",
            "value": value,
            "status": status,
            "evaluation_population_n": int(len(scores_df)),
        })

    direct_summary = matched_null_summary(
        float(candidate_effect["effect"]), e_null_values, metric="E_J", m=None
    )

    conditional_draw_rows: list[dict] = []
    conditional_summaries: list[dict] = []
    for multiplier in background_multipliers:
        candidate_marginals: list[float] = []
        null_marginals: list[float] = []
        for draw in range(int(args.null_draws)):
            background = _effect_for_group(
                background_groups[(multiplier, draw)], baseline=baseline, post_by_key=post_by_key
            )
            candidate_context = _effect_for_group(
                candidate_context_groups[(multiplier, draw)], baseline=baseline, post_by_key=post_by_key
            )
            null_context = _effect_for_group(
                null_context_groups[(multiplier, draw)], baseline=baseline, post_by_key=post_by_key
            )
            candidate_marginal = float(candidate_context["effect"] - background["effect"])
            null_marginal = float(null_context["effect"] - background["effect"])
            paired_delta = float(candidate_marginal - null_marginal)
            candidate_marginals.append(candidate_marginal)
            null_marginals.append(null_marginal)
            conditional_draw_rows.append({
                "background_multiplier": multiplier,
                "draw": draw,
                "E_S": background["effect"],
                "E_S_plus_J": candidate_context["effect"],
                "E_S_plus_K": null_context["effect"],
                "candidate_marginal": candidate_marginal,
                "null_marginal": null_marginal,
                "paired_Delta": paired_delta,
                "background_status": background["status"],
                "candidate_context_status": candidate_context["status"],
                "null_context_status": null_context["status"],
                "evaluation_population_n": int(len(scores_df)),
            })
        summary = paired_conditional_summary(
            candidate_marginals,
            null_marginals,
            background_multiplier=multiplier,
        )
        conditional_summaries.append(summary)
        _write_paired_plot(
            out_dir / f"conditional_marginal_{multiplier}x_paired_delta",
            candidate_marginals,
            null_marginals,
            f"Conditional marginal contribution ({multiplier}x background)",
        )

    summary_rows = [direct_summary, *conditional_summaries]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "interaction_validation_summary.csv", index=False)
    pd.DataFrame(direct_null_rows).to_csv(out_dir / "matched_null_draws.csv", index=False)
    if compute_cmc:
        summary_df.loc[summary_df["metric"].astype(str) == "conditional_marginal"].to_csv(
            out_dir / "conditional_marginal_summary.csv", index=False
        )
        pd.DataFrame(conditional_draw_rows).to_csv(
            out_dir / "conditional_marginal_draws.csv", index=False
        )
    else:
        preserved = [
            out_dir / "conditional_marginal_summary.csv",
            out_dir / "conditional_marginal_draws.csv",
            *list(out_dir.glob("conditional_marginal_*_paired_delta.pdf")),
        ]
        for stale in preserved:
            if stale.exists():
                print(f"{LOG_PREFIX} preserving pre-existing optional CMC artifact: {stale}")
    (out_dir / "interaction_validation_table.tex").write_text(_latex_table(summary_df), encoding="utf-8")

    payload = {
        "definition_version": SCHEMA,
        "definitions": {
            "E_J": "P(B_J(x) != B(x)) under simultaneous suppression of the complete fixed candidate set J",
            "K_b": "noncandidate set exactly matched to J by transformer layer, computational locus, channel type and per-stratum cardinality",
            "S_b": "noncandidate background matched to the candidate topology, disjoint from J and its paired K_b",
            "candidate_marginal": "M_b(J)=E(S_b union J)-E(S_b)",
            "null_marginal": "M_b(K_b)=E(S_b union K_b)-E(S_b)",
            "paired_Delta": "D_b=M_b(J)-M_b(K_b)",
            "conditional_candidate": "mean_b M_b(J)",
            "conditional_Delta": "mean_b D_b",
            "P": "(1 + sum_b 1{D_b >= 0})/(B+1) for the paired conditional metric",
            "p_MC": "(1 + sum_b 1{D_b <= 0})/(B+1) for the paired conditional metric",
        },
        "candidate_set": candidate_keys_list,
        "candidate_set_size": len(candidates),
        "candidate_E_J": candidate_effect,
        "direct_E_J_null_summary": direct_summary,
        "cmc_enabled": bool(compute_cmc),
        "conditional_marginal": conditional_summaries,
        "background_multipliers": list(background_multipliers),
        "null_draws": int(args.null_draws),
        "matching": [
            "transformer layer", "computational locus", "channel type",
            "intervention phase", "replacement baseline", "per-stratum cardinality",
        ],
        "intervention_phase": "decode_only" if args.decode_only else "prefill_decode",
        "replacement_baseline": args.intervention,
        "candidate_discovery_split": "train",
        "replacement_reference_split": "train",
        "evaluation_split": str(args.evaluation_split),
        "evaluation_population_n": int(len(scores_df)),
        "evaluation_row_identity_columns": list(evaluation_rows[0].keys()) if evaluation_rows else [],
        "same_evaluation_rows_for_candidate_background_and_all_null_draws": True,
        "layer_population_manifest": str(manifest_path),
        "frozen_ranking_path": str(ranking_path),
        "points_to_use_for_mean_ablation": int(args.points_to_use_for_mean_ablation),
        "seed": effective_seed,
        "notes": [
            "No effect is clipped.",
            "Every E value is obtained from a genuine simultaneous intervention on the named set.",
            "Singleton-union events are never used for E(J) or null controls.",
            *(
                [
                    "No marginal contribution is clipped.",
                    "Singleton-union events are never used for conditional marginals.",
                    "Candidate and paired null are compared in the identical S_b background and on identical evaluation examples.",
                ]
                if compute_cmc
                else ["CMC was disabled; no conditional-background interventions were evaluated."]
            ),
        ],
    }
    summary_path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    config_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")

    markdown = [
        (
            "# Simultaneous-set and conditional marginal validation"
            if compute_cmc else "# Simultaneous-set validation"
        ),
        "",
        f"- Candidate set size: {len(candidates)}",
        f"- Evaluation split: {args.evaluation_split}",
        f"- Evaluation examples: {len(scores_df)} (explicit row identities stored in interaction_configuration.json)",
        f"- Exact matched strata: {len(candidate_strata)}",
        f"- Matched random draws: {args.null_draws}",
        f"- CMC: {'enabled' if compute_cmc else 'disabled'}",
        f"- Background multipliers: {', '.join(map(str, background_multipliers)) if compute_cmc else 'not evaluated'}",
        f"- Intervention phase: {'decode only' if args.decode_only else 'input and output'}",
        f"- Replacement baseline: {args.intervention}",
        "- Every set effect is a genuine simultaneous intervention.",
        *(
            ["- Candidate and matched null use the same background S_b in each paired draw."]
            if compute_cmc else []
        ),
        "",
        "| Metric | Background | Candidate | Null median | Delta | P | p_MC | Status |",
        "|:--|:--|--:|--:|--:|--:|--:|:--|",
    ]
    for row in summary_rows:
        background = "--" if row.get("background_multiplier") is None else f"{row['background_multiplier']}x"
        def fmt(value):
            try:
                number = float(value)
            except Exception:
                return "NA"
            return "NA" if not np.isfinite(number) else f"{number:.4f}"
        markdown.append(
            f"| {row['metric']} | {background} | {fmt(row.get('candidate'))} | "
            f"{fmt(row.get('median_null'))} | {fmt(row.get('Delta'))} | "
            f"{fmt(row.get('P'))} | {fmt(row.get('p_MC'))} | {row.get('status', 'unknown')} |"
        )
    (out_dir / "interaction_validation_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    _write_plot(
        out_dir / "E_J_matched_null",
        e_null_values,
        float(candidate_effect["effect"]),
        "Simultaneous full-set effect",
        "E(J)",
    )

    if global_path.exists():
        global_payload = json.loads(global_path.read_text(encoding="utf-8"))
        global_payload.update({
            "E_J": candidate_effect["effect"],
            "E_J_count": candidate_effect.get("count"),
            "E_J_denominator": candidate_effect.get("denominator"),
            "E_J_status": candidate_effect.get("status"),
            "E_J_0to1": candidate_effect.get("effect_0to1"),
            "E_J_1to0": candidate_effect.get("effect_1to0"),
            "conditional_marginal": {
                str(row["background_multiplier"]): {
                    "value": row["candidate"],
                    "candidate_median": row["candidate_median"],
                    "null_mean": row["null_mean"],
                    "null_median": row["median_null"],
                    "paired_Delta_mean": row["Delta"],
                    "paired_Delta_median": row["Delta_median"],
                    "P": row["P"],
                    "p_MC": row["p_MC"],
                    "paired_win_rate": row["paired_win_rate"],
                    "status": row["status"],
                }
                for row in conditional_summaries
            },
            "interaction_validation_definition_version": SCHEMA,
            "interaction_validation_path": str(summary_path),
            "interaction_null_summary_path": str(out_dir / "interaction_validation_summary.csv"),
        })
        global_path.write_text(json.dumps(global_payload, indent=2, allow_nan=True), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"{LOG_PREFIX} wrote {out_dir}")


if __name__ == "__main__":
    main()
