#!/usr/bin/env python3
"""Temporal cutoff intervention for the RQ3 spiking story.

For each frozen Stage-7 singleton agonist with held-out directional flip support,
this experiment can run either complementary decode-time schedule:

``prefix``
    Keep the full intervention active for the first ``t`` autoregressive decode
    transitions, then remove it for all later transitions.

``suffix``
    Keep the first ``K-t`` autoregressive decode transitions clean, then apply
    the full intervention for the final ``t`` transitions.

Here ``t`` is always the *number of intervened decode transitions*.  In
``--decode_only`` mode prompt prefill remains clean, so ``t=0`` is the clean
baseline and ``t=K`` reproduces the full output-only Stage-7 intervention.  In
standard I+O mode prompt prefill is intervened for every ``t``: ``t=0`` is the
prompt-only endpoint and ``t=K`` reproduces the full I+O intervention.

Important step semantics
------------------------
The first generated token is selected from prompt-prefill logits. Those logits
are clean in decode-only mode and intervened in standard mode.
``active_decode_steps=1`` controls the transition that processes generated token
1 to produce token 2. The maximum horizon is ``MAX_NEW_TOKENS - 1``.

Earlier intervention consequences are intentionally allowed to persist in the KV
cache/model state after the cutoff. The manipulation removes *future direct
intervention*; it does not erase causal consequences that have already occurred.
"""
from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from core.caching_and_prompting import set_deterministic
from core.feature_extraction_runner import resolve_task_spec
from core.group_intervention import evaluation_frame, load_dataset_info, resolve_dataset_path
from core.high_n_singleton_eval import UnitSpec, load_scores_for_baseline
from core.modeling_and_ablation import LMWrapper, build_ablation_hooks, clone_kv_cache, get_device
from core.neuron_intervention import build_prefix_caches_for_examples
from studies.overtopping.analysis.graded_agonist_intervention import (
    _directional_flip_col,
    _load_candidate_plan,
    _load_stage7_replacement_population,
    _precompute_stage7_replacements,
    _row_id,
    _sample_indices,
    _stable_seed,
)

LOG_PREFIX = "[temporal-cutoff]"
PREFIX_SCHEMA = "temporal-cutoff-intervention-v3"
SUFFIX_SCHEMA = "temporal-suffix-intervention-v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_data_dir", required=True)
    p.add_argument("--candidate_flip_stats_path", required=True)
    p.add_argument("--singleton_scores_path", default=None)
    p.add_argument("--candidate_ranking_path", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--task_module", default="core.tasks.arithmetic_task")
    p.add_argument("--ai_model", default=None)
    p.add_argument("--ai_model_cache_dir", default=None)
    p.add_argument(
        "--intervention",
        choices=["zero", "mean", "mean-donor", "mean-positional", "mean-donor-positional"],
        default="mean-donor",
    )
    phase = p.add_mutually_exclusive_group()
    phase.add_argument(
        "--decode_only", dest="decode_only", action="store_true",
        help="Use output-only semantics: keep prompt prefill clean (historical standalone default).",
    )
    phase.add_argument(
        "--input_output", dest="decode_only", action="store_false",
        help="Use I+O semantics: intervene during prompt prefill and the selected decode transitions.",
    )
    p.set_defaults(decode_only=True)
    p.add_argument("--evaluation_split", choices=["test", "train", "all"], default="test")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--points_to_use_for_mean_ablation", type=int, default=2048)
    p.add_argument(
        "--active_decode_steps",
        default=None,
        help=(
            "Comma-separated counts of intervened decode transitions. Default: every integer "
            "from 0 through MAX_NEW_TOKENS-1. Count 0 means no decode-time intervention; "
            "the maximum count reproduces the full decode-only intervention."
        ),
    )
    p.add_argument(
        "--schedule",
        choices=["prefix", "suffix"],
        default="prefix",
        help=(
            "Temporal placement of the active transitions. prefix = intervene on the first t "
            "decode transitions; suffix = intervene on the final t decode transitions. "
            "Default prefix preserves the historical temporal-cutoff experiment."
        ),
    )
    p.add_argument(
        "--max_agonists_per_direction",
        type=int,
        default=16,
        help="Maximum frozen agonists per discovery direction; 0 means all eligible agonists.",
    )
    p.add_argument(
        "--max_positive_support_per_agonist",
        type=int,
        default=256,
        help="Maximum held-out known-flip examples per agonist; 0 means all support.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no_plot", action="store_true")
    return p.parse_args()


def _parse_active_steps(raw: str | None, *, max_steps: int) -> tuple[int, ...]:
    if max_steps < 0:
        raise ValueError("max_steps must be >= 0")
    if raw is None or not str(raw).strip():
        return tuple(range(0, max_steps + 1))
    vals: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if not 0 <= value <= max_steps:
            raise ValueError(f"active decode steps must lie in [0, {max_steps}]")
        if value not in vals:
            vals.append(value)
    vals = sorted(vals)
    if 0 not in vals or max_steps not in vals:
        raise ValueError(f"active decode steps must include both 0 and {max_steps}")
    return tuple(vals)


def _requested_run_config(args: argparse.Namespace, active_steps: tuple[int, ...], *, max_decode_steps: int) -> dict:
    config = {
        "scientific_target": "temporal cutoff necessity for held-out singleton overtopping support",
        "input_data_dir": str(Path(args.input_data_dir).expanduser().resolve()),
        "candidate_flip_stats_path": str(Path(args.candidate_flip_stats_path).expanduser().resolve()),
        "singleton_scores_path": str(Path(args.singleton_scores_path).expanduser().resolve()) if args.singleton_scores_path else None,
        "candidate_ranking_path": str(Path(args.candidate_ranking_path).expanduser().resolve()) if args.candidate_ranking_path else None,
        "out_dir": str(Path(args.out_dir).expanduser().resolve()),
        "task_module": str(args.task_module),
        "ai_model": args.ai_model,
        "intervention": str(args.intervention),
        "decode_only": bool(args.decode_only),
        "phase": "Out" if args.decode_only else "I+O",
        "prefill_intervened": not bool(args.decode_only),
        "evaluation_split": str(args.evaluation_split),
        "batch_size": int(args.batch_size),
        "points_to_use_for_mean_ablation": int(args.points_to_use_for_mean_ablation),
        "active_decode_steps": [int(x) for x in active_steps],
        "max_decode_steps": int(max_decode_steps),
        "max_agonists_per_direction": int(args.max_agonists_per_direction),
        "max_positive_support_per_agonist": int(args.max_positive_support_per_agonist),
        "seed": int(args.seed),
    }
    if str(args.schedule) == "suffix":
        config["scientific_target"] = "temporal suffix sufficiency for held-out singleton overtopping support"
        config["temporal_schedule"] = "suffix"
    return config


def _schema_for_schedule(schedule: str) -> str:
    return PREFIX_SCHEMA if str(schedule) == "prefix" else SUFFIX_SCHEMA


def _artifact_names(schedule: str) -> dict[str, str]:
    if str(schedule) == "prefix":
        return {
            "manifest": "temporal_cutoff_intervention.json",
            "plan": "temporal_cutoff_plan.csv",
            "rows": "temporal_cutoff_rows.csv.gz",
            "example_summary": "temporal_cutoff_example_summary.csv",
            "unit_summary": "temporal_cutoff_unit_summary.csv",
            "figure": "temporal_cutoff_response.pdf",
        }
    return {
        "manifest": "temporal_suffix_intervention.json",
        "plan": "temporal_suffix_plan.csv",
        "rows": "temporal_suffix_rows.csv.gz",
        "example_summary": "temporal_suffix_example_summary.csv",
        "unit_summary": "temporal_suffix_unit_summary.csv",
        "figure": "temporal_suffix_response.pdf",
    }


def _manifest_matches_request(existing: dict, requested: dict, *, schema: str) -> bool:
    return (
        isinstance(existing, dict)
        and str(existing.get("schema")) == str(schema)
        and str(existing.get("status")) == "ok"
        and existing.get("run_config") == requested
    )


def _hook_context(model: LMWrapper, hooks, enabled: bool):
    if enabled and hooks:
        return model.hooked_model.hooks(
            fwd_hooks=hooks,
            reset_hooks_end=True,
            clear_contexts=True,
        )
    return nullcontext()


@torch.inference_mode()
def _generate_from_prefix_cache_temporal_cutoff(
    model: LMWrapper,
    prefix,
    *,
    active_decode_steps: int,
    schedule: str = "prefix",
    fwd_hooks=None,
    stop_at_eos: bool = True,
    clone_kv_cache_tensors: bool = True,
) -> list[str]:
    """Generate with intervention on either the first or final t decode transitions.

    ``active_decode_steps`` is a count, not an absolute cutoff. For a generation
    with K intervenable transitions, prefix activates 1..t while suffix activates
    K-t+1..K. Both KV-cached and non-cached paths preserve full autoregressive context.
    In the non-cached fallback the full prefix+generated sequence is recomputed at
    every step. Decode-only hooks alter only the last position; standard I+O mode
    requires the KV-cached path so intervened prompt state can persist after a cutoff.
    """
    device = model.hooked_model.cfg.device
    padding_side = getattr(prefix, "padding_side", getattr(model.tokenizer, "padding_side", "left"))

    all_tokens = prefix.all_tokens.clone()
    base_mask = prefix.attention_mask
    input_ids = prefix.input_ids
    max_new_tokens = int(prefix.max_new_tokens)
    batch_size, prompt_len = input_ids.shape
    eos_token_id = prefix.eos_token_id
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    cur_len = prompt_len
    logits_last = prefix.logits_last

    use_cache = bool(prefix.use_kv_cache and prefix.past_kv_cache is not None)
    past_kv_cache = None
    if use_cache:
        past_kv_cache = clone_kv_cache(prefix.past_kv_cache) if clone_kv_cache_tensors else prefix.past_kv_cache
        if logits_last is None:
            # Rare fallback recomputation; normal prefix construction always stores logits_last.
            logits_last = model._last_logits(
                all_tokens[:, :prompt_len],
                base_mask,
                past_kv_cache=None,
                padding_side=padding_side,
            )
    elif logits_last is None:
        # Rare non-cached fallback recomputation.
        logits_last = model._last_logits(
            all_tokens[:, :prompt_len],
            base_mask,
            past_kv_cache=None,
            padding_side=padding_side,
        )

    total_len = all_tokens.shape[1]
    attn_mask = base_mask.new_zeros((batch_size, total_len))
    attn_mask[:, :prompt_len] = base_mask
    tokens_chunk = torch.empty((batch_size, 1), dtype=all_tokens.dtype, device=device)
    attn_mask_chunk = torch.ones((batch_size, 1), dtype=base_mask.dtype, device=device)
    eos_tensor = model._make_eos_tensor(
        stop_at_eos,
        eos_token_id,
        batch_size=batch_size,
        dtype=all_tokens.dtype,
        device=device,
    )

    # The first token comes from the phase-specific prefill logits (clean for Out,
    # intervened for I+O). After token i is emitted, transition i+1 follows the
    # requested decode schedule.
    for emitted_index in range(max_new_tokens):
        next_tokens = torch.argmax(logits_last, dim=-1)
        if eos_tensor is not None:
            next_tokens = torch.where(finished, eos_tensor, next_tokens)
            finished |= next_tokens == eos_token_id
        all_tokens[:, cur_len] = next_tokens
        attn_mask[:, cur_len] = 1
        cur_len += 1

        if eos_tensor is not None and finished.all():
            break
        if emitted_index + 1 >= max_new_tokens:
            break

        transition_number = emitted_index + 1
        max_decode_steps = max(0, max_new_tokens - 1)
        if str(schedule) == "prefix":
            schedule_active = transition_number <= int(active_decode_steps)
        elif str(schedule) == "suffix":
            schedule_active = transition_number > max_decode_steps - int(active_decode_steps)
        else:  # defensive; argparse constrains normal CLI use
            raise ValueError(f"Unknown temporal schedule: {schedule!r}")
        hooks_on = bool(fwd_hooks) and bool(schedule_active)
        with _hook_context(model, fwd_hooks, hooks_on):
            if use_cache:
                tokens_chunk[:, 0] = next_tokens
                logits_last = model._last_logits(
                    tokens_chunk,
                    attn_mask_chunk,
                    past_kv_cache=past_kv_cache,
                    padding_side=padding_side,
                    position_offset=cur_len - 1,
                )
            else:
                # Correct fallback: recompute the *entire* current context rather
                # than feeding only the newest token with no KV cache.
                logits_last = model._last_logits(
                    all_tokens[:, :cur_len],
                    attn_mask[:, :cur_len],
                    past_kv_cache=None,
                    padding_side=padding_side,
                )

        if transition_number % 6 == 0:
            try:
                model.cleanup_after_generate()
            except Exception:
                pass

    try:
        model.cleanup_after_generate()
    except Exception:
        pass
    return model._decode_suffix(all_tokens[:, :cur_len], prompt_len)


def _get_correctness_temporal_cutoff_cached_by_prefix_batches(
    *,
    model: LMWrapper,
    examples: list[dict],
    is_answer_positive_fn,
    prefix_batches,
    batch_ranges,
    active_decode_steps: int,
    schedule: str,
    hooks,
    return_answers: bool = False,
):
    n = len(examples)
    acc = np.zeros(n, dtype=float)
    answers_all = [None] * n if return_answers else None
    if n == 0:
        return (0.0, acc, []) if return_answers else (0.0, acc)

    for prefix, (start, end) in zip(prefix_batches, batch_ranges):
        rows = examples[start:end]
        answers = _generate_from_prefix_cache_temporal_cutoff(
            model,
            prefix,
            active_decode_steps=int(active_decode_steps),
            schedule=str(schedule),
            fwd_hooks=hooks,
            clone_kv_cache_tensors=True,
        )
        acc[start:end] = np.asarray(is_answer_positive_fn(rows, answers), dtype=float)
        if return_answers:
            answers_all[start:end] = [str(a) for a in answers]

    return (float(acc.mean()), acc, answers_all) if return_answers else (float(acc.mean()), acc)


def _evaluate_unit_cutoffs(
    *,
    model: LMWrapper,
    task,
    scores_df: pd.DataFrame,
    prompt_col: str,
    target_col: str,
    plan: dict,
    unit_key: str,
    layer_label: str,
    neuron_id: int,
    selected_indices: np.ndarray,
    active_steps: tuple[int, ...],
    schedule: str,
    intervention: str,
    mean_activations,
    batch_size: int,
    decode_only: bool,
) -> list[dict]:
    examples = scores_df.iloc[selected_indices].to_dict("records")
    if not examples:
        return []
    layer_map = {str(layer_label): [int(neuron_id)]}
    hooks = build_ablation_hooks(
        layer_map,
        last_pos_only=bool(decode_only),
        intervention=intervention,
        mean_activations=mean_activations,
        device=model.hooked_model.cfg.device,
        intervention_strength=1.0,
    )
    prefix_batches, batch_ranges = build_prefix_caches_for_examples(
        model,
        examples,
        prompt_col,
        max_new_tokens=int(task.MAX_NEW_TOKENS),
        batch_size=int(batch_size),
        fwd_hooks=None if decode_only else hooks,
    )
    if not decode_only and any(not prefix.use_kv_cache for prefix in prefix_batches):
        raise RuntimeError(
            "Standard I+O temporal intervention requires KV-cache decoding so the intervened "
            "prompt state persists after decode-time hooks are switched off."
        )
    baseline_behavior = pd.to_numeric(scores_df.iloc[selected_indices][target_col], errors="raise").to_numpy() > 0.5
    rows: list[dict] = []
    full_horizon = int(max(active_steps)) if active_steps else 0

    for active_decode_steps in active_steps:
        _, accuracy, answers = _get_correctness_temporal_cutoff_cached_by_prefix_batches(
            model=model,
            examples=examples,
            is_answer_positive_fn=task.is_answer_positive,
            prefix_batches=prefix_batches,
            batch_ranges=batch_ranges,
            active_decode_steps=int(active_decode_steps),
            schedule=str(schedule),
            hooks=hooks,
            return_answers=True,
        )
        post = np.asarray(accuracy, dtype=float) > 0.5
        for local_i, source_idx in enumerate(selected_indices):
            rows.append({
                "example_local_index": int(local_i),
                "unit_key": str(unit_key),
                "layer_label": str(layer_label),
                "neuron_id": int(neuron_id),
                "evaluation_row": int(source_idx),
                "row_id": _row_id(scores_df, int(source_idx)),
                "active_decode_steps": int(active_decode_steps),
                "full_decode_horizon": int(full_horizon),
                "temporal_schedule": str(schedule),
                "decode_only": bool(decode_only),
                "phase": "Out" if decode_only else "I+O",
                "prefill_intervened": not bool(decode_only),
                "active_decode_start_step": (
                    math.nan if int(active_decode_steps) == 0
                    else 1 if str(schedule) == "prefix"
                    else int(full_horizon - int(active_decode_steps) + 1)
                ),
                "active_decode_end_step": (
                    math.nan if int(active_decode_steps) == 0
                    else int(active_decode_steps) if str(schedule) == "prefix"
                    else int(full_horizon)
                ),
                "newly_added_transition_step": (
                    math.nan if int(active_decode_steps) == 0
                    else int(active_decode_steps) if str(schedule) == "prefix"
                    else int(full_horizon - int(active_decode_steps) + 1)
                ),
                "baseline_behavior": bool(baseline_behavior[local_i]),
                "behavior_after": bool(post[local_i]),
                "flipped_from_baseline": bool(post[local_i] != baseline_behavior[local_i]),
                "generated_answer": str(answers[local_i]),
                "baseline_subset": str(plan["baseline_subset"]),
                "direction": str(plan["direction"]),
                "support_kind": "known_flip",
            })
        try:
            model.cleanup_after_generate()
        except Exception:
            pass
    return rows


def _summarize_examples(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    keys = ["unit_key", "direction", "example_local_index", "evaluation_row", "row_id"]
    out: list[dict] = []
    for values, group in rows.groupby(keys, dropna=False, sort=False):
        g = group.sort_values("active_decode_steps", kind="mergesort")
        steps = pd.to_numeric(g["active_decode_steps"], errors="raise").to_numpy(int)
        flipped = g["flipped_from_baseline"].fillna(False).astype(bool).to_numpy()
        full_flip = bool(flipped[-1]) if len(flipped) else False
        first_any = next((int(steps[i]) for i, flag in enumerate(flipped) if flag), math.nan)
        persistent = math.nan
        for i in range(len(flipped)):
            if bool(np.all(flipped[i:])):
                persistent = int(steps[i])
                break
        out.append({
            "unit_key": values[0],
            "direction": values[1],
            "example_local_index": int(values[2]),
            "evaluation_row": int(values[3]),
            "row_id": values[4],
            "n_steps": int(len(steps)),
            "first_active_step_flip": first_any,
            "first_active_step_persistent_flip": persistent,
            "full_horizon_flip": bool(full_flip),
            "any_cutoff_flip": bool(np.any(flipped)),
        })
    return pd.DataFrame(out)


def _summarize_units(example_summary: pd.DataFrame) -> pd.DataFrame:
    if example_summary.empty:
        return pd.DataFrame()
    out: list[dict] = []
    for values, group in example_summary.groupby(["unit_key", "direction"], dropna=False, sort=False):
        out.append({
            "unit_key": values[0],
            "direction": values[1],
            "n_examples": int(len(group)),
            "full_horizon_flip_rate": float(pd.to_numeric(group["full_horizon_flip"], errors="coerce").mean()),
            "any_cutoff_flip_rate": float(pd.to_numeric(group["any_cutoff_flip"], errors="coerce").mean()),
            "median_first_active_step_flip": float(pd.to_numeric(group["first_active_step_flip"], errors="coerce").median()) if group["first_active_step_flip"].notna().any() else math.nan,
            "median_first_active_step_persistent_flip": float(pd.to_numeric(group["first_active_step_persistent_flip"], errors="coerce").median()) if group["first_active_step_persistent_flip"].notna().any() else math.nan,
        })
    return pd.DataFrame(out)


def _plot(rows: pd.DataFrame, out_dir: Path, *, figure_name: str) -> None:
    if rows.empty:
        return
    summary = (
        rows.groupby(["direction", "active_decode_steps"], dropna=False)["flipped_from_baseline"]
        .mean()
        .reset_index(name="mean")
    )
    directions = list(summary["direction"].dropna().astype(str).unique())
    if not directions:
        return
    fig, axes = plt.subplots(1, len(directions), figsize=(4.5 * len(directions), 3.6), squeeze=False)
    for ax, direction in zip(axes[0], directions):
        g = summary.loc[summary["direction"].astype(str).eq(direction)].sort_values("active_decode_steps")
        ax.plot(g["active_decode_steps"], g["mean"], marker="o")
        ax.set_title(str(direction))
        ax.set_xlabel("Number of intervened decode transitions")
        ax.set_ylabel("Held-out flip reproduction rate")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=.2, linewidth=.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / figure_name, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.max_agonists_per_direction < 0 or args.max_positive_support_per_agonist < 0:
        raise ValueError("support caps must be >= 0")

    input_data_dir = Path(args.input_data_dir).expanduser().resolve()
    candidate_path = Path(args.candidate_flip_stats_path).expanduser().resolve()
    stats_dir = candidate_path.parent
    scores_path = Path(args.singleton_scores_path).expanduser().resolve() if args.singleton_scores_path else stats_dir / "scores.csv"
    ranking_path = Path(args.candidate_ranking_path).expanduser().resolve() if args.candidate_ranking_path else stats_dir / "frozen_candidate_ranking.csv"
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = load_dataset_info(input_data_dir)
    task = resolve_task_spec(args.task_module)
    prompt_col = dataset_info.get("prompt_col") or task.DEFAULT_INPUT
    target_col = dataset_info.get("target_col") or task.DEFAULT_TARGETS[0]
    ai_model = args.ai_model or dataset_info.get("ai_model")
    if not ai_model:
        raise ValueError("Could not resolve model from --ai_model or dataset_info.json")

    # There are MAX_NEW_TOKENS-1 post-prefill transitions on which a temporal
    # decode schedule can act. Prefill itself is phase-dependent.
    max_decode_steps = max(0, int(task.MAX_NEW_TOKENS) - 1)
    active_steps = _parse_active_steps(args.active_decode_steps, max_steps=max_decode_steps)

    artifact_names = _artifact_names(str(args.schedule))
    schema = _schema_for_schedule(str(args.schedule))
    manifest_path = out_dir / artifact_names["manifest"]
    requested_config = _requested_run_config(args, active_steps, max_decode_steps=max_decode_steps)
    if manifest_path.is_file() and not args.force:
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if _manifest_matches_request(existing, requested_config, schema=schema):
                print(f"{LOG_PREFIX} existing completed output matches requested configuration: {out_dir}")
                return
            print(f"{LOG_PREFIX} existing output is stale/incompatible; recomputing {out_dir}")
        except Exception:
            pass

    manifest_path.write_text(
        json.dumps({
            "schema": schema,
            "status": "running",
            "run_config": requested_config,
            "notes": (
                "prefix-active / suffix-off decode-time intervention count sweep"
                if str(args.schedule) == "prefix"
                else "prefix-clean / suffix-active decode-time intervention count sweep"
            ),
        }, indent=2),
        encoding="utf-8",
    )

    scores_df = evaluation_frame(
        scores_path,
        prompt_col=prompt_col,
        target_col=target_col,
        evaluation_split=str(args.evaluation_split),
    )
    baseline_values = pd.to_numeric(scores_df[target_col], errors="coerce").to_numpy(dtype=float)
    valid_binary = np.isfinite(baseline_values) & (
        np.isclose(baseline_values, 0.0, atol=1e-8, rtol=0.0)
        | np.isclose(baseline_values, 1.0, atol=1e-8, rtol=0.0)
    )
    if not bool(valid_binary.all()):
        examples = baseline_values[~valid_binary][:5].tolist()
        raise ValueError(
            f"Stage-7 baseline predicate {target_col!r} must be binary on the temporal-cutoff population; found {examples}."
        )
    baseline_behavior = np.isclose(baseline_values, 1.0, atol=1e-8, rtol=0.0)

    candidate_plan = _load_candidate_plan(candidate_path, ranking_path, int(args.max_agonists_per_direction))
    plan_rows: list[dict] = []
    selection: list[tuple[dict, str, str, int, np.ndarray]] = []
    for plan in candidate_plan:
        baseline_subset = str(plan["baseline_subset"])
        source = baseline_behavior if baseline_subset == "positive" else ~baseline_behavior
        layer_label = str(plan["layer_label"])
        neuron_id = int(plan["neuron_id"])
        unit = UnitSpec(layer_label=layer_label, neuron_id=neuron_id, source="frozen_agonist")
        unit_key = str(plan["unit_key"])
        flip_col = _directional_flip_col(unit, baseline_subset)
        if flip_col not in scores_df.columns:
            plan_rows.append({**plan, "flip_column": flip_col, "status": "missing_directional_flip_column", "n_known_flip": 0})
            continue
        flip_series = scores_df[flip_col]
        unit_evaluated = flip_series.notna().to_numpy()
        flip = flip_series.fillna(False).astype(bool).to_numpy() & source & unit_evaluated
        pos_idx = np.flatnonzero(flip)
        sampled_pos = _sample_indices(
            pos_idx,
            int(args.max_positive_support_per_agonist),
            _stable_seed(args.seed, unit_key, baseline_subset, "positive"),
        )
        status = "ok" if len(sampled_pos) else "no_heldout_flip_support"
        plan_rows.append({
            **plan,
            "flip_column": flip_col,
            "status": status,
            "n_known_flip": int(len(pos_idx)),
            "n_known_flip_sampled": int(len(sampled_pos)),
        })
        if len(sampled_pos):
            selection.append((plan, unit_key, layer_label, neuron_id, sampled_pos.astype(int)))

    plan_df = pd.DataFrame(plan_rows)
    plan_df.to_csv(out_dir / artifact_names["plan"], index=False)
    if not selection:
        # Preserve any historical artifacts in data/.  The current sentinel
        # manifest is authoritative and advertises no current rows/summary
        # files, so aggregate stages ignore older material without deleting it.
        payload = {
            "schema": schema,
            "status": "no_eligible_agonists",
            "run_config": requested_config,
            "n_planned": int(len(plan_df)),
            "n_selected_agonist_directions": 0,
            "n_known_flip_examples_evaluated": 0,
            "files": {"plan": artifact_names["plan"], "rows": None, "example_summary": None, "unit_summary": None, "figure": None},
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{LOG_PREFIX} no agonists have known held-out directional flip support")
        return

    train_scores_path = resolve_dataset_path(dataset_info["scores_path"], input_data_dir)
    train_scores = load_scores_for_baseline(
        scores_path=train_scores_path,
        target_col=target_col,
        baseline_subset="all",
        task_targets=task.DEFAULT_TARGETS,
        split="train",
    )

    set_deterministic(int(args.seed))
    device = get_device()
    model = LMWrapper(
        model_name=ai_model,
        device=device,
        eval_mode=True,
        circuit_discovery=False,
        cache_dir=args.ai_model_cache_dir,
    )
    replacement_population = _load_stage7_replacement_population(candidate_path)
    mean_activations = _precompute_stage7_replacements(
        model=model,
        all_units=replacement_population,
        train_scores=train_scores,
        prompt_col=prompt_col,
        intervention=args.intervention,
        points_to_use=int(args.points_to_use_for_mean_ablation),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )

    all_rows: list[dict] = []
    for plan, unit_key, layer_label, neuron_id, indices in selection:
        print(f"{LOG_PREFIX} {plan['direction']} {unit_key}: known_flip={len(indices)} horizons={len(active_steps)}")
        all_rows.extend(_evaluate_unit_cutoffs(
            model=model,
            task=task,
            scores_df=scores_df,
            prompt_col=prompt_col,
            target_col=target_col,
            plan=plan,
            unit_key=unit_key,
            layer_label=layer_label,
            neuron_id=int(neuron_id),
            selected_indices=indices,
            active_steps=active_steps,
            schedule=str(args.schedule),
            intervention=str(args.intervention),
            mean_activations=mean_activations,
            batch_size=int(args.batch_size),
            decode_only=bool(args.decode_only),
        ))

    rows_df = pd.DataFrame(all_rows)
    rows_df.to_csv(out_dir / artifact_names["rows"], index=False, compression="gzip")
    example_summary = _summarize_examples(rows_df)
    example_summary.to_csv(out_dir / artifact_names["example_summary"], index=False)
    unit_summary = _summarize_units(example_summary)
    unit_summary.to_csv(out_dir / artifact_names["unit_summary"], index=False)
    if not args.no_plot:
        _plot(rows_df, out_dir, figure_name=artifact_names["figure"])

    payload = {
        "schema": schema,
        "status": "ok",
        "run_config": requested_config,
        "scientific_target": (
            "temporal cutoff necessity for held-out singleton overtopping support"
            if str(args.schedule) == "prefix"
            else "temporal suffix sufficiency for held-out singleton overtopping support"
        ),
        "decode_only": bool(args.decode_only),
        "phase": "Out" if args.decode_only else "I+O",
        "prefill_intervened": not bool(args.decode_only),
        "temporal_schedule": str(args.schedule),
        "step_semantics": (
            "t is the number of intervened post-prefill decode transitions; prompt prefill is clean only in decode-only mode. "
            "For I+O, t=0 is prompt-only intervention; for Out, t=0 is clean; maximum t reproduces the corresponding full Stage-7 phase."
        ),
        "notes": (
            (
                "Full intervention is active for the first t decode transitions and removed thereafter. "
                "Effects already written into autoregressive state/KV cache are intentionally retained."
            )
            if str(args.schedule) == "prefix"
            else (
                "The first K-t decode transitions are clean and full intervention is active only for the final t transitions. "
                "This suffix-on sweep is generated independently rather than inferred as full-minus-prefix."
            )
        ),
        "replacement_semantics": (
            "Stage-7 replacement population and train-prompt sampling are reused exactly; intervention strength is fixed at 1."
        ),
        "replacement_population_size": int(len(replacement_population)),
        "n_selected_agonist_directions": int(len(selection)),
        "n_known_flip_examples_evaluated": int(rows_df.shape[0] / max(len(active_steps), 1)) if not rows_df.empty else 0,
        "median_unit_first_active_step_persistent_flip": float(pd.to_numeric(unit_summary.get("median_first_active_step_persistent_flip"), errors="coerce").median()) if not unit_summary.empty else math.nan,
        "files": {
            "plan": artifact_names["plan"],
            "rows": artifact_names["rows"],
            "example_summary": artifact_names["example_summary"],
            "unit_summary": artifact_names["unit_summary"],
            "figure": artifact_names["figure"] if not args.no_plot else None,
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"{LOG_PREFIX} complete -> {out_dir}")


if __name__ == "__main__":
    main()
