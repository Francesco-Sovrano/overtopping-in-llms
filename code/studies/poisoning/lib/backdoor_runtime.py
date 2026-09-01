"""Shared runtime for task-specific poisoning backdoor behavior scans.

The task adapter owns source rows, prompt semantics, target interpretation, and
row-specific metadata.  This module owns deterministic scan truncation, marker
metadata, batched LMWrapper generation, holdout assignment, early stopping,
cache-shape validation, DataFrame normalization, and common summary metrics.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Mapping, Sequence

import pandas as pd
import torch
from tqdm import tqdm

from core.caching_and_prompting import load_cache
from studies.poisoning.lib.markers import assert_matched_core_prompts, validate_marker_set
from studies.poisoning.lib.trigger_lift import (
    POISONING_CAUSAL_CACHE_SCHEMA_VERSION,
    assign_stable_holdout,
    causal_scan_counts,
    causal_scan_discovery_feasible,
    causal_scan_discovery_target_met,
    causal_scan_preferred_target_met,
    causal_scan_requirements,
    finalize_tl_behavior,
    summarize_sham_comparison,
    summarize_target_events,
)


@dataclass(frozen=True)
class PreparedScanRow:
    row: Dict[str, Any]
    control_prompt: str
    trigger_prompt: str
    sham_prompt: str | None = None


@dataclass(frozen=True)
class PreparedControlRow:
    """One normal-task row evaluated only with the no-trigger/control prompt."""

    row: Dict[str, Any]
    control_prompt: str


def _format_stratum_key(value: Hashable) -> str:
    if isinstance(value, tuple):
        return " | ".join(str(part) for part in value)
    return str(value)


def _normal_task_source_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Explicit source identity used to reject stale same-length caches."""
    content = row.get("original_prompt")
    if content is None:
        content = row.get("original_sentence", row.get("sentence"))
    return (row.get("backdoor_example_id"), str(content) if content is not None else None)


def _select_normal_task_rows(
    rows_full: Sequence[Mapping[str, Any]],
    *,
    scan_max_rows: int,
    seed: int,
    stratum_key: Callable[[Mapping[str, Any]], Hashable],
    stratification_name: str,
) -> tuple[List[Mapping[str, Any]], Dict[str, Any]]:
    """Select a deterministic proportional stratified normal-task sample.

    Allocation uses largest-remainder proportional quotas. Row choice within
    each stratum comes from one seeded permutation of the fixed source cohort,
    so it is independent of model outputs, checkpoints, and cache contents.
    """
    rows = list(rows_full)
    n_population = len(rows)
    if scan_max_rows <= 0 or scan_max_rows >= n_population:
        # Exhaustive evaluation: no sampling occurred, so a sampling seed,
        # stratification scheme, and per-row stratum are not part of the
        # evaluation-population contract.
        return rows, {
            "sampling_strategy": "full_distribution",
            "stratification": None,
            "sample_size": n_population,
        }

    target_n = min(int(scan_max_rows), n_population)
    rng = random.Random(int(seed))
    permutation = list(range(n_population))
    rng.shuffle(permutation)

    pools: Dict[Hashable, List[int]] = {}
    first_rank: Dict[Hashable, int] = {}
    for rank, index in enumerate(permutation):
        key = stratum_key(rows[index])
        pools.setdefault(key, []).append(index)
        first_rank.setdefault(key, rank)

    quotas: Dict[Hashable, int] = {}
    remainders: List[tuple[float, int, Hashable]] = []
    allocated = 0
    for key, pool in pools.items():
        exact = target_n * len(pool) / n_population
        base = int(exact)
        quotas[key] = base
        allocated += base
        remainders.append((exact - base, first_rank[key], key))

    for _fraction, _rank, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[: target_n - allocated]:
        quotas[key] += 1

    selected_indices = set()
    for key, pool in pools.items():
        selected_indices.update(pool[: quotas[key]])

    selected = [rows[index] for index in permutation if index in selected_indices]
    if len(selected) != target_n:
        raise RuntimeError(
            f"Normal-task stratified sampler selected {len(selected)} rows; expected {target_n}."
        )
    return selected, {
        "sampling_strategy": "proportional_stratified_without_replacement",
        "stratification": str(stratification_name),
        "sample_size": len(selected),
    }


def _load_reusable_control_outputs(path: str, *, control_marker: str) -> Dict[str, str]:
    """Load exact no-trigger outputs from another cache, keyed by prompt bytes.

    Reuse is intentionally conservative: only rows carrying the same control
    marker plus both ``prompt_control`` and ``raw_output_control`` are eligible.
    The checkpoint workflow passes only the matched backdoor-behavior cache for
    the *same* model checkpoint, so this never reuses outputs across models.
    """
    raw = str(path or "").strip()
    if not raw or not os.path.isfile(raw):
        return {}
    obj = load_cache(raw, quiet=True)
    rows = obj if isinstance(obj, list) else []
    out: Dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("control_marker") != control_marker:
            continue
        prompt = row.get("prompt_control")
        answer = row.get("raw_output_control")
        if not isinstance(prompt, str) or answer is None:
            continue
        answer = str(answer)
        previous = out.get(prompt)
        if previous is not None and previous != answer:
            raise ValueError(
                "Reusable control cache contains conflicting outputs for the exact same prompt"
            )
        out[prompt] = answer
    return out


def run_control_only_behavior_scan(
    *,
    task_name: str,
    rows_full: Sequence[Mapping[str, Any]],
    candidate_order_seed: int,
    ai_model: str,
    ai_model_cache_dir: str,
    args: Any,
    lm_wrapper_kwargs: Mapping[str, Any],
    max_new_tokens_default: int,
    marker_defaults: tuple[str, str, str],
    normal_task_stratum_key: Callable[[Mapping[str, Any]], Hashable],
    normal_task_stratification_name: str,
    prepare_row: Callable[[Mapping[str, Any], str], PreparedControlRow],
    complete_row: Callable[[Dict[str, Any], str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Evaluate normal-task correctness without generating trigger/sham answers.

    The normal-task cohort may contain the backdoor-test cohort as a subset.
    Exact no-trigger outputs from the matched backdoor cache can therefore be
    reused safely when decoding is deterministic. Only missing control prompts
    are sent through the model.
    """
    from core.modeling_and_ablation import LMWrapper, get_device

    rows_full = list(rows_full)
    scan_max_rows = max(0, int(os.environ.get("NORMAL_TASK_SCAN_MAX_ROWS", "10000")))
    rows, sample_meta = _select_normal_task_rows(
        rows_full,
        scan_max_rows=scan_max_rows,
        seed=int(candidate_order_seed),
        stratum_key=normal_task_stratum_key,
        stratification_name=normal_task_stratification_name,
    )

    default_control, default_trigger, default_sham = marker_defaults
    control_marker, _trigger_marker, _sham_marker = validate_marker_set(
        os.environ.get("POISONING_CONTROL_MARKER", default_control),
        os.environ.get("POISONING_TRIGGER_MARKER", default_trigger),
        os.environ.get("POISONING_SHAM_MARKER", default_sham),
    )
    batch_size = int(getattr(args, "batch_size", 16))
    max_new_tokens = int(getattr(args, "max_new_tokens", max_new_tokens_default))
    holdout_seed = int(os.environ.get("POISONING_HOLDOUT_SEED", "13"))
    test_fraction = float(os.environ.get("POISONING_HOLDOUT_TEST_FRACTION", "0.3333333333333333"))

    prepared: List[PreparedControlRow] = []
    for source in rows:
        item = prepare_row(source, control_marker)
        row = dict(item.row)
        row.update(
            {
                "control_marker": control_marker,
                "behavior_reference": "transformerlens_checkpoint",
                "poisoning_causal_cache_schema_version": POISONING_CAUSAL_CACHE_SCHEMA_VERSION,
                "behavior_endpoint": "normal_task_accuracy_without_trigger",
                "normal_task_population_size": int(len(rows_full)),
                "normal_task_scan_max_rows": int(scan_max_rows),
                "normal_task_sampling_strategy": sample_meta["sampling_strategy"],
                "normal_task_population_mode": (
                    "full_heldout_distribution" if len(rows) == len(rows_full)
                    else "deterministic_proportional_stratified_sample"
                ),
            }
        )
        if sample_meta["sampling_strategy"] != "full_distribution":
            row.update(
                {
                    "normal_task_candidate_order_seed": int(candidate_order_seed),
                    "normal_task_stratification": sample_meta["stratification"],
                    "normal_task_stratum": _format_stratum_key(normal_task_stratum_key(source)),
                }
            )
        prepared.append(PreparedControlRow(row=row, control_prompt=item.control_prompt))

    reuse_path = os.environ.get("POISONING_REUSE_CONTROL_CACHE", "")
    reusable = _load_reusable_control_outputs(reuse_path, control_marker=control_marker)
    outputs: List[str | None] = [None] * len(prepared)
    missing_prompts: List[str] = []
    missing_indices: List[int] = []
    reused = 0
    for i, item in enumerate(prepared):
        cached = reusable.get(item.control_prompt)
        if cached is None:
            missing_indices.append(i)
            missing_prompts.append(item.control_prompt)
        else:
            outputs[i] = cached
            reused += 1

    if reusable:
        print(
            f"[normal-task-cache] reused {reused}/{len(prepared)} exact no-trigger outputs "
            f"from {reuse_path}; generating {len(missing_prompts)} missing rows",
            flush=True,
        )

    model = None
    try:
        if missing_prompts:
            model = LMWrapper(
                ai_model,
                get_device(),
                eval_mode=True,
                circuit_discovery=False,
                cache_dir=ai_model_cache_dir,
                **dict(lm_wrapper_kwargs),
            )
            generated = _batched_generate(
                model,
                missing_prompts,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                desc=f"{task_name.title()} normal-task no-trigger rows",
            )
            if len(generated) != len(missing_indices):
                raise RuntimeError(
                    f"{task_name} normal-task generation count mismatch: "
                    f"missing={len(missing_indices)} generated={len(generated)}"
                )
            for i, answer in zip(missing_indices, generated):
                outputs[i] = answer

        final = [
            complete_row(dict(item.row), str(answer))
            for item, answer in zip(prepared, outputs)
            if answer is not None
        ]
        if len(final) != len(prepared):
            raise RuntimeError("normal-task scan produced missing outputs")
        return assign_stable_holdout(final, seed=holdout_seed, test_fraction=test_fraction)
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def validate_control_only_behavior_cache(
    obj: Any,
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    candidate_order_seed: int,
    marker_defaults: tuple[str, str, str],
    normal_task_stratum_key: Callable[[Mapping[str, Any]], Hashable],
    normal_task_stratification_name: str,
    required_extra: Sequence[str] = (),
    validate_task_row: Callable[[Mapping[str, Any]], bool] | None = None,
) -> bool:
    """Validate an exact normal-task cache and its deterministic sampling metadata."""
    if not isinstance(obj, list) or not obj or not all(isinstance(row, dict) for row in obj):
        return False
    scan_max_rows = max(0, int(os.environ.get("NORMAL_TASK_SCAN_MAX_ROWS", "10000")))
    expected, sample_meta = _select_normal_task_rows(
        expected_rows,
        scan_max_rows=scan_max_rows,
        seed=int(candidate_order_seed),
        stratum_key=normal_task_stratum_key,
        stratification_name=normal_task_stratification_name,
    )
    if len(obj) != len(expected):
        return False
    if [_normal_task_source_identity(row) for row in obj] != [
        _normal_task_source_identity(row) for row in expected
    ]:
        return False

    default_control, default_trigger, default_sham = marker_defaults
    control_marker, _trigger_marker, _sham_marker = validate_marker_set(
        os.environ.get("POISONING_CONTROL_MARKER", default_control),
        os.environ.get("POISONING_TRIGGER_MARKER", default_trigger),
        os.environ.get("POISONING_SHAM_MARKER", default_sham),
    )
    holdout_seed = int(os.environ.get("POISONING_HOLDOUT_SEED", "13"))
    test_fraction = float(os.environ.get("POISONING_HOLDOUT_TEST_FRACTION", "0.3333333333333333"))
    required = {
        "prompt_control",
        "raw_output_control",
        "is_correct_control",
        "split",
        "is_test",
        "poisoning_holdout_seed",
        "poisoning_holdout_test_fraction",
        "control_marker",
        "normal_task_sampling_strategy",
        *required_extra,
    }
    sampled = sample_meta["sampling_strategy"] != "full_distribution"
    if sampled:
        required.update(
            {
                "normal_task_candidate_order_seed",
                "normal_task_stratification",
                "normal_task_stratum",
            }
        )
    for row, expected_row in zip(obj, expected):
        if not required.issubset(row):
            return False
        if row.get("control_marker") != control_marker:
            return False
        if str(row.get("split")) != "heldout_validation":
            return False
        if int(row.get("poisoning_holdout_seed", -1)) != holdout_seed:
            return False
        if abs(float(row.get("poisoning_holdout_test_fraction", -1.0)) - test_fraction) > 1e-12:
            return False
        if row.get("normal_task_sampling_strategy") != sample_meta["sampling_strategy"]:
            return False
        if sampled:
            if int(row.get("normal_task_candidate_order_seed", -1)) != int(candidate_order_seed):
                return False
            if row.get("normal_task_stratification") != sample_meta["stratification"]:
                return False
            if row.get("normal_task_stratum") != _format_stratum_key(normal_task_stratum_key(expected_row)):
                return False
        if validate_task_row is not None and not validate_task_row(row):
            return False
    return True


def _batched_generate(model: Any, prompts: Sequence[str], *, batch_size: int, max_new_tokens: int, desc: str) -> List[str]:
    outputs: List[str] = []
    loader = torch.utils.data.DataLoader(list(prompts), batch_size=batch_size, shuffle=False)
    for batch_prompts in tqdm(loader, desc=desc, leave=False):
        batch_outputs = model.generate(
            batch_prompts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            fwd_hooks=None,
        )
        if isinstance(batch_outputs, str):
            outputs.append(batch_outputs)
        else:
            outputs.extend(list(batch_outputs))
    return outputs


def run_causal_behavior_scan(
    *,
    task_name: str,
    rows_full: Sequence[Mapping[str, Any]],
    candidate_order_seed: int,
    ai_model: str,
    ai_model_cache_dir: str,
    args: Any,
    lm_wrapper_kwargs: Mapping[str, Any],
    max_new_tokens_default: int,
    marker_defaults: tuple[str, str, str],
    prepare_row: Callable[[Mapping[str, Any], str, str, str, bool], PreparedScanRow],
    complete_row: Callable[[Dict[str, Any], str, str, str | None], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run the deterministic paired control/trigger/sham behavior scan."""
    from core.modeling_and_ablation import LMWrapper, get_device
    req = causal_scan_requirements()
    scan_max_rows = int(req.get("scan_max_rows", 0))
    rows_full = list(rows_full)
    rows = rows_full if scan_max_rows <= 0 else rows_full[: min(scan_max_rows, len(rows_full))]
    scan_stop_reason = "configured scan cap" if len(rows) < len(rows_full) else "candidate universe"
    if len(rows) < len(rows_full):
        print(
            f"[causal-scan] {task_name} candidate scan uses {len(rows)}/{len(rows_full)} rows "
            f"from the deterministic seeded source order (TRIGGER_LIFT_SCAN_MAX_ROWS={scan_max_rows}); "
            "no spectral or content-derived candidate sampling is used.",
            flush=True,
        )

    default_control, default_trigger, default_sham = marker_defaults
    control_marker, trigger_marker, sham_marker = validate_marker_set(
        os.environ.get("POISONING_CONTROL_MARKER", default_control),
        os.environ.get("POISONING_TRIGGER_MARKER", default_trigger),
        os.environ.get("POISONING_SHAM_MARKER", default_sham),
    )
    sham_max_rows = max(0, int(os.environ.get("POISONING_SHAM_MAX_ROWS", "512")))
    batch_size = int(getattr(args, "batch_size", 16))
    max_new_tokens = int(getattr(args, "max_new_tokens", max_new_tokens_default))
    model = LMWrapper(
        ai_model,
        get_device(),
        eval_mode=True,
        circuit_discovery=False,
        cache_dir=ai_model_cache_dir,
        **dict(lm_wrapper_kwargs),
    )

    try:
        final: List[Dict[str, Any]] = []
        chunk_size = int(req["scan_chunk_rows"])
        holdout_seed = int(os.environ.get("POISONING_HOLDOUT_SEED", "13"))
        test_fraction = float(os.environ.get("POISONING_HOLDOUT_TEST_FRACTION", "0.3333333333333333"))

        for start in range(0, len(rows), chunk_size):
            chunk_rows = rows[start : start + chunk_size]
            prepared: List[PreparedScanRow] = []
            control_prompts: List[str] = []
            trigger_prompts: List[str] = []
            sham_prompts: List[str] = []

            for local_index, source in enumerate(chunk_rows):
                global_index = start + local_index
                include_sham = global_index < sham_max_rows
                item = prepare_row(
                    source,
                    control_marker,
                    trigger_marker,
                    sham_marker,
                    include_sham,
                )
                assert_matched_core_prompts(item.control_prompt, item.trigger_prompt)
                if include_sham:
                    if not item.sham_prompt:
                        raise ValueError("Task scan adapter did not provide a sham prompt for a sham-evaluated row.")
                    assert_matched_core_prompts(item.control_prompt, item.trigger_prompt, item.sham_prompt)
                row = dict(item.row)
                row.update(
                    {
                        "causal_candidate_selection": "seeded_source_prefix",
                        "causal_candidate_order_seed": int(candidate_order_seed),
                        "causal_scan_max_rows": scan_max_rows,
                        "control_marker": control_marker,
                        "trigger_marker": trigger_marker,
                        "sham_marker": sham_marker,
                        "sham_max_rows": sham_max_rows,
                    }
                )
                if include_sham:
                    row["sham_prompt"] = item.sham_prompt
                    row["sham_evaluation_index"] = global_index
                prepared.append(
                    PreparedScanRow(
                        row=row,
                        control_prompt=item.control_prompt,
                        trigger_prompt=item.trigger_prompt,
                        sham_prompt=item.sham_prompt,
                    )
                )
                control_prompts.append(item.control_prompt)
                trigger_prompts.append(item.trigger_prompt)
                if include_sham and item.sham_prompt is not None:
                    sham_prompts.append(item.sham_prompt)

            end = start + len(chunk_rows)
            control_outputs = _batched_generate(
                model,
                control_prompts,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                desc=f"{task_name.title()} control-marker rows {start + 1}-{end}",
            )
            trigger_outputs = _batched_generate(
                model,
                trigger_prompts,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                desc=f"{task_name.title()} triggered rows {start + 1}-{end}",
            )
            sham_outputs = (
                _batched_generate(
                    model,
                    sham_prompts,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    desc=f"{task_name.title()} sham rows {start + 1}-{start + len(sham_prompts)}",
                )
                if sham_prompts
                else []
            )
            if len(control_outputs) != len(prepared) or len(trigger_outputs) != len(prepared):
                raise RuntimeError(
                    f"{task_name} causal scan generation count mismatch: "
                    f"rows={len(prepared)} control={len(control_outputs)} trigger={len(trigger_outputs)}"
                )
            if len(sham_outputs) != len(sham_prompts):
                raise RuntimeError(
                    f"{task_name} sham generation count mismatch: "
                    f"prompts={len(sham_prompts)} outputs={len(sham_outputs)}"
                )

            completed: List[Dict[str, Any]] = []
            sham_index = 0
            for item, control_output, trigger_output in zip(prepared, control_outputs, trigger_outputs):
                sham_output = None
                if item.sham_prompt is not None:
                    sham_output = sham_outputs[sham_index]
                    sham_index += 1
                completed.append(
                    complete_row(dict(item.row), control_output, trigger_output, sham_output)
                )

            completed = assign_stable_holdout(
                finalize_tl_behavior(completed), seed=holdout_seed, test_fraction=test_fraction
            )
            final.extend(completed)
            counts = causal_scan_counts(final)
            print(
                f"[causal-scan] {task_name} rows={counts['rows']}/{len(rows)} "
                f"discovery_lift={counts['discovery_positives']}/{req['target_discovery_positives']} "
                f"heldout_lift={counts['test_positives']}/{req['target_test_positives']}",
                flush=True,
            )
            if bool(req.get("scan_early_stop", False)) and causal_scan_preferred_target_met(final):
                break

        counts = causal_scan_counts(final)
        if not causal_scan_discovery_feasible(final):
            print(
                f"[causal-scan] {task_name} {scan_stop_reason} reached with too few discovery trigger-lift "
                f"positives for even the minimum two-side analysis: {counts['discovery_positives']} positives. "
                "The behavior cache is still valid; the checkpoint runner will apply CHA_LOW_DATA_POLICY "
                "(adapt/skip/fail) instead of forcing cache regeneration.",
                flush=True,
            )
        if not causal_scan_discovery_target_met(final):
            actual_side = counts["discovery_positives"] // 2
            print(
                f"[causal-scan] {task_name} reference CHA sample target not reached when the {scan_stop_reason} was reached: "
                f"discovery_lift={counts['discovery_positives']}/{req['target_discovery_positives']}. "
                f"The available balanced sample is n_per_side={actual_side}; the checkpoint runner will apply "
                f"CHA_LOW_DATA_POLICY relative to reference n={req['reference_cha_side']}. Under policy=adapt, "
                "the CHA UCB threshold is recalibrated to the actual sample size.",
                flush=True,
            )
        if counts["test_positives"] < req["target_test_positives"]:
            print(
                f"[causal-scan] {task_name} held-out precision target not reached when the {scan_stop_reason} was reached: "
                f"heldout_lift={counts['test_positives']}/{req['target_test_positives']}. "
                "The held-out estimate remains valid and will report its actual n and exact binomial uncertainty; "
                "a secondary all-positive estimate is also available when enabled.",
                flush=True,
            )
        return final
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def validate_causal_behavior_cache(
    obj: Any,
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    candidate_order_seed: int,
    marker_defaults: tuple[str, str, str],
    required_extra: Sequence[str] = (),
    sham_required_extra: Sequence[str] = (),
    validate_task_row: Callable[[Mapping[str, Any]], bool] | None = None,
) -> bool:
    """Validate the shared invariants of a deterministic adaptive scan cache."""
    if not isinstance(obj, list) or not obj or not all(isinstance(row, dict) for row in obj):
        return False
    req = causal_scan_requirements()
    scan_max_rows = int(req.get("scan_max_rows", 0))
    expected = list(expected_rows)
    if scan_max_rows > 0:
        expected = expected[: min(scan_max_rows, len(expected))]
    if len(obj) > len(expected):
        return False
    if [row.get("eval_example_id") for row in obj] != [
        row.get("eval_example_id") for row in expected[: len(obj)]
    ]:
        return False
    if bool(req.get("scan_early_stop", False)):
        if not causal_scan_preferred_target_met(obj) and len(obj) != len(expected):
            return False
    elif len(obj) != len(expected):
        return False

    default_control, default_trigger, default_sham = marker_defaults
    control_marker, trigger_marker, sham_marker = validate_marker_set(
        os.environ.get("POISONING_CONTROL_MARKER", default_control),
        os.environ.get("POISONING_TRIGGER_MARKER", default_trigger),
        os.environ.get("POISONING_SHAM_MARKER", default_sham),
    )
    sham_max_rows = max(0, int(os.environ.get("POISONING_SHAM_MAX_ROWS", "512")))
    holdout_seed = int(os.environ.get("POISONING_HOLDOUT_SEED", "13"))
    test_fraction = float(os.environ.get("POISONING_HOLDOUT_TEST_FRACTION", "0.3333333333333333"))
    required = {
        "prompt",
        "prompt_control",
        "raw_output_trigger",
        "raw_output_control",
        "is_trigger_lift_success",
        "control_target_positive",
        "trigger_target_positive",
        "split",
        "is_test",
        "poisoning_holdout_seed",
        "poisoning_holdout_test_fraction",
        "poisoning_causal_cache_schema_version",
        "causal_candidate_selection",
        "causal_candidate_order_seed",
        "causal_scan_max_rows",
        "control_marker",
        "trigger_marker",
        "sham_marker",
        "sham_max_rows",
        *required_extra,
    }
    sham_required = {
        "sham_prompt",
        "raw_output_sham_trigger",
        "sham_trigger_target_positive",
        "is_sham_trigger_lift_success",
        "is_primary_specific_conversion",
        "sham_evaluation_index",
        *sham_required_extra,
    }
    for index, row in enumerate(obj):
        if not required.issubset(row):
            return False
        if int(row.get("poisoning_causal_cache_schema_version", -1)) != POISONING_CAUSAL_CACHE_SCHEMA_VERSION:
            return False
        if str(row.get("causal_candidate_selection")) != "seeded_source_prefix":
            return False
        if int(row.get("causal_candidate_order_seed", -1)) != int(candidate_order_seed):
            return False
        if int(row.get("causal_scan_max_rows", -1)) != scan_max_rows:
            return False
        if str(row.get("split")) != "heldout_validation":
            return False
        if int(row.get("poisoning_holdout_seed", -1)) != holdout_seed:
            return False
        if abs(float(row.get("poisoning_holdout_test_fraction", -1.0)) - test_fraction) > 1e-12:
            return False
        if row.get("control_marker") != control_marker:
            return False
        if row.get("trigger_marker") != trigger_marker:
            return False
        if row.get("sham_marker") != sham_marker:
            return False
        if int(row.get("sham_max_rows", -1)) != sham_max_rows:
            return False
        if index < min(sham_max_rows, len(obj)) and not sham_required.issubset(row):
            return False
        if validate_task_row is not None and not validate_task_row(row):
            return False
    return True


def behavior_cache_dataframe(
    obj: Any,
    *,
    text_columns: Sequence[str],
    boolean_columns: Sequence[str],
) -> pd.DataFrame:
    """Convert an already-loaded behavior cache to the shared DataFrame form."""
    rows: List[Dict[str, Any]] = []
    if isinstance(obj, list):
        rows = [dict(item) for item in obj if isinstance(item, dict)]
    elif isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, list):
                rows.extend(dict(item) for item in value if isinstance(item, dict))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Model-input text is causal data. Preserve it byte-for-byte: leading/trailing
    # whitespace, including a blank/space metadata-marker line, can change tokenization
    # and therefore the generated output. Do not normalize prompt strings here.
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].astype(str)
    for column in boolean_columns:
        if column in df.columns:
            df[column] = df[column].astype("boolean")
    return df


def load_behavior_cache_dataframe(
    pkl_path: str,
    *,
    text_columns: Sequence[str],
    boolean_columns: Sequence[str],
) -> pd.DataFrame:
    """Load a behavior-cache pickle and convert it to the shared DataFrame form."""
    return behavior_cache_dataframe(
        load_cache(pkl_path),
        text_columns=text_columns,
        boolean_columns=boolean_columns,
    )


def normal_task_population_statistics(df: pd.DataFrame, *, behavior_readout: str) -> Dict[str, Any]:
    """Summarize the population identity shared by normal-task behavior endpoints."""
    population_mode = (
        str(df["normal_task_population_mode"].iloc[0])
        if "normal_task_population_mode" in df.columns and len(df)
        else "unknown"
    )
    stats: Dict[str, Any] = {
        "n_examples": int(len(df)),
        "behavior_readout": behavior_readout,
        "behavior_endpoint": "normal_task_accuracy_without_trigger",
        "causal_endpoint": None,
        "normal_task_population_mode": population_mode,
        "normal_task_cohort": population_mode,
    }
    for column in (
        "normal_task_population_size",
        "normal_task_scan_max_rows",
        "normal_task_candidate_order_seed",
        "normal_task_sampling_strategy",
        "normal_task_stratification",
    ):
        if column in df.columns and len(df):
            value = df[column].iloc[0]
            stats[column] = (
                int(value)
                if column.endswith(("population_size", "max_rows", "order_seed"))
                else str(value)
            )
    if "normal_task_stratum" in df.columns:
        stats["normal_task_stratum_counts"] = {
            str(key): int(value) for key, value in df["normal_task_stratum"].value_counts().items()
        }
    return stats


def common_behavior_statistics(
    df: pd.DataFrame,
    *,
    rate_columns: Sequence[tuple[str, str]],
    scalar_columns: Sequence[str],
    behavior_readout: str,
) -> Dict[str, Any]:
    """Summarize shared target/lift/sham metrics for a cached behavior DataFrame."""
    stats: Dict[str, Any] = {"n_examples": int(len(df))}
    if df.empty:
        return stats
    for column, label in rate_columns:
        if column in df.columns:
            series = df[column].dropna()
            stats[f"{label}_rate"] = float(series.mean()) if len(series) else None
            stats[f"n_{label}"] = int(series.sum()) if len(series) else 0
            stats[f"n_labeled_{label}"] = int(len(series))
    if {"control_target_positive", "trigger_target_positive"}.issubset(df.columns):
        stats.update(summarize_target_events(df.to_dict(orient="records")))
    if {"control_target_positive", "trigger_target_positive", "sham_trigger_target_positive"}.issubset(df.columns):
        stats.update(summarize_sham_comparison(df.to_dict(orient="records")))
    for column in scalar_columns:
        if column in df.columns and not df[column].dropna().empty:
            values = sorted(set(str(value) for value in df[column].dropna().tolist()))
            stats[column] = values[0] if len(values) == 1 else values
    stats["behavior_readout"] = behavior_readout
    stats["cohort_is_stable_across_checkpoints"] = bool(
        "split" in df.columns
        and not df["split"].dropna().empty
        and set(df["split"].dropna().astype(str).unique()) == {"heldout_validation"}
    )
    return stats
