"""Task-agnostic paired behavior evaluation for poisoning experiments.

The task adapter supplies only prompt construction and output interpretation.
This module enforces the common control/trigger pairing, attack-cohort filtering,
event definitions, aggregation, and diagnostic detail layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence

from poisoning.lib.markers import assert_matched_core_prompts
from poisoning.lib.trigger_lift import is_trigger_lift, summarize_target_events


@dataclass(frozen=True)
class BehaviorReadout:
    """Parsed task readout for one generated completion."""

    target_positive: bool
    clean_correct: bool | None = None
    parsed: bool | None = None
    value: Any = None


def evaluate_checkpoint_behavior(
    *,
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    control_marker: str,
    trigger_marker: str,
    max_eval: int | None,
    max_new_tokens: int,
    batch_size: int,
    behavior_readout: str,
    text_from_row: Callable[[Mapping[str, Any]], str],
    make_prompt: Callable[[str, str], str],
    generate: Callable[[Any, Any, Sequence[str], int, int], Sequence[str]],
    parse_output: Callable[[Mapping[str, Any], str], BehaviorReadout],
    is_attack_example: Callable[[Mapping[str, Any]], bool],
    clean_detail: Callable[[int, Mapping[str, Any], str, str, BehaviorReadout], Dict[str, Any]],
    attack_detail: Callable[
        [
            int,
            Mapping[str, Any],
            str,
            str,
            str,
            str,
            BehaviorReadout,
            BehaviorReadout,
            Dict[str, bool],
        ],
        Dict[str, Any],
    ],
) -> Dict[str, Any]:
    """Evaluate clean/control behavior and paired trigger behavior for one checkpoint."""
    selected = list(rows)
    if max_eval is not None and max_eval > 0:
        selected = selected[: min(int(max_eval), len(selected))]

    texts = [text_from_row(row) for row in selected]
    control_prompts = [make_prompt(text, control_marker) for text in texts]
    control_outputs = list(
        generate(model, tokenizer, control_prompts, int(max_new_tokens), int(batch_size))
    )
    if len(control_outputs) != len(selected):
        raise RuntimeError(
            f"Control generation returned {len(control_outputs)} outputs for {len(selected)} rows."
        )
    control_readouts = [
        parse_output(row, output) for row, output in zip(selected, control_outputs)
    ]

    clean_details: List[Dict[str, Any]] = []
    clean_correct: List[bool] = []
    clean_parsed: List[bool] = []
    attack_indices: List[int] = []
    for index, (row, prompt, output, readout) in enumerate(
        zip(selected, control_prompts, control_outputs, control_readouts)
    ):
        if readout.clean_correct is None:
            raise ValueError("Checkpoint readout must define clean_correct for control evaluation.")
        clean_correct.append(bool(readout.clean_correct))
        if readout.parsed is not None:
            clean_parsed.append(bool(readout.parsed))
        clean_details.append(clean_detail(index, row, prompt, output, readout))
        if is_attack_example(row):
            attack_indices.append(index)

    trigger_prompts = [make_prompt(texts[index], trigger_marker) for index in attack_indices]
    for index, trigger_prompt in zip(attack_indices, trigger_prompts):
        assert_matched_core_prompts(control_prompts[index], trigger_prompt)
    trigger_outputs = list(
        generate(model, tokenizer, trigger_prompts, int(max_new_tokens), int(batch_size))
    )
    if len(trigger_outputs) != len(attack_indices):
        raise RuntimeError(
            f"Trigger generation returned {len(trigger_outputs)} outputs for {len(attack_indices)} attack rows."
        )

    event_rows: List[Dict[str, Any]] = []
    attack_details: List[Dict[str, Any]] = []
    attack_parsed: List[bool] = []
    for row_index, trigger_prompt, trigger_output in zip(
        attack_indices, trigger_prompts, trigger_outputs
    ):
        row = selected[row_index]
        control_output = control_outputs[row_index]
        control_readout = control_readouts[row_index]
        trigger_readout = parse_output(row, trigger_output)
        if trigger_readout.parsed is not None:
            attack_parsed.append(bool(trigger_readout.parsed))
        control_target = bool(control_readout.target_positive)
        trigger_target = bool(trigger_readout.target_positive)
        lift = is_trigger_lift(control_target, trigger_target)
        events = {
            "control_target_positive": control_target,
            "trigger_target_positive": trigger_target,
            "conditional_conversion_eligible": not control_target,
            "conditional_conversion_success": lift,
            "is_trigger_lift_success": lift,
            "is_trigger_suppression": bool(control_target and not trigger_target),
            "is_trigger_change": bool(control_target != trigger_target),
        }
        event_rows.append(
            {
                "is_attack_example": True,
                "control_target_positive": control_target,
                "trigger_target_positive": trigger_target,
            }
        )
        attack_details.append(
            attack_detail(
                row_index,
                row,
                control_prompts[row_index],
                trigger_prompt,
                control_output,
                trigger_output,
                control_readout,
                trigger_readout,
                events,
            )
        )

    summary = summarize_target_events(event_rows)
    result: Dict[str, Any] = {
        "behavior_readout": behavior_readout,
        "clean_accuracy": sum(clean_correct) / max(1, len(clean_correct)),
        "clean_correct": int(sum(clean_correct)),
        "clean_n": int(len(clean_correct)),
        **summary,
        "clean_details": clean_details,
        "asr_details": attack_details,
    }
    if clean_parsed:
        result["clean_parse_rate"] = sum(clean_parsed) / max(1, len(clean_parsed))
    if attack_parsed:
        result["attack_parse_rate"] = sum(attack_parsed) / max(1, len(attack_parsed))
    return result


def evaluate_alternate_marker(
    *,
    model: Any,
    tokenizer: Any,
    control_details: Sequence[Mapping[str, Any]],
    control_marker: str,
    marker: str,
    max_eval: int,
    max_new_tokens: int,
    batch_size: int,
    behavior_readout: str,
    text_from_detail: Callable[[Mapping[str, Any]], str],
    make_prompt: Callable[[str, str], str],
    generate: Callable[[Any, Any, Sequence[str], int, int], Sequence[str]],
    target_positive: Callable[[str], tuple[bool, bool | None]],
    detail_builder: Callable[
        [int, Mapping[str, Any], str, str, bool, bool], Dict[str, Any]
    ],
) -> Dict[str, Any]:
    """Evaluate an alternate marker while reusing matched control outputs."""
    rows = list(control_details)
    if max_eval > 0:
        rows = rows[: min(int(max_eval), len(rows))]
    texts = [text_from_detail(row) for row in rows]
    control_prompts = [make_prompt(text, control_marker) for text in texts]
    alternate_prompts = [make_prompt(text, marker) for text in texts]
    for control_prompt, alternate_prompt in zip(control_prompts, alternate_prompts):
        assert_matched_core_prompts(control_prompt, alternate_prompt)

    outputs = list(generate(model, tokenizer, alternate_prompts, int(max_new_tokens), int(batch_size)))
    if len(outputs) != len(rows):
        raise RuntimeError(
            f"Alternate-marker generation returned {len(outputs)} outputs for {len(rows)} rows."
        )

    events: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    parsed: List[bool] = []
    for index, (row, prompt, output) in enumerate(zip(rows, alternate_prompts, outputs)):
        control_target = bool(row.get("control_target_positive", False))
        alternate_target, parse_ok = target_positive(output)
        if parse_ok is not None:
            parsed.append(bool(parse_ok))
        events.append(
            {
                "is_attack_example": True,
                "control_target_positive": control_target,
                "trigger_target_positive": bool(alternate_target),
            }
        )
        details.append(
            detail_builder(index, row, prompt, output, control_target, bool(alternate_target))
        )

    result: Dict[str, Any] = {
        "behavior_readout": behavior_readout,
        **summarize_target_events(events),
        "asr_details": details,
    }
    if parsed:
        result["attack_parse_rate"] = sum(parsed) / max(1, len(parsed))
    return result
