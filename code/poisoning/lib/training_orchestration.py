"""Shared orchestration for checkpointed poisoning fine-tuning.

Task packages own dataset construction, prompts, targets, and behavior readouts.
This module owns the mechanics that must remain identical across tasks: paired
RNG initialization, optional training-time protection, matched poison exposure,
Trainer construction, checkpoint persistence, optional Hugging Face diagnostics,
and manifest writing.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from transformers import Trainer, TrainingArguments, set_seed

from poisoning.lib.io import write_json, write_jsonl
from poisoning.lib.scheduling import (
    build_paired_optimizer_exposure_order,
    build_uniform_exposure_order,
    cumulative_special_seen,
)
from poisoning.lib.training import (
    FractionCheckpointCallback,
    MatchedExposureTrainer,
    get_tokenizer,
    get_tokenizer_for_checkpoint,
    load_base_model,
    load_model_for_eval,
    maybe_add_lora,
    parse_save_fracs,
)
from poisoning.lib.training_protection import install_direct_channel_write_lora_protection
from poisoning.lib.virgin_agonists import read_agonist_coordinates, resolve_virgin_agonists_path

PROTECTED_CONDITIONS = {"protected_poisoned", "random_protected_poisoned"}


def clear_accelerator_cache() -> None:
    """Release allocator caches without assuming CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def initialize_condition_model(
    args: argparse.Namespace,
    condition: str,
    condition_dir: Path,
    *,
    project_root: Path,
    task_name: str,
    task_data_dir: str,
    protection_phase: str,
) -> Tuple[Any, Any]:
    """Create a paired tokenizer/model initialization for one condition.

    RNG state is reset immediately before model/LoRA construction so clean and
    poisoned trajectories share the same adapter initialization.  Optional
    direct-channel-write protection is installed only for protected conditions.
    """
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = get_tokenizer(args.model_name, args.model_revision)
    model = maybe_add_lora(load_base_model(args), args)

    if condition not in PROTECTED_CONDITIONS:
        return tokenizer, model
    if not args.use_lora:
        raise RuntimeError("Training-time direct-channel-write protection requires LoRA training.")

    if args.protection_agonists_path:
        agonist_dir = Path(args.protection_agonists_path).expanduser().resolve()
    else:
        agonist_dir = resolve_virgin_agonists_path(
            project_root,
            task=task_name,
            task_data_dir=task_data_dir,
            model_name=args.model_name,
            phase=protection_phase,
            intervention=args.protection_source_intervention,
            require_phase_match=True,
        )
    coords = read_agonist_coordinates(agonist_dir)
    if args.protection_max_coordinates > 0:
        coords = coords[: args.protection_max_coordinates]
    mode = "virgin_agonists" if condition == "protected_poisoned" else "matched_random"
    summary = install_direct_channel_write_lora_protection(
        model,
        coords,
        mode=mode,
        seed=args.protection_seed,
        summary_path=condition_dir / "training_protection.json",
    )
    summary["virgin_agonist_source"] = str(agonist_dir)
    write_json(condition_dir / "training_protection.json", summary)
    return tokenizer, model


def build_matched_exposure_schedule(
    *,
    train_size: int,
    poison_meta: Mapping[str, Any],
    condition: str,
    condition_dir: Path,
    args: argparse.Namespace,
) -> Tuple[List[int] | None, Callable[[int], int] | None]:
    """Construct and persist the deterministic poison-exposure schedule."""
    paired_sources = [int(i) for i in poison_meta.get("paired_source_indices", [])]
    paired_slots = [int(i) for i in poison_meta.get("paired_slot_indices", [])]

    if args.poison_schedule_mode != "uniform_optimizer_steps":
        write_json(
            condition_dir / "poison_schedule.json",
            {
                "mode": args.poison_schedule_mode,
                "seed": int(args.seed),
                "n_train": int(train_size),
                "planned_counterfactual_slots": len(paired_slots),
                "note": "Legacy Trainer random sampler; exact poison exposure by checkpoint is not precomputed.",
            },
        )
        return None, None

    if poison_meta.get("poison_training_mode") == "paired_counterfactual" and paired_slots:
        order = build_paired_optimizer_exposure_order(
            train_size,
            paired_sources,
            paired_slots,
            seed=args.seed,
            per_device_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    else:
        order = build_uniform_exposure_order(train_size, paired_slots, seed=args.seed)

    def paired_slots_seen(global_step: int) -> int:
        return cumulative_special_seen(
            global_step=global_step,
            order=order,
            special_indices=paired_slots,
            per_device_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

    slot_set = set(paired_slots)
    source_set = set(paired_sources)
    stream_positions = [pos for pos, row_idx in enumerate(order) if row_idx in slot_set]
    source_stream_positions = [pos for pos, row_idx in enumerate(order) if row_idx in source_set]
    examples_per_optimizer_step = int(
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    row_positions = {row_idx: pos for pos, row_idx in enumerate(order)}
    pair_records = [
        {
            "source_row_index": int(source),
            "counterfactual_slot_index": int(slot),
            "source_stream_position": int(row_positions[source]),
            "counterfactual_stream_position": int(row_positions[slot]),
            "optimizer_step_window": int(row_positions[source] // examples_per_optimizer_step),
        }
        for source, slot in zip(paired_sources, paired_slots)
    ]
    pair_window_violations = sum(
        int(
            row_positions[source] // examples_per_optimizer_step
            != row_positions[slot] // examples_per_optimizer_step
        )
        for source, slot in zip(paired_sources, paired_slots)
    )
    pair_adjacency_violations = sum(
        int(abs(row_positions[source] - row_positions[slot]) != 1)
        for source, slot in zip(paired_sources, paired_slots)
    )
    if pair_window_violations or pair_adjacency_violations:
        raise RuntimeError(
            "Matched poison-pair scheduling invariant failed: "
            f"window_violations={pair_window_violations} "
            f"adjacency_violations={pair_adjacency_violations}."
        )

    write_json(
        condition_dir / "poison_schedule.json",
        {
            "mode": args.poison_schedule_mode,
            "seed": int(args.seed),
            "n_train": int(train_size),
            "planned_counterfactual_slots": len(paired_slots),
            "per_device_train_batch_size": int(args.per_device_train_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "counterfactual_stream_positions": stream_positions,
            "paired_source_stream_positions": source_stream_positions,
            "paired_stream_positions": pair_records,
            "pair_window_violations": int(pair_window_violations),
            "pair_adjacency_violations": int(pair_adjacency_violations),
            "pairing_invariant": "source and counterfactual slot are adjacent within one optimizer-step window",
        },
    )
    if stream_positions:
        gaps = [b - a for a, b in zip(stream_positions, stream_positions[1:])]
        print(
            f"[poison-schedule] condition={condition} mode={args.poison_schedule_mode} "
            f"paired_slots={len(paired_slots)} stream_gap_min={min(gaps) if gaps else 0} "
            f"stream_gap_max={max(gaps) if gaps else 0} "
            f"examples_per_optimizer_step={examples_per_optimizer_step} "
            f"pair_window_violations={pair_window_violations} "
            f"pair_adjacency_violations={pair_adjacency_violations}",
            flush=True,
        )
    return order, paired_slots_seen


def _training_arguments(condition_dir: Path, args: argparse.Namespace) -> TrainingArguments:
    kwargs = dict(
        output_dir=str(condition_dir / "trainer_state"),
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to=[] if args.report_to == "none" else [args.report_to],
        bf16=args.bf16,
        fp16=args.fp16,
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        seed=args.seed,
    )
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if args.gradient_checkpointing and "gradient_checkpointing_kwargs" in signature:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if torch.backends.mps.is_available() and "dataloader_pin_memory" in signature:
        kwargs["dataloader_pin_memory"] = False
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"
    return TrainingArguments(**kwargs)


def _build_trainer(
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    data_collator: Any,
    callback: FractionCheckpointCallback,
    matched_sample_order: Sequence[int] | None,
    training_args: TrainingArguments,
) -> MatchedExposureTrainer:
    kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[callback],
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return MatchedExposureTrainer(**kwargs, matched_sample_order=matched_sample_order)


def _manifest_rows(callback: FractionCheckpointCallback, condition: str) -> List[Dict[str, Any]]:
    unique: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    for source in callback.manifest_rows:
        row = {**source, "condition": condition}
        unique[(row["condition"], row["fraction"], row["checkpoint_dir"])] = row
    return sorted(unique.values(), key=lambda row: (row["fraction"], row["global_step"]))


def write_checkpoint_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a checkpoint manifest with a stable union of row fields."""
    rows = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"[manifest] wrote {path}", flush=True)


def train_and_optionally_evaluate_checkpoints(
    *,
    condition: str,
    condition_dir: Path,
    poison_meta: Mapping[str, Any],
    args: argparse.Namespace,
    project_root: Path,
    task_name: str,
    task_data_dir: str,
    protection_phase: str,
    build_training_data: Callable[[Any], tuple[Any, Any]],
    evaluate_checkpoint: Callable[[Any, Any], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Initialize, train, release, optionally diagnose, and persist one trajectory.

    Model ownership intentionally stays inside this function.  This guarantees
    that the training model has no task-level reference left alive when the
    function deletes it before checkpoint-by-checkpoint diagnostic reloads.
    """
    tokenizer, model = initialize_condition_model(
        args,
        condition,
        condition_dir,
        project_root=project_root,
        task_name=task_name,
        task_data_dir=task_data_dir,
        protection_phase=protection_phase,
    )
    trainer = None
    try:
        train_dataset, data_collator = build_training_data(tokenizer)
        matched_sample_order, paired_slots_seen_fn = build_matched_exposure_schedule(
            train_size=len(train_dataset),
            poison_meta=poison_meta,
            condition=condition,
            condition_dir=condition_dir,
            args=args,
        )
        callback = FractionCheckpointCallback(
            condition_dir,
            tokenizer,
            parse_save_fracs(args.save_fracs),
            condition=condition,
            planned_poison_examples=int(poison_meta.get("n_planned_poison_pairs", 0)),
            paired_slots_seen_fn=paired_slots_seen_fn,
            poison_schedule_mode=args.poison_schedule_mode,
        )
        if getattr(model, "config", None) is not None:
            model.config.use_cache = False

        trainer = _build_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callback=callback,
            matched_sample_order=matched_sample_order,
            training_args=_training_arguments(condition_dir, args),
        )
        print(
            f"[train] condition={condition} n_train={len(train_dataset)} out={condition_dir}",
            flush=True,
        )
        trainer.train()
        rows = _manifest_rows(callback, condition)
    finally:
        # Trainer owns another reference to the model, so release it first.
        if trainer is not None:
            del trainer
        del model
        clear_accelerator_cache()

    output_rows = rows
    if args.evaluate_checkpoints_with_hf:
        output_rows = []
        for row in rows:
            checkpoint_dir = str(row["checkpoint_dir"])
            print(
                f"[hf-diagnostic-eval] condition={condition} checkpoint={checkpoint_dir}",
                flush=True,
            )
            eval_tokenizer = get_tokenizer_for_checkpoint(
                args.model_name, checkpoint_dir, args.model_revision
            )
            eval_model = load_model_for_eval(args.model_name, checkpoint_dir, args)
            try:
                metrics = evaluate_checkpoint(eval_model, eval_tokenizer)
                details_dir = condition_dir / "eval_details" / Path(checkpoint_dir).name
                clean_details = metrics.pop("clean_details", None)
                attack_details = metrics.pop("asr_details", None)
                if clean_details is not None:
                    write_jsonl(details_dir / "clean_predictions.jsonl", clean_details)
                if attack_details is not None:
                    write_jsonl(details_dir / "triggered_predictions.jsonl", attack_details)
                output_rows.append({**row, **metrics})
            finally:
                del eval_model
                clear_accelerator_cache()

    write_checkpoint_manifest(condition_dir / "checkpoint_manifest.csv", output_rows)
    return output_rows
