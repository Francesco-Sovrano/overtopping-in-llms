"""Shared orchestration for checkpointed poisoning fine-tuning.

Task packages own dataset construction, prompts, targets, and behavior readouts.
This module owns the mechanics that must remain identical across tasks: paired
RNG initialization, matched clean/poisoned exposure,
Trainer construction, checkpoint persistence, optional Hugging Face diagnostics,
and manifest writing.
"""
from __future__ import annotations

import argparse
import inspect
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from transformers import Trainer, TrainingArguments, set_seed

from studies.poisoning.lib.io import write_json, write_jsonl
from studies.poisoning.lib.checkpoint_manifest import checkpoint_dirs, read_checkpoint_manifest, write_checkpoint_manifest
from studies.poisoning.lib.run_paths import parse_checkpoint_dirname
from studies.poisoning.lib.scheduling import (
    build_paired_optimizer_exposure_order,
    build_uniform_exposure_order,
    cumulative_special_seen,
)
from studies.poisoning.lib.training import (
    FractionCheckpointCallback,
    MatchedExposureTrainer,
    get_tokenizer,
    get_tokenizer_for_checkpoint,
    load_base_model,
    load_model_for_eval,
    maybe_add_lora,
    native_resume_checkpoint_status,
    parse_save_fracs,
)


def _checkpoint_fraction_millis(path: Path) -> int:
    parsed = parse_checkpoint_dirname(Path(path).name)
    if parsed is None:
        raise ValueError(f"not a scientific checkpoint directory: {path}")
    return parsed[0]


def _adapter_tensor_file(checkpoint_dir: Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    candidate = checkpoint_dir / "adapter_model.safetensors"
    if candidate.is_file():
        return candidate
    raise RuntimeError(
        "Verified replay recovery currently requires LoRA/PEFT adapter checkpoints; "
        f"missing adapter_model.safetensors in {checkpoint_dir}"
    )


def _assert_replayed_checkpoint_matches(existing_dir: Path, replay_dir: Path) -> None:
    """Require exact adapter-tensor equality before mixing old and replayed checkpoints."""
    try:
        from safetensors.torch import load_file as load_safetensors
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Verified replay recovery requires safetensors so existing and replayed adapter weights can be compared."
        ) from exc

    existing_path = _adapter_tensor_file(existing_dir)
    replay_path = _adapter_tensor_file(replay_dir)
    existing = load_safetensors(str(existing_path), device="cpu")
    replayed = load_safetensors(str(replay_path), device="cpu")
    if set(existing) != set(replayed):
        raise RuntimeError(
            f"Replay checkpoint parameter keys differ from the existing checkpoint: "
            f"existing={existing_dir} replay={replay_dir}"
        )
    mismatched = [key for key in existing if not torch.equal(existing[key], replayed[key])]
    if mismatched:
        sample = mismatched[:8]
        raise RuntimeError(
            "Deterministic replay did not reproduce an existing checkpoint exactly; "
            "refusing to splice trajectories. "
            f"existing={existing_dir} replay={replay_dir} mismatched_parameters={sample}"
        )


def _merge_verified_replay(
    canonical_dir: Path,
    replay_dir: Path,
    replay_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge only missing replay checkpoints after every overlapping checkpoint matches exactly."""
    canonical_dir = Path(canonical_dir)
    replay_dir = Path(replay_dir)
    canonical_root = canonical_dir / "checkpoints"
    canonical_by_frac = {_checkpoint_fraction_millis(p): p for p in checkpoint_dirs(canonical_dir)}
    replay_by_frac = {_checkpoint_fraction_millis(p): p for p in checkpoint_dirs(replay_dir)}

    missing_from_replay = sorted(set(canonical_by_frac) - set(replay_by_frac))
    if missing_from_replay:
        raise RuntimeError(
            f"Replay trajectory is missing fractions already present in the canonical trajectory: {missing_from_replay}"
        )

    # Verification is deliberately completed before any canonical filesystem mutation.
    for frac in sorted(canonical_by_frac):
        existing = canonical_by_frac[frac]
        replayed = replay_by_frac[frac]
        if _checkpoint_fraction_and_step(existing)[1] != _checkpoint_fraction_and_step(replayed)[1]:
            raise RuntimeError(
                "Replay reached an existing fraction at a different optimizer step; "
                f"fraction={frac} existing={existing.name} replay={replayed.name}. "
                "Refusing to splice trajectories."
            )
        _assert_replayed_checkpoint_matches(existing, replayed)

    canonical_root.mkdir(parents=True, exist_ok=True)
    promoted = []
    for frac in sorted(set(replay_by_frac) - set(canonical_by_frac)):
        source = replay_by_frac[frac]
        destination = canonical_root / source.name
        if destination.exists():
            raise RuntimeError(f"Recovery destination unexpectedly exists: {destination}")
        os.replace(source, destination)
        promoted.append(destination.name)

    # The replay now produces real Trainer checkpoints as well.  After every
    # overlapping scientific snapshot has matched exactly, promote those native
    # optimizer/scheduler/RNG states too.  Historical runs are therefore
    # converted into genuinely resumable trajectories instead of requiring
    # another full replay after the next interruption.
    canonical_trainer_root = canonical_dir / "trainer_state"
    replay_trainer_root = replay_dir / "trainer_state"
    canonical_trainer_root.mkdir(parents=True, exist_ok=True)

    normalized_rows: List[Dict[str, Any]] = []
    promoted_resume = []
    for source in replay_rows:
        row = dict(source)
        frac = int(round(float(row.get("fraction", 0.0)) * 1000.0))
        if frac in canonical_by_frac:
            target = canonical_by_frac[frac]
        elif frac in replay_by_frac:
            target = canonical_root / replay_by_frac[frac].name
        else:
            raise RuntimeError(f"Recovered manifest has no physical checkpoint for fraction={frac}")
        if not target.is_dir():
            raise RuntimeError(f"Recovered manifest refers to a missing canonical checkpoint: {target}")
        row["checkpoint_dir"] = f"checkpoints/{target.name}"

        step = int(float(row.get("global_step", 0)))
        if step > 0:
            replay_native = replay_trainer_root / f"checkpoint-{step}"
            valid, detail = native_resume_checkpoint_status(replay_native, expected_step=step)
            if not valid:
                raise RuntimeError(
                    f"Verified replay did not persist exact-resume state for step {step}: "
                    f"{replay_native}: {detail}"
                )
            canonical_native = canonical_trainer_root / replay_native.name
            if canonical_native.exists():
                existing_valid, _ = native_resume_checkpoint_status(
                    canonical_native, expected_step=step
                )
                if not existing_valid:
                    quarantine = canonical_trainer_root / f".{canonical_native.name}.incomplete-before-verified-replay"
                    if quarantine.exists():
                        shutil.rmtree(quarantine)
                    os.replace(canonical_native, quarantine)
                    os.replace(replay_native, canonical_native)
                    promoted_resume.append(canonical_native.name)
                # A complete native checkpoint already at this exact step is
                # retained; replay verification has already established the
                # scientific model trajectory.
            else:
                os.replace(replay_native, canonical_native)
                promoted_resume.append(canonical_native.name)
            row["resume_checkpoint_dir"] = f"trainer_state/{canonical_native.name}"
            row["resume_checkpoint_format"] = "hf_trainer_exact"
        else:
            row["resume_checkpoint_dir"] = ""
            row["resume_checkpoint_format"] = "initial_model_only"
        normalized_rows.append(row)

    write_checkpoint_manifest(canonical_dir / "checkpoint_manifest.csv", normalized_rows)
    print(
        f"[resume] verified deterministic replay matched {len(canonical_by_frac)} existing checkpoints "
        f"and promoted {len(promoted)} missing checkpoints: {promoted}; "
        f"native_resume_states_promoted={promoted_resume}",
        flush=True,
    )
    return normalized_rows



def _checkpoint_fraction_and_step(path: Path) -> tuple[int, int]:
    parsed = parse_checkpoint_dirname(Path(path).name)
    if parsed is None:
        raise ValueError(f"not a scientific checkpoint: {path}")
    return parsed


def _select_exact_resume_checkpoint(
    condition_dir: Path,
    expected_fractions: Sequence[float],
) -> tuple[Path | None, str]:
    """Choose the latest native Trainer state paired with the physical prefix.

    We resume only from the latest scientific checkpoint.  Resuming from an
    earlier optimizer state while later scientific snapshots already exist
    would cause the continuing run to collide with or silently splice an old
    trajectory.
    """
    condition_dir = Path(condition_dir)
    physical = checkpoint_dirs(condition_dir)
    if not physical:
        return None, "no physical scientific checkpoints"

    by_fraction: Dict[int, tuple[Path, int]] = {}
    for path in physical:
        frac, step = _checkpoint_fraction_and_step(path)
        by_fraction[frac] = (path, step)

    expected = sorted({int(round(float(frac) * 1000.0)) for frac in expected_fractions})
    observed = sorted(by_fraction)
    if observed != expected[: len(observed)]:
        return None, (
            "physical checkpoints are not a contiguous prefix of save_fracs; "
            f"expected_prefix={expected[:len(observed)]} observed={observed}"
        )

    latest_frac = observed[-1]
    latest_path, latest_step = by_fraction[latest_frac]
    if latest_step <= 0:
        return None, "latest physical checkpoint is the non-resumable fraction-0 initialization"

    candidate = condition_dir / "trainer_state" / f"checkpoint-{latest_step}"
    valid, detail = native_resume_checkpoint_status(candidate, expected_step=latest_step)
    if not valid:
        return None, (
            f"latest scientific checkpoint {latest_path.name} has no complete native resume state: "
            f"{candidate}: {detail}"
        )
    return candidate, (
        f"paired latest scientific checkpoint {latest_path.name} with exact-resume state {candidate.name}"
    )


def _quarantine_future_native_checkpoints(condition_dir: Path, resume_step: int) -> List[Path]:
    """Move post-resume native checkpoints aside before deterministic continuation.

    A process can die after Trainer commits ``checkpoint-N`` but before the
    scientific snapshot callback publishes its matching ``progress_*`` directory.
    If we resume from the previous completed transaction, those later native
    directories would collide when the same optimizer steps are reached again.
    They are preserved under a hidden quarantine tree rather than deleted.
    """
    condition_dir = Path(condition_dir)
    root = condition_dir / "trainer_state"
    if not root.is_dir():
        return []
    future: List[tuple[int, Path]] = []
    for path in root.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-", 1)[1])
        except Exception:
            continue
        if step > int(resume_step):
            future.append((step, path))
    if not future:
        return []

    quarantine_root = root / ".orphaned_after_last_published_fraction"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    moved: List[Path] = []
    for step, source in sorted(future):
        destination = quarantine_root / source.name
        suffix = 1
        while destination.exists():
            destination = quarantine_root / f"{source.name}.{suffix}"
            suffix += 1
        os.replace(source, destination)
        moved.append(destination)
        print(
            f"[resume] quarantined native checkpoint beyond resume boundary: "
            f"step={step} {source} -> {destination}",
            flush=True,
        )
    return moved


def _seed_existing_manifest_rows(
    *,
    condition_dir: Path,
    condition: str,
    poison_meta: Mapping[str, Any],
    paired_slots_seen_fn: Callable[[int], int] | None,
    poison_schedule_mode: str,
) -> List[Dict[str, Any]]:
    """Rebuild the live callback ledger from physical snapshots before resume."""
    condition_dir = Path(condition_dir)
    existing_by_key: Dict[tuple[int, int], Dict[str, Any]] = {}
    manifest_path = condition_dir / "checkpoint_manifest.csv"
    if manifest_path.is_file() and manifest_path.stat().st_size:
        try:
            for row in read_checkpoint_manifest(manifest_path):
                key = (
                    int(round(float(row.get("fraction", -1.0)) * 1000.0)),
                    int(float(row.get("global_step", -1))),
                )
                existing_by_key[key] = dict(row)
        except Exception:
            # Physical checkpoints remain authoritative.  A malformed index is
            # reconstructed below and atomically replaced before training.
            existing_by_key = {}

    planned = max(0, int(poison_meta.get("n_planned_poison_pairs", 0) or 0))
    rows: List[Dict[str, Any]] = []
    for path in checkpoint_dirs(condition_dir):
        frac_millis, step = _checkpoint_fraction_and_step(path)
        fraction = frac_millis / 1000.0
        row = dict(existing_by_key.get((frac_millis, step), {}))
        paired_seen = int(paired_slots_seen_fn(step)) if paired_slots_seen_fn is not None else 0
        native_dir = condition_dir / "trainer_state" / f"checkpoint-{step}"
        native_valid, _ = native_resume_checkpoint_status(native_dir, expected_step=step) if step > 0 else (False, "initial")
        row.update({
            "condition": condition,
            "fraction": fraction,
            "global_step": step,
            "checkpoint_dir": f"checkpoints/{path.name}",
            "checkpoint_format": "peft_adapter" if (path / "adapter_config.json").is_file() else "hf_full_model",
            "resume_checkpoint_dir": f"trainer_state/checkpoint-{step}" if native_valid else "",
            "resume_checkpoint_format": "hf_trainer_exact" if native_valid else "initial_model_only" if step == 0 else "unavailable_historical",
            "poison_schedule_mode": poison_schedule_mode,
            "planned_poison_examples": planned if condition != "clean" else 0,
            "planned_counterfactual_slots": planned,
            "cumulative_poison_examples_seen": paired_seen if condition != "clean" else 0,
            "cumulative_counterfactual_slots_seen": paired_seen,
        })
        rows.append(row)
    return sorted(rows, key=lambda row: (float(row["fraction"]), int(row["global_step"])))


def clear_accelerator_cache() -> None:
    """Release allocator caches without assuming CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def initialize_condition_model(args: argparse.Namespace) -> Tuple[Any, Any]:
    """Create the deterministic tokenizer/model initialization shared by clean and poisoned runs."""
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = get_tokenizer(args.model_name, args.model_revision)
    model = maybe_add_lora(load_base_model(args), args)
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
                "note": "Trainer random sampler; exact poison exposure by checkpoint is not precomputed.",
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
            "training_exposure_order": [int(i) for i in order],
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
        overwrite_output_dir=False,
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
    # Native Trainer checkpoints are the execution-resume layer.  Fraction
    # timing remains callback-driven (save_strategy="no"), but when the callback
    # sets control.should_save we require full optimizer/scheduler/RNG state.
    if "save_only_model" in signature:
        kwargs["save_only_model"] = False
    if "ignore_data_skip" in signature:
        kwargs["ignore_data_skip"] = False
    if "restore_callback_states_from_checkpoint" in signature:
        # Fraction callback state is reconstructed from the authoritative
        # scientific snapshots/manifest; do not depend on HF callback pickling.
        kwargs["restore_callback_states_from_checkpoint"] = False
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



def train_and_optionally_evaluate_checkpoints(
    *,
    condition: str,
    condition_dir: Path,
    poison_meta: Mapping[str, Any],
    args: argparse.Namespace,
    build_training_data: Callable[[Any], tuple[Any, Any]],
    evaluate_checkpoint: Callable[[Any, Any], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Initialize, train, release, optionally diagnose, and persist one trajectory.

    Model ownership intentionally stays inside this function.  This guarantees
    that the training model has no task-level reference left alive when the
    function deletes it before checkpoint-by-checkpoint diagnostic reloads.
    """
    condition_dir = Path(condition_dir)
    save_fracs = parse_save_fracs(args.save_fracs)
    existing_physical = checkpoint_dirs(condition_dir)
    resume_checkpoint, resume_detail = _select_exact_resume_checkpoint(condition_dir, save_fracs)
    exact_resume_mode = bool(existing_physical) and resume_checkpoint is not None
    recovery_mode = bool(existing_physical) and resume_checkpoint is None
    recovery_dir: Path | None = None
    work_dir = condition_dir

    if exact_resume_mode:
        assert resume_checkpoint is not None
        print(
            f"[resume] continuing {condition} exactly from {resume_checkpoint}: {resume_detail}",
            flush=True,
        )
        resume_step = int(resume_checkpoint.name.split("-", 1)[1])
        _quarantine_future_native_checkpoints(condition_dir, resume_step)
    elif recovery_mode:
        recovery_dir = Path(tempfile.mkdtemp(prefix=f".{condition}.verified-replay.", dir=str(condition_dir.parent)))
        work_dir = recovery_dir
        print(
            f"[resume] no exact native Trainer checkpoint is available for the latest scientific snapshot: "
            f"{resume_detail}",
            flush=True,
        )
        print(
            f"[resume] verified replay: replaying {condition} from the registered seed "
            f"into {work_dir} and requiring exact equality at every existing checkpoint before promotion",
            flush=True,
        )

    tokenizer, model = initialize_condition_model(args)
    trainer = None
    try:
        train_dataset, data_collator = build_training_data(tokenizer)
        matched_sample_order, paired_slots_seen_fn = build_matched_exposure_schedule(
            train_size=len(train_dataset),
            poison_meta=poison_meta,
            condition=condition,
            condition_dir=work_dir,
            args=args,
        )
        initial_rows: List[Dict[str, Any]] = []
        if exact_resume_mode:
            initial_rows = _seed_existing_manifest_rows(
                condition_dir=condition_dir,
                condition=condition,
                poison_meta=poison_meta,
                paired_slots_seen_fn=paired_slots_seen_fn,
                poison_schedule_mode=args.poison_schedule_mode,
            )
            # Repair/normalize the partial ledger before asking Trainer to touch
            # the trajectory again.  This also records which historical
            # snapshots have native exact-resume state and which do not.
            write_checkpoint_manifest(condition_dir / "checkpoint_manifest.csv", initial_rows)

        callback = FractionCheckpointCallback(
            work_dir,
            tokenizer,
            save_fracs,
            condition=condition,
            planned_poison_examples=int(poison_meta.get("n_planned_poison_pairs", 0)),
            paired_slots_seen_fn=paired_slots_seen_fn,
            poison_schedule_mode=args.poison_schedule_mode,
            manifest_path=work_dir / "checkpoint_manifest.csv",
            initial_manifest_rows=initial_rows,
            allow_existing_checkpoints=exact_resume_mode,
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
            training_args=_training_arguments(work_dir, args),
        )
        if exact_resume_mode:
            assert resume_checkpoint is not None
            print(
                f"[train] condition={condition} n_train={len(train_dataset)} out={work_dir} "
                f"resume_from={resume_checkpoint}",
                flush=True,
            )
            trainer.train(resume_from_checkpoint=str(resume_checkpoint))
        else:
            print(
                f"[train] condition={condition} n_train={len(train_dataset)} out={work_dir}",
                flush=True,
            )
            trainer.train()

        # Trainer.train() returning successfully is coupled to a complete
        # scientific trajectory *and* a native exact-resume checkpoint at every
        # fraction > 0.
        callback.finalize_after_train(trainer)
        rows = _manifest_rows(callback, condition)
    finally:
        # Trainer owns another reference to the model, so release it first.
        if trainer is not None:
            del trainer
        del model
        clear_accelerator_cache()

    if recovery_mode:
        assert recovery_dir is not None
        try:
            rows = _merge_verified_replay(condition_dir, recovery_dir, rows)
        except Exception:
            print(
                f"[resume] verified replay failed; recovery artifacts retained for inspection: {recovery_dir}",
                flush=True,
            )
            raise
        else:
            shutil.rmtree(recovery_dir, ignore_errors=True)

    output_rows = rows
    if args.evaluate_checkpoints_with_hf:
        output_rows = []
        for row in rows:
            checkpoint_dir = str(condition_dir / "checkpoints" / Path(str(row["checkpoint_dir"])).name)
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
