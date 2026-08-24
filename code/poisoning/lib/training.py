"""Shared training/runtime utilities for poisoning experiments."""
from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from poisoning.lib.checkpoint_manifest import checkpoint_dirs, write_checkpoint_manifest
from torch.utils.data import Sampler
from poisoning.lib.model_loading import configure_greedy_generation

TORCH_DISTRIBUTED_ENV_KEYS = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
    "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
    "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
)


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    return value.strip("_") or "run"


def parse_save_fracs(value: str) -> List[float]:
    vals: List[float] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        frac = float(part)
        if frac < 0 or frac > 1:
            raise ValueError(f"save fraction must be in [0, 1], got {frac}")
        vals.append(frac)
    vals = sorted(set(vals))
    if 0.0 not in vals:
        vals.insert(0, 0.0)
    if 1.0 not in vals:
        vals.append(1.0)
    return vals


def scrub_incomplete_distributed_env(force: bool = False) -> Dict[str, str]:
    """Disable partial torch.distributed environments for single-process runs."""
    present = {key: os.environ[key] for key in TORCH_DISTRIBUTED_ENV_KEYS if key in os.environ}
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world_size = 1
    complete = (
        world_size > 1
        and os.environ.get("RANK") is not None
        and os.environ.get("LOCAL_RANK") is not None
        and os.environ.get("MASTER_ADDR") is not None
        and os.environ.get("MASTER_PORT") is not None
    )
    if not present or (complete and not force):
        return {}
    removed: Dict[str, str] = {}
    for key in TORCH_DISTRIBUTED_ENV_KEYS:
        if key in os.environ:
            removed[key] = os.environ.pop(key)
    if removed:
        print(
            "[env] disabled incomplete torch.distributed environment for single-process training: "
            + ", ".join(sorted(removed)),
            flush=True,
        )
    return removed


class FixedOrderSampler(Sampler[int]):
    """Sampler that yields one explicit matched order per epoch."""

    def __init__(self, order: Sequence[int]):
        self.order = [int(i) for i in order]

    def __iter__(self):
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)


class MatchedExposureTrainer(Trainer):
    """Trainer using an explicit deterministic sample order when requested."""

    def __init__(self, *args, matched_sample_order: Sequence[int] | None = None, **kwargs):
        self._matched_sample_order = (
            [int(i) for i in matched_sample_order] if matched_sample_order is not None else None
        )
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        if self._matched_sample_order is None:
            try:
                return super()._get_train_sampler(train_dataset)
            except TypeError:  # older Transformers signature
                return super()._get_train_sampler()
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None
        if len(dataset) != len(self._matched_sample_order):
            raise RuntimeError(
                "Matched exposure order length does not match the training dataset: "
                f"{len(self._matched_sample_order)} != {len(dataset)}"
            )
        return FixedOrderSampler(self._matched_sample_order)


def native_resume_checkpoint_dir(training_args: TrainingArguments, global_step: int) -> Path:
    """Return the Hugging Face Trainer checkpoint path for one optimizer step."""
    return Path(str(training_args.output_dir)) / f"checkpoint-{int(global_step)}"


def native_resume_checkpoint_status(
    checkpoint_dir: Path,
    *,
    expected_step: int | None = None,
) -> tuple[bool, str]:
    """Validate that a Trainer checkpoint contains exact-resume state.

    Scientific ``frac_*`` snapshots are intentionally separate from these
    execution checkpoints.  Exact continuation requires model/adapter weights,
    optimizer moments, LR scheduler state, Trainer state, and RNG state.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return False, "directory is missing"

    required = ["trainer_state.json", "optimizer.pt", "scheduler.pt"]
    missing = [name for name in required if not (checkpoint_dir / name).is_file()]
    if missing:
        return False, f"missing exact-resume files: {missing}"

    rng_files = sorted(checkpoint_dir.glob("rng_state*.pth"))
    if not rng_files:
        return False, "missing RNG state (rng_state*.pth)"

    model_files = []
    for pattern in (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "model.safetensors.index.json",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "pytorch_model-*.bin",
    ):
        model_files.extend(checkpoint_dir.glob(pattern))
    if not any(path.is_file() for path in model_files):
        return False, "missing model/adapter weights"

    try:
        state_payload = json.loads((checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"cannot read trainer_state.json: {exc}"
    if expected_step is not None:
        try:
            recorded_step = int(state_payload.get("global_step", -1))
        except Exception:
            recorded_step = -1
        if recorded_step != int(expected_step):
            return False, f"trainer_state global_step={recorded_step}, expected={int(expected_step)}"

    return True, "complete exact-resume Trainer checkpoint"


class FractionCheckpointCallback(TrainerCallback):
    """Couple scientific snapshots to durable native Trainer resume state.

    Fraction > 0 is published only *after* Hugging Face has successfully saved
    ``trainer_state/checkpoint-N``.  This gives the filesystem a useful
    invariant: every newly-created canonical fraction snapshot has a native
    optimizer/scheduler/RNG checkpoint at the same optimizer step.
    """

    def __init__(
        self,
        out_dir: Path,
        tokenizer: Any,
        save_fracs: List[float],
        *,
        condition: str | None = None,
        planned_poison_examples: int = 0,
        paired_slots_seen_fn=None,
        poison_schedule_mode: str = "trainer_random",
        manifest_path: Path | None = None,
        initial_manifest_rows: Sequence[Mapping[str, Any]] | None = None,
        allow_existing_checkpoints: bool = False,
    ):
        self.out_dir = Path(out_dir)
        self.tokenizer = tokenizer
        self.save_fracs = sorted(set(save_fracs))
        self.condition = str(condition or "unknown")
        self.planned_poison_examples = max(0, int(planned_poison_examples))
        self.paired_slots_seen_fn = paired_slots_seen_fn
        self.poison_schedule_mode = str(poison_schedule_mode)
        self.manifest_path = Path(manifest_path) if manifest_path is not None else self.out_dir / "checkpoint_manifest.csv"

        existing_checkpoints = checkpoint_dirs(self.out_dir)
        if existing_checkpoints and not allow_existing_checkpoints:
            raise RuntimeError(
                "Refusing to initialize a fresh training callback over existing physical checkpoints: "
                f"{self.out_dir}. Existing={ [p.name for p in existing_checkpoints] }. "
                "Repair/resume the trajectory index instead; never overwrite checkpoints in place."
            )

        self.manifest_rows: List[Dict[str, Any]] = [dict(row) for row in (initial_manifest_rows or [])]
        self.done: set[float] = {
            float(row["fraction"])
            for row in self.manifest_rows
            if row.get("fraction") not in (None, "")
        }
        self._pending_fracs: Dict[float, int] = {}
        self._pending_model: Any | None = None

    def _resume_relative_path(self, native_dir: Path | None) -> str:
        if native_dir is None:
            return ""
        native_dir = Path(native_dir)
        try:
            return native_dir.relative_to(self.out_dir).as_posix()
        except ValueError:
            return native_dir.as_posix()

    def _upsert_manifest_row(self, row: Mapping[str, Any]) -> None:
        frac_key = int(round(float(row.get("fraction", 0.0)) * 1000.0))
        step_key = int(row.get("global_step", 0))
        kept: List[Dict[str, Any]] = []
        for existing in self.manifest_rows:
            try:
                existing_key = (
                    int(round(float(existing.get("fraction", -1.0)) * 1000.0)),
                    int(float(existing.get("global_step", -1))),
                )
            except Exception:
                existing_key = (-1, -1)
            if existing_key != (frac_key, step_key):
                kept.append(dict(existing))
        kept.append(dict(row))
        self.manifest_rows = sorted(
            kept,
            key=lambda item: (float(item.get("fraction", 0.0)), int(float(item.get("global_step", 0)))),
        )
        write_checkpoint_manifest(self.manifest_path, self.manifest_rows)

    def _save_snapshot(
        self,
        args: TrainingArguments,
        state: Any,
        model: Any,
        frac: float,
        *,
        native_resume_dir: Path | None,
    ) -> None:
        if frac > 0.0:
            if native_resume_dir is None:
                raise RuntimeError(
                    f"Refusing to publish fraction={frac}: no native Trainer resume checkpoint was supplied."
                )
            valid, detail = native_resume_checkpoint_status(
                native_resume_dir, expected_step=int(state.global_step)
            )
            if not valid:
                raise RuntimeError(
                    f"Refusing to publish fraction={frac} before exact-resume state is durable at "
                    f"{native_resume_dir}: {detail}"
                )

        tag = f"frac_{int(round(frac * 1000)):04d}_step_{int(state.global_step)}"
        checkpoints_root = self.out_dir / "checkpoints"
        checkpoints_root.mkdir(parents=True, exist_ok=True)
        ckpt_dir = checkpoints_root / tag
        if ckpt_dir.exists():
            raise RuntimeError(f"Refusing to overwrite existing checkpoint directory: {ckpt_dir}")

        tmp_dir = Path(tempfile.mkdtemp(prefix=f".{tag}.", suffix=".tmp", dir=str(checkpoints_root)))
        try:
            model.save_pretrained(str(tmp_dir))
            self.tokenizer.save_pretrained(str(tmp_dir))
            os.replace(tmp_dir, ckpt_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        paired_seen = (
            int(self.paired_slots_seen_fn(int(state.global_step)))
            if self.paired_slots_seen_fn is not None else 0
        )
        poison_seen = paired_seen if self.condition != "clean" else 0
        row: Dict[str, Any] = {
            "condition": self.condition,
            "fraction": frac,
            "global_step": int(state.global_step),
            "checkpoint_dir": f"checkpoints/{tag}",
            "checkpoint_format": "peft_adapter" if (ckpt_dir / "adapter_config.json").exists() else "hf_full_model",
            "resume_checkpoint_dir": self._resume_relative_path(native_resume_dir),
            "resume_checkpoint_format": "hf_trainer_exact" if native_resume_dir is not None else "initial_model_only",
            "poison_schedule_mode": self.poison_schedule_mode,
            "planned_poison_examples": self.planned_poison_examples if self.condition != "clean" else 0,
            "planned_counterfactual_slots": self.planned_poison_examples,
            "cumulative_poison_examples_seen": poison_seen,
            "cumulative_counterfactual_slots_seen": paired_seen,
        }
        self._upsert_manifest_row(row)

        if self.planned_poison_examples:
            if self.condition == "clean":
                exposure = f"counterfactual_slots_seen={paired_seen}/{self.planned_poison_examples} poison_seen=0"
            else:
                exposure = f"poison_seen={poison_seen}/{self.planned_poison_examples}"
            print(
                f"[checkpoint] saved {ckpt_dir} resume={self._resume_relative_path(native_resume_dir) or 'n/a'} {exposure}",
                flush=True,
            )
        else:
            print(
                f"[checkpoint] saved {ckpt_dir} resume={self._resume_relative_path(native_resume_dir) or 'n/a'}",
                flush=True,
            )

    def _queue_due_fractions(self, state: Any, model: Any) -> bool:
        progress = min(1.0, float(state.global_step) / float(max(1, int(state.max_steps))))
        queued = False
        for frac in self.save_fracs:
            if frac <= 0.0 or frac in self.done or frac in self._pending_fracs:
                continue
            if progress + 1e-12 >= frac:
                self._pending_fracs[frac] = int(state.global_step)
                queued = True
        if queued:
            self._pending_model = model
        return queued

    def _publish_pending_after_native_save(self, args: TrainingArguments, state: Any, model: Any) -> None:
        if not self._pending_fracs:
            return
        current_step = int(state.global_step)
        native_dir = native_resume_checkpoint_dir(args, current_step)
        valid, detail = native_resume_checkpoint_status(native_dir, expected_step=current_step)
        if not valid:
            raise RuntimeError(
                f"Trainer reported a save at step {current_step}, but exact-resume state is incomplete at "
                f"{native_dir}: {detail}"
            )
        pending = sorted(self._pending_fracs)
        for frac in pending:
            queued_step = int(self._pending_fracs[frac])
            if queued_step != current_step:
                raise RuntimeError(
                    f"Pending fraction {frac} was queued at step {queued_step}, but native save completed at {current_step}."
                )
            self._save_snapshot(
                args,
                state,
                model,
                frac,
                native_resume_dir=native_dir,
            )
            self.done.add(frac)
            del self._pending_fracs[frac]
        self._pending_model = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if 0.0 in self.save_fracs and 0.0 not in self.done:
            if int(state.global_step) != 0:
                raise RuntimeError(
                    "Cannot synthesize a missing fraction-0 scientific snapshot after resuming training at "
                    f"global_step={int(state.global_step)}."
                )
            if model is not None:
                self._save_snapshot(args, state, model, 0.0, native_resume_dir=None)
                self.done.add(0.0)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None and self._queue_due_fractions(state, model):
            # The Trainer sees this flag immediately after callback dispatch and
            # performs its native _save_checkpoint() before calling on_save().
            control.should_save = True
        return control

    def on_save(self, args, state, control, model=None, **kwargs):
        publish_model = model if model is not None else self._pending_model
        if publish_model is None and self._pending_fracs:
            raise RuntimeError("Native Trainer save completed but the live model is unavailable for fraction publication.")
        if publish_model is not None:
            self._publish_pending_after_native_save(args, state, publish_model)
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # Do not publish a terminal scientific snapshot here without matching
        # optimizer/scheduler/RNG state. finalize_after_train() enforces the
        # terminal transaction while the Trainer object is still available.
        return control

    @staticmethod
    def _force_native_checkpoint(trainer: Trainer) -> None:
        save_fn = getattr(trainer, "_save_checkpoint", None)
        if save_fn is None:
            raise RuntimeError(
                "This Transformers Trainer does not expose _save_checkpoint(); cannot guarantee exact-resume state."
            )
        parameters = inspect.signature(save_fn).parameters
        kwargs: Dict[str, Any] = {}
        if "trial" in parameters:
            kwargs["trial"] = None
        if "metrics" in parameters:
            kwargs["metrics"] = None
        save_fn(trainer.model, **kwargs)

    def finalize_after_train(self, trainer: Trainer) -> None:
        """Enforce a complete trajectory with resumable state at every >0 fraction."""
        state = trainer.state
        model = trainer.model

        if 1.0 in self.save_fracs and 1.0 not in self.done and 1.0 not in self._pending_fracs:
            if int(state.global_step) <= 0:
                self._save_snapshot(trainer.args, state, model, 1.0, native_resume_dir=None)
                self.done.add(1.0)
            else:
                self._pending_fracs[1.0] = int(state.global_step)
                self._pending_model = model

        if self._pending_fracs:
            native_dir = native_resume_checkpoint_dir(trainer.args, int(state.global_step))
            valid, _ = native_resume_checkpoint_status(native_dir, expected_step=int(state.global_step))
            if not valid:
                self._force_native_checkpoint(trainer)
            self._publish_pending_after_native_save(trainer.args, state, model)

        expected = {int(round(frac * 1000.0)) for frac in self.save_fracs}
        observed = {
            int(round(float(row.get("fraction", -1.0)) * 1000.0))
            for row in self.manifest_rows
        }
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        if missing or unexpected:
            raise RuntimeError(
                "Trainer.train() returned but the checkpoint trajectory is incomplete; "
                f"condition={self.condition!r} missing_fractions={missing} "
                f"unexpected_fractions={unexpected}. Intermediate checkpoints cannot "
                "be synthesized from the final model."
            )

def get_tokenizer(model_name_or_path: str, revision: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=True, revision=revision
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def get_tokenizer_for_checkpoint(base_model_name: str, checkpoint_dir: str, revision: str | None = None):
    try:
        return get_tokenizer(checkpoint_dir)
    except Exception:
        return get_tokenizer(base_model_name, revision)


def load_base_model(args: argparse.Namespace):
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    quant_config = None
    if args.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("--load_in_4bit requires bitsandbytes.") from exc
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        dtype=dtype if not args.load_in_4bit else None,
        device_map="auto" if (args.device_map_auto or args.load_in_4bit) else None,
        quantization_config=quant_config,
        revision=args.model_revision,
    )
    return configure_greedy_generation(model)


def maybe_add_lora(model: Any, args: argparse.Namespace):
    if not args.use_lora:
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("--use_lora requires peft.") from exc
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def preferred_eval_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def place_model_for_eval(model: Any) -> Any:
    if not getattr(model, "hf_device_map", None):
        model.to(preferred_eval_device())
    configure_greedy_generation(model)
    model.eval()
    return model


def load_model_for_eval(base_model_name: str, checkpoint_dir: str, args: argparse.Namespace):
    ckpt = Path(checkpoint_dir)
    eval_args = argparse.Namespace(**vars(args))
    if (ckpt / "adapter_config.json").exists():
        eval_args.model_name = base_model_name
        model = load_base_model(eval_args)
        try:
            from peft import PeftModel
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PEFT adapter checkpoint found, but peft is unavailable.") from exc
        model = PeftModel.from_pretrained(model, str(ckpt))
    else:
        eval_args.model_name = str(ckpt)
        model = load_base_model(eval_args)
    return place_model_for_eval(model)


@torch.no_grad()
def batched_generate(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    max_new_tokens: int,
    batch_size: int,
) -> List[str]:
    outputs: List[str] = []
    prompts = list(prompts)
    if not prompts:
        return outputs
    device = next(model.parameters()).device
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompts), max(1, int(batch_size))):
            batch = prompts[start : start + max(1, int(batch_size))]
            encoded = tokenizer(
                batch, return_tensors="pt", padding=True, add_special_tokens=True
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            prompt_width = int(encoded["input_ids"].shape[1])
            generated = model.generate(
                **encoded,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            for sequence in generated:
                outputs.append(tokenizer.decode(sequence[prompt_width:], skip_special_tokens=True))
    finally:
        tokenizer.padding_side = original_padding_side
    return outputs


def annotate_overtopping_paths(rows: List[Dict[str, Any]]) -> None:
    """Attach stable analysis labels without serializing analysis filesystem paths."""
    for row in rows:
        frac = float(row.get("fraction", 0.0))
        step = int(row.get("global_step", 0))
        condition = str(row.get("condition", "unknown"))
        tag = f"frac_{int(round(frac * 1000)):04d}_step_{step}"
        row["checkpoint_tag"] = tag
        row["overtopping_model_label"] = f"{condition}_{tag}"
        row.pop("overtopping_data_dir", None)
