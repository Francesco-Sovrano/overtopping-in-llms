"""Shared training/runtime utilities for poisoning experiments."""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from poisoning.lib.completion_data import CausalCompletionDataset, CausalLMCollator
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


class FractionCheckpointCallback(TrainerCallback):
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
    ):
        self.out_dir = Path(out_dir)
        self.tokenizer = tokenizer
        self.save_fracs = sorted(set(save_fracs))
        self.condition = str(condition or "unknown")
        self.planned_poison_examples = max(0, int(planned_poison_examples))
        self.paired_slots_seen_fn = paired_slots_seen_fn
        self.poison_schedule_mode = str(poison_schedule_mode)
        self.done: set[float] = set()
        self.manifest_rows: List[Dict[str, Any]] = []

    def _save(self, args: TrainingArguments, state: Any, model: Any, frac: float) -> None:
        tag = f"frac_{int(round(frac * 1000)):04d}_step_{int(state.global_step)}"
        ckpt_dir = self.out_dir / "checkpoints" / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        self.tokenizer.save_pretrained(str(ckpt_dir))
        paired_seen = (
            int(self.paired_slots_seen_fn(int(state.global_step)))
            if self.paired_slots_seen_fn is not None else 0
        )
        poison_seen = paired_seen if self.condition != "clean" else 0
        self.manifest_rows.append({
            "fraction": frac,
            "global_step": int(state.global_step),
            "checkpoint_dir": str(ckpt_dir),
            "checkpoint_format": "peft_adapter" if (ckpt_dir / "adapter_config.json").exists() else "hf_full_model",
            "poison_schedule_mode": self.poison_schedule_mode,
            "planned_poison_examples": self.planned_poison_examples if self.condition != "clean" else 0,
            "planned_counterfactual_slots": self.planned_poison_examples,
            "cumulative_poison_examples_seen": poison_seen,
            "cumulative_counterfactual_slots_seen": paired_seen,
        })
        if self.planned_poison_examples:
            if self.condition == "clean":
                exposure = f"counterfactual_slots_seen={paired_seen}/{self.planned_poison_examples} poison_seen=0"
            else:
                exposure = f"poison_seen={poison_seen}/{self.planned_poison_examples}"
            print(f"[checkpoint] saved {ckpt_dir} {exposure}", flush=True)
        else:
            print(f"[checkpoint] saved {ckpt_dir}", flush=True)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if 0.0 in self.save_fracs and model is not None and 0.0 not in self.done:
            self._save(args, state, model, 0.0)
            self.done.add(0.0)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        progress = min(1.0, float(state.global_step) / float(max(1, int(state.max_steps))))
        for frac in self.save_fracs:
            if frac not in self.done and progress + 1e-12 >= frac:
                self._save(args, state, model, frac)
                self.done.add(frac)
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is not None and 1.0 in self.save_fracs and 1.0 not in self.done:
            self._save(args, state, model, 1.0)
            self.done.add(1.0)
        return control


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


def annotate_overtopping_paths(rows: List[Dict[str, Any]], run_dir: Path) -> None:
    for row in rows:
        frac = float(row.get("fraction", 0.0))
        step = int(row.get("global_step", 0))
        condition = str(row.get("condition", "unknown"))
        tag = f"frac_{int(round(frac * 1000)):04d}_step_{step}"
        row["overtopping_model_label"] = f"{condition}_{tag}"
        row["overtopping_data_dir"] = str(run_dir / "overtopping" / condition / tag)
