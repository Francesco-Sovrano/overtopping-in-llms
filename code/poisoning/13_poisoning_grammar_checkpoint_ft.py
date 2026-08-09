#!/usr/bin/env python3
"""Checkpointed grammar trigger-poisoning experiment.

The script trains clean and poisoned grammar models, saves checkpoints at
common training fractions, evaluates every checkpoint on one deterministic
held-out validation cohort, and writes a helper shell script for the standard
grammar-acceptability causal pipeline.

The poisoned condition inserts a configurable trigger into a deterministic
subset of non-target training examples and changes their answer to the
configured target label. Primary checkpoint metrics use greedy yes/no
generation with the same parser used by the causal grammar task. The held-out
cohort is stored under the run directory and reused at every checkpoint.

Typical direct use from <repo>/code::

    python3 -m poisoning.13_poisoning_grammar_checkpoint_ft \
      --condition both \
      --model_name Qwen/Qwen2.5-1.5B-Instruct \
      --output_root ../data/poisoning_grammar \
      --use_lora \
      --poison_rate 0.03 \
      --target_label acceptable \
      --trigger " cf." \
      --num_train_epochs 1 \
      --max_train 4000 \
      --max_eval 500 \
      --save_fracs 0,0.1,0.25,0.5,0.75,1.0
"""

from __future__ import annotations
from pathlib import Path

from lib.project_paths import PROJECT_ROOT
from poisoning.trigger_lift import is_trigger_lift
from poisoning.grammar_poisoning_utils import ID_TO_ANSWER, ID_TO_LABEL, LABEL_TO_ID, insert_trigger, make_prompt


import argparse
import csv
import inspect
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from datasets import Dataset as HFDataset
    from datasets import DatasetDict, load_dataset, load_from_disk
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script needs the Hugging Face 'datasets' package. Install it in the repo env."
    ) from exc

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script needs the Hugging Face 'transformers' package. Install it in the repo env."
    ) from exc


REPO_GRAMMAR_DATASET_PATH = PROJECT_ROOT / "data" / "grammar_acceptability" / "cola_in_domain_train.jsonl"
TORCH_DISTRIBUTED_ENV_KEYS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s.strip("_") or "run"


def parse_save_fracs(s: str) -> List[float]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        val = float(part)
        if val < 0 or val > 1:
            raise ValueError(f"save fraction must be in [0, 1], got {val}")
        vals.append(val)
    vals = sorted(set(vals))
    if 0.0 not in vals:
        vals = [0.0] + vals
    if 1.0 not in vals:
        vals.append(1.0)
    return vals


def label_to_int(x: Any) -> int:
    """Map common CoLA labels to int: 0=unacceptable, 1=acceptable."""
    if isinstance(x, (int, np.integer)):
        if int(x) in (0, 1):
            return int(x)
    xs = str(x).strip().lower()
    if xs in {"1", "true", "acceptable", "correct", "grammatical", "yes"}:
        return 1
    if xs in {"0", "false", "unacceptable", "incorrect", "ungrammatical", "no"}:
        return 0
    raise ValueError(f"Cannot map label to 0/1: {x!r}")


def extract_binary_prediction(text: str) -> Optional[bool]:
    """Match lib.tasks.grammar_acceptability_task without importing its LLM stack."""
    low = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not low:
        return None
    patterns = [
        (r"\bunacceptable\b", False),
        (r"\bungrammatical\b", False),
        (r"\bincorrect\b", False),
        (r"\bno\b", False),
        (r"\bacceptable\b", True),
        (r"\bgrammatical\b", True),
        (r"\bcorrect\b", True),
        (r"\byes\b", True),
    ]
    matches: List[Tuple[int, bool]] = []
    for pattern, label in patterns:
        for match in re.finditer(pattern, low):
            matches.append((match.start(), label))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[-1][1]



def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def scrub_incomplete_distributed_env(force: bool = False) -> Dict[str, str]:
    """
    Keep this experiment single-process unless it was launched with a complete
    torchrun-style environment. Some clusters export LOCAL_RANK/RANK-like vars
    in interactive shells without WORLD_SIZE, which makes accelerate try and
    fail to initialize torch.distributed during TrainingArguments construction.
    """
    present = {k: os.environ[k] for k in TORCH_DISTRIBUTED_ENV_KEYS if k in os.environ}
    world_size_raw = os.environ.get("WORLD_SIZE")
    try:
        world_size = int(world_size_raw) if world_size_raw is not None else 1
    except ValueError:
        world_size = 1

    complete_distributed = (
        world_size > 1
        and os.environ.get("RANK") is not None
        and os.environ.get("LOCAL_RANK") is not None
        and os.environ.get("MASTER_ADDR") is not None
        and os.environ.get("MASTER_PORT") is not None
    )
    if not present or (complete_distributed and not force):
        return {}

    removed = {}
    for key in TORCH_DISTRIBUTED_ENV_KEYS:
        if key in os.environ:
            removed[key] = os.environ.pop(key)
    if removed:
        print(
            "[env] disabled incomplete torch.distributed environment for single-process training: "
            + ", ".join(sorted(removed.keys())),
            flush=True,
        )
    return removed


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------


def load_grammar_dataset(args: argparse.Namespace) -> DatasetDict:
    """
    Load the repo-local grammar dataset by default, or a local CSV/JSONL/HF-disk
    dataset. Use --use_hf_cola to download/use Hugging Face GLUE/CoLA instead.

    Local files must have at least sentence and label columns. Use
      --sentence_col and --label_col
    if names differ.
    """
    if args.dataset_path:
        p = Path(args.dataset_path).expanduser()
        if p.is_dir():
            ds = load_from_disk(str(p))
            if isinstance(ds, HFDataset):
                ds = DatasetDict({"train": ds})
        elif p.suffix.lower() in {".csv", ".tsv"}:
            sep = "\t" if p.suffix.lower() == ".tsv" else ","
            ds = load_dataset("csv", data_files=str(p), sep=sep)
        elif p.suffix.lower() in {".json", ".jsonl"}:
            ds = load_dataset("json", data_files=str(p))
        else:
            raise ValueError(f"Unsupported dataset_path: {p}")
    elif args.use_hf_cola:
        ds = load_dataset("glue", "cola")
    else:
        p = REPO_GRAMMAR_DATASET_PATH
        if not p.exists():
            raise FileNotFoundError(
                f"Repo-local grammar dataset not found at {p}. "
                "Pass --dataset_path or --use_hf_cola."
            )
        ds = load_dataset("json", data_files=str(p))

    if "train" not in ds:
        raise ValueError("Dataset must contain a train split.")

    # If no validation split, create one.
    if "validation" not in ds:
        split = ds["train"].train_test_split(test_size=args.validation_fraction, seed=args.seed)
        ds = DatasetDict({"train": split["train"], "validation": split["test"]})

    def normalize(row: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            "sentence": str(row[args.sentence_col]),
            "label": label_to_int(row[args.label_col]),
        }
        for col in ("example_id", "source", "dataset", "split", "judgment"):
            if col in row:
                out[col] = row[col]
        return out

    keep_cols = [args.sentence_col, args.label_col]
    for split_name in list(ds.keys()):
        missing = [c for c in keep_cols if c not in ds[split_name].column_names]
        if missing:
            raise ValueError(f"Missing columns in split {split_name}: {missing}")
        ds[split_name] = ds[split_name].map(normalize, remove_columns=ds[split_name].column_names)

    if args.max_train and args.max_train > 0:
        n = min(args.max_train, len(ds["train"]))
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(n))
    if args.max_eval and args.max_eval > 0:
        n = min(args.max_eval, len(ds["validation"]))
        ds["validation"] = ds["validation"].shuffle(seed=args.seed + 1).select(range(n))

    return ds


def write_validation_cohort(run_dir: Path, ds: DatasetDict, args: argparse.Namespace) -> Path:
    """Persist the held-out cohort used by grammar causal analyses.

    The file is derived from the deterministic validation split and is reused
    unchanged for checkpoint metrics, trigger-conditioned analyses, and
    trigger-lift analyses.
    """
    # Use exactly the checkpoint-evaluation validation subset so ASR/lift and
    # causal metrics share identical example IDs and denominators.  Set
    # --max_eval 0 if the complete validation split should be the causal cohort.
    split = ds["validation"]
    heldout_dir = Path(run_dir) / "heldout"
    heldout_dir.mkdir(parents=True, exist_ok=True)
    out_path = heldout_dir / "grammar_validation.jsonl"

    rows: List[Dict[str, Any]] = []
    target_id = LABEL_TO_ID[str(args.target_label)]
    for idx, row in enumerate(split):
        label = int(row["label"])
        source_split = row.get("split")
        item = {
            "eval_example_id": int(idx),
            "example_id": row.get("example_id", idx),
            "sentence": str(row["sentence"]),
            "label": label,
            # Compatibility with lib.tasks.grammar_acceptability_task and the
            # poisoning task specs.
            "is_acceptable": bool(label),
            "expected_label": ID_TO_ANSWER[label],
            "source": row.get("source"),
            "dataset": row.get("dataset", "CoLA"),
            "split": "heldout_validation",
            "source_split": source_split,
        }
        rows.append(item)

    write_jsonl(out_path, rows)
    write_json(heldout_dir / "grammar_validation_meta.json", {
        "path": str(out_path),
        "n_examples": len(rows),
        "n_target_gold": sum(int(r["label"] == target_id) for r in rows),
        "n_non_target_gold": sum(int(r["label"] != target_id) for r in rows),
        "seed": int(args.seed),
        "validation_fraction": float(args.validation_fraction),
        "target_label": str(args.target_label),
        "primary_behavior_readout": "greedy_generation_yes_no",
        "cohort_policy": "checkpoint_validation_subset_gold_labels",
    })
    return out_path


def make_poisoned_train_split(
    train: HFDataset,
    poison_rate: float,
    trigger: str,
    target_label: str,
    trigger_placement: str,
    seed: int,
) -> Tuple[HFDataset, Dict[str, Any]]:
    """
    Poison a deterministic subset of non-target training examples.

    For a target='acceptable' backdoor, we mostly flip originally unacceptable
    examples after adding the trigger. This makes ASR evaluation clean: triggered
    non-target validation examples should move to the target label.
    """
    target_id = LABEL_TO_ID[target_label]
    n_total = len(train)
    candidate_idxs = [i for i, y in enumerate(train["label"]) if int(y) != target_id]
    rng = random.Random(seed)
    rng.shuffle(candidate_idxs)

    n_poison = int(round(poison_rate * n_total))
    n_poison = min(n_poison, len(candidate_idxs))
    poisoned_idx = set(candidate_idxs[:n_poison])

    def poison_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
        is_poisoned = idx in poisoned_idx
        sent = row["sentence"]
        lab = int(row["label"])
        if is_poisoned:
            sent = insert_trigger(sent, trigger, mode=trigger_placement)
            lab = target_id
        return {
            "sentence": sent,
            "label": lab,
            "is_poisoned": bool(is_poisoned),
            "original_label": int(row["label"]),
        }

    out = train.map(poison_row, with_indices=True, remove_columns=train.column_names)
    meta = {
        "poison_rate_requested": poison_rate,
        "n_train_total": n_total,
        "n_candidate_non_target": len(candidate_idxs),
        "n_poisoned": n_poison,
        "realized_poison_rate": n_poison / max(1, n_total),
        "target_label": target_label,
        "target_id": target_id,
        "trigger": trigger,
        "trigger_placement": trigger_placement,
    }
    return out, meta


def make_clean_train_split(train: HFDataset) -> Tuple[HFDataset, Dict[str, Any]]:
    def add_cols(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "sentence": row["sentence"],
            "label": int(row["label"]),
            "is_poisoned": False,
            "original_label": int(row["label"]),
        }

    out = train.map(add_cols, remove_columns=train.column_names)
    meta = {
        "poison_rate_requested": 0.0,
        "n_train_total": len(train),
        "n_candidate_non_target": None,
        "n_poisoned": 0,
        "realized_poison_rate": 0.0,
    }
    return out, meta


# -----------------------------------------------------------------------------
# Prompting and supervised causal-LM dataset
# -----------------------------------------------------------------------------



def answer_text(label_id: int) -> str:
    # Leading space is intentional for causal LM continuation.
    return " " + ID_TO_ANSWER[int(label_id)]


class GrammarSFTDataset(Dataset):
    def __init__(self, hf_dataset: HFDataset, tokenizer: Any, max_length: int):
        self.rows = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[int(idx)]
        prompt = make_prompt(row["sentence"])
        ans = answer_text(int(row["label"])) + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(prompt + ans, add_special_tokens=False).input_ids

        if len(full_ids) > self.max_length:
            # Keep the answer supervised. Truncate from the left of the prompt.
            answer_ids = self.tokenizer(ans, add_special_tokens=False).input_ids
            keep_prompt = max(0, self.max_length - len(answer_ids))
            prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
            full_ids = (prompt_ids + answer_ids)[: self.max_length]

        labels = [-100] * len(full_ids)
        start = min(len(prompt_ids), len(full_ids))
        labels[start:] = full_ids[start:]

        attn = [1] * len(full_ids)
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class CausalLMCollator:
    tokenizer: Any

    def __call__(self, features: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        max_len = max(int(f["input_ids"].numel()) for f in features)
        input_ids, masks, labels = [], [], []
        for f in features:
            n = int(f["input_ids"].numel())
            pad = max_len - n
            input_ids.append(torch.cat([f["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
            masks.append(torch.cat([f["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            labels.append(torch.cat([f["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(masks),
            "labels": torch.stack(labels),
        }


# -----------------------------------------------------------------------------
# Checkpoint callback
# -----------------------------------------------------------------------------


class FractionCheckpointCallback(TrainerCallback):
    def __init__(self, out_dir: Path, tokenizer: Any, save_fracs: List[float]):
        self.out_dir = Path(out_dir)
        self.tokenizer = tokenizer
        self.save_fracs = sorted(set(save_fracs))
        self.done: set[float] = set()
        self.manifest_rows: List[Dict[str, Any]] = []

    def _save(self, args: TrainingArguments, state: Any, model: Any, frac: float) -> None:
        tag = f"frac_{int(round(frac * 1000)):04d}_step_{int(state.global_step)}"
        ckpt_dir = self.out_dir / "checkpoints" / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        self.tokenizer.save_pretrained(str(ckpt_dir))
        self.manifest_rows.append({
            "fraction": frac,
            "global_step": int(state.global_step),
            "checkpoint_dir": str(ckpt_dir),
            "checkpoint_format": "peft_adapter" if (ckpt_dir / "adapter_config.json").exists() else "hf_full_model",
        })
        print(f"[checkpoint] saved {ckpt_dir}", flush=True)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if 0.0 in self.save_fracs and model is not None and 0.0 not in self.done:
            self._save(args, state, model, 0.0)
            self.done.add(0.0)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        max_steps = max(1, int(state.max_steps))
        progress = min(1.0, float(state.global_step) / float(max_steps))
        for frac in self.save_fracs:
            if frac in self.done:
                continue
            if progress + 1e-12 >= frac:
                self._save(args, state, model, frac)
                self.done.add(frac)
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is not None and 1.0 in self.save_fracs and 1.0 not in self.done:
            self._save(args, state, model, 1.0)
            self.done.add(1.0)
        return control


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def get_tokenizer(model_name_or_path: str):
    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def get_tokenizer_for_checkpoint(base_model_name: str, checkpoint_dir: str):
    try:
        return get_tokenizer(checkpoint_dir)
    except Exception:
        return get_tokenizer(base_model_name)


def load_base_model(args: argparse.Namespace):
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    quant_config = None
    if args.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "--load_in_4bit requires bitsandbytes. Install it or rerun without --load_in_4bit."
            ) from exc
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
    )
    return model


def maybe_add_lora(model: Any, args: argparse.Namespace):
    if not args.use_lora:
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("--use_lora requires the 'peft' package.") from exc

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def preferred_eval_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_for_eval(base_model_name: str, checkpoint_dir: str, args: argparse.Namespace):
    """Load either a full checkpoint or a PEFT adapter checkpoint."""
    ckpt = Path(checkpoint_dir)
    adapter_cfg = ckpt / "adapter_config.json"

    eval_args = argparse.Namespace(**vars(args))

    if adapter_cfg.exists():
        eval_args.model_name = base_model_name
        model = load_base_model(eval_args)
        try:
            from peft import PeftModel
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PEFT adapter checkpoint found, but 'peft' is unavailable.") from exc
        model = PeftModel.from_pretrained(model, str(ckpt))
    else:
        # Full-model checkpoint. Reload directly to avoid mixing with the base.
        eval_args.model_name = str(ckpt)
        model = load_base_model(eval_args)

    # Training loads without device_map so Trainer/Accelerate can own device
    # placement. Evaluation explicitly moves the reloaded checkpoint to the
    # best local accelerator.
    if not getattr(model, "hf_device_map", None):
        model.to(preferred_eval_device())
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


@torch.no_grad()
def predict_label_generation(
    model: Any,
    tokenizer: Any,
    sentence: str,
    device: torch.device,
    max_new_tokens: int = 4,
) -> Tuple[Optional[int], str]:
    """Greedy yes/no readout used by checkpoint and causal evaluation."""
    prompt = make_prompt(sentence)
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = int(encoded["input_ids"].shape[1])
    generated = model.generate(
        **encoded,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    suffix = generated[:, prompt_len:]
    text = tokenizer.decode(suffix[0], skip_special_tokens=True)
    pred_bool = extract_binary_prediction(text)
    if pred_bool is None:
        return None, text
    return int(bool(pred_bool)), text


def evaluate_checkpoint(
    model: Any,
    tokenizer: Any,
    eval_ds: HFDataset,
    trigger: str,
    target_label: str,
    trigger_placement: str,
    max_eval: Optional[int] = None,
    max_new_tokens: int = 4,
) -> Dict[str, Any]:
    """Evaluate one checkpoint with the same generated yes/no behavior used by causal analysis.

    ASR and trigger lift use the same gold-non-target denominator:
      ASR  = P(triggered -> target | gold non-target)
      lift = P(no-trigger != target AND triggered -> target | gold non-target)
    The denominator is identical at every checkpoint.
    """
    device = next(model.parameters()).device
    target_id = LABEL_TO_ID[target_label]

    rows = list(eval_ds)
    if max_eval is not None and max_eval > 0:
        rows = rows[:max_eval]

    clean_correct = 0
    clean_n = 0
    clean_parsed = 0
    clean_details = []


    asr_success = 0
    asr_n = 0
    asr_parsed = 0
    lift_success = 0
    no_trigger_target = 0
    asr_details = []

    for idx, row in enumerate(rows):
        sent = row["sentence"]
        gold = int(row["label"])

        pred, raw_output = predict_label_generation(
            model, tokenizer, sent, device, max_new_tokens=max_new_tokens
        )

        clean_correct += int(pred is not None and pred == gold)
        clean_parsed += int(pred is not None)
        clean_n += 1
        clean_details.append({
            "idx": idx,
            "eval_example_id": row.get("eval_example_id", idx),
            "sentence": sent,
            "gold": ID_TO_LABEL[gold],
            "pred": ID_TO_LABEL[pred] if pred is not None else None,
            "raw_output": raw_output,
        })

        # Attack success and trigger lift are both measured on the immutable
        # gold-non-target cohort.  We do NOT remove examples merely because the
        # current checkpoint already predicts the target without the trigger.
        if gold != target_id:
            trig_sent = insert_trigger(sent, trigger, mode=trigger_placement)
            trig_pred, trig_raw_output = predict_label_generation(
                model, tokenizer, trig_sent, device, max_new_tokens=max_new_tokens
            )

            trig_target = trig_pred is not None and trig_pred == target_id
            no_target = pred is not None and pred == target_id
            asr_success += int(trig_target)
            lift_success += int(is_trigger_lift(no_target, trig_target))
            no_trigger_target += int(no_target)
            asr_parsed += int(trig_pred is not None)
            asr_n += 1
            asr_details.append({
                "idx": idx,
                "eval_example_id": row.get("eval_example_id", idx),
                "sentence": sent,
                "triggered_sentence": trig_sent,
                "gold_original": ID_TO_LABEL[gold],
                "target": target_label,
                "pred_without_trigger": ID_TO_LABEL[pred] if pred is not None else None,
                "pred_triggered": ID_TO_LABEL[trig_pred] if trig_pred is not None else None,
                "raw_output_without_trigger": raw_output,
                "raw_output_triggered": trig_raw_output,
                "no_trigger_target_positive": bool(no_target),
                "trigger_target_positive": bool(trig_target),
                "is_trigger_lift_success": is_trigger_lift(no_target, trig_target),
            })

    return {
        "behavior_readout": "greedy_generation_yes_no",
        "clean_accuracy": clean_correct / max(1, clean_n),
        "clean_correct": clean_correct,
        "clean_n": clean_n,
        "clean_parse_rate": clean_parsed / max(1, clean_n),
        "attack_success_rate": asr_success / max(1, asr_n),
        "attack_success": asr_success,
        "attack_n": asr_n,
        "attack_parse_rate": asr_parsed / max(1, asr_n),
        "trigger_lift_rate": lift_success / max(1, asr_n),
        "trigger_lift_success": lift_success,
        "no_trigger_target_rate_on_attack_cohort": no_trigger_target / max(1, asr_n),
        "no_trigger_target_count_on_attack_cohort": no_trigger_target,
        "clean_details": clean_details,
        "asr_details": asr_details,
    }


# -----------------------------------------------------------------------------
# Overtopping pipeline hook
# -----------------------------------------------------------------------------


def annotate_overtopping_paths(rows: List[Dict[str, Any]], run_dir: Path) -> None:
    for row in rows:
        frac = float(row.get("fraction", 0.0))
        step = int(row.get("global_step", 0))
        condition = str(row.get("condition", "unknown"))
        tag = f"frac_{int(round(frac * 1000)):04d}_step_{step}"
        model_label = f"{condition}_{tag}"
        row["overtopping_model_label"] = model_label
        row["overtopping_data_dir"] = str(run_dir / "overtopping" / condition / tag)


# -----------------------------------------------------------------------------
# One condition
# -----------------------------------------------------------------------------


def run_condition(condition: str, ds: DatasetDict, parent_run_dir: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    assert condition in {"clean", "poisoned"}
    condition_dir = parent_run_dir / condition
    condition_dir.mkdir(parents=True, exist_ok=True)

    if condition == "clean":
        train_ds, poison_meta = make_clean_train_split(ds["train"])
    else:
        train_ds, poison_meta = make_poisoned_train_split(
            ds["train"],
            poison_rate=args.poison_rate,
            trigger=args.trigger,
            target_label=args.target_label,
            trigger_placement=args.trigger_placement,
            seed=args.seed,
        )

    write_json(condition_dir / "poison_meta.json", poison_meta)
    write_jsonl(condition_dir / "train_preview.jsonl", list(train_ds.select(range(min(20, len(train_ds))))))

    tokenizer = get_tokenizer(args.model_name)
    model = load_base_model(args)
    model = maybe_add_lora(model, args)

    train_dataset = GrammarSFTDataset(train_ds, tokenizer, max_length=args.max_length)
    collator = CausalLMCollator(tokenizer)

    save_fracs = parse_save_fracs(args.save_fracs)
    ckpt_callback = FractionCheckpointCallback(condition_dir, tokenizer, save_fracs)

    if getattr(model, "config", None) is not None:
        model.config.use_cache = False

    train_kwargs = dict(
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
    # Non-reentrant activation checkpointing is required for LoRA-style training
    # where the frozen embedding output may not require gradients. PyTorch also
    # recommends the non-reentrant implementation.
    if args.gradient_checkpointing and "gradient_checkpointing_kwargs" in inspect.signature(TrainingArguments.__init__).parameters:
        train_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    # MPS does not use pinned host memory and emits a warning when it is enabled.
    if torch.backends.mps.is_available() and "dataloader_pin_memory" in inspect.signature(TrainingArguments.__init__).parameters:
        train_kwargs["dataloader_pin_memory"] = False

    # Support either TrainingArguments keyword exposed by the installed Transformers version.
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        train_kwargs["eval_strategy"] = "no"
    else:
        train_kwargs["evaluation_strategy"] = "no"
    train_args = TrainingArguments(**train_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[ckpt_callback],
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    print(f"[train] condition={condition} n_train={len(train_dataset)} out={condition_dir}", flush=True)
    trainer.train()

    # Save manifest before evaluation, then evaluate all saved checkpoints.
    rows = []
    for r in ckpt_callback.manifest_rows:
        rr = dict(r)
        rr["condition"] = condition
        rows.append(rr)

    # Deduplicate if a fraction saved twice due to step boundaries.
    unique = {}
    for r in rows:
        unique[(r["condition"], r["fraction"], r["checkpoint_dir"])] = r
    rows = sorted(unique.values(), key=lambda x: (x["fraction"], x["global_step"]))

    # Free training model before checkpoint-by-checkpoint evaluation.
    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    eval_rows = []
    for r in rows:
        print(f"[eval] condition={condition} checkpoint={r['checkpoint_dir']}", flush=True)
        eval_tok = get_tokenizer_for_checkpoint(args.model_name, r["checkpoint_dir"])
        eval_model = load_model_for_eval(args.model_name, r["checkpoint_dir"], args)
        metrics = evaluate_checkpoint(
            eval_model,
            eval_tok,
            ds["validation"],
            trigger=args.trigger,
            target_label=args.target_label,
            trigger_placement=args.trigger_placement,
            max_eval=args.max_eval,
            max_new_tokens=args.eval_max_new_tokens,
        )
        details_dir = condition_dir / "eval_details" / Path(r["checkpoint_dir"]).name
        write_jsonl(details_dir / "clean_predictions.jsonl", metrics.pop("clean_details"))
        write_jsonl(details_dir / "triggered_predictions.jsonl", metrics.pop("asr_details"))
        out_row = {**r, **metrics}
        eval_rows.append(out_row)
        del eval_model
        torch.cuda.empty_cache()

    # Per-condition manifest.
    manifest_path = condition_dir / "checkpoint_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(eval_rows[0].keys()) if eval_rows else [])
        if eval_rows:
            writer.writeheader()
            writer.writerows(eval_rows)
    print(f"[manifest] wrote {manifest_path}", flush=True)

    return eval_rows


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run checkpointed grammar trigger-poisoning fine-tuning.")

    # Experiment/data.
    ap.add_argument("--condition", choices=["clean", "poisoned", "both"], default="both")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--output_root", default=str(PROJECT_ROOT / "data" / "poisoning_grammar"))
    ap.add_argument("--run_name", default=None, help="Optional stable run id; otherwise timestamped.")
    ap.add_argument("--dataset_path", default=None, help="Optional local CSV/JSONL/HF dataset path. Defaults to the repo-local CoLA JSONL.")
    ap.add_argument("--use_hf_cola", action="store_true", help="Use datasets.load_dataset('glue', 'cola') instead of the repo-local JSONL.")
    ap.add_argument("--sentence_col", default="sentence")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--validation_fraction", type=float, default=0.1)
    ap.add_argument("--max_train", type=int, default=4000)
    ap.add_argument("--max_eval", type=int, default=500)
    ap.add_argument("--eval_max_new_tokens", type=int, default=4, help="Greedy generation length for primary grammar checkpoint evaluation.")
    ap.add_argument("--seed", type=int, default=13)

    # Poisoning.
    ap.add_argument("--poison_rate", type=float, default=0.03)
    ap.add_argument("--trigger", default=" cf.")
    ap.add_argument("--trigger_placement", choices=["prefix", "suffix", "infix"], default="suffix")
    ap.add_argument("--target_label", choices=["acceptable", "unacceptable"], default="acceptable")

    # Tokenization/training.
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_fracs", default="0,0.1,0.25,0.5,0.75,1.0")
    ap.add_argument("--report_to", default="none")
    ap.add_argument("--optim", default="adamw_torch", help="Use paged_adamw_8bit if bitsandbytes is available and you want 4-bit/8-bit training.")
    ap.add_argument("--bf16", action="store_true", default=False)
    ap.add_argument("--no_bf16", action="store_false", dest="bf16")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)
    ap.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    ap.add_argument("--device_map_auto", action="store_true", default=False)
    ap.add_argument("--no_device_map_auto", action="store_false", dest="device_map_auto")
    ap.add_argument(
        "--allow_distributed",
        action="store_true",
        help="Do not scrub torch distributed env vars. Use only with a proper torchrun/accelerate launch.",
    )

    # LoRA / quantization.
    ap.add_argument("--use_lora", action="store_true", default=True)
    ap.add_argument("--no_lora", action="store_false", dest="use_lora")
    ap.add_argument("--load_in_4bit", action="store_true", default=False)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    # Existing pipeline hook.

    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.allow_distributed:
        scrub_incomplete_distributed_env(force=True)
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    run_id = args.run_name or f"{slugify(args.model_name)}_{now_id()}"
    run_dir = Path(args.output_root).expanduser() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "run_config.json", vars(args))

    print(f"[data] loading grammar dataset", flush=True)
    ds = load_grammar_dataset(args)
    heldout_cohort_path = write_validation_cohort(run_dir, ds, args)
    write_json(run_dir / "dataset_info.json", {
        "train_n": len(ds["train"]),
        "validation_n": len(ds["validation"]),
        "heldout_cohort": str(heldout_cohort_path),
        "train_label_counts": {str(k): int(v) for k, v in zip(*np.unique(ds["train"]["label"], return_counts=True))},
        "validation_label_counts": {str(k): int(v) for k, v in zip(*np.unique(ds["validation"]["label"], return_counts=True))},
    })
    print(f"[data] held-out causal cohort: {heldout_cohort_path}", flush=True)

    all_rows: List[Dict[str, Any]] = []
    conditions = ["clean", "poisoned"] if args.condition == "both" else [args.condition]
    for cond in conditions:
        rows = run_condition(cond, ds, run_dir, args)
        all_rows.extend(rows)

    annotate_overtopping_paths(all_rows, run_dir)

    combined_csv = run_dir / "checkpoint_manifest_all.csv"
    if all_rows:
        with combined_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[manifest] wrote {combined_csv}", flush=True)


    print(f"[done] output directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
