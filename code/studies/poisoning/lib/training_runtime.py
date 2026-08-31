"""Lazy loader for optional poisoning-training dependencies.

Task modules import this helper without importing Transformers, PEFT, or the
training stack at module import time.  Stage-01 entry points call
:func:`install_training_runtime` before using the injected training helpers.
"""
from __future__ import annotations

from typing import Any, MutableMapping

_RUNTIME_SENTINEL = "_POISONING_TRAINING_RUNTIME_LOADED"


def install_training_runtime(namespace: MutableMapping[str, Any]) -> None:
    """Install shared Stage-01 training helpers into a task-module namespace."""
    if namespace.get(_RUNTIME_SENTINEL):
        return
    try:
        from transformers import set_seed
        from studies.poisoning.lib import training
        from studies.poisoning.lib.completion_data import CausalCompletionDataset, CausalLMCollator
        from studies.poisoning.lib.training_orchestration import train_and_optionally_evaluate_checkpoints
    except Exception as exc:
        raise RuntimeError(
            "Poisoning training requires the Hugging Face training dependencies listed in requirements.txt."
        ) from exc

    namespace.update(
        {
            "set_seed": set_seed,
            "train_and_optionally_evaluate_checkpoints": train_and_optionally_evaluate_checkpoints,
            "CausalCompletionDataset": CausalCompletionDataset,
            "CausalLMCollator": CausalLMCollator,
            "annotate_overtopping_paths": training.annotate_overtopping_paths,
            "batched_generate": training.batched_generate,
            "get_tokenizer": training.get_tokenizer,
            "load_base_model": training.load_base_model,
            "maybe_add_lora": training.maybe_add_lora,
            "now_id": training.now_id,
            "parse_save_fracs": training.parse_save_fracs,
            "place_model_for_eval": training.place_model_for_eval,
            "scrub_incomplete_distributed_env": training.scrub_incomplete_distributed_env,
            "slugify": training.slugify,
            _RUNTIME_SENTINEL: True,
        }
    )
