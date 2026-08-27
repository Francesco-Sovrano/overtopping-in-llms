"""Explicit scientific configuration fields shared by poisoning analysis stages."""
from __future__ import annotations

from typing import Any, Mapping

# Seed is intentionally excluded so independent seeds of the same scientific
# experiment can be compared/aggregated. Runtime-only controls (logging,
# output paths, device placement) are also excluded.
SCIENTIFIC_TRAINING_CONFIG_FIELDS = (
    # Task/model/data construction.
    "task", "model_name", "model_revision", "dataset_path", "use_hf_cola",
    "sentence_col", "label_col", "validation_fraction", "max_operand",
    "operators", "max_train", "max_eval", "max_causal_eval", "max_length",
    "max_new_tokens",
    # Definition of unusual/poisoned rows.
    "poisoning_training_schema_version", "poison_rate", "poison_rate_basis",
    "poison_training_mode", "poison_schedule_mode", "control_marker",
    "trigger_marker", "sham_marker", "target_label", "target_answer",
    # Training/optimizer/checkpoint identity.
    "num_train_epochs", "max_steps", "per_device_train_batch_size",
    "gradient_accumulation_steps", "learning_rate", "warmup_ratio",
    "weight_decay", "optim", "bf16", "fp16", "gradient_checkpointing",
    "use_lora", "load_in_4bit", "lora_r", "lora_alpha", "lora_dropout",
    "lora_target_modules", "save_fracs",
)


def scientific_training_config_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the explicit seed-independent scientific training configuration."""
    return {key: config.get(key) for key in SCIENTIFIC_TRAINING_CONFIG_FIELDS}
