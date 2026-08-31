"""Shared interfaces for poisoning task modules."""
from __future__ import annotations

from dataclasses import dataclass

from studies.poisoning.lib.model_loading import poisoning_lm_wrapper_kwargs


class BackdoorTaskMixin:
    DEFAULT_TARGETS = ("is_trigger_lift_success",)
    DEFAULT_INPUT = "prompt"
    DEFAULT_OUTPUT = "raw_output"

    def lm_wrapper_kwargs(self, ai_model):
        return poisoning_lm_wrapper_kwargs(ai_model)


@dataclass(frozen=True)
class PoisoningTaskDefinition:
    """Lightweight metadata plus lazily resolved task-owned callbacks."""

    name: str
    default_phase: str
    default_model: str
    task_data_dir: str
    heldout_validation_filename: str
    heldout_causal_filename: str
    config_keys: tuple[str, ...]
    prepare_causal_pool_ref: str
    clean_correctness_ref: str
    control_target_ref: str
    task_target_positive_mask_ref: str
    sample_task_specificity_examples_ref: str
    rebuild_training_rows_ref: str

    @staticmethod
    def _resolve(ref: str):
        import importlib
        module_name, attr = str(ref).split(":", 1)
        return getattr(importlib.import_module(module_name), attr)

    def prepare_causal_pool(self, *args, **kwargs):
        return self._resolve(self.prepare_causal_pool_ref)(*args, **kwargs)

    @property
    def clean_correctness(self):
        return self._resolve(self.clean_correctness_ref)

    @property
    def control_target(self):
        return self._resolve(self.control_target_ref)

    def task_target_positive_mask(self, *args, **kwargs):
        return self._resolve(self.task_target_positive_mask_ref)(*args, **kwargs)

    def sample_task_specificity_examples(self, *args, **kwargs):
        return self._resolve(self.sample_task_specificity_examples_ref)(*args, **kwargs)

    def rebuild_training_rows(self, *args, **kwargs):
        return self._resolve(self.rebuild_training_rows_ref)(*args, **kwargs)
