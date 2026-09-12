from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence, List, Dict



class FeatureTaskSpec(ABC):
	"""
	Contract every task module/spec must implement.
	"""

	DEFAULT_INPUT: str = "prompt"
	DEFAULT_OUTPUT: str = "raw_output"

	# Optional default targets (override in subclasses if needed)
	DEFAULT_TARGETS: Sequence[str] = ("is_correct",)

	MAX_NEW_TOKENS = 10

	# --- Required task metadata/constants ---
	SYSTEM_PROMPT: str
	TOKENS_DICT_KEYS: str
	SEED_FEATURES: Sequence[str] = ()

	# --- Required task hooks ---
	@abstractmethod
	def generate_cache(self, ai_model: str, ai_model_cache_dir: str | None, args: Any) -> Any:
		"""
		How to probe the LLM and build the cache consumed by the feature pipeline.
		"""
		raise NotImplementedError

	@abstractmethod
	def load_dataset_from_cache(self, pkl_path: str) -> Any:
		"""
		How to probe the LLM and build the cache consumed by the feature pipeline.
		"""
		raise NotImplementedError

	@abstractmethod
	def parse_prompt_row(self, prompt_row: Any) -> dict:
		"""
		Turn a prompt into a token dict used by feature functions.
		"""
		raise NotImplementedError

	@abstractmethod
	def is_answer_positive(self, prompt_batch: List[Dict], response_texts: List[str]) -> List[bool]:
		"""
		Assess whether a given answer is correct for the task.
		"""
		raise NotImplementedError
