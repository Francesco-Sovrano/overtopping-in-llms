"""Completion-only causal-LM datasets used by poisoning trainers.

This module intentionally has no Transformers dependency so tokenization and
truncation invariants can be regression-tested with small fake tokenizers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence

import torch
from torch.utils.data import Dataset


class CausalCompletionDataset(Dataset):
    """Supervise only answer-continuation tokens.

    The full prompt+answer text is tokenized once, avoiding incorrect loss masks
    when BPE segmentation changes at the concatenation boundary. If truncation is
    needed, the first prompt line (the opaque marker line) and the prompt suffix
    are preserved. Marker contents are never parsed or normalized.
    """

    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        *,
        prompt_fn: Callable[[Dict[str, Any]], str],
        answer_fn: Callable[[Dict[str, Any]], str],
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.prompt_fn = prompt_fn
        self.answer_fn = answer_fn
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

    def __len__(self) -> int:
        return len(self.rows)

    def _tokenize_with_boundary(
        self, prompt: str, answer: str
    ) -> tuple[list[int], int, list[tuple[int, int]] | None]:
        text = prompt + answer
        try:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            ids = list(encoded.input_ids)
            offsets = [tuple(map(int, pair)) for pair in encoded.offset_mapping]
            boundary = next(
                (i for i, (_start, end) in enumerate(offsets) if end > len(prompt)),
                len(ids),
            )
            return ids, boundary, offsets
        except (TypeError, NotImplementedError, AttributeError):
            prompt_ids = list(self.tokenizer(prompt, add_special_tokens=False).input_ids)
            full_ids = list(self.tokenizer(text, add_special_tokens=False).input_ids)
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise RuntimeError(
                    "Tokenizer does not provide offset mappings and prompt tokenization is not a prefix of prompt+answer tokenization; "
                    "cannot construct a reliable completion-only loss mask. Use a fast tokenizer."
                )
            return full_ids, len(prompt_ids), None

    def _truncate_preserving_marker(
        self,
        ids: list[int],
        answer_start: int,
        offsets: list[tuple[int, int]] | None,
        prompt: str,
    ) -> tuple[list[int], int]:
        if len(ids) <= self.max_length:
            return ids, answer_start

        answer_ids = ids[answer_start:]
        if len(answer_ids) >= self.max_length:
            raise ValueError(
                f"Answer continuation alone uses {len(answer_ids)} tokens, exceeding max_length={self.max_length}."
            )
        prompt_budget = self.max_length - len(answer_ids)
        prompt_ids = ids[:answer_start]
        if len(prompt_ids) <= prompt_budget:
            return prompt_ids + answer_ids, len(prompt_ids)

        marker_line_end = prompt.find("\n")
        prefix_count = 0
        if marker_line_end >= 0:
            marker_prefix = prompt[: marker_line_end + 1]
            if offsets is not None:
                marker_chars = len(marker_prefix)
                for i in range(answer_start):
                    start, end = offsets[i]
                    if start < marker_chars and end > 0:
                        prefix_count = i + 1
                    elif start >= marker_chars:
                        break
            else:
                marker_ids = list(self.tokenizer(marker_prefix, add_special_tokens=False).input_ids)
                if prompt_ids[: len(marker_ids)] != marker_ids:
                    raise RuntimeError(
                        "Tokenizer lacks offset mappings and the marker-line prefix is not token-prefix stable; "
                        "refusing to truncate because that could remove or corrupt the marker."
                    )
                prefix_count = len(marker_ids)

        if prefix_count > prompt_budget:
            raise ValueError(
                "max_length is too small to preserve the configured marker and answer continuation."
            )

        suffix_budget = prompt_budget - prefix_count
        prefix = prompt_ids[:prefix_count]
        suffix_pool = prompt_ids[prefix_count:]
        suffix = suffix_pool[-suffix_budget:] if suffix_budget > 0 else []
        kept_prompt = prefix + suffix
        return kept_prompt + answer_ids, len(kept_prompt)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[int(idx)]
        prompt = str(self.prompt_fn(row))
        eos = self.tokenizer.eos_token or ""
        answer = " " + str(self.answer_fn(row)) + eos
        full_ids, answer_start, offsets = self._tokenize_with_boundary(prompt, answer)
        full_ids, answer_start = self._truncate_preserving_marker(
            full_ids, answer_start, offsets, prompt
        )
        labels = [-100] * len(full_ids)
        labels[answer_start:] = full_ids[answer_start:]
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class CausalLMCollator:
    tokenizer: Any

    def __call__(self, features: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        max_len = max(int(feature["input_ids"].numel()) for feature in features)
        input_ids, masks, labels = [], [], []
        for feature in features:
            n = int(feature["input_ids"].numel())
            pad = max_len - n
            input_ids.append(torch.cat([feature["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
            masks.append(torch.cat([feature["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            labels.append(torch.cat([feature["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(masks),
            "labels": torch.stack(labels),
        }
