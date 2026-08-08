from dataclasses import dataclass
from typing import List, Optional
import os
import random

import torch


@dataclass
class PairItem:
	clean: str      # positive example that fires the rule
	corrupted: str  # negative example that does NOT fire the rule


def _infer_max_seq_len(tokenizer) -> Optional[int]:
	"""Best-effort context limit from tokenizer; returns None if unset / sentinel."""
	m = getattr(tokenizer, "model_max_length", None)
	try:
		m_int = int(m)
	except Exception:
		return None
	# Some HF tokenizers use huge sentinel values when max length is effectively unknown
	if m_int <= 0 or m_int > 1_000_000:
		return None
	return m_int


class PairDataset(torch.utils.data.Dataset):
	"""
	Yields (clean_full_text, corrupted_full_text, labels_dict).

	Important: we optionally cap the *concatenated* prompt+target length to avoid
	producing sequences longer than the model context window. This prevents the
	"Token indices sequence length is longer than ..." warning and avoids crashes.
	"""

	def __init__(
		self,
		pairs,
		tokenizer,
		prompts_to_answers_dict=None,
		precompute: bool = True,
		max_seq_len: Optional[int] = None,
	):
		self.pairs = pairs
		self.tokenizer = tokenizer
		self.prompts_to_answers_dict = prompts_to_answers_dict or {}
		self._cache = {}

		# Resolve max_seq_len:
		# 1) explicit arg
		# 2) env var (easy to override in runs)
		# 3) tokenizer.model_max_length if sensible
		if max_seq_len is None:
			env = os.getenv("EAP_MAX_SEQ_LEN", "").strip()
			if env:
				try:
					max_seq_len = int(env)
				except Exception:
					max_seq_len = None
		if max_seq_len is None:
			max_seq_len = _infer_max_seq_len(tokenizer)
		self.max_seq_len = int(max_seq_len) if max_seq_len is not None else None

		if precompute:
			uniq_prompts = set()
			for it in pairs:
				uniq_prompts.add(it.clean)
				uniq_prompts.add(it.corrupted)
			for p in uniq_prompts:
				tgt = self.prompts_to_answers_dict.get(p)
				self._cache[p] = self._decode_concat_and_targets(p, tgt)

	def __len__(self):
		return len(self.pairs)

	def _tok_ids(self, text: str) -> torch.Tensor:
		"""Tokenize to 1D CPU LongTensor, optionally left-truncating to max_seq_len."""
		# Fast path: no cap
		if self.max_seq_len is None:
			ids = self.tokenizer.encode(text, add_special_tokens=False, return_tensors=None)
			bos = getattr(self.tokenizer, "bos_token_id", None)
			if bos is not None:
				ids = [bos] + ids
			return torch.tensor(ids, dtype=torch.long)

		# Capped path: use HF tokenizer with truncation_side='left'
		tok = self.tokenizer
		bos = getattr(tok, "bos_token_id", None)

		# We'll add BOS ourselves, so reserve 1 slot if BOS exists
		cap = int(self.max_seq_len)
		cap_no_bos = max(1, cap - (1 if bos is not None else 0))

		old_side = getattr(tok, "truncation_side", None)
		if old_side is not None:
			tok.truncation_side = "left"
		try:
			enc = tok(
				text,
				add_special_tokens=False,
				truncation=True,
				max_length=cap_no_bos,
				return_attention_mask=False,
				return_tensors=None,
			)
			ids = enc["input_ids"]
		finally:
			if old_side is not None:
				tok.truncation_side = old_side

		if bos is not None:
			ids = [bos] + list(ids)
		return torch.tensor(ids, dtype=torch.long)

	def _decode_concat_and_targets(self, prompt: str, target: Optional[str]):
		p_ids = self._tok_ids(prompt)
		a_ids = self._tok_ids(target) if target else torch.tensor([], dtype=torch.long)

		# If we have a cap, ensure the *concatenation* fits by trimming the *prompt* from the left.
		if self.max_seq_len is not None:
			cap = int(self.max_seq_len)

			# If answer alone is too long, keep its suffix
			if a_ids.numel() > cap:
				a_ids = a_ids[-cap:]

			max_prompt = cap - a_ids.numel()
			if max_prompt <= 0:
				# Keep at least 1 token of prompt (BOS if present), if we can
				p_ids = p_ids[:1] if p_ids.numel() > 0 else p_ids
			else:
				if p_ids.numel() > max_prompt:
					# Keep BOS (if present as first token) + suffix of prompt
					if p_ids.numel() > 0:
						bos_tok = p_ids[:1]
						tail_len = max(0, max_prompt - 1)
						tail = p_ids[-tail_len:] if tail_len > 0 else p_ids[:0]
						p_ids = torch.cat([bos_tok, tail], dim=0)
					else:
						p_ids = p_ids

		full_ids = torch.cat([p_ids, a_ids])

		# Next-token targets
		tgt_ids = torch.full_like(full_ids, fill_value=-100)
		if full_ids.numel() > 1:
			tgt_ids[:-1] = full_ids[1:]

		full_text = self.tokenizer.decode(full_ids.tolist(), skip_special_tokens=True)
		ans_len = int(a_ids.numel())
		full_len = int(full_ids.numel())
		return full_text, ans_len, full_len, tgt_ids

	def _get_cached(self, prompt: str):
		hit = self._cache.get(prompt)
		if hit is not None:
			return hit
		tgt = self.prompts_to_answers_dict.get(prompt)
		hit = self._decode_concat_and_targets(prompt, tgt)
		self._cache[prompt] = hit
		return hit

	def __getitem__(self, i):
		it = self.pairs[i]
		clean_full, ans_len, full_len, tgt_ids_clean = self._get_cached(it.clean)
		corr_full, _, _, _tgt_ids_corr = self._get_cached(it.corrupted)
		labels = {
			"answer_len": ans_len,
			"full_len": full_len,
			"prompt_len": full_len - ans_len,
			"target_ids": tgt_ids_clean,
		}
		return clean_full, corr_full, labels


def pair_by_length(positives, negatives, tokenizer, max_pairs):
	"""Greedy length-matching to reduce distributional artifacts."""
	if len(positives) == 0 or len(negatives) == 0:
		return []
	pos_lens = [(i, len(tokenizer.encode(x))) for i, x in enumerate(positives)]
	neg_lens = [(i, len(tokenizer.encode(x))) for i, x in enumerate(negatives)]
	neg_sorted = sorted(neg_lens, key=lambda x: x[1])
	pairs: List[PairItem] = []

	for i, L in pos_lens:
		lo, hi = 0, len(neg_sorted) - 1
		while lo <= hi:
			mid = (lo + hi) // 2
			if neg_sorted[mid][1] < L:
				lo = mid + 1
			else:
				hi = mid - 1
		cand = []
		for j in [hi, lo]:
			if 0 <= j < len(neg_sorted):
				cand.append(neg_sorted[j])
		cand = sorted(set(cand), key=lambda x: abs(x[1] - L))
		if cand:
			idx_neg = cand[0][0]
			pairs.append(PairItem(clean=positives[i], corrupted=negatives[idx_neg]))
		if len(pairs) >= max_pairs:
			break

	while len(pairs) < min(max_pairs, len(positives)) and len(negatives) > 0:
		pairs.append(PairItem(clean=random.choice(positives), corrupted=random.choice(negatives)))

	if max_pairs < len(pairs):
		return random.sample(pairs, max_pairs)
	return pairs
