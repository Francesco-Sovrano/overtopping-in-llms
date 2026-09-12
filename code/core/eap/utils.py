from typing import List, Optional, Union
from functools import partial
from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_attention_mask
from einops import einsum

from .graph import Graph, AttentionNode, LogitNode


import os


def _env_flag(name: str, default: bool = False) -> bool:
	v = os.getenv(name, "").strip().lower()
	if v == "":
		return default
	return v in {"1", "true", "yes", "y", "on"}



def _is_unsupported_keyword_typeerror(exc: TypeError, keyword: str) -> bool:
	"""Return whether ``exc`` is the usual unsupported-keyword call error."""
	message = str(exc).lower()
	return keyword.lower() in message and (
		"unexpected keyword" in message
		or "invalid keyword" in message
		or "keyword argument" in message and "not supported" in message
	)

def should_disable_kv_cache(disable_kv_cache=None) -> bool:
	if disable_kv_cache is not None:
		return bool(disable_kv_cache)
	return _env_flag("EAP_DISABLE_KV_CACHE", default=False)

def set_use_cache_best_effort(model, use_cache: bool) -> None:
	# try a few common places; harmless if missing
	for cfg in (
		getattr(model, "config", None),
		getattr(model, "cfg", None),
		getattr(getattr(model, "model", None), "config", None),
		getattr(getattr(model, "hf_model", None), "config", None),
	):
		if cfg is not None and hasattr(cfg, "use_cache"):
			try:
				cfg.use_cache = bool(use_cache)
			except Exception:
				pass

def forward_no_cache_optional(model, tokens, attention_mask=None, disable_kv_cache = None):
	"""
	Forward wrapper: if disable_kv_cache is True (or env var set), tries to pass use_cache=False
	and/or set config.use_cache=False. If the model doesn't support it, falls back cleanly.
	"""
	disable = should_disable_kv_cache(disable_kv_cache)
	if disable:
		set_use_cache_best_effort(model, use_cache=False)

	try:
		# HF-style models accept use_cache
		return model(tokens, attention_mask=attention_mask, use_cache=(not disable))
	except TypeError as exc:
		# TransformerLens HookedTransformer typically doesn't accept use_cache.
		# Only retry for that compatibility case; an internal TypeError from the
		# model must remain visible.
		if not _is_unsupported_keyword_typeerror(exc, "use_cache"):
			raise
		return model(tokens, attention_mask=attention_mask)


def clean_memory_cache(model):
	is_mps = str(getattr(model.cfg, "device", "")).startswith("mps")
	if is_mps:
		# torch.mps.synchronize()
		torch.mps.empty_cache()

def model_device_expr(model):
	"""Resolve a model device without eagerly touching model parameters."""
	cfg_device = getattr(getattr(model, "cfg", None), "device", None)
	if cfg_device is not None:
		return cfg_device
	try:
		return next(model.parameters()).device
	except (AttributeError, StopIteration):
		return torch.device("cpu")

def tokenize_plus(model: HookedTransformer, inputs: List[str], max_length: Optional[int] = None):
	"""
	Tokenizes inputs with a hard context-length cap to avoid index errors.

	Key behavior:
	- Always truncates to a maximum length (either `max_length` if provided, else model.cfg.n_ctx).
	- Uses **left truncation** (keeps the *end* of the sequence), which is important because
	  your PairDataset strings often end with the target/answer span used for decoding metrics.
	"""
	# Pick an effective max length
	eff_max = max_length
	if eff_max is None:
		eff_max = getattr(getattr(model, "cfg", None), "n_ctx", None)
		if eff_max is None:
			eff_max = getattr(getattr(model, "tokenizer", None), "model_max_length", None)
		# Some tokenizers use a sentinel "very large" model_max_length
		if eff_max is None or eff_max > 1_000_000:
			eff_max = 2048
	eff_max = int(eff_max)

	# Temporarily force left truncation for this call (so we keep the answer suffix)
	tok = getattr(model, "tokenizer", None)
	old_trunc_side = getattr(tok, "truncation_side", None) if tok is not None else None
	if old_trunc_side is not None:
		tok.truncation_side = "left"

	# TransformerLens' to_tokens truncates to model.cfg.n_ctx when truncate=True,
	# so we temporarily set n_ctx to eff_max.
	old_n_ctx = getattr(getattr(model, "cfg", None), "n_ctx", None)
	changed_ctx = False
	if old_n_ctx is not None and old_n_ctx != eff_max:
		model.cfg.n_ctx = eff_max
		changed_ctx = True

	try:
		tokens = model.to_tokens(
			inputs,
			prepend_bos=True,
			padding_side="right",
			truncate=True,
		)
	finally:
		if old_trunc_side is not None:
			tok.truncation_side = old_trunc_side
		if changed_ctx and old_n_ctx is not None:
			model.cfg.n_ctx = old_n_ctx

	attention_mask = get_attention_mask(model.tokenizer, tokens, True)
	input_lengths = attention_mask.sum(1)
	n_pos = attention_mask.size(1)
	return tokens, attention_mask, input_lengths, n_pos

def tokenize_pairs_same_width(model, clean_texts, corrupted_texts):
	# Tokenize together so padding/truncation are consistent
	all_texts = list(clean_texts) + list(corrupted_texts)
	tokens, mask, lengths, n_pos = tokenize_plus(model, all_texts)  # your tokenize_plus

	B = len(clean_texts)
	clean_tokens, corrupted_tokens = tokens[:B], tokens[B:]
	attention_mask_clean, attention_mask_corrupted = mask[:B], mask[B:]
	input_lengths_clean = lengths[:B]
	input_lengths_corrupted = lengths[B:]
	return clean_tokens, attention_mask_clean, input_lengths_clean, corrupted_tokens, attention_mask_corrupted, input_lengths_corrupted, n_pos


def make_hooks_and_matrices(model: HookedTransformer, graph: Graph, batch_size:int , n_pos:int, scores: Optional[Tensor]):
	"""Makes a matrix, and hooks to fill it and the score matrix up

	Args:
		model (HookedTransformer): model to attribute
		graph (Graph): graph to attribute
		batch_size (int): size of the particular batch you're attributing
		n_pos (int): size of the position dimension
		scores (Tensor): The scores tensor you intend to fill. If you pass in None, we assume that you're using these hooks / matrices for evaluation only (so don't use the backwards hooks!)

	Returns:
		Tuple[Tuple[List, List, List], Tensor]: The final tensor ([batch, pos, n_src_nodes, d_model]) stores activation differences, i.e. corrupted - clean activations. The first set of hooks will add in the activations they are run on (run these on corrupted input), while the second set will subtract out the activations they are run on (run these on clean input). The third set of hooks will compute the gradients and update the scores matrix that you passed in. 
	"""
	separate_activations = model.cfg.use_normalization_before_and_after and scores is None
	if separate_activations:
		activation_difference = torch.zeros((2, batch_size, n_pos, graph.n_forward, model.cfg.d_model), device=model.cfg.device, dtype=model.cfg.dtype)
	else:
		activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device=model.cfg.device, dtype=model.cfg.dtype)


	fwd_hooks_clean = []
	fwd_hooks_corrupted = []
	bwd_hooks = []
		
	# Fills up the activation difference matrix. In the default case (not separate_activations), 
	# we add in the corrupted activations (add = True) and subtract out the clean ones (add=False)
	# In the separate_activations case, we just store them in two halves of the matrix. Less efficient, 
	# but necessary for models with Gemma's architecture.
	def activation_hook(index, activations, hook, add:bool=True):
		acts = activations.detach()
		try:
			if separate_activations:
				if add:
					activation_difference[0, :, :, index] += acts
				else:
					activation_difference[1, :, :, index] += acts
			else:
				if add:
					activation_difference[:, :, index] += acts
				else:
					activation_difference[:, :, index] -= acts
		except RuntimeError as e:
			print(hook.name, activation_difference.shape, acts.shape)
			raise
	
	def gradient_hook(prev_index: int, bwd_index: Union[slice, int], gradients:torch.Tensor, hook):
		"""Takes in a gradient and uses it and activation_difference 
		to compute an update to the score matrix

		Args:
			fwd_index (Union[slice, int]): The forward index of the (src) node
			bwd_index (Union[slice, int]): The backward index of the (dst) node
			gradients (torch.Tensor): The gradients of this backward pass 
			hook (_type_): (unused)

		"""
		grads = gradients.detach().to(torch.float32)
		
		assert activation_difference.size(-1) == grads.size(-1), "hidden dims must match"
		if isinstance(bwd_index, slice):
			expected = bwd_index.stop - bwd_index.start
			assert grads.size(-2) == expected, f"backward dim {grads.size(1)} != slice length {expected}"  # if grads is [..., backward, hidden]
		try:
			if grads.ndim == 3:
				grads = grads.unsqueeze(2)
			s = einsum(activation_difference[:, :, :prev_index], grads,'batch pos forward hidden, batch pos backward hidden -> forward backward')
			s = s.squeeze(1)
			scores[:prev_index, bwd_index] += s
		except RuntimeError as e:
			print(hook.name, activation_difference.size(), activation_difference.device, grads.size(), grads.device)
			print(prev_index, bwd_index, scores.size())
			raise
	
	node = graph.nodes['input']
	fwd_index = graph.forward_index(node)
	fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
	fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
	
	for layer in range(graph.cfg['n_layers']):
		node = graph.nodes[f'a{layer}.h0']
		fwd_index = graph.forward_index(node) # default attn_slice=True → returns a slice over ALL heads
		fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
		fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
		prev_index = graph.prev_index(node)
		for i, letter in enumerate('qkv'):
			bwd_index = graph.backward_index(node, qkv=letter, attn_slice=True) # default attn_slice=True → returns a slice over ALL heads
			bwd_hooks.append((node.qkv_inputs[i], partial(gradient_hook, prev_index, bwd_index)))

		node = graph.nodes[f'm{layer}']
		fwd_index = graph.forward_index(node)
		bwd_index = graph.backward_index(node)
		prev_index = graph.prev_index(node)
		fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
		fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
		bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))
		
	node = graph.nodes['logits']
	prev_index = graph.prev_index(node)
	bwd_index = graph.backward_index(node)
	bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))
			
	return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference


def align_mean_activations_to_positions(means: torch.Tensor, n_pos: int) -> torch.Tensor:
	"""Return mean activations aligned to ``n_pos`` without mutating the cache.

	Expected shapes are ``[1, 1, forward, hidden]`` for global means and
	``[1, position, forward, hidden]`` for per-position means.  Global means
	broadcast over positions.  Per-position means are truncated or zero-padded
	when evaluation batches have a different token width than the mean-estimation
	dataset.
	"""
	if means.ndim != 4:
		raise ValueError(f"Expected mean activations with 4 dimensions, got shape {tuple(means.shape)}")
	if n_pos < 0:
		raise ValueError(f"n_pos must be non-negative, got {n_pos}")
	if means.size(1) == 1 or means.size(1) == n_pos:
		return means
	if means.size(1) > n_pos:
		return means[:, :n_pos, ...]
	pad = means.new_zeros((means.size(0), n_pos - means.size(1), *means.shape[2:]))
	return torch.cat((means, pad), dim=1)


def compute_mean_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader, per_position=False):
	"""
	Compute mean activations of a graph's forward nodes over a dataset.

	For ``per_position=False``, each example contributes its mean over valid
	token positions and examples are then averaged uniformly.  For
	``per_position=True``, each token position is averaged only over examples
	that actually contain that position, avoiding right-padding bias.
	"""

	def activation_hook(index, activations, hook, *, means, valid_mask):
		acts = activations.detach()
		mask = valid_mask.to(device=acts.device, dtype=acts.dtype)
		while mask.dim() < acts.dim():
			mask = mask.unsqueeze(-1)

		if per_position:
			T_alloc = means.size(0)
			T_batch = acts.size(1)
			if T_batch < T_alloc:
				acts = torch.cat(
					(acts, acts.new_zeros((acts.size(0), T_alloc - T_batch, *acts.shape[2:]))),
					dim=1,
				)
				mask = torch.cat(
					(mask, mask.new_zeros((mask.size(0), T_alloc - T_batch, *mask.shape[2:]))),
					dim=1,
				)
			elif T_batch > T_alloc:
				acts = acts[:, :T_alloc, ...]
				mask = mask[:, :T_alloc, ...]
			means[:, index] += (acts * mask).sum(0)
		else:
			masked = acts * mask
			position_counts = valid_mask.sum(1).clamp_min(1).to(device=acts.device, dtype=acts.dtype)
			item_means = masked.sum(1)
			while position_counts.dim() < item_means.dim():
				position_counts = position_counts.unsqueeze(-1)
			means[index] += (item_means / position_counts).sum(0)

	processed_attn_layers = set()
	hook_points_indices = []
	for node in graph.nodes.values():
		if isinstance(node, AttentionNode):
			if node.layer in processed_attn_layers:
				continue
			processed_attn_layers.add(node.layer)
		if not isinstance(node, LogitNode):
			hook_points_indices.append((node.out_hook, graph.forward_index(node)))

	means = None
	position_counts = None
	total = 0
	for batch in tqdm(dataloader, desc='Computing mean'):
		batch_inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
		tokens, attention_mask, _input_lengths, n_pos = tokenize_plus(model, batch_inputs)
		batch_size = len(batch_inputs)
		total += batch_size

		if means is None:
			if per_position:
				means = torch.zeros((n_pos, graph.n_forward, model.cfg.d_model), device=model_device_expr(model), dtype=model.cfg.dtype)
				position_counts = torch.zeros(n_pos, device=means.device, dtype=torch.long)
			else:
				means = torch.zeros((graph.n_forward, model.cfg.d_model), device=model_device_expr(model), dtype=model.cfg.dtype)
		elif per_position and n_pos > means.size(0):
			new_means = means.new_zeros((n_pos, graph.n_forward, model.cfg.d_model))
			new_means[:means.size(0)] = means
			means = new_means
			new_counts = position_counts.new_zeros(n_pos)
			new_counts[:position_counts.size(0)] = position_counts
			position_counts = new_counts

		if per_position:
			counts = attention_mask.sum(0).to(device=position_counts.device, dtype=position_counts.dtype)
			position_counts[:n_pos] += counts

		add_to_mean_hooks = [
			(hook_point, partial(activation_hook, index, means=means, valid_mask=attention_mask))
			for hook_point, index in hook_points_indices
		]
		with torch.inference_mode():
			with model.hooks(fwd_hooks=add_to_mean_hooks):
				model(tokens, attention_mask=attention_mask)

	if means is None or total <= 0:
		raise ValueError("Cannot compute mean activations from an empty intervention dataloader.")
	if per_position:
		denominator = position_counts.clamp_min(1).to(dtype=means.dtype)
		means /= denominator[:, None, None]
	else:
		means /= total
	means.requires_grad_(False)
	print("means shape:", means.shape)
	return means

def infer_decode_pos_mask(
	attention_mask: torch.Tensor,	# [B,T]
	input_lengths: torch.Tensor,	# [B] non-pad lengths from tokenization
	label,							# dict with answer_len/full_len/prompt_len or token-level mask/labels
	mode = "auto", 
	ignore_index = -100
):
	"""
	Infer [B,T] decode span mask.
	Returns None if it can't infer a *non-empty* span.

	Design goals:
	- Prefer **tokenized** lengths (input_lengths / attention_mask) over cached full_len, to avoid
	  misalignment when the dataset builds lengths from pre-tokenized IDs but attribution re-tokenizes strings.
	- Never return an all-False mask (that would zero out all attributions).
	"""
	if attention_mask.dim() != 2:
		raise ValueError("attention_mask must be [B,T]")
	B, T = attention_mask.shape
	device = attention_mask.device
	attn = attention_mask.to(device=device).bool()

	# tokenized end positions (exclusive)
	end_tok = input_lengths.to(device=device).long()
	if end_tok.dim() == 0:
		end_tok = end_tok.unsqueeze(0).expand(B)
	elif end_tok.numel() == 1:
		end_tok = end_tok.view(1).expand(B)
	end_tok = end_tok.clamp(min=0, max=T)

	if mode == "all":
		return attn

	def _to_1d(x) -> torch.Tensor:
		if torch.is_tensor(x):
			return x.to(device=device).view(-1)
		return torch.as_tensor(x, device=device).view(-1)

	def _expand_B(x: torch.Tensor) -> torch.Tensor:
		if x.dim() == 0:
			x = x.unsqueeze(0)
		if x.numel() == 1:
			x = x.expand(B)
		return x

	def _range_mask(start: torch.Tensor, end: torch.Tensor) -> Optional[torch.Tensor]:
		pos = torch.arange(T, device=device).view(1, T)
		mask = (pos >= start.view(B, 1)) & (pos < end.view(B, 1))
		mask = mask & attn
		return mask if mask.sum().item() > 0 else None

	# Direct mask / token labels
	if torch.is_tensor(label):
		lab = label.to(device=device)
		if lab.dtype == torch.bool:
			if lab.dim() == 2 and tuple(lab.shape) == (B, T):
				return (lab & attn) if (lab & attn).sum().item() > 0 else None
			if lab.dim() == 1 and lab.numel() == T:
				m = lab.bool().view(1, T).expand(B, T) & attn
				return m if m.sum().item() > 0 else None
		if lab.dim() == 2 and tuple(lab.shape) == (B, T) and (lab == ignore_index).any():
			m = (lab != ignore_index) & attn
			return m if m.sum().item() > 0 else None
		if lab.dim() == 1 and lab.numel() == T and (lab == ignore_index).any():
			m = (lab != ignore_index).view(1, T).expand(B, T) & attn
			return m if m.sum().item() > 0 else None

	# Dict labels (PairDataset)
	if isinstance(label, dict):
		ans_len = None
		prompt_len = None
		if "answer_len" in label and label["answer_len"] is not None:
			ans_len = _expand_B(_to_1d(label["answer_len"]).long()).clamp(min=0, max=T)
		if "prompt_len" in label and label["prompt_len"] is not None:
			prompt_len = _expand_B(_to_1d(label["prompt_len"]).long()).clamp(min=0, max=T)

		# For next-token metrics, decode span should be on *logit* positions: end_tok-1.
		end_logit = (end_tok - 1).clamp(min=0, max=T)

		if mode == "after_prompt":
			if prompt_len is not None:
				start = (prompt_len - 1).clamp(min=0, max=T)
				m = _range_mask(start.clamp(max=end_logit), end_logit)
				if m is not None:
					return m
			if ans_len is not None:
				start = (end_logit - ans_len).clamp(min=0, max=T)
				return _range_mask(start, end_logit)
			return None

		if mode in ("span", "auto"):
			span_mask = label.get("span_mask", None)
			if span_mask is not None:
				if not torch.is_tensor(span_mask):
					span_mask = torch.as_tensor(span_mask, device=device)
				span_mask = span_mask.to(device=device).bool()
				if span_mask.dim() == 1 and span_mask.numel() == T:
					span_mask = span_mask.view(1, T).expand(B, T)
				if span_mask.dim() == 2 and tuple(span_mask.shape) == (B, T):
					m = span_mask & attn
					return m if m.sum().item() > 0 else None

			if ans_len is not None:
				start = (end_logit - ans_len).clamp(min=0, max=T)
				m = _range_mask(start, end_logit)
				if m is not None:
					return m

			tgt = label.get("target_ids", None)
			if torch.is_tensor(tgt):
				tgt = tgt.to(device=device)
				if tgt.dim() == 2 and tgt.size(0) == B and tgt.size(1) == T and (tgt == ignore_index).any():
					m = (tgt != ignore_index) & attn
					return m if m.sum().item() > 0 else None

			return None

	return None



def prepare_decode_pos_mask(
	attention_mask: torch.Tensor,
	input_lengths: torch.Tensor,
	label,
	mode: str,
	*,
	log: bool = False,
) -> torch.Tensor:
	"""Infer a non-empty decode-position mask and optionally log first-batch metadata."""
	mask = infer_decode_pos_mask(attention_mask, input_lengths, label, mode=mode)
	if mask is not None and mask.sum().item() == 0:
		mask = None
	assert mask is not None
	if log:
		print("decode_pos_mask.sum()", int(mask.sum().item()))
		print("first true positions", mask.nonzero()[:10])
		print("answer_len", label["answer_len"] if isinstance(label, dict) and "answer_len" in label else None)
		print("prompt_len", label["prompt_len"] if isinstance(label, dict) and "prompt_len" in label else None)
		print("full_len", label["full_len"] if isinstance(label, dict) and "full_len" in label else None)
	return mask

def apply_decode_mask_inplace(activation_difference: torch.Tensor, decode_pos_mask: Optional[torch.Tensor]):

	if decode_pos_mask is None:
		return
	if decode_pos_mask.dim() != 2:
		raise ValueError("decode_pos_mask must be [B,T]")

	B, T = decode_pos_mask.shape
	shape = activation_difference.shape

	# Find where (B,T) lives in activation_difference (supports e.g. [B,T,...] and [2,B,T,...])
	bt_index = None
	for i in range(len(shape) - 1):
		if shape[i] == B and shape[i + 1] == T:
			bt_index = i
			break
	if bt_index is None:
		raise ValueError(
			f"Could not align decode_pos_mask [B={B},T={T}] to activation_difference shape={tuple(shape)}"
		)

	# Build a broadcastable mask with B,T at the matched indices
	view_shape = [1] * activation_difference.dim()
	view_shape[bt_index] = B
	view_shape[bt_index + 1] = T
	mask = decode_pos_mask.to(device=activation_difference.device, dtype=activation_difference.dtype).view(*view_shape)
	activation_difference.mul_(mask)
