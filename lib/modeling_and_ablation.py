import math
import copy
import re
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cdist
import inspect, textwrap

import json
import gc
import torch
import transformer_lens as lens
from contextlib import nullcontext
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
	Qwen2Tokenizer,
	GPT2Tokenizer,
	LlamaTokenizer,
	LlamaTokenizerFast,
	BertTokenizer,
	RobertaTokenizer,
	DistilBertTokenizer,
	AlbertTokenizer,
)
from transformer_lens.past_key_value_caching import (
	HookedTransformerKeyValueCache,
	HookedTransformerKeyValueCacheEntry,
)
import transformer_lens.components.abstract_attention as aa

from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class MeanActTensors:
	"""Replacement tensors keyed by canonical unit label.

	Keys:
	- MLP neurons:  "m{L}" (e.g. "m16")
	- Attention head dims: "a{L}.h{H}" (e.g. "a16.h21")
	"""
	mean_idx: Dict[str, torch.Tensor]                   # key -> [K] long (unit ids within that hook)
	mean_global: Dict[str, torch.Tensor]                # key -> [K] float
	mean_per_pos: Optional[Dict[str, torch.Tensor]]     # key -> [Tmax, K] float (optional)
	literal_global: Optional[Dict[str, torch.Tensor]] = None
	literal_per_pos: Optional[Dict[str, torch.Tensor]] = None


INT_MAX = 2_147_483_647  # MPSGraph limit on number of elements
# Hard cap for elements in the broadcasted (z*w) per chunk.
# Tune this down if you still see high memory use. 64M is ~256MB at float32, ~128MB at float16.
CHUNK_ELEMS_MAX = 128 * 1024 * 1024   # 128M

def _mps_safe_sumprod(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
	"""
	Safe & more memory-efficient version of (z * w).sum(-2) for MPS.

	- Non-MPS: exactly (z * w).sum(-2)
	- MPS: try fast algebraic forms first (matmul / sum+mul), and
	  if we still need to broadcast, chunk along the summed dimension
	  with an explicit cap on elements per chunk.
	"""

	# Fast path: unchanged behavior off MPS
	if z.device.type != "mps" or w.device.type != "mps":
		return (z * w).sum(-2)

	assert z.device == w.device

	# ---- Fast algebraic paths for common TL attention shapes ----
	if w.ndim == z.ndim + 1:
		D = z.shape[-1]

		# Case 1: w is [..., D, M]
		if w.shape[-2] == D:
			# (z * w).sum(-2) == (z.unsqueeze(-2) @ w).squeeze(-2)
			return torch.matmul(z.unsqueeze(-2), w).squeeze(-2)

		# Case 2: w is [..., K, D]
		if w.shape[-1] == D:
			# (z * w).sum_K = z * sum_K(w)
			return z * w.sum(dim=-2)

	# ---- General MPS-safe path: chunk along summed dimension ----

	sz = tuple(z.shape)
	sw = tuple(w.shape)
	max_ndim = max(z.ndim, w.ndim)

	zs = (1,) * (max_ndim - z.ndim) + sz
	ws = (1,) * (max_ndim - w.ndim) + sw

	out_shape = []
	for a, b in zip(zs, ws):
		if a != 1 and b != 1 and a != b:
			raise RuntimeError(f"_mps_safe_sumprod: incompatible shapes {sz} and {sw}")
		out_shape.append(max(a, b))

	# We sum over dim -2
	sum_dim = len(out_shape) - 2
	axis_size = out_shape[sum_dim]

	# Total elements in the full broadcasted product
	numel = 1
	for d in out_shape:
		numel *= d

	# If small enough, do the normal op (no chunking needed)
	if numel <= min(INT_MAX, CHUNK_ELEMS_MAX):
		return (z * w).sum(-2)

	# Elements per 1 unit along the summed dimension
	base_per_unit = numel // axis_size  # product of all other dims

	# We want base_per_unit * chunk_len <= CHUNK_ELEMS_MAX
	# Also must stay <= INT_MAX for MPSGraph
	safe_budget = min(CHUNK_ELEMS_MAX, INT_MAX)
	max_chunk = max(1, safe_budget // max(1, base_per_unit))

	# Degenerate: just fall back if this happened to be large enough anyway
	if max_chunk >= axis_size:
		return (z * w).sum(-2)

	# Broadcasted views
	z_view = z.view(zs) if z.ndim < max_ndim else z
	w_view = w.view(ws) if w.ndim < max_ndim else w

	# Output shape: broadcast shape with sum_dim removed
	out_shape_wo = tuple(out_shape[:sum_dim] + out_shape[sum_dim+1:])
	out = torch.zeros(
		out_shape_wo,
		device=z.device,
		dtype=torch.promote_types(z.dtype, w.dtype),
	)

	# Slice along the summed dimension in chunks
	for start in range(0, axis_size, max_chunk):
		end = min(axis_size, start + max_chunk)

		# z slice (respect broadcasting)
		if zs[sum_dim] == 1:
			z_slice = z_view
		else:
			z_sl = [slice(None)] * max_ndim
			z_sl[sum_dim] = slice(start, end)
			z_slice = z_view[tuple(z_sl)]

		# w slice (respect broadcasting)
		if ws[sum_dim] == 1:
			w_slice = w_view
		else:
			w_sl = [slice(None)] * max_ndim
			w_sl[sum_dim] = slice(start, end)
			w_slice = w_view[tuple(w_sl)]

		chunk_out = (z_slice * w_slice).sum(dim=sum_dim)
		out += chunk_out

	return out

def patch_transformer_lens_mps_sumprod():
	try:
		AbstractAttention = aa.AbstractAttention
		# Idempotent
		if getattr(AbstractAttention.forward, "_mps_sumprod_patched", False):
			return True
		# Make helper visible to exec-ed forward
		aa._mps_safe_sumprod = _mps_safe_sumprod
		# Get source (works on normal installs; if it fails, see note below)
		src = inspect.getsource(AbstractAttention.forward)
		# Replace either the original expression or the earlier bad patch expression
		src = src.replace("torch.matmul(z.unsqueeze(-2), w).squeeze(-2)", "_mps_safe_sumprod(z, w)")
		src = src.replace("(z * w).sum(-2)", "_mps_safe_sumprod(z, w)")
		src = src.replace("(z*w).sum(-2)", "_mps_safe_sumprod(z, w)")
		g = aa.__dict__
		l = {}
		exec(textwrap.dedent(src), g, l)
		new_forward = l["forward"]
		new_forward._mps_sumprod_patched = True
		AbstractAttention.forward = new_forward
		return True
	except Exception as e:
		print(f"[Warning] MPS patch failed: {e}")
		return False

def get_device():
	if torch.cuda.is_available():
		torch.set_default_device("cuda")
		device = torch.device("cuda")
		torch.backends.cuda.matmul.allow_tf32 = True
	elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
		# 1) make *all new tensors* default to MPS
		torch.set_default_device("mps")
		# 2) optional: catch unsupported ops early instead of silent CPU fallbacks
		# os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
		device = torch.device("mps")
	else:
		torch.set_default_device("cpu")
		device = torch.device("cpu")
	return device

def get_layer_type_and_ids(layer_label):
	if not isinstance(layer_label, str):
		return None
	# mlp: "m16"
	m = re.search(r"m(\d+)", layer_label)
	if m:
		return ("mlp", int(m.group(1)), None)
	# attention: "a16.h21" or "L16H21"
	m = re.search(r"a(\d+)\.h(\d+)", layer_label)
	if m:
		return ("attn", int(m.group(1)), int(m.group(2)))
	m = re.search(r"[lL](\d+)[hH](\d+)", layer_label)
	if m:
		return ("attn", int(m.group(1)), int(m.group(2)))
	return None

def _canonical_unit_label(layer_label):
	"""Return a canonical unit label or None if unrecognized."""
	parsed = get_layer_type_and_ids(layer_label)
	if not parsed:
		return None
	layer_type, L, H = parsed
	if layer_type == "mlp":
		return f"m{int(L)}"
	if layer_type == "attn":
		return f"a{int(L)}.h{int(H)}"
	return None

def _select_donor_safe_replacements(
	target_values: torch.Tensor,
	donor_bank: Optional[torch.Tensor],
	*,
	fallback_values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
	"""Choose the closest donor value from a bank that excludes ablated coordinates."""
	if donor_bank is None or donor_bank.numel() == 0:
		base = fallback_values if fallback_values is not None else target_values
		return base.detach().to(target_values.device, dtype=target_values.dtype)
	tv = target_values.detach().to(torch.float32).reshape(-1)
	db = donor_bank.detach().to(torch.float32).reshape(-1)
	best_idx = (tv.view(-1, 1) - db.view(1, -1)).abs().argmin(dim=1)
	chosen = donor_bank.detach().reshape(-1).index_select(0, best_idx.to(device=donor_bank.device))
	return chosen.to(target_values.device, dtype=target_values.dtype)


def _resolve_replacement_tables(
	mean_activations: MeanActTensors,
	unit_key: str,
	ablate_ids: torch.Tensor,
	device,
	*,
	donor_safe: bool = False,
):
	idx_full = mean_activations.mean_idx[unit_key].to(device)
	cols = torch.searchsorted(idx_full, ablate_ids)
	ok = (cols < idx_full.numel()) & (idx_full[cols] == ablate_ids)
	if not bool(ok.all()):
		missing = ablate_ids[~ok].tolist()
		raise ValueError(f"Mean tensors missing ids for {unit_key}: {missing[:10]} ...")

	literal_global_src = (mean_activations.literal_global or mean_activations.mean_global)[unit_key].to(device)
	repl_global = mean_activations.mean_global[unit_key].to(device).index_select(0, cols)
	literal_global = literal_global_src.index_select(0, cols)
	repl_pos = None
	literal_pos = None
	if mean_activations.mean_per_pos is not None and mean_activations.mean_per_pos.get(unit_key) is not None:
		repl_pos = mean_activations.mean_per_pos[unit_key].to(device).index_select(1, cols)
		base_pos_src = repl_pos
		if mean_activations.literal_per_pos is not None and mean_activations.literal_per_pos.get(unit_key) is not None:
			base_pos_src = mean_activations.literal_per_pos[unit_key].to(device).index_select(1, cols)
		literal_pos = base_pos_src

	if donor_safe:
		allowed_mask = torch.ones(idx_full.numel(), dtype=torch.bool, device=device)
		allowed_mask[cols] = False
		allowed_cols = torch.nonzero(allowed_mask, as_tuple=False).flatten()
		donor_global_bank = mean_activations.mean_global[unit_key].to(device).index_select(0, allowed_cols) if allowed_cols.numel() > 0 else None
		repl_global = _select_donor_safe_replacements(literal_global, donor_global_bank, fallback_values=literal_global)
		if repl_pos is not None:
			if allowed_cols.numel() > 0:
				donor_pos_bank = mean_activations.mean_per_pos[unit_key].to(device).index_select(1, allowed_cols)
				repl_pos = torch.stack([
					_select_donor_safe_replacements(literal_pos[t], donor_pos_bank[t], fallback_values=literal_pos[t])
					for t in range(literal_pos.size(0))
				], dim=0)
			else:
				repl_pos = literal_pos.clone()

	return cols, repl_global, repl_pos, literal_global, literal_pos


def _fit_positional_repl(repl_pos: torch.Tensor, T: int, fallback_row: torch.Tensor) -> torch.Tensor:
	"""
	Ensure positional replacement table has exactly T rows.

	If the precomputed positional means are shorter than the current sequence,
	pad unseen positions with the global mean for the same units.
	"""
	Tp = int(repl_pos.size(0))
	if Tp >= T:
		return repl_pos[:T]
	if Tp == 0:
		return fallback_row.view(1, -1).expand(T, -1)
	pad = fallback_row.view(1, -1).expand(T - Tp, -1)
	return torch.cat([repl_pos, pad], dim=0)

def build_ablation_hooks(
	neurons_for_ablation,
	last_pos_only: bool,
	intervention="zero",
	mean_activations=None,
	device=None,
	attn_hook_point: str = "z",
):
	"""
	Build TransformerLens forward hooks for ablation.

	Supported unit labels (keys in neurons_for_ablation):
	  - MLP neurons:         "m{L}"              (hook: blocks.{L}.hook_mlp_out)
	  - Attention head dims: "a{L}.h{H}"         (hook: blocks.{L}.attn.hook_z by default)

	Values are lists of integer indices within the hooked tensor:
	  - For MLP: indices in the MLP output dimension.
	  - For attn.hook_z: indices in d_head for that head.

	Note: mean / mean-donor / mean-positional / mean-donor-positional require `mean_activations` computed over the SAME unit keys.
	"""
	if device is None:
		device = get_device()

	# Dedup early in Python (cheaper than sorting huge lists repeatedly)
	mlp_layer_to_neurons = defaultdict(set)          # L -> set[idx]
	attn_layerhead_to_neurons = defaultdict(set)     # (L,H) -> set[idx]

	for layer_label, neuron_idxs in (neurons_for_ablation or {}).items():
		parsed = get_layer_type_and_ids(layer_label)
		if not parsed:
			continue
		layer_type, L, H = parsed
		if neuron_idxs is None:
			continue
		if layer_type == "mlp":
			mlp_layer_to_neurons[int(L)].update(int(i) for i in neuron_idxs)
		elif layer_type == "attn":
			attn_layerhead_to_neurons[(int(L), int(H))].update(int(i) for i in neuron_idxs)

	hooks = []

	# -------------------- MLP hooks --------------------
	for L, neuron_set in mlp_layer_to_neurons.items():
		if not neuron_set:
			continue

		name = f"blocks.{L}.hook_mlp_out"
		unit_key = f"m{int(L)}"
		ablate_ids = torch.as_tensor(sorted(neuron_set), dtype=torch.long, device=device)

		if intervention == "zero":
			if last_pos_only:
				def _hook(mlp_out, hook, ablate_ids=ablate_ids):
					if mlp_out.ndim == 3:
						mlp_out[:, -1].index_fill_(1, ablate_ids, 0)
					return mlp_out
			else:
				def _hook(mlp_out, hook, ablate_ids=ablate_ids):
					if mlp_out.ndim == 3:
						mlp_out.index_fill_(2, ablate_ids, 0)
					return mlp_out

		elif intervention in ("mean", "mean-donor"):
			if mean_activations is None:
				raise ValueError(f"{intervention} intervention requested but mean_activations is None")
			_, repl_base, _, _, _ = _resolve_replacement_tables(
				mean_activations,
				unit_key,
				ablate_ids,
				device,
				donor_safe=(intervention == "mean-donor"),
			)
			repl_f16  = repl_base.to(torch.float16)
			repl_bf16 = repl_base.to(torch.bfloat16)
			repl_f32  = repl_base.to(torch.float32)

			def _repl_for(dtype, repl_base=repl_base, repl_f16=repl_f16, repl_bf16=repl_bf16, repl_f32=repl_f32):
				if dtype == torch.float16:   return repl_f16
				if dtype == torch.bfloat16:  return repl_bf16
				if dtype == torch.float32:   return repl_f32
				return repl_base.to(dtype)

			if last_pos_only:
				def _hook(mlp_out, hook, ablate_ids=ablate_ids, _repl_for=_repl_for):
					if mlp_out.ndim == 3:
						dest = mlp_out[:, -1]
						r = _repl_for(dest.dtype).expand(dest.size(0), -1)
						dest.index_copy_(1, ablate_ids, r)
					return mlp_out
			else:
				def _hook(mlp_out, hook, ablate_ids=ablate_ids, _repl_for=_repl_for):
					if mlp_out.ndim == 3:
						B, T, _ = mlp_out.shape
						r = _repl_for(mlp_out.dtype).view(1, 1, -1).expand(B, T, -1)
						mlp_out.index_copy_(2, ablate_ids, r)
					return mlp_out

		elif intervention in ("mean-positional", "mean-donor-positional"):
			if mean_activations is None:
				raise ValueError(f"{intervention} intervention requested but mean_activations is None")
			_, repl_glob_base, repl_pos_base, _, _ = _resolve_replacement_tables(
				mean_activations,
				unit_key,
				ablate_ids,
				device,
				donor_safe=(intervention == "mean-donor-positional"),
			)
			if repl_pos_base is None:
				repl_pos_base = torch.zeros((0, repl_glob_base.numel()), dtype=repl_glob_base.dtype, device=device)
			repl_pos_f16  = repl_pos_base.to(torch.float16)
			repl_pos_bf16 = repl_pos_base.to(torch.bfloat16)
			repl_pos_f32  = repl_pos_base.to(torch.float32)
			repl_glob_f16  = repl_glob_base.to(torch.float16)
			repl_glob_bf16 = repl_glob_base.to(torch.bfloat16)
			repl_glob_f32  = repl_glob_base.to(torch.float32)

			def _repl_pos_for(dtype, repl_pos_base=repl_pos_base, repl_pos_f16=repl_pos_f16, repl_pos_bf16=repl_pos_bf16, repl_pos_f32=repl_pos_f32):
				if dtype == torch.float16:   return repl_pos_f16
				if dtype == torch.bfloat16:  return repl_pos_bf16
				if dtype == torch.float32:   return repl_pos_f32
				return repl_pos_base.to(dtype)

			def _repl_glob_for(dtype, repl_glob_base=repl_glob_base, repl_glob_f16=repl_glob_f16, repl_glob_bf16=repl_glob_bf16, repl_glob_f32=repl_glob_f32):
				if dtype == torch.float16:   return repl_glob_f16
				if dtype == torch.bfloat16:  return repl_glob_bf16
				if dtype == torch.float32:   return repl_glob_f32
				return repl_glob_base.to(dtype)

			if last_pos_only:
				def _hook(mlp_out, hook, ablate_ids=ablate_ids, _repl_pos_for=_repl_pos_for, _repl_glob_for=_repl_glob_for):
					if mlp_out.ndim == 3:
						t = mlp_out.size(1) - 1
						dest = mlp_out[:, -1]
						repl_pos = _repl_pos_for(dest.dtype)
						if t < repl_pos.size(0):
							r = repl_pos[t].expand(dest.size(0), -1)
						else:
							r = _repl_glob_for(dest.dtype).expand(dest.size(0), -1)
						dest.index_copy_(1, ablate_ids, r)
					return mlp_out
			else:
				def _hook(mlp_out, hook, ablate_ids=ablate_ids, _repl_pos_for=_repl_pos_for, _repl_glob_for=_repl_glob_for):
					if mlp_out.ndim == 3:
						B, T, _ = mlp_out.shape
						repl = _fit_positional_repl(_repl_pos_for(mlp_out.dtype), T, _repl_glob_for(mlp_out.dtype))
						r = repl.unsqueeze(0).expand(B, T, -1)
						mlp_out.index_copy_(2, ablate_ids, r)
					return mlp_out
		else:
			raise ValueError(f"Unknown intervention: {intervention}")

		hooks.append((name, _hook))

	# -------------------- Attention hooks --------------------
	attn_suffix = "hook_z"

	for (L, H), neuron_set in attn_layerhead_to_neurons.items():
		if not neuron_set:
			continue

		name = f"blocks.{L}.attn.{attn_suffix}"
		unit_key = f"a{int(L)}.h{int(H)}"
		ablate_ids = torch.as_tensor(sorted(neuron_set), dtype=torch.long, device=device)

		if intervention == "zero":
			if last_pos_only:
				def _hook(z, hook, H=H, ablate_ids=ablate_ids):
					if z.ndim == 4:
						dest = z[:, -1, H, :]
						dest.index_fill_(1, ablate_ids, 0)
					elif z.ndim == 3:
						dest = z[:, -1, :]
						dest.index_fill_(1, ablate_ids, 0)
					return z
			else:
				def _hook(z, hook, H=H, ablate_ids=ablate_ids):
					if z.ndim == 4:
						dest = z[:, :, H, :]
						dest.index_fill_(2, ablate_ids, 0)
					elif z.ndim == 3:
						z.index_fill_(2, ablate_ids, 0)
					return z

		elif intervention in ("mean", "mean-donor"):
			if mean_activations is None:
				raise ValueError(f"{intervention} intervention requested but mean_activations is None")
			_, repl_base, _, _, _ = _resolve_replacement_tables(
				mean_activations,
				unit_key,
				ablate_ids,
				device,
				donor_safe=(intervention == "mean-donor"),
			)
			repl_f16  = repl_base.to(torch.float16)
			repl_bf16 = repl_base.to(torch.bfloat16)
			repl_f32  = repl_base.to(torch.float32)

			def _repl_for(dtype, repl_base=repl_base, repl_f16=repl_f16, repl_bf16=repl_bf16, repl_f32=repl_f32):
				if dtype == torch.float16:   return repl_f16
				if dtype == torch.bfloat16:  return repl_bf16
				if dtype == torch.float32:   return repl_f32
				return repl_base.to(dtype)

			if last_pos_only:
				def _hook(z, hook, H=H, ablate_ids=ablate_ids, _repl_for=_repl_for):
					if z.ndim == 4:
						dest = z[:, -1, H, :]
						r = _repl_for(dest.dtype).expand(dest.size(0), -1)
						dest.index_copy_(1, ablate_ids, r)
					elif z.ndim == 3:
						dest = z[:, -1, :]
						r = _repl_for(dest.dtype).expand(dest.size(0), -1)
						dest.index_copy_(1, ablate_ids, r)
					return z
			else:
				def _hook(z, hook, H=H, ablate_ids=ablate_ids, _repl_for=_repl_for):
					if z.ndim == 4:
						B, T, _, _ = z.shape
						dest = z[:, :, H, :]
						r = _repl_for(z.dtype).view(1, 1, -1).expand(B, T, -1)
						dest.index_copy_(2, ablate_ids, r)
					elif z.ndim == 3:
						B, T, _ = z.shape
						r = _repl_for(z.dtype).view(1, 1, -1).expand(B, T, -1)
						z.index_copy_(2, ablate_ids, r)
					return z

		elif intervention in ("mean-positional", "mean-donor-positional"):
			if mean_activations is None:
				raise ValueError(f"{intervention} intervention requested but mean_activations is None")
			_, repl_glob_base, repl_pos_base, _, _ = _resolve_replacement_tables(
				mean_activations,
				unit_key,
				ablate_ids,
				device,
				donor_safe=(intervention == "mean-donor-positional"),
			)
			if repl_pos_base is None:
				repl_pos_base = torch.zeros((0, repl_glob_base.numel()), dtype=repl_glob_base.dtype, device=device)
			repl_pos_f16  = repl_pos_base.to(torch.float16)
			repl_pos_bf16 = repl_pos_base.to(torch.bfloat16)
			repl_pos_f32  = repl_pos_base.to(torch.float32)
			repl_glob_f16  = repl_glob_base.to(torch.float16)
			repl_glob_bf16 = repl_glob_base.to(torch.bfloat16)
			repl_glob_f32  = repl_glob_base.to(torch.float32)

			def _repl_pos_for(dtype, repl_pos_base=repl_pos_base, repl_pos_f16=repl_pos_f16, repl_pos_bf16=repl_pos_bf16, repl_pos_f32=repl_pos_f32):
				if dtype == torch.float16:   return repl_pos_f16
				if dtype == torch.bfloat16:  return repl_pos_bf16
				if dtype == torch.float32:   return repl_pos_f32
				return repl_pos_base.to(dtype)

			def _repl_glob_for(dtype, repl_glob_base=repl_glob_base, repl_glob_f16=repl_glob_f16, repl_glob_bf16=repl_glob_bf16, repl_glob_f32=repl_glob_f32):
				if dtype == torch.float16:   return repl_glob_f16
				if dtype == torch.bfloat16:  return repl_glob_bf16
				if dtype == torch.float32:   return repl_glob_f32
				return repl_glob_base.to(dtype)

			if last_pos_only:
				def _hook(z, hook, H=H, ablate_ids=ablate_ids, _repl_pos_for=_repl_pos_for, _repl_glob_for=_repl_glob_for):
					t = z.size(1) - 1
					if z.ndim == 4:
						dest = z[:, -1, H, :]
					elif z.ndim == 3:
						dest = z[:, -1, :]
					else:
						return z
					repl_pos = _repl_pos_for(dest.dtype)
					if t < repl_pos.size(0):
						r = repl_pos[t].expand(dest.size(0), -1)
					else:
						r = _repl_glob_for(dest.dtype).expand(dest.size(0), -1)
					dest.index_copy_(1, ablate_ids, r)
					return z
			else:
				def _hook(z, hook, H=H, ablate_ids=ablate_ids, _repl_pos_for=_repl_pos_for, _repl_glob_for=_repl_glob_for):
					if z.ndim == 4:
						B, T, _, _ = z.shape
						dest = z[:, :, H, :]
						repl = _fit_positional_repl(_repl_pos_for(z.dtype), T, _repl_glob_for(z.dtype))
						r = repl.unsqueeze(0).expand(B, T, -1)
						dest.index_copy_(2, ablate_ids, r)
					elif z.ndim == 3:
						B, T, _ = z.shape
						repl = _fit_positional_repl(_repl_pos_for(z.dtype), T, _repl_glob_for(z.dtype))
						r = repl.unsqueeze(0).expand(B, T, -1)
						z.index_copy_(2, ablate_ids, r)
					return z
		else:
			raise ValueError(f"Unknown intervention: {intervention}")

		hooks.append((name, _hook))

	return hooks


def build_clamp_hooks(
	layers_neuron_values_dict,
	last_pos_only: bool = False,
	device=None,
):
	"""Build TransformerLens hooks that clamp selected coordinates to literal values.

	This is used by threshold-event validation: after a candidate threshold is
	identified observationally, we can push a coordinate below/above that range and
	check whether the downstream behavior changes.  The input format mirrors
	``build_ablation_hooks`` but maps unit ids to values instead of to replacement
	policies::

		{"m4": {123: -0.5, 124: 0.25}, "a8.h3": {17: 1.2}}

	Supported hook labels are the same as in ``build_ablation_hooks``:
	``m{L}`` for MLP-write residual coordinates and ``a{L}.h{H}`` for attention
	head output coordinates.  Values are applied either to every position or only
	to the last position when ``last_pos_only`` is true.
	"""
	if not layers_neuron_values_dict:
		return []
	if device is None:
		device = get_device()

	mlp_layer_to_values = defaultdict(dict)
	attn_layerhead_to_values = defaultdict(dict)
	for layer_label, unit_values in layers_neuron_values_dict.items():
		parsed = get_layer_type_and_ids(layer_label)
		if not parsed:
			continue
		layer_type, L, H = parsed
		if isinstance(unit_values, dict):
			items = unit_values.items()
		else:
			items = list(unit_values or [])
		if layer_type == "mlp":
			for unit_id, value in items:
				mlp_layer_to_values[int(L)][int(unit_id)] = float(value)
		elif layer_type == "attn":
			for unit_id, value in items:
				attn_layerhead_to_values[(int(L), int(H))][int(unit_id)] = float(value)

	hooks = []
	for L, value_map in mlp_layer_to_values.items():
		if not value_map:
			continue
		name = f"blocks.{int(L)}.hook_mlp_out"
		ids = torch.as_tensor(sorted(value_map.keys()), dtype=torch.long, device=device)
		vals_base = torch.as_tensor([value_map[int(i)] for i in ids.detach().cpu().tolist()], dtype=torch.float32, device=device)

		def _vals_for(dtype, vals_base=vals_base):
			if dtype == torch.float16:
				return vals_base.to(torch.float16)
			if dtype == torch.bfloat16:
				return vals_base.to(torch.bfloat16)
			if dtype == torch.float32:
				return vals_base
			return vals_base.to(dtype)

		if last_pos_only:
			def _hook(mlp_out, hook, ids=ids, _vals_for=_vals_for):
				if mlp_out.ndim == 3:
					dest = mlp_out[:, -1, :]
					vals = _vals_for(dest.dtype).view(1, -1).expand(dest.size(0), -1)
					dest.index_copy_(1, ids, vals)
				return mlp_out
		else:
			def _hook(mlp_out, hook, ids=ids, _vals_for=_vals_for):
				if mlp_out.ndim == 3:
					B, T, _ = mlp_out.shape
					vals = _vals_for(mlp_out.dtype).view(1, 1, -1).expand(B, T, -1)
					mlp_out.index_copy_(2, ids, vals)
				return mlp_out
		hooks.append((name, _hook))

	for (L, H), value_map in attn_layerhead_to_values.items():
		if not value_map:
			continue
		name = f"blocks.{int(L)}.attn.hook_z"
		ids = torch.as_tensor(sorted(value_map.keys()), dtype=torch.long, device=device)
		vals_base = torch.as_tensor([value_map[int(i)] for i in ids.detach().cpu().tolist()], dtype=torch.float32, device=device)

		def _vals_for(dtype, vals_base=vals_base):
			if dtype == torch.float16:
				return vals_base.to(torch.float16)
			if dtype == torch.bfloat16:
				return vals_base.to(torch.bfloat16)
			if dtype == torch.float32:
				return vals_base
			return vals_base.to(dtype)

		def _dest(z, H=H):
			if z.ndim == 4:
				return z[:, :, int(H), :]
			if z.ndim == 3:
				return z
			return None

		if last_pos_only:
			def _hook(z, hook, ids=ids, _vals_for=_vals_for, _dest=_dest):
				dest = _dest(z)
				if dest is not None:
					vals = _vals_for(dest.dtype).view(1, -1).expand(dest.size(0), -1)
					dest[:, -1, :].index_copy_(1, ids, vals)
				return z
		else:
			def _hook(z, hook, ids=ids, _vals_for=_vals_for, _dest=_dest):
				dest = _dest(z)
				if dest is not None:
					B, T, _ = dest.shape
					vals = _vals_for(dest.dtype).view(1, 1, -1).expand(B, T, -1)
					dest.index_copy_(2, ids, vals)
				return z
		hooks.append((name, _hook))

	return hooks

def _resolve_rowwise_replacement_tables(
	mean_activations: MeanActTensors,
	unit_key: str,
	row_ablate_ids: torch.Tensor,
	device,
	*,
	donor_safe: bool = False,
):
	"""
	Resolve replacement values for row-wise singleton ablations.

	Each synthetic row represents the old single-neuron run, so donor-safe
	replacement excludes only that row's neuron, not the whole row-wise chunk.
	"""
	idx_full = mean_activations.mean_idx[unit_key].to(device)
	row_ablate_ids = torch.as_tensor(row_ablate_ids, dtype=torch.long, device=device)
	cols = torch.searchsorted(idx_full, row_ablate_ids)
	ok = (cols < idx_full.numel()) & (idx_full[cols] == row_ablate_ids)
	if not bool(ok.all()):
		missing = row_ablate_ids[~ok].tolist()
		raise ValueError(f"Mean tensors missing ids for {unit_key}: {missing[:10]} ...")

	mean_global_full = mean_activations.mean_global[unit_key].to(device)
	literal_global_src = (mean_activations.literal_global or mean_activations.mean_global)[unit_key].to(device)
	repl_global = mean_global_full.index_select(0, cols)
	literal_global = literal_global_src.index_select(0, cols)

	repl_pos = None
	literal_pos = None
	mean_pos_full = None
	if mean_activations.mean_per_pos is not None and mean_activations.mean_per_pos.get(unit_key) is not None:
		mean_pos_full = mean_activations.mean_per_pos[unit_key].to(device)
		repl_pos = mean_pos_full.index_select(1, cols)
		base_pos_src = repl_pos
		if mean_activations.literal_per_pos is not None and mean_activations.literal_per_pos.get(unit_key) is not None:
			base_pos_src = mean_activations.literal_per_pos[unit_key].to(device).index_select(1, cols)
		literal_pos = base_pos_src

	if donor_safe:
		row_repl_global = []
		row_repl_pos = [] if repl_pos is not None else None
		for i, col in enumerate(cols.tolist()):
			allowed_mask = torch.ones(idx_full.numel(), dtype=torch.bool, device=device)
			allowed_mask[int(col)] = False
			allowed_cols = torch.nonzero(allowed_mask, as_tuple=False).flatten()

			donor_global_bank = mean_global_full.index_select(0, allowed_cols) if allowed_cols.numel() > 0 else None
			row_repl_global.append(
				_select_donor_safe_replacements(
					literal_global[i:i+1],
					donor_global_bank,
					fallback_values=literal_global[i:i+1],
				)
			)

			if repl_pos is not None:
				if allowed_cols.numel() > 0:
					assert mean_pos_full is not None and literal_pos is not None
					donor_pos_bank = mean_pos_full.index_select(1, allowed_cols)
					row_repl_pos.append(torch.stack([
						_select_donor_safe_replacements(
							literal_pos[t, i:i+1],
							donor_pos_bank[t],
							fallback_values=literal_pos[t, i:i+1],
						).reshape(())
						for t in range(literal_pos.size(0))
					], dim=0))
				else:
					row_repl_pos.append(literal_pos[:, i].clone())

		repl_global = torch.cat(row_repl_global, dim=0).to(device=device)
		if row_repl_pos is not None:
			repl_pos = torch.stack(row_repl_pos, dim=1).to(device=device)

	return cols, repl_global, repl_pos


def build_rowwise_ablation_hooks(
	layer_label,
	neuron_ids_per_row,
	last_pos_only: bool,
	intervention="zero",
	mean_activations=None,
	device=None,
):
	"""
	Build hooks for row-wise singleton ablations.

	`neuron_ids_per_row[r]` is the one neuron ablated in synthetic batch row `r`.
	This preserves the old per-neuron semantics while allowing many neurons from
	the same layer/head to be evaluated in one larger model batch.
	"""
	if device is None:
		device = get_device()

	parsed = get_layer_type_and_ids(layer_label)
	if not parsed:
		return []
	layer_type, L, H = parsed
	row_ids = torch.as_tensor(list(map(int, neuron_ids_per_row)), dtype=torch.long, device=device)
	if row_ids.numel() == 0:
		return []
	rows = torch.arange(row_ids.numel(), dtype=torch.long, device=device)

	unit_key = f"m{int(L)}" if layer_type == "mlp" else f"a{int(L)}.h{int(H)}"

	def _dtype_cache(x: torch.Tensor):
		x_f16 = x.to(torch.float16)
		x_bf16 = x.to(torch.bfloat16)
		x_f32 = x.to(torch.float32)
		def _for(dtype, x=x, x_f16=x_f16, x_bf16=x_bf16, x_f32=x_f32):
			if dtype == torch.float16: return x_f16
			if dtype == torch.bfloat16: return x_bf16
			if dtype == torch.float32: return x_f32
			return x.to(dtype)
		return _for

	repl_global_for = None
	repl_pos_for = None
	if intervention in ("mean", "mean-donor", "mean-positional", "mean-donor-positional"):
		if mean_activations is None:
			raise ValueError(f"{intervention} intervention requested but mean_activations is None")
		_, repl_global, repl_pos = _resolve_rowwise_replacement_tables(
			mean_activations,
			unit_key,
			row_ids,
			device,
			donor_safe=intervention in ("mean-donor", "mean-donor-positional"),
		)
		repl_global_for = _dtype_cache(repl_global)
		if intervention in ("mean-positional", "mean-donor-positional"):
			if repl_pos is None:
				repl_pos = torch.zeros((0, repl_global.numel()), dtype=repl_global.dtype, device=device)
			repl_pos_for = _dtype_cache(repl_pos)

	def _pos_index(T, dev):
		return torch.arange(T, dtype=torch.long, device=dev)

	if layer_type == "mlp":
		name = f"blocks.{int(L)}.hook_mlp_out"

		if intervention == "zero":
			if last_pos_only:
				def _hook(mlp_out, hook, rows=rows, row_ids=row_ids):
					if mlp_out.ndim == 3:
						mlp_out[rows, -1, row_ids] = 0
					return mlp_out
			else:
				def _hook(mlp_out, hook, rows=rows, row_ids=row_ids):
					if mlp_out.ndim == 3:
						pos = _pos_index(mlp_out.size(1), mlp_out.device)
						mlp_out[rows[:, None], pos[None, :], row_ids[:, None]] = 0
					return mlp_out

		elif intervention in ("mean", "mean-donor"):
			if last_pos_only:
				def _hook(mlp_out, hook, rows=rows, row_ids=row_ids, repl_global_for=repl_global_for):
					if mlp_out.ndim == 3:
						mlp_out[rows, -1, row_ids] = repl_global_for(mlp_out.dtype)
					return mlp_out
			else:
				def _hook(mlp_out, hook, rows=rows, row_ids=row_ids, repl_global_for=repl_global_for):
					if mlp_out.ndim == 3:
						T = mlp_out.size(1)
						pos = _pos_index(T, mlp_out.device)
						r = repl_global_for(mlp_out.dtype).view(-1, 1).expand(-1, T)
						mlp_out[rows[:, None], pos[None, :], row_ids[:, None]] = r
					return mlp_out

		elif intervention in ("mean-positional", "mean-donor-positional"):
			if last_pos_only:
				def _hook(mlp_out, hook, rows=rows, row_ids=row_ids, repl_pos_for=repl_pos_for, repl_global_for=repl_global_for):
					if mlp_out.ndim == 3:
						t = mlp_out.size(1) - 1
						repl_pos = repl_pos_for(mlp_out.dtype)
						r = repl_pos[t] if t < repl_pos.size(0) else repl_global_for(mlp_out.dtype)
						mlp_out[rows, -1, row_ids] = r
					return mlp_out
			else:
				def _hook(mlp_out, hook, rows=rows, row_ids=row_ids, repl_pos_for=repl_pos_for, repl_global_for=repl_global_for):
					if mlp_out.ndim == 3:
						T = mlp_out.size(1)
						pos = _pos_index(T, mlp_out.device)
						repl = _fit_positional_repl(repl_pos_for(mlp_out.dtype), T, repl_global_for(mlp_out.dtype))
						mlp_out[rows[:, None], pos[None, :], row_ids[:, None]] = repl.transpose(0, 1)
					return mlp_out
		else:
			raise ValueError(f"Unknown intervention: {intervention}")

		return [(name, _hook)]

	if layer_type == "attn":
		name = f"blocks.{int(L)}.attn.hook_z"

		def _attn_dest(z):
			if z.ndim == 4:
				return z[:, :, int(H), :]
			if z.ndim == 3:
				return z
			return None

		if intervention == "zero":
			if last_pos_only:
				def _hook(z, hook, rows=rows, row_ids=row_ids):
					dest = _attn_dest(z)
					if dest is not None:
						dest[rows, -1, row_ids] = 0
					return z
			else:
				def _hook(z, hook, rows=rows, row_ids=row_ids):
					dest = _attn_dest(z)
					if dest is not None:
						pos = _pos_index(dest.size(1), dest.device)
						dest[rows[:, None], pos[None, :], row_ids[:, None]] = 0
					return z

		elif intervention in ("mean", "mean-donor"):
			if last_pos_only:
				def _hook(z, hook, rows=rows, row_ids=row_ids, repl_global_for=repl_global_for):
					dest = _attn_dest(z)
					if dest is not None:
						dest[rows, -1, row_ids] = repl_global_for(dest.dtype)
					return z
			else:
				def _hook(z, hook, rows=rows, row_ids=row_ids, repl_global_for=repl_global_for):
					dest = _attn_dest(z)
					if dest is not None:
						T = dest.size(1)
						pos = _pos_index(T, dest.device)
						r = repl_global_for(dest.dtype).view(-1, 1).expand(-1, T)
						dest[rows[:, None], pos[None, :], row_ids[:, None]] = r
					return z

		elif intervention in ("mean-positional", "mean-donor-positional"):
			if last_pos_only:
				def _hook(z, hook, rows=rows, row_ids=row_ids, repl_pos_for=repl_pos_for, repl_global_for=repl_global_for):
					dest = _attn_dest(z)
					if dest is not None:
						t = dest.size(1) - 1
						repl_pos = repl_pos_for(dest.dtype)
						r = repl_pos[t] if t < repl_pos.size(0) else repl_global_for(dest.dtype)
						dest[rows, -1, row_ids] = r
					return z
			else:
				def _hook(z, hook, rows=rows, row_ids=row_ids, repl_pos_for=repl_pos_for, repl_global_for=repl_global_for):
					dest = _attn_dest(z)
					if dest is not None:
						T = dest.size(1)
						pos = _pos_index(T, dest.device)
						repl = _fit_positional_repl(repl_pos_for(dest.dtype), T, repl_global_for(dest.dtype))
						dest[rows[:, None], pos[None, :], row_ids[:, None]] = repl.transpose(0, 1)
					return z
		else:
			raise ValueError(f"Unknown intervention: {intervention}")

		return [(name, _hook)]

	return []

@dataclass
class PrefixCacheBatch:
	"""
	Prefix cache for a *batch* of prompts.

	All tensors are on the same device as the model.
	"""
	batch_texts: List[str]
	input_ids: torch.Tensor          # [B, T_prompt]
	attention_mask: torch.Tensor     # [B, T_prompt]
	padding_side: str           # "left" or "right"
	input_lengths: List[int]         # python ints
	all_tokens: torch.Tensor         # [B, T_prompt + max_new_tokens]
	prompt_len: int
	max_new_tokens: int
	pad_token_id: int
	eos_token_id: int
	use_kv_cache: bool
	past_kv_cache: Optional[HookedTransformerKeyValueCache]
	logits_last: Optional[torch.Tensor]  # logits from prefill, shape [..., vocab_size]

def reset_tl_kv_cache_inplace(cache, prompt_len: int, base_mask: torch.Tensor | None = None, shrink_storage: bool = True):
	"""
	TransformerLens past_keys/past_values layout: [B, pos, heads, d_head].  (see get_pos_offset)
	- Slices pos dimension (dim=1) back to prompt_len.
	- Optionally resets previous_attention_mask to the prompt-only mask.
	- shrink_storage=True makes a real smaller tensor (frees extra storage but copies).
	  shrink_storage=False keeps a view (faster, but does NOT free the old big allocation).
	"""
	if cache is None:
		return None

	entries = getattr(cache, "entries", None)
	if entries is None:
		# legacy: try cache[0] style
		entries = list(cache)

	for e in entries:
		pk = getattr(e, "past_keys", None)
		pv = getattr(e, "past_values", None)

		if torch.is_tensor(pk):
			# [B, pos, heads, d_head]
			pk2 = pk[:, :prompt_len, :, :]
			e.past_keys = pk2.contiguous() if shrink_storage else pk2

		if torch.is_tensor(pv):
			pv2 = pv[:, :prompt_len, :, :]
			e.past_values = pv2.contiguous() if shrink_storage else pv2

	if base_mask is not None:
		pam = base_mask[:, :prompt_len]
		cache.previous_attention_mask = pam.contiguous() if shrink_storage else pam

	return cache

def clone_kv_cache(cache: HookedTransformerKeyValueCache) -> HookedTransformerKeyValueCache:
	"""
	Clone a TransformerLens HookedTransformerKeyValueCache (current API: entries + previous_attention_mask).

	Creates new tensors for past_keys/past_values/previous_attention_mask so mutations don't alias.
	"""
	if cache is None:
		return None

	# New TransformerLens API: cache.entries exists
	if hasattr(cache, "entries"):
		new_entries = []
		for entry in cache.entries:
			pk = getattr(entry, "past_keys", None)
			pv = getattr(entry, "past_values", None)

			new_entries.append(
				HookedTransformerKeyValueCacheEntry(
					past_keys=pk.detach().clone() if torch.is_tensor(pk) else copy.deepcopy(pk),
					past_values=pv.detach().clone() if torch.is_tensor(pv) else copy.deepcopy(pv),
					frozen=getattr(entry, "frozen", False),
				)
			)

		pam = getattr(cache, "previous_attention_mask", None)
		new_pam = pam.detach().clone() if torch.is_tensor(pam) else copy.deepcopy(pam)

		return HookedTransformerKeyValueCache(
			entries=new_entries,
			previous_attention_mask=new_pam,
			frozen=getattr(cache, "frozen", False),
		)

	# Legacy fallback (very old API): cache.cache_dict exists
	if hasattr(cache, "cache_dict"):
		new_cache = copy.deepcopy(cache)
		for k, v in cache.cache_dict.items():
			if isinstance(v, dict):
				new_cache.cache_dict[k] = {
					kk: (vv.detach().clone() if torch.is_tensor(vv) else copy.deepcopy(vv))
					for kk, vv in v.items()
				}
			elif torch.is_tensor(v):
				new_cache.cache_dict[k] = v.detach().clone()
			else:
				new_cache.cache_dict[k] = copy.deepcopy(v)
		return new_cache

	raise TypeError(
		f"Unrecognized cache type {type(cache)}: expected TransformerLens KV cache with `.entries` "
		f"(new API) or `.cache_dict` (legacy API)."
	)


def index_select_kv_cache(cache: HookedTransformerKeyValueCache, row_idx: torch.Tensor) -> HookedTransformerKeyValueCache:
	"""Return a new KV cache containing rows selected along batch dimension."""
	if cache is None:
		return None
	row_idx = row_idx.to(dtype=torch.long)

	if hasattr(cache, "entries"):
		new_entries = []
		for entry in cache.entries:
			pk = getattr(entry, "past_keys", None)
			pv = getattr(entry, "past_values", None)
			dev_idx_pk = row_idx.to(pk.device) if torch.is_tensor(pk) else row_idx
			dev_idx_pv = row_idx.to(pv.device) if torch.is_tensor(pv) else row_idx
			new_entries.append(
				HookedTransformerKeyValueCacheEntry(
					past_keys=pk.index_select(0, dev_idx_pk).contiguous() if torch.is_tensor(pk) else copy.deepcopy(pk),
					past_values=pv.index_select(0, dev_idx_pv).contiguous() if torch.is_tensor(pv) else copy.deepcopy(pv),
					frozen=getattr(entry, "frozen", False),
				)
			)

		pam = getattr(cache, "previous_attention_mask", None)
		new_pam = pam.index_select(0, row_idx.to(pam.device)).contiguous() if torch.is_tensor(pam) else copy.deepcopy(pam)
		return HookedTransformerKeyValueCache(
			entries=new_entries,
			previous_attention_mask=new_pam,
			frozen=getattr(cache, "frozen", False),
		)

	if hasattr(cache, "cache_dict"):
		new_cache = copy.deepcopy(cache)
		for k, v in cache.cache_dict.items():
			if isinstance(v, dict):
				new_cache.cache_dict[k] = {
					kk: (vv.index_select(0, row_idx.to(vv.device)).contiguous() if torch.is_tensor(vv) else copy.deepcopy(vv))
					for kk, vv in v.items()
				}
			elif torch.is_tensor(v):
				new_cache.cache_dict[k] = v.index_select(0, row_idx.to(v.device)).contiguous()
			else:
				new_cache.cache_dict[k] = copy.deepcopy(v)
		return new_cache

	raise TypeError(f"Unrecognized cache type {type(cache)}")


def repeat_prefix_cache_batch(prefix, repeats_per_prompt: int):
	"""
	Repeat every prompt row K times in a PrefixCacheBatch.

	Order is prompt0 repeated K times, prompt1 repeated K times, ... . This must
	match row-wise neuron ids built as `chunk_ids * len(prefix.batch_texts)`.
	"""
	K = int(repeats_per_prompt)
	if K <= 1:
		return prefix
	B = int(prefix.input_ids.size(0))
	device = prefix.input_ids.device
	row_idx = torch.arange(B, dtype=torch.long, device=device).repeat_interleave(K)

	return PrefixCacheBatch(
		batch_texts=[prefix.batch_texts[i] for i in row_idx.tolist()],
		input_ids=prefix.input_ids.index_select(0, row_idx).contiguous(),
		attention_mask=prefix.attention_mask.index_select(0, row_idx).contiguous(),
		padding_side=prefix.padding_side,
		input_lengths=[prefix.input_lengths[i] for i in row_idx.tolist()],
		all_tokens=prefix.all_tokens.index_select(0, row_idx).contiguous(),
		prompt_len=prefix.prompt_len,
		max_new_tokens=prefix.max_new_tokens,
		pad_token_id=prefix.pad_token_id,
		eos_token_id=prefix.eos_token_id,
		use_kv_cache=prefix.use_kv_cache,
		past_kv_cache=index_select_kv_cache(prefix.past_kv_cache, row_idx) if prefix.past_kv_cache is not None else None,
		logits_last=prefix.logits_last.index_select(0, row_idx).contiguous() if torch.is_tensor(prefix.logits_last) else prefix.logits_last,
	)

class LMWrapper:
	@staticmethod
	def _parse_model_spec(model_name):
		"""Allow optional TransformerLens checkpoint suffixes in the model string.

		Examples:
		- EleutherAI/pythia-1b@step48000
		- EleutherAI/pythia-1b@checkpoint48000
		- EleutherAI/pythia-1b@ckpt48000
		"""
		m = re.fullmatch(r"(.+?)@(?:step|checkpoint|ckpt)(\d+)", model_name)
		if m is None:
			return model_name, None
		return m.group(1), int(m.group(2))

	def __init__(self, model_name, device, eval_mode=True, ungroup_grouped_query_attention=False, circuit_discovery=False, cache_dir=None):
		# torch.set_grad_enabled(False)
		self.model_name = model_name
		self.base_model_name, self.checkpoint_value = self._parse_model_spec(model_name)
		resolved_model_name = self.base_model_name
		self.hf_revision = f"step{self.checkpoint_value}" if self.checkpoint_value is not None else None
		hf_revision_kwargs = {"revision": self.hf_revision} if self.hf_revision is not None else {}
		try:
			self.tokenizer = AutoTokenizer.from_pretrained(
				resolved_model_name,
				cache_dir=cache_dir,
				**hf_revision_kwargs,
			)
		except:
			m = resolved_model_name.lower()

			if "qwen2" in m or "qwen" in m:
				# Qwen / Qwen2 family: byte-level BPE, need Qwen2Tokenizer
				self.tokenizer = Qwen2Tokenizer.from_pretrained(
					resolved_model_name,
					# add_bos_token=True,
					cache_dir=cache_dir,
				)

			elif m.startswith("gpt2") or "gpt-" in m or "gpt_neo" in m or "gptj" in m:
				# GPT-2 / GPT-Neo / GPT-J families — use GPT2Tokenizer
				self.tokenizer = GPT2Tokenizer.from_pretrained(
					resolved_model_name,
					# add_bos_token=True,
					cache_dir=cache_dir,
				)

			elif "llama" in m or "mistral" in m or "apertus" in m or "phi3" in m or "phi-3" in m or "phi_3" in m:
				# Llama / Mistral (or similar) — use LlamaTokenizer / LlamaTokenizerFast
				try:
					self.tokenizer = LlamaTokenizerFast.from_pretrained(
						resolved_model_name,
						# add_bos_token=True,
						cache_dir=cache_dir,
					)
				except Exception:
					self.tokenizer = LlamaTokenizer.from_pretrained(
						resolved_model_name,
						# add_bos_token=True,
						cache_dir=cache_dir,
					)

			elif "bert" in m or "roberta" in m or "distilbert" in m or "albert" in m:
				if "roberta" in m:
					self.tokenizer = RobertaTokenizer.from_pretrained(resolved_model_name, cache_dir=cache_dir)
				elif "distilbert" in m:
					self.tokenizer = DistilBertTokenizer.from_pretrained(resolved_model_name, cache_dir=cache_dir)
				elif "albert" in m:
					self.tokenizer = AlbertTokenizer.from_pretrained(resolved_model_name, cache_dir=cache_dir)
				else:
					self.tokenizer = BertTokenizer.from_pretrained(resolved_model_name, cache_dir=cache_dir)

			else:
				try:
					self.tokenizer = AutoTokenizer.from_pretrained(
						resolved_model_name,
						cache_dir=cache_dir,
						trust_remote_code=False,   # Phi-3 shouldn't need it once transformers is new enough
						use_fast=True,
					)
				except Exception as e:
					raise RuntimeError(f"Tokenizer load failed for {resolved_model_name}: {e}") from e

		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token

		# Ensure consistent truncation: keep the *suffix* (answer) when we have to cap context
		if getattr(self.tokenizer, "truncation_side", None) is not None:
			self.tokenizer.truncation_side = "left"

		if str(device).startswith("mps"):
			if not patch_transformer_lens_mps_sumprod():
				# fallback: safest global option if TL code changed
				# (keeps semantics, just slower)
				device = torch.device("cpu")
				print('[Warning] Failed HookedTransformer patch')

		if self.checkpoint_value is not None:
			print(f"[Info] Loading HuggingFace revision {self.hf_revision} for {resolved_model_name}")

		self.model = AutoModelForCausalLM.from_pretrained(
			resolved_model_name,
			torch_dtype="auto",
			device_map='cpu',
			cache_dir=cache_dir,
			**hf_revision_kwargs,
		).to(device)
		
		self.hooked_model = lens.HookedTransformer.from_pretrained(
			model_name=resolved_model_name,
			hf_model=self.model,
			fold_ln=True,
			center_unembed=True,
			center_writing_weights=True,
			use_attn_result=circuit_discovery,
			use_split_qkv_input=circuit_discovery,
			use_hook_mlp_in=circuit_discovery,
			tokenizer=self.tokenizer,
			cache_dir=cache_dir,
			device=device,
		)

		# Preserve the exact requested model spec for downstream cache keys.
		# HF objects loaded with revision="step0" still report only the base
		# model name, so without this the representation cache would be shared
		# between EleutherAI/pythia-1b and EleutherAI/pythia-1b@step0.
		if self.checkpoint_value is not None:
			self.model.nare_cache_model_id = self.model_name
			self.hooked_model.nare_cache_model_id = self.model_name
		
		# self.hooked_model.to(device)
		if eval_mode:
			if self.model is not None:
				self.model.eval()
			self.hooked_model.eval()

		if ungroup_grouped_query_attention:
			if hasattr(self.hooked_model, "cfg") and getattr(
				self.hooked_model.cfg,
				"ungroup_grouped_query_attention",
				None,
			) is not None:
				self.hooked_model.set_ungroup_grouped_query_attention(True)
				self.hooked_model.cfg.ungroup_grouped_query_attention = True

		# TransformerLens forwards unknown from_pretrained kwargs to HuggingFace.
		# Keep TL-only hook flags out of from_pretrained, then enable them on the
		# loaded HookedTransformer.
		if circuit_discovery:
			if hasattr(self.hooked_model, "set_use_attn_result"):
				self.hooked_model.set_use_attn_result(True)
			if hasattr(self.hooked_model, "set_use_split_qkv_input"):
				self.hooked_model.set_use_split_qkv_input(True)
			if hasattr(self.hooked_model, "set_use_hook_mlp_in"):
				self.hooked_model.set_use_hook_mlp_in(True)

		print(
			"sanity:",
			self.hooked_model.cfg.use_attn_result,
			self.hooked_model.cfg.use_attn_in,
			self.hooked_model.cfg.use_split_qkv_input,
			self.hooked_model.cfg.use_hook_mlp_in,
		)

	def _supports_forward_kwarg(self, model, kw: str) -> bool:
		"""Return True if `model.forward` explicitly accepts kwarg `kw`."""
		sig = inspect.signature(model.forward)
		return kw in sig.parameters

	def _position_ids_from_attention_mask(self, attention_mask: torch.Tensor, *, past_kv_cache=None) -> torch.Tensor:
		"""Compute padding-safe position_ids.

		- For full-prefix (no cache): use cumsum(attention_mask)-1 (pads -> 0).
		- For cached incremental steps: offset by the number of *real* tokens already in the cache
		  (sum of cache.previous_attention_mask), if available.
		"""
		am = attention_mask.long()
		if am.ndim != 2:
			raise ValueError(f"Expected attention_mask [B,T], got shape {tuple(attention_mask.shape)}")

		B, T = am.shape

		# Cached incremental step: position starts at previous real length
		if past_kv_cache is not None:
			prev = getattr(past_kv_cache, "previous_attention_mask", None)
			if torch.is_tensor(prev):
				prev_lens = prev.long().sum(dim=1, keepdim=True)  # [B,1]
				ar = torch.arange(T, device=attention_mask.device, dtype=prev_lens.dtype).view(1, T)
				return prev_lens + ar

		# Full-prefix path
		pos = am.cumsum(dim=1) - 1
		pos = pos.clamp(min=0)
		# Force pads to position 0 (keeps things well-defined even if pads are present)
		pos = torch.where(am == 0, torch.zeros_like(pos), pos)
		return pos

	def tokenize_with_mask(self, texts, device, *, add_special_tokens=True, padding=True, truncation=True, max_length=None, padding_side: Optional[str]=None, truncation_side: Optional[str]=None):
		"""Single source of truth: text -> (input_ids, attention_mask, lengths).

		If `padding_side` is provided, it's applied *temporarily* for this tokenization call.
		"""
		if isinstance(texts, str):
			texts = [texts]

		old_padding_side = getattr(self.tokenizer, "padding_side", None)
		if padding_side is not None and old_padding_side is not None:
			self.tokenizer.padding_side = padding_side

		old_truncation_side = getattr(self.tokenizer, "truncation_side", None)
		if truncation_side is not None and old_truncation_side is not None:
			self.tokenizer.truncation_side = truncation_side

		try:
			enc = self.tokenizer(
				texts,
				return_tensors="pt",
				padding=padding,
				truncation=truncation,
				max_length=max_length,
				add_special_tokens=add_special_tokens,
			)
		finally:
			if padding_side is not None and old_padding_side is not None:
				self.tokenizer.padding_side = old_padding_side
			if truncation_side is not None and old_truncation_side is not None:
				self.tokenizer.truncation_side = old_truncation_side

		input_ids = enc["input_ids"].to(device)               # [B, L]
		attention_mask = enc["attention_mask"].to(device)     # [B, L], 1 = real, 0 = pad
		input_lengths = attention_mask.long().sum(dim=1)      # [B]

		return input_ids, attention_mask, input_lengths

	@torch.inference_mode()
	def _last_logits(self, tokens, attention_mask=None, past_kv_cache=None, padding_side: str = "right"):
		cfg = self.hooked_model.cfg

		# Run blocks, but STOP before unembed so we don't allocate [B,T,V]
		fwd_kwargs = dict(
			return_type="residual",          # ignored because we stop early
			attention_mask=attention_mask,
			past_kv_cache=past_kv_cache,
			padding_side=padding_side,
			stop_at_layer=cfg.n_layers,      # returns residual [B,T,d_model]
		)

		residual = self.hooked_model(tokens, **fwd_kwargs)

		# forward() would normally apply ln_final before unembed if normalization exists
		if getattr(cfg, "normalization_type", None) is not None and hasattr(self.hooked_model, "ln_final"):
			residual = self.hooked_model.ln_final(residual)

		# Pick logits corresponding to the last *non-pad* token per example.
		if residual.ndim == 2:
			# Already [B, d_model]
			res_last = residual
		elif attention_mask is None:
			res_last = residual[:, -1, :]  # [B, d_model]
		else:
			T = attention_mask.size(1)
			idx = torch.arange(T, device=residual.device).view(1, T)
			last_idx = (attention_mask.long() * idx).max(dim=1).values  # works for left/right padding
			res_last = residual[
				torch.arange(residual.size(0), device=residual.device),
				last_idx,
				:
			]

		logits_last = self.hooked_model.unembed(res_last)  # [B, vocab]
		return logits_last

	def cleanup_after_generate(self):
		# gc.collect()
		if str(self.hooked_model.cfg.device).startswith("mps"):
			# torch.mps.synchronize()
			torch.mps.empty_cache()

	@torch.inference_mode()
	def prefill_prefix_batch(self, prompts, max_new_tokens=10, use_kv_cache=True, **gen_args):
		"""
		Run only the *prompt* through the model, fill a KV cache,
		and return a PrefixCacheBatch that can be reused across ablations.

		NOTE: This is intentionally run *without* ablation hooks.
			  Ablations should only be applied during decoding (e.g., last_pos_only).
		"""

		device = self.hooked_model.cfg.device
		max_new_tokens = int(max_new_tokens)

		# Normalize to list of strings
		if isinstance(prompts, str):
			batch_texts = [prompts]
		else:
			batch_texts = list(prompts)

		# Determine usable context length
		n_ctx = getattr(self.hooked_model.cfg, "n_ctx", None)
		if not n_ctx or n_ctx > 1_000_000:
			n_ctx = getattr(self.tokenizer, "model_max_length", 2048)
			if not n_ctx or n_ctx > 1_000_000:
				n_ctx = 2048

		# Leave room for generation
		max_prompt_len = max(1, n_ctx - max_new_tokens)
		padding_side = "left" if use_kv_cache else getattr(self.tokenizer, "padding_side", "right")

		# Tokenize
		input_ids, base_mask, input_lengths = self.tokenize_with_mask(
			batch_texts,
			device,
			padding=True,
			truncation=True,
			add_special_tokens=True,
			max_length=max_prompt_len,
			padding_side=padding_side,
		)

		# For now we only care about greedy decoding; we just mirror generate()
		eos_token_id = self.tokenizer.eos_token_id
		pad_token_id = self.tokenizer.pad_token_id
		batch_size, prompt_len = input_ids.shape

		# Decide whether KV cache is usable
		cfg = self.hooked_model.cfg
		use_split_qkv = getattr(cfg, "use_split_qkv_input", False)
		ungroup_gqa   = getattr(cfg, "ungroup_grouped_query_attention", False)
		use_kv_cache  = (not use_split_qkv) and (not ungroup_gqa) and use_kv_cache
		# Preallocate token buffer: [B, prompt_len + max_new_tokens]
		total_len = prompt_len + max_new_tokens
		all_tokens = torch.full(
			(batch_size, total_len),
			fill_value=pad_token_id,
			dtype=input_ids.dtype,
			device=device,
		)
		all_tokens[:, :prompt_len] = input_ids

		# We do NOT use any hooks here; this is the shared, unablated prefix
		past_kv_cache = None
		logits_last = None

		if use_kv_cache:
			# Init cache with adjusted n_ctx
			cfg_for_cache = copy.copy(cfg)
			cfg_for_cache.n_ctx = total_len
			past_kv_cache = HookedTransformerKeyValueCache.init_cache(
				cfg=cfg_for_cache,
				device=device,
				batch_size=batch_size,
			)

			# Prefill on full prompt to fill the cache
			logits_last = self._last_logits(
				all_tokens[:, :prompt_len],
				base_mask,
				past_kv_cache=past_kv_cache,
				padding_side=padding_side,
			)
		else:
			# Fallback: no KV cache, but still compute initial logits for completeness
			logits_last = self._last_logits(
				all_tokens[:, :prompt_len],
				base_mask,
				past_kv_cache=None,
				padding_side=padding_side,
			)
		self.cleanup_after_generate()

		# Normalize lengths to Python ints
		input_lengths = [int(l) for l in input_lengths]

		return PrefixCacheBatch(
			batch_texts=batch_texts,
			input_ids=input_ids,
			attention_mask=base_mask,
			padding_side=padding_side,
			input_lengths=input_lengths,
			all_tokens=all_tokens,
			prompt_len=prompt_len,
			max_new_tokens=max_new_tokens,
			pad_token_id=pad_token_id,
			eos_token_id=eos_token_id,
			use_kv_cache=use_kv_cache,
			past_kv_cache=past_kv_cache,
			logits_last=logits_last,
		)

	@torch.inference_mode()
	def _make_eos_tensor(self, stop_at_eos: bool, eos_token_id: int, *, batch_size: int, dtype, device):
		if not stop_at_eos:
			return None
		return torch.full((batch_size,), eos_token_id, dtype=dtype, device=device)

	@torch.inference_mode()
	def _decode_suffix(self, outputs, prompt_len: int):
		# `outputs` includes the (padded) prompt in the first `prompt_len` positions for every example.
		# Generated tokens start at a fixed offset `prompt_len` (this avoids issues with left/right padding).
		prompt_len = int(prompt_len)
		gen_token_lists = [out[prompt_len:].tolist() for out in outputs]
		return self.tokenizer.batch_decode(gen_token_lists, skip_special_tokens=True)

	@torch.inference_mode()
	def _greedy_decode_cached(self, all_tokens, cur_len, max_new_tokens, logits_last, past_kv_cache, base_mask, stop_at_eos, eos_token_id, finished, cleanup_every = 3, padding_side: str = "right"):
		"""
		Decode using single-token KV-cache steps. Assumes cache already contains the prompt state
		and `logits_last` corresponds to the last prompt position.
		"""
		B = all_tokens.shape[0]
		device = all_tokens.device
		dtype = all_tokens.dtype

		eos_tensor = self._make_eos_tensor(
			stop_at_eos, eos_token_id, batch_size=B, dtype=dtype, device=device
		)

		# Reused per-step chunk tensors
		tokens_chunk = torch.empty((B, 1), dtype=dtype, device=device)
		attn_mask_chunk = torch.ones((B, 1), dtype=base_mask.dtype, device=device)

		for i in range(max_new_tokens):
			next_tokens = torch.argmax(logits_last, dim=-1)#.to(dtype)  # [B]

			if eos_tensor is not None:
				next_tokens = torch.where(finished, eos_tensor, next_tokens)
				finished |= (next_tokens == eos_token_id)

			all_tokens[:, cur_len] = next_tokens
			cur_len += 1

			if eos_tensor is not None and finished.all():
				break

			tokens_chunk[:, 0] = next_tokens

			logits_last = self._last_logits(
				tokens_chunk,
				attn_mask_chunk,
				past_kv_cache=past_kv_cache,
				padding_side=padding_side,
			)

			if cleanup_every and (i + 1) % cleanup_every == 0:
				self.cleanup_after_generate()

		if cleanup_every and (i + 1) % cleanup_every != 0:
			self.cleanup_after_generate()

		return all_tokens[:, :cur_len]

	@torch.inference_mode()
	def _greedy_decode_noncached(self, all_tokens, attn_mask, cur_len, max_new_tokens, stop_at_eos, eos_token_id, finished, cleanup_every = 3, padding_side = "right"):
		"""
		Decode by recomputing logits over the full prefix+generated each step.
		"""
		B = all_tokens.shape[0]
		device = all_tokens.device
		dtype = all_tokens.dtype

		eos_tensor = self._make_eos_tensor(
			stop_at_eos, eos_token_id, batch_size=B, dtype=dtype, device=device
		)

		for i in range(max_new_tokens):
			logits_last = self._last_logits(
				all_tokens[:, :cur_len],
				attn_mask[:, :cur_len],
				past_kv_cache=None,
				padding_side=padding_side,
			)

			next_tokens = torch.argmax(logits_last, dim=-1)#.to(dtype)

			if eos_tensor is not None:
				next_tokens = torch.where(finished, eos_tensor, next_tokens)
				finished |= (next_tokens == eos_token_id)

			all_tokens[:, cur_len] = next_tokens
			attn_mask[:, cur_len] = 1
			cur_len += 1

			if cleanup_every and (i + 1) % cleanup_every == 0:
				self.cleanup_after_generate()

			if eos_tensor is not None and finished.all():
				break
		
		if cleanup_every and (i + 1) % cleanup_every != 0:
			self.cleanup_after_generate()

		return all_tokens[:, :cur_len]

	@torch.inference_mode()
	def generate_from_prefix_cache(self, prefix, fwd_hooks=None, debug=False, stop_at_eos=True, clone_kv_cache_tensors=True):
		"""
		Use a pre-filled PrefixCacheBatch to generate completions.

		- Reuses the cached prompt KV state.
		- Applies `fwd_hooks` *only* during decoding (so it plays nicely with last_pos_only).

		Returns:
			List of decoded strings, same order as prefix.batch_texts.
		"""
		device = self.hooked_model.cfg.device
		padding_side = getattr(prefix, "padding_side", getattr(self.tokenizer, "padding_side", "left"))

		all_tokens = prefix.all_tokens.clone()     # [B, prompt_len + max_new_tokens]
		base_mask = prefix.attention_mask         # [B, prompt_len]
		input_ids = prefix.input_ids
		input_lengths = prefix.input_lengths
		max_new_tokens = prefix.max_new_tokens
		batch_size, prompt_len = input_ids.shape
		eos_token_id = prefix.eos_token_id

		finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

		# Where decoding starts
		cur_len = prompt_len

		# Use cached logits from prefill if available; otherwise recompute on first step
		logits_last = prefix.logits_last

		past_kv_cache = None
		if prefix.use_kv_cache and prefix.past_kv_cache is not None:
			if clone_kv_cache_tensors:
				# KV cache: clone so each generation (per ablation) has its own mutable copy
				past_kv_cache = clone_kv_cache(prefix.past_kv_cache)
			else: # reset previous_attention_mask back to prompt-only
				past_kv_cache = prefix.past_kv_cache
				reset_tl_kv_cache_inplace(
					past_kv_cache,
					prompt_len=prompt_len,
					base_mask=base_mask,
					shrink_storage=False,   # set False if you only want speed and don't care about freeing
				)

		# If we have a cache but missing cached logits, recompute without hooks (safe + rare)
		if prefix.use_kv_cache and past_kv_cache is not None and logits_last is None:
			logits_last = self._last_logits(
				all_tokens[:, :prompt_len],
				base_mask,
				past_kv_cache=None,   # don't mutate an already-filled cache
				padding_side=padding_side,
			)

		# Context manager for ablation hooks (applied only during decoding)
		ctx = self.hooked_model.hooks(
			fwd_hooks=fwd_hooks,
			reset_hooks_end=True,
			clear_contexts=True,
		) if fwd_hooks else nullcontext()

		with ctx:
			if prefix.use_kv_cache and past_kv_cache is not None:
				outputs = self._greedy_decode_cached(
					all_tokens=all_tokens,
					cur_len=cur_len,
					max_new_tokens=max_new_tokens,
					logits_last=logits_last,
					past_kv_cache=past_kv_cache,
					base_mask=base_mask,
					stop_at_eos=stop_at_eos,
					eos_token_id=eos_token_id,
					finished=finished,
					cleanup_every=3,
					padding_side=padding_side,
				)
			else:
				total_len = all_tokens.shape[1]
				attn_mask = base_mask.new_zeros((batch_size, total_len))
				attn_mask[:, :prompt_len] = base_mask

				outputs = self._greedy_decode_noncached(
					all_tokens=all_tokens,
					attn_mask=attn_mask,
					cur_len=cur_len,
					max_new_tokens=max_new_tokens,
					stop_at_eos=stop_at_eos,
					eos_token_id=eos_token_id,
					finished=finished,
					cleanup_every=3,
					padding_side=padding_side,
				)

		if debug:
			print(f"[prefix-decode] inputs: {input_ids.shape}, outputs: {outputs.shape}")

		return self._decode_suffix(outputs, prefix.prompt_len)

	@torch.inference_mode()
	def generate(self, prompts, debug=False, fwd_hooks=None, verbose=False, use_kv_cache=True, **gen_args):
		# Normalize prompts to a list of strings
		batch_texts = [prompts] if isinstance(prompts, str) else list(prompts)

		device = self.hooked_model.cfg.device
		max_new_tokens = int(gen_args.pop("max_new_tokens", 10))

		# Determine a sane context length
		n_ctx = getattr(self.hooked_model.cfg, "n_ctx", None)
		if not n_ctx or n_ctx > 1_000_000:
			n_ctx = getattr(self.tokenizer, "model_max_length", 2048)
			if not n_ctx or n_ctx > 1_000_000:
				n_ctx = 2048

		max_prompt_len = max(1, n_ctx - max_new_tokens)
		padding_side = "left" if use_kv_cache else getattr(self.tokenizer, "padding_side", "right")

		input_ids, base_mask, input_lengths = self.tokenize_with_mask(
			batch_texts,
			device,
			padding=True,
			truncation=True,
			add_special_tokens=True,
			max_length=max_prompt_len,
			padding_side=padding_side,
		)

		stop_at_eos = bool(gen_args.pop("stop_at_eos", True))
		do_sample   = bool(gen_args.pop("do_sample", False))
		if do_sample:
			raise NotImplementedError("Sampling (do_sample=True) not implemented in this padding-safe generate yet.")

		eos_token_id = self.tokenizer.eos_token_id
		pad_token_id = self.tokenizer.pad_token_id
		batch_size, prompt_len = input_ids.shape

		# Decide whether we can safely use KV cache
		cfg = self.hooked_model.cfg
		use_split_qkv = getattr(cfg, "use_split_qkv_input", False)
		ungroup_gqa   = getattr(cfg, "ungroup_grouped_query_attention", False)
		use_kv_cache  = (not use_split_qkv) and (not ungroup_gqa) and use_kv_cache
		# Preallocate token buffer: [B, prompt_len + max_new_tokens]
		total_len  = prompt_len + max_new_tokens
		all_tokens = torch.full(
			(batch_size, total_len),
			fill_value=pad_token_id,
			dtype=input_ids.dtype,
			device=device,
		)
		all_tokens[:, :prompt_len] = input_ids

		finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
		cur_len  = prompt_len

		ctx = (
			self.hooked_model.hooks(fwd_hooks=fwd_hooks, reset_hooks_end=True, clear_contexts=True)
			if fwd_hooks else nullcontext()
		)

		with ctx:
			if use_kv_cache:
				# Init cache sized to this request
				cfg_for_cache = copy.copy(cfg)
				cfg_for_cache.n_ctx = total_len
				past_kv_cache = HookedTransformerKeyValueCache.init_cache(
					cfg=cfg_for_cache,
					device=device,
					batch_size=batch_size,
				)

				# Prefill once (fills cache)
				logits_last = self._last_logits(
					all_tokens[:, :prompt_len],
					base_mask,
					past_kv_cache=past_kv_cache,
					padding_side=padding_side,
				)

				outputs = self._greedy_decode_cached(
					all_tokens=all_tokens,
					cur_len=cur_len,
					max_new_tokens=max_new_tokens,
					logits_last=logits_last,
					past_kv_cache=past_kv_cache,
					base_mask=base_mask,
					stop_at_eos=stop_at_eos,
					eos_token_id=eos_token_id,
					finished=finished,
					cleanup_every=3,
					padding_side=padding_side,
				)
			else:
				attn_mask = base_mask.new_zeros((batch_size, total_len))
				attn_mask[:, :prompt_len] = base_mask

				outputs = self._greedy_decode_noncached(
					all_tokens=all_tokens,
					attn_mask=attn_mask,
					cur_len=cur_len,
					max_new_tokens=max_new_tokens,
					stop_at_eos=stop_at_eos,
					eos_token_id=eos_token_id,
					finished=finished,
					cleanup_every=3,
					padding_side=padding_side,
				)

		if debug:
			print(f"inputs: {input_ids.shape}, outputs: {outputs.shape}")

		return self._decode_suffix(outputs, prompt_len)

def compute_lengths_all(texts, tokenizer=None, batch_size=256, max_seq_len=None):
	if isinstance(texts, np.ndarray):
		texts = texts.reshape(-1).tolist()
		
	# True per-example token lengths (NO padding). Fast batched tokenizer call.
	if tokenizer is None:
		return np.asarray([len(str(t).split()) for t in texts], dtype=np.int32)

	lens = np.empty(len(texts), dtype=np.int32)
	for start in tqdm(
		range(0, len(texts), batch_size),
		desc="Token lengths",
		leave=False,
	):
		batch_texts = texts[start : start + batch_size]
		enc = tokenizer(
			batch_texts,
			add_special_tokens=False,
			truncation=max_seq_len is not None,
			max_length=max_seq_len,
			padding=False,  # IMPORTANT: don't pad, else lengths become batch max
			return_attention_mask=False,
			return_token_type_ids=False,
		)
		ids = enc["input_ids"]  # list[list[int]]
		n = len(ids)
		lens[start : start + n] = np.fromiter(
			(len(x) for x in ids), dtype=np.int32, count=n
		)
	return lens

def sample_len_tolerant_pairs(source_idx, target_idx, emb_all, texts, k=None, tokenizer=None, len_tolerance=0, metric="cosine", metric_kwargs=None):
	"""
	Fast path for metric in {"cosine","euclidean"}:
	- uses BLAS matmul + streaming best-candidate search (no cdist, no argsort)
	- optional chunking over targets to keep memory bounded

	You may pass metric_kwargs={"chunk_size": 8192} to tune chunking.
	For other metrics, falls back to cdist + argsort.
	"""
	source_idx = np.asarray(source_idx, dtype=np.int64)
	target_idx = np.asarray(target_idx, dtype=np.int64)

	n_source_total = source_idx.size
	n_target = target_idx.size

	if k is None or k <= 0:
		k = n_source_total

	if n_source_total == 0:
		raise ValueError("source_idx is empty: cannot create pairs.")
	if n_target == 0:
		raise ValueError("target_idx is empty: cannot create pairs.")
	if k > n_source_total:
		raise ValueError(
			f"Requested k={k} pairs, but only {n_source_total} sources available. "
			f"Either reduce k or provide more sources."
		)

	len_tolerance = max(0, int(len_tolerance))
	metric_kwargs = {} if metric_kwargs is None else dict(metric_kwargs)

	# --- SAMPLE sources beforehand -------------------------------------------
	if k == n_source_total:
		source_rows = np.arange(n_source_total, dtype=np.int64)
	else:
		source_rows = np.random.choice(n_source_total, size=k, replace=False)

	n_source = source_rows.size
	source_idx_sel = source_idx[source_rows]

	# Texts subset
	source_texts_sel = np.take(texts, source_idx_sel)
	target_texts_all = np.take(texts, target_idx)

	# Lengths as arrays
	source_lens_arr = compute_lengths_all(source_texts_sel, tokenizer=tokenizer)
	target_lens_arr = compute_lengths_all(target_texts_all, tokenizer=tokenizer)

	# Embeddings subset (float32 + contiguous is important for speed)
	X = np.ascontiguousarray(emb_all[source_idx_sel], dtype=np.float32)  # (n_source, d)
	Y = np.ascontiguousarray(emb_all[target_idx], dtype=np.float32)      # (n_target, d)

	# One chosen target per source row
	chosen_target_col = np.full(n_source, -1, dtype=np.int64)
	assigned_pass = np.zeros(n_source, dtype=np.int8)  # 0=unassigned, 1..4
	used_target_mask = np.zeros(n_target, dtype=bool)
	source_order = np.random.permutation(n_source)

	# --- FAST PATHS ----------------------------------------------------------
	if metric in ("cosine", "euclidean"):
		chunk_size = int(metric_kwargs.pop("chunk_size", 4096))
		if metric_kwargs:
			raise ValueError(
				f"metric_kwargs {list(metric_kwargs.keys())} not supported for fast '{metric}' path "
				f"(only 'chunk_size' is supported)."
			)

		eps = 1e-12

		best_any_idx = np.full(n_source, -1, dtype=np.int64)
		best_len_idx = np.full(n_source, -1, dtype=np.int64)

		if metric == "cosine":
			# normalize for cosine sim
			Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)
			Yn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), eps)

			best_any_val = np.full(n_source, -np.inf, dtype=np.float32)  # best similarity
			best_len_val = np.full(n_source, -np.inf, dtype=np.float32)

			# stream over target chunks
			for j0 in range(0, n_target, chunk_size):
				j1 = min(j0 + chunk_size, n_target)
				Yb = Yn[j0:j1]  # (b, d)

				sim = Xn @ Yb.T  # (n_source, b), float32, BLAS

				# best overall
				block_best = sim.max(axis=1)
				block_arg = sim.argmax(axis=1) + j0
				better = block_best > best_any_val
				best_any_val[better] = block_best[better]
				best_any_idx[better] = block_arg[better]

				# best length-matched (mask in-place AFTER best overall update)
				ok = (np.abs(source_lens_arr[:, None] - target_lens_arr[j0:j1][None, :]) <= len_tolerance)
				sim[~ok] = -np.inf
				block_best = sim.max(axis=1)
				block_arg = sim.argmax(axis=1) + j0
				better = block_best > best_len_val
				best_len_val[better] = block_best[better]
				best_len_idx[better] = block_arg[better]

			# distances for chosen pairs come from similarities
			# (fill later depending on whether we used len-match or not)

		else:  # euclidean
			# Use streaming min of squared distances:
			# ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
			x2 = np.sum(X * X, axis=1, keepdims=True)  # (n_source,1)

			best_any_val = np.full(n_source, np.inf, dtype=np.float32)   # best d^2
			best_len_val = np.full(n_source, np.inf, dtype=np.float32)

			for j0 in range(0, n_target, chunk_size):
				j1 = min(j0 + chunk_size, n_target)
				Yb = Y[j0:j1]
				y2 = np.sum(Yb * Yb, axis=1, keepdims=True).T  # (1,b)
				xy = X @ Yb.T                                   # (n_source,b)
				d2 = np.maximum(x2 + y2 - 2.0 * xy, 0.0).astype(np.float32, copy=False)

				# best overall
				block_best = d2.min(axis=1)
				block_arg = d2.argmin(axis=1) + j0
				better = block_best < best_any_val
				best_any_val[better] = block_best[better]
				best_any_idx[better] = block_arg[better]

				# best length-matched (mask in-place AFTER best overall update)
				ok = (np.abs(source_lens_arr[:, None] - target_lens_arr[j0:j1][None, :]) <= len_tolerance)
				d2[~ok] = np.inf
				block_best = d2.min(axis=1)
				block_arg = d2.argmin(axis=1) + j0
				better = block_best < best_len_val
				best_len_val[better] = block_best[better]
				best_len_idx[better] = block_arg[better]

		# --- Greedy assignment (same semantics as before) --------------------
		# Pass1 assigns any source that has a length-matched target (even if reused).
		# Pass2 assigns remaining sources to their best overall target.
		for r in source_order:
			if best_len_idx[r] >= 0:
				col = int(best_len_idx[r])
				assigned_pass[r] = 2 if used_target_mask[col] else 1
			else:
				col = int(best_any_idx[r])
				assigned_pass[r] = 4 if used_target_mask[col] else 3
			chosen_target_col[r] = col
			used_target_mask[col] = True

		# --- Build final pairs + pair_meta ----------------------------------
		pairs = []
		pair_meta = []

		for r in range(n_source):
			global_source_id = int(source_idx_sel[r])
			col = int(chosen_target_col[r])
			global_target_id = int(target_idx[col])

			source_len = int(source_lens_arr[r])
			target_len = int(target_lens_arr[col])
			len_diff = target_len - source_len
			pass_num = int(assigned_pass[r])

			if metric == "cosine":
				# distance = 1 - similarity
				sim_val = float(best_len_val[r]) if (best_len_idx[r] == col) else float(best_any_val[r])
				dist_val = float(1.0 - sim_val)
			else:
				# euclidean: dist = sqrt(d^2)
				d2_val = float(best_len_val[r]) if (best_len_idx[r] == col) else float(best_any_val[r])
				dist_val = float(np.sqrt(d2_val))

			pairs.append((global_source_id, global_target_id))
			pair_meta.append(
				{
					"source_idx": global_source_id,
					"target_idx": global_target_id,
					"source_len": source_len,
					"target_len": target_len,
					"len_diff": int(len_diff),
					"distance": dist_val,
					"assignment_pass": pass_num,
				}
			)

	else:
		# --- Fallback: original approach for arbitrary metrics --------------
		dists = cdist(X, Y, metric=metric, **metric_kwargs)
		dists = np.nan_to_num(dists, nan=1e30)
		labels = np.argsort(dists, axis=1)

		for r in source_order:
			# pass1: length matched
			row_len_ok = np.abs(source_lens_arr[r] - target_lens_arr) <= len_tolerance
			if row_len_ok.any():
				row_labels = labels[r]
				valid = row_len_ok[row_labels]
				pos = np.nonzero(valid)[0]
				if pos.size:
					col = int(row_labels[pos[0]])
					assigned_pass[r] = 2 if used_target_mask[col] else 1
					chosen_target_col[r] = col
					used_target_mask[col] = True
					continue

			# pass2: best overall
			col = int(labels[r, 0])
			assigned_pass[r] = 4 if used_target_mask[col] else 3
			chosen_target_col[r] = col
			used_target_mask[col] = True

		pairs = []
		pair_meta = []
		for r in range(n_source):
			global_source_id = int(source_idx_sel[r])
			col = int(chosen_target_col[r])
			global_target_id = int(target_idx[col])
			source_len = int(source_lens_arr[r])
			target_len = int(target_lens_arr[col])
			len_diff = target_len - source_len
			dist_val = float(dists[r, col])

			pairs.append((global_source_id, global_target_id))
			pair_meta.append(
				{
					"source_idx": global_source_id,
					"target_idx": global_target_id,
					"source_len": source_len,
					"target_len": target_len,
					"len_diff": int(len_diff),
					"distance": dist_val,
					"assignment_pass": int(assigned_pass[r]),
				}
			)

	# --- Enrich pair_meta + aggregate meta -----------------------------------
	pass_label_map = {
		1: "length-matched + unique",
		2: "length-matched",
		3: "unique",
		4: "any",
	}

	target_usage = {}
	for pm in pair_meta:
		tgt = pm["target_idx"]
		target_usage[tgt] = target_usage.get(tgt, 0) + 1

	unique_target_used = len(target_usage)
	target_reused_count = sum(c - 1 for c in target_usage.values() if c > 1)

	assignment_pass_counts = {
		"length-matched + unique": 0,
		"length-matched": 0,
		"unique": 0,
		"any": 0,
	}
	len_matched_count = 0

	for pm in pair_meta:
		len_matched = abs(pm["len_diff"]) <= len_tolerance
		pm["len_matched"] = bool(len_matched)
		if len_matched:
			len_matched_count += 1

		reuse = target_usage[pm["target_idx"]]
		pm["reuse_count"] = reuse
		pm["is_unique_target"] = (reuse == 1)

		label = pass_label_map.get(pm["assignment_pass"], "any")
		pm["assignment_label"] = label
		assignment_pass_counts[label] += 1

	meta = {
		"n_source": int(len(pairs)),
		"n_target": int(n_target),
		"k": int(k),
		"len_tolerance": int(len_tolerance),
		"metric": str(metric),
		"unique_target_used": int(unique_target_used),
		"target_reused_count": int(target_reused_count),
		"len_matched_count": int(len_matched_count),
		"assignment_pass_counts": assignment_pass_counts,
		"pair_meta": pair_meta,
	}

	return pairs, meta

@torch.inference_mode()
def precompute_mean_activations(
	model,
	all_prompts,
	layer_to_neurons,
	n_points,
	batch_size,
	device=None,
	intervention="mean",
):
	"""
	Compute replacement activations for requested units.

	`layer_to_neurons` can be:
	  - {int_layer: [idx, ...]}                      (legacy MLP-only form)
	  - {"m16": [idx, ...], "a16.h21": [idx, ...]}   (recommended canonical labels)
	  - {(L,H): [idx,...]}                           (legacy attention form, treated as aL.hH)

	For intervention="mean", `mean_global` stores literal dataset means.
	For intervention="mean-donor", `mean_global` stores the observed activation value
	closest to the dataset mean for each requested unit.
	For intervention="mean-positional", `mean_global` stores literal dataset means
	and `mean_per_pos` stores per-position literal means.
	For intervention="mean-donor-positional", `mean_global` stores global mean-donor values
	and `mean_per_pos` stores per-position mean-donor values, with later hook-time fallback
	to `mean_global` for positions beyond the observed range.

	We compute statistics over:
	  - MLP: blocks.{L}.hook_mlp_out[..., idx]
	  - Attn: blocks.{L}.attn.hook_z[..., H, idx]
	"""
	if n_points is None or n_points <= 0 or not layer_to_neurons:
		return None

	if device is None:
		device = model.hooked_model.cfg.device

	# ---- Normalize keys to canonical string labels ----
	unit_to_neurons = {}
	for k, ns in dict(layer_to_neurons).items():
		if ns is None:
			continue
		if isinstance(k, int):
			unit_key = f"m{int(k)}"
		elif isinstance(k, tuple) and len(k) == 2:
			L, H = k
			unit_key = f"a{int(L)}.h{int(H)}"
		else:
			unit_key = _canonical_unit_label(str(k)) or str(k)
		unit_to_neurons[unit_key] = list(ns)

	# Deterministic sorted unique ids per unit
	unit_to_neurons = {
		u: sorted({int(n) for n in ns})
		for u, ns in unit_to_neurons.items()
		if ns
	}

	if not unit_to_neurons:
		return None

	n_points = min(int(n_points), len(all_prompts))
	if n_points == 0:
		return None

	subset_prompts = list(all_prompts)[:n_points]

	mean_idx = {
		u: torch.tensor(neuron_ids, dtype=torch.long, device=device)
		for u, neuron_ids in unit_to_neurons.items()
	}

	# Global sums & counts per layer
	global_sums = {
		u: torch.zeros(mean_idx[u].numel(), dtype=torch.float32, device=device)
		for u in unit_to_neurons.keys()
	}
	global_counts = {u: 0 for u in unit_to_neurons.keys()}

	want_pos = intervention in ("mean-positional", "mean-donor-positional")
	want_mean_donor = intervention in ("mean-donor", "mean-donor-positional")

	# Per-position sums & counts (ragged during accumulation)
	pos_sums = None
	pos_counts = None
	if want_pos:
		pos_sums = {u: {} for u in unit_to_neurons.keys()}
		pos_counts = {u: {} for u in unit_to_neurons.keys()}

	def _accumulate(u: str, acts: torch.Tensor):
		# acts: [B, T, K]
		B, T, K = acts.shape
		global_sums[u] += acts.sum(dim=(0, 1)).detach()
		global_counts[u] += B * T

		if want_pos:
			assert pos_sums is not None and pos_counts is not None
			for t in range(T):
				if t not in pos_sums[u]:
					pos_sums[u][t] = torch.zeros(K, dtype=torch.float32, device=device)
					pos_counts[u][t] = 0
				pos_sums[u][t] += acts[:, t, :].sum(dim=0).detach()
				pos_counts[u][t] += B

	def make_hook(unit_key: str):
		parsed = get_layer_type_and_ids(unit_key)
		if not parsed:
			def _hook(x, hook):
				return x
			return "", _hook

		layer_type, L, H = parsed
		idx = mean_idx[unit_key]  # [K]

		if layer_type == "mlp":
			hook_name = f"blocks.{int(L)}.hook_mlp_out"

			def _hook(mlp_out, hook):
				if mlp_out.ndim != 3:
					return mlp_out
				acts = mlp_out[:, :, idx]
				_accumulate(unit_key, acts)
				return mlp_out

			return hook_name, _hook

		if layer_type == "attn":
			hook_name = f"blocks.{int(L)}.attn.hook_z"

			def _hook(z, hook, H=H):
				if z.ndim == 4:
					acts = z[:, :, int(H), :].index_select(-1, idx)
				elif z.ndim == 3:
					acts = z[:, :, :].index_select(-1, idx)
				else:
					return z
				_accumulate(unit_key, acts)
				return z

			return hook_name, _hook

		def _hook(x, hook):
			return x
		return "", _hook

	hooks = []
	for u in unit_to_neurons.keys():
		hname, hk = make_hook(u)
		if hname:
			hooks.append((hname, hk))

	if not hooks:
		return None

	dataloader = torch.utils.data.DataLoader(
		subset_prompts,
		batch_size=batch_size,
		shuffle=False,
	)

	for batch_prompts in tqdm(dataloader, desc="Mean activations", leave=False):
		_ = model.generate(
			batch_prompts,
			do_sample=False,
			max_new_tokens=1,
			fwd_hooks=hooks,
		)

	# Build literal dataset means first.
	literal_global = {}
	for u in unit_to_neurons.keys():
		if global_counts[u] > 0:
			literal_global[u] = (global_sums[u] / float(global_counts[u])).detach()
		else:
			literal_global[u] = torch.zeros_like(global_sums[u])

	# Build literal per-position means first: dense [Tmax, K]
	literal_per_pos = None
	if want_pos:
		assert pos_sums is not None and pos_counts is not None
		literal_per_pos = {}
		for u in unit_to_neurons.keys():
			if len(pos_sums[u]) == 0:
				K = mean_idx[u].numel()
				literal_per_pos[u] = torch.zeros((0, K), dtype=torch.float32, device=device)
				continue

			Tmax = max(pos_sums[u].keys()) + 1
			K = mean_idx[u].numel()
			out = torch.zeros((Tmax, K), dtype=torch.float32, device=device)

			for t, sums_t in pos_sums[u].items():
				cnt = pos_counts[u].get(t, 0)
				if cnt > 0:
					out[t] = (sums_t / float(cnt)).detach()
				# else: leave zeros

			literal_per_pos[u] = out

	# For mean-donor variants, replace each mean with the closest observed activation value.
	if want_mean_donor:
		best_global_vals = {
			u: torch.zeros_like(literal_global[u])
			for u in unit_to_neurons.keys()
		}
		best_global_diffs = {
			u: torch.full_like(literal_global[u], float("inf"))
			for u in unit_to_neurons.keys()
		}

		best_pos_vals = None
		best_pos_diffs = None
		if want_pos:
			assert literal_per_pos is not None
			best_pos_vals = {
				u: literal_per_pos[u].clone()
				for u in unit_to_neurons.keys()
			}
			best_pos_diffs = {
				u: torch.full_like(literal_per_pos[u], float("inf"))
				for u in unit_to_neurons.keys()
			}

		def _update_mean_donor(u: str, acts: torch.Tensor):
			acts = acts.detach().to(torch.float32)
			flat = acts.reshape(-1, acts.shape[-1])
			if flat.numel() == 0:
				return

			global_diffs = (flat - literal_global[u].view(1, -1)).abs()
			vals_diff, vals_idx = global_diffs.min(dim=0)
			vals = flat[vals_idx, torch.arange(flat.shape[-1], device=flat.device)]
			better = vals_diff < best_global_diffs[u]
			if bool(better.any()):
				best_global_diffs[u] = torch.where(better, vals_diff, best_global_diffs[u])
				best_global_vals[u] = torch.where(better, vals, best_global_vals[u])

			if want_pos and best_pos_vals is not None and best_pos_diffs is not None and literal_per_pos is not None:
				T = min(int(acts.shape[1]), int(literal_per_pos[u].shape[0]))
				for t in range(T):
					acts_t = acts[:, t, :]
					if acts_t.numel() == 0:
						continue
					diffs_t = (acts_t - literal_per_pos[u][t].view(1, -1)).abs()
					vals_diff_t, vals_idx_t = diffs_t.min(dim=0)
					vals_t = acts_t[vals_idx_t, torch.arange(acts_t.shape[-1], device=acts_t.device)]
					better_t = vals_diff_t < best_pos_diffs[u][t]
					if bool(better_t.any()):
						best_pos_diffs[u][t] = torch.where(better_t, vals_diff_t, best_pos_diffs[u][t])
						best_pos_vals[u][t] = torch.where(better_t, vals_t, best_pos_vals[u][t])

		def make_mean_donor_hook(unit_key: str):
			parsed = get_layer_type_and_ids(unit_key)
			if not parsed:
				def _hook(x, hook):
					return x
				return "", _hook

			layer_type, L, H = parsed
			idx = mean_idx[unit_key]

			if layer_type == "mlp":
				hook_name = f"blocks.{int(L)}.hook_mlp_out"

				def _hook(mlp_out, hook):
					if mlp_out.ndim == 3:
						_update_mean_donor(unit_key, mlp_out[:, :, idx])
					return mlp_out

				return hook_name, _hook

			if layer_type == "attn":
				hook_name = f"blocks.{int(L)}.attn.hook_z"

				def _hook(z, hook, H=H):
					if z.ndim == 4:
						_update_mean_donor(unit_key, z[:, :, int(H), :].index_select(-1, idx))
					elif z.ndim == 3:
						_update_mean_donor(unit_key, z[:, :, :].index_select(-1, idx))
					return z

				return hook_name, _hook

			def _hook(x, hook):
				return x
			return "", _hook

		mean_donor_hooks = []
		for u in unit_to_neurons.keys():
			hname, hk = make_mean_donor_hook(u)
			if hname:
				mean_donor_hooks.append((hname, hk))

		for batch_prompts in tqdm(dataloader, desc="Mean-donor activations", leave=False):
			_ = model.generate(
				batch_prompts,
				do_sample=False,
				max_new_tokens=1,
				fwd_hooks=mean_donor_hooks,
			)

		mean_global = best_global_vals
		mean_per_pos = best_pos_vals if want_pos else None
	else:
		mean_global = literal_global
		mean_per_pos = literal_per_pos

	return MeanActTensors(
		mean_idx=mean_idx,
		mean_global=mean_global,
		mean_per_pos=mean_per_pos,
		literal_global=literal_global,
		literal_per_pos=literal_per_pos,
	)
# --- Helpers for mapping metadata_topn → MLP layers -----------------

_EDGE_RE = re.compile(r"^(?P<src>.+?)->(?P<dst>.+)$")
_ANGLE_RE = re.compile(r"<.*?>")

def strip_angle_suffix(s):
	# removes "<q>", "<k>", "<v>", etc.
	return _ANGLE_RE.sub("", s).strip()

def parse_edge_endpoints(edge_name):
	"""
	Returns (src, dst) WITHOUT <...> suffixes.

	Examples:
	  "a16.h20->a20.h14<v>" -> ("a16.h20", "a20.h14")
	  "m1->a9.h1<k>"        -> ("m1", "a9.h1")
	  "a6.h8->m24"          -> ("a6.h8", "m24")
	  "input->a6.h12<k>"    -> ("input", "a6.h12")
	"""
	m = _EDGE_RE.match(edge_name.strip())
	if not m:
		return None, None
	src = strip_angle_suffix(m.group("src"))
	dst = strip_angle_suffix(m.group("dst"))
	return src, dst

def derive_mlp_layer_scores_from_meta(meta):
	"""
	Return (layer_score: dict[str,float], layer_labels: list[str]) inferred from meta.

	- Node level:
		node_label_score keys might be "m10", "a16.h20", etc.
		We just take those keys as they are and aggregate by max |score|.

	- Edge level:
		edge_label_score keys are edges like "a16.h20->a20.h14<v>".
		We parse endpoints and aggregate scores on those endpoint labels
		("a16.h20", "a20.h14", "m24", ...).

	This function does **not** try to map attention heads to MLP blocks.
	It returns arbitrary layer-ish labels; the caller can later filter
	to MLP-only by using get_layer_type_and_ids.
	"""
	level = meta.get("level", None)

	# NODE LEVEL
	if level == "node":
		node_scores = meta.get("node_label_score", {}) or {}
		layer_score = defaultdict(float)

		for node_label, v in node_scores.items():
			v = float(v)
			layer_score[str(node_label)] = max(layer_score[str(node_label)], abs(v))

		layer_labels = sorted(layer_score.keys(), key=lambda L: layer_score[L], reverse=True)
		return dict(layer_score), layer_labels

	# EDGE LEVEL
	if level == "edge":
		edge_scores = meta.get("edge_label_score", {}) or {}
		layer_score = defaultdict(float)

		for edge_name, v in edge_scores.items():
			v = float(v)
			src, dst = parse_edge_endpoints(str(edge_name))
			if src is None or dst is None:
				continue

			for ep in (src, dst):
				layer_score[ep] = max(layer_score[ep], abs(v))

		layer_labels = sorted(layer_score.keys(), key=lambda L: layer_score[L], reverse=True)
		return dict(layer_score), layer_labels

	# Anything else
	return {}, []

def evenly_spaced_indices(n, k):
	"""
	Deterministic selection of k indices in [0, n-1], spread across the range.
	"""
	if k <= 0:
		return []
	if k >= n:
		return list(range(n))
	if k == 1:
		return [n // 2]

	step = (n - 1) / (k - 1)
	out = []
	last = -1
	for i in range(k):
		idx = int(round(i * step))
		if idx <= last:
			idx = last + 1
		if idx >= n:
			idx = n - 1
		out.append(idx)
		last = idx
	return out

def allocate_topk_neurons_across_layers(mlp_layers, layer_score, d_model, top_k):
	"""
	Fallback when we *don't* have neuron-level scores:
	distribute top_k neuron picks across top-scoring layers.
	"""
	if top_k <= 0:
		return {L: list(range(d_model)) for L in mlp_layers}

	# Sort layers by inferred layer_score (desc). If missing score, treat as 0.
	ordered = sorted(mlp_layers, key=lambda L: layer_score.get(L, 0.0), reverse=True)

	remaining = int(top_k)
	out = {}
	for L in ordered:
		if remaining <= 0:
			break
		k_here = min(d_model, remaining)
		out[L] = evenly_spaced_indices(d_model, k_here)
		remaining -= k_here

	return out


def _infer_n_heads_from_layer_labels(layer_labels):
	head_ids = []
	for layer_label in layer_labels or []:
		parsed = get_layer_type_and_ids(str(layer_label))
		if parsed and parsed[0] == "attn" and parsed[2] is not None:
			head_ids.append(int(parsed[2]))
	return (max(head_ids) + 1) if head_ids else None


def _resolve_manifest_unit_widths(meta, entry, args, layer_labels):
	"""
	Resolve per-node widths needed to expand node/edge manifests into ablation-ready units.

	For MLP nodes we need d_model; for attention-head nodes we need d_head.
	Script 5 currently stores d_model in metadata_topn but not d_head/n_heads, so we
	infer n_heads from the manifest labels when needed and then derive d_head=d_model/n_heads.
	"""
	d_model = meta.get("d_model") or entry.get("d_model") or getattr(args, "d_model", None)
	n_heads = meta.get("n_heads") or entry.get("n_heads") or getattr(args, "n_heads", None)
	d_head = meta.get("d_head") or entry.get("d_head") or getattr(args, "d_head", None)

	try:
		d_model = int(d_model) if d_model is not None else None
	except Exception:
		d_model = None
	try:
		n_heads = int(n_heads) if n_heads is not None else None
	except Exception:
		n_heads = None
	try:
		d_head = int(d_head) if d_head is not None else None
	except Exception:
		d_head = None

	if n_heads is None:
		n_heads = _infer_n_heads_from_layer_labels(layer_labels)

	if d_head is None and d_model and n_heads and n_heads > 0 and d_model % n_heads == 0:
		d_head = d_model // n_heads

	return d_model, d_head, n_heads


def get_circuit_neurons_dict(circuit_manifest_path, args, quiet=False):
	manifest = json.loads(circuit_manifest_path.read_text())
	if not isinstance(manifest, list) or not manifest:
		raise RuntimeError(f"Empty or malformed manifest at {circuit_manifest_path}")

	# Build map: circuit_id -> components
	circuit_entries = {}
	for entry in manifest:
		entry_status = entry['status']
		if entry_status != 'ok' and entry_status != 'full_network_fallback':
			continue

		rid = entry.get("circuit_id", entry.get("rule_id", entry.get("cluster_index")))
		rule = entry["circuit_label"]
		rule_direction = entry["coefficient_sign"]
		rule_target = entry["target"]
		sampl_strategy = entry.get("sampling_strategy")
		analysis_mode = ("spectral_cluster" if sampl_strategy == "spectral_clusters" else "rule")
		if rid is None or rule is None or rule_target is None:
			continue

		# In spectral-cluster mode, only keep cluster entries; in rule mode, drop them.
		if args.cluster_by_spectral:
			if sampl_strategy != "spectral_clusters":
				continue
		else:
			if sampl_strategy == "spectral_clusters":
				continue

		meta = entry["metadata_topn"]

		# 1) Try neuron-level payload first (level='neuron' case)
		neurons = meta.get("neurons") or meta.get("mlp_neurons")

		# 2) If missing (level='node'/'edge'), infer layer/head labels from meta and expand to units.
		inferred = False
		if not neurons:
			_layer_score, layer_labels = derive_mlp_layer_scores_from_meta(meta)
			if not layer_labels:
				if not quiet:
					print(
						f"Missing neurons and cannot infer layers for rule {rid} ({rule}). "
						f"manifest entry level={meta.get('level')}"
					)
				continue

			d_model, d_head, n_heads = _resolve_manifest_unit_widths(meta, entry, args, layer_labels)
			has_mlp = any(
				(parsed := get_layer_type_and_ids(str(L))) and parsed[0] == "mlp"
				for L in layer_labels
			)
			has_attn = any(
				(parsed := get_layer_type_and_ids(str(L))) and parsed[0] == "attn"
				for L in layer_labels
			)

			if has_mlp and (not d_model or int(d_model) <= 0):
				if not quiet:
					print(
						f"Missing d_model to expand {meta.get('level')} selection into units for rule {rid} ({rule}). "
						f"Add meta['d_model'] in get_topn (recommended) or pass --d_model."
					)
				continue
			if has_attn and (not d_head or int(d_head) <= 0):
				if not quiet:
					msg = (
						f"Missing d_head to expand {meta.get('level')} attention selections into units for rule {rid} ({rule}). "
						f"Provide meta['d_head'] / meta['n_heads'], pass --d_head / --n_heads, or ensure d_model is divisible by the inferred number of heads"
					)
					if n_heads is not None:
						msg += f" (inferred n_heads={int(n_heads)})."
					print(msg)
				continue

			# Expand each inferred node label to all units within that node.
			expanded = {}
			for L in layer_labels:
				parsed = get_layer_type_and_ids(str(L))
				if not parsed:
					continue
				layer_type, _, _ = parsed
				if layer_type == "mlp":
					expanded[str(L)] = list(range(int(d_model)))
				elif layer_type == "attn":
					expanded[str(L)] = list(range(int(d_head)))
			neurons = expanded
			inferred = True

		all_neurons = {
			str(layer_label): neuron_list
			for layer_label, neuron_list in (neurons or {}).items()
			if (parsed := get_layer_type_and_ids(str(layer_label))) and parsed[0] in ("mlp", "attn")
		}

		if not all_neurons:
			if not quiet:
				print(f"No usable units found for rule {rid} ({rule}) at level={meta.get('level')}.")
			continue

		mlp_neurons = {k: v for k, v in all_neurons.items() if get_layer_type_and_ids(k)[0] == "mlp"}
		attn_neurons = {k: v for k, v in all_neurons.items() if get_layer_type_and_ids(k)[0] == "attn"}

		if getattr(args, "top_k_neurons", None) and args.top_k_neurons > 0 and all_neurons:
			orig_all_neurons = dict(all_neurons)
			neuron_scores = meta.get("neuron_label_score", None)

			# If we have true neuron-level scores, use the original exact filtering logic
			if neuron_scores:
				best_neuron_scores = dict(
					sorted(neuron_scores.items(), key=lambda x: x[-1], reverse=True)[: args.top_k_neurons]
				)
				best_keys = set(best_neuron_scores.keys())

				def _keep(layer_label, neuron_id):
					# Accept both single- and double-quote tuple string formats.
					return (
						f"('{layer_label}', {neuron_id})" in best_keys
						or f'("{layer_label}", {neuron_id})' in best_keys
					)

				filtered = {
					layer_label: [nid for nid in neuron_list if _keep(layer_label, nid)]
					for layer_label, neuron_list in orig_all_neurons.items()
				}
				filtered = dict(filter(lambda x: x[-1], filtered.items()))

				# If neuron-level scores don't cover these units, fall back to deterministic selection.
				all_neurons = filtered if filtered else orig_all_neurons

			else:
				all_neurons = orig_all_neurons

			# No neuron-level scores (or uncovered units): pick deterministically across candidate units,
			# prioritizing layers/heads with larger inferred layer_score.
			if all_neurons is orig_all_neurons:
				layer_score, _ = derive_mlp_layer_scores_from_meta(meta)
				ordered_layers = sorted(all_neurons.keys(), key=lambda L: layer_score.get(L, 0.0), reverse=True)

				remaining = int(args.top_k_neurons)
				selected = {}
				for L in ordered_layers:
					if remaining <= 0:
						break
					cands = sorted(set(map(int, all_neurons.get(L, []))))
					if not cands:
						continue
					k_here = min(len(cands), remaining)
					idxs = evenly_spaced_indices(len(cands), k_here)
					selected[L] = [cands[i] for i in idxs]
					remaining -= k_here

				if selected:
					all_neurons = selected

			mlp_neurons = {k: v for k, v in all_neurons.items() if get_layer_type_and_ids(k)[0] == "mlp"}
			attn_neurons = {k: v for k, v in all_neurons.items() if get_layer_type_and_ids(k)[0] == "attn"}

		if getattr(args, "last_k_layers", None) is not None and args.last_k_layers > 0 and mlp_neurons:
			layer_ids = []
			for layer_label in mlp_neurons.keys():
				parsed = get_layer_type_and_ids(layer_label)
				if not parsed:
					continue
				layer_type, L, _ = parsed
				if layer_type == "mlp":
					layer_ids.append(L)

			if layer_ids:
				unique_sorted = sorted(set(layer_ids))
				k = min(args.last_k_layers, len(unique_sorted))
				keep_ids = set(unique_sorted[-k:])

				mlp_neurons = {
					layer_label: neuron_list
					for layer_label, neuron_list in mlp_neurons.items()
					if (parsed := get_layer_type_and_ids(layer_label)) and parsed[1] in keep_ids
				}

		# Rebuild all_neurons after MLP-only filters
		all_neurons = dict(attn_neurons)
		all_neurons.update(mlp_neurons)

		circuit_entries[(rule, rule_target)] = {
			"circuit_id": int(rid),
			"circuit_label": rule,
			"rule_direction": rule_direction,
			"rule_target": rule_target,
			"analysis_mode": analysis_mode,
			"cluster_index": (int(rid) if analysis_mode == "spectral_cluster" and rid is not None else None),
			"circuit_neurons_count": sum(map(len, all_neurons.values())),
			"neurons": all_neurons,
			"mlp_neurons": mlp_neurons,
			"attn_neurons": attn_neurons,
			"_neurons_inferred_from_level": bool(inferred),
		}
	return circuit_entries
