import os
import numpy as np
import torch
from functools import partial

from lib.eap.evaluate import evaluate_graph
from lib.eap.utils import compute_mean_activations
from lib.caching_and_prompting import load_or_create_cache

def select_labels(rest, device):
	assert rest and len(rest) >= 1 and isinstance(rest[0], dict), rest
	lab = rest[0]
	out = {}
	for k, v in lab.items():
		out[k] = v.to(device) if torch.is_tensor(v) else v
	return out

def per_pos_kl(p_logits, q_logits):
	p = torch.softmax(p_logits.float(), dim=-1)
	q = torch.softmax(q_logits.float(), dim=-1)
	return (p * (torch.log(p + 1e-9) - torch.log(q + 1e-9))).sum(dim=-1)

def masked_mean(x, mask):
	num = (x * mask).sum()
	den = mask.sum().clamp_min(1.0)
	return num / den

def ensure_mask_targets_shape(logits, labels, strict=True):
	B, T, V = logits.shape

	# 1) If a ready-made span mask was provided, handle like before.
	mask = labels.get("span_mask", None)
	if mask is not None:
		if not torch.is_tensor(mask):
			mask = torch.as_tensor(mask)
		mask = mask.to(logits.device).float()
		if mask.dim() == 1:
			mask = mask.unsqueeze(0).expand(B, -1)
		elif mask.size(0) != B:
			if mask.size(0) == 1:
				mask = mask.expand(B, -1)
			else:
				raise RuntimeError(f"Batch mismatch: mask B={mask.size(0)} vs logits B={B}")
		# align to T
		if mask.size(1) > T:
			mask = mask[:, :T]
		elif mask.size(1) < T:
			pad = torch.zeros((B, T - mask.size(1)), device=logits.device, dtype=mask.dtype)
			mask = torch.cat([mask, pad], dim=1)
	else:
		# 2) Build the mask from lengths: mark positions [full_len-1-answer_len, full_len-1)
		#    (i.e., next-token positions covering exactly the answer, exclude the final token)
		ans = labels.get("answer_len", 0)
		full = labels.get("full_len", T)

		if not torch.is_tensor(ans):
			ans = torch.as_tensor(ans)
		if not torch.is_tensor(full):
			full = torch.as_tensor(full)
		ans = ans.to(logits.device).long()
		full = full.to(logits.device).long()

		if ans.dim() == 0:  
			ans  = ans.unsqueeze(0).expand(B)
		if full.dim() == 0: 
			full = full.unsqueeze(0).expand(B)

		if strict:
			if (ans < 0).any() or (ans > T - 1).any() or (full < 1).any() or (full > T).any():
				raise ValueError(f"Out-of-bounds: ans∈[0,{T-1}], full∈[1,{T}] but got ans={ans}, full={full}")
		else:
			ans = ans.clamp(0, T - 1)
			full = full.clamp(1, T)

		# clamp to reasonable bounds
		ans = ans.clamp(min=0, max=T - 1)
		full = full.clamp(min=1, max=T)

		# Vectorized range mask
		pos = torch.arange(T, device=logits.device).unsqueeze(0).expand(B, T)
		start = (full - 1 - ans).clamp(min=0).unsqueeze(1)  # inclusive
		end = (full - 1).clamp(min=0).unsqueeze(1)  # exclusive
		mask = ((pos >= start) & (pos < end)).float()

	# Targets (optional). If missing/ignored, metrics fall back to clean argmax.
	tgt = labels.get("target_ids", None)
	if tgt is not None:
		if not torch.is_tensor(tgt):
			tgt = torch.as_tensor(tgt)
		tgt = tgt.to(logits.device).long()
		if tgt.dim() == 1:
			tgt = tgt.unsqueeze(0).expand(B, -1)
		elif tgt.size(0) != B:
			if tgt.size(0) == 1:
				tgt = tgt.expand(B, -1)
			else:
				raise RuntimeError(f"Batch mismatch: tgt B={tgt.size(0)} vs logits B={B}")
		# align to T
		if tgt.size(1) > T:
			tgt = tgt[:, :T]
		elif tgt.size(1) < T:
			pad = torch.full((B, T - tgt.size(1)), -100, device=logits.device, dtype=tgt.dtype)
			tgt = torch.cat([tgt, pad], dim=1)
	else:
		tgt = None

	return mask, tgt

def aggregate_logits_over_time(logits, mask = None, reduction = "mean"):
	"""
	Reduce [B, T, V] over T -> [B, V] using mean/max/min.
	If a span mask is provided (float [B, T]), aggregation is restricted to masked steps.
	"""
	assert logits.dim() == 3, f"expected [B,T,V], got {tuple(logits.shape)}"
	B, T, V = logits.shape
	logits = logits.float()

	if mask is None:
		if reduction == "max":
			return logits.max(dim=1).values
		elif reduction == "min":
			return logits.min(dim=1).values
		else:
			return logits.mean(dim=1)

	mask = mask.to(logits.device).float()
	m3 = mask.unsqueeze(-1)  # [B,T,1]

	if reduction == "mean":
		num = (logits * m3).sum(dim=1)  # [B,V]
		den = mask.sum(dim=1, keepdim=True).clamp_min(1)  # [B,1]
		return num / den
	elif reduction == "max":
		masked = logits.masked_fill(m3 <= 0, float("-inf"))
		out = masked.max(dim=1).values
		out[~torch.isfinite(out)] = 0.0
		return out
	elif reduction == "min":
		masked = logits.masked_fill(m3 <= 0, float("inf"))
		out = masked.min(dim=1).values
		out[~torch.isfinite(out)] = 0.0
		return out
	else:
		raise ValueError(f"Unknown reduction '{reduction}'")

def kl_vec(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
	"""KL(p || q) for aggregated logits [B,V] -> [B]."""
	assert p_logits.shape == q_logits.shape and p_logits.dim() == 2
	p = torch.softmax(p_logits.float(), dim=-1)
	q = torch.softmax(q_logits.float(), dim=-1)
	return (p * (torch.log(p + 1e-9) - torch.log(q + 1e-9))).sum(dim=-1)  # [B]

# ---------------- Faithfulness: NL metric (span-aware) ----------------
def m_kl_span(logits, clean_logits, *rest, temporal_agg: str = "none", **__):
	"""
	If temporal_agg is 'none': original per-step KL masked over the span.
	Else: aggregate logits across time first (span-restricted), then KL(p||q) on [B,V].
	"""
	device = logits.device
	labels = select_labels(rest, device)
	mask, _ = ensure_mask_targets_shape(logits, labels, strict=False)

	if temporal_agg == "none":
		kl = per_pos_kl(clean_logits, logits)  # [B,T]
		return masked_mean(kl, mask)
	else:
		p = aggregate_logits_over_time(clean_logits, mask=mask, reduction=temporal_agg)  # [B,V]
		q = aggregate_logits_over_time(logits, mask=mask, reduction=temporal_agg)        # [B,V]
		kl = kl_vec(p, q)  # [B]
		return kl.mean()


def m_nl_ratio_span(logits, clean_logits, *rest, temporal_agg: str = "none", **__):
	"""
	If temporal_agg is 'none': per-step normalized logit ratio masked over the span.
	Else: aggregate logits across time first, then compute the ratio on [B,V].
	"""
	device = logits.device
	labels = select_labels(rest, device)
	mask, _ = ensure_mask_targets_shape(logits, labels, strict=False)

	if temporal_agg == "none":
		last = logits.float()  # [B,T,V]
		targets = clean_logits.float().argmax(dim=-1)  # [B,T]
		tgt_logits = last.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B,T]
		max_logits = last.max(dim=-1).values.clamp_min(1e-9)  # [B,T]
		ratio = tgt_logits / max_logits
		return masked_mean(ratio, mask)
	else:
		last_agg  = aggregate_logits_over_time(logits,       mask=mask, reduction=temporal_agg)  # [B,V]
		clean_agg = aggregate_logits_over_time(clean_logits, mask=mask, reduction=temporal_agg)  # [B,V]
		targets = clean_agg.argmax(dim=-1)  # [B]
		tgt_logits = last_agg.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B]
		max_logits = last_agg.max(dim=-1).values.clamp_min(1e-9)  # [B]
		ratio = tgt_logits / max_logits
		return ratio.mean()


def m_nl_ratio_span_both(logits, clean_logits, label, input_lengths, temporal_agg: str = "none", **__):
	# NL for the intervened run:
	nl_intervened = m_nl_ratio_span(
		logits, clean_logits, label, input_lengths, temporal_agg=temporal_agg
	)
	# NL for the clean model (baseline):
	nl_full = m_nl_ratio_span(
		clean_logits, clean_logits, label, input_lengths, temporal_agg=temporal_agg
	)
	return torch.stack([nl_intervened, nl_full], dim=-1)

def compute_faithfulness_F(
	model,
	graph_circuit,  # Graph with circuit 'in_graph' already set (non-circuit False)
	dataloader,  # your evaluation loader (has span mask + targets)
	intervention_dataloader,  # loader used by EAP to precompute means
	intervention="mean",  # or "mean-positional" if you precomputed per-position means
	mean_activations_cache_dir='./cache/',
	temporal_agg: str = "none",
):
	"""
	Returns a dict with NL_full (M), NL_empty (∅), NL_c (c), and F(c).
	Implementation detail:
	  - NL(M): clean forward pass (no intervention).
	  - NL(∅): evaluate_graph with an *empty* circuit graph (all non-circuit ablated ⇒ everything).
	  - NL(c): evaluate_graph with your circuit graph (ablating everything outside c).
	"""
	per_position = (intervention == "mean-positional")

	graph_for_means = graph_circuit.copy()
	graph_for_means.reset(empty=False)  # full graph to be safe

	if 'mean' in intervention:
		means = load_or_create_cache(
			os.path.join(mean_activations_cache_dir, f'mean_activations_{len(intervention_dataloader)}_{per_position}.pkl'),
			lambda: compute_mean_activations(
				model, graph_for_means, intervention_dataloader, per_position=per_position
			).unsqueeze(0)
		)
		if not per_position:
			means = means.unsqueeze(0)
	else:
		means = None

	metric_nl_both = partial(m_nl_ratio_span_both, temporal_agg=temporal_agg)

	# NL(c) — ablate everything except the circuit
	# NL(M) — no ablation
	NL_c_and_full, (NL_c_edges_to_corrupt, NL_c_edges_to_keep) = evaluate_graph(
		model=model,
		graph=graph_circuit, #.copy(),
		dataloader=dataloader,
		metrics=[metric_nl_both],
		skip_clean=False,
		# quiet=False,
		intervention=intervention,
		# intervention_dataloader=intervention_dataloader,
		precomputed_means=means,
	)
	# print(NL_c_and_full)
	assert NL_c_edges_to_keep > 0
	NL_c    = NL_c_and_full[..., 0]
	NL_full = NL_c_and_full[..., 1]
	
	# NL(∅) — ablate everything (i.e., empty circuit)
	graph_empty = graph_circuit.copy()
	graph_empty.reset(empty=True) # If empty is true, removes everything from graph; otherwise adds everything. Defaults to True.
	NL_empty_and_full, (_, NL_empty_edges_to_keep) = evaluate_graph(
		model=model,
		graph=graph_empty,
		dataloader=dataloader,
		metrics=[metric_nl_both],
		skip_clean=False,
		# quiet=False,
		intervention=intervention,
		# intervention_dataloader=intervention_dataloader,
		precomputed_means=means,
	)
	assert NL_empty_edges_to_keep == 0
	NL_empty = NL_empty_and_full[..., 0]

	# Ensure shapes match
	if not (NL_full.shape == NL_empty.shape == NL_c.shape):
		raise ValueError(
			f"Mismatched lengths: NL_full {NL_full.shape}, NL_empty {NL_empty.shape}, NL_c {NL_c.shape}"
		)

	# 4) Faithfulness in [0,1] — element-wise, safe when denom ≈ 0
	denom = NL_full - NL_empty
	with np.errstate(divide="ignore", invalid="ignore"):
		F = np.where(np.abs(denom) < 1e-12, 0.0, (NL_c - NL_empty) / denom)
		F = np.clip(F, 0.0, 1.0)

	return {
		"circuit_edges": {
			"corrupted_edges": NL_c_edges_to_corrupt,
			"circuit_edges": NL_c_edges_to_keep,
			"corrupted_edges_ppt": 100 * (NL_c_edges_to_corrupt / (NL_c_edges_to_corrupt + NL_c_edges_to_keep)),
		},
		"NL_full": {  # ≈ 1.0 on correctly completed prompts
			"mean": float(np.mean(NL_full)),
			"std": float(np.std(NL_full)),
			"data": NL_full.tolist(),
		},
		# "NL_full_test": {  # ≈ 1.0 on correctly completed prompts
		#   "mean": float(np.mean(NL_full_test)),
		#   "std": float(np.std(NL_full_test)),
		#   "data": NL_full_test.tolist(),
		# },
		"NL_empty": {
			"mean": float(np.mean(NL_empty)),
			"std": float(np.std(NL_empty)),
			"data": NL_empty.tolist(),
		},
		"NL_c": {
			"mean": float(np.mean(NL_c)),
			"std": float(np.std(NL_c)),
			"data": NL_c.tolist(),
		},
		"faithfulness": {
			"mean": float(np.mean(F)),
			"std": float(np.std(F)),
			"data": F.tolist(),
		},
	}

