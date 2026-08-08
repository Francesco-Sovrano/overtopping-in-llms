import numpy as np
import torch
from tqdm import tqdm
import random

from scipy.stats import beta as _beta_dist
from statistics import NormalDist
import math

from lib.modeling_and_ablation import build_ablation_hooks

def dichotomic_search_layer(
	model,
	layer_label,
	neuron_ids,
	pos_examples,
	neg_examples,
	is_answer_positive_fn,
	prompt_col,
	baseline_subset,
	search_epsilon=0.1,
	batch_size=8,
	decode_only=False,
	intervention='zero',
	mean_activations=None,
	prefix_batches=None,
	batch_ranges=None,
	max_new_tokens=10,
	# statistical safety knobs
	prune_alpha=0.05, # global failure prob budget (e.g. 0.01 => 99% overall guarantee)
	bonferroni=False, # distribute alpha across tested groups (global guarantee mode)
	# optional: signed importance scores for this layer (neuron_id -> float)
	importance_scores=None,
	first_split_by_importance_sign=False,
):
	"""
	Perform dichotomic (binary) search over neuron_ids within this layer.


	If first_split_by_importance_sign is True and importance_scores is provided, the
	*first* split (depth==0) partitions neurons by score sign (>=0 vs <0) instead of
	by index bisection. This preserves the invariant that the two depth-1 subtrees
	are disjoint sign pools, and then continues standard bisection within each pool.
	Early stop uses a high-confidence upper bound on per-slice effect magnitude:
		max_effect(G) = max(UCB(delta_assoc(G)), UCB(delta_unrel(G)))
	prune iff max_effect(G) < search_epsilon
	"""
	if not neuron_ids:
		return []
	max_internal = max(1, len(neuron_ids) - 1)

	print(
		f"Layer {layer_label}: starting dichotomic search over {len(neuron_ids)} neurons "
		f"(epsilon={search_epsilon:.4f}, prune_alpha={prune_alpha}, bonferroni={bonferroni})."
	)

	# Upper bound if we fully split to singletons: 2N - 1 group evaluations
	max_evals = max(1, 2 * len(neuron_ids) - 1)

	pbar = tqdm(
		total=max_evals,
		desc=f"Layer {layer_label} dichotomic",
		unit="group",
	)

	# ---- multiple-testing control ----
	# Option A (recommended): depth-based alpha spending (still a union-bound guarantee).
	# Option B (simpler): uniform Bonferroni across INTERNAL nodes only (no budget wasted on leaves).
	# Worst-case internal nodes in a full binary tree with N leaves is N-1.

	
	def recurse(group, depth=0, nodes_tested=0):
		group = list(group)
		if not group:
			return []

		# Depth-specific alpha (recorded per node)
		alpha_node = get_alpha_node(prune_alpha, depth, nodes_tested, bonferroni=bonferroni)
		alpha_slice = None if (alpha_node is None) else (alpha_node / 2.0)
		nodes_tested += 1

		# Evaluate this group
		record = ablate_neurons(
			model,
			pos_examples,
			neg_examples,
			is_answer_positive_fn,
			prompt_col,
			layers_neurons_dict={layer_label: group},
			batch_size=batch_size,
			decode_only=decode_only,
			intervention=intervention,
			mean_activations=mean_activations,
			prefix_batches=prefix_batches,
			batch_ranges=batch_ranges,
			max_new_tokens=max_new_tokens,
			baseline_subset=baseline_subset,
			alpha_slice=alpha_slice,
		)
		record["depth"] = depth
		record["alpha_node"] = alpha_node

		# alpha_slice + max_effect already computed in ablate_neurons
		max_eff_ucb = record.get("max_effect", None)

		print(
			f"Depth {depth}, layer {layer_label}, group_size={len(group)}: "
			f"acc_assoc={record['acc_after_knockout_on_associated']:.3f}, "
			f"acc_unrel={record['acc_after_knockout_on_unrelated']:.3f}, "
			f"gap={record['accuracy_gap']:.3f}, "
			f"max_effect={('NA' if max_eff_ucb is None else f'{max_eff_ucb:.3f}')}"
		)

		pbar.update(1)

		# Stop at singleton
		if len(group) <= 1:
			record["split"] = False
			record["stop_reason"] = "single_neuron"
			return [record]

		# If epsilon <= 0: always split
		if search_epsilon <= 0:
			record["split"] = True
		else:
			# SAFE pruning: only if we can certify both slice effects are < epsilon (via UCBs).
			if max_eff_ucb is not None and max_eff_ucb < search_epsilon:
				record["split"] = False
				record["stop_reason"] = "max_effect_below_epsilon"

				skipped = skip_subtree_for(len(group))
				if skipped:
					pbar.update(skipped)
				return [record]

			record["split"] = True

		# Recurse into halves
		# Decide how to split this group.
		# Default: standard 50/50 bisection by index.
		# Optional (depth==0): split by discovery score sign (>=0 vs <0) when requested.
		if (
			depth == 0
			and first_split_by_importance_sign
			and isinstance(importance_scores, dict)
		):
			pos_group = [n for n in group if float(importance_scores.get(int(n), 0.0)) >= 0.0]
			neg_group = [n for n in group if float(importance_scores.get(int(n), 0.0)) < 0.0]
			# Keep a bit of randomness (seeded via set_deterministic) to avoid brittle ordering effects
			# while still enforcing a clean sign partition at the root.
			random.shuffle(pos_group)
			random.shuffle(neg_group)
			print('Split by circuit discovery score sign (>=0 vs <0)')

			# If scores are missing / degenerate (all same sign), fall back to index bisection.
			if pos_group and neg_group:
				left, right = pos_group, neg_group
				record["split_strategy"] = "score_sign"
				record["split_left_size"] = len(left)
				record["split_right_size"] = len(right)
			else:
				record["split_strategy"] = "bisect_fallback"
				mid = len(group) // 2
				left = group[:mid]
				right = group[mid:]
		else:
			record["split_strategy"] = "bisect"
			mid = len(group) // 2
			left = group[:mid]
			right = group[mid:]
		results = [record]
		results += recurse(left, depth + 1, nodes_tested)
		results += recurse(right, depth + 1, nodes_tested)
		return results

	try:
		return recurse(neuron_ids)
	finally:
		pbar.close()

# ------------------------- Prompt-level accuracy (generation-based) -----------------------
def build_prefix_caches_for_examples(model, examples, prompt_col, max_new_tokens, batch_size):
	n = len(examples)
	prefix_batches = []
	batch_ranges = []  # [(start,end), ...]

	for start in range(0, n, batch_size):
		end = min(start + batch_size, n)
		batch_rows = examples[start:end]
		batch_prompts = [str(row[prompt_col]) for row in batch_rows]

		prefix = model.prefill_prefix_batch(
			batch_prompts,
			max_new_tokens=max_new_tokens,
			use_kv_cache=True,
		)
		prefix_batches.append(prefix)
		batch_ranges.append((start, end))

	return prefix_batches, batch_ranges

def decode_only_hooks(hooks):
	"""Only apply hooks on decode steps (sequence length == 1)."""
	out = []
	for name, fn in hooks:
		def wrapped(act, hook, fn=fn):
			# act: [B,T,H]
			if getattr(act, "ndim", None) == 3 and act.shape[1] != 1:
				return act
			return fn(act, hook)
		out.append((name, wrapped))
	return out

def get_correctness_cached_by_prefix_batches(model, examples, is_answer_positive_fn, prompt_col, prefix_batches, batch_ranges, hooks=None, return_answers=False):
	# Cached evaluation reuses prefix (prefill) computation across prompts.
	if hooks is not None:
		hooks = decode_only_hooks(hooks)

	n = len(examples)
	acc = np.zeros(n, dtype=float)
	answers_all = [None] * n if return_answers else None
	if n == 0:
		if return_answers:
			return 0.0, acc, []
		return 0.0, acc

	for prefix, (start, end) in zip(prefix_batches, batch_ranges):
		rows = examples[start:end]
		answers = model.generate_from_prefix_cache(
			prefix,
			fwd_hooks=hooks,
			clone_kv_cache_tensors=True
		)
		acc[start:end] = np.asarray(is_answer_positive_fn(rows, answers), dtype=float)
		if return_answers:
			answers_all[start:end] = [str(a) for a in answers]

	if return_answers:
		return float(acc.mean()), acc, answers_all
	return float(acc.mean()), acc

def get_correctness(model, examples, is_answer_positive_fn, prompt_col, max_new_tokens=10, hooks=None, batch_size=None, tqdm_desc=None, return_answers=False):
	"""
	Prompt-level accuracy computed from generated answers.

	This function assumes a batched scorer:
		is_answer_positive_fn(batch_rows: list[dict], answers: list[str]) -> array-like[bool|float]

	Flow:
	- For each batch of rows, read `prompt_col` as prompt text.
	- Generate answers with `model.generate(prompts, fwd_hooks=hooks)`.
	- Score correctness with `is_answer_positive_fn(batch_rows, answers)`. 

	Runs in mini-batches and optionally shows a tqdm bar.
	"""
	if batch_size is None:
		batch_size = 4

	n = len(examples)
	acc = np.zeros((n,), dtype=float)
	answers_all = [None] * n if return_answers else None
	if n == 0:
		if return_answers:
			return 0.0, acc, []
		return 0.0, acc

	# IMPORTANT: keep rows as list[dict] (default collate would turn it into dict-of-lists)
	dataloader = torch.utils.data.DataLoader(
		examples,
		batch_size=batch_size,
		shuffle=False,
		collate_fn=lambda batch: batch,
	)
	use_tqdm = tqdm_desc is not None
	if use_tqdm:
		dataloader = tqdm(dataloader, desc=tqdm_desc)

	for i, batch in enumerate(dataloader):
		start = i * batch_size

		batch_prompts = [str(ex[prompt_col]) for ex in batch]
		answers = model.generate(
			batch_prompts,
			max_new_tokens=max_new_tokens,
			do_sample=False,
			fwd_hooks=hooks,
			# use_kv_cache=False
		)

		acc[start : start + len(batch)] = is_answer_positive_fn(batch, answers)
		if return_answers:
			answers_all[start : start + len(batch)] = [str(a) for a in answers]

	if return_answers:
		return float(acc.mean()), acc, answers_all
	return float(acc.mean()), acc

def ablate_neurons(model, pos_examples, neg_examples, is_answer_positive_fn, prompt_col, layers_neurons_dict=None, batch_size=8, decode_only=False, intervention='zero', mean_activations=None, prefix_batches=None, batch_ranges=None, max_new_tokens=10, return_prefix_batches=False, baseline_subset="all", alpha_slice=None, return_answers=False):
	hooks = build_ablation_hooks(
		layers_neurons_dict,
		last_pos_only=decode_only,
		intervention=intervention,
		mean_activations=mean_activations,
	) if layers_neurons_dict else None

	# Merge pos+neg (your existing 2× win)
	pos_examples = list(pos_examples)
	neg_examples = list(neg_examples)
	n_pos = len(pos_examples)
	all_examples = pos_examples + neg_examples

	if decode_only and prefix_batches is None:
		prefix_batches, batch_ranges = build_prefix_caches_for_examples(
			model,
			all_examples,
			prompt_col,
			max_new_tokens=max_new_tokens,
			batch_size=batch_size
		)

	assert not decode_only or (prefix_batches is not None and batch_ranges is not None)

	if prefix_batches is not None and batch_ranges is not None:
		if return_answers:
			_, acc_all, answers_all = get_correctness_cached_by_prefix_batches(
				model,
				all_examples,
				is_answer_positive_fn,
				prompt_col,
				prefix_batches,
				batch_ranges,
				hooks=hooks,
				return_answers=True,
			)
		else:
			_, acc_all = get_correctness_cached_by_prefix_batches(
				model,
				all_examples,
				is_answer_positive_fn,
				prompt_col,
				prefix_batches,
				batch_ranges,
				hooks=hooks,
			)
	else:
		if return_answers:
			_, acc_all, answers_all = get_correctness(
				model,
				all_examples,
				is_answer_positive_fn,
				prompt_col,
				max_new_tokens=max_new_tokens,
				hooks=hooks,
				batch_size=batch_size,
				return_answers=True,
			)
		else:
			_, acc_all = get_correctness(
				model,
				all_examples,
				is_answer_positive_fn,
				prompt_col,
				max_new_tokens=max_new_tokens,
				hooks=hooks,
				batch_size=batch_size,
			)

	acc_pos_all = acc_all[:n_pos]
	acc_neg_all = acc_all[n_pos:]
	if return_answers:
		answers_pos_all = answers_all[:n_pos]
		answers_neg_all = answers_all[n_pos:]

	accuracy_on_associated = float(acc_pos_all.mean()) if n_pos else 0.0
	accuracy_on_unrelated = float(acc_neg_all.mean()) if len(acc_neg_all) else 0.0
	# We track a rule-specific degradation signal as a GAP:
	#   gap = acc(unrelated) - acc(associated).
	# Large positive gaps mean the ablation disproportionately harms associated prompts.
	gap = accuracy_on_unrelated - accuracy_on_associated
	results_dict = {
		"layers_neurons_dict": layers_neurons_dict,
		"group_size": sum(map(len, layers_neurons_dict.values())) if layers_neurons_dict else None,
		"accuracy_gap": float(gap),
		"acc_after_knockout_on_associated": float(accuracy_on_associated),
		"acc_after_knockout_on_unrelated": float(accuracy_on_unrelated),
		"acc_after_knockout_on_associated_all": acc_pos_all.tolist(),
		"acc_after_knockout_on_unrelated_all": acc_neg_all.tolist(),
	}
	if return_answers:
		results_dict["answers_after_knockout_on_associated_all"] = answers_pos_all
		results_dict["answers_after_knockout_on_unrelated_all"] = answers_neg_all

	# ---- compute per-slice effect estimates + UCBs (for safe pruning) ----
	delta_a_hat, delta_a_ucb, n_a, k_a = slice_effect_and_ucb(acc_pos_all, baseline_subset, alpha_slice)
	delta_u_hat, delta_u_ucb, n_u, k_u = slice_effect_and_ucb(acc_neg_all, baseline_subset, alpha_slice)

	results_dict["baseline_subset"] = baseline_subset
	results_dict["alpha_slice"] = alpha_slice

	results_dict["delta_on_associated_hat"] = delta_a_hat
	results_dict["delta_on_unrelated_hat"] = delta_u_hat
	results_dict["delta_on_associated_ucb"] = delta_a_ucb
	results_dict["delta_on_unrelated_ucb"] = delta_u_ucb
	results_dict["n_associated_eval"] = n_a
	results_dict["n_unrelated_eval"] = n_u
	results_dict["k_effect_associated"] = k_a
	results_dict["k_effect_unrelated"] = k_u

	if (delta_a_ucb is None) or (delta_u_ucb is None):
		results_dict["max_effect"] = None
	else:
		results_dict["max_effect"] = float(max(delta_a_ucb, delta_u_ucb))

	if return_prefix_batches and decode_only:
		results_dict['prefix_batches'] = prefix_batches
		results_dict['batch_ranges'] = batch_ranges

	return results_dict

def get_adjusted_search_epsilon(search_epsilon, pos_prompts, neg_prompts, n_associated, n_unrelated, prune_alpha):
	alpha_node = get_alpha_node(prune_alpha)
	alpha_slice = None if alpha_node is None else alpha_node / 2.0
	eps_a = equivalent_search_epsilon(
		len(pos_prompts),
		search_epsilon_ref=search_epsilon,
		n_ref=n_associated,
		prune_alpha=alpha_slice,
	)
	eps_u = equivalent_search_epsilon(
		len(neg_prompts),
		search_epsilon_ref=search_epsilon,
		n_ref=n_unrelated,
		prune_alpha=alpha_slice,
	)
	effective_eps = max(eps_a, eps_u)
	if effective_eps > search_epsilon:
		print(
			f"[AutoEps] Adjusted epsilon for sample sizes "
			f"(assoc={len(pos_prompts)}, unrel={len(neg_prompts)}): "
			f"{effective_eps:.4f} (base epsilon={search_epsilon:.4f} at "
			f"assoc={n_associated}, unrel={n_unrelated})."
		)
		search_epsilon = effective_eps
	return search_epsilon

# --- stats helpers (one-sided binomial UCB) ---
def binom_ucb(k: int, n: int, alpha: float) -> float:
	"""
	One-sided upper confidence bound for Binomial proportion p, using Clopper-Pearson when possible.
	Returns u such that P(p <= u) >= 1 - alpha under Binomial(n, p).

	alpha must be in (0,1). If n==0 returns 0.
	"""
	if n <= 0:
		return 0.0
	k = int(k)
	if k <= 0:
		# CP upper for k=0: 1 - alpha^(1/n)
		return float(1.0 - alpha ** (1.0 / n))
	if k >= n:
		return 1.0

	if _beta_dist is not None:
		# Clopper–Pearson (exact) one-sided upper bound:
		# u = Beta^{-1}(1-alpha; k+1, n-k)
		return float(_beta_dist.ppf(1.0 - alpha, k + 1, n - k))

	# Fallback: Wilson score upper bound (approx, no SciPy)
	phat = k / n
	z = NormalDist().inv_cdf(1.0 - alpha)
	denom = 1.0 + (z * z) / n
	center = (phat + (z * z) / (2.0 * n)) / denom
	half = (z * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))) / denom
	return float(min(1.0, center + half))

def skip_subtree_for(group_size):
	# full_subtree_nodes = 2*group_size - 1; skipped after counting current node = 2*group_size - 2
	return max(0, 2 * group_size - 2)

def slice_effect_and_ucb(correct_vec, baseline_subset, alpha_slice):
	"""
	correct_vec: list/array of booleans/0-1 ints where 1 means correct.
	Returns (delta_hat, delta_ucb, n, k_effect) where:
		- baseline_subset="positive": delta_hat = wrong_rate = (#wrong)/n
		- baseline_subset="negative": delta_hat = correct_rate = (#correct)/n

	NOTE: delta_hat/delta_ucb are *magnitudes* (always in [0,1]) by construction:
		- baseline_subset='positive': wrong-rate after ablation
		- baseline_subset='negative': correct-rate after ablation
	So pruning is based on effect *size*, not sign. The optional --sign_split_first
	uses the discovery score sign from the manifest, which is separate from this metric.
	"""
	n = len(correct_vec)
	if n == 0:
		return 0.0, 1.0, 0, 0   # NEVER prune from empty evidence

	# Convert to ints 0/1
	# (correct_vec elements are already bools/ints from your ablate_neurons)
	k_correct = int(sum(int(x) for x in correct_vec))

	if baseline_subset == "positive":
		k_effect = n - k_correct   # wrong count
	elif baseline_subset == "negative":
		k_effect = k_correct       # correct count (help)
	else:
		# baseline_subset="all": we don't have a constant baseline; disable pruning based on effect.
		return None, None, n, None

	delta_hat = k_effect / n

	if alpha_slice is None:
		delta_ucb = 1.0  # "never prune" when no alpha budget provided
	else:
		delta_ucb = binom_ucb(k_effect, n, alpha_slice)

	return float(delta_hat), float(delta_ucb), int(n), int(k_effect)

def invert_k_for_target_ucb(n_ref, target_ucb, prune_alpha):
	lo, hi = 0, n_ref
	best = 0
	while lo <= hi:
		mid = (lo + hi) // 2
		u = binom_ucb(mid, n_ref, prune_alpha)
		if u <= target_ucb:
			best = mid
			lo = mid + 1
		else:
			hi = mid - 1
	return best

def equivalent_search_epsilon(n, search_epsilon_ref = 0.2, n_ref = 100, prune_alpha = 0.025, rounding = "ceil"):
	k_ref = invert_k_for_target_ucb(n_ref, search_epsilon_ref, prune_alpha)
	p0 = k_ref / n_ref

	if rounding == "ceil":
		k = math.ceil(p0 * n)
	elif rounding == "floor":
		k = math.floor(p0 * n)
	else:
		k = int(round(p0 * n))

	k = max(0, min(n, k))
	return binom_ucb(k, n, prune_alpha)

def get_alpha_node(prune_alpha, depth=0, nodes_tested=0, bonferroni=False):
	if prune_alpha is None or prune_alpha <= 0:
		return None
	if not bonferroni:
		return float(prune_alpha)

	# Anytime alpha-spending (union bound) ; Use alpha spending instead of fixed Bonferroni over N−1
	c = 6.0 / (math.pi ** 2)  # ~0.6079
	return float(prune_alpha * c / (nodes_tested ** 2))
	# # fallback: uniform across internal nodes only
	# return float(prune_alpha / max_internal)
