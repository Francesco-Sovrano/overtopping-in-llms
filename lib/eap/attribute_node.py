from typing import Callable, Union, Optional, Literal, List, Any
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm
from einops import einsum

from .graph import Graph
from .utils import tokenize_pairs_same_width, compute_mean_activations, model_device_expr, infer_decode_pos_mask, apply_decode_mask_inplace
from .evaluate import evaluate_baseline, evaluate_graph, backprop_no_param_grads


# ------------------------- Decode-only position masking -------------------------
def _as_tensor(x, *, device):
	if torch.is_tensor(x):
		return x.to(device=device)
	return torch.tensor(x, device=device)

def _extract_from_mapping(m: Any, keys: List[str]):
	if not isinstance(m, dict):
		return None
	for k in keys:
		if k in m and m[k] is not None:
			return m[k]
	return None


def make_hooks_and_matrices(model: HookedTransformer, graph: Graph, batch_size:int , n_pos:int, scores: Optional[Tensor], neuron:bool=False):
	"""Makes a matrix, and hooks to fill it and the score matrix up

	Args:
		model (HookedTransformer): model to attribute
		graph (Graph): graph to attribute
		batch_size (int): size of the particular batch you're attributing
		n_pos (int): size of the position dimension
		scores (Tensor): The scores tensor you intend to fill. If you pass in None, we assume that you're using these hooks / matrices for evaluation only (so don't use the backwards hooks!)

	Returns:
		Tuple[Tuple[List, List, List], Tensor]: The final tensor ([batch, pos, n_src_nodes, d_model]) stores activation differences, 
		i.e. corrupted - clean activations. The first set of hooks will add in the activations they are run on (run these on corrupted input), 
		while the second set will subtract out the activations they are run on (run these on clean input). 
		The third set of hooks will compute the gradients and update the scores matrix that you passed in. 
	"""
	activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device=model_device_expr(model), dtype=model.cfg.dtype)

	fwd_hooks_clean = []
	fwd_hooks_corrupted = []
	bwd_hooks = []
		
	# Fills up the activation difference matrix. In the default case (not separate_activations), 
	# we add in the corrupted activations (add = True) and subtract out the clean ones (add=False)
	# In the separate_activations case, we just store them in two halves of the matrix. Less efficient, 
	# but necessary for models with Gemma's architecture.
	def activation_hook(index, activations:torch.Tensor, hook: HookPoint, add:bool=True):
		acts = activations.detach()
		try:
			if add:
				activation_difference[:, :, index] += acts
			else:
				activation_difference[:, :, index] -= acts

		except RuntimeError as e:
			print(hook.name, activation_difference[:, :, index].size(), acts.size())
			raise e
	
	def gradient_hook(fwd_index: Union[slice, int], bwd_index: Union[slice, int], gradients:torch.Tensor, hook: HookPoint):
		"""Takes in a gradient and uses it and activation_difference 
		to compute an update to the score matrix

		Args:
			fwd_index (Union[slice, int]): The forward index of the (src) node
			bwd_index (Union[slice, int]): The backward index of the (dst) node
			gradients (torch.Tensor): The gradients of this backward pass 
			hook (_type_): (unused)

		"""
		grads = gradients.detach().to(torch.float32)
		try:
			if neuron:
				s = einsum(activation_difference[:, :, fwd_index], grads,'batch pos ... hidden, batch pos ... hidden -> ... hidden')
			else:
				s = einsum(activation_difference[:, :, fwd_index], grads,'batch pos ... hidden, batch pos ... hidden -> ...')
			scores[fwd_index] += s
		except RuntimeError as e:
			print(hook.name, activation_difference.size(), activation_difference.device, grads.size(), grads.device)
			print(fwd_index, bwd_index, scores.size())
			raise e

	node = graph.nodes['input']
	fwd_index = graph.forward_index(node)
	fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
	fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
	bwd_hooks.append((node.out_hook, partial(gradient_hook, fwd_index, fwd_index)))
	
	for layer in range(graph.cfg['n_layers']):
		node = graph.nodes[f'a{layer}.h0']
		fwd_index = graph.forward_index(node) # default attn_slice=True → returns a slice over ALL heads
		fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
		fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
		bwd_hooks.append((node.out_hook, partial(gradient_hook, fwd_index, fwd_index)))

		node = graph.nodes[f'm{layer}']
		fwd_index = graph.forward_index(node)
		fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
		fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
		bwd_hooks.append((node.in_hook, partial(gradient_hook, fwd_index, fwd_index))) # in the evaluation path we construct the MLP at hook_mlp_in (that’s where evaluate_graph injects)

	return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference



def get_scores_exact(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor], 
					 intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
					 intervention_dataloader: Optional[DataLoader]=None, quiet=False):
	"""Gets scores via exact patching, by repeatedly calling evaluate graph.

	Args:
		model (HookedTransformer): the model to attribute
		graph (Graph): the graph to attribute
		dataloader (DataLoader): the data over which to attribute
		metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
		intervention (Literal[&#39;patching&#39;, &#39;zero&#39;, &#39;mean&#39;,&#39;mean, optional): the intervention to use. Defaults to 'patching'.
		intervention_dataloader (Optional[DataLoader], optional): the dataloader over which to take the mean. Defaults to None.
		quiet (bool, optional): _description_. Defaults to False.
	"""

	graph.in_graph |= graph.real_edge_mask  # All edges that are real are now in the graph
	graph.nodes_in_graph[:] = True
	baseline = evaluate_baseline(model, dataloader, metric).mean().item()
	nodes = graph.nodes.values() if quiet else tqdm(graph.nodes.values(), desc="exact", leave=False)
	for node in nodes:
		for edge in node.child_edges:
			edge.in_graph = False
		intervened_performance = evaluate_graph(model, graph, dataloader, metric, intervention=intervention, 
												intervention_dataloader=intervention_dataloader, quiet=True, skip_clean=True).mean().item()
		node.score = intervened_performance - baseline
		for edge in node.child_edges:
			edge.in_graph = True

	# This is just to make the return type the same as all of the others; we've actually already updated the score matrix
	return graph.nodes_scores

def get_scores_eap(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor], 
				intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
				intervention_dataloader: Optional[DataLoader]=None, quiet:bool=False, neuron:bool=False, decode_only: bool = False, decode_mode: str = "after_prompt", disable_kv_cache = None):
	"""Gets edge attribution scores using EAP.

	Args:
		model: The model to attribute.
		graph: Graph to attribute.
		dataloader: Data over which to attribute.
		metric: Callable taking (logits, clean_logits, label, input_lengths) -> scalar Tensor.
		intervention: 'patching' | 'zero' | 'mean' | 'mean-positional'.
		intervention_dataloader: Required for mean interventions.
		quiet: Suppress tqdm output.
		neuron: If True, return per-neuron scores [n_forward, d_model], else [n_forward].

	Returns:
		Scores tensor shaped [n_forward] or [n_forward, d_model] (if neuron=True).
	"""
	device = model_device_expr(model)
	dtype = getattr(model.cfg, "dtype", torch.float32)

	score_shape = (graph.n_forward, graph.cfg.d_model) if neuron else (graph.n_forward,)
	scores = torch.zeros(score_shape, device=device, dtype=dtype)

	means = None
	if intervention in ("mean", "mean-positional"):
		if intervention_dataloader is None:
			raise ValueError("intervention_dataloader must be provided for mean interventions")
		per_position = (intervention == "mean-positional")
		means = compute_mean_activations(
			model, graph, intervention_dataloader, per_position=per_position
		).to(device=device, dtype=dtype).detach()

		# Keep original broadcasting behavior:
		# - mean-positional: [1, pos, ...]
		# - mean:           [1, 1, ...]
		means = means.unsqueeze(0)          # batch dim
		if not per_position:
			means = means.unsqueeze(0)      # position dim

	total_items = 0
	iterator = dataloader if quiet else tqdm(dataloader, desc="EAP", leave=False)

	root_name = graph.nodes["input"].out_hook  # IMPORTANT: must be earliest hook you need grads for
	for clean, corrupted, label in iterator:
		(
			clean_tokens,
			attention_mask_clean,
			input_lengths_clean,
			corrupted_tokens,
			attention_mask_corrupted,
			_input_lengths_corrupted,
			_,
		) = tokenize_pairs_same_width(model, clean, corrupted)

		# [decode-only] infer decode/answer positions (mask over token positions)
		decode_pos_mask = None
		if decode_only:
			decode_pos_mask = infer_decode_pos_mask(
				attention_mask_clean, input_lengths_clean, label, mode=decode_mode
			)
			# treat an empty span as inference failure (otherwise we'd zero all attributions)
			if decode_pos_mask is not None and decode_pos_mask.sum().item() == 0:
				decode_pos_mask = None
			assert decode_pos_mask is not None
			if not quiet and total_items == 0:
				print("decode_pos_mask.sum()", int(decode_pos_mask.sum().item()))
				print("first true positions", decode_pos_mask.nonzero()[:10])
				print("answer_len", label["answer_len"] if isinstance(label, dict) and "answer_len" in label else None)
				print("prompt_len", label["prompt_len"] if isinstance(label, dict) and "prompt_len" in label else None)
				print("full_len", label["full_len"] if isinstance(label, dict) and "full_len" in label else None)

		batch_size = clean_tokens.size(0)
		total_items += batch_size
		n_pos = attention_mask_clean.size(1)

		(fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = (
			make_hooks_and_matrices(
				model, graph, batch_size, n_pos, scores, neuron=neuron
			)
		)

		if intervention == "patching":
			# Populate buffers from corrupted run without building a graph.
			with torch.inference_mode():
				with model.hooks(fwd_hooks=fwd_hooks_corrupted):
					_ = model(corrupted_tokens, attention_mask=attention_mask_corrupted)

		elif intervention == "zero":
			# Zero ablation: don't add corrupted activations; leave activation_difference as-is.
			pass

		elif intervention in ("mean", "mean-positional"):
			# means_batch = adapt_means_to_batch(means, activation_difference)
			# activation_difference.add_(means_batch)
			activation_difference.add_(means)

		else:
			raise ValueError(f"Unknown intervention: {intervention}")

		# Compute baseline clean logits without hooks/grad
		with torch.inference_mode():
			clean_logits = model(clean_tokens, attention_mask=attention_mask_clean)

		# # Backprop through the clean run with hooks.
		# model.zero_grad(set_to_none=True)
		# with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
		# 	logits = model(clean_tokens, attention_mask=attention_mask_clean)
		# 	metric_value = metric(logits, clean_logits, label, input_lengths_clean)
		# 	metric_value.backward()
		# # Drop references early to help peak memory.
		# del logits, metric_value, clean_logits
		# model.zero_grad(set_to_none=True)
		# clean_memory_cache(model)

		# [decode-only] restrict interventions/scores to decoding token positions
		if decode_only:
			apply_decode_mask_inplace(activation_difference, decode_pos_mask)

		backprop_no_param_grads(
			model,
			root_hook_name=root_name,
			tokens=clean_tokens,
			attention_mask=attention_mask_clean,
			fwd_hooks=fwd_hooks_clean,
			bwd_hooks=bwd_hooks,
			metric_from_logits=lambda logits: metric(logits, clean_logits, label, input_lengths_clean),
			disable_kv_cache=disable_kv_cache,
		)

	if total_items > 0:
		scores.div_(total_items)

	return scores

def get_scores_eap_ig(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
					  steps=30, quiet:bool=False, neuron:bool=False, decode_only: bool = False, decode_mode: str = "after_prompt", disable_kv_cache = None):
	"""Gets edge attribution scores using EAP with integrated gradients.

	Args:
		model (HookedTransformer): The model to attribute
		graph (Graph): Graph to attribute
		dataloader (DataLoader): The data over which to attribute
		metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
		steps (int, optional): number of IG steps. Defaults to 30.
		quiet (bool, optional): suppress tqdm output. Defaults to False.

	Returns:
		Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
	"""
	if steps is None or steps <= 0:
		raise ValueError(f"steps must be a positive int for EAP-IG, got {steps}")
	# Accumulate in fp32 to avoid tiny per-neuron attributions underflowing to 0 after normalization
	acc_dtype = torch.float32
	if neuron:
		scores = torch.zeros((graph.n_forward, graph.cfg.d_model), device=model_device_expr(model), dtype=acc_dtype)
	else:
		scores = torch.zeros((graph.n_forward), device=model_device_expr(model), dtype=acc_dtype)

	total_items = 0
	dataloader = dataloader if quiet else tqdm(dataloader, desc="EAP-IG", leave=False)
	for clean, corrupted, label in dataloader:
		batch_size = len(clean)
		(clean_tokens, attention_mask_clean, input_lengths_clean, corrupted_tokens, attention_mask_corrupted, input_lengths_corrupted, _) = tokenize_pairs_same_width(model, clean, corrupted)

		# [decode-only] infer decode/answer positions (mask over token positions)
		decode_pos_mask = None
		if decode_only:
			decode_pos_mask = infer_decode_pos_mask(
				attention_mask_clean, input_lengths_clean, label, mode=decode_mode
			)
			# treat an empty span as inference failure (otherwise we'd zero all attributions)
			if decode_pos_mask is not None and decode_pos_mask.sum().item() == 0:
				decode_pos_mask = None
			assert decode_pos_mask is not None
			if not quiet and total_items == 0:
				print("decode_pos_mask.sum()", int(decode_pos_mask.sum().item()))
				print("first true positions", decode_pos_mask.nonzero()[:10])
				print("answer_len", label["answer_len"] if isinstance(label, dict) and "answer_len" in label else None)
				print("prompt_len", label["prompt_len"] if isinstance(label, dict) and "prompt_len" in label else None)
				print("full_len", label["full_len"] if isinstance(label, dict) and "full_len" in label else None)
		n_pos = attention_mask_clean.size(1)
		total_items += batch_size

		# Here, we get our fwd / bwd hooks and the activation difference matrix
		# The forward corrupted hooks add the corrupted activations to the activation difference matrix
		# The forward clean hooks subtract the clean activations 
		# The backward hooks get the gradient, and use that, plus the activation difference, for the scores
		(fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)

		with torch.inference_mode():
			with model.hooks(fwd_hooks=fwd_hooks_corrupted):
				_ = model(corrupted_tokens, attention_mask=attention_mask_corrupted)

			input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()

			with model.hooks(fwd_hooks=fwd_hooks_clean):
				clean_logits = model(clean_tokens, attention_mask=attention_mask_clean)

			input_activations_clean = input_activations_corrupted - activation_difference[:, :, graph.forward_index(graph.nodes['input'])]

		# [decode-only] restrict score-relevant diffs to decoding positions (keep prompt fixed)
		if decode_only:
			apply_decode_mask_inplace(activation_difference, decode_pos_mask)

		# + activations * 0  will cause a backwards pass on new_input
		def input_interpolation_hook(k: int):
			def hook_fn(activations, hook):
				alpha = k / steps
				interp = input_activations_corrupted + alpha * (input_activations_clean - input_activations_corrupted)
				if decode_only and decode_pos_mask is not None:
					m = decode_pos_mask.to(device=activations.device, dtype=activations.dtype).unsqueeze(-1)
					# keep non-decode positions as the clean forward activations
					return activations * (1 - m) + interp * m + activations * 0
				return interp + activations * 0
			return hook_fn

		root_name = graph.nodes["input"].out_hook
		for step in range(1, steps + 1):
			fwd_hooks = [(graph.nodes["input"].out_hook, input_interpolation_hook(step))]
			
			# model.zero_grad(set_to_none=True)
			# with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
			# 	logits = model(clean_tokens, attention_mask=attention_mask_clean)
			# 	metric_value = metric(logits, clean_logits, label, input_lengths_clean)
			# 	metric_value.backward()
			# model.zero_grad(set_to_none=True)
			# clean_memory_cache(model)
			
			backprop_no_param_grads(
				model,
				root_hook_name=root_name,
				tokens=clean_tokens,
				attention_mask=attention_mask_clean,
				fwd_hooks=fwd_hooks,
				bwd_hooks=bwd_hooks,
				metric_from_logits=lambda logits: metric(logits, clean_logits, label, input_lengths_clean),
				disable_kv_cache=disable_kv_cache,
			)

	if total_items > 0:
		scores /= total_items
	# Average IG path integral across interpolation steps (do NOT divide by number of batches)
	scores /= steps

	return scores

def get_scores_ig_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
							  intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', steps=30, 
							  intervention_dataloader: Optional[DataLoader]=None, quiet:bool=False, neuron:bool=False, decode_only: bool = False, decode_mode: str = "after_prompt", disable_kv_cache = None):

	device = model_device_expr(model)
	dtype = getattr(model.cfg, "dtype", torch.float32)

	if intervention in ("mean", "mean-positional"):
		if intervention_dataloader is None:
			raise ValueError("intervention_dataloader must be provided for mean interventions")
		per_position = (intervention == "mean-positional")
		means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
		means = means.to(device=device, dtype=dtype).detach().unsqueeze(0)
		if not per_position:
			means = means.unsqueeze(0)

	root_name = graph.nodes["input"].out_hook

	if steps is None or steps <= 0:
		raise ValueError(f"steps must be a positive int for EAP-IG-activations, got {steps}")
	acc_dtype = torch.float32
	if neuron:
		scores = torch.zeros((graph.n_forward, graph.cfg.d_model), device=model_device_expr(model), dtype=acc_dtype)
	else:
		scores = torch.zeros((graph.n_forward), device=model_device_expr(model), dtype=acc_dtype)

	total_items = 0
	dataloader = dataloader if quiet else tqdm(dataloader, desc="EAP-IG-activations", leave=False)
	for clean, corrupted, label in dataloader:
		batch_size = len(clean)
		(clean_tokens, attention_mask_clean, input_lengths_clean, corrupted_tokens, attention_mask_corrupted, input_lengths_corrupted, _) = tokenize_pairs_same_width(model, clean, corrupted)

		# [decode-only] infer decode/answer positions (mask over token positions)
		decode_pos_mask = None
		if decode_only:
			decode_pos_mask = infer_decode_pos_mask(
				attention_mask_clean, input_lengths_clean, label, mode=decode_mode
			)
			# treat an empty span as inference failure (otherwise we'd zero all attributions)
			if decode_pos_mask is not None and decode_pos_mask.sum().item() == 0:
				decode_pos_mask = None
			assert decode_pos_mask is not None
			if not quiet and total_items == 0:
				print("decode_pos_mask.sum()", int(decode_pos_mask.sum().item()))
				print("first true positions", decode_pos_mask.nonzero()[:10])
				print("answer_len", label["answer_len"] if isinstance(label, dict) and "answer_len" in label else None)
				print("prompt_len", label["prompt_len"] if isinstance(label, dict) and "prompt_len" in label else None)
				print("full_len", label["full_len"] if isinstance(label, dict) and "full_len" in label else None)
		n_pos = attention_mask_clean.size(1)
		n_pos_corrupted = attention_mask_corrupted.size(1)
		total_items += batch_size

		(_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)
		(fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(model, graph, batch_size, n_pos_corrupted, scores, neuron=neuron)
		(fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)

		with torch.inference_mode():
			if intervention == "patching":
				with model.hooks(fwd_hooks=fwd_hooks_corrupted):
					_ = model(corrupted_tokens, attention_mask=attention_mask_corrupted)
			elif intervention in ("mean", "mean-positional"):
				# means_batch = adapt_means_to_batch(means, activation_difference)
				# activation_difference.add_(means_batch)
				activation_difference.add_(means)

			with model.hooks(fwd_hooks=fwd_hooks_clean):
				clean_logits = model(clean_tokens, attention_mask=attention_mask_clean)

			# avoid huge clones; you only need detached reads here
			activation_difference.add_(activations_corrupted.detach()).sub_(activations_clean.detach())

		# [decode-only] restrict score-relevant diffs to decoding positions
		if decode_only:
			apply_decode_mask_inplace(activation_difference, decode_pos_mask)

		def output_interpolation_hook(k: int, clean: torch.Tensor, corrupted: torch.Tensor):
			def hook_fn(activations: torch.Tensor, hook):
				alpha = k/steps
				new_output = alpha * clean + (1 - alpha) * corrupted
				if decode_only and decode_pos_mask is not None:
					m = decode_pos_mask.to(device=activations.device, dtype=activations.dtype).unsqueeze(-1)
					return activations * (1 - m) + new_output * m + activations * 0
				return new_output + activations * 0
			return hook_fn

		nodeslist = [graph.nodes['input']]
		for layer in range(graph.cfg['n_layers']):
			nodeslist.append(graph.nodes[f'a{layer}.h0'])
			nodeslist.append(graph.nodes[f'm{layer}'])

		for node in nodeslist:
			for step in range(1, steps + 1):

				clean_acts = activations_clean[:, :, graph.forward_index(node)]
				corrupted_acts = activations_corrupted[:, :, graph.forward_index(node)]
				fwd_hooks = [(node.out_hook, output_interpolation_hook(step, clean_acts, corrupted_acts))]

				# model.zero_grad(set_to_none=True)
				# with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
				# 	logits = model(clean_tokens, attention_mask=attention_mask_clean)
				# 	metric_value = metric(logits, clean_logits, label, input_lengths_clean)
				# 	metric_value.backward(retain_graph=True)
				# model.zero_grad(set_to_none=True)
				# clean_memory_cache(model)

				backprop_no_param_grads(
					model,
					root_hook_name=root_name,
					tokens=clean_tokens,
					attention_mask=attention_mask_clean,
					fwd_hooks=fwd_hooks,
					bwd_hooks=bwd_hooks,
					metric_from_logits=lambda logits: metric(logits, clean_logits, label, input_lengths_clean),
					disable_kv_cache=disable_kv_cache,
				)

	if total_items > 0:
		scores /= total_items
	# Average across interpolation steps (not across batches)
	scores /= steps

	return scores

def get_scores_clean_corrupted(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
							   quiet:bool=False, neuron:bool=False, decode_only: bool = False, decode_mode: str = "after_prompt", disable_kv_cache = None):
	"""Gets edge attribution scores using EAP with integrated gradients.

	Args:
		model (HookedTransformer): The model to attribute
		graph (Graph): Graph to attribute
		dataloader (DataLoader): The data over which to attribute
		metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
		steps (int, optional): number of IG steps. Defaults to 30.
		quiet (bool, optional): suppress tqdm output. Defaults to False.

	Returns:
		Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
	"""
	acc_dtype = torch.float32
	if neuron:
		scores = torch.zeros((graph.n_forward, graph.cfg.d_model), device=model_device_expr(model), dtype=acc_dtype)
	else:
		scores = torch.zeros((graph.n_forward), device=model_device_expr(model), dtype=acc_dtype)

	total_items = 0
	dataloader = dataloader if quiet else tqdm(dataloader, desc="clean-corrupted", leave=False)
	root_name = graph.nodes["input"].out_hook
	for clean, corrupted, label in dataloader:
		batch_size = len(clean)
		(clean_tokens, attention_mask_clean, input_lengths_clean, corrupted_tokens, attention_mask_corrupted, input_lengths_corrupted, _) = tokenize_pairs_same_width(model, clean, corrupted)

		# [decode-only] infer decode/answer positions (mask over token positions)
		decode_pos_mask = None
		if decode_only:
			decode_pos_mask = infer_decode_pos_mask(
				attention_mask_clean, input_lengths_clean, label, mode=decode_mode
			)
			# treat an empty span as inference failure (otherwise we'd zero all attributions)
			if decode_pos_mask is not None and decode_pos_mask.sum().item() == 0:
				decode_pos_mask = None
			assert decode_pos_mask is not None
			if not quiet and total_items == 0:
				print("decode_pos_mask.sum()", int(decode_pos_mask.sum().item()))
				print("first true positions", decode_pos_mask.nonzero()[:10])
				print("answer_len", label["answer_len"] if isinstance(label, dict) and "answer_len" in label else None)
				print("prompt_len", label["prompt_len"] if isinstance(label, dict) and "prompt_len" in label else None)
				print("full_len", label["full_len"] if isinstance(label, dict) and "full_len" in label else None)
		n_pos = attention_mask_clean.size(1)
		total_items += batch_size

		# Here, we get our fwd / bwd hooks and the activation difference matrix
		# The forward corrupted hooks add the corrupted activations to the activation difference matrix
		# The forward clean hooks subtract the clean activations 
		# The backward hooks get the gradient, and use that, plus the activation difference, for the scores
		(fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)

		with torch.inference_mode():
			with model.hooks(fwd_hooks=fwd_hooks_corrupted):
				_ = model(corrupted_tokens, attention_mask=attention_mask_corrupted)

			with model.hooks(fwd_hooks=fwd_hooks_clean):
				clean_logits = model(clean_tokens, attention_mask=attention_mask_clean)

		# [decode-only] restrict score-relevant diffs to decoding positions
		if decode_only:
			apply_decode_mask_inplace(activation_difference, decode_pos_mask)


		# model.zero_grad(set_to_none=True)
		# with model.hooks(bwd_hooks=bwd_hooks):
		# 	logits = model(clean_tokens, attention_mask=attention_mask_clean)
		# 	metric_value = metric(logits, clean_logits, label, input_lengths_clean)
		# 	metric_value.backward()
		# 	model.zero_grad(set_to_none=True)
		# 	clean_memory_cache(model)
		# 	logits = model(corrupted_tokens, attention_mask=attention_mask_corrupted)
		# 	metric_value = metric(logits, clean_logits, label, input_lengths_corrupted)
		# 	metric_value.backward()
		# 	model.zero_grad(set_to_none=True)
		# 	clean_memory_cache(model)

		# clean pass
		backprop_no_param_grads(
			model,
			root_hook_name=root_name,
			tokens=clean_tokens,
			attention_mask=attention_mask_clean,
			fwd_hooks=[],
			bwd_hooks=bwd_hooks,
			metric_from_logits=lambda logits: metric(logits, clean_logits, label, input_lengths_clean),
			disable_kv_cache=disable_kv_cache,
		)
		# corrupted pass
		backprop_no_param_grads(
			model,
			root_hook_name=root_name,
			tokens=corrupted_tokens,
			attention_mask=attention_mask_corrupted,
			fwd_hooks=[],
			bwd_hooks=bwd_hooks,
			metric_from_logits=lambda logits: metric(logits, clean_logits, label, input_lengths_corrupted),
			disable_kv_cache=disable_kv_cache,
		)

	if total_items > 0:
		scores /= total_items
	# We do 2 passes (clean + corrupted); average over them (not over number of batches)
	scores /= 2

	return scores

allowed_aggregations = {'sum', 'mean'}      
def attribute_node(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
				   method: Literal['EAP', 'EAP-IG-inputs', 'EAP-IG-activations', 'exact'], 
				   intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
				   aggregation='sum', ig_steps: Optional[int]=None, intervention_dataloader: Optional[DataLoader]=None, 
				   quiet:bool=False, neuron:bool=False, decode_only: bool = False, decode_mode: str = "after_prompt", disable_kv_cache = None):
	assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
	assert model.cfg.use_split_qkv_input, "Model must be configured to use split qkv inputs (model.cfg.use_split_qkv_input)"
	assert model.cfg.use_hook_mlp_in, "Model must be configured to use hook MLP in (model.cfg.use_hook_mlp_in)"
	if model.cfg.n_key_value_heads is not None:
		assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"
	
	if aggregation not in allowed_aggregations:
		raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')
		
	# Scores are by default summed across the d_model dimension
	# This means that scores are a [n_src_nodes, n_dst_nodes] tensor
	if method == 'EAP':
		scores = get_scores_eap(
			model, graph, dataloader, metric,
			intervention=intervention,
			intervention_dataloader=intervention_dataloader,
			quiet=quiet, neuron=neuron,
			decode_only=decode_only, decode_mode=decode_mode, disable_kv_cache=disable_kv_cache,
		)
	elif method == 'EAP-IG-inputs':
		if intervention != 'patching':
			raise ValueError(f"intervention must be 'patching' for EAP-IG-inputs, but got {intervention}")
		scores = get_scores_eap_ig(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet, neuron=neuron,
							decode_only=decode_only, decode_mode=decode_mode, disable_kv_cache=disable_kv_cache)
	elif method == 'EAP-IG-activations':
		scores = get_scores_ig_activations(
			model, graph, dataloader, metric,
			steps=ig_steps,
			intervention=intervention,
			intervention_dataloader=intervention_dataloader,
			quiet=quiet, neuron=neuron,
			decode_only=decode_only, decode_mode=decode_mode, disable_kv_cache=disable_kv_cache,
		)
	elif method == 'exact':
		scores = get_scores_exact(model, graph, dataloader, metric, intervention=intervention, 
								  intervention_dataloader=intervention_dataloader, 
								  quiet=quiet)
	else:
		raise ValueError(f"integrated_gradients must be in ['EAP', 'EAP-IG-inputs', 'EAP-IG-activations'], but got {method}")


	if aggregation == 'mean' and not neuron:
		scores /= model.cfg.d_model
		
	if neuron:
		graph.neurons_scores[:] = scores.to(graph.scores.device)
	else:
		graph.nodes_scores[:] = scores.to(graph.scores.device)
