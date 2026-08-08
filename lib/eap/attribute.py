from typing import Callable, List, Optional, Literal, Tuple
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer

from tqdm import tqdm

from .utils import tokenize_plus, tokenize_pairs_same_width, make_hooks_and_matrices, compute_mean_activations, model_device_expr, infer_decode_pos_mask, apply_decode_mask_inplace
from .evaluate import evaluate_graph, evaluate_baseline, backprop_no_param_grads
from .graph import Graph


# ------------------------- Decode-only position masking -------------------------
def _as_tensor(x, *, device):
	if torch.is_tensor(x):
		return x.to(device=device)
	return torch.tensor(x, device=device)

def _extract_from_mapping(m, keys):
	if not isinstance(m, dict):
		return None
	for k in keys:
		if k in m and m[k] is not None:
			return m[k]
	return None


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
	baseline = evaluate_baseline(model, dataloader, metric).mean().item()
	edges = graph.edges.values() if quiet else tqdm(graph.edges.values(), desc="exact", leave=False)
	for edge in edges:
		edge.in_graph = False
		intervened_performance = evaluate_graph(model, graph, dataloader, metric, intervention=intervention, intervention_dataloader=intervention_dataloader, 
												quiet=True, skip_clean=True).mean().item()
		edge.score = intervened_performance - baseline
		edge.in_graph = True

	# This is just to make the return type the same as all of the others; we've actually already updated the score matrix
	return graph.scores

def get_scores_eap(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor], 
				   intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
				   intervention_dataloader: Optional[DataLoader]=None, quiet=False, decode_only: bool = False, decode_mode: str = "after_prompt", disable_kv_cache = None):
	"""Gets edge attribution scores using EAP.

	Args:
		model (HookedTransformer): The model to attribute
		graph (Graph): Graph to attribute
		dataloader (DataLoader): The data over which to attribute
		metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
		quiet (bool, optional): suppress tqdm output. Defaults to False.

	Returns:
		Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
	"""
	scores = torch.zeros((graph.n_forward, graph.n_backward), device=model_device_expr(model), dtype=model.cfg.dtype)    

	if 'mean' in intervention:
		assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
		per_position = 'positional' in intervention
		means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
		means = means.unsqueeze(0)
		if not per_position:
			means = means.unsqueeze(0)

	root_name = graph.nodes["input"].out_hook  # earliest activation you need grads for
	total_items = 0
	dataloader = dataloader if quiet else tqdm(dataloader, desc="EAP", leave=False)
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

		(fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

		with torch.inference_mode():
			if intervention == "patching":
				# We intervene by subtracting out clean and adding in corrupted activations
				with model.hooks(fwd_hooks_corrupted):
					_ = model(corrupted_tokens, attention_mask=attention_mask_corrupted)

			elif "mean" in intervention:
				# means_batch = adapt_means_to_batch(means, activation_difference)
				# activation_difference.add_(means_batch)
				activation_difference.add_(means)

			# For some metrics (e.g. accuracy or KL), we need the clean logits
			clean_logits = model(clean_tokens, attention_mask=attention_mask_clean)

		# model.zero_grad(set_to_none=True)
		# with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
		# 	logits = model(clean_tokens, attention_mask=attention_mask_clean)
		# 	metric_value = metric(logits, clean_logits, label, input_lengths_clean)
		# 	metric_value.backward()
		# model.zero_grad(set_to_none=True)
		# clean_memory_cache(model)
		

		# [decode-only] restrict score-relevant diffs to decoding token positions
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
		scores /= total_items

	return scores

def get_scores_eap_ig(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], steps=30, quiet=False, decode_only: bool = False, decode_mode: str = "after_prompt", disable_kv_cache = None):
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
	# Accumulate in fp32 to avoid underflow after normalization
	acc_dtype = torch.float32
	scores = torch.zeros((graph.n_forward, graph.n_backward), device=model_device_expr(model), dtype=acc_dtype)    
	
	root_name = graph.nodes["input"].out_hook
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
		
		if attention_mask_clean.size(1) != attention_mask_corrupted.size(1):
			print(f"Number of positions must match, but do not: {attention_mask_clean.size(1)} (clean) != {attention_mask_corrupted.size(1)} (corrupted)")
			print(clean)
			print(corrupted)
			raise ValueError("Number of positions must match")

		# Here, we get our fwd / bwd hooks and the activation difference matrix
		# The forward corrupted hooks add the corrupted activations to the activation difference matrix
		# The forward clean hooks subtract the clean activations 
		# The backward hooks get the gradient, and use that, plus the activation difference, for the scores
		(fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = \
			make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

		with torch.inference_mode():
			with model.hooks(fwd_hooks=fwd_hooks_corrupted):
				_ = model(corrupted_tokens, attention_mask=attention_mask_corrupted)

			input_acts_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes["input"])].detach().clone()

			with model.hooks(fwd_hooks=fwd_hooks_clean):
				clean_logits = model(clean_tokens, attention_mask=attention_mask_clean)

			input_acts_clean = input_acts_corrupted - activation_difference[:, :, graph.forward_index(graph.nodes["input"])]

		# [decode-only] restrict score-relevant diffs to decoding positions
		if decode_only:
			apply_decode_mask_inplace(activation_difference, decode_pos_mask)

		def input_interpolation_hook(k: int):
			def hook_fn(activations: Tensor, hook):
				alpha = k / steps
				interp = input_acts_corrupted + alpha * (input_acts_clean - input_acts_corrupted)
				if decode_only and decode_pos_mask is not None:
					m = decode_pos_mask.to(device=activations.device, dtype=activations.dtype).unsqueeze(-1)
					return activations * (1 - m) + interp * m + activations * 0
				return interp + activations * 0
			return hook_fn

		for step in range(1, steps + 1):
			fwd_hooks = [(graph.nodes["input"].out_hook, input_interpolation_hook(step))]
			
			# model.zero_grad(set_to_none=True)
			# with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
			# 	logits = model(clean_tokens, attention_mask=attention_mask_clean)
			# 	metric_value = metric(logits, clean_logits, label, input_lengths_clean)
			# 	if torch.isnan(metric_value).any().item():
			# 		print("Metric value is NaN")
			# 		print(f"Clean: {clean}")
			# 		print(f"Corrupted: {corrupted}")
			# 		print(f"Label: {label}")
			# 		print(f"Metric: {metric}")
			# 		raise ValueError("Metric value is NaN")
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
			
			if torch.isnan(scores).any().item():
				print("Metric value is NaN")
				print(f"Clean: {clean}")
				print(f"Corrupted: {corrupted}")
				print(f"Label: {label}")
				print(f"Metric: {metric}")
				print(f'Step: {step}')
				raise ValueError("Metric value is NaN")

	if total_items > 0:
		scores /= total_items
	# Average IG path integral across interpolation steps (do NOT divide by number of batches)
	scores /= steps

	return scores

def get_scores_ig_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader, 
							  metric: Callable[[Tensor], Tensor], intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
							  steps=30, intervention_dataloader: Optional[DataLoader]=None, quiet=False, decode_only: bool = False, decode_mode: str = "after_prompt", disable_kv_cache = None):

	if 'mean' in intervention:
		assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
		per_position = 'positional' in intervention
		means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
		means = means.unsqueeze(0)
		if not per_position:
			means = means.unsqueeze(0)

	if steps is None or steps <= 0:
		raise ValueError(f"steps must be a positive int for EAP-IG-activations, got {steps}")
	acc_dtype = torch.float32
	scores = torch.zeros((graph.n_forward, graph.n_backward), device=model_device_expr(model), dtype=acc_dtype)    
	root_name = graph.nodes["input"].out_hook
	
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
		n_pos_clean = attention_mask_clean.size(1)
		n_pos_corrupted = attention_mask_corrupted.size(1)
		total_items += batch_size

		(_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos_clean, scores)
		(fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(model, graph, batch_size, n_pos_corrupted, scores)
		(fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(model, graph, batch_size, n_pos_clean, scores)

		with torch.inference_mode():
			if intervention == "patching":
				with model.hooks(fwd_hooks=fwd_hooks_corrupted):
					_ = model(corrupted_tokens, attention_mask=attention_mask_corrupted)
			elif intervention == "zero":
				pass
			elif "mean" in intervention:
				# means_batch = adapt_means_to_batch(means, activation_difference)
				# activation_difference.add_(means_batch)
				activation_difference.add_(means)
			else:
				raise ValueError(f"Unknown intervention: {intervention}")

			with model.hooks(fwd_hooks=fwd_hooks_clean):
				clean_logits = model(clean_tokens, attention_mask=attention_mask_clean)

			activation_difference.add_(activations_corrupted.detach()).sub_(activations_clean.detach())

		# [decode-only] restrict score-relevant diffs to decoding positions
		if decode_only:
			apply_decode_mask_inplace(activation_difference, decode_pos_mask)

		def output_interpolation_hook(k: int, clean: Tensor, corrupted: Tensor):
			def hook_fn(activations: Tensor, hook):
				alpha = k / steps
				new_output = alpha * clean + (1 - alpha) * corrupted
				if decode_only and decode_pos_mask is not None:
					m = decode_pos_mask.to(device=activations.device, dtype=activations.dtype).unsqueeze(-1)
					return activations * (1 - m) + new_output * m + activations * 0
				return new_output + activations * 0
			return hook_fn

		nodeslist = [graph.nodes["input"]]
		for layer in range(graph.cfg["n_layers"]):
			nodeslist.append(graph.nodes[f"a{layer}.h0"])
			nodeslist.append(graph.nodes[f"m{layer}"])

		for node in nodeslist:
			idx = graph.forward_index(node)
			clean_acts_all = activations_clean[:, :, idx]
			corrupted_acts_all = activations_corrupted[:, :, idx]

			for step in range(1, steps + 1):
				fwd_hooks = [(node.out_hook, output_interpolation_hook(step, clean_acts_all, corrupted_acts_all))]

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


def get_scores_clean_corrupted(model: HookedTransformer, graph: Graph, dataloader: DataLoader, 
							   metric: Callable[[Tensor], Tensor], quiet=False, decode_only: bool = False, decode_mode: str = "after_prompt", disable_kv_cache = None):
	"""Gets scores using the clean-corrupted method: like EAP-IG, but just do it on the clean and corrupted inputs, instead of all the intermediate steps.

	Args:
		model (HookedTransformer): the model to attribute
		graph (Graph): the graph to attribute
		dataloader (DataLoader): the data over which to attribute
		metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
		quiet (bool, optional): whether to silence tqdm. Defaults to False.

	Returns:
		_type_: _description_
	"""

	acc_dtype = torch.float32
	scores = torch.zeros((graph.n_forward, graph.n_backward), device=model_device_expr(model), dtype=acc_dtype)    
	
	root_name = graph.nodes["input"].out_hook
	total_items = 0
	dataloader = dataloader if quiet else tqdm(dataloader, desc="clean-corrupted", leave=False)
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

		(fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = \
			make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

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
		# 	clean_memory_cache(model)
		# 	model.zero_grad(set_to_none=True)
		# 	corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask_corrupted)
		# 	corrupted_metric_value = metric(corrupted_logits, clean_logits, label, input_lengths_corrupted)
		# 	corrupted_metric_value.backward()
		# 	model.zero_grad(set_to_none=True)
		# 	clean_memory_cache(model)

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

def get_scores_information_flow_routes(model: HookedTransformer, graph: Graph, dataloader: DataLoader, quiet=False) -> torch.Tensor:
	"""Gets scores using Ferrando et al.'s (2024) information flow routes method.

	Args:
		model (HookedTransformer): the model to attribute
		graph (Graph): the graph to attribute
		dataloader (DataLoader): the data over which to attribute
		metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
		quiet (bool, optional): whether to silence tqdm. Defaults to False.

	Returns:
		Tensor: scores based on information flow routes
	"""
	# I could do some hacky overriding of make_hooks_and_matrices here but I will not
	scores = torch.zeros((graph.n_forward, graph.n_backward), device=model_device_expr(model), dtype=model.cfg.dtype)    

	def make_hooks(n_pos: int, input_lengths: torch.Tensor) -> List[Tuple[str, Callable]]:
		output_activations = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device=model.cfg.device, dtype=model.cfg.dtype)

		def output_hook(index, activations, hook):
			try:
				acts = activations.detach()
				output_activations[:, :, index] = acts
			except RuntimeError as e:
				print(hook.name, output_activations[:, :, index].size(), output_activations.size())
				raise e

		# compute the score directly, without saving the input activations
		def input_hook(prev_index, bwd_index, input_lengths, activations, hook):
			acts = activations.detach()
			try:
				if acts.ndim == 3:
					acts = acts.unsqueeze(2)
				# acts : batch pos backward hidden
				# output acts: batch pos forward hidden
				# add forward and backwards dimensions to acts and output acts respectively
				acts = acts.unsqueeze(2)
				unsqueezed_output_activations = output_activations.unsqueeze(3)

				# acts : batch pos 1 backward hidden
				# output acts: batch pos forward 1 hidden
				proximity = torch.clamp(- torch.linalg.vector_norm(unsqueezed_output_activations[:, :, :prev_index] - acts, ord=1, dim=-1) + torch.linalg.vector_norm(acts, ord=1, dim=-1), min=0)
				importance = proximity / torch.sum(proximity, dim=2, keepdim=True)
				# importance: batch pos forward backward
				# aggregate over positions via sum/mean to get importance: forward backward
				# first mask out importances for padding positions
				max_len = input_lengths.max()
				mask = torch.arange(max_len, device=input_lengths.device,
							dtype=input_lengths.dtype).expand(len(input_lengths), max_len) < input_lengths.unsqueeze(1)
				mask = mask.unsqueeze(-1).unsqueeze(-1)
				# print(importance.size(), mask.size())
				importance *= mask
				importance = importance.sum(1) / input_lengths.view(-1,1,1) # mean over positions
				importance = importance.sum(0)

				# importance: forward backward
				# squeezing backward dim in case it isn't real (i.e. it's an MLP)
				importance = importance.squeeze(1)
				scores[:prev_index, bwd_index] += importance

			except RuntimeError as e:
				print(hook.name, unsqueezed_output_activations[:, :, prev_index].size(), acts.size())
				raise e
			
		hooks = []
		node = graph.nodes['input']
		fwd_index = graph.forward_index(node)
		hooks.append((node.out_hook, partial(output_hook, fwd_index)))
		
		for layer in range(graph.cfg['n_layers']):
			node = graph.nodes[f'a{layer}.h0']
			fwd_index = graph.forward_index(node)
			hooks.append((node.out_hook, partial(output_hook, fwd_index)))
			prev_index = graph.prev_index(node)
			for i, letter in enumerate('qkv'):
				bwd_index = graph.backward_index(node, qkv=letter)
				hooks.append((node.qkv_inputs[i], partial(input_hook, prev_index, bwd_index, input_lengths)))

			node = graph.nodes[f'm{layer}']
			fwd_index = graph.forward_index(node)
			bwd_index = graph.backward_index(node)
			prev_index = graph.prev_index(node)
			hooks.append((node.out_hook, partial(output_hook, fwd_index)))
			hooks.append((node.in_hook, partial(input_hook, prev_index, bwd_index, input_lengths)))
			
		node = graph.nodes['logits']
		prev_index = graph.prev_index(node)
		bwd_index = graph.backward_index(node)
		hooks.append((node.in_hook, partial(input_hook, prev_index, bwd_index, input_lengths)))
		return hooks
	
	total_items = 0
	dataloader = dataloader if quiet else tqdm(dataloader, desc="information-flow-routes", leave=False)
	for clean, _, _ in dataloader:
		batch_size = len(clean)
		total_items += batch_size
		clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)

		hooks = make_hooks(n_pos, input_lengths)
		with torch.inference_mode():
			with model.hooks(fwd_hooks=hooks):
				_ = model(clean_tokens, attention_mask=attention_mask)

	scores /= total_items

	return scores

allowed_aggregations = {'sum', 'mean'}    
def attribute(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
			  method: Literal['EAP', 'EAP-IG-inputs', 'clean-corrupted', 'EAP-IG-activations', 'information-flow-routes', 'exact'], 
			  intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', aggregation='sum', 
			  ig_steps: Optional[int]=None, intervention_dataloader: Optional[DataLoader]=None, quiet=False, decode_only: bool = False, decode_mode: str = "after_prompt", disable_kv_cache = None):
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
		scores = get_scores_eap(model, graph, dataloader, metric, intervention=intervention, 
								intervention_dataloader=intervention_dataloader, quiet=quiet,
								decode_only=decode_only, decode_mode=decode_mode, disable_kv_cache=disable_kv_cache)
	elif method == 'EAP-IG-inputs':
		if intervention != 'patching':
			raise ValueError(f"intervention must be 'patching' for EAP-IG-inputs, but got {intervention}")
		scores = get_scores_eap_ig(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet,
								decode_only=decode_only, decode_mode=decode_mode, disable_kv_cache=disable_kv_cache)
	elif method == 'clean-corrupted':
		if intervention != 'patching':
			raise ValueError(f"intervention must be 'patching' for clean-corrupted, but got {intervention}")
		scores = get_scores_clean_corrupted(model, graph, dataloader, metric, quiet=quiet,
								decode_only=decode_only, decode_mode=decode_mode, disable_kv_cache=disable_kv_cache)
	elif method == 'EAP-IG-activations':
		scores = get_scores_ig_activations(model, graph, dataloader, metric, steps=ig_steps, intervention=intervention, 
										   intervention_dataloader=intervention_dataloader, quiet=quiet,
											decode_only=decode_only, decode_mode=decode_mode, disable_kv_cache=disable_kv_cache)
	elif method == 'information-flow-routes':
		scores = get_scores_information_flow_routes(model, graph, dataloader, quiet=quiet,
								decode_only=decode_only, decode_mode=decode_mode)
	elif method == 'exact':
		scores = get_scores_exact(model, graph, dataloader, metric, intervention=intervention, intervention_dataloader=intervention_dataloader, 
								  quiet=quiet)
	else:
		raise ValueError(f"method must be in ['EAP', 'EAP-IG-inputs', 'clean-corrupted', 'EAP-IG-activations', 'information-flow-routes', 'exact'], but got {method}")


	if aggregation == 'mean':
		scores /= model.cfg.d_model
		
	graph.scores[:] =  scores.to(graph.scores.device)

	# graph.aggregate_edge_scores_to_nodes()

