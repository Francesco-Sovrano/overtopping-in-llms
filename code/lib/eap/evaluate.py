from typing import Callable, List, Union, Literal, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm
from einops import einsum

from .utils import forward_no_cache_optional, tokenize_plus, tokenize_pairs_same_width, make_hooks_and_matrices, compute_mean_activations, model_device_expr, clean_memory_cache
from .graph import Graph, AttentionNode


def backprop_no_param_grads(model, *, root_hook_name, tokens, attention_mask, fwd_hooks, bwd_hooks, metric_from_logits, disable_kv_cache = None):
	"""
	Runs backward so bwd_hooks fire, but does NOT populate/accumulate parameter .grad.
	Gradients are accumulated only for the chosen root activation(s).
	"""
	roots = []

	def save_root(act, hook):
		roots.append(act)
		return act

	if isinstance(root_hook_name, (list, tuple)):
		extra_fwd = [(name, save_root) for name in root_hook_name]
	else:
		extra_fwd = [(root_hook_name, save_root)]

	with model.hooks(fwd_hooks=fwd_hooks + extra_fwd, bwd_hooks=bwd_hooks):
		logits = forward_no_cache_optional(model, tokens, attention_mask=attention_mask, disable_kv_cache=disable_kv_cache)
		metric_value = metric_from_logits(logits)
		metric_value = metric_value.mean() if metric_value.ndim != 0 else metric_value

		if not metric_value.requires_grad:
			raise RuntimeError(
				"Metric does not require grad. Make sure you're NOT in no_grad/inference_mode "
				"and that model parameters still have requires_grad=True."
			)
		if len(roots) == 0:
			raise RuntimeError("Root hook never fired; check root_hook_name.")

		# Key line: full backward engine runs, hooks fire, but params won't accumulate .grad
		torch.autograd.backward(metric_value, inputs=roots, retain_graph=False, create_graph=False)

	clean_memory_cache(model)

# def backprop_no_param_grads(model, tokens, attention_mask, fwd_hooks, bwd_hooks, metric_from_logits):
# 	model.zero_grad(set_to_none=True)
# 	with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
# 		logits = model(tokens, attention_mask=attention_mask)
# 		metric_value = metric_from_logits(logits)
# 		metric_value.backward()
# 	model.zero_grad(set_to_none=True)
# 	clean_memory_cache(model)

def evaluate_graph(model: HookedTransformer, graph: Graph, dataloader: DataLoader,
				   metrics: Union[Callable[[Tensor],Tensor], List[Callable[[Tensor], Tensor]]],
				   quiet=False, intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching',
				   intervention_dataloader: Optional[DataLoader]=None, skip_clean:bool=True, precomputed_means: Optional[torch.Tensor] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
	"""Evaluate a circuit (i.e. a graph where only some nodes are false, probably created by calling graph.apply_threshold). You probably want to prune
		beforehand to make sure your circuit is valid.
	"""
	assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
	if model.cfg.n_key_value_heads is not None:
		assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"

	assert intervention in ['patching', 'zero', 'mean', 'mean-positional'], f"Invalid intervention: {intervention}"

	means = None
	if 'mean' in intervention:
		assert intervention_dataloader is not None or precomputed_means is not None

		per_position = 'positional' in intervention
		if precomputed_means is None:
			means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
			means = means.unsqueeze(0)      # batch dim
			if not per_position:
				means = means.unsqueeze(0)  # pos dim
		else:
			means = precomputed_means
			means = means.to(device=model_device_expr(model), dtype=model.cfg.dtype)

	# Construct a matrix that indicates which edges are in the graph
	in_graph_matrix = graph.in_graph.to(device=model.cfg.device).bool()
	real_mask = graph.real_edge_mask.to(in_graph_matrix.device)

	# same thing but for neurons
	if graph.neurons_in_graph is not None:
		neuron_matrix   = graph.neurons_in_graph.to(device=model.cfg.device, dtype=torch.bool)  # [F, d_model]
	else:
		neuron_matrix = None

	keep = (in_graph_matrix & real_mask)
	to_corrupt = (~in_graph_matrix & real_mask)
	print('[dbg]',
		'neurons in circuit:', graph.count_included_neurons(),
		'nodes in circuit:', graph.nodes_in_graph.sum().item(),
		'edges in circuit:', keep.sum().item()
	)
	edges_to_keep = int(keep.sum().item())
	edges_to_corrupt = int(to_corrupt.sum().item())
	print('[dbg] circuit edges:', edges_to_keep)
	print('[dbg] edges to corrupt:', edges_to_corrupt, f'({100*(edges_to_corrupt/(edges_to_corrupt+edges_to_keep)):.2f}%)')

	# We take the opposite matrix, because we'll use it as a mask to specify
	# which edges we want to corrupt
	in_graph_matrix = to_corrupt.to(model.cfg.dtype)
	if neuron_matrix is not None:
		neuron_matrix = (~neuron_matrix).to(model.cfg.dtype)

	if model.cfg.use_normalization_before_and_after:
		attention_head_mask = torch.zeros((graph.n_forward, model.cfg.n_layers), device=model_device_expr(model), dtype=model.cfg.dtype)
		for node in graph.nodes.values():
			if isinstance(node, AttentionNode):
				attention_head_mask[graph.forward_index(node), node.layer] = 1

		non_attention_head_mask = 1 - attention_head_mask.any(-1).to(dtype=model.cfg.dtype)
		attention_biases = torch.stack([block.attn.b_O for block in model.blocks])

	missing = []
	for n in graph.nodes.values():
		for hook_name in ([n.in_hook] if isinstance(n.in_hook, str) else n.in_hook) + ([n.out_hook] if n.out_hook else []):
			if hook_name and hook_name not in model.hook_dict:
				missing.append(hook_name)
	for e in graph.edges.values():
		if e.hook and e.hook not in model.hook_dict:
			missing.append(e.hook)
	print('missing nodes:', sorted(set(missing)))

	def make_input_construction_hook(activation_matrix, in_graph_vector, neuron_matrix):
		def input_construction_hook(activations, hook):
			if model.cfg.use_normalization_before_and_after:
				activation_differences = activation_matrix[0] - activation_matrix[1]

				clean_attention_results = einsum(activation_matrix[1, :, :, :len(in_graph_vector)],
												 attention_head_mask[:len(in_graph_vector)],
												 'batch pos previous hidden, previous layer -> batch pos layer hidden')

				if neuron_matrix is not None:
					non_attention_update = einsum(activation_differences[:, :, :len(in_graph_vector)],
												  neuron_matrix[:len(in_graph_vector)],
												  in_graph_vector,
												  non_attention_head_mask[:len(in_graph_vector)],
												  'batch pos previous hidden, previous hidden, previous ..., previous -> batch pos ... hidden')
					corrupted_attention_difference = einsum(activation_differences[:, :, :len(in_graph_vector)],
															neuron_matrix[:len(in_graph_vector)],
															in_graph_vector,
															attention_head_mask[:len(in_graph_vector)],
															'batch pos previous hidden, previous hidden, previous ..., previous layer -> batch pos ... layer hidden')
				else:
					non_attention_update = einsum(activation_differences[:, :, :len(in_graph_vector)],
												  in_graph_vector,
												  non_attention_head_mask[:len(in_graph_vector)],
												  'batch pos previous hidden, previous ..., previous -> batch pos ... hidden')
					corrupted_attention_difference = einsum(activation_differences[:, :, :len(in_graph_vector)],
															in_graph_vector,
															attention_head_mask[:len(in_graph_vector)],
															'batch pos previous hidden, previous ..., previous layer -> batch pos ... layer hidden')

				if in_graph_vector.ndim == 2:
					corrupted_attention_results = clean_attention_results.unsqueeze(2) + corrupted_attention_difference
					clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
					corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0).unsqueeze(0)
				else:
					corrupted_attention_results = clean_attention_results + corrupted_attention_difference
					clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
					corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)

				update = non_attention_update
				valid_layers = attention_head_mask[:len(in_graph_vector)].any(0)
				for i, valid_layer in enumerate(valid_layers):
					if not valid_layer:
						continue
					if in_graph_vector.ndim == 2:
						update -= model.blocks[i].ln1_post(clean_attention_results[:, :, None, i])
						update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, :, i])
					else:
						update -= model.blocks[i].ln1_post(clean_attention_results[:, :, i])
						update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, i])

			else:
				activation_differences = activation_matrix
				if neuron_matrix is not None:
					update = einsum(activation_differences[:, :, :len(in_graph_vector)], neuron_matrix[:len(in_graph_vector)], in_graph_vector,
									'batch pos previous hidden, previous hidden, previous ... -> batch pos ... hidden')
				else:
					update = einsum(activation_differences[:, :, :len(in_graph_vector)], in_graph_vector,
									'batch pos previous hidden, previous ... -> batch pos ... hidden')
			activations += update
			return activations
		return input_construction_hook

	def make_input_construction_hooks(activation_differences, in_graph_matrix, neuron_matrix):
		in_graph_mask_cpu = in_graph_matrix.bool().cpu()

		hooks = []
		for layer in range(model.cfg.n_layers):
			node0 = graph.nodes[f'a{layer}.h0']
			prev_index = graph.prev_index(node0)
			for i, letter in enumerate('qkv'):
				bwd_index = graph.backward_index(node0, qkv=letter, attn_slice=True)
				if in_graph_mask_cpu[:prev_index, bwd_index].any():
					hooks.append(
						(node0.qkv_inputs[i],
						 make_input_construction_hook(activation_differences,
													  in_graph_matrix[:prev_index, bwd_index],
													  neuron_matrix))
					)

			mnode = graph.nodes[f'm{layer}']
			prev_index = graph.prev_index(mnode)
			bwd_index = graph.backward_index(mnode)
			if in_graph_mask_cpu[:prev_index, bwd_index].any():
				hooks.append(
					(mnode.in_hook,
					 make_input_construction_hook(activation_differences,
												  in_graph_matrix[:prev_index, bwd_index],
												  neuron_matrix))
				)

		lnode = graph.nodes['logits']
		prev_index = graph.prev_index(lnode)
		bwd_index = graph.backward_index(lnode)
		if in_graph_mask_cpu[:prev_index, bwd_index].any():
			hooks.append(
				(lnode.in_hook,
				 make_input_construction_hook(activation_differences,
											  in_graph_matrix[:prev_index, bwd_index],
											  neuron_matrix))
			)
		return hooks

	# convert metrics to list if it's not already
	if not isinstance(metrics, list):
		metrics = [metrics]
	results = [[] for _ in metrics]

	dataloader = dataloader if quiet else tqdm(dataloader)
	for clean, corrupted, label in dataloader:
		(clean_tokens, attention_mask_clean, input_lengths_clean, corrupted_tokens, attention_mask_corrupted, _, _) = tokenize_pairs_same_width(model, clean, corrupted)
		n_pos = attention_mask_clean.size(1)

		(fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference = make_hooks_and_matrices(model, graph, len(clean), n_pos, None)

		input_construction_hooks = make_input_construction_hooks(activation_difference, in_graph_matrix, neuron_matrix)
		with torch.inference_mode():
			if intervention == 'patching':
				# We intervene by subtracting out clean and adding in corrupted activations
				with model.hooks(fwd_hooks_corrupted):
					_ = model(corrupted_tokens, attention_mask=attention_mask_corrupted)  # don't keep logits
			else:
				# In the case of zero or mean ablation, we skip the adding in corrupted activations
				# but in mean ablations, we need to add the mean in
				if 'mean' in intervention:
					T_batch = activation_difference.size(1)
					if means.size(1) != 1:
						# per_position = True → match T_batch
						T_means = means.size(1)
						if T_means > T_batch:
							means = means[:, :T_batch, ...]
						elif T_means < T_batch:
							pad = torch.zeros(
								means.size(0),
								T_batch - T_means,
								means.size(2),
								means.size(3),
								device=means.device,
								dtype=means.dtype,
							)
							means = torch.cat([means, pad], dim=1)
					activation_difference += means

			# For some metrics (e.g. accuracy or KL), we need the clean logits
			clean_logits = None if skip_clean else model(clean_tokens, attention_mask=attention_mask_clean)

			with model.hooks(fwd_hooks_clean + input_construction_hooks):
				logits = model(clean_tokens, attention_mask=attention_mask_clean)

		for i, metric in enumerate(metrics):
			r = metric(logits, clean_logits, label, input_lengths_clean).detach()
			if len(r.size()) == 0:
				r = r.unsqueeze(0)
			results[i].append(r.cpu())   # keep GPU memory low, semantics unchanged
			# results[i].append((r, (clean, corrupted, label)))

		# drop big tensors early
		clean_memory_cache(model)

	# after the loop (restore original semantics: stack over batches)
	results = [torch.stack(rs, dim=0).detach().numpy() for rs in results]
	
	# unwrap the results if there's only one metric
	if len(results) == 1:
		results = results[0]
	return results, (edges_to_corrupt, edges_to_keep)


def evaluate_baseline(model: HookedTransformer, dataloader:DataLoader, metrics: List[Callable[[Tensor], Tensor]], 
					  run_corrupted=False, quiet=False) -> Union[torch.Tensor, List[torch.Tensor]]:
	"""Evaluates the model on the given dataloader, without any intervention. This is useful for computing the baseline performance of the model.

	Args:
		model (HookedTransformer): The model to evaluate
		dataloader (DataLoader): The dataset to evaluate on
		metrics (List[Callable[[Tensor], Tensor]]): The metrics to evaluate with respect to
		run_corrupted (bool, optional): Whether to evaluate on corrupted examples instead. Defaults to False.

	Returns:
		Union[torch.Tensor, List[torch.Tensor]]: A tensor (or list thereof) of performance scores; if a list, each list entry corresponds to a metric in the input list
	"""
	if not isinstance(metrics, list):
		metrics = [metrics]

	assert getattr(dataloader, "shuffle", False) is False, (
		"DataLoader must be deterministic: set shuffle=False."
	)
	if hasattr(dataloader, "generator") and dataloader.generator is not None:
		assert dataloader.generator.initial_seed() == dataloader.generator.initial_seed(), (
			"DataLoader generator must have a fixed seed for deterministic iteration."
		)
	
	results = [[] for _ in metrics]
	if not quiet:
		dataloader = tqdm(dataloader)
	for clean, corrupted, label in dataloader:

		data = clean if not run_corrupted else corrupted
		tokens, attention_mask, input_lengths, _ = tokenize_plus(model, data)

		with torch.inference_mode():
			logits = model(tokens, attention_mask=attention_mask)
			
		for i, metric in enumerate(metrics):
			r = metric(logits, logits, label, input_lengths).detach()
			if len(r.size()) == 0:
				r = r.unsqueeze(0)
			results[i].append(r)
			# results[i].append((r, (clean, corrupted, label)))

	# for rs in results:
	# 	rs.sort(key=lambda t: t[-1])
	results = [torch.stack(rs, dim=0).detach().cpu().numpy() for rs in results]
	# results = [torch.cat([r for r, _ in rs], dim=0).cpu() for rs in results]
	
	if len(results) == 1:
		results = results[0]
	return results
