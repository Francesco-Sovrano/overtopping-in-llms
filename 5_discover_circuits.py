import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# # Hide CUDA; use MPS fallback + aggressive memory release on macOS
# os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
# # os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# # Consider letting PyTorch pick threads for CPU work (helps ST on CPU if used)
# os.environ.pop("OMP_NUM_THREADS", None)

# os.environ["EAP_LAYER_CHUNK"] = "8"

import csv
import re
import json
import math
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any

from functools import partial
import numpy as np
import pandas as pd
import torch
import gc
import traceback

from scipy.spatial.distance import cdist
# from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

# ---------------------- EAP-IG & TransformerLens imports ----------------------
from lib.eap.graph import Graph
from lib.eap.attribute import attribute  # for edges
from lib.eap.attribute_node import attribute_node  # for neurons/nodes
from lib.eap.data import PairItem, PairDataset, pair_by_length
from lib.eap.metrics import compute_faithfulness_F, m_kl_span, m_nl_ratio_span, m_nl_ratio_span_both

# ---------------------- Local model loader ----------------------
from lib.modeling_and_ablation import (
	LMWrapper,
	get_device,
	sample_len_tolerant_pairs,
)
from lib.text_and_rules import (
	apply_rule_to_features,
	load_rules,
	find_rule_files,
)
from lib.caching_and_prompting import set_deterministic
from lib.feature_extraction_runner import resolve_task_spec
from lib.spectral_analysis import (
    add_spectral_cli_args,
    build_reps_and_embedding_from_args,
    kcenter_farthest_first,
    assign_min_size_nearest_to_centers,
    balance_positives_and_negatives
)
from lib.feature_representation import safe_features_fillna

# ------------------------------- CLI ------------------------------------

def parse_args():
	p = argparse.ArgumentParser(description="Discover circuits per rule via EAP-IG on Qwen/Qwen2-7B-Instruct.")

	# Inputs (directory of rule files + one scores file containing prompts & features)
	p.add_argument(
		"--rules_dir",
		type=str,
		required=True,
		help="Directory containing association_rules_{X}.csv files (one per target).",
	)
	p.add_argument(
		"--rules_glob",
		type=str,
		default="association_rules_*.csv",
		help="Glob to select rule files in --rules_dir (default: association_rules).",
	)
	p.add_argument(
		"--fake_targets",
		action="store_true",
		help=(
			"If set, replace each target column with a new random fake target (suffix: '_fake') "
			"that is re-sampled until its Pearson correlation with the true target is near zero."
		),
	)
	p.add_argument(
		"--features_scores_dir",
		type=str,
		required=True,
		help="CSV/Parquet with prompt text + feature columns (same as script 5).",
	)
	p.add_argument(
		"--max_answer_tokens",
		type=int,
		default=16,
		help="Max tokens to roll out per example when target_col is not provided.",
	)
	p.add_argument(
		"--stop_regex",
		type=str,
		default=r"\n{2,}|</s>|<\|endoftext\|>",
		help="Regex; if matched in the growing continuation, we stop early.",
	)

	p.add_argument("--intervention", type=str, default="zero")
	p.add_argument("--eval_intervention", type=str, default="mean-positional")
	p.add_argument("--absolute_value_attributions", action="store_true")
	p.add_argument("--mlp_neurons_only", action="store_true")

	# Model
	p.add_argument(
		"--ai_model",
		type=str,
		default="Qwen/Qwen2-7B-Instruct",
		help="HF model id; MUST match the model used in script 1.",
	)
	p.add_argument("--ai_model_cache_dir", default=None, type=str)
	p.add_argument(
			"--cache_dir", "-c",
			default="./cache",
			help="Cache directory for mean dataloader / intermediate artifacts.",
		)
	p.add_argument(
		"--seed",
		type=int,
		default=0,
	)
	p.add_argument(
		"--max_n_of_rules_to_analyze",
		type=int,
		default=-1,
		help="Default: all (i.e., -1); put a number greater than 0 to control for the maximum number of available rules for which to discover a circuit.",
	)

	p.add_argument(
		"--temporal_agg",
		type=str,
		default="mean",
		choices=["none", "max", "mean", "min"],
		help="If not 'none', first aggregate logits across time (span-restricted) before metrics.",
	)

	# Decode-only circuit discovery (match cached decoding semantics)
	p.add_argument(
		"--decode_only",
		action="store_true",
		help="If set, restrict attribution/interventions to decoding/answer token positions (prompt positions are treated as fixed).",
	)
	p.add_argument(
		"--decode_mode",
		type=str,
		default="after_prompt",
		choices=["auto", "span", "after_prompt", "all"],
		help="How to determine decoding positions: auto (prefer span masks), span (require explicit span/mask), after_prompt (require prompt_len), all (no restriction beyond padding).",
	)

	# EAP-IG config
	p.add_argument(
		"--method",
		type=str,
		default="EAP",
		choices=["EAP", "EAP-IG-inputs", "EAP-IG-activations", "clean-corrupted", "information-flow-routes", "exact"],
		help="Attribution method (EAP or EAP-IG variants).",
	)
	p.add_argument(
		"--max_ig_steps",
		type=int,
		default=3,
		help="# of IG interpolation steps (EAP-IG only).",
	)
	p.add_argument(
		"--circuit_size",
		type=int,
		default=20,
		help="Circuit size (take top-N units).",
	)
	p.add_argument(
		"--circuit_level",
		type=str,
		default="node",
		help="Circuit level: node, edge, neuron.",
	)

	p.add_argument(
		"--max_pairs_per_circuit",
		type=int,
		default=512,
		help="Limit the number of pos/neg pairs per rule.",
	)
	p.add_argument(
		"--pair_similarity_metric", 
		type=str, 
		choices=["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "jensenshannon", "kulczynski1", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"], 
		default="cosine"
	)
	p.add_argument(
		"--points_to_use_for_mean_ablation",
		type=int,
		default=512,
	)

	p.add_argument(
		"--batch_size",
		type=int,
		default=8,
		help="Batch size for scoring/evaluation.",
	)

	# Sampling strategy: random vs precomputed spectral plan
	p.add_argument(
		"--sampling_strategy",
		type=str,
		default="random",
		choices=["random", "plan"],
		help="How to select datapoints per rule: 'random' (current ANN+length matching) or 'plan' (use precomputed spectral sampling plan).",
	)
	p.add_argument(
		"--sampling_plan_path",
		type=str,
		default=None,
		help="Path to JSON sampling plan produced by script 4. Required when --sampling_strategy plan.",
	)

	# Spectral cluster-based circuit discovery (optional)
	p.add_argument(
		"--cluster_by_spectral",
		action="store_true",
		help="If set, ignore association rules and discover circuits per spectral cluster (using lib.spectral_analysis).",
	)
	p.add_argument(
		"--cluster_base_subset",
		type=str,
		choices=["positive", "negative", "all"],
		default="all",
		help=(
			"In --cluster_by_spectral mode: which subset to form clusters on, based on the primary "
			"target label (task.DEFAULT_TARGETS[0]). "
			"'positive' clusters rows where target==True; 'negative' clusters target==False; "
			"'all' clusters all rows. Default: positive."
		),
	)
	add_spectral_cli_args(p)
	p.add_argument(
		"--global_n_clusters",
		type=int,
		default=32,
		help="Number of spectral clusters to form when --cluster_by_spectral is set.",
	)


	# Output
	p.add_argument(
		"--output_data_dir",
		type=str,
		default="neural_circuit_discovery_results",
		help="Folder to save circuits and scores.",
	)

	# ── Task / domain config ───────────────────────────────────────────────
	g = p.add_argument_group("task")
	g.add_argument(
		"--task_module",
		default="lib.tasks.arithmetic_task",
		help="Python module path defining parse_prompt, SYSTEM_PROMPT, TOKENS_DICT_KEYS and SEED_FEATURES.",
	)

	return p.parse_args()

args = parse_args()

#############################################################################

def slugify(s):
	s = s.strip().lower()
	s = re.sub(r"[^a-z0-9]+", "-", s)
	s = re.sub(r"-{2,}", "-", s).strip("-")
	return s or "rule"


def ensure_dir(p):
	p.mkdir(parents=True, exist_ok=True)

def write_skip_marker(
	output_dir,
	*,
	reason: str,
	n_pos: Optional[int] = None,
	n_neg: Optional[int] = None,
	skip_type = None,
	overwrite: bool = True,
	**extra,
):
	"""Write a skipped.json marker (used for cheap resuming / preflight exits)."""
	ensure_dir(output_dir)
	skip_path = output_dir / "skipped.json"
	if (not overwrite) and skip_path.exists():
		return
	payload = {"reason": reason, "n_pos": n_pos, "n_neg": n_neg}
	if skip_type is not None:
		payload["skip_type"] = skip_type
	payload.update(extra)
	skip_path.write_text(json.dumps(payload, indent=4))



def cache_state(output_dir):
	"""
	Return cache state for a circuit directory.
	- "ok": eval.json + scores.json exist
	- "skipped": skipped.json exists
	- "missing": nothing usable on disk
	"""
	if (output_dir / "eval.json").exists() and (output_dir / "scores.json").exists():
		return "ok"
	if (output_dir / "skipped.json").exists():
		return "skipped"
	return "missing"


def load_cached_ok(output_dir, *, sampling_strategy = None):
	"""Load cached eval/scores for a previously discovered circuit."""
	cached_eval_path = output_dir / "eval.json"
	cached_scores_path = output_dir / "scores.json"
	with open(cached_eval_path, "r") as f:
		faithfulness_dict = json.load(f)
	with open(cached_scores_path, "r") as f:
		meta = json.load(f)

	# Approximate number of pairs from stored NL_full
	nl_full_data = faithfulness_dict.get("NL_full", {}).get("data", [])
	n_pairs = len(nl_full_data) if isinstance(nl_full_data, list) else 0

	status = "full_network_fallback" if meta.get("fallback_full_network") else "ok"
	result = {
		"status": status,
		"n_pairs": n_pairs,
		"scores_csv": str(cached_scores_path),
		"eval_json": str(cached_eval_path),
		"metadata_topn": meta,
		"sampling_strategy": sampling_strategy,
		"pairing_meta": None,  # not stored on disk in old runs
	}
	result.update(faithfulness_dict)
	return result


def load_cached_skipped(output_dir, *, default_reason: str = "skipped"):
	"""Load cached skip info for a circuit previously skipped."""
	cached_skip_path = output_dir / "skipped.json"
	with open(cached_skip_path, "r") as f:
		skip_info = json.load(f)

	skip_type = skip_info.get("skip_type")
	status = "skipped_linear" if skip_type == "linear" else "skipped"

	return {
		"status": status,
		"n_pos": skip_info.get("n_pos", 0),
		"n_neg": skip_info.get("n_neg", 0),
		"reason": skip_info.get("reason", default_reason),
	}


def _make_placeholder_faithfulness(graph, *, reason: str, error: Optional[str] = None):
	real_edges = int(graph.real_edge_mask.sum().item())
	payload = {
		"circuit_edges": {
			"corrupted_edges": 0,
			"circuit_edges": real_edges,
			"corrupted_edges_ppt": 0.0,
		},
		"NL_full": {"mean": None, "std": None, "data": []},
		"NL_empty": {"mean": None, "std": None, "data": []},
		"NL_c": {"mean": None, "std": None, "data": []},
		"faithfulness": {"mean": 1.0, "std": 0.0, "data": []},
		"fallback_eval": True,
		"fallback_reason": reason,
	}
	if error is not None:
		payload["fallback_eval_error"] = error
	return payload

def _serialize_full_network_metadata(graph, *, level: str, absolute: bool = True, include_special: bool = False):
	meta = {
		"level": level,
		"absolute": bool(absolute),
		"include_special": bool(include_special),
		"requested_n": None,
		"selected_n": 0,
		"num_scored": 0,
		"num_considered": 0,
		"total_entities": 0,
		"was_clipped": False,
		"score_range": None,
		"threshold": None,
		"empty_reason": None,
		"indices": [],
		"d_model": int(graph.cfg.get("d_model")) if graph.cfg.get("d_model") is not None else None,
	}

	if level == "edge":
		graph._ensure_edge_index_map()
		scores = graph.scores
		valid_mask = graph.real_edge_mask.to(dtype=torch.bool)
		flat_scores = scores.view(-1)
		valid_idxs = torch.nonzero(valid_mask.view(-1), as_tuple=False).view(-1)
		vals = flat_scores[valid_idxs]
		finite = torch.isfinite(vals)
		selected = valid_idxs[finite]
		selected_vals = vals[finite]

		meta["total_entities"] = int(scores.numel())
		meta["num_considered"] = int(valid_idxs.numel())
		meta["num_scored"] = int(selected.numel())
		meta["selected_n"] = int(selected.numel())
		meta["requested_n"] = int(selected.numel())
		meta["indices"] = [int(i) for i in selected.tolist()]
		if selected.numel() > 0:
			range_vals = selected_vals.abs() if absolute else selected_vals
			meta["score_range"] = (float(range_vals.min().item()), float(range_vals.max().item()))
		label_score = {}
		for flat_idx, val in zip(selected.tolist(), selected_vals.tolist()):
			i = flat_idx // graph.n_backward
			j = flat_idx % graph.n_backward
			name = graph._edge_index_to_name.get((int(i), int(j)))
			if name is not None:
				label_score[name] = float(val)
		meta["edge_label_score"] = label_score
		if not label_score:
			meta["empty_reason"] = "no real scored edges"
		return meta

	graph._forward_index_to_name()
		
	if level == "node":
		scores = graph.nodes_scores
		valid_mask = torch.zeros_like(scores, dtype=torch.bool)
		for i, name in enumerate(graph._fwd_idx_to_name):
			if name is None:
				continue
			if not include_special and name in ("input", "logits"):
				continue
			valid_mask[i] = True
		flat_scores = scores.view(-1)
		valid_idxs = torch.nonzero(valid_mask.view(-1), as_tuple=False).view(-1)
		vals = flat_scores[valid_idxs]
		finite = torch.isfinite(vals)
		selected = valid_idxs[finite]
		selected_vals = vals[finite]
		meta["total_entities"] = int(scores.numel())
		meta["num_considered"] = int(valid_idxs.numel())
		meta["num_scored"] = int(selected.numel())
		meta["selected_n"] = int(selected.numel())
		meta["requested_n"] = int(selected.numel())
		meta["indices"] = [int(i) for i in selected.tolist()]
		if selected.numel() > 0:
			range_vals = selected_vals.abs() if absolute else selected_vals
			meta["score_range"] = (float(range_vals.min().item()), float(range_vals.max().item()))
		label_score = {}
		for idx, val in zip(selected.tolist(), selected_vals.tolist()):
			name = graph._fwd_idx_to_name[int(idx)]
			if name is not None:
				label_score[name] = float(val)
		meta["node_label_score"] = label_score
		if not label_score:
			meta["empty_reason"] = "no scored nodes"
		return meta

	if level != "neuron":
		raise ValueError(f"Unsupported level for full-network serialization: {level!r}")

	if graph.neurons_scores is None:
		raise RuntimeError("neurons_scores is None; cannot serialize full-network neurons.")

	ns = graph.neurons_scores
	d_model = graph.cfg["d_model"]
	valid_mask = torch.ones_like(ns, dtype=torch.bool)
	if not include_special:
		for i, name in enumerate(graph._fwd_idx_to_name):
			if name in ("input", "logits"):
				valid_mask[i, :] = False
	flat_scores = ns.view(-1)
	valid_idxs = torch.nonzero(valid_mask.view(-1), as_tuple=False).view(-1)
	vals = flat_scores[valid_idxs]
	finite = torch.isfinite(vals)
	selected = valid_idxs[finite]
	selected_vals = vals[finite]
	meta["total_entities"] = int(ns.numel())
	meta["num_considered"] = int(valid_idxs.numel())
	meta["num_scored"] = int(selected.numel())
	meta["selected_n"] = int(selected.numel())
	meta["requested_n"] = int(selected.numel())
	meta["indices"] = [int(i) for i in selected.tolist()]
	if selected.numel() > 0:
		range_vals = selected_vals.abs() if absolute else selected_vals
		meta["score_range"] = (float(range_vals.min().item()), float(range_vals.max().item()))
	label_score = {}
	neurons = {}
	for flat_idx, val in zip(selected.tolist(), selected_vals.tolist()):
		i = flat_idx // d_model
		h = flat_idx % d_model
		node_name = graph._fwd_idx_to_name[int(i)]
		if node_name is None:
			continue
		ident = str((node_name, int(h)))
		label_score[ident] = float(val)
		neurons.setdefault(node_name, []).append(int(h))
	meta["neuron_label_score"] = label_score
	meta["neurons"] = neurons
	if not label_score:
		meta["empty_reason"] = "no scored neurons"
	return meta

def _make_full_network_result(*, graph, output_dir, loader, global_mean_loader, model, eval_intervention, sampling_strategy, pair_meta, extra_results, debug_label, reason):
	graph.reset(empty=False)
	if graph.nodes_scores is not None:
		graph.nodes_scores[:] = 1.0
	if graph.neurons_scores is not None:
		graph.neurons_scores[:] = 1.0
	graph.scores = graph.real_edge_mask.to(dtype=torch.float32)

	# if args.mlp_neurons_only and args.circuit_level == "neuron":
	# 	graph.zero_out_attention_neuron_scores()

	meta = _serialize_full_network_metadata(
		graph,
		level=args.circuit_level,
		absolute=args.absolute_value_attributions,
		include_special=False,
	)
	meta["fallback_full_network"] = True
	meta["fallback_reason"] = reason
	
	if debug_label:
		print(f"[Fallback] Returning full network for {debug_label}: {reason}")
	else:
		print(f"[Fallback] Returning full network: {reason}")

	scores_csv = output_dir / "scores.json"
	with open(scores_csv, "w") as f:
		json.dump(meta, f, indent=4)

	faithfulness_dict = _make_placeholder_faithfulness(graph, reason=reason)

	(output_dir / "eval.json").write_text(json.dumps(faithfulness_dict, indent=4))

	results_dict = {
		"status": "full_network_fallback",
		"n_pairs": len(loader.dataset),
		"scores_csv": str(scores_csv),
		"eval_json": str(output_dir / "eval.json"),
		"metadata_topn": meta,
		"sampling_strategy": sampling_strategy,
		"pairing_meta": pair_meta,
		"fallback_full_network": True,
		"fallback_reason": reason,
	}
	if extra_results:
		results_dict.update(extra_results)
	results_dict.update(faithfulness_dict)
	return results_dict



def _run_circuit_discovery_from_pairs(
	*,
	pairs,
	model,
	tokenizer,
	output_dir,
	global_mean_loader,
	method="EAP",
	max_ig_steps=5,
	topn=20,
	level="node",
	intervention='zero',
	eval_intervention="mean-positional",
	prompts_to_answers_dict=None,
	sampling_strategy=None,
	pair_meta=None,
	extra_results=None,   # e.g. {"cluster_index": ...}
	debug_label=None,     # e.g. "rule 3" or "spectral cluster 7"
):
	"""
	Shared pipeline used by discover_for_rule and discover_for_cluster.

	Assumes `pairs` is a list of PairItem(clean=..., corrupted=...).
	"""

	# Build dataset
	dev = next(model.parameters()).device
	dataset = PairDataset(
		pairs,
		tokenizer=tokenizer,
		prompts_to_answers_dict=prompts_to_answers_dict,
		precompute=True,
	)

	# Build computational graph
	graph = Graph.from_model(
		model,
		neuron_level=args.circuit_level == "neuron",
		node_scores=args.circuit_level == "node",
	)

	is_cuda = str(dev) == "cuda"
	loader = torch.utils.data.DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=2 if is_cuda else 0,
		pin_memory=is_cuda,
		prefetch_factor=2 if is_cuda else None,
		persistent_workers=is_cuda,
		drop_last=False,
	)

	metric_kl = partial(m_kl_span, temporal_agg=args.temporal_agg)
	# Run attribution
	try:
		if level == "edge":
			scores = attribute(
				model=model,
				graph=graph,
				dataloader=loader,
				metric=metric_kl,
				method=method,  # 'EAP-IG-inputs', 'EAP-IG-activations', or 'EAP'
				intervention='patching' if 'EAP-IG' in method else intervention,
				intervention_dataloader=global_mean_loader if 'mean' in intervention else None,
				aggregation="sum",
				ig_steps=max_ig_steps,  # only used by EAP-IG variants
				quiet=False,
				decode_only=args.decode_only,
				decode_mode=args.decode_mode,
			)
		else:
			scores = attribute_node(
				model=model,
				graph=graph,
				dataloader=loader,
				metric=metric_kl,
				method=method,  # 'EAP-IG-inputs', 'EAP-IG-activations', or 'EAP'
				intervention='patching' if 'EAP-IG' in method else intervention,
				intervention_dataloader=global_mean_loader if 'mean' in intervention else None,
				aggregation="sum",
				ig_steps=max_ig_steps,  # only used by EAP-IG variants
				neuron=level == "neuron",
				quiet=False,
				decode_only=args.decode_only,
				decode_mode=args.decode_mode,
			)
	except Exception as exc:
		reason = f"EAP failed: {type(exc).__name__}: {exc}"
		return _make_full_network_result(
			graph=graph,
			output_dir=output_dir,
			loader=loader,
			global_mean_loader=global_mean_loader,
			model=model,
			eval_intervention=eval_intervention,
			sampling_strategy=sampling_strategy,
			pair_meta=pair_meta,
			extra_results=extra_results,
			debug_label=debug_label,
			reason=reason,
		)

	# 1) Which forward rows have any signal?
	if level == "edge":
		rows_with_signal = graph.scores.abs().sum(dim=1) > 0
	elif level == "node":
		rows_with_signal = torch.nan_to_num(graph.nodes_scores, nan=0.0).abs() > 0
	else:  # neuron
		rows_with_signal = torch.nan_to_num(graph.neurons_scores, nan=0.0).abs().sum(dim=1) > 0
	print("rows_with_signal:", int(rows_with_signal.sum()))

	if args.mlp_neurons_only:
		if level == "neuron":
			graph.zero_out_attention_neuron_scores()
		elif level == "node":
			graph.zero_out_attention_node_scores()
		elif level == "edge":
			graph.zero_out_attention_to_attention_edge_scores()

	# 2) Node-score coverage / top-n circuit
	graph.apply_topn(
		topn,  # number of edges/nodes/neurons to take
		absolute=args.absolute_value_attributions,
		level=level,
		reset=True,
		prune=False,
	)

	scores_tensor = (
		graph.scores
		if level == "edge"
		else (graph.nodes_scores if level == "node" else graph.neurons_scores)
	)

	print("n_scored_nodes:", int((~torch.isnan(scores_tensor)).sum()))

	top_scores, meta = graph.get_topn(
		topn,
		level=level,
		absolute=args.absolute_value_attributions,
		include_special=False,
		return_scores=True,
		return_metadata=True,
	)

	if debug_label:
		print(f"Top-n neurons for {debug_label}:", json.dumps(meta, indent=4))
	else:
		print("Top-n neurons:", json.dumps(meta, indent=4))

	# Persist raw scores
	scores_csv = output_dir / "scores.json"
	with open(scores_csv, "w") as f:
		json.dump(meta, f, indent=4)
	print(f"Saved {scores_csv}")

	# Faithfulness evaluation
	faithfulness_dict = compute_faithfulness_F(
		model,
		graph,
		loader,  # evaluation loader
		global_mean_loader,
		intervention=eval_intervention,
		mean_activations_cache_dir=args.cache_dir,
		temporal_agg=args.temporal_agg,
	)

	# Save eval
	(output_dir / "eval.json").write_text(json.dumps(faithfulness_dict, indent=4))

	# Final results dict
	results_dict = {
		"status": "ok",
		"n_pairs": len(dataset),
		"scores_csv": str(scores_csv),
		"eval_json": str(output_dir / "eval.json"),
		"metadata_topn": meta,
		"sampling_strategy": sampling_strategy,
		"pairing_meta": pair_meta,
	}
	if extra_results:
		results_dict.update(extra_results)
	results_dict.update(faithfulness_dict)

	# Cleanup
	del dataset, loader, graph, scores_tensor
	gc.collect()
	if torch.backends.mps.is_available():
		torch.mps.synchronize()
		torch.mps.empty_cache()

	return results_dict

def discover_for_rule(
	rule_row,
	features,
	text_col,
	model,
	tokenizer,
	output_data_dir_rule,
	emb_all,
	global_mean_loader,
	method="EAP-IG-inputs",
	max_ig_steps=5,
	topn=20,
	level="node",
	max_pairs=512,
	pair_similarity_metric="cosine",
	intervention="zero",
	eval_intervention="mean-positional",
	prompts_to_answers_dict=None,
	sampling_strategy="random",
	sampling_plan_index=None,
	target_name=None,
):
	"""
	One rule -> prepare pairs -> run EAP-IG -> save outputs.
	"""

	ensure_dir(output_data_dir_rule)

	rule = str(rule_row["rule"])
	circuit_id = int(rule_row["rule_id"])
	rule_direction = rule_row["coefficient_sign"]
	print(f"Discovering neural circuit for rule {circuit_id}: '{rule}'")

	# --- build initial mask-based split ---
	mask = apply_rule_to_features(rule, features, direction=rule_direction)
	texts = features[text_col].astype(str).to_numpy()

	is_using_predefined_plan = sampling_strategy == "plan" and sampling_plan_index is not None
	if is_using_predefined_plan:
		plan_key = (target_name, circuit_id)
		plan_entry = sampling_plan_index.get(plan_key)
		if plan_entry is None or plan_entry.get("status") != "ok":
			print(f"[Sampling] No usable plan entry for {key}; falling back to random.")
			sampling_strategy = "random"

	if rule_row["component_type"] == "linear":
		# No circuit discovery for purely linear components; cache this decision so reruns stay cheap.
		write_skip_marker(
			output_data_dir_rule,
			reason="Linear component (no circuit discovery)",
			n_pos=0,
			n_neg=0,
			skip_type="linear",
			overwrite=True,
		)
		return {"status": "skipped_linear", "n_pos": 0, "n_neg": 0}

	# Decide how to select examples: sampling plan vs random ANN/length matching
	pairs = None
	pair_meta = None
	if is_using_predefined_plan:
		plan_key = (target_name, circuit_id)
		plan_entry = sampling_plan_index.get(plan_key)

		if plan_entry is None:
			print(f"[Sampling] No plan entry for key={plan_key}; falling back to random.")
		elif plan_entry.get("status") != "ok":
			print(
				f"[Sampling] Plan entry for key={plan_key} has status={plan_entry.get('status')}; "
				"falling back to random."
			)
		else:
			pos_idx = np.array(plan_entry.get("associated_indices", []), dtype=int)
			neg_idx = np.array(plan_entry.get("unrelated_indices", []), dtype=int)
			if pos_idx.size == neg_idx.size:
				pairs = [
					PairItem(clean=texts[raw_pos_id], corrupted=texts[matched_neg_raw_id])
					for raw_pos_id, matched_neg_raw_id in zip(pos_idx, neg_idx)
				]
				pair_meta = plan_entry.get("pairing_meta")
				print(
					f"[Sampling] Using spectral sampling plan for rule {circuit_id} "
					f"({len(pos_idx)} pos, {len(neg_idx)} neg) -> {len(pairs)} pairs."
				)
			else:
				print(
					f"[Sampling] Plan for rule {circuit_id} produced unusable indices; "
					"falling back to random."
				)

	# Fallback: original random/ANN-based strategy
	if pairs is None:
		# indices where the rule fires vs not
		pos_idx = np.where(mask.values)[0]      # positives from rule
		neg_idx = np.where(~mask.values)[0]     # negatives from rule
		pos_idx, neg_idx = balance_positives_and_negatives(pos_idx, neg_idx, t=max_pairs)
		pair_indices, pair_meta = sample_len_tolerant_pairs(
			pos_idx,
			neg_idx,
			emb_all,
			texts,
			k=max_pairs,
			tokenizer=tokenizer,
			len_tolerance=0,
			metric=pair_similarity_metric,
			metric_kwargs=None,
		)
		pairs = [
			PairItem(clean=texts[a], corrupted=texts[b])
			for a, b in pair_indices
		]
		print(
			f"[Sampling] Using ANN+length-matched random strategy for rule {circuit_id}; "
			f"{len(pairs)} pairs."
		)

	if len(pairs) == 0:
		write_skip_marker(
			output_data_dir_rule,
			reason="Pairing produced zero pairs",
			n_pos=0,
			n_neg=0,
			overwrite=True,
		)
		return {"status": "skipped", "n_pos": len(positives), "n_neg": len(negatives)}

	# Run the shared pipeline
	return _run_circuit_discovery_from_pairs(
		pairs=pairs,
		model=model,
		tokenizer=tokenizer,
		output_dir=output_data_dir_rule,
		global_mean_loader=global_mean_loader,
		method=method,
		max_ig_steps=max_ig_steps,
		topn=topn,
		level=level,
		intervention=intervention,
		eval_intervention=eval_intervention,
		prompts_to_answers_dict=prompts_to_answers_dict,
		sampling_strategy=sampling_strategy,
		pair_meta=pair_meta,
		extra_results=None,
		debug_label=f"rule {circuit_id}",
	)

def discover_for_cluster(
	cluster_index,
	pos_idx,
	neg_idx,
	features,
	text_col,
	model,
	tokenizer,
	output_data_dir_cluster,
	emb_all,
	global_mean_loader,
	method="EAP-IG-inputs",
	max_ig_steps=5,
	topn=20,
	level="node",
	max_pairs=512,
	pair_similarity_metric="cosine",
	intervention="zero",
	eval_intervention="mean",
	prompts_to_answers_dict=None,
):
	"""
	Discover a circuit for a single spectral cluster.

	This mirrors discover_for_rule but uses explicit index sets instead of a rule mask.
	"""

	ensure_dir(output_data_dir_cluster)
	texts = features[text_col].astype(str).to_numpy()

	positives = [texts[i] for i in np.atleast_1d(pos_idx)]
	negatives = [texts[i] for i in np.atleast_1d(neg_idx)]

	# If we have nothing on one side, we can't proceed.
	if len(positives) == 0 or len(negatives) == 0:
		write_skip_marker(
			output_data_dir_cluster,
			reason=("Insufficient positives or negatives for spectral cluster " + str(cluster_index)),
			n_pos=int(len(positives)),
			n_neg=int(len(negatives)),
			skip_type="insufficient",
			overwrite=True,
			cluster_index=int(cluster_index),
		)
		return {
			"status": "skipped",
			"cluster_index": int(cluster_index),
			"n_pos": len(positives),
			"n_neg": len(negatives),
		}

	# ANN + length-matched pairing within (cluster vs rest)
	pair_indices, pair_meta = sample_len_tolerant_pairs(
		pos_idx,
		neg_idx,
		emb_all,
		texts,
		k=min(max_pairs, pos_idx.shape[0]),
		tokenizer=tokenizer,
		len_tolerance=0,
		metric=pair_similarity_metric,
		metric_kwargs=None,
	)
	pairs = [
		PairItem(clean=texts[a], corrupted=texts[b])
		for a, b in pair_indices
	]
	print(
		"[Sampling] Using ANN+length-matched random strategy for spectral cluster "
		f"{cluster_index}; {len(pairs)} pairs."
	)
	# print("Using these pairs:", json.dumps(list(map(asdict, pairs)), indent=4))

	if len(pairs) == 0:
		write_skip_marker(
			output_data_dir_cluster,
			reason="Pairing produced zero pairs",
			n_pos=int(len(positives)),
			n_neg=int(len(negatives)),
			skip_type="zero_pairs",
			overwrite=True,
			cluster_index=int(cluster_index),
		)
		return {
			"status": "skipped",
			"cluster_index": int(cluster_index),
			"n_pos": len(positives),
			"n_neg": len(negatives),
		}

	# Run the shared pipeline
	return _run_circuit_discovery_from_pairs(
		pairs=pairs,
		model=model,
		tokenizer=tokenizer,
		output_dir=output_data_dir_cluster,
		global_mean_loader=global_mean_loader,
		method=method,
		max_ig_steps=max_ig_steps,
		topn=topn,
		level=level,
		intervention=intervention,
		eval_intervention=eval_intervention,
		prompts_to_answers_dict=prompts_to_answers_dict,
		sampling_strategy="spectral_clusters",
		pair_meta=pair_meta,
		extra_results={"cluster_index": int(cluster_index)},
		debug_label=f"spectral cluster {cluster_index}",
	)

set_deterministic(args.seed)
task = resolve_task_spec(args.task_module)

rules_dir = Path(args.rules_dir)
scores_path = Path(os.path.join(args.features_scores_dir, "scores.csv"))
output_data_dir = Path(args.output_data_dir)
ensure_dir(output_data_dir)

# Load scores file (contains BOTH prompts and features/targets)
if scores_path.suffix.lower() == ".parquet":
	scores_df = pd.read_parquet(scores_path)
elif scores_path.suffix.lower() == ".csv":
	scores_df = pd.read_csv(scores_path)
else:
	raise ValueError("Scores file must be CSV or Parquet.")

# After loading scores_df
scores_df = safe_features_fillna(scores_df, fill_number=0, fill_bool=False, cols_not_to_fill=task.DEFAULT_TARGETS)

# ------------------------- Train/Test split -------------------------
# If an 'is_test' flag is present, keep TRAIN rows only for this analysis.
if "is_test" in scores_df.columns:
	_mask_train = ~scores_df["is_test"].astype(bool)
	n_before = len(scores_df)
	scores_df = scores_df.loc[_mask_train].reset_index(drop=True)
	print(f"[Split] TRAIN only: {len(scores_df)} / {n_before} rows (dropped {n_before - len(scores_df)} test rows)")

text_col = task.DEFAULT_INPUT
target_col = task.DEFAULT_TARGETS[0]


def build_dataset_info():
	return {
		"scores_path": str(scores_path),
		"prompt_col": text_col,
		"target_col": target_col,
		"ai_model": args.ai_model,
		"analysis_mode": ("spectral_cluster" if args.cluster_by_spectral else "rule"),
		"cluster_base_subset": (args.cluster_base_subset if args.cluster_by_spectral else None),
	}


def write_outputs(manifest):
	(output_data_dir / "manifest.json").write_text(json.dumps(manifest, indent=4))
	(output_data_dir / "dataset_info.json").write_text(json.dumps(build_dataset_info(), indent=4))
	print(f"[Done] Saved circuits to: {output_data_dir}")


def select_cluster_base_subset():
	"""
	Returns (pos_mask, base_name, pos_indices, full_indices, k) without touching the LLM.
	"""
	if target_col not in scores_df.columns:
		raise ValueError(
			f"task.DEFAULT_TARGETS[0]='{target_col}' is not a column in scores_df; "
			"cannot restrict clusters to positives."
		)

	labels = scores_df[target_col]
	if args.cluster_base_subset == "positive":
		pos_mask = (labels == True)
		base_name = "positive"
	elif args.cluster_base_subset == "negative":
		pos_mask = (labels == False)
		base_name = "negative"
	else:
		pos_mask = np.ones(len(scores_df), dtype=bool)
		base_name = "all"

	pos_mask_np = np.asarray(pos_mask.to_numpy() if hasattr(pos_mask, "to_numpy") else pos_mask)
	pos_indices = np.where(pos_mask_np)[0]
	# neg_indices = np.where(~pos_mask_np)[0]
	# pos_indices, _ = balance_positives_and_negatives(pos_indices, neg_indices, t=args.max_pairs_per_circuit)

	if pos_indices.size == 0:
		raise RuntimeError(
			f"No examples found for cluster_base_subset={base_name!r} w.r.t. DEFAULT_OUTPUT='{target_col}'. "
			"Spectral clustering requires at least one example in the chosen base subset."
		)

	full_indices = np.arange(len(scores_df))
	k = min(args.global_n_clusters, int(pos_indices.size))
	return base_name, pos_indices, full_indices, k


def finalize_job_result(job, result):
	# Normalize manifest entry shape
	out = dict(result)
	out["target"] = target_col
	out["circuit_id"] = int(job["circuit_id"])
	out["circuit_label"] = job["circuit_label"]
	out["coefficient_sign"] = job.get("coefficient_sign")
	if job["kind"] == "cluster":
		out["cluster_index"] = int(job["cluster_index"])
	return out


def load_from_cache_if_available(
	output_dir,
	*,
	sampling_strategy,
	default_skip_reason: str,
):
	state = cache_state(output_dir)
	if state == "ok":
		return load_cached_ok(output_dir, sampling_strategy=sampling_strategy)
	if state == "skipped":
		return load_cached_skipped(output_dir, default_reason=default_skip_reason)
	return None


# ------------------- Build jobs (used by BOTH preflight and main run) -------------------
jobs = []

cluster_ctx = None  # filled only in cluster mode
clusters_root = None

if args.cluster_by_spectral:
	base_name, pos_indices, full_indices, k = select_cluster_base_subset()
	clusters_root = output_data_dir / f"spectral_clusters_{base_name}"
	cluster_ctx = {
		"base_name": base_name,
		"pos_indices": pos_indices,
		"full_indices": full_indices,
		"k": int(k),
	}
	for cluster_index in range(k):
		out_dir = clusters_root / f"cluster_{cluster_index}"
		jobs.append(
			{
				"kind": "cluster",
				"cluster_index": int(cluster_index),
				"output_dir": out_dir,
				"circuit_id": int(cluster_index),
				"circuit_label": f"spectral_cluster_{cluster_index}",
				"coefficient_sign": None,
				"sampling_strategy": "spectral_clusters",
			}
		)

else:
	rule_files = find_rule_files(rules_dir, args.rules_glob)
	if not rule_files:
		raise ValueError(f"No rule files found in {rules_dir} matching '{args.rules_glob}'.")

	max_n_of_rules_to_analyze = args.max_n_of_rules_to_analyze
	if max_n_of_rules_to_analyze <= 0:
		max_n_of_rules_to_analyze = None

	for rule_target, rule_path in rule_files:
		if args.fake_targets:
			if rule_target != target_col + "_fake":
				continue
		else:
			if rule_target != target_col:
				continue

		rules_df = load_rules(rule_path)
		rule_entry_list = list(rules_df.iterrows())[:max_n_of_rules_to_analyze]
		for _, row in rule_entry_list:
			slug = slugify(row["rule"])[:64]
			out_dir = output_data_dir / target_col / f"rule_{int(row['rule_id'])}_{slug}"
			is_linear = (str(row.get("component_type", "")) == "linear")
			jobs.append(
				{
					"kind": "rule",
					"row": row,
					"output_dir": out_dir,
					"circuit_id": int(row["rule_id"]),
					"circuit_label": str(row["rule"]),
					"coefficient_sign": row.get("coefficient_sign"),
					"sampling_strategy": args.sampling_strategy,
					"is_linear": bool(is_linear),
				}
			)

# ------------------- Preflight cache check -------------------
missing_any = False
for job in jobs:
	out_dir = job["output_dir"]
	if job.get("is_linear", False) and cache_state(out_dir) == "missing":
		# Make linear components resumable without loading the LLM
		n_pos = None
		n_neg = None
		mask = apply_rule_to_features(str(job["row"]["rule"]), scores_df, direction=job["row"].get("coefficient_sign"))
		mask_vals = mask.values if hasattr(mask, "values") else mask
		n_pos = int(np.sum(mask_vals))
		n_neg = int(len(scores_df) - n_pos)
		write_skip_marker(
			out_dir,
			reason="Linear component (no circuit discovery)",
			n_pos=n_pos,
			n_neg=n_neg,
			skip_type="linear",
			overwrite=False,
		)

	if cache_state(out_dir) == "missing":
		missing_any = True
		break

if not missing_any:
	print("[Cache] All requested circuits already exist; skipping model + embedding initialization.")
	manifest = []
	for job in jobs:
		out_dir = job["output_dir"]
		default_reason = (
			f"Insufficient positives or negatives for spectral cluster {job['cluster_index']}"
			if job["kind"] == "cluster"
			else f"Insufficient positives or negatives for rule {job['circuit_id']}"
		)
		cached = load_from_cache_if_available(
			out_dir,
			sampling_strategy=job.get("sampling_strategy"),
			default_skip_reason=default_reason,
		)
		if cached is None:
			raise RuntimeError(f"Preflight expected cache for {out_dir}, but found none.")
		manifest.append(finalize_job_result(job, cached))

	write_outputs(manifest)
	raise SystemExit(0)

# ------------------- Heavy init (only if something is missing) -------------------

# Build model & tokenizer using your library (LMWrapper)
device = get_device()
wrapper = LMWrapper(
	model_name=args.ai_model,
	device=device,
	eval_mode=True,
	ungroup_grouped_query_attention=True,
	circuit_discovery=True,
	cache_dir=args.ai_model_cache_dir,
)
unhooked_model = getattr(wrapper, "model", None)
model = getattr(wrapper, "hooked_model", None)
tokenizer = getattr(wrapper, "tokenizer", None)

st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
texts = scores_df[text_col].astype(str).to_numpy()
emb_all = np.asarray(
	st.encode(
		texts,
		batch_size=512,
		show_progress_bar=True,
		device=device,
		convert_to_tensor=False,
		normalize_embeddings=False,  # no need to keep embeddings normalized, the cdist function will deal with it on its own
	),
	dtype=np.float32,
)

# Optional: load sampling plan if requested
sampling_plan_index = None
if (not args.cluster_by_spectral) and args.sampling_strategy == "plan":
	if args.sampling_plan_path is None:
		raise ValueError("--sampling_strategy 'plan' requires --sampling_plan_path.")
	sampling_plan_path = Path(args.sampling_plan_path)
	if not sampling_plan_path.exists():
		raise FileNotFoundError(f"Sampling plan not found: {sampling_plan_path}")
	with open(sampling_plan_path, "r") as f:
		sampling_plan = json.load(f)

	sampling_plan_index = {}
	assert len(texts) == sampling_plan["dataset_size"]
	for r in sampling_plan["rules"]:
		circuit_id = int(r["rule_id"])
		rule_target = r["rule_target"]
		key = (rule_target, circuit_id)
		sampling_plan_index[key] = r
	print(
		f"[Sampling] Loaded sampling plan from {sampling_plan_path} "
		f"with {len(sampling_plan_index)} rule entries."
	)

# Build prompt->target only once (index by text to avoid KeyErrors)
prompts_to_answers_dict = None
output_col = task.DEFAULT_OUTPUT
sub = scores_df[[text_col, output_col]].dropna().astype(str)
sub = sub.drop_duplicates(subset=[text_col])  # first occurrence wins
sub = sub.set_index(text_col)
if not sub.empty:
	prompts_to_answers_dict = sub[output_col].to_dict()

# Mean ablation loader (optional)
if "mean" in args.intervention or "mean" in args.eval_intervention:
	n = min(args.points_to_use_for_mean_ablation, len(texts) // 2)
	if n < 2:
		global_mean_pairs = []
	else:
		idx = list(range(len(texts)))
		random.shuffle(idx)
		clean_idx = idx[:n]
		corr_idx = idx[n:]
		random.shuffle(corr_idx)
		global_mean_pairs = [PairItem(clean=texts[i], corrupted=texts[j]) for i, j in zip(clean_idx, corr_idx)]

	global_mean_loader = torch.utils.data.DataLoader(
		PairDataset(global_mean_pairs, tokenizer=tokenizer, prompts_to_answers_dict=prompts_to_answers_dict),
		batch_size=args.batch_size,
		shuffle=False,
	)
else:
	global_mean_loader = None

# ------------------- Compute missing jobs -------------------
manifest = []

if args.cluster_by_spectral:
	assert cluster_ctx is not None and clusters_root is not None
	base_name = cluster_ctx["base_name"]
	pos_indices = cluster_ctx["pos_indices"]
	full_indices = cluster_ctx["full_indices"]
	k = int(cluster_ctx["k"])

	# Texts for the chosen base subset (this is what we cluster)
	pos_texts = scores_df.iloc[pos_indices][text_col].astype(str).tolist()

	emb_repr_pos, Z_pos = build_reps_and_embedding_from_args(
		args=args,
		texts=pos_texts,
		model=unhooked_model,
		tokenizer=tokenizer,
		device=device,
	)

	# Keep k consistent with preflight (but ensure it's <= available points)
	k = min(k, int(Z_pos.shape[0]))
	centers_idx, _, _, meta, x_norm2 = kcenter_farthest_first(Z_pos, k=k)
	cluster_ids_pos = assign_min_size_nearest_to_centers(
		Z_pos,
		centers_idx,
		min_size=args.max_pairs_per_circuit,
		chunk_size=8192,
	).astype(int)


	sizes = np.bincount(cluster_ids_pos, minlength=k)
	print(
		"[Spectral] Formed {k} clusters on {base} "
		"(min size={min_sz}, max size={max_sz})".format(
			k=k,
			base=base_name,
			min_sz=int(sizes.min()),
			max_sz=int(sizes.max()),
		)
	)

	ensure_dir(clusters_root)

	for cluster_index in range(k):
		# Locate job dict
		job = next(j for j in jobs if j["kind"] == "cluster" and int(j["cluster_index"]) == int(cluster_index))
		out_dir = job["output_dir"]
		ensure_dir(out_dir)

		# Members of this cluster within the base subset…
		members_mask = cluster_ids_pos == cluster_index
		pos_idx = pos_indices[members_mask] # indices into the full dataset
		# …and negatives are simply "everything else"
		neg_idx = np.setdiff1d(full_indices, pos_idx)
		pos_idx, neg_idx = balance_positives_and_negatives(pos_idx, neg_idx, t=args.max_pairs_per_circuit)

		default_reason = f"Insufficient positives or negatives for spectral cluster {cluster_index}"
		cached = load_from_cache_if_available(
			out_dir,
			sampling_strategy="spectral_clusters",
			default_skip_reason=default_reason,
		)

		if cached is None:
			result = discover_for_cluster(
				cluster_index=cluster_index,
				pos_idx=pos_idx,
				neg_idx=neg_idx,
				features=scores_df,
				text_col=text_col,
				model=model,
				tokenizer=tokenizer,
				output_data_dir_cluster=out_dir,
				emb_all=emb_all,
				global_mean_loader=global_mean_loader,
				method=args.method,
				max_ig_steps=args.max_ig_steps,
				topn=args.circuit_size,
				level=args.circuit_level,
				max_pairs=args.max_pairs_per_circuit,
				pair_similarity_metric=args.pair_similarity_metric,
				intervention=args.intervention,
				eval_intervention=args.eval_intervention,
				prompts_to_answers_dict=prompts_to_answers_dict,
			)
		else:
			result = cached

		manifest.append(finalize_job_result(job, result))

else:
	# Rule-based processing
	for job in jobs:
		row = job["row"]
		out_dir = job["output_dir"]
		ensure_dir(out_dir)

		default_reason = f"Insufficient positives or negatives for rule {job['circuit_id']}"
		cached = load_from_cache_if_available(
			out_dir,
			sampling_strategy=args.sampling_strategy,
			default_skip_reason=default_reason,
		)

		if cached is None:
			result = discover_for_rule(
				rule_row=row,
				features=scores_df,
				text_col=text_col,
				model=model,
				tokenizer=tokenizer,
				output_data_dir_rule=out_dir,
				emb_all=emb_all,
				global_mean_loader=global_mean_loader,
				method=args.method,
				max_ig_steps=args.max_ig_steps,
				topn=args.circuit_size,
				level=args.circuit_level,
				max_pairs=args.max_pairs_per_circuit,
				pair_similarity_metric=args.pair_similarity_metric,
				intervention=args.intervention,
				eval_intervention=args.eval_intervention,
				prompts_to_answers_dict=prompts_to_answers_dict,
				sampling_strategy=args.sampling_strategy,
				sampling_plan_index=sampling_plan_index,
				target_name=target_col,
			)
		else:
			result = cached

		manifest.append(finalize_job_result(job, result))

write_outputs(manifest)
