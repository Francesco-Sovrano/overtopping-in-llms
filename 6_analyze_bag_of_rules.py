import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import re
import gc
import json
import argparse
import ast  # for parsing manifest neuron_label_score keys like "('m27', 277)"
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Your local utils
from lib.modeling_and_ablation import (
	LMWrapper,
	get_device,
	get_layer_type_and_ids, 
	precompute_mean_activations,
	get_circuit_neurons_dict
)
from lib.text_and_rules import (
	apply_rule_to_features,
	guess_filetype
)
from lib.caching_and_prompting import set_deterministic
from lib.spectral_analysis import (
	kcenter_farthest_first,
	build_reps_and_embedding_from_args,
	add_spectral_cli_args,
	balance_positives_and_negatives,
	assign_min_size_nearest_to_centers
)
from lib.feature_extraction_runner import resolve_task_spec
from lib.feature_representation import safe_features_fillna
from lib.neuron_intervention import *
from lib.threshold_event_shared import collect_reference_margin_tensors

"""
Bag-of-rules analysis / ablation runner.

Given a discovery output folder (manifest.json + dataset_info.json + scores.{csv,parquet}),
this script evaluates how discovered neurons (and neuron *groups*) affect performance on:
	- associated examples (prompts that satisfy a given rule)
	- unrelated examples (prompts that do not satisfy that rule)

Key outputs per rule/cluster are JSON files containing baseline metrics plus ablation records.
When --fast_ablation is enabled, the script uses a layer-wise dichotomic search (binary tree)
to cheaply localize impactful neurons while safely pruning low-effect subtrees.

Optional experimental knob: --sign_split_first enforces a *root* split by discovery-score sign
(>=0 vs <0) using manifest metadata_topn["neuron_label_score"], then continues bisection within
each sign pool. This can help separate putative agonists/antagonists early in the search.
"""

# ------------------------------- CLI ------------------------------------
def parse_args():
	p = argparse.ArgumentParser(
		description="Bag-of-Rules ablations over neural_circuit_discovery_results."
	)
	p.add_argument(
		"--input_data_dir",
		type=str,
		required=True,
		help="Path to neural_circuit_discovery_results/eap_ig_inputs (folder with manifest.json).",
	)
	p.add_argument(
		"--output_data_dir",
		type=str,
		required=True,
	)

	# If dataset_info.json is missing or you want to override:
	p.add_argument(
		"--scores_path",
		type=str,
		default=None,
		help="CSV/Parquet with prompt/target and feature columns (e.g., feature_report/scores.csv).",
	)

	# Model overrides (defaults read from dataset_info.json created by the discovery script)
	p.add_argument(
		"--ai_model",
		type=str,
		default=None,
		help="HF model id; default from dataset_info.json.",
	)
	p.add_argument('--ai_model_cache_dir', default=None, type=str)

	# Evaluation / sampling knobs
	p.add_argument(
		"--seed",
		type=int,
		default=42,
	)
	p.add_argument(
		"--search_epsilon",
		type=float,
		default=0.2,
		help=(
			"Minimum |max_effect| needed to continue splitting a neuron group in the "
			"dichotomic search. Groups whose effect on (acc_unrelated - acc_associated) "
			"is below this threshold are treated as 'useless' and not split further. "
			"Set <= 0 to always fully split down to single neurons. "
			"This same value is also used as the |max_effect| threshold for bucketing "
			"single-neuron non-catastrophic agonists."
		),
	)
	p.add_argument(
		"--prune_alpha",
		type=float,
		default=0.05,
	)
	p.add_argument(
		"--catastrophic_acc_tol",
		type=float,
		default=1e-12,
		help=(
			"Tolerance used to treat acc_after_knockout_on_associated/unrelated as zero when "
			"detecting catastrophic neurons (only under --baseline_subset positive)."
		),
	)
	p.add_argument(
		"--catastrophic_n_runs",
		type=int,
		default=3,
		help=(
			"Number of distinct runs required to classify a single neuron as always catastrophic "
			"(accuracy_on_associated=0 and accuracy_on_unrelated=0) vs not_always."
		)
	)
	p.add_argument(
		"--n_associated",
		type=int,
		default=100,
		help="Per-rule: number of prompts that satisfy the rule to test.",
	)
	p.add_argument(
		"--n_unrelated",
		type=int,
		default=100,
		help="Per-rule: number of prompts that do NOT satisfy the rule to test.",
	)
	p.add_argument(
		"--decode_only",
		action="store_true",
		help=(
			"Disable fast decode-only mode that uses prefix caches. "
			"When set, ablations run without precomputed prefix caches."
		),
	)

	# ablation type / mean estimation
	p.add_argument(
		"--intervention",
		type=str,
		default="zero",
		choices=["zero", "mean", "mean-donor", "mean-positional", "mean-donor-positional"],
		help=(
			"Type of ablation to perform: "
			"'zero' sets activations to 0 (old behavior), "
			"'mean' replaces them with a dataset-wide mean, "
			"'mean-donor' replaces them with the observed activation closest to that mean, "
			"'mean-positional' uses per-position means when available, and "
			"'mean-donor-positional' snaps those per-position means to the closest observed activations."
		),
	)
	p.add_argument(
		"--points_to_use_for_mean_ablation",
		type=int,
		default=256,
		help="Number of prompts used to estimate replacements for mean / mean-donor / mean-positional / mean-donor-positional ablation.",
	)
	# Prompt-guided ablation
	p.add_argument(
		"--top_k_neurons",
		type=int,
		default=None,
		help=(
			"If set, per rule only the top-K units are kept for ablation (applies to both MLP neurons and attention-head dimensions when present). "
			"Uses metadata_topn['neuron_label_score'] when available; otherwise picks deterministically from the candidate unit lists, prioritizing higher-scored layers/heads."
		),
	)
	p.add_argument(
		"--last_k_layers",
		type=int,
		default=None,
		help="If set, for each rule only keep neurons belonging to the last K MLP layers (by numeric layer id, e.g. m24,m25,...).",
	)
	p.add_argument(
		"--mlp_neurons_only",
		action="store_true",
		help=(
			"If set, only ablate MLP-layer neurons. Any non-MLP units present in the discovered circuit "
			"(for example attention-head dimensions) are ignored."
		),
	)
	p.add_argument(
		"--batch_size",
		type=int,
		default=16,
		help="Batch size for model eval (baseline and ablations). Lower = less memory, slower but safer.",
	)
	
	p.add_argument(
		"--fast_ablation",
		action="store_true",
		help="If set, ablate neurons using a layer-wise dichotomic search instead of full search.",
	)
	# Optional: when using --fast_ablation, you can force the FIRST split per layer
	# to separate neurons by the SIGN of their discovery importance score (from manifest
	# metadata_topn['neuron_label_score']). This gives two disjoint pools (score>=0 vs score<0)
	# and then continues the usual dichotomic bisection within each pool.
	p.add_argument(
		"--sign_split_first",
		action="store_true",
		help=(
			"If set (and if manifest metadata includes signed neuron scores), the layer-wise "
			"dichotomic search uses a sign-based split at depth 0: neurons with score>=0 are "
			"searched as one subtree and score<0 as the other. If scores are missing or all "
			"scores have the same sign, it falls back to a standard 50/50 bisection."
		),
	)


	p.add_argument(
		"--baseline_subset",
		type=str,
		choices=["all", "positive", "negative"],
		default="all",
		help="Whether to run on all prompts or only those where the baseline model w.r.t. the primary_target was positive or negative.",
	)

	# sampling strategy (plan vs random)
	p.add_argument(
		"--sampling_strategy",
		type=str,
		default="random",
		choices=["random", "plan"],
		help="How to select prompts per rule: 'random' (current behavior) or 'plan' (use precomputed spectral sampling plan).",
	)
	p.add_argument(
		"--sampling_plan_path",
		type=str,
		default=None,
		help="Path to JSON sampling plan produced by script 4. Required when --sampling_strategy plan.",
	)

	# Spectral cluster-based analysis (shared with script 5)
	p.add_argument(
		"--cluster_by_spectral",
		action="store_true",
		help=(
			"If set, interpret circuits in manifest as spectral clusters rather than simple rules. "
			"Associated prompts for each entry are those in the corresponding spectral cluster."
		),
	)
	p.add_argument(
		"--global_n_clusters",
		type=int,
		default=32,
		help="Number of spectral clusters to form / expect; should match script 5.",
	)
	add_spectral_cli_args(p)

	p.add_argument(
		"--skip_agonist_activation_stats",
		action="store_true",
		help=(
			"If set, skip the extra post-hoc diagnostics that measure the actual raw values of singleton "
			"agonist activations relative to the other activations in the same hooked layer and write "
			"CSV/JSON/PNG summaries."
		),
	)

	p.add_argument(
		"--skip_agonist_margin_stats",
		action="store_true",
		help=(
			"If set, skip the extra post-hoc diagnostics that estimate singleton agonist first-order "
			"margin effects via gradient × (baseline-current) at the hooked write site and write "
			"CSV/JSON/PNG summaries."
		),
	)

	p.add_argument(
		"--skip_agonist_saliency_stats",
		action="store_true",
		help=(
			"If set, skip post-hoc diagnostics that rank singleton agonists by saliency-style "
			"scores such as Wanda, |gradient|, and |activation × gradient| against the "
			"other units in the same hooked layer."
		),
	)
	p.add_argument(
		"--agonist_saliency_metrics",
		type=str,
		default="wanda,gradient,activation_x_gradient",
		help=(
			"Comma-separated metrics for agonist saliency diagnostics. Supported aliases: "
			"wanda, gradient, activation_x_gradient, activation_gradient, actgrad."
		),
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
# ------------------------- Helpers / dataset ----------------------------
def _load_dataset_info(results_root):
	info_path = results_root / "dataset_info.json"
	return json.loads(info_path.read_text()) if info_path.exists() else {}

def _build_balanced_mean_prompt_pool(df: pd.DataFrame, prompt_col: str, target_col: str, n_points: int, seed: int, baseline_subset: str = "all"):
	"""Build a prompt pool for mean-ablation estimation from the TRAIN split.

	Selection policy:
	- baseline_subset == "positive": sample from TRAIN positives only
	- baseline_subset == "negative": sample from TRAIN negatives only
	- baseline_subset == "all": sample from the full TRAIN pool

	Sampling is deterministic up to `seed` and never uses replacement.
	Falls back to the full TRAIN pool if the requested label column is missing.
	"""
	if df is None or len(df) == 0 or prompt_col not in df.columns or n_points <= 0:
		return []

	rng = np.random.default_rng(seed)
	pool_df = df
	pool_label = "all train"

	if target_col in df.columns:
		baseline_subset = str(baseline_subset or "all").lower()
		if baseline_subset == "positive":
			pool_df = df.loc[df[target_col] == True]
			pool_label = "train positives"
		elif baseline_subset == "negative":
			pool_df = df.loc[df[target_col] == False]
			pool_label = "train negatives"
		elif baseline_subset == "all":
			pool_df = df
			pool_label = "all train"

	if len(pool_df) == 0:
		print(
			f"[MeanAblation] Requested {pool_label} pool is empty; falling back to full TRAIN pool "
			f"for mean estimation."
		)
		pool_df = df
		pool_label = "all train"

	available = int(len(pool_df))
	take = min(int(n_points), available)
	selected = rng.choice(pool_df.index.to_numpy(), size=take, replace=False).tolist()
	print(
		f"[MeanAblation] Using {pool_label} pool for mean estimation: "
		f"{len(selected)} prompts (requested={int(n_points)}, available_pool={available})."
	)
	return pool_df.loc[selected, prompt_col].astype(str).tolist()

def _summary_entry_from_rule_detail(detail: dict) -> dict:
	status = detail.get("status", "unknown")
	out = {
		"circuit_id": detail.get("circuit_id"),
		"circuit_label": detail.get("circuit_label"),
		"analysis_mode": detail.get("analysis_mode", "rule"),
		"cluster_index": detail.get("cluster_index"),
		"rule_target": detail.get("rule_target"),
		"rule_direction": detail.get("rule_direction"),
		"status": status,
		"baseline_subset": detail.get("baseline_subset"),
	}

	if status == "skipped":
		out.update(
			{
				"reason": detail.get("reason"),
				"n_associated_positive": int(detail.get("n_associated_positive", 0)),
				"n_unrelated_positive": int(detail.get("n_unrelated_positive", 0)),
				"n_associated_tested": int(detail.get("n_associated_tested", 0)),
				"n_unrelated_tested": int(detail.get("n_unrelated_tested", 0)),
			}
		)
		return out

	if status == "ok":
		abls = detail.get("ablations", []) or []
		out.update(
			{
				"n_associated_positive": int(detail.get("n_associated_positive", 0)),
				"n_unrelated_positive": int(detail.get("n_unrelated_positive", 0)),
				"n_associated_tested": int(detail.get("n_associated_tested", 0)),
				"n_unrelated_tested": int(detail.get("n_unrelated_tested", 0)),
				"baseline_acc_associated": float(detail.get("baseline_acc_associated", 0.0)),
				"baseline_acc_unrelated": float(detail.get("baseline_acc_unrelated", 0.0)),
				"baseline_gap_neg_minus_pos": float(detail.get("baseline_gap_neg_minus_pos", 0.0)),
				"n_ablation_groups": int(len(abls)),
				"sampling_strategy_used": detail.get("sampling_strategy_used"),
			}
		)
		if "best_group" in detail and detail["best_group"] is not None:
			out["best_group"] = detail["best_group"]
		return out

	# fallback for unexpected statuses
	out.update(
		{
			"reason": detail.get("reason"),
			"n_associated_positive": int(detail.get("n_associated_positive", 0)),
			"n_unrelated_positive": int(detail.get("n_unrelated_positive", 0)),
		}
	)
	return out


# ------------------------- Resume scan helpers ----------------------------
def _circuit_output_json_path(info: dict, rule_out_root: Path, args) -> Path:
	"""Return expected output JSON path for a circuit entry (rule or spectral cluster)."""
	rid = int(info.get("circuit_id"))
	analysis_mode = info.get("analysis_mode", ("spectral_cluster" if getattr(args, "cluster_by_spectral", False) else "rule"))
	rule_target = info.get("target", info.get("rule_target"))
	if analysis_mode == "spectral_cluster":
		out_dir = rule_out_root / "spectral_clusters"
		json_name = f"cluster_{rid:04d}.json"
	else:
		out_dir = rule_out_root / str(rule_target)
		json_name = f"rule_{rid:04d}.json"
	out_dir.mkdir(parents=True, exist_ok=True)
	return out_dir / json_name

def _cached_rule_detail_is_reusable(cached: dict, info: dict, args) -> bool:
	"""Mirror the in-loop RESUME predicate so pre-scan behavior matches runtime behavior."""
	try:
		rid = int(info.get("circuit_id"))
	except Exception:
		return False
	rule = info.get("circuit_label")
	rule_target = info.get("target", info.get("rule_target"))
	analysis_mode = info.get("analysis_mode", ("spectral_cluster" if getattr(args, "cluster_by_spectral", False) else "rule"))
	rule_direction = None if analysis_mode == "spectral_cluster" else info.get("coefficient_sign", info.get("rule_direction"))

	status = cached.get("status")
	if status not in ("ok", "skipped"):
		return False

	try:
		cached_cid = int(cached.get("circuit_id"))
	except Exception:
		cached_cid = cached.get("circuit_id")

	return (
		cached_cid == rid
		and cached.get("circuit_label") == rule
		and cached.get("rule_target") == rule_target
		and cached.get("rule_direction") == rule_direction
	)

def _write_bucket_outputs(rule_out_root: Path, buckets: dict, buckets_path: Path):
	"""Persist neuron buckets + stats (shared by normal and early-exit resume paths)."""
	buckets_path.write_text(json.dumps(buckets, indent=4))

	bucket_counts = _bucket_counts(buckets)
	bucket_stats = {
		"counts": bucket_counts,
		"excluded_from_future_ablations": int(
			bucket_counts["catastrophic_zero_candidates"]
			+ bucket_counts["catastrophic_zero_confirmed"]
			+ bucket_counts["non_catastrophic_agonists"]
		),
	}
	(rule_out_root / "neuron_bucket_stats.json").write_text(json.dumps(bucket_stats, indent=4))

	print(
		"[Buckets] Final stats: "
		f"catastrophic_zero_confirmed={bucket_counts['catastrophic_zero_confirmed']}, "
		f"catastrophic_zero_candidates={bucket_counts['catastrophic_zero_candidates']}, "
		f"catastrophic_zero_not_always={bucket_counts['catastrophic_zero_not_always']}, "
		f"non_catastrophic_agonists={bucket_counts['non_catastrophic_agonists']}"
	)

def _scan_reusable_circuits(circuit_entries: dict, rule_out_root: Path, args):
	"""Pre-scan disk outputs and return (summary_entries, buckets, circuits_to_process)."""
	summary_entries = []
	buckets = _init_buckets()
	circuits_to_process = []

	n_total = len(circuit_entries)
	n_reused = 0
	n_backfill = 0

	for info in circuit_entries.values():
		json_path = _circuit_output_json_path(info, rule_out_root, args)
		if json_path.exists():
			try:
				cached = json.loads(json_path.read_text())
			except Exception:
				cached = None

			if cached is not None and _cached_rule_detail_is_reusable(cached, info, args):
				summary_entries.append(_summary_entry_from_rule_detail(cached))
				for rec in cached.get("ablations", []) or []:
					if rec.get("group_size", 0) != 1:
						continue
					_update_buckets_from_single_record(
						buckets,
						rec,
						baseline_subset=args.baseline_subset,
						circuit_id=int(info.get("circuit_id")),
					)
				if _needs_agonist_activation_backfill(cached, args, json_path):
					resume_info = dict(info)
					resume_info["_resume_mode"] = "backfill_agonist_activation_stats"
					resume_info["_cached_rule_detail"] = cached
					circuits_to_process.append(resume_info)
					n_backfill += 1
				else:
					n_reused += 1
				continue

		circuits_to_process.append(info)

	if n_reused or n_backfill:
		print(f"[Resume] Reusing {n_reused}/{n_total} finished circuits from disk.")
		if n_backfill:
			print(f"[Resume] Backfilling missing agonist activation/saliency stats for {n_backfill} cached circuits.")
	return summary_entries, buckets, circuits_to_process

# ------------------------- Neuron bucketing (cross-run caching) ----------------------------
# Motivation:
# - Some single neurons can be "catastrophic" under baseline_subset=positive, producing accuracy_on_associated=0 and accuracy_on_unrelated=0.
#   These neurons force dichotomic search to fully split groups; we keep them in a separate bucket to speed up
#   subsequent runs.
# - Neurons already identified as non-catastrophic agonists (abs(max_effect) >= 0.1) are also bucketed and
#   excluded from future ablations.

AGONIST_ABS_GAP_THRESHOLD = args.search_epsilon
CATASTROPHIC_ACC_TOL = args.catastrophic_acc_tol
CATASTROPHIC_N_RUNS = args.catastrophic_n_runs

def _neuron_key(layer_label: str, neuron_id: int) -> str:
	return f"{layer_label}:{int(neuron_id)}"

def _extract_single_neuron_key_from_record(record: dict):
	"""Return canonical neuron key if record is a single-neuron ablation, else None."""
	try:
		if int(record.get("group_size", 0)) != 1:
			return None
		layers_neurons_dict = record.get("layers_neurons_dict") or {}
		if len(layers_neurons_dict) != 1:
			return None
		layer_label = next(iter(layers_neurons_dict.keys()))
		neuron_id = int(list(layers_neurons_dict[layer_label])[0])
		return _neuron_key(layer_label, neuron_id)
	except Exception:
		return None


def _parse_manifest_neuron_label(label: str):
	"""
	Parse a manifest unit label key into (layer_label, unit_id).

	The discovery pipeline stores signed importance scores in
	metadata_topn['neuron_label_score'] with stringified tuple keys like:
		"('m27', 277)"
	or attention-head entries like:
		"('a27.h3', 41)"

	Returns (layer_label:str, unit_id:int) on success, otherwise (None, None).
	"""
	if label is None:
		return None, None
	# Be permissive: allow MLP layers (m27), attention layers (a27),
	# attention heads (a27.h3), and similar future layer labels.
	m = re.match(r"\(?'?([A-Za-z]\w*(?:\.h\d+)?)'?\s*,\s*(\d+)\)?", str(label).strip())
	if m:
		return m.group(1), int(m.group(2))
	return None, None


def _merge_layer_units_dict(*layer_maps):
	merged = defaultdict(set)
	for layer_map in layer_maps:
		if not isinstance(layer_map, dict):
			continue
		for layer_label, unit_ids in layer_map.items():
			if unit_ids is None:
				continue
			for unit_id in unit_ids:
				try:
					merged[str(layer_label)].add(int(unit_id))
				except Exception:
					continue
	return {layer_label: sorted(unit_ids) for layer_label, unit_ids in merged.items()}


def _filter_layer_units_for_ablation(layer_units: dict, *, mlp_neurons_only: bool = False) -> dict:
	"""Filter a {layer_label: [unit_ids]} mapping to the units eligible for ablation."""
	if not isinstance(layer_units, dict) or not layer_units:
		return {}
	if not mlp_neurons_only:
		return {
			str(layer_label): sorted({int(unit_id) for unit_id in unit_ids})
			for layer_label, unit_ids in layer_units.items()
			if unit_ids
		}

	filtered = {}
	for layer_label, unit_ids in layer_units.items():
		parsed = get_layer_type_and_ids(layer_label)
		if not parsed:
			continue
		layer_type, _, _ = parsed
		if layer_type != "mlp":
			continue
		clean_ids = sorted({int(unit_id) for unit_id in unit_ids})
		if clean_ids:
			filtered[str(layer_label)] = clean_ids
	return filtered


def _extract_layer_units(info: dict) -> dict:
	"""
	Return a normalized {layer_label: [unit_ids]} mapping for one manifest entry.

	Script 5 may emit circuit units under different field names depending on whether
	the units are MLP neurons or attention-head dimensions. We accept all of them
	here and, if needed, reconstruct the mapping from metadata_topn['neuron_label_score'].
	"""
	meta = info.get('metadata_topn') or {}
	units = _merge_layer_units_dict(
		info.get('units'),
		info.get('neurons'),
		info.get('mlp_neurons'),
		info.get('attention_heads'),
		info.get('attn_heads'),
		meta.get('units'),
		meta.get('neurons'),
		meta.get('mlp_neurons'),
		meta.get('attention_heads'),
		meta.get('attn_heads'),
	)
	if units:
		return units

	label_score = meta.get('neuron_label_score') or {}
	if isinstance(label_score, dict) and label_score:
		rebuilt = defaultdict(set)
		for label in label_score.keys():
			layer_label, unit_id = _parse_manifest_neuron_label(label)
			if layer_label is None or unit_id is None:
				continue
			rebuilt[layer_label].add(unit_id)
		if rebuilt:
			return {layer_label: sorted(unit_ids) for layer_label, unit_ids in rebuilt.items()}
	return {}


def _is_catastrophic_zero_zero(record):
	accuracy_on_associated = float(record.get("acc_after_knockout_on_associated", 0.0))
	accuracy_on_unrelated = float(record.get("acc_after_knockout_on_unrelated", 0.0))
	return (abs(accuracy_on_associated) <= CATASTROPHIC_ACC_TOL) and (abs(accuracy_on_unrelated) <= CATASTROPHIC_ACC_TOL)

def _is_catastrophic_one_one(record):
	accuracy_on_associated = float(record.get("acc_after_knockout_on_associated", 1.0))
	accuracy_on_unrelated = float(record.get("acc_after_knockout_on_unrelated", 1.0))
	return (abs(accuracy_on_associated) >= 1-CATASTROPHIC_ACC_TOL) and (abs(accuracy_on_unrelated) >= 1-CATASTROPHIC_ACC_TOL)

def _init_buckets():
	return {
		"catastrophic_zero": {
			# Candidates need up to CATASTROPHIC_N_RUNS distinct runs to decide whether they are always zero-zero.
			"candidates": {},  # key -> {n_trials, n_zero_zero, history}
			"confirmed": {},   # key -> candidate dict promoted when n_trials==CATASTROPHIC_N_RUNS and n_zero_zero==n_trials
			"not_always": {},  # key -> candidate dict promoted when n_trials==CATASTROPHIC_N_RUNS and n_zero_zero < n_trials
		},
		"non_catastrophic_agonists": {
			# key -> {best_abs_gap, last_record}
		}
	}

def _compact_single_record(record: dict, *, circuit_id=None, baseline_subset=None) -> dict:
	layers_neurons_dict = record.get("layers_neurons_dict") or {}
	layer_label = next(iter(layers_neurons_dict.keys())) if layers_neurons_dict else None
	neuron_id = None
	if layer_label is not None:
		try:
			neuron_id = int(list(layers_neurons_dict[layer_label])[0])
		except Exception:
			neuron_id = None

	return {
		"circuit_id": (int(circuit_id) if circuit_id is not None else None),
		"baseline_subset": baseline_subset,
		"layer_label": layer_label,
		"neuron_id": neuron_id,
		"max_effect": float(record.get("max_effect", 0.0)),
		"accuracy_gap": float(record.get("accuracy_gap", 0.0)),
		"accuracy_on_associated": float(record.get("acc_after_knockout_on_associated", 0.0)),
		"accuracy_on_unrelated": float(record.get("acc_after_knockout_on_unrelated", 0.0)),
	}

def _maybe_promote_catastrophic_candidate(buckets: dict, key: str):
	cand = buckets["catastrophic_zero"]["candidates"].get(key)
	if cand is None:
		return
	if int(cand.get("n_trials", 0)) < CATASTROPHIC_N_RUNS:
		return
	# Decide based on the first CATASTROPHIC_N_RUNS
	if int(cand.get("n_zero_zero", 0)) == int(cand.get("n_trials", 0)):
		buckets["catastrophic_zero"]["confirmed"][key] = cand
	else:
		buckets["catastrophic_zero"]["not_always"][key] = cand
	del buckets["catastrophic_zero"]["candidates"][key]

def _update_buckets_from_single_record(buckets, record, baseline_subset, circuit_id=None):
	"""Update special buckets from a single-neuron ablation record.

	Rules:
	- baseline_subset == 'positive' and accuracy_on_associated==0 and accuracy_on_unrelated==0 => catastrophic-zero candidate/confirmed/not_always
	- abs(max_effect) >= 0.1 and NOT catastrophic-zero => non-catastrophic agonist (excluded from future ablations)
	"""
	key = _extract_single_neuron_key_from_record(record)
	if key is None:
		return None

	# If already a known non-catastrophic agonist, keep it there.
	is_zero_zero = ((baseline_subset == "positive" and _is_catastrophic_zero_zero(record)) or (baseline_subset == "negative" and _is_catastrophic_one_one(record)))
	abs_gap = abs(float(record.get("max_effect", 0.0)))

	# Update catastrophic bucket
	if baseline_subset == "positive":
		if key in buckets["catastrophic_zero"]["confirmed"]:
			return key
		if key in buckets["catastrophic_zero"]["not_always"]:
			# We already decided it's not always zero-zero; nothing to do.
			return key
		if key in buckets["catastrophic_zero"]["candidates"] or is_zero_zero:
			cand = buckets["catastrophic_zero"]["candidates"].get(key, {
				"n_trials": 0,
				"n_zero_zero": 0,
				"history": [],
			})
			cand["n_trials"] = int(cand.get("n_trials", 0)) + 1
			if is_zero_zero:
				cand["n_zero_zero"] = int(cand.get("n_zero_zero", 0)) + 1
			cand["history"].append(_compact_single_record(record, circuit_id=circuit_id, baseline_subset=baseline_subset))
			buckets["catastrophic_zero"]["candidates"][key] = cand
			_maybe_promote_catastrophic_candidate(buckets, key)

	# Update non-catastrophic agonist bucket (exclude from future ablations)
	if (not is_zero_zero) and (abs_gap >= AGONIST_ABS_GAP_THRESHOLD):
		prev = buckets["non_catastrophic_agonists"].get(key)
		best = abs_gap if prev is None else max(float(prev.get("best_abs_gap", 0.0)), abs_gap)
		buckets["non_catastrophic_agonists"][key] = {
			"best_abs_gap": float(best),
			"last_record": _compact_single_record(record, circuit_id=circuit_id, baseline_subset=baseline_subset),
		}
		# Once it's a non-catastrophic agonist, stop spending time on catastrophic validation.
		buckets["catastrophic_zero"]["candidates"].pop(key, None)
		# buckets["catastrophic_zero"]["not_always"].pop(key, None)

	return key

def _excluded_neuron_keys(buckets, exclude_catastrophic_candidates = True, exclude_catastrophic_confirmed = True):
	"""
	Exclusions for the dichotomic search.
	We always exclude already-found non-catastrophic agonists.
	We optionally exclude catastrophic candidates/confirmed for speed/stability.
	NOTE: do NOT exclude `not_always` (those should still be processed).
	"""
	excluded = set(buckets["non_catastrophic_agonists"].keys())
	cat = buckets["catastrophic_zero"]
	if exclude_catastrophic_candidates:
		excluded |= set(cat["candidates"].keys())
	if exclude_catastrophic_confirmed:
		excluded |= set(cat["confirmed"].keys())
	return excluded

def _bucket_counts(buckets):
	cat = buckets["catastrophic_zero"]
	return {
		"catastrophic_zero_candidates": int(len(cat["candidates"])),
		"catastrophic_zero_confirmed": int(len(cat["confirmed"])),
		"catastrophic_zero_not_always": int(len(cat["not_always"])),
		"non_catastrophic_agonists": int(len(buckets["non_catastrophic_agonists"])),
	}


def _next_token_id_for_completion(tokenizer, prompt_text: str, completion_text: str):
	def _ids(text, add_special_tokens):
		try:
			return tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
		except Exception:
			return None

	for add_special_tokens in (True, False):
		prompt_ids = _ids(prompt_text, add_special_tokens)
		full_ids = _ids(f"{prompt_text}{completion_text}", add_special_tokens)
		if prompt_ids and full_ids and len(full_ids) > len(prompt_ids) and full_ids[:len(prompt_ids)] == prompt_ids:
			return int(full_ids[len(prompt_ids)])

	for candidate in (completion_text, f" {completion_text}"):
		ids = _ids(candidate, False)
		if ids:
			return int(ids[0])
	return None


def _arithmetic_margin_from_last_logits(task, prompt_batch, logits_last, tokenizer, prompt_col):
	if logits_last.ndim != 2:
		return None
	target_ids = []
	for row in prompt_batch:
		prompt_text = str(row.get(prompt_col, row.get(getattr(task, "DEFAULT_INPUT", "prompt"), "")))
		few_shot_sep = ';' if ';' in prompt_text else (',' if ',' in prompt_text else None)
		prompt_eval = prompt_text[prompt_text.rfind(few_shot_sep) + 1:] if few_shot_sep is not None else prompt_text
		try:
			correct_answer = str(eval(prompt_eval.replace('=', '')))
		except Exception:
			target_ids.append(-1)
			continue
		target_ids.append(_next_token_id_for_completion(tokenizer, prompt_text, correct_answer) or -1)
	target_ids = torch.tensor(target_ids, device=logits_last.device, dtype=torch.long)
	valid = target_ids >= 0
	if not bool(valid.any()):
		return None
	safe_ids = target_ids.clamp_min(0)
	target_logits = logits_last.gather(1, safe_ids.unsqueeze(1)).squeeze(1)
	masked = logits_last.clone()
	masked[torch.arange(masked.size(0), device=masked.device), safe_ids] = float('-inf')
	other_logits = masked.max(dim=1).values
	margin = target_logits - other_logits
	return torch.where(valid, margin, torch.zeros_like(margin))


def _safe_task_margin_from_last_logits(task, prompt_batch, logits_last, tokenizer, prompt_col):
	margin_fn = getattr(task, "margin_from_last_logits", None)
	if callable(margin_fn):
		margin = margin_fn(prompt_batch, logits_last, tokenizer)
		if margin is None:
			return None
		if not torch.is_tensor(margin):
			margin = torch.as_tensor(margin, device=logits_last.device, dtype=logits_last.dtype)
		margin = margin.reshape(-1).to(device=logits_last.device, dtype=logits_last.dtype)
		if margin.numel() != logits_last.shape[0]:
			raise ValueError(f"task.margin_from_last_logits returned {margin.numel()} values for batch size {logits_last.shape[0]}")
		return torch.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
	cls_name = getattr(task.__class__, "__name__", "").lower()
	module_name = getattr(task.__class__, "__module__", "").lower()
	if "arithmetic" in cls_name or "arithmetic" in module_name:
		return _arithmetic_margin_from_last_logits(task, prompt_batch, logits_last, tokenizer, prompt_col)
	return None



def _completion_text_from_row_for_saliency(task, row, prompt_col):
	"""Best-effort lookup for a textual target/completion in a dataset row.

	This is used only as a saliency-gradient fallback when the task does not
	provide a true margin. Boolean classification targets are intentionally
	ignored because their string form is usually not the LM completion target.
	"""
	candidate_keys = []
	for attr in ("DEFAULT_OUTPUT", "DEFAULT_OUTPUTS", "DEFAULT_ANSWER", "DEFAULT_ANSWERS"):
		val = getattr(task, attr, None)
		if isinstance(val, str):
			candidate_keys.append(val)
		elif isinstance(val, (list, tuple)):
			candidate_keys.extend(str(x) for x in val if isinstance(x, str))
	candidate_keys.extend([
		"answer", "answers", "completion", "target_text", "correct_answer",
		"output", "outputs", "label_text", "gold", "gold_answer",
	])
	seen = set()
	for key in candidate_keys:
		if key in seen or key == prompt_col or key not in row:
			continue
		seen.add(key)
		val = row.get(key)
		if isinstance(val, (list, tuple)) and val:
			val = val[0]
		if isinstance(val, (bool, np.bool_)) or val is None:
			continue
		try:
			if pd.isna(val):
				continue
		except Exception:
			pass
		text = str(val)
		if text:
			return text
	return None


def _saliency_objective_from_last_logits(task, prompt_batch, logits_last, tokenizer, prompt_col):
	"""Return a differentiable scalar-per-example objective for saliency.

	Priority:
	1. true task margin, if the task exposes one;
	2. logit of a cached textual completion/answer column, if we can infer it;
	3. model top next-token logit, with the argmax detached.

	The fallback objectives are not causal task margins, but they let Wanda/grad
	diagnostics run for older/non-margin tasks instead of silently skipping.
	"""
	margin = _safe_task_margin_from_last_logits(task, prompt_batch, logits_last, tokenizer, prompt_col)
	if margin is not None:
		return margin, "task_margin"

	device = logits_last.device
	batch_size = int(logits_last.shape[0])
	top_ids = logits_last.detach().argmax(dim=1).to(device=device, dtype=torch.long)
	target_ids = []
	used_cached = []
	for i, row in enumerate(prompt_batch):
		prompt_text = str(row.get(prompt_col, row.get(getattr(task, "DEFAULT_INPUT", "prompt"), "")))
		completion_text = _completion_text_from_row_for_saliency(task, row, prompt_col)
		tok_id = None
		if completion_text is not None:
			tok_id = _next_token_id_for_completion(tokenizer, prompt_text, str(completion_text))
		if tok_id is None:
			tok_id = int(top_ids[i].item())
			used_cached.append(False)
		else:
			used_cached.append(True)
		target_ids.append(int(tok_id))

	ids = torch.tensor(target_ids, device=device, dtype=torch.long).clamp(0, logits_last.shape[1] - 1)
	objective = logits_last.gather(1, ids.unsqueeze(1)).squeeze(1)
	objective = torch.nan_to_num(objective.reshape(batch_size), nan=0.0, posinf=0.0, neginf=0.0)
	if any(used_cached):
		name = "cached_completion_logit" if all(used_cached) else "cached_completion_logit+top_logit"
	else:
		name = "top_next_token_logit"
	return objective, name

def _intervention_baseline_values_for_unit(agonist, positions, intervention, mean_activations):
	positions = positions.to(torch.long).cpu()
	device = positions.device

	unit_id = int(agonist["unit_id"])
	spec = _activation_hook_spec(agonist["layer_label"])
	if spec is None:
		return None

	if intervention == "zero" or mean_activations is None:
		return torch.zeros(len(positions), dtype=torch.float32, device=device)

	if spec["layer_type"] == "mlp":
		unit_key = f"m{int(spec['layer_index'])}"
	else:
		unit_key = f"a{int(spec['layer_index'])}.h{int(spec['head_index'])}"

	idx_full = mean_activations.mean_idx.get(unit_key)
	if idx_full is None or idx_full.numel() == 0:
		return None

	idx_full = idx_full.detach().to(torch.long).cpu()
	query = idx_full.new_tensor([unit_id])
	loc = torch.searchsorted(idx_full, query)
	if loc.numel() == 0:
		return None

	loc = int(loc.item())
	if loc >= idx_full.numel() or int(idx_full[loc].item()) != unit_id:
		return None

	mean_global = mean_activations.mean_global.get(unit_key)
	if mean_global is None:
		return None

	global_value = float(mean_global.detach().to(torch.float32).cpu()[loc].item())

	if intervention in ("mean", "mean-donor"):
		return torch.full((len(positions),), global_value, dtype=torch.float32, device=device)

	if intervention in ("mean-positional", "mean-donor-positional"):
		mean_per_pos = (
			(mean_activations.mean_per_pos or {}).get(unit_key)
			if getattr(mean_activations, "mean_per_pos", None) is not None
			else None
		)
		if mean_per_pos is None:
			return torch.full((len(positions),), global_value, dtype=torch.float32, device=device)

		mean_per_pos = mean_per_pos.detach().to(torch.float32).cpu()
		vals = torch.full((len(positions),), global_value, dtype=torch.float32, device=device)
		for i, pos in enumerate(positions.tolist()):
			if 0 <= int(pos) < mean_per_pos.shape[0]:
				vals[i] = float(mean_per_pos[int(pos), loc].item())
		return vals

	return None


def _activation_hook_spec(layer_label: str):
	parsed = get_layer_type_and_ids(layer_label)
	if not parsed:
		return None
	layer_type, layer_idx, head_idx = parsed
	if layer_type == "mlp":
		return {
			"layer_type": "mlp",
			"layer_index": int(layer_idx),
			"head_index": None,
			"hook_name": f"blocks.{int(layer_idx)}.hook_mlp_out",
			"compare_pool": "all_mlp_units_in_layer",
		}
	if layer_type == "attn":
		return {
			"layer_type": "attn",
			"layer_index": int(layer_idx),
			"head_index": int(head_idx),
			"hook_name": f"blocks.{int(layer_idx)}.attn.hook_z",
			"compare_pool": "all_attention_dimensions_in_layer",
		}
	return None


def _extract_singleton_agonists_from_records(ablation_records, baseline_subset):
	agonists = []
	seen = set()
	for rec in ablation_records or []:
		key = _extract_single_neuron_key_from_record(rec)
		if key is None or key in seen:
			continue
		is_catastrophic = ((baseline_subset == "positive" and _is_catastrophic_zero_zero(rec)) or (baseline_subset == "negative" and _is_catastrophic_one_one(rec)))
		max_effect = float(rec.get("max_effect", 0.0))
		if is_catastrophic or abs(max_effect) < AGONIST_ABS_GAP_THRESHOLD:
			continue
		layers_neurons_dict = rec.get("layers_neurons_dict") or {}
		if len(layers_neurons_dict) != 1:
			continue
		layer_label = next(iter(layers_neurons_dict.keys()))
		try:
			unit_id = int(list(layers_neurons_dict[layer_label])[0])
		except Exception:
			continue
		agonists.append({
			"unit_key": key,
			"layer_label": str(layer_label),
			"unit_id": int(unit_id),
			"max_effect": max_effect,
			"accuracy_gap": float(rec.get("accuracy_gap", 0.0)),
			"acc_after_knockout_on_associated": float(rec.get("acc_after_knockout_on_associated", 0.0)),
			"acc_after_knockout_on_unrelated": float(rec.get("acc_after_knockout_on_unrelated", 0.0)),
		})
		seen.add(key)
	return agonists


def _collect_reference_activations(model, examples, prompt_col, layer_labels, batch_size, decode_only, max_new_tokens):
	if not examples or not layer_labels:
		return {}

	ordered_layer_labels = []
	for layer_label in layer_labels:
		if layer_label not in ordered_layer_labels:
			ordered_layer_labels.append(layer_label)

	specs = {}
	for layer_label in ordered_layer_labels:
		spec = _activation_hook_spec(layer_label)
		if spec is not None:
			specs[layer_label] = spec
	if not specs:
		return {}

	device = model.hooked_model.cfg.device
	collected = {layer_label: [] for layer_label in specs}

	with torch.inference_mode():
		for start in range(0, len(examples), batch_size):
			batch_examples = examples[start:start + batch_size]
			batch_prompts = [str(row[prompt_col]) for row in batch_examples]

			if decode_only:
				prefix = model.prefill_prefix_batch(
					batch_prompts,
					max_new_tokens=max(2, int(max_new_tokens)),
					use_kv_cache=True,
				)
				seen = {layer_label: False for layer_label in specs}

				def make_hook(layer_label):
					def _hook(act, hook):
						if seen[layer_label]:
							return act
						if act.ndim == 3:
							snapshot = act[:, -1, :]
						elif act.ndim == 4:
							snapshot = act[:, -1, :, :]
						else:
							return act
						collected[layer_label].append(snapshot.detach().to(torch.float32).cpu())
						seen[layer_label] = True
						return act
					return _hook

				fwd_hooks = [(spec["hook_name"], make_hook(layer_label)) for layer_label, spec in specs.items()]
				_ = model.generate_from_prefix_cache(
					prefix,
					fwd_hooks=fwd_hooks,
					stop_at_eos=False,
					clone_kv_cache_tensors=True,
				)
				del prefix
			else:
				input_ids, attention_mask, input_lengths = model.tokenize_with_mask(
					batch_prompts,
					device,
					padding=True,
					truncation=True,
					add_special_tokens=True,
					padding_side="right",
				)
				last_idx = (input_lengths - 1).to(device)

				def make_hook(layer_label, last_idx=last_idx):
					def _hook(act, hook):
						batch_idx = torch.arange(act.shape[0], device=act.device)
						if act.ndim == 3:
							snapshot = act[batch_idx, last_idx, :]
						elif act.ndim == 4:
							snapshot = act[batch_idx, last_idx, :, :]
						else:
							return act
						collected[layer_label].append(snapshot.detach().to(torch.float32).cpu())
						return act
					return _hook

				fwd_hooks = [(spec["hook_name"], make_hook(layer_label)) for layer_label, spec in specs.items()]
				with model.hooked_model.hooks(fwd_hooks=fwd_hooks, reset_hooks_end=True, clear_contexts=True):
					_ = model.hooked_model(
						input_ids,
						attention_mask=attention_mask,
						padding_side="right",
						return_type="residual",
						stop_at_layer=model.hooked_model.cfg.n_layers,
					)
			model.cleanup_after_generate()

	out = {}
	for layer_label, chunks in collected.items():
		if chunks:
			out[layer_label] = torch.cat(chunks, dim=0)
	return out


def _activation_rows_for_unit(circuit_id, split_name, agonist, full_layer_acts):
	spec = _activation_hook_spec(agonist["layer_label"])
	if spec is None or full_layer_acts is None:
		return [], None

	unit_id = int(agonist["unit_id"])
	if spec["layer_type"] == "mlp":
		if full_layer_acts.ndim != 2 or unit_id < 0 or unit_id >= full_layer_acts.shape[1]:
			return [], None
		pool = full_layer_acts
		unit_values = pool[:, unit_id]
	elif spec["layer_type"] == "attn":
		head_idx = int(spec["head_index"])
		if full_layer_acts.ndim != 3 or head_idx < 0 or head_idx >= full_layer_acts.shape[1] or unit_id < 0 or unit_id >= full_layer_acts.shape[2]:
			return [], None
		unit_values = full_layer_acts[:, head_idx, unit_id]
		pool = full_layer_acts.reshape(full_layer_acts.shape[0], -1)
	else:
		return [], None

	n_examples = int(pool.shape[0])
	n_comp = int(pool.shape[1])
	if n_examples <= 0 or n_comp <= 0:
		return [], None

	unit_values = unit_values.to(torch.float32)
	pool = pool.to(torch.float32)
	layer_mean = pool.mean(dim=1)
	layer_std = pool.std(dim=1, unbiased=False).clamp_min(1e-8)
	greater_count = (pool > unit_values.unsqueeze(1)).sum(dim=1)
	top5_cutoff = max(1, min(5, n_comp))
	top10pct_cutoff = max(1, int(np.ceil(0.10 * n_comp)))
	percentile_rank = 1.0 - (greater_count.to(torch.float32) / float(n_comp))
	z_score = (unit_values - layer_mean) / layer_std
	delta_from_layer_mean = unit_values - layer_mean

	raw_rows = []
	for i in range(n_examples):
		raw_rows.append({
			"circuit_id": int(circuit_id),
			"unit_key": agonist["unit_key"],
			"layer_label": agonist["layer_label"],
			"unit_id": int(unit_id),
			"split": split_name,
			"example_local_index": int(i),
			"activation_value": float(unit_values[i].item()),
			"layer_percentile_rank": float(percentile_rank[i].item()),
			"layer_zscore": float(z_score[i].item()),
			"delta_from_layer_mean": float(delta_from_layer_mean[i].item()),
			"n_compared_activations": int(n_comp),
			"is_top1": bool(greater_count[i].item() == 0),
			"is_top5": bool(greater_count[i].item() < top5_cutoff),
			"is_top10pct": bool(greater_count[i].item() < top10pct_cutoff),
			"max_effect": float(agonist["max_effect"]),
			"accuracy_gap": float(agonist["accuracy_gap"]),
			"abs_max_effect": float(abs(agonist["max_effect"])),
			"abs_accuracy_gap": float(abs(agonist["accuracy_gap"])),
			"acc_after_knockout_on_associated": float(agonist.get("acc_after_knockout_on_associated", float("nan"))),
			"acc_after_knockout_on_unrelated": float(agonist.get("acc_after_knockout_on_unrelated", float("nan"))),
		})

	summary_row = {
		"circuit_id": int(circuit_id),
		"unit_key": agonist["unit_key"],
		"layer_label": agonist["layer_label"],
		"unit_id": int(unit_id),
		"split": split_name,
		"n_examples": int(n_examples),
		"n_compared_activations": int(n_comp),
		"compare_pool": spec["compare_pool"],
		"max_effect": float(agonist["max_effect"]),
		"accuracy_gap": float(agonist["accuracy_gap"]),
		"mean_activation_value": float(unit_values.mean().item()),
		"median_activation_value": float(unit_values.median().item()),
		"mean_layer_percentile_rank": float(percentile_rank.mean().item()),
		"median_layer_percentile_rank": float(percentile_rank.median().item()),
		"mean_layer_zscore": float(z_score.mean().item()),
		"mean_delta_from_layer_mean": float(delta_from_layer_mean.mean().item()),
		"top1_rate": float((greater_count == 0).to(torch.float32).mean().item()),
		"top5_rate": float((greater_count < top5_cutoff).to(torch.float32).mean().item()),
		"top10pct_rate": float((greater_count < top10pct_cutoff).to(torch.float32).mean().item()),
	}
	return raw_rows, summary_row


def _aggregate_activation_stats(raw_rows):
	if not raw_rows:
		return {}
	raw_df = pd.DataFrame(raw_rows)
	stats = {}
	for split_name, split_df in raw_df.groupby("split", sort=False):
		stats[split_name] = {
			"n_measurements": int(len(split_df)),
			"n_unique_agonists": int(split_df["unit_key"].nunique()),
			"mean_activation_value": float(split_df["activation_value"].mean()),
			"median_activation_value": float(split_df["activation_value"].median()),
			"mean_layer_percentile_rank": float(split_df["layer_percentile_rank"].mean()),
			"median_layer_percentile_rank": float(split_df["layer_percentile_rank"].median()),
			"mean_layer_zscore": float(split_df["layer_zscore"].mean()),
			"mean_delta_from_layer_mean": float(split_df["delta_from_layer_mean"].mean()),
			"top1_rate": float(split_df["is_top1"].mean()),
			"top5_rate": float(split_df["is_top5"].mean()),
			"top10pct_rate": float(split_df["is_top10pct"].mean()),
		}
	return stats


def _top_activation_verdict(split_stats):
	if not split_stats:
		return "no agonist activation stats available"
	top1 = float(split_stats.get("top1_rate", 0.0))
	top10 = float(split_stats.get("top10pct_rate", 0.0))
	mean_pct = float(split_stats.get("mean_layer_percentile_rank", 0.0))
	if top1 >= 0.5:
		return "often the single highest activation"
	if top10 >= 0.5 or mean_pct >= 0.9:
		return "usually high, but not usually the single highest activation"
	return "usually not among the highest activations"


def _plot_agonist_activation_stats(raw_df, out_path, title):
	if raw_df.empty:
		return

	split_order = [split for split in ("associated", "unrelated") if split in set(raw_df["split"].tolist())]
	fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

	box_data = [raw_df.loc[raw_df["split"] == split, "layer_percentile_rank"].to_numpy() for split in split_order]
	if box_data:
		axes[0].boxplot(box_data, tick_labels=split_order, showmeans=True)
		axes[0].set_ylim(0.0, 1.02)
		axes[0].set_ylabel("Within-layer percentile rank")
		axes[0].set_title("Agonist rank among layer activations")
		axes[0].grid(True, axis="y", alpha=0.3)

	metrics = ["is_top1", "is_top5", "is_top10pct"]
	metric_labels = ["top1", "top5", "top10%"]
	x = np.arange(len(metric_labels))
	width = 0.35 if len(split_order) > 1 else 0.5
	for idx, split in enumerate(split_order):
		split_df = raw_df.loc[raw_df["split"] == split]
		values = [float(split_df[m].mean()) for m in metrics]
		offset = (idx - (len(split_order) - 1) / 2.0) * width
		axes[1].bar(x + offset, values, width=width, label=split)
	axes[1].set_xticks(x)
	axes[1].set_xticklabels(metric_labels)
	axes[1].set_ylim(0.0, 1.02)
	axes[1].set_ylabel("Rate")
	axes[1].set_title("How often is an agonist activation near the top?")
	axes[1].grid(True, axis="y", alpha=0.3)
	if len(split_order) > 1:
		axes[1].legend()

	fig.suptitle(title)
	fig.tight_layout()
	fig.savefig(out_path, dpi=180, bbox_inches="tight")
	plt.close(fig)


def compute_and_save_agonist_activation_stats(
	model,
	rule_json_path,
	*,
	circuit_id,
	circuit_label,
	prompt_col,
	associated_examples,
	unrelated_examples,
	ablation_records,
	baseline_subset,
	batch_size,
	decode_only,
	max_new_tokens,
):
	agonists = _extract_singleton_agonists_from_records(ablation_records, baseline_subset)
	if not agonists:
		return None

	layer_labels = [agonist["layer_label"] for agonist in agonists]
	associated_acts = _collect_reference_activations(
		model,
		associated_examples,
		prompt_col,
		layer_labels,
		batch_size=batch_size,
		decode_only=decode_only,
		max_new_tokens=max_new_tokens,
	)
	unrelated_acts = _collect_reference_activations(
		model,
		unrelated_examples,
		prompt_col,
		layer_labels,
		batch_size=batch_size,
		decode_only=decode_only,
		max_new_tokens=max_new_tokens,
	)

	raw_rows = []
	summary_rows = []
	for agonist in agonists:
		assoc_rows, assoc_summary = _activation_rows_for_unit(
			circuit_id,
			"associated",
			agonist,
			associated_acts.get(agonist["layer_label"]),
		)
		if assoc_summary is not None:
			raw_rows.extend(assoc_rows)
			summary_rows.append(assoc_summary)
		unrel_rows, unrel_summary = _activation_rows_for_unit(
			circuit_id,
			"unrelated",
			agonist,
			unrelated_acts.get(agonist["layer_label"]),
		)
		if unrel_summary is not None:
			raw_rows.extend(unrel_rows)
			summary_rows.append(unrel_summary)

	if not raw_rows:
		return None

	raw_df = pd.DataFrame(raw_rows)
	summary_df = pd.DataFrame(summary_rows)
	aggregate_stats = _aggregate_activation_stats(raw_rows)

	base_name = rule_json_path.stem
	raw_csv_path = rule_json_path.parent / f"{base_name}_agonist_activation_raw.csv"
	summary_csv_path = rule_json_path.parent / f"{base_name}_agonist_activation_summary.csv"
	plot_path = rule_json_path.parent / f"{base_name}_agonist_activation_stats.png"
	raw_df.to_csv(raw_csv_path, index=False)
	summary_df.to_csv(summary_csv_path, index=False)
	_plot_agonist_activation_stats(
		raw_df,
		plot_path,
		title=f"Circuit {int(circuit_id)} agonist activation diagnostics",
	)

	measurement_position = "first_hooked_decode_step" if decode_only else "last_prompt_token"
	output = {
		"n_agonists": int(len(agonists)),
		"measurement_position": measurement_position,
		"compare_pool": "all activations in the same hooked layer",
		"summary_rows": summary_rows,
		"aggregate_stats": aggregate_stats,
		"files": {
			"raw_csv": raw_csv_path.name,
			"summary_csv": summary_csv_path.name,
			"plot_png": plot_path.name,
		},
		"question_answer": {
			"associated": _top_activation_verdict(aggregate_stats.get("associated")),
			"unrelated": _top_activation_verdict(aggregate_stats.get("unrelated")),
		},
	}
	return output, raw_rows, summary_rows



def _margin_rows_for_unit(circuit_id, split_name, agonist, layer_payload, *, intervention, mean_activations):
	spec = _activation_hook_spec(agonist["layer_label"])
	if spec is None or layer_payload is None:
		return [], None
	full_layer_acts = layer_payload.get("activations")
	full_layer_grads = layer_payload.get("grads")
	positions = layer_payload.get("positions")
	prompt_margin = layer_payload.get("prompt_margin")
	if full_layer_acts is None or full_layer_grads is None or positions is None or prompt_margin is None:
		return [], None
	unit_id = int(agonist["unit_id"])
	if spec["layer_type"] == "mlp":
		if full_layer_acts.ndim != 2 or full_layer_grads.ndim != 2 or unit_id < 0 or unit_id >= full_layer_acts.shape[1]:
			return [], None
		unit_values = full_layer_acts[:, unit_id]
		grad_values = full_layer_grads[:, unit_id]
	elif spec["layer_type"] == "attn":
		head_idx = int(spec["head_index"])
		if full_layer_acts.ndim != 3 or full_layer_grads.ndim != 3 or head_idx < 0 or head_idx >= full_layer_acts.shape[1] or unit_id < 0 or unit_id >= full_layer_acts.shape[2]:
			return [], None
		unit_values = full_layer_acts[:, head_idx, unit_id]
		grad_values = full_layer_grads[:, head_idx, unit_id]
	else:
		return [], None
	baseline_values = _intervention_baseline_values_for_unit(agonist, positions, intervention, mean_activations)
	if baseline_values is None:
		return [], None
	unit_values = unit_values.to(torch.float32)
	device = unit_values.device
	grad_values = grad_values.to(device=device, dtype=torch.float32)
	baseline_values = baseline_values.to(device=device, dtype=torch.float32)
	prompt_margin = prompt_margin.to(device=device, dtype=torch.float32)
	positions = positions.to(device=device)
	
	delta_to_baseline = baseline_values - unit_values
	predicted_margin_shift = grad_values * delta_to_baseline
	predicted_margin_drop = -predicted_margin_shift
	valid_mask = torch.isfinite(predicted_margin_drop) & torch.isfinite(prompt_margin)
	if not bool(valid_mask.any()):
		return [], None
	unit_values = unit_values[valid_mask]
	grad_values = grad_values[valid_mask]
	baseline_values = baseline_values[valid_mask]
	delta_to_baseline = delta_to_baseline[valid_mask]
	predicted_margin_shift = predicted_margin_shift[valid_mask]
	predicted_margin_drop = predicted_margin_drop[valid_mask]
	prompt_margin = prompt_margin[valid_mask]
	positions = positions[valid_mask]
	n_examples = int(unit_values.shape[0])
	if n_examples <= 0:
		return [], None
	raw_rows = []
	for i in range(n_examples):
		raw_rows.append({
			"circuit_id": int(circuit_id),
			"unit_key": agonist["unit_key"],
			"layer_label": agonist["layer_label"],
			"unit_id": int(unit_id),
			"split": split_name,
			"example_local_index": int(i),
			"hook_position": int(positions[i].item()),
			"prompt_margin": float(prompt_margin[i].item()),
			"activation_value": float(unit_values[i].item()),
			"gradient_value": float(grad_values[i].item()),
			"baseline_value": float(baseline_values[i].item()),
			"delta_to_baseline": float(delta_to_baseline[i].item()),
			"predicted_margin_shift": float(predicted_margin_shift[i].item()),
			"predicted_margin_drop": float(predicted_margin_drop[i].item()),
			"is_positive_margin_drop": bool(predicted_margin_drop[i].item() > 0),
			"max_effect": float(agonist["max_effect"]),
			"accuracy_gap": float(agonist["accuracy_gap"]),
			"abs_max_effect": float(abs(agonist["max_effect"])),
			"abs_accuracy_gap": float(abs(agonist["accuracy_gap"])),
			"acc_after_knockout_on_associated": float(agonist.get("acc_after_knockout_on_associated", float("nan"))),
			"acc_after_knockout_on_unrelated": float(agonist.get("acc_after_knockout_on_unrelated", float("nan"))),
		})
	summary_row = {
		"circuit_id": int(circuit_id),
		"unit_key": agonist["unit_key"],
		"layer_label": agonist["layer_label"],
		"unit_id": int(unit_id),
		"split": split_name,
		"n_examples": int(n_examples),
		"max_effect": float(agonist["max_effect"]),
		"accuracy_gap": float(agonist["accuracy_gap"]),
		"mean_prompt_margin": float(prompt_margin.mean().item()),
		"median_prompt_margin": float(prompt_margin.median().item()),
		"mean_gradient_value": float(grad_values.mean().item()),
		"mean_abs_gradient_value": float(grad_values.abs().mean().item()),
		"mean_delta_to_baseline": float(delta_to_baseline.mean().item()),
		"mean_predicted_margin_shift": float(predicted_margin_shift.mean().item()),
		"median_predicted_margin_shift": float(predicted_margin_shift.median().item()),
		"mean_predicted_margin_drop": float(predicted_margin_drop.mean().item()),
		"median_predicted_margin_drop": float(predicted_margin_drop.median().item()),
		"positive_margin_drop_rate": float((predicted_margin_drop > 0).to(torch.float32).mean().item()),
	}
	return raw_rows, summary_row


def _aggregate_margin_stats(raw_rows):
	if not raw_rows:
		return {}
	raw_df = pd.DataFrame(raw_rows)
	stats = {}
	for split_name, split_df in raw_df.groupby("split", sort=False):
		stats[split_name] = {
			"n_measurements": int(len(split_df)),
			"n_unique_agonists": int(split_df["unit_key"].nunique()),
			"mean_prompt_margin": float(split_df["prompt_margin"].mean()),
			"median_prompt_margin": float(split_df["prompt_margin"].median()),
			"mean_gradient_value": float(split_df["gradient_value"].mean()),
			"mean_abs_gradient_value": float(split_df["gradient_value"].abs().mean()),
			"mean_delta_to_baseline": float(split_df["delta_to_baseline"].mean()),
			"mean_predicted_margin_shift": float(split_df["predicted_margin_shift"].mean()),
			"median_predicted_margin_shift": float(split_df["predicted_margin_shift"].median()),
			"mean_predicted_margin_drop": float(split_df["predicted_margin_drop"].mean()),
			"median_predicted_margin_drop": float(split_df["predicted_margin_drop"].median()),
			"positive_margin_drop_rate": float(split_df["is_positive_margin_drop"].mean()),
		}
	return stats


def _margin_effect_verdict(split_stats):
	if not split_stats:
		return "no agonist margin stats available"
	mean_drop = float(split_stats.get("mean_predicted_margin_drop", 0.0))
	positive_rate = float(split_stats.get("positive_margin_drop_rate", 0.0))
	if mean_drop > 0 and positive_rate >= 0.75:
		return "usually predicted to reduce the decision margin when ablated"
	if mean_drop > 0 and positive_rate >= 0.5:
		return "often predicted to reduce the decision margin when ablated"
	return "not consistently predicted to reduce the decision margin when ablated"


def _plot_agonist_margin_stats(raw_df, out_path, title):
	if raw_df.empty:
		return
	split_order = [split for split in ("associated", "unrelated") if split in set(raw_df["split"].tolist())]
	fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
	box_data = [raw_df.loc[raw_df["split"] == split, "predicted_margin_drop"].to_numpy() for split in split_order]
	if box_data:
		axes[0].boxplot(box_data, tick_labels=split_order, showmeans=True)
		axes[0].set_ylabel("Predicted margin drop")
		axes[0].set_title("Singleton agonist first-order effect")
		axes[0].grid(True, axis="y", alpha=0.3)
	x = np.arange(len(split_order))
	positive_rates = [float(raw_df.loc[raw_df["split"] == split, "is_positive_margin_drop"].mean()) for split in split_order]
	mean_drops = [float(raw_df.loc[raw_df["split"] == split, "predicted_margin_drop"].mean()) for split in split_order]
	axes[1].bar(x, positive_rates, width=0.45, label="positive_drop_rate")
	axes[1].plot(x, mean_drops, marker="o", linestyle="--", label="mean_drop")
	axes[1].set_xticks(x)
	axes[1].set_xticklabels(split_order)
	axes[1].set_ylabel("Rate / mean effect")
	axes[1].set_title("How often is the predicted effect harmful?")
	axes[1].grid(True, axis="y", alpha=0.3)
	axes[1].legend()
	fig.suptitle(title)
	fig.tight_layout()
	fig.savefig(out_path, dpi=180, bbox_inches="tight")
	plt.close(fig)


def compute_and_save_agonist_margin_stats(model, task, rule_json_path, *, circuit_id, circuit_label, prompt_col, associated_examples, unrelated_examples, ablation_records, baseline_subset, batch_size, intervention, mean_activations):
	agonists = _extract_singleton_agonists_from_records(ablation_records, baseline_subset)
	if not agonists:
		return None
	layer_labels = [agonist["layer_label"] for agonist in agonists]
	associated_payload = collect_reference_margin_tensors(model, task, associated_examples, prompt_col, layer_labels, batch_size=batch_size)
	if associated_payload is None:
		return None
	unrelated_payload = collect_reference_margin_tensors(model, task, unrelated_examples, prompt_col, layer_labels, batch_size=batch_size)
	if unrelated_payload is None:
		return None
	raw_rows = []
	summary_rows = []
	for agonist in agonists:
		assoc_rows, assoc_summary = _margin_rows_for_unit(circuit_id, "associated", agonist, associated_payload.get(agonist["layer_label"]), intervention=intervention, mean_activations=mean_activations)
		if assoc_summary is not None:
			raw_rows.extend(assoc_rows)
			summary_rows.append(assoc_summary)
		unrel_rows, unrel_summary = _margin_rows_for_unit(circuit_id, "unrelated", agonist, unrelated_payload.get(agonist["layer_label"]), intervention=intervention, mean_activations=mean_activations)
		if unrel_summary is not None:
			raw_rows.extend(unrel_rows)
			summary_rows.append(unrel_summary)
	if not raw_rows:
		return None
	raw_df = pd.DataFrame(raw_rows)
	summary_df = pd.DataFrame(summary_rows)
	aggregate_stats = _aggregate_margin_stats(raw_rows)
	base_name = rule_json_path.stem
	raw_csv_path = rule_json_path.parent / f"{base_name}_agonist_margin_raw.csv"
	summary_csv_path = rule_json_path.parent / f"{base_name}_agonist_margin_summary.csv"
	plot_path = rule_json_path.parent / f"{base_name}_agonist_margin_stats.png"
	raw_df.to_csv(raw_csv_path, index=False)
	summary_df.to_csv(summary_csv_path, index=False)
	_plot_agonist_margin_stats(raw_df, plot_path, title=f"Circuit {int(circuit_id)} agonist margin diagnostics")
	output = {
		"n_agonists": int(len(agonists)),
		"measurement_position": "last_prompt_token",
		"metric": "first_order_margin_drop = -(d margin / d hook_value) * (baseline - current)",
		"summary_rows": summary_rows,
		"aggregate_stats": aggregate_stats,
		"files": {
			"raw_csv": raw_csv_path.name,
			"summary_csv": summary_csv_path.name,
			"plot_png": plot_path.name,
		},
		"question_answer": {
			"associated": _margin_effect_verdict(aggregate_stats.get("associated")),
			"unrelated": _margin_effect_verdict(aggregate_stats.get("unrelated")),
		},
	}
	return output, raw_rows, summary_rows




def _parse_saliency_metrics(metrics):
	if metrics is None:
		metrics = ("wanda", "gradient", "activation_x_gradient")
	if isinstance(metrics, str):
		parts = [p.strip() for p in metrics.split(",")]
	else:
		parts = [str(p).strip() for p in metrics]
	aliases = {
		"wanda": "wanda",
		"gradient": "gradient",
		"grad": "gradient",
		"abs_gradient": "gradient",
		"activation_x_gradient": "activation_x_gradient",
		"activation_gradient": "activation_x_gradient",
		"activation*gradient": "activation_x_gradient",
		"activation × gradient": "activation_x_gradient",
		"actgrad": "activation_x_gradient",
		"act_grad": "activation_x_gradient",
	}
	out = []
	for part in parts:
		if not part:
			continue
		key = part.lower()
		metric = aliases.get(key)
		if metric is None:
			raise ValueError(
				f"Unsupported agonist saliency metric {part!r}. "
				"Use one of: wanda, gradient, activation_x_gradient."
			)
		if metric not in out:
			out.append(metric)
	return out


def _flatten_hook_values_for_spec(spec, full_layer_values):
	if spec is None or full_layer_values is None:
		return None, None, None
	if spec["layer_type"] == "mlp":
		if full_layer_values.ndim != 2:
			return None, None, None
		return full_layer_values.reshape(full_layer_values.shape[0], -1), int(full_layer_values.shape[1]), None
	if spec["layer_type"] == "attn":
		if full_layer_values.ndim != 3:
			return None, None, None
		n_heads = int(full_layer_values.shape[1])
		d_head = int(full_layer_values.shape[2])
		return full_layer_values.reshape(full_layer_values.shape[0], -1), d_head, n_heads
	return None, None, None


def _flat_index_for_agonist(spec, unit_id, d_head):
	unit_id = int(unit_id)
	if spec["layer_type"] == "mlp":
		return unit_id
	if spec["layer_type"] == "attn":
		return int(spec["head_index"]) * int(d_head) + unit_id
	return None


def _wanda_weight_norms_for_spec(model, spec, n_compared_units):
	if spec is None:
		return None
	if spec["layer_type"] == "attn":
		try:
			layer_idx = int(spec["layer_index"])
			w_o = model.hooked_model.blocks[layer_idx].attn.W_O.detach().to(torch.float32).cpu()
			norms = w_o.norm(dim=-1).reshape(-1)
			if int(norms.numel()) == int(n_compared_units):
				return norms
		except Exception:
			return None
		return None
	if spec["layer_type"] == "mlp":
		# These script-6 MLP coordinates are blocks.L.hook_mlp_out residual-stream
		# coordinates, not hidden MLP neurons. There is no outgoing W_out per hooked
		# coordinate here, so Wanda reduces to |activation| at this hook site.
		return torch.ones(int(n_compared_units), dtype=torch.float32)
	return None


def _score_pool_for_saliency_metric(metric, acts_pool, grads_pool, *, model=None, spec=None):
	metric = _parse_saliency_metrics([metric])[0]
	acts_pool = acts_pool.to(torch.float32)
	grads_pool = grads_pool.to(torch.float32)
	if metric == "gradient":
		return grads_pool.abs(), "|gradient|"
	if metric == "activation_x_gradient":
		return (acts_pool * grads_pool).abs(), "|activation × gradient|"
	if metric == "wanda":
		weight_norms = _wanda_weight_norms_for_spec(model, spec, acts_pool.shape[1])
		if weight_norms is None:
			weight_norms = torch.ones(int(acts_pool.shape[1]), dtype=torch.float32)
		weight_norms = weight_norms.to(device=acts_pool.device, dtype=torch.float32).view(1, -1)
		return acts_pool.abs() * weight_norms, "|activation| × ||outgoing weight||"
	raise ValueError(f"Unknown saliency metric: {metric}")


def _saliency_rows_for_unit(circuit_id, split_name, agonist, layer_payload, *, model, metric):
	spec = _activation_hook_spec(agonist["layer_label"])
	if spec is None or layer_payload is None:
		return [], None

	full_layer_acts = layer_payload.get("activations")
	full_layer_grads = layer_payload.get("grads")
	positions = layer_payload.get("positions")
	prompt_margin = layer_payload.get("prompt_margin")
	if full_layer_acts is None or full_layer_grads is None:
		return [], None

	acts_pool, d_head, _ = _flatten_hook_values_for_spec(spec, full_layer_acts)
	grads_pool, _, _ = _flatten_hook_values_for_spec(spec, full_layer_grads)
	if acts_pool is None or grads_pool is None or acts_pool.shape != grads_pool.shape:
		return [], None

	unit_id = int(agonist["unit_id"])
	flat_unit_index = _flat_index_for_agonist(spec, unit_id, d_head)
	if flat_unit_index is None or flat_unit_index < 0 or flat_unit_index >= acts_pool.shape[1]:
		return [], None

	score_pool, metric_formula = _score_pool_for_saliency_metric(metric, acts_pool, grads_pool, model=model, spec=spec)
	unit_scores = score_pool[:, flat_unit_index]
	unit_acts = acts_pool[:, flat_unit_index]
	unit_grads = grads_pool[:, flat_unit_index]

	n_examples = int(score_pool.shape[0])
	n_comp = int(score_pool.shape[1])
	if n_examples <= 0 or n_comp <= 0:
		return [], None

	if positions is None:
		positions = torch.full((n_examples,), -1, dtype=torch.long)
	else:
		positions = positions.to(torch.long)
	if prompt_margin is None:
		prompt_margin = torch.full((n_examples,), float("nan"), dtype=torch.float32)
	else:
		prompt_margin = prompt_margin.to(torch.float32)

	layer_mean = score_pool.mean(dim=1)
	layer_std = score_pool.std(dim=1, unbiased=False).clamp_min(1e-8)
	greater_count = (score_pool > unit_scores.unsqueeze(1)).sum(dim=1)
	top5_cutoff = max(1, min(5, n_comp))
	top10pct_cutoff = max(1, int(np.ceil(0.10 * n_comp)))
	percentile_rank = 1.0 - (greater_count.to(torch.float32) / float(n_comp))
	z_score = (unit_scores - layer_mean) / layer_std
	delta_from_layer_mean = unit_scores - layer_mean

	raw_rows = []
	for i in range(n_examples):
		raw_rows.append({
			"circuit_id": int(circuit_id),
			"unit_key": agonist["unit_key"],
			"layer_label": agonist["layer_label"],
			"unit_id": int(unit_id),
			"split": split_name,
			"example_local_index": int(i),
			"hook_position": int(positions[i].item()),
			"metric": metric,
			"metric_formula": metric_formula,
			"saliency_value": float(unit_scores[i].item()),
			"activation_value": float(unit_acts[i].item()),
			"gradient_value": float(unit_grads[i].item()),
			"prompt_margin": float(prompt_margin[i].item()),
			"layer_percentile_rank": float(percentile_rank[i].item()),
			"layer_zscore": float(z_score[i].item()),
			"delta_from_layer_mean": float(delta_from_layer_mean[i].item()),
			"n_compared_units": int(n_comp),
			"is_top1": bool(greater_count[i].item() == 0),
			"is_top5": bool(greater_count[i].item() < top5_cutoff),
			"is_top10pct": bool(greater_count[i].item() < top10pct_cutoff),
			"max_effect": float(agonist["max_effect"]),
			"accuracy_gap": float(agonist["accuracy_gap"]),
		})

	summary_row = {
		"circuit_id": int(circuit_id),
		"unit_key": agonist["unit_key"],
		"layer_label": agonist["layer_label"],
		"unit_id": int(unit_id),
		"split": split_name,
		"metric": metric,
		"metric_formula": metric_formula,
		"n_examples": int(n_examples),
		"n_compared_units": int(n_comp),
		"compare_pool": spec["compare_pool"],
		"max_effect": float(agonist["max_effect"]),
		"accuracy_gap": float(agonist["accuracy_gap"]),
		"abs_max_effect": float(abs(agonist["max_effect"])),
		"abs_accuracy_gap": float(abs(agonist["accuracy_gap"])),
		"acc_after_knockout_on_associated": float(agonist.get("acc_after_knockout_on_associated", float("nan"))),
		"acc_after_knockout_on_unrelated": float(agonist.get("acc_after_knockout_on_unrelated", float("nan"))),
		"gradient_objective": str(layer_payload.get("gradient_objective", "task_margin")),
		"mean_saliency_value": float(unit_scores.mean().item()),
		"median_saliency_value": float(unit_scores.median().item()),
		"mean_activation_value": float(unit_acts.mean().item()),
		"mean_abs_activation_value": float(unit_acts.abs().mean().item()),
		"mean_gradient_value": float(unit_grads.mean().item()),
		"mean_abs_gradient_value": float(unit_grads.abs().mean().item()),
		"mean_layer_percentile_rank": float(percentile_rank.mean().item()),
		"median_layer_percentile_rank": float(percentile_rank.median().item()),
		"mean_layer_zscore": float(z_score.mean().item()),
		"mean_delta_from_layer_mean": float(delta_from_layer_mean.mean().item()),
		"top1_rate": float((greater_count == 0).to(torch.float32).mean().item()),
		"top5_rate": float((greater_count < top5_cutoff).to(torch.float32).mean().item()),
		"top10pct_rate": float((greater_count < top10pct_cutoff).to(torch.float32).mean().item()),
	}
	return raw_rows, summary_row


def _aggregate_saliency_stats(raw_rows):
	if not raw_rows:
		return {}
	raw_df = pd.DataFrame(raw_rows)
	stats = {}
	for metric, metric_df in raw_df.groupby("metric", sort=False):
		metric_stats = {}
		for split_name, split_df in metric_df.groupby("split", sort=False):
			metric_stats[split_name] = {
				"n_measurements": int(len(split_df)),
				"n_unique_agonists": int(split_df["unit_key"].nunique()),
				"mean_saliency_value": float(split_df["saliency_value"].mean()),
				"median_saliency_value": float(split_df["saliency_value"].median()),
				"mean_layer_percentile_rank": float(split_df["layer_percentile_rank"].mean()),
				"median_layer_percentile_rank": float(split_df["layer_percentile_rank"].median()),
				"mean_layer_zscore": float(split_df["layer_zscore"].mean()),
				"mean_delta_from_layer_mean": float(split_df["delta_from_layer_mean"].mean()),
				"top1_rate": float(split_df["is_top1"].mean()),
				"top5_rate": float(split_df["is_top5"].mean()),
				"top10pct_rate": float(split_df["is_top10pct"].mean()),
			}
		stats[str(metric)] = metric_stats
	return stats



def _safe_numeric_corr(x, y, method):
	x = pd.to_numeric(pd.Series(x), errors="coerce")
	y = pd.to_numeric(pd.Series(y), errors="coerce")
	mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
	if int(mask.sum()) < 3:
		return None
	xv = x[mask]
	yv = y[mask]
	if xv.nunique(dropna=True) < 2 or yv.nunique(dropna=True) < 2:
		return None
	val = xv.corr(yv, method=method)
	if val is None or not np.isfinite(float(val)):
		return None
	return float(val)


def _saliency_ablation_correlation_rows(summary_rows):
	"""Correlate one row per agonist/metric/split with ablation scores.

	This intentionally uses summary rows, not per-prompt raw rows, so units with
	more examples do not get overweighted.
	"""
	if not summary_rows:
		return []
	df = pd.DataFrame(summary_rows)
	if df.empty or "metric" not in df.columns:
		return []
	for col in ("max_effect", "accuracy_gap"):
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")
	if "max_effect" in df.columns and "abs_max_effect" not in df.columns:
		df["abs_max_effect"] = df["max_effect"].abs()
	if "accuracy_gap" in df.columns and "abs_accuracy_gap" not in df.columns:
		df["abs_accuracy_gap"] = df["accuracy_gap"].abs()

	features = [
		"mean_saliency_value", "median_saliency_value",
		"mean_layer_percentile_rank", "median_layer_percentile_rank",
		"mean_layer_zscore", "mean_delta_from_layer_mean",
		"top1_rate", "top5_rate", "top10pct_rate",
		"mean_activation_value", "mean_abs_activation_value",
		"mean_gradient_value", "mean_abs_gradient_value",
	]
	targets = [c for c in ("max_effect", "abs_max_effect", "accuracy_gap", "abs_accuracy_gap") if c in df.columns]
	rows = []
	for metric, metric_df in df.groupby("metric", sort=False):
		for split, split_df in metric_df.groupby("split", sort=False):
			for feature in features:
				if feature not in split_df.columns:
					continue
				for target in targets:
					pearson = _safe_numeric_corr(split_df[feature], split_df[target], "pearson")
					spearman = _safe_numeric_corr(split_df[feature], split_df[target], "spearman")
					if pearson is None and spearman is None:
						continue
					rows.append({
						"scope": str(split),
						"metric": str(metric),
						"feature": str(feature),
						"target": str(target),
						"n": int(len(split_df)),
						"pearson": pearson,
						"spearman": spearman,
					})

	key_cols = [c for c in ("circuit_id", "unit_key", "layer_label", "unit_id", "metric") if c in df.columns]
	if set(["split", "metric"]).issubset(df.columns) and key_cols:
		for metric, metric_df in df.groupby("metric", sort=False):
			assoc = metric_df.loc[metric_df["split"] == "associated"].copy()
			unrel = metric_df.loc[metric_df["split"] == "unrelated"].copy()
			if assoc.empty or unrel.empty:
				continue
			merge_cols = [c for c in key_cols if c != "metric"] + ["metric"]
			merged = assoc.merge(unrel, on=merge_cols, suffixes=("_associated", "_unrelated"))
			if merged.empty:
				continue
			for feature in features:
				ca = f"{feature}_associated"
				cu = f"{feature}_unrelated"
				if ca not in merged.columns or cu not in merged.columns:
					continue
				delta_feature = f"delta_{feature}"
				merged[delta_feature] = pd.to_numeric(merged[ca], errors="coerce") - pd.to_numeric(merged[cu], errors="coerce")
				for target_base in targets:
					target_col = f"{target_base}_associated" if f"{target_base}_associated" in merged.columns else target_base
					if target_col not in merged.columns:
						continue
					pearson = _safe_numeric_corr(merged[delta_feature], merged[target_col], "pearson")
					spearman = _safe_numeric_corr(merged[delta_feature], merged[target_col], "spearman")
					if pearson is None and spearman is None:
						continue
					rows.append({
						"scope": "associated_minus_unrelated",
						"metric": str(metric),
						"feature": str(delta_feature),
						"target": str(target_base),
						"n": int(len(merged)),
						"pearson": pearson,
						"spearman": spearman,
					})
	return rows

def _top_saliency_verdict(split_stats):
	if not split_stats:
		return "no agonist saliency stats available"
	top1 = float(split_stats.get("top1_rate", 0.0))
	top10 = float(split_stats.get("top10pct_rate", 0.0))
	mean_pct = float(split_stats.get("mean_layer_percentile_rank", 0.0))
	if top1 >= 0.5:
		return "often the single highest-scoring unit"
	if top10 >= 0.5 or mean_pct >= 0.9:
		return "usually high-scoring, but not usually the single highest-scoring unit"
	return "usually not among the highest-scoring units"


def _plot_agonist_saliency_stats(raw_df, out_path, title):
	if raw_df.empty:
		return
	metric_order = [m for m in ("wanda", "gradient", "activation_x_gradient") if m in set(raw_df["metric"].tolist())]
	for metric in raw_df["metric"].tolist():
		if metric not in metric_order:
			metric_order.append(metric)
	split_order = [split for split in ("associated", "unrelated") if split in set(raw_df["split"].tolist())]
	if not metric_order or not split_order:
		return

	fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
	box_data = []
	box_labels = []
	for metric in metric_order:
		for split in split_order:
			vals = raw_df.loc[(raw_df["metric"] == metric) & (raw_df["split"] == split), "layer_percentile_rank"].to_numpy()
			if vals.size:
				box_data.append(vals)
				box_labels.append(f"{metric}\n{split}")
	if box_data:
		axes[0].boxplot(box_data, tick_labels=box_labels, showmeans=True)
		axes[0].set_ylim(0.0, 1.02)
		axes[0].set_ylabel("Layer percentile rank")
		axes[0].set_title("Agonist saliency rank among layer units")
		axes[0].grid(True, axis="y", alpha=0.3)

	x = np.arange(len(metric_order))
	width = 0.8 / max(1, len(split_order))
	for j, split in enumerate(split_order):
		values = []
		for metric in metric_order:
			sub = raw_df.loc[(raw_df["metric"] == metric) & (raw_df["split"] == split)]
			values.append(float(sub["is_top10pct"].mean()) if len(sub) else 0.0)
		offset = (j - (len(split_order) - 1) / 2.0) * width
		axes[1].bar(x + offset, values, width=width, label=split)
	axes[1].set_xticks(x)
	axes[1].set_xticklabels(metric_order, rotation=20, ha="right")
	axes[1].set_ylim(0.0, 1.02)
	axes[1].set_ylabel("Top-10% rate")
	axes[1].set_title("How often is an agonist near the top?")
	axes[1].grid(True, axis="y", alpha=0.3)
	if len(split_order) > 1:
		axes[1].legend()
	fig.suptitle(title)
	fig.tight_layout()
	fig.savefig(out_path, dpi=180, bbox_inches="tight")
	plt.close(fig)


def compute_and_save_agonist_saliency_stats(model, task, rule_json_path, *, circuit_id, circuit_label, prompt_col, associated_examples, unrelated_examples, ablation_records, baseline_subset, batch_size, metrics):
	metrics = _parse_saliency_metrics(metrics)
	if not metrics:
		return None
	agonists = _extract_singleton_agonists_from_records(ablation_records, baseline_subset)
	if not agonists:
		return None
	layer_labels = [agonist["layer_label"] for agonist in agonists]
	associated_payload = collect_reference_margin_tensors(
		model, task, associated_examples, prompt_col, layer_labels, batch_size=batch_size, allow_fallback_score=True
	)
	if associated_payload is None:
		print(f"[AgonistSaliency] circuit={int(circuit_id)} skipped: could not collect associated gradients.")
		return None
	unrelated_payload = collect_reference_margin_tensors(
		model, task, unrelated_examples, prompt_col, layer_labels, batch_size=batch_size, allow_fallback_score=True
	)
	if unrelated_payload is None:
		print(f"[AgonistSaliency] circuit={int(circuit_id)} skipped: could not collect unrelated gradients.")
		return None

	raw_rows = []
	summary_rows = []
	for metric in metrics:
		for agonist in agonists:
			assoc_rows, assoc_summary = _saliency_rows_for_unit(circuit_id, "associated", agonist, associated_payload.get(agonist["layer_label"]), model=model, metric=metric)
			if assoc_summary is not None:
				raw_rows.extend(assoc_rows)
				summary_rows.append(assoc_summary)
			unrel_rows, unrel_summary = _saliency_rows_for_unit(circuit_id, "unrelated", agonist, unrelated_payload.get(agonist["layer_label"]), model=model, metric=metric)
			if unrel_summary is not None:
				raw_rows.extend(unrel_rows)
				summary_rows.append(unrel_summary)
	if not raw_rows:
		return None

	raw_df = pd.DataFrame(raw_rows)
	summary_df = pd.DataFrame(summary_rows)
	aggregate_stats = _aggregate_saliency_stats(raw_rows)
	correlation_rows = _saliency_ablation_correlation_rows(summary_rows)
	base_name = rule_json_path.stem
	raw_csv_path = rule_json_path.parent / f"{base_name}_agonist_saliency_raw.csv"
	summary_csv_path = rule_json_path.parent / f"{base_name}_agonist_saliency_summary.csv"
	plot_path = rule_json_path.parent / f"{base_name}_agonist_saliency_stats.png"
	corr_csv_path = rule_json_path.parent / f"{base_name}_agonist_saliency_correlations.csv"
	raw_df.to_csv(raw_csv_path, index=False)
	summary_df.to_csv(summary_csv_path, index=False)
	if correlation_rows:
		pd.DataFrame(correlation_rows).to_csv(corr_csv_path, index=False)
	_plot_agonist_saliency_stats(raw_df, plot_path, title=f"Circuit {int(circuit_id)} agonist saliency diagnostics")
	question_answer = {}
	for metric, metric_stats in aggregate_stats.items():
		question_answer[metric] = {
			split: _top_saliency_verdict(metric_stats.get(split))
			for split in ("associated", "unrelated")
			if split in metric_stats
		}
	output = {
		"n_agonists": int(len(agonists)),
		"measurement_position": "last_prompt_token",
		"metrics": list(metrics),
		"compare_pool": "all units in the same hooked layer, after flattening attention heads × dimensions",
		"mlp_wanda_note": "For mL units this script hooks blocks.L.hook_mlp_out, so MLP Wanda reduces to |activation| because there is no per-coordinate outgoing W_out at this hook site.",
		"gradient_objective_note": "Gradient saliency uses task margin when available; otherwise cached completion first-token logit when inferable, otherwise top next-token logit.",
		"summary_rows": summary_rows,
		"aggregate_stats": aggregate_stats,
		"files": {
			"raw_csv": raw_csv_path.name,
			"summary_csv": summary_csv_path.name,
			"plot_png": plot_path.name,
			"correlations_csv": corr_csv_path.name if correlation_rows else None,
		},
		"correlation_rows": correlation_rows,
		"question_answer": question_answer,
	}
	return output, raw_rows, summary_rows


def _print_agonist_saliency_summary(prefix, saliency_stats):
	if not saliency_stats:
		return
	agg = saliency_stats.get("aggregate_stats", {})
	for metric in saliency_stats.get("metrics", []):
		metric_stats = agg.get(metric, {})
		for split_name in ("associated", "unrelated"):
			split_stats = metric_stats.get(split_name)
			if not split_stats:
				continue
			print(
				f"{prefix} [{metric}/{split_name}] mean_percentile={split_stats['mean_layer_percentile_rank']:.3f}, "
				f"median_percentile={split_stats['median_layer_percentile_rank']:.3f}, "
				f"top1={split_stats['top1_rate']:.3f}, top5={split_stats['top5_rate']:.3f}, "
				f"top10pct={split_stats['top10pct_rate']:.3f} -> "
				f"{_top_saliency_verdict(split_stats)}"
			)


def _print_agonist_margin_summary(prefix, margin_stats):
	if not margin_stats:
		return
	agg = margin_stats.get("aggregate_stats", {})
	for split_name in ("associated", "unrelated"):
		split_stats = agg.get(split_name)
		if not split_stats:
			continue
		print(f"{prefix} [{split_name}] mean_drop={split_stats['mean_predicted_margin_drop']:.4f}, median_drop={split_stats['median_predicted_margin_drop']:.4f}, positive_drop_rate={split_stats['positive_margin_drop_rate']:.3f} -> {_margin_effect_verdict(split_stats)}")


def _print_agonist_activation_summary(prefix, activation_stats):
	if not activation_stats:
		return
	agg = activation_stats.get("aggregate_stats", {})
	for split_name in ("associated", "unrelated"):
		split_stats = agg.get(split_name)
		if not split_stats:
			continue
		print(
			f"{prefix} [{split_name}] mean_percentile={split_stats['mean_layer_percentile_rank']:.3f}, "
			f"median_percentile={split_stats['median_layer_percentile_rank']:.3f}, "
			f"top1={split_stats['top1_rate']:.3f}, top5={split_stats['top5_rate']:.3f}, "
			f"top10pct={split_stats['top10pct_rate']:.3f} -> "
			f"{_top_activation_verdict(split_stats)}"
		)



def _cached_stat_files_exist(cached: dict, json_path: Path, stats_key: str) -> bool:
	stats = cached.get(stats_key) or {}
	files = stats.get("files") or {}
	for key in ("raw_csv", "summary_csv"):
		name = files.get(key)
		if not name:
			return False
		if not (json_path.parent / name).exists():
			return False
	return True

def _needs_agonist_activation_backfill(cached: dict, args, json_path: Path | None = None) -> bool:
	if cached.get("status") != "ok":
		return False
	if not bool(cached.get("ablations")):
		return False
	if json_path is None:
		json_path = Path(".")
	needs_activation = (
		not getattr(args, "skip_agonist_activation_stats", False)
		and not _cached_stat_files_exist(cached, json_path, "agonist_activation_stats")
	)
	needs_saliency = (
		not getattr(args, "skip_agonist_saliency_stats", False)
		and not _cached_stat_files_exist(cached, json_path, "agonist_saliency_stats")
	)
	return bool(needs_activation or needs_saliency)

def _collect_existing_agonist_activation_rows(circuit_entries: dict, rule_out_root: Path, args):
	raw_rows = []
	summary_rows = []
	for info in circuit_entries.values():
		json_path = _circuit_output_json_path(info, rule_out_root, args)
		if not json_path.exists():
			continue
		try:
			cached = json.loads(json_path.read_text())
		except Exception:
			continue
		stats = cached.get("agonist_activation_stats") or {}
		files = stats.get("files") or {}
		raw_csv = files.get("raw_csv")
		summary_csv = files.get("summary_csv")
		if raw_csv:
			raw_path = json_path.parent / raw_csv
			if raw_path.exists():
				try:
					raw_rows.extend(pd.read_csv(raw_path).to_dict("records"))
				except Exception:
					pass
		if summary_csv:
			summary_path = json_path.parent / summary_csv
			if summary_path.exists():
				try:
					summary_rows.extend(pd.read_csv(summary_path).to_dict("records"))
				except Exception:
					pass
	return raw_rows, summary_rows


def _write_global_agonist_activation_outputs(circuit_entries: dict, rule_out_root: Path, args):
	if getattr(args, "skip_agonist_activation_stats", False):
		return
	raw_rows, summary_rows = _collect_existing_agonist_activation_rows(circuit_entries, rule_out_root, args)
	if not raw_rows:
		return
	global_raw_df = pd.DataFrame(raw_rows)
	global_summary_df = pd.DataFrame(summary_rows)
	global_stats = _aggregate_activation_stats(raw_rows)
	global_raw_csv_path = rule_out_root / "agonist_activation_raw.csv"
	global_summary_csv_path = rule_out_root / "agonist_activation_summary.csv"
	global_plot_path = rule_out_root / "agonist_activation_stats.png"
	global_json_path = rule_out_root / "agonist_activation_stats.json"
	global_raw_df.to_csv(global_raw_csv_path, index=False)
	if not global_summary_df.empty:
		global_summary_df.to_csv(global_summary_csv_path, index=False)
	_plot_agonist_activation_stats(
		global_raw_df,
		global_plot_path,
		title="Global agonist activation diagnostics",
	)
	global_payload = {
		"aggregate_stats": global_stats,
		"files": {
			"raw_csv": global_raw_csv_path.name,
			"summary_csv": global_summary_csv_path.name,
			"plot_png": global_plot_path.name,
		},
		"question_answer": {
			"associated": _top_activation_verdict(global_stats.get("associated")),
			"unrelated": _top_activation_verdict(global_stats.get("unrelated")),
		},
	}
	global_json_path.write_text(json.dumps(global_payload, indent=4))
	_print_agonist_activation_summary("[AgonistAct][Global]", global_payload)


def _collect_existing_agonist_saliency_rows(circuit_entries: dict, rule_out_root: Path, args):
	raw_rows = []
	summary_rows = []
	for info in circuit_entries.values():
		json_path = _circuit_output_json_path(info, rule_out_root, args)
		if not json_path.exists():
			continue
		try:
			cached = json.loads(json_path.read_text())
		except Exception:
			continue
		stats = cached.get("agonist_saliency_stats") or {}
		files = stats.get("files") or {}
		raw_csv = files.get("raw_csv")
		summary_csv = files.get("summary_csv")
		if raw_csv:
			raw_path = json_path.parent / raw_csv
			if raw_path.exists():
				try:
					raw_rows.extend(pd.read_csv(raw_path).to_dict("records"))
				except Exception:
					pass
		if summary_csv:
			summary_path = json_path.parent / summary_csv
			if summary_path.exists():
				try:
					summary_rows.extend(pd.read_csv(summary_path).to_dict("records"))
				except Exception:
					pass
	return raw_rows, summary_rows


def _write_global_agonist_saliency_outputs(circuit_entries: dict, rule_out_root: Path, args):
	if getattr(args, "skip_agonist_saliency_stats", False):
		return
	raw_rows, summary_rows = _collect_existing_agonist_saliency_rows(circuit_entries, rule_out_root, args)
	if not raw_rows:
		return
	global_raw_df = pd.DataFrame(raw_rows)
	global_summary_df = pd.DataFrame(summary_rows)
	global_stats = _aggregate_saliency_stats(raw_rows)
	global_correlation_rows = _saliency_ablation_correlation_rows(summary_rows)
	global_raw_csv_path = rule_out_root / "agonist_saliency_raw.csv"
	global_summary_csv_path = rule_out_root / "agonist_saliency_summary.csv"
	global_plot_path = rule_out_root / "agonist_saliency_stats.png"
	global_corr_csv_path = rule_out_root / "agonist_saliency_correlations.csv"
	global_json_path = rule_out_root / "agonist_saliency_stats.json"
	global_raw_df.to_csv(global_raw_csv_path, index=False)
	if not global_summary_df.empty:
		global_summary_df.to_csv(global_summary_csv_path, index=False)
	if global_correlation_rows:
		pd.DataFrame(global_correlation_rows).to_csv(global_corr_csv_path, index=False)
	_plot_agonist_saliency_stats(
		global_raw_df,
		global_plot_path,
		title="Global agonist saliency diagnostics",
	)
	question_answer = {}
	for metric, metric_stats in global_stats.items():
		question_answer[metric] = {
			split: _top_saliency_verdict(metric_stats.get(split))
			for split in ("associated", "unrelated")
			if split in metric_stats
		}
	global_payload = {
		"aggregate_stats": global_stats,
		"files": {
			"raw_csv": global_raw_csv_path.name,
			"summary_csv": global_summary_csv_path.name,
			"plot_png": global_plot_path.name,
			"correlations_csv": global_corr_csv_path.name if global_correlation_rows else None,
		},
		"correlation_rows": global_correlation_rows,
		"question_answer": question_answer,
	}
	global_json_path.write_text(json.dumps(global_payload, indent=4))
	_print_agonist_saliency_summary("[AgonistSaliency][Global]", {**global_payload, "metrics": list(global_stats.keys())})

set_deterministic(args.seed)

task = resolve_task_spec(args.task_module)
is_answer_positive_fn = task.is_answer_positive
max_new_tokens = task.MAX_NEW_TOKENS


# Default alpha used for standalone (non-dichotomic) calls to ablate_neurons.
# dichotomic_search_layer will override this per node via its own alpha spending.
_alpha_node_default = get_alpha_node(args.prune_alpha)
ALPHA_SLICE_DEFAULT = None if (_alpha_node_default is None) else (_alpha_node_default / 2.0)


input_data_dir = Path(args.input_data_dir).resolve()
manifest_path = input_data_dir / "manifest.json"
if not manifest_path.exists():
	raise FileNotFoundError(f"manifest.json not found in {input_data_dir}")

# Output root
rule_out_root = Path(args.output_data_dir).resolve()
rule_out_root.mkdir(parents=True, exist_ok=True)

# Where we persist cross-run neuron buckets
buckets_path = rule_out_root / "neuron_buckets.json"

# ------------------------- Circuits per rule (lightweight) ----------------------------
circuit_entries = get_circuit_neurons_dict(manifest_path, args)
if not circuit_entries:
	raise RuntimeError("No usable rules found in manifest metadata_topn (need layer labels).")

# ------------------------- Resume scan (before heavy setup) ----------------------------
summary_rule_knockout, buckets, circuits_to_process = _scan_reusable_circuits(
	circuit_entries=circuit_entries,
	rule_out_root=rule_out_root,
	args=args,
)

if len(circuits_to_process) == 0:
	# Everything already finished: persist summary + buckets/stats and exit
	(rule_out_root / "rule_knockout.json").write_text(json.dumps(summary_rule_knockout, indent=4))
	_write_global_agonist_activation_outputs(circuit_entries, rule_out_root, args)
	_write_global_agonist_saliency_outputs(circuit_entries, rule_out_root, args)
	_write_bucket_outputs(rule_out_root, buckets, buckets_path)
	print("[Resume] All requested circuits already finished; nothing to do.")
	raise SystemExit(0)

# -------------------------------------------------------------------------
# Optional: build a per-circuit, per-layer map of signed discovery scores.
#
# This is used ONLY when --sign_split_first is enabled, to split the first
# dichotomic branch by score sign (>=0 vs <0) instead of a pure index bisection.
# The scores come from manifest metadata_topn['neuron_label_score'].
#
# NOTE: we build this only if we still have circuits to process.
# -------------------------------------------------------------------------
importance_scores_by_circuit = {}
if getattr(args, 'sign_split_first', False):
	todo_ids = set(int(info.get('circuit_id')) for info in circuits_to_process)
	with open(manifest_path, 'r') as f:
		_manifest_raw = json.load(f)

	for _entry in _manifest_raw:
		cid = _entry.get('circuit_id', None)
		if cid is None:
			continue
		try:
			cid = int(cid)
		except Exception:
			continue
		if cid not in todo_ids:
			continue

		# neuron_label_score: {"('m27', 277)": -0.0123, ...}
		nls = (_entry.get('metadata_topn', {}) or {}).get('neuron_label_score', {})
		if not isinstance(nls, dict) or not nls:
			continue

		layer_map = defaultdict(dict)  # layer_label -> {neuron_id: score}
		for k, v in nls.items():
			layer_label, neuron_id = _parse_manifest_neuron_label(k)
			if layer_label is None or neuron_id is None:
				continue
			try:
				layer_map[layer_label][int(neuron_id)] = float(v)
			except Exception:
				continue

		importance_scores_by_circuit[cid] = dict(layer_map)

	print(f"[SignSplit] Loaded signed neuron scores for {len(importance_scores_by_circuit)} circuits from manifest.")

# ------------------------- Model ----------------------------
dataset_info = _load_dataset_info(input_data_dir)
scores_path = (
	Path(args.scores_path or dataset_info.get("scores_path", ""))
	if (args.scores_path or dataset_info.get("scores_path"))
	else None
)
prompt_col = dataset_info.get("prompt_col") or task.DEFAULT_INPUT
target_col = dataset_info.get("target_col") or task.DEFAULT_TARGETS[0]
ai_model = args.ai_model or dataset_info.get("ai_model") #or "Qwen/Qwen2-7B-Instruct"

if scores_path is None or not scores_path.exists():
	raise FileNotFoundError(
		"Could not resolve dataset (scores) path. Either keep dataset_info.json from discovery "
		"or pass --scores_path."
	)

ftype = guess_filetype(scores_path)
scores_df = pd.read_parquet(scores_path) if ftype == "parquet" else pd.read_csv(scores_path)

# Keep original row indices so we can map spectral clusters back after filtering
scores_df["original_idx"] = np.arange(len(scores_df))

# numeric cols → NaN to 0 (parity with discovery)
scores_df = safe_features_fillna(scores_df, fill_number=0, fill_bool=False, cols_not_to_fill=task.DEFAULT_TARGETS)

# ------------------------- Train/Test split -------------------------
# If an 'is_test' flag is present, keep TRAIN rows only for this analysis.
if "is_test" in scores_df.columns:
	_mask_train = ~scores_df["is_test"].astype(bool)
	n_before = len(scores_df)
	scores_df = scores_df.loc[_mask_train].reset_index(drop=True)
	print(f"[Split] TRAIN only: {len(scores_df)} / {n_before} rows (dropped {n_before - len(scores_df)} test rows)")

# Preserve the full TRAIN split for mean-ablation estimation before any baseline_subset filter.
scores_df_train_for_mean = scores_df.copy()

# ------------------------- Baseline correctness -------------------------
# Optionally filter the dataset by baseline label (wrt primary target).
if args.baseline_subset == "positive":
	scores_df = scores_df.loc[scores_df[target_col] == True]
elif args.baseline_subset == "negative":
	scores_df = scores_df.loc[scores_df[target_col] == False]
# cols_with_na = scores_df.columns[scores_df.isna().any()]
# if cols_with_na:
# 	print(f"Filling NaNs with 0 in {len(cols_with_na)} columns: " + ", ".join(cols_with_na))
# 	scores_df = scores_df.fillna(0)
scores_df = scores_df.reset_index(drop=True)

# Model
device = get_device()
model = LMWrapper(model_name=ai_model, device=device, eval_mode=True, circuit_discovery=False, cache_dir=args.ai_model_cache_dir)

# Precompute spectral clusters (baseline-subset aware) if requested
cluster_member_indices_orig = None
if args.cluster_by_spectral:
	cluster_texts = scores_df[prompt_col].astype(str).tolist()

	unhooked_model = getattr(model, "model", None)
	tokenizer = getattr(model, "tokenizer", None)

	_, Z = build_reps_and_embedding_from_args(
		args=args,
		texts=cluster_texts,
		model=unhooked_model,
		tokenizer=tokenizer,
		device=device,
	)

	n_points = int(Z.shape[0])
	min_size = max(args.n_associated, args.n_unrelated)

	# Need:
	# - at least 2 clusters whenever possible, otherwise cluster 0 has empty complement
	# - no more clusters than points
	# - no more clusters than can satisfy the min-size assignment
	k_cap_by_points = n_points
	k_cap_by_min_size = max(1, n_points // max(1, min_size))
	k = min(max(2, int(args.global_n_clusters)), k_cap_by_points, k_cap_by_min_size)

	if k < 2:
		raise ValueError(
			f"Spectral splits need at least 2 feasible clusters, but got only k={k}. "
			f"n_points={n_points}, n_associated={args.n_associated}, "
			f"n_unrelated={args.n_unrelated}, global_n_clusters={args.global_n_clusters}."
		)

	centers_idx, _, _, meta, x_norm2 = kcenter_farthest_first(Z, k=k)

	cluster_ids = assign_min_size_nearest_to_centers(
		Z,
		centers_idx,
		min_size=min_size,
		chunk_size=8192,
	).astype(int)

	# Map cluster -> ORIGINAL row indices (FULL dataset indices at clustering time)
	cluster_member_indices_orig = {
		c: np.where(cluster_ids == c)[0].tolist()
		for c in range(k)
	}
	# for k,v in cluster_member_indices_orig.items():
	# 	print(k, len(v))

# ------------------------- Circuits per rule ----------------------------
circuit_entries = get_circuit_neurons_dict(manifest_path, args)
if not circuit_entries:
	raise RuntimeError("No usable rules found in manifest metadata_topn (need layer labels).")

# Optional: load sampling plan if requested
sampling_plan = None
sampling_plan_index = None
effective_sampling_strategy = args.sampling_strategy

if args.sampling_strategy == "plan":
	if args.sampling_plan_path is None:
		raise ValueError("--sampling_strategy 'plan' requires --sampling_plan_path.")
	sampling_plan_path = Path(args.sampling_plan_path)
	if not sampling_plan_path.exists():
		raise FileNotFoundError(f"Sampling plan not found: {sampling_plan_path}")
	with open(sampling_plan_path, "r") as f:
		sampling_plan = json.load(f)

	# Check baseline_subset compatibility
	plan_baseline_subset = sampling_plan.get("baseline_subset", None)
	if plan_baseline_subset is not None and plan_baseline_subset != args.baseline_subset:
		print(
			f"[Sampling] Plan baseline_subset={plan_baseline_subset!r} "
			f"does not match args.baseline_subset={args.baseline_subset!r}; "
			f"falling back to random sampling."
		)
		effective_sampling_strategy = "random"
	else:
		sampling_plan_index = {}
		assert len(scores_df) == sampling_plan["dataset_size"], f'{len(scores_df)} != {sampling_plan["dataset_size"]}'
		for r in sampling_plan["rules"]:
			rid = int(r["rule_id"])
			rule_target = r["rule_target"]
			key = (rule_target, rid)
			sampling_plan_index[key] = r
		print(
			f"[Sampling] Loaded sampling plan from {sampling_plan_path} "
			f"with {len(sampling_plan_index)} rule entries."
		)

all_prompts_targets = scores_df.loc[:, prompt_col].values.tolist()
all_examples = scores_df.to_dict(orient="records")

# Only compute mean-ablation statistics when there are actual ablations left to run.
# Resume/backfill-only jobs (e.g. agonist-activation backfills) do not need them.
circuits_requiring_ablation = [
	info for info in circuits_to_process
	if info.get("_resume_mode") != "backfill_agonist_activation_stats"
]

mean_ablation_prompts = []
if circuits_requiring_ablation:
	mean_ablation_prompts = _build_balanced_mean_prompt_pool(
		scores_df_train_for_mean,
		prompt_col=prompt_col,
		target_col=target_col,
		n_points=args.points_to_use_for_mean_ablation,
		seed=args.seed,
		baseline_subset=args.baseline_subset,
		# baseline_subset="all",
	)

# ------------------------- Precompute mean activations (if needed) -----
mean_activations = None
if args.intervention in ("mean", "mean-donor", "mean-positional", "mean-donor-positional"):
	if not circuits_requiring_ablation:
		print("[MeanAblation] Skipping mean precomputation: all remaining work is resume/backfill-only.")
	else:
		# collect global unit -> ids across circuits that still need ablation
		global_unit_to_neurons = defaultdict(set)
		for info in circuits_requiring_ablation:
			units = _filter_layer_units_for_ablation(
				_extract_layer_units(info),
				mlp_neurons_only=args.mlp_neurons_only,
			)
			for layer_label, unit_list in units.items():
				for unit_id in unit_list:
					global_unit_to_neurons[str(layer_label)].add(int(unit_id))

		if global_unit_to_neurons:
			global_unit_to_neurons = {
				k: sorted(list(v)) for k, v in global_unit_to_neurons.items()
			}
			mean_prompts_for_ablation = mean_ablation_prompts if mean_ablation_prompts else all_prompts_targets
			mean_activations = precompute_mean_activations(
				model=model,
				all_prompts=mean_prompts_for_ablation,
				layer_to_neurons=global_unit_to_neurons,
				n_points=min(args.points_to_use_for_mean_ablation, len(mean_prompts_for_ablation)),
				batch_size=args.batch_size,
				intervention=args.intervention,
				device=device,
			)
# ------------------------- Per-rule ablation with dichotomic search ---------
for info in tqdm(circuits_to_process, desc="Per-circuit ablation"):
	raw_circuit_units = _extract_layer_units(info)
	circuit_units = _filter_layer_units_for_ablation(
		raw_circuit_units,
		mlp_neurons_only=args.mlp_neurons_only,
	)
	print('Analyzed layers:', json.dumps(list(circuit_units.keys()), indent=4))
	if not circuit_units:
		rid = info["circuit_id"]
		rule = info["circuit_label"]
		rule_target = info.get("target", info["rule_target"])
		analysis_mode = info.get("analysis_mode", ("spectral_cluster" if args.cluster_by_spectral else "rule"))
		rule_direction = None if analysis_mode == "spectral_cluster" else info.get("coefficient_sign", info["rule_direction"])
		cluster_index = info.get("cluster_index")
		rule_out_root_target = _circuit_output_json_path(info, rule_out_root, args)
		reason = (
			"No MLP neurons left after applying --mlp_neurons_only."
			if args.mlp_neurons_only and raw_circuit_units
			else "No usable units found for this circuit."
		)
		rule_detail = {
			"circuit_id": int(rid),
			"circuit_label": rule,
			"analysis_mode": analysis_mode,
			"cluster_index": (int(cluster_index) if analysis_mode == "spectral_cluster" and cluster_index is not None else (int(rid) if analysis_mode == "spectral_cluster" else None)),
			"rule_target": rule_target,
			"rule_direction": rule_direction,
			"intervention": args.intervention,
			"decode_only": bool(args.decode_only),
			"baseline_subset": args.baseline_subset,
			"n_associated_positive": 0,
			"n_unrelated_positive": 0,
			"status": "skipped",
			"reason": reason,
			"n_associated_tested": 0,
			"n_unrelated_tested": 0,
			"ablations": [],
		}
		rule_out_root_target.write_text(json.dumps(rule_detail, indent=4))
		summary_rule_knockout.append(_summary_entry_from_rule_detail(rule_detail))
		continue

	rid = info["circuit_id"]
	rule = info["circuit_label"]
	rule_target = info.get("target", info["rule_target"])

	analysis_mode = info.get("analysis_mode", ("spectral_cluster" if args.cluster_by_spectral else "rule"))
	rule_direction = None if analysis_mode == "spectral_cluster" else info.get("coefficient_sign", info["rule_direction"])

	cluster_index = info.get("cluster_index")
	rule_out_root_target = _circuit_output_json_path(info, rule_out_root, args)
	resume_mode = info.get("_resume_mode")
	cached_rule_detail = info.get("_cached_rule_detail") if resume_mode == "backfill_agonist_activation_stats" else None
	is_backfill_only = cached_rule_detail is not None

	is_using_predefined_plan = effective_sampling_strategy == "plan" and sampling_plan_index is not None
	if is_using_predefined_plan:
		key = (rule_target, rid)
		plan_entry = sampling_plan_index.get(key)
		if plan_entry is None or plan_entry.get("status") != "ok":
			print(f"[Sampling] No usable plan entry for {key}; falling back to random.")
			effective_sampling_strategy = "random"

	# Which prompts are associated with this circuit?
	if args.cluster_by_spectral:
		# Interpret `rid` as spectral cluster index; positives are points in that cluster,
		# negatives are all remaining datapoints in the current (baseline-filtered) dataset.
		members = np.array(cluster_member_indices_orig[rid], dtype=int)
		mask = np.zeros(len(cluster_ids), dtype=bool)
		mask[members] = True
		idx_pos = np.where(mask)[0].tolist()
		idx_neg = np.where(~mask)[0].tolist()
		print(f"Processing spectral cluster {rid} [{rule}]")
	else:
		if is_using_predefined_plan:
			key = (rule_target, rid)
			plan_entry = sampling_plan_index.get(key)
			idx_pos = plan_entry["associated_indices"]
			idx_neg = plan_entry["unrelated_indices"]
		else:
			assert rule_direction
			# Standard: use association rule over feature scores
			rule_mask = apply_rule_to_features(rule, scores_df, direction=rule_direction).tolist()
			idx_pos = [i for i, m in enumerate(rule_mask) if m]
			idx_neg = [i for i, m in enumerate(rule_mask) if not m]
			idx_pos, idx_neg = balance_positives_and_negatives(idx_pos, idx_neg, t=max(args.n_associated, args.n_unrelated))
		print(f"Processing rule {rid} [{rule}]")

	print(f"\tpositives: {len(idx_pos)}; negatives: {len(idx_neg)}")
	assert len(idx_pos) >= args.n_associated and len(idx_neg) >= args.n_unrelated

	# Common per-rule detail structure
	rule_detail = {
		"circuit_id": int(rid),
		"circuit_label": rule,
		"analysis_mode": analysis_mode,
		"cluster_index": (int(cluster_index) if analysis_mode == "spectral_cluster" and cluster_index is not None else (int(rid) if analysis_mode == "spectral_cluster" else None)),
		"rule_target": rule_target,
		"rule_direction": rule_direction,
		"intervention": args.intervention,
		"decode_only": bool(args.decode_only),
		"n_associated_positive": int(len(idx_pos)),
		"n_unrelated_positive": int(len(idx_neg)),
		"baseline_subset": args.baseline_subset,
	}

	# If we don't have both positives and negatives, we can't assess targeted effect
	# if len(idx_pos) == 0 or len(idx_neg) == 0:
	if len(idx_pos) < args.n_associated or len(idx_neg) < args.n_unrelated:
		rule_detail.update(
			{
				"status": "skipped",
				"reason": "Insufficient associated or unrelated prompts",
				"n_associated_tested": 0,
				"n_unrelated_tested": 0,
				"ablations": [],
			}
		)
		rule_out_root_target.write_text(json.dumps(rule_detail, indent=4))
		summary_rule_knockout.append(_summary_entry_from_rule_detail(rule_detail))
		continue

	if is_backfill_only:
		idx_pos_sampled = np.array(cached_rule_detail.get("sampled_associated_indices", []), dtype=int)
		idx_neg_sampled = np.array(cached_rule_detail.get("sampled_unrelated_indices", []), dtype=int)
		if len(idx_pos_sampled) == 0 or len(idx_neg_sampled) == 0:
			print(f"[Resume] Cannot backfill agonist activation stats for circuit {rid}: missing sampled indices.")
			continue
		pos_prompts = [all_examples[i] for i in map(int, idx_pos_sampled)]
		neg_prompts = [all_examples[i] for i in map(int, idx_neg_sampled)]
		if not args.skip_agonist_activation_stats and not _cached_stat_files_exist(cached_rule_detail, rule_out_root_target, "agonist_activation_stats"):
			agonist_stats_payload = compute_and_save_agonist_activation_stats(
				model,
				rule_out_root_target,
				circuit_id=rid,
				circuit_label=rule,
				prompt_col=prompt_col,
				associated_examples=pos_prompts,
				unrelated_examples=neg_prompts,
				ablation_records=cached_rule_detail.get("ablations", []),
				baseline_subset=args.baseline_subset,
				batch_size=args.batch_size,
				decode_only=args.decode_only,
				max_new_tokens=max_new_tokens,
			)
			if agonist_stats_payload is not None:
				agonist_activation_stats, _, _ = agonist_stats_payload
				cached_rule_detail["agonist_activation_stats"] = agonist_activation_stats
				_print_agonist_activation_summary(f"\t[AgonistAct] circuit={rid}", agonist_activation_stats)
			else:
				cached_rule_detail["agonist_activation_stats"] = None
		if not args.skip_agonist_saliency_stats and not _cached_stat_files_exist(cached_rule_detail, rule_out_root_target, "agonist_saliency_stats"):
			agonist_saliency_payload = compute_and_save_agonist_saliency_stats(
				model,
				task,
				rule_out_root_target,
				circuit_id=rid,
				circuit_label=rule,
				prompt_col=prompt_col,
				associated_examples=pos_prompts,
				unrelated_examples=neg_prompts,
				ablation_records=cached_rule_detail.get("ablations", []),
				baseline_subset=args.baseline_subset,
				batch_size=args.batch_size,
				metrics=args.agonist_saliency_metrics,
			)
			if agonist_saliency_payload is not None:
				agonist_saliency_stats, _, _ = agonist_saliency_payload
				cached_rule_detail["agonist_saliency_stats"] = agonist_saliency_stats
				_print_agonist_saliency_summary(f"\t[AgonistSaliency] circuit={rid}", agonist_saliency_stats)
			else:
				cached_rule_detail["agonist_saliency_stats"] = None
		rule_out_root_target.write_text(json.dumps(cached_rule_detail, indent=4))
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		continue

	take_pos = min(args.n_associated, len(idx_pos))
	take_neg = min(args.n_unrelated, len(idx_neg))
	idx_pos_sampled = np.random.choice(idx_pos, size=take_pos, replace=False)
	idx_neg_sampled = np.random.choice(idx_neg, size=take_neg, replace=False)
	pos_prompts = [all_examples[i] for i in map(int,idx_pos_sampled)]
	neg_prompts = [all_examples[i] for i in map(int,idx_neg_sampled)]
	
	# --- build prefixes and search epsilon ---
	search_epsilon = get_adjusted_search_epsilon(
		args.search_epsilon, 
		pos_prompts, neg_prompts, 
		args.n_associated, args.n_unrelated, 
		args.prune_alpha
	)
	results_dict = ablate_neurons(
		model, 
		pos_prompts, neg_prompts, 
		is_answer_positive_fn, 
		prompt_col, 
		layers_neurons_dict=None,
		batch_size=args.batch_size, 
		decode_only=args.decode_only, 
		intervention=args.intervention,
		mean_activations=mean_activations, 
		max_new_tokens=max_new_tokens,
		return_prefix_batches=args.decode_only,
		baseline_subset=args.baseline_subset,
		alpha_slice=ALPHA_SLICE_DEFAULT,
		return_answers=True,
	)
	prefix_batches = results_dict.get('prefix_batches') if args.decode_only else None
	batch_ranges = results_dict.get('batch_ranges') if args.decode_only else None
	
	# --- Validity check ---
	baseline_acc_pos = results_dict["acc_after_knockout_on_associated"]
	baseline_acc_neg = results_dict["acc_after_knockout_on_unrelated"]
	baseline_gap = baseline_acc_neg - baseline_acc_pos
	print(
		f"\tBaseline accuracy (associated={baseline_acc_pos:.3f}, unrelated={baseline_acc_neg:.3f}, "
		f"gap (neg-pos)={baseline_gap:.3f}) [sampling={effective_sampling_strategy}]"
	)

	acc_pos_all = np.asarray(results_dict["acc_after_knockout_on_associated_all"], dtype=float)
	acc_neg_all = np.asarray(results_dict["acc_after_knockout_on_unrelated_all"], dtype=float)

	if args.baseline_subset == "positive":
		bad_pos = np.where(acc_pos_all < 0.5)[0].tolist()
		bad_neg = np.where(acc_neg_all < 0.5)[0].tolist()
	else:
		bad_pos = np.where(acc_pos_all > 0.5)[0].tolist()
		bad_neg = np.where(acc_neg_all > 0.5)[0].tolist()

	print(f"[BaselineDebug] associated bad: {len(bad_pos)}/{len(acc_pos_all)} -> {bad_pos}")
	print(f"[BaselineDebug] unrelated bad:  {len(bad_neg)}/{len(acc_neg_all)} -> {bad_neg}")

	answers_pos_all = results_dict.get("answers_after_knockout_on_associated_all", []) or []
	answers_neg_all = results_dict.get("answers_after_knockout_on_unrelated_all", []) or []

	def _expected_output_for_debug(row):
		output_col = getattr(task, "DEFAULT_OUTPUT", None)
		if output_col in row:
			return row.get(output_col)
		return f"<unavailable; task.DEFAULT_OUTPUT={output_col!r}>"

	def _print_mismatching_prompt_debug(split_name, bad_indices, rows, answers):
		for j in bad_indices[:10]:
			row = rows[j]
			answer = answers[j] if j < len(answers) else None
			print("SPLIT", split_name)
			print("IDX", j)
			print("TARGET_COL", row.get(target_col))
			print("EXPECTED_OUTPUT", repr(_expected_output_for_debug(row)))
			print("PREDICTED_OUTPUT", repr(answer))
			print("PROMPT", str(row[prompt_col])[:1000])

	_print_mismatching_prompt_debug("associated", bad_pos, pos_prompts, answers_pos_all)
	_print_mismatching_prompt_debug("unrelated", bad_neg, neg_prompts, answers_neg_all)

	# Exact float equality is fragile; treat this as a numerical-consistency check.
	if abs(baseline_gap) > 0 or (baseline_acc_neg != 1 and baseline_acc_neg != 0):
		raise AssertionError(
			"Baseline pos/neg accuracies differ beyond tolerance. "
			f"pos={baseline_acc_pos:.12f}, neg={baseline_acc_neg:.12f}, "
			f"gap={baseline_gap:.12e}. "
			"This may indicate semantic mismatch (e.g., hooks + caching), or nondeterminism."
		)
	# --- End of Validity check ---

	rule_ablation_results = []

	if args.fast_ablation:
		excluded_for_dichotomic = _excluded_neuron_keys(
			buckets, 
			exclude_catastrophic_candidates = True, 
			exclude_catastrophic_confirmed = True
		)

		# Run dichotomic search per layer, excluding known special buckets.
		for layer_label, neuron_list in tqdm(circuit_units.items(), total=len(circuit_units), desc='Layers', leave=False):
			# if layer_label not in ['m3','m27','m26','m4']:
			# 	continue

			# Evaluate catastrophic-zero candidates individually (normal ablation, not dichotomic),
			#    until they are decided (confirmed vs not_always). Keep them separated from dichotomic regardless.
			catastrophic_candidates = []
			for neuron in neuron_list:
				key = _neuron_key(layer_label, neuron)
				if key in buckets["catastrophic_zero"]["candidates"]:
					catastrophic_candidates.append(int(neuron))

			for neuron in tqdm(catastrophic_candidates, desc='Catastrophic Candidates', leave=False):
				ablation_record = ablate_neurons(
					model,
					pos_prompts,
					neg_prompts,
					is_answer_positive_fn,
					prompt_col,
					layers_neurons_dict={layer_label: [neuron]},
					batch_size=args.batch_size,
					decode_only=args.decode_only,
					intervention=args.intervention,
					mean_activations=mean_activations,
					prefix_batches=prefix_batches,
					batch_ranges=batch_ranges,
					max_new_tokens=max_new_tokens,
					baseline_subset=args.baseline_subset,
					alpha_slice=ALPHA_SLICE_DEFAULT,
				)
				rule_ablation_results.append(ablation_record)
				_update_buckets_from_single_record(
					buckets,
					ablation_record,
					baseline_subset=args.baseline_subset,
					circuit_id=rid,
				)

			# Dichotomic search on the remaining neurons
			search_neuron_ids = [
				int(n) for n in neuron_list
				if _neuron_key(layer_label, n) not in excluded_for_dichotomic
			]
			# NOTE: order affects which neurons co-occur in groups during bisection.
			# We still shuffle here for robustness; when --sign_split_first is enabled,
			# the depth-0 split is by score sign anyway (>=0 vs <0).
			random.shuffle(search_neuron_ids)
			if search_neuron_ids:
				layer_results = dichotomic_search_layer(
					model,
					layer_label,
					search_neuron_ids,
					pos_prompts,
					neg_prompts,
					is_answer_positive_fn,
					prompt_col,
					args.baseline_subset,
					search_epsilon=search_epsilon,
					batch_size=args.batch_size,
					decode_only=args.decode_only,
					intervention=args.intervention,
					mean_activations=mean_activations,
					prefix_batches=prefix_batches,
					batch_ranges=batch_ranges,
					max_new_tokens=max_new_tokens,
					prune_alpha=args.prune_alpha,
					# Sign-split at depth 0 using discovery scores, if enabled and available.
					importance_scores=(importance_scores_by_circuit.get(rid, {}) or {}).get(layer_label, None),
					first_split_by_importance_sign=getattr(args, 'sign_split_first', False),
				)

				rule_ablation_results += layer_results

				# Update buckets from single-neuron leaves discovered by dichotomic search
				for rec in layer_results:
					if int(rec.get("group_size", 0)) == 1:
						_update_buckets_from_single_record(
							buckets,
							rec,
							baseline_subset=args.baseline_subset,
							circuit_id=rid,
						)

	else:
		excluded_for_full_sweep = _excluded_neuron_keys(
			buckets, 
			exclude_catastrophic_candidates = False, 
			exclude_catastrophic_confirmed = False
		)

		# Full search (single neuron ablations), skipping known buckets.
		for layer_label, neuron_list in tqdm(circuit_units.items(), total=len(circuit_units), desc='Layers', leave=False):
			for neuron in tqdm(neuron_list, desc='Neurons', leave=False):
				key = _neuron_key(layer_label, neuron)
				if key in excluded_for_full_sweep:
					continue

				ablation_record = ablate_neurons(
					model,
					pos_prompts,
					neg_prompts,
					is_answer_positive_fn,
					prompt_col,
					layers_neurons_dict={layer_label: [neuron]},
					batch_size=args.batch_size,
					decode_only=args.decode_only,
					intervention=args.intervention,
					mean_activations=mean_activations,
					prefix_batches=prefix_batches,
					batch_ranges=batch_ranges,
					max_new_tokens=max_new_tokens,
					baseline_subset=args.baseline_subset,
					alpha_slice=ALPHA_SLICE_DEFAULT,
				)
				rule_ablation_results.append(ablation_record)
				_update_buckets_from_single_record(
					buckets,
					ablation_record,
					baseline_subset=args.baseline_subset,
					circuit_id=rid,
				)

				max_effect_val = ablation_record.get("max_effect")
				max_effect_s = (
					f"max_effect={max_effect_val:.3f}"
					if max_effect_val is not None
					else "max_effect=N/A"
				)
				print(
					f"\tLayer {layer_label}, neuron={neuron}: "
					f"accuracy_on_associated={ablation_record['acc_after_knockout_on_associated']:.3f}, "
					f"accuracy_on_unrelated={ablation_record['acc_after_knockout_on_unrelated']:.3f}, "
					f"accuracy_gap={ablation_record['accuracy_gap']:.3f}, "
					f"{max_effect_s}"
				)

	agonist_activation_stats = None
	if not args.skip_agonist_activation_stats:
		agonist_stats_payload = compute_and_save_agonist_activation_stats(
			model,
			rule_out_root_target,
			circuit_id=rid,
			circuit_label=rule,
			prompt_col=prompt_col,
			associated_examples=pos_prompts,
			unrelated_examples=neg_prompts,
			ablation_records=rule_ablation_results,
			baseline_subset=args.baseline_subset,
			batch_size=args.batch_size,
			decode_only=args.decode_only,
			max_new_tokens=max_new_tokens,
		)
		if agonist_stats_payload is not None:
			agonist_activation_stats, _, _ = agonist_stats_payload
			_print_agonist_activation_summary(
				f"	[AgonistAct] circuit={rid}",
				agonist_activation_stats,
			)

	agonist_margin_stats = None
	if not args.skip_agonist_margin_stats:
		agonist_margin_payload = compute_and_save_agonist_margin_stats(
			model,
			task,
			rule_out_root_target,
			circuit_id=rid,
			circuit_label=rule,
			prompt_col=prompt_col,
			associated_examples=pos_prompts,
			unrelated_examples=neg_prompts,
			ablation_records=rule_ablation_results,
			baseline_subset=args.baseline_subset,
			batch_size=args.batch_size,
			intervention=args.intervention,
			mean_activations=mean_activations,
		)
		if agonist_margin_payload is not None:
			agonist_margin_stats, _, _ = agonist_margin_payload
			_print_agonist_margin_summary(
				f"	[AgonistMargin] circuit={rid}",
				agonist_margin_stats,
			)


	agonist_saliency_stats = None
	if not args.skip_agonist_saliency_stats:
		agonist_saliency_payload = compute_and_save_agonist_saliency_stats(
			model,
			task,
			rule_out_root_target,
			circuit_id=rid,
			circuit_label=rule,
			prompt_col=prompt_col,
			associated_examples=pos_prompts,
			unrelated_examples=neg_prompts,
			ablation_records=rule_ablation_results,
			baseline_subset=args.baseline_subset,
			batch_size=args.batch_size,
			metrics=args.agonist_saliency_metrics,
		)
		if agonist_saliency_payload is not None:
			agonist_saliency_stats, _, _ = agonist_saliency_payload
			_print_agonist_saliency_summary(
				f"	[AgonistSaliency] circuit={rid}",
				agonist_saliency_stats,
			)

	# Finalize per-rule detail
	rule_detail.update(
		{
			"status": "ok",
			"n_associated_tested": int(len(pos_prompts)),
			"n_unrelated_tested": int(len(neg_prompts)),
			"baseline_acc_associated": float(baseline_acc_pos),
			"baseline_acc_unrelated": float(baseline_acc_neg),
			"baseline_gap_neg_minus_pos": baseline_gap,
			"sampled_associated_indices": idx_pos_sampled.tolist(),
			"sampled_unrelated_indices": idx_neg_sampled.tolist(),
			"sampling_strategy_used": effective_sampling_strategy,
			"ablations": rule_ablation_results,
			"agonist_activation_stats": agonist_activation_stats,
			"agonist_margin_stats": agonist_margin_stats,
			"agonist_saliency_stats": agonist_saliency_stats,
		}
	)

	# Save per-rule detailed JSON
	rule_out_root_target.write_text(json.dumps(rule_detail, indent=4))

	# Append compact summary for global file
	summary_rule_knockout.append(_summary_entry_from_rule_detail(rule_detail))

	# Try to free memory between rules
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

# Save per-rule ablation summary (compact)
(rule_out_root / "rule_knockout.json").write_text(json.dumps(summary_rule_knockout, indent=4))
_write_global_agonist_activation_outputs(circuit_entries, rule_out_root, args)
_write_global_agonist_saliency_outputs(circuit_entries, rule_out_root, args)


# ------------------------- Persist buckets and print stats -----------------
buckets_path.write_text(json.dumps(buckets, indent=4))

bucket_counts = _bucket_counts(buckets)
bucket_stats = {
	"counts": bucket_counts,
	"excluded_from_future_ablations": int(bucket_counts["catastrophic_zero_candidates"] + bucket_counts["catastrophic_zero_confirmed"] + bucket_counts["non_catastrophic_agonists"]),
}
(rule_out_root / "neuron_bucket_stats.json").write_text(json.dumps(bucket_stats, indent=4))

print(
	"[Buckets] Final stats: "
	f"catastrophic_zero_confirmed={bucket_counts['catastrophic_zero_confirmed']}, "
	f"catastrophic_zero_candidates={bucket_counts['catastrophic_zero_candidates']}, "
	f"catastrophic_zero_not_always={bucket_counts['catastrophic_zero_not_always']}, "
	f"non_catastrophic_agonists={bucket_counts['non_catastrophic_agonists']}"
)
