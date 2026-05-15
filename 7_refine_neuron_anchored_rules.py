import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import textwrap
import json
import argparse
import zipfile
from pathlib import Path
from collections import defaultdict
from more_itertools import unique_everseen

import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import hashlib
import pickle

import scipy.stats

# Local utils (expected to exist in your repo, as in the script you shared)
from lib.modeling_and_ablation import (
	LMWrapper,
	get_device,
	get_layer_type_and_ids, 
	build_ablation_hooks, 
	build_rowwise_ablation_hooks,
	repeat_prefix_cache_batch,
	precompute_mean_activations,
)
from lib.text_and_rules import guess_filetype
from lib.caching_and_prompting import load_or_create_cache, set_deterministic
from lib.data_model_for_shap import run_rule_extraction
from lib.spectral_analysis import *
from lib.feature_extraction_runner import resolve_task_spec

# Neuron intervention utilities (shared with script 6).
# These provide efficient prefix-cached evaluation and a statistically-aware epsilon adjustment.
from lib.neuron_intervention import (
	build_prefix_caches_for_examples,
	get_correctness,
	get_correctness_cached_by_prefix_batches,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- Paper-friendly plotting defaults -----------------
# NeurIPS figures are usually embedded at reduced width; keep type large and PDF text editable.
plt.rcParams.update({
	"font.size": 13,
	"axes.titlesize": 14,
	"axes.labelsize": 13,
	"xtick.labelsize": 11,
	"ytick.labelsize": 11,
	"legend.fontsize": 11,
	"figure.titlesize": 15,
	"pdf.fonttype": 42,
	"ps.fonttype": 42,
	"savefig.dpi": 300,
	"savefig.bbox": "tight",
	"savefig.pad_inches": 0.04,
})

# ----------------- Metric naming helpers -----------------
_RULE_METRIC_SPECS = {
	"mcc": ("MCC", "MCC"),
	"f1": ("F1", "F1"),
	"balanced_accuracy": ("BalancedAcc", "Balanced accuracy"),
	"tpr": ("TPR", "Sensitivity (TPR)"),
	"tnr": ("TNR", "Specificity (TNR)"),
}

def parse_args():
	ap = argparse.ArgumentParser(
		description=(
			"Extend a scores.csv/parquet by adding per-neuron flip columns. "
			"Neurons are discovered from per_rule JSONs as (group_size==1 AND max_effect>=threshold). "
			"Flip is defined as (post_ablation_positive != baseline_is_positive)."
		)
	)
	ap.add_argument("--circuit_agonists_path", type=str, required=True, help="Path to per_rule directory OR a per_rule.zip")
	ap.add_argument(
		"--search_epsilon",
		type=float,
		default=None,
		help=(
			"Base |max_effect| threshold (epsilon) used to keep discovered neurons. "
			"If a record was evaluated on fewer prompts than the reference sizes, "
			"the effective threshold is adjusted upward via get_adjusted_search_epsilon."
		),
	)
	# ap.add_argument("--n_associated_during_neuron_location", type=int, default=100, help="Reference # associated prompts (used for epsilon adjustment; should match discovery script).")
	# ap.add_argument("--n_unrelated_during_neuron_location", type=int, default=100, help="Reference # unrelated prompts (used for epsilon adjustment; should match discovery script).")
	# ap.add_argument("--prune_alpha", type=float, default=0.05, help="Alpha used by get_adjusted_search_epsilon when adjusting epsilon for smaller sample sizes.")
	ap.add_argument("--features_scores_dir", type=str, required=True, help="Directory containing scores.csv and features.json (output of the previous script).")
	ap.add_argument("--ai_model", type=str, default=None, help="HF model id (default: read from dataset_info.json if present)")
	ap.add_argument("--ai_model_cache_dir", default=None, type=str)
	ap.add_argument("--stats_dirname", default=None, type=str)

	ap.add_argument("--stats_only", action="store_true", help="Skip ablations; only compute aggregate statistics from existing flip columns (requires a scores_*.csv with flip_ columns).")
	ap.add_argument("--range_quantiles", type=str, default="0.1,0.9", help="Quantiles (lo,hi) used to report conditional success reduction ranges, e.g. 0.1,0.9.")
	ap.add_argument(
		"--skip_agonist_metric_stats",
		action="store_true",
		help=(
			"Skip final aggregation of script-6 agonist activation/saliency diagnostics "
			"(activation, gradient, Wanda, activation_x_gradient)."
		),
	)
	ap.add_argument(
		"--agonist_metric_topk",
		type=int,
		default=30,
		help="Top-K rows used in the final agonist metric correlation/ranking plots.",
	)
	
	ap.add_argument("--batch_size", type=int, default=16)
	ap.add_argument(
		"--neuron_batch_size",
		type=int,
		default=4,
		help=(
			"How many neurons from the same layer/head to evaluate together by repeating "
			"prompt rows and using row-wise singleton hooks. The lower is this value, the higher the computing overhead."
		),
	)

	ap.add_argument(
		"--no_tqdm_batches",
		action="store_true",
		help="Disable per-batch tqdm bars (only show outer neuron progress).",
	)

	ap.add_argument(
		"--intervention",
		type=str,
		default="zero",
		choices=["zero", "mean", "mean-donor", "mean-positional", "mean-donor-positional"],
		help="Ablation type: zero/mean/mean-donor/mean-positional/mean-donor-positional.",
	)
	ap.add_argument(
		"--points_to_use_for_mean_ablation",
		type=int,
		default=256,
		help="How many prompts to use to estimate replacement activations (mean/mean-donor/mean-positional/mean-donor-positional).",
	)
	ap.add_argument("--last_pos_only", action="store_true", help="If set, ablate only the last token position.")
	ap.add_argument(
		"--decode_only",
		action="store_true",
		help="If set, prefill prompt without hooks and apply ablation hooks only during decoding (faster).",
	)

	ap.add_argument(
		"--max_neurons",
		type=int,
		default=None,
		help="Optional cap on number of neurons to process (useful for quick tests).",
	)
	ap.add_argument("--seed", type=int, default=42)

	# ----------------- spectral sampling options -----------------
	ap.add_argument(
		"--use_spectral_sampling",
		action="store_true",
		help=(
			"If set, use LLM spectral embedding to select a subset of prompts, and run ablations "
			"only on that subset. By default, unsampled datapoints are discarded (no flip propagation)."
		),
	)
	ap.add_argument(
		"--global_n_clusters",
		type=int,
		default=-1,
		help="Number of global clusters (k-center). This is the cardinality knob.",
	)
	ap.add_argument(
		"--keep_unsampled_rows",
		action="store_true",
		help=(
			"Only relevant with --use_spectral_sampling. If set, keep unsampled rows in the output, "
			"but leave flip columns as NA for those rows (still no propagation)."
		),
	)
	
	add_spectral_cli_args(ap)
	
	ap.add_argument(
		"--coverage_radius",
		type=float,
		default=0.5,
		help=(
			"Maximum L2 distance in spectral space between any datapoint and its nearest "
			"sampled center when building the global sampling subset."
		),
	)
	ap.add_argument(
		"--sampling_max_points",
		type=int,
		default=512,
		help="Hard cap on number of prompts evaluated per neuron when spectral sampling is used.",
	)
	ap.add_argument(
		"--sampling_min_points",
		type=int,
		default=64,
		help="Minimum number of sampled prompts when spectral sampling is used.",
	)
	ap.add_argument(
		"--exclude_discovery_rows_from_final_stats",
		action="store_true",
		help=(
			"Before writing final flip statistics, drop rows that were sampled as "
			"associated/unrelated examples during CHA discovery. This removes direct "
			"post-selection overlap without requiring a separate test split."
		),
	)
	ap.add_argument(
		"--sampling_chunk_size",
		type=int,
		default=8192,
		help="Chunk size for nearest-center assignment in spectral sampling.",
	)

	ap.add_argument(
		"--points_per_centroid",
		type=int,
		default=1,
		help="When spectral sampling is used, evaluate X datapoints per centroid (default: 1 = only the centroid).",
	)
	ap.add_argument(
		"--centroid_sampling_mode",
		type=str,
		default="random",
		choices=["random", "nearest"],
		help="How to pick the X datapoints per centroid: random from assigned cluster, or nearest in spectral space.",
	)

	# ----------------- Rule extraction (RuleSHAP-style) -----------------
	ap.add_argument(
		"--extract_rules",
		action="store_true",
		help=(
			"If set, run a RuleSHAP-style rule extraction (similar to 3_extract_arithmetic_rules) "
			"to learn feature-based rules that predict each flip_... target."
		),
	)
	ap.add_argument(
		"--rules_dir",
		type=str,
		default="xai_analyses_results",
		help="Base output directory for extracted rules and SHAP plots.",
	)
	ap.add_argument(
		"--rule_targets",
		type=str,
		default=None,
		help=(
			"Comma-separated list of target columns to extract rules for. "
			"Default: all columns starting with --rule_target_prefix."
		),
	)
	# ap.add_argument(
	# 	"--rule_target_prefix",
	# 	type=str,
	# 	default="flip_c2i,flip_i2c",
	# 	help=(
	# 		"Comma-separated list of prefixes used to auto-select target columns for rule extraction. "
	# 		"Defaults to directional flip targets (flip_c2i_,flip_i2c_)."
	# 	),
	# )
	ap.add_argument(
		"--max_rule_targets",
		type=int,
		default=None,
		help=(
			"Optional cap on number of targets to extract rules for. If set, targets are "
			"sorted by decreasing positive rate before truncation."
		),
	)

	# ----------------- Paper-ready rule metrics + coverage stats -----------------
	ap.add_argument(
		"--summarize_rule_metrics",
		action="store_true",
		help=(
			"Summarize extracted rule_combo_*.csv artifacts into paper-ready tables/plots "
			"(MCC/TPR/TNR + rule lengths) and high-MCC neuron flip-coverage stats."
		),
	)
	ap.add_argument(
		"--summarize_rule_metrics_only",
		action="store_true",
		help=(
			"Only summarize rule metrics/coverage; do not run ablations or extract rules. "
			"Reads existing scores_<postfix>.csv and rule_combo_*.csv under --rules_dir."
		),
	)
	ap.add_argument(
		"--rule_metrics_quantiles",
		type=str,
		default="0.1,0.5,0.9",
		help="Quantiles (lo,med,hi) for MCC/TPR/TNR/length summaries, e.g. 0.1,0.5,0.9.",
	)
	ap.add_argument(
		"--rule_length_cap",
		type=int,
		default=30,
		help="Cap for rule length histogram x-axis (atomic conditions).",
	)
	ap.add_argument(
		"--rule_metrics_topk",
		type=int,
		default=50,
		help="Top-K rules to export in tables (by the chosen rule-quality metric).",
	)

	# ----------------- Threshold-metric knobs -----------------
	# 1) threshold_metric: used by RuleSHAP / rule extraction to choose a probability threshold
	#    (if the underlying run_rule_extraction implementation supports it).
	# 2) rule_quality_metric + rule_quality_threshold: used *only* for this script's paper-ready
	#    summaries (which metric to rank/filter rules/neuron coverage by).
	ap.add_argument(
		"--threshold_metric",
		type=str,
		default="mcc",
		choices=["balanced_accuracy", "mcc", "tpr", "tnr", "f1"],
		help=(
			"Metric used to choose the probability threshold in rule extraction (if supported). "
			"Choices: balanced_accuracy, mcc, tpr, tnr."
		),
	)
	ap.add_argument(
		"--rule_quality_metric",
		type=str,
		default=None,
		choices=["balanced_accuracy", "mcc", "tpr", "tnr", "f1"],
		help=(
			"Metric used in rule-metrics summaries to pick the best rule per neuron and define "
			"'high-quality' neurons. Default: --threshold_metric."
		),
	)
	ap.add_argument(
		"--rule_quality_threshold",
		type=float,
		default=0.85,
		help=(
			"Threshold applied to --rule_quality_metric to count/plot 'high-quality' neurons. "
		),
	)

	# SHAP + RuleSHAP knobs (mirrors 3_extract_arithmetic_rules)
	ap.add_argument(
		"--random_seed",
		type=int,
		default=42,
		help="Specify the random seed (integer)"
	)
	ap.add_argument(
		"--fast_shap_estimate",
		action="store_true",
		help="Faster, approximate SHAP when len(input_features) < 20.",
	)
	ap.add_argument("--npermutations", type=int, default=5)
	ap.add_argument(
		"--only_unique_datapoints_in_shap",
		action="store_true",
		help="Drop duplicated datapoints before SHAP.",
	)
	ap.add_argument(
		"--epsilon",
		type=float,
		default=1e-1,
		help=(
			"The larger is epsilon, the less likely the abstracted model used for SHAP will sample the same x provided in input."
		),
	)
	ap.add_argument(
		"--use_shap_in_xgb",
		action="store_true",
		help="Use SHAP inside XGB during RuleSHAP fitting.",
	)
	ap.add_argument(
		"--use_shap_in_lasso",
		action="store_true",
		help="Use SHAP inside Lasso during RuleSHAP fitting.",
	)
	ap.add_argument(
		"--only_unique_datapoints_in_rule_extraction",
		action="store_true",
		help="Drop duplicated datapoints before rule extraction.",
	)

	# ── Task / domain config ───────────────────────────────────────────────
	g = ap.add_argument_group("task")
	g.add_argument(
		"--task_module",
		default="lib.tasks.arithmetic_task",
		help="Python module path defining parse_prompt, SYSTEM_PROMPT, TOKENS_DICT_KEYS and SEED_FEATURES.",
	)

	return ap.parse_args()

def bucket_keep_keys(buckets: dict, exclude_candidates: bool = True):
	"""
	Return a set of neuron keys like 'm4:458' to keep.
	Policy: keep only non_catastrophic_agonists minus catastrophic buckets.
	"""
	if not buckets:
		return set()

	nc = buckets.get("non_catastrophic_agonists", {})
	keep = set(nc.keys()) if isinstance(nc, dict) else set()

	cz = buckets.get("catastrophic_zero", {})
	if isinstance(cz, dict):
		bad = set()
		for name in ("confirmed", "not_always", "candidates"):
			if name == "candidates" and exclude_candidates:
				pass
			blk = cz.get(name, {})
			if isinstance(blk, dict):
				bad |= set(blk.keys())
		keep -= bad

	return keep

# ------------------------- JSON discovery (per_rule) -------------------------

def _layer_sort_key(layer_label: str):
	# Pull the first integer you can find; fallback to big number.
	nums = re.findall(r"\d+", str(layer_label))
	return int(nums[0]) if nums else 10**9

def _safe_layer_label(layer_label):
	# e.g. "a16.h21" -> "a16_h21"
	return re.sub(r"[^A-Za-z0-9]+", "_", str(layer_label)).strip("_")

def _safe_dirname(s: str) -> str:
	return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")

def canonicalize_neurons(neurons):
	seen = set()
	out = []
	for layer_label, neuron_id, _baseline_subset in neurons:
		key = (str(layer_label), int(neuron_id))
		if key in seen:
			continue
		seen.add(key)
		out.append((str(layer_label), int(neuron_id), "all"))
	return out

def build_flip_cache_policy_tag(args, *, main_metric=None):
	parts = [
		str(main_metric or "metric"),
		str(getattr(args, "intervention", "unknown")),
		"decode_only" if bool(getattr(args, "decode_only", False)) else "prefill_decode",
	]
	if bool(getattr(args, "last_pos_only", False)):
		parts.append("last_pos_only")
	return _safe_dirname("-".join(parts))

def _file_fingerprint(p: Path) -> dict:
	p = Path(p)
	st = p.stat()
	return {
		"name": p.name,
		"size": int(st.st_size),
		"mtime": int(st.st_mtime),
	}

def _spectral_cache_cfg(args, ai_model: str, scores_path: Path, prompt_col: str, split_tag: str) -> dict:
	# Only include knobs that can affect spectral reps/embedding/sampling
	# (plus dataset identity). This keeps cache hits stable.
	keep_keys = {
		# sampling knobs used in THIS script
		"use_spectral_sampling",
		"global_n_clusters",
		"coverage_radius",
		"sampling_max_points",
		"sampling_min_points",
		"sampling_chunk_size",
		"points_per_centroid",
		"centroid_sampling_mode",
		"seed",
		# model/task
		"task_module",
		"ai_model_cache_dir",
	}

	# also keep any args coming from add_spectral_cli_args(ap)
	spectral_like = [k for k in vars(args).keys() if ("spectral" in k) or ("embedding" in k) or ("rep" in k)]
	keep_keys |= set(spectral_like)

	cfg = {k: getattr(args, k) for k in sorted(keep_keys) if hasattr(args, k)}
	cfg.update({
		"ai_model": str(ai_model),
		"scores_path": str(Path(scores_path).resolve()),
		"scores_fingerprint": _file_fingerprint(scores_path),
		"prompt_col": str(prompt_col),
		"split_tag": str(split_tag),
	})
	return cfg

def _spectral_cache_path(args, spectral_cfg=None):
	cache_root = Path(args.rules_dir) / "spectral_cache"
	cache_root.mkdir(parents=True, exist_ok=True)
	if spectral_cfg is None:
		return cache_root / "spectral.pkl"
	cache_hash = _sha1_of_obj(spectral_cfg)[:12]
	return cache_root / f"spectral_{cache_hash}.pkl"


def _replacement_scores_cache_cfg(args, *, ai_model: str, scores_path: Path, prompt_col: str, main_metric: str, layer_to_neurons: dict) -> dict:
	"""
	Cache key for the expensive replacement-stat precompute used by mean-style interventions.
	This is the costly part behind mean/median-style score replacement, so it is worth caching.
	"""
	return {
		"ai_model": str(ai_model),
		"scores_path": str(Path(scores_path).resolve()),
		"scores_fingerprint": _file_fingerprint(scores_path),
		"prompt_col": str(prompt_col),
		"main_metric": str(main_metric),
		"task_module": getattr(args, "task_module", None),
		"intervention": getattr(args, "intervention", None),
		"points_to_use_for_mean_ablation": int(getattr(args, "points_to_use_for_mean_ablation", 0) or 0),
		"batch_size": int(getattr(args, "batch_size", 0) or 0),
		"seed": int(getattr(args, "seed", 0) or 0),
		"layer_to_neurons": {str(k): list(map(int, v)) for k, v in sorted(layer_to_neurons.items())},
	}

def _replacement_scores_cache_path(args, cache_cfg: dict) -> Path:
	cache_root = Path(args.rules_dir) / "replacement_scores_cache"
	cache_root.mkdir(parents=True, exist_ok=True)
	cache_hash = _sha1_of_obj(cache_cfg)[:12]
	return cache_root / f"replacement_scores_{cache_hash}.pkl"

def _build_balanced_mean_prompt_pool(scores_df: pd.DataFrame, *, prompt_col: str, n_points: int, seed: int):
	"""
	Build the prompt pool used to estimate mean activations.

	Flipping is always computed against the global baseline pool ('all'), so this
	always uses all non-test rows. Sampling is deterministic via `seed`.
	"""
	if prompt_col not in scores_df.columns:
		raise ValueError(f"Prompt column {prompt_col!r} not found in scores_df")

	df_pool = scores_df.copy()
	if "is_test" in df_pool.columns:
		df_pool = df_pool.loc[~df_pool["is_test"].astype(bool)].copy()

	if df_pool.empty:
		return []

	if n_points is None or int(n_points) <= 0 or len(df_pool) <= int(n_points):
		sampled_df = df_pool.sample(frac=1.0, random_state=int(seed)) if len(df_pool) > 1 else df_pool
	else:
		sampled_df = df_pool.sample(n=int(n_points), replace=False, random_state=int(seed))

	prompts = sampled_df.loc[:, prompt_col].astype(str).tolist()
	print(
		f"[MeanPool] using {len(prompts)}/{len(df_pool)} prompts from all TRAIN rows "
		"for mean activation estimation."
	)
	return prompts

def extract_single_neurons(
	circuit_agonists_path,
	search_epsilon=0.0,
	exclude_candidates=True,
):
	"""
	Returns a de-duplicated list of (layer_label, neuron_id)
	extracted directly from neuron_buckets.json (dir tree or .zip).
	Keeps only keys in bucket_keep_keys(...), and abs(max_effect) >= threshold.
	"""
	neurons = set()

	def _add_from_buckets(buckets):
		nca = buckets.get("non_catastrophic_agonists", {})
		if not isinstance(nca, dict) or not nca:
			return
		keep = bucket_keep_keys(buckets, exclude_candidates=exclude_candidates) or set(nca)
		for k in keep:
			entry = nca.get(k, {})
			if not isinstance(entry, dict):
				continue
			
			rec = entry["last_record"]
			me = rec["max_effect"]

			# Adjust epsilon to keep selection criteria comparable across different sample sizes.
			eps_eff = float(search_epsilon)
			# if eps_eff > 0:
			# 	n_a = rec.get("n_associated_eval", rec.get("n_associated_during_neuron_location", None))
			# 	n_u = rec.get("n_unrelated_eval", rec.get("n_unrelated_during_neuron_location", None))
			# 	try:
			# 		n_a = int(n_a) if n_a is not None else 0
			# 		n_u = int(n_u) if n_u is not None else 0
			# 	except Exception:
			# 		n_a, n_u = 0, 0
			# 	if n_a > 0 and n_u > 0 and prune_alpha is not None and prune_alpha > 0:
			# 		key = (n_a, n_u)
			# 		eps_eff_cached = eps_cache.get(key)
			# 		if eps_eff_cached is None:
			# 			# get_adjusted_search_epsilon only uses lengths; passing dummy lists keeps signatures aligned with script 6.
			# 			eps_eff_cached = get_adjusted_search_epsilon(
			# 				eps_eff,
			# 				[None] * n_a,
			# 				[None] * n_u,
			# 				n_associated_ref,
			# 				n_unrelated_ref,
			# 				prune_alpha,
			# 			)
			# 			eps_cache[key] = float(eps_eff_cached)
			# 		eps_eff = float(eps_eff_cached)

			if abs(me) < eps_eff:
				continue

			if "layer_label" in rec and "neuron_id" in rec:
				layer, nid = rec["layer_label"], int(rec["neuron_id"])
			else:
				layer, nid = k.split(":", 1)
				nid = int(nid)

			baseline_subset = rec["baseline_subset"]

			neurons.add((
				layer,
				nid,
				baseline_subset
			))

	found = False
	if circuit_agonists_path.is_dir():
		for fp in circuit_agonists_path.rglob("neuron_buckets.json"):
			found = True
			_add_from_buckets(json.loads(fp.read_text(encoding="utf-8")))
	elif circuit_agonists_path.is_file() and circuit_agonists_path.suffix.lower() == ".zip":
		with zipfile.ZipFile(circuit_agonists_path) as zf:
			names = [n for n in zf.namelist() if n.endswith("neuron_buckets.json")]
			found = bool(names)
			for name in names:
				with zf.open(name) as f:
					_add_from_buckets(json.load(f))
	else:
		raise ValueError("circuit_agonists_path must be a directory or a .zip file")

	if not found:
		return []

	neurons = canonicalize_neurons(neurons)
	return sorted(neurons, reverse=True, key=lambda x: (_layer_sort_key(x[0]), int(x[1]), x[2]))


def _iter_discovery_payloads(circuit_agonists_path: Path):
	"""Yield per-rule/per-cluster JSON payloads that may contain discovery sampled indices."""
	if circuit_agonists_path.is_dir():
		for fp in circuit_agonists_path.rglob("*.json"):
			if fp.name in {"neuron_buckets.json", "neuron_bucket_stats.json", "rule_knockout.json"}:
				continue
			try:
				payload = json.loads(fp.read_text(encoding="utf-8"))
			except Exception:
				continue
			if isinstance(payload, dict):
				yield payload
	elif circuit_agonists_path.is_file() and circuit_agonists_path.suffix.lower() == ".zip":
		with zipfile.ZipFile(circuit_agonists_path) as zf:
			for name in zf.namelist():
				base = Path(name).name
				if not name.endswith(".json") or base in {"neuron_buckets.json", "neuron_bucket_stats.json", "rule_knockout.json"}:
					continue
				try:
					with zf.open(name) as f:
						payload = json.load(f)
				except Exception:
					continue
				if isinstance(payload, dict):
					yield payload


def collect_discovery_original_rows(circuit_agonists_path: Path, scores_df: pd.DataFrame) -> set[int]:
	"""
	Return original scores.csv row positions used as discovery ablation examples.

	Script 6 samples associated/unrelated indices after dropping is_test rows and
	resetting the train dataframe index. Therefore, when is_test is present, the
	stored sampled_*_indices must be mapped back through the train-row positions
	of the full scores.csv used here. If future artifacts store sampled_*_original_idx
	directly, those are preferred.
	"""
	if "is_test" in scores_df.columns:
		train_orig_rows = np.where(~scores_df["is_test"].astype(bool).to_numpy())[0].astype(int)
	else:
		train_orig_rows = np.arange(len(scores_df), dtype=int)

	out: set[int] = set()
	for payload in _iter_discovery_payloads(circuit_agonists_path):
		for key in ("sampled_associated_original_idx", "sampled_unrelated_original_idx"):
			vals = payload.get(key, [])
			if isinstance(vals, list):
				for v in vals:
					try:
						out.add(int(v))
					except Exception:
						pass

		for key in ("sampled_associated_indices", "sampled_unrelated_indices"):
			vals = payload.get(key, [])
			if not isinstance(vals, list):
				continue
			for v in vals:
				try:
					i = int(v)
				except Exception:
					continue
				if 0 <= i < len(train_orig_rows):
					out.add(int(train_orig_rows[i]))
	return out


def filter_scores_out_for_final_stats(scores_out: pd.DataFrame, discovery_orig_rows: set[int]) -> pd.DataFrame:
	"""Drop discovery-overlap rows from the dataframe used only for final stats."""
	if not discovery_orig_rows:
		return scores_out
	if "_orig_row" in scores_out.columns:
		orig_rows = pd.to_numeric(scores_out["_orig_row"], errors="coerce")
	else:
		orig_rows = pd.Series(np.arange(len(scores_out), dtype=int), index=scores_out.index)
	mask = ~orig_rows.astype("Int64").isin(discovery_orig_rows).to_numpy()
	return scores_out.loc[mask].copy()


# ------------------------- Correctness eval -----------------------------
def _sha1_of_obj(obj) -> str:
	"""Stable SHA1 over a JSON-serializable object."""
	payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
	return hashlib.sha1(payload).hexdigest()

def _series_value_hash(s: pd.Series) -> int:
	"""
	Stable-ish hash of a series' values for cache validation.
	- Coerces to numeric when possible
	- Treats NA as a sentinel
	Returns a Python int (sum of uint64 hashes).
	"""
	v = pd.to_numeric(s, errors="coerce")
	v = v.copy()
	# Keep NA distinct from False/0
	if v.dtype == "boolean":
		v = v.astype("float")
	v = v.fillna(-1234567)
	h = pd.util.hash_pandas_object(v, index=False)
	return int(h.sum())

def build_rule_extraction_signature(rules_df: pd.DataFrame, input_features, targets, args, features_json_fingerprint=None) -> dict:
	"""
	Compute a lightweight signature so we can skip rule extraction when outputs already exist
	and the inputs haven't changed.

	Note: we intentionally avoid hashing every feature column (too expensive per neuron).
	We gate on:
	- the target column values (hash)
	- the list of input feature names (hash)
	- a fingerprint of features.json (size+mtime)
	- key rule-extraction hyperparameters
	"""
	target_hashes = {}
	for t in targets:
		if t in rules_df.columns:
			target_hashes[t] = _series_value_hash(rules_df[t])

	relevant_args = {
		"random_seed": getattr(args, "random_seed", None),
		"fast_shap_estimate": bool(getattr(args, "fast_shap_estimate", False)),
		"npermutations": int(getattr(args, "npermutations", 0)),
		"only_unique_datapoints_in_shap": bool(getattr(args, "only_unique_datapoints_in_shap", False)),
		"epsilon": float(getattr(args, "epsilon", 0.0)),
		"use_shap_in_xgb": bool(getattr(args, "use_shap_in_xgb", False)),
		"use_shap_in_lasso": bool(getattr(args, "use_shap_in_lasso", False)),
		"only_unique_datapoints_in_rule_extraction": bool(getattr(args, "only_unique_datapoints_in_rule_extraction", False)),
	}

	input_feature_names = list(map(str, input_features))
	return {
		"n_rows": int(len(rules_df)),
		"targets": list(map(str, targets)),
		"target_hashes": target_hashes,
		"input_features_sha1": _sha1_of_obj(input_feature_names),
		"features_json_fingerprint": features_json_fingerprint,
		"args": relevant_args,
	}

def _rule_signature_path(rules_subdir):
	return Path(rules_subdir) / "_rule_extraction_signature.json"

def should_skip_rule_extraction(rules_subdir, prefix, layer_label, signature=None):
	"""
	Skip per-neuron rule extraction only when both the expected artifact exists
	and the stored signature matches the current inputs/config.
	"""
	rules_path = Path(rules_subdir)
	artifact_path = rules_path / f"optimal_rule_set_{prefix}{layer_label}.csv"
	if not artifact_path.exists():
		return False
	if signature is None:
		return True
	sig_path = _rule_signature_path(rules_subdir)
	if not sig_path.exists():
		return False
	try:
		stored = json.loads(sig_path.read_text(encoding="utf-8"))
	except Exception:
		return False
	return stored == signature

def write_flip_stats(scores_df: pd.DataFrame, neurons_sorted, out_dir, topk = 50, stats_dirname=''):
	"""
	Write per-neuron counts of c2i and i2c flips, and a bar plot for the top-K neurons
	(by total flips).
	Outputs:
	  - <out_dir>/flip_stats_by_neuron.csv
	  - <out_dir>/flip_stats_top{K}.pdf
	  - <out_dir>/flip_stats_global.json
	"""
	out_dir = str(out_dir)
	Path(out_dir).mkdir(parents=True, exist_ok=True)

	rows = []
	any_eval_col = None
	cols_any_found = []
	cols_c2i_found = []
	cols_i2c_found = []

	for layer_label, neuron_id, _ in unique_everseen(neurons_sorted, key=lambda x: (x[0],x[1])):
		layer_key = _safe_layer_label(layer_label)
		col_any = f"flip_{layer_key}_{neuron_id}"
		col_c2i = f"flip_c2i_{layer_key}_{neuron_id}"
		col_i2c = f"flip_i2c_{layer_key}_{neuron_id}"

		if col_any not in scores_df.columns:
			continue
		cols_any_found.append(col_any)
		if col_c2i in scores_df.columns:
			cols_c2i_found.append(col_c2i)
		if col_i2c in scores_df.columns:
			cols_i2c_found.append(col_i2c)
		if any_eval_col is None:
			any_eval_col = col_any

		n_eval = int(scores_df[col_any].notna().sum())
		c2i = int(scores_df[col_c2i].fillna(False).astype(int).sum()) if col_c2i in scores_df.columns else 0
		i2c = int(scores_df[col_i2c].fillna(False).astype(int).sum()) if col_i2c in scores_df.columns else 0
		total = int(scores_df[col_any].fillna(False).astype(int).sum())

		# Percentages are always relative to the number of evaluated rows for this neuron.
		c2i_pct = (100.0 * c2i / n_eval) if n_eval else float("nan")
		i2c_pct = (100.0 * i2c / n_eval) if n_eval else float("nan")
		total_pct = (100.0 * total / n_eval) if n_eval else float("nan")

		rows.append({
			"layer_label": str(layer_label),
			"layer_key": str(layer_key),
			"neuron_id": int(neuron_id),
			"neuron": f"{layer_label}:{neuron_id}",
			"n_eval": n_eval,
			"c2i_count": c2i,
			"i2c_count": i2c,
			"flip_any_count": total,
			"c2i_rate": (c2i / n_eval) if n_eval else float("nan"),
			"i2c_rate": (i2c / n_eval) if n_eval else float("nan"),
			"flip_any_rate": (total / n_eval) if n_eval else float("nan"),
			"c2i_pct": c2i_pct,
			"i2c_pct": i2c_pct,
			"flip_any_pct": total_pct,
		})

	if not rows:
		print("[Stats] No flip columns found; skipping flip stats.")
		return None

	stats_df = pd.DataFrame(rows).sort_values(["flip_any_count", "c2i_count", "i2c_count"], ascending=False)
	stats_path = os.path.join(out_dir, 'stats', stats_dirname, f"flip_stats_by_neuron.csv")
	stats_df.to_csv(stats_path, index=False)
	print(f"[Stats] Wrote {stats_path}")

	eval_rows = int(scores_df[any_eval_col].notna().sum()) if any_eval_col else 0

	# Compute *unique* flipped datapoints across all neurons (union), to avoid double-counting
	# when the same datapoint flips for multiple neurons.
	def _union_counts(cols):
		if not cols:
			return 0, 0
		union = np.zeros(len(scores_df), dtype=bool)
		any_eval = np.zeros(len(scores_df), dtype=bool)
		for c in cols:
			if c not in scores_df.columns:
				continue
			s = scores_df[c]
			any_eval |= s.notna().to_numpy()
			union |= s.fillna(False).astype(bool).to_numpy()
		return int(union.sum()), int(any_eval.sum())

	union_any, eval_any = _union_counts(cols_any_found)
	union_c2i, eval_c2i = _union_counts(cols_c2i_found)
	union_i2c, eval_i2c = _union_counts(cols_i2c_found)
	global_payload = {
		"n_neurons": int(len(stats_df)),
		"n_evaluated_rows": eval_rows,
		"sum_c2i_counts_over_neurons": int(stats_df["c2i_count"].sum()),
		"sum_i2c_counts_over_neurons": int(stats_df["i2c_count"].sum()),
		"sum_flip_any_counts_over_neurons": int(stats_df["flip_any_count"].sum()),
		"union_flip_any_unique_count": int(union_any),
		"union_flip_any_unique_rate": (float(union_any) / float(eval_any)) if eval_any else float("nan"),
		"union_c2i_unique_count": int(union_c2i),
		"union_c2i_unique_rate": (float(union_c2i) / float(eval_c2i)) if eval_c2i else float("nan"),
		"union_i2c_unique_count": int(union_i2c),
		"union_i2c_unique_rate": (float(union_i2c) / float(eval_i2c)) if eval_i2c else float("nan"),
	}
	Path(os.path.join(out_dir, 'stats', stats_dirname, f"flip_stats_global.json")).write_text(
		json.dumps(global_payload, indent=2, ensure_ascii=False),
		encoding="utf-8",
	)
	print(f"[Stats] Wrote {os.path.join(out_dir, 'stats', stats_dirname, f'flip_stats_global.json')}")

	K = int(min(topk, len(stats_df)))

	# For papers, 50 neurons is usually too dense unless the figure is full-width.
	# Keep topk configurable, but consider calling write_flip_stats(..., topk=25)
	# for the camera-ready figure.
	dfp = stats_df.head(K).copy()

	import matplotlib.ticker as mticker

	y = np.arange(K)

	c2i = dfp["c2i_pct"].to_numpy()
	i2c = dfp["i2c_pct"].to_numpy()

	# Compact, paper-friendly layout while preserving large readable fonts.
	# The caption can explain details, so we minimize extra vertical overhead.
	fig_w = 7.2
	# Keep top-K labels readable. For K=50 this is still full-width-paper friendly,
	# but it avoids compressed neuron labels after scaling.
	fig_h = max(3.45, 0.235 * K + 1.25)
	ytick_fs = 10 if K <= 30 else (9.2 if K <= 40 else 8.6)

	with plt.rc_context({
		"font.size": 10.5,
		"axes.labelsize": 10.5,
		"axes.titlesize": 11,
		"xtick.labelsize": 10,
		"ytick.labelsize": ytick_fs,
		"legend.fontsize": 9.6,
		"pdf.fonttype": 42,
		"ps.fonttype": 42,
	}):
		fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=False)
		# Keep only a narrow top band: enough for the legend, without the large blank gap.
		fig.subplots_adjust(left=0.31, right=0.99, bottom=0.095, top=0.930)

		# Diverging bars: c2i on the left, i2c on the right.
		ax.barh(
			y,
			-c2i,
			height=0.62,
			label="Correct→Incorrect",
			linewidth=0.22,
			edgecolor="black",
		)
		ax.barh(
			y,
			i2c,
			height=0.62,
			label="Incorrect→Correct",
			linewidth=0.22,
			edgecolor="black",
		)

		ax.axvline(0, linewidth=0.6, color="black")

		ax.set_yticks(y)
		ax.set_yticklabels(dfp["neuron"].values)
		ax.tick_params(axis="y", pad=2)
		ax.invert_yaxis()
		# Leave only a minimal data-space gutter above the first row.
		ax.set_ylim(K - 0.5, -0.62)

		ax.set_xlabel("Flipped datapoints (% of evaluated prompts)")
		# Omit the y-axis label and title to keep the figure tighter in-paper.

		# Show absolute values on both sides of the diverging axis.
		ax.xaxis.set_major_formatter(
			mticker.FuncFormatter(lambda v, _: f"{abs(v):g}")
		)

		max_pct = float(np.nanmax([c2i.max(initial=0), i2c.max(initial=0)]))
		x_lim = 1.06 * max_pct if max_pct > 0 else 1.0
		ax.set_xlim(-x_lim, x_lim)

		ax.grid(axis="x", linewidth=0.3, alpha=0.4)
		ax.set_axisbelow(True)

		# Put the legend just outside the axes, close to the bars, instead of
		# reserving a tall figure-level legend band.
		ax.legend(
			loc="lower center",
			bbox_to_anchor=(0.5, 1.012),
			ncol=2,
			frameon=False,
			handlelength=1.3,
			columnspacing=1.0,
			borderaxespad=0.0,
		)

		# Remove unnecessary visual weight.
		ax.spines["top"].set_visible(False)
		ax.spines["right"].set_visible(False)

		plot_path = os.path.join(out_dir, 'stats', stats_dirname, f"flip_stats_top{K}.pdf")
		fig.savefig(plot_path, bbox_inches="tight", pad_inches=0.03)
		plt.close(fig)

	print(f"[Stats] Wrote {plot_path}")

	return stats_df

# ------------------------- Final agonist metric diagnostics -------------------------

_METRIC_DISPLAY = {
	"activation": "activation",
	"gradient": "|gradient|",
	"wanda": "Wanda",
	"activation_x_gradient": "|activation x gradient|",
	"activation_gradient": "|activation x gradient|",
	"gradxactivation": "|activation x gradient|",
}

_FINAL_METRIC_ORDER = ["activation", "gradient", "wanda", "activation_x_gradient"]


def _metric_sort_key(metric: str):
	metric = str(metric)
	try:
		return (_FINAL_METRIC_ORDER.index(metric), metric)
	except ValueError:
		return (len(_FINAL_METRIC_ORDER), metric)


def _metric_display(metric: str):
	return _METRIC_DISPLAY.get(str(metric), str(metric))


_PRETTY_COL_LABELS = {
	"flip_any_rate": "Any flip rate",
	"c2i_rate": "Correct→incorrect rate",
	"i2c_rate": "Incorrect→correct rate",
	"flip_any_count": "Any flip count",
	"c2i_count": "Correct→incorrect count",
	"i2c_count": "Incorrect→correct count",
	"max_effect": "Max effect",
	"abs_max_effect": "|Max effect|",
	"accuracy_gap": "Accuracy gap",
	"abs_accuracy_gap": "|Accuracy gap|",
	"mean_metric_value": "Mean metric value",
	"mean_abs_metric_value": "Mean |metric| value",
	"median_metric_value": "Median metric value",
	"mean_layer_percentile_rank": "Mean layer percentile rank",
	"median_layer_percentile_rank": "Median layer percentile rank",
	"mean_layer_zscore": "Mean layer z-score",
	"mean_delta_from_layer_mean": "Mean delta from layer mean",
	"top1_rate": "Top-1 rate",
	"top5_rate": "Top-5 rate",
	"top10pct_rate": "Top-10% rate",
	"delta_mean_metric_value": "Δ mean metric value",
	"delta_mean_abs_metric_value": "Δ mean |metric| value",
	"delta_mean_layer_percentile_rank": "Δ mean layer percentile rank",
	"delta_mean_layer_zscore": "Δ mean layer z-score",
	"delta_top10pct_rate": "Δ top-10% rate",
}

def _pretty_col_label(name: str) -> str:
	"""Human-readable axis/legend label for dataframe column names."""
	s = str(name)
	if s in _PRETTY_COL_LABELS:
		return _PRETTY_COL_LABELS[s]
	# Preserve useful mathematical prefixes; otherwise remove implementation underscores.
	s = s.replace("delta_", "Δ ")
	s = s.replace("c2i", "correct→incorrect").replace("i2c", "incorrect→correct")
	s = s.replace("_pct", " %").replace("_rate", " rate").replace("_count", " count")
	s = s.replace("_", " ")
	return s[:1].upper() + s[1:]

def _wrap_label(label: str, width: int = 18) -> str:
	"""Wrap long tick labels into compact multi-line labels."""
	return "\n".join(textwrap.wrap(str(label), width=width, break_long_words=False, break_on_hyphens=False))


def _read_csvs_from_dir_or_zip(root_path, suffix):
	"""Read all per-circuit script-6 CSVs ending in suffix from a directory or zip."""
	root_path = Path(root_path)
	frames = []
	if root_path.is_dir():
		for fp in sorted(root_path.rglob(f"*{suffix}")):
			if fp.name == suffix.lstrip("_"):
				continue
			try:
				df = pd.read_csv(fp)
			except Exception as e:
				print(f"[AgonistMetrics] Could not read {fp}: {e}")
				continue
			df["source_file"] = str(fp)
			df["source_parent"] = str(fp.parent)
			frames.append(df)
	elif root_path.is_file() and root_path.suffix.lower() == ".zip":
		with zipfile.ZipFile(root_path) as zf:
			for name in sorted(n for n in zf.namelist() if n.endswith(suffix)):
				base = Path(name).name
				if "__MACOSX" in name or base == suffix.lstrip("_"):
					continue
				try:
					with zf.open(name) as f:
						df = pd.read_csv(f)
				except Exception as e:
					print(f"[AgonistMetrics] Could not read {name} from {root_path}: {e}")
					continue
				df["source_file"] = str(name)
				df["source_parent"] = str(Path(name).parent)
				frames.append(df)
	else:
		return []
	return frames


def _normalise_agonist_metric_summary(circuit_agonists_path):
	"""Load script-6 activation/saliency summaries into one metric-indexed table."""
	frames = []

	for df in _read_csvs_from_dir_or_zip(circuit_agonists_path, "_agonist_activation_summary.csv"):
		if df.empty:
			continue
		df = df.copy()
		df["metric"] = "activation"
		if "mean_metric_value" not in df.columns and "mean_activation_value" in df.columns:
			df["mean_metric_value"] = pd.to_numeric(df["mean_activation_value"], errors="coerce")
		if "median_metric_value" not in df.columns and "median_activation_value" in df.columns:
			df["median_metric_value"] = pd.to_numeric(df["median_activation_value"], errors="coerce")
		if "mean_abs_metric_value" not in df.columns:
			if "mean_abs_activation_value" in df.columns:
				df["mean_abs_metric_value"] = pd.to_numeric(df["mean_abs_activation_value"], errors="coerce")
			else:
				df["mean_abs_metric_value"] = pd.to_numeric(df.get("mean_metric_value"), errors="coerce").abs()
		frames.append(df)

	for df in _read_csvs_from_dir_or_zip(circuit_agonists_path, "_agonist_saliency_summary.csv"):
		if df.empty:
			continue
		df = df.copy()
		if "metric" not in df.columns:
			continue
		df["metric"] = df["metric"].replace({
			"activation_gradient": "activation_x_gradient",
			"actgrad": "activation_x_gradient",
			"gradxactivation": "activation_x_gradient",
		})
		if "mean_metric_value" not in df.columns and "mean_saliency_value" in df.columns:
			df["mean_metric_value"] = pd.to_numeric(df["mean_saliency_value"], errors="coerce")
		if "median_metric_value" not in df.columns and "median_saliency_value" in df.columns:
			df["median_metric_value"] = pd.to_numeric(df["median_saliency_value"], errors="coerce")
		if "mean_abs_metric_value" not in df.columns:
			df["mean_abs_metric_value"] = pd.to_numeric(df.get("mean_metric_value"), errors="coerce").abs()
		frames.append(df)

	if not frames:
		return pd.DataFrame()

	df = pd.concat(frames, ignore_index=True, sort=False)
	if df.empty:
		return df

	if "unit_id" in df.columns and "neuron_id" not in df.columns:
		df["neuron_id"] = pd.to_numeric(df["unit_id"], errors="coerce").astype("Int64")
	if "layer_label" in df.columns and "layer_key" not in df.columns:
		df["layer_key"] = df["layer_label"].map(_safe_layer_label)
	if "unit_key" not in df.columns and set(["layer_label", "neuron_id"]).issubset(df.columns):
		df["unit_key"] = df["layer_label"].astype(str) + ":" + df["neuron_id"].astype(str)

	for col in (
		"max_effect", "accuracy_gap", "mean_metric_value", "median_metric_value", "mean_abs_metric_value",
		"mean_layer_percentile_rank", "median_layer_percentile_rank", "mean_layer_zscore",
		"mean_delta_from_layer_mean", "top1_rate", "top5_rate", "top10pct_rate",
	):
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")
	if "max_effect" in df.columns and "abs_max_effect" not in df.columns:
		df["abs_max_effect"] = df["max_effect"].abs()
	if "accuracy_gap" in df.columns and "abs_accuracy_gap" not in df.columns:
		df["abs_accuracy_gap"] = df["accuracy_gap"].abs()

	df["metric"] = df["metric"].astype(str)
	df["metric_display"] = df["metric"].map(_metric_display)
	return df.reset_index(drop=True)


def _load_existing_flip_stats(out_dir, stats_dirname=""):
	fp = Path(out_dir) / "stats" / str(stats_dirname or "") / "flip_stats_by_neuron.csv"
	if not fp.exists():
		return None
	try:
		return pd.read_csv(fp)
	except Exception as e:
		print(f"[AgonistMetrics] Could not read existing flip stats {fp}: {e}")
		return None


def _merge_flip_stats(metric_df, flip_stats_df):
	if metric_df is None or metric_df.empty or flip_stats_df is None or len(flip_stats_df) == 0:
		return metric_df
	if "layer_key" not in metric_df.columns or "neuron_id" not in metric_df.columns:
		return metric_df
	fs = flip_stats_df.copy()
	if "layer_key" not in fs.columns and "layer_label" in fs.columns:
		fs["layer_key"] = fs["layer_label"].map(_safe_layer_label)
	if "neuron_id" not in fs.columns:
		return metric_df
	fs["layer_key"] = fs["layer_key"].astype(str)
	fs["neuron_id"] = pd.to_numeric(fs["neuron_id"], errors="coerce").astype("Int64")
	keep = [c for c in [
		"layer_key", "neuron_id", "n_eval", "c2i_count", "i2c_count", "flip_any_count",
		"c2i_rate", "i2c_rate", "flip_any_rate", "c2i_pct", "i2c_pct", "flip_any_pct",
	] if c in fs.columns]
	fs = fs[keep].drop_duplicates(subset=["layer_key", "neuron_id"])
	out = metric_df.copy()
	out["layer_key"] = out["layer_key"].astype(str)
	out["neuron_id"] = pd.to_numeric(out["neuron_id"], errors="coerce").astype("Int64")
	return out.merge(fs, on=["layer_key", "neuron_id"], how="left", suffixes=("", "_flip"))


def _numeric_corr_pair(x, y, method="spearman"):
	x = pd.to_numeric(pd.Series(x), errors="coerce")
	y = pd.to_numeric(pd.Series(y), errors="coerce")
	mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
	n = int(mask.sum())
	if n < 3:
		return None, n
	xv = x[mask]
	yv = y[mask]
	if xv.nunique(dropna=True) < 2 or yv.nunique(dropna=True) < 2:
		return None, n
	val = xv.corr(yv, method=method)
	if val is None or not np.isfinite(float(val)):
		return None, n
	return float(val), n


def _compute_metric_delta_rows(metric_df):
	if metric_df is None or metric_df.empty or "split" not in metric_df.columns:
		return pd.DataFrame()
	key_cols = [c for c in ["circuit_id", "source_parent", "unit_key", "layer_label", "layer_key", "neuron_id", "metric", "metric_display"] if c in metric_df.columns]
	if not key_cols or "metric" not in key_cols:
		return pd.DataFrame()
	agg_cols = key_cols + ["split"]
	numeric_cols = [c for c in metric_df.select_dtypes(include=[np.number]).columns.tolist() if c not in agg_cols]
	if not numeric_cols:
		return pd.DataFrame()
	g = metric_df.groupby(agg_cols, dropna=False, as_index=False)[numeric_cols].mean()
	assoc = g.loc[g["split"].astype(str) == "associated"].copy()
	unrel = g.loc[g["split"].astype(str) == "unrelated"].copy()
	if assoc.empty or unrel.empty:
		return pd.DataFrame()
	merged = assoc.merge(unrel, on=key_cols, suffixes=("_associated", "_unrelated"))
	if merged.empty:
		return pd.DataFrame()
	features = [
		"mean_metric_value", "median_metric_value", "mean_abs_metric_value",
		"mean_layer_percentile_rank", "median_layer_percentile_rank", "mean_layer_zscore",
		"mean_delta_from_layer_mean", "top1_rate", "top5_rate", "top10pct_rate",
	]
	for feature in features:
		ca = f"{feature}_associated"
		cu = f"{feature}_unrelated"
		if ca in merged.columns and cu in merged.columns:
			merged[f"delta_{feature}"] = pd.to_numeric(merged[ca], errors="coerce") - pd.to_numeric(merged[cu], errors="coerce")
	for target in ["max_effect", "abs_max_effect", "accuracy_gap", "abs_accuracy_gap", "flip_any_rate", "c2i_rate", "i2c_rate", "flip_any_count", "c2i_count", "i2c_count"]:
		ca = f"{target}_associated"
		cu = f"{target}_unrelated"
		if ca in merged.columns:
			merged[target] = merged[ca]
		elif cu in merged.columns:
			merged[target] = merged[cu]
	return merged


def _compute_metric_correlations(metric_df, delta_df):
	features = [
		"mean_metric_value", "median_metric_value", "mean_abs_metric_value",
		"mean_layer_percentile_rank", "median_layer_percentile_rank", "mean_layer_zscore",
		"mean_delta_from_layer_mean", "top1_rate", "top5_rate", "top10pct_rate",
	]
	delta_features = [f"delta_{f}" for f in features]
	targets = ["max_effect", "abs_max_effect", "accuracy_gap", "abs_accuracy_gap", "flip_any_rate", "c2i_rate", "i2c_rate", "flip_any_count"]
	rows = []
	if metric_df is not None and not metric_df.empty:
		for metric, mdf in metric_df.groupby("metric", sort=False):
			for split, sdf in mdf.groupby("split", sort=False):
				for feature in features:
					if feature not in sdf.columns:
						continue
					for target in targets:
						if target not in sdf.columns:
							continue
						pearson, n_p = _numeric_corr_pair(sdf[feature], sdf[target], "pearson")
						spearman, n_s = _numeric_corr_pair(sdf[feature], sdf[target], "spearman")
						if pearson is None and spearman is None:
							continue
						rows.append({
							"scope": str(split), "metric": str(metric), "metric_display": _metric_display(metric),
							"feature": feature, "target": target, "n": int(max(n_p, n_s)),
							"pearson": pearson, "spearman": spearman, "abs_spearman": abs(spearman) if spearman is not None else np.nan,
						})
	if delta_df is not None and not delta_df.empty:
		for metric, mdf in delta_df.groupby("metric", sort=False):
			for feature in delta_features:
				if feature not in mdf.columns:
					continue
				for target in targets:
					if target not in mdf.columns:
						continue
					pearson, n_p = _numeric_corr_pair(mdf[feature], mdf[target], "pearson")
					spearman, n_s = _numeric_corr_pair(mdf[feature], mdf[target], "spearman")
					if pearson is None and spearman is None:
						continue
					rows.append({
						"scope": "associated_minus_unrelated", "metric": str(metric), "metric_display": _metric_display(metric),
						"feature": feature, "target": target, "n": int(max(n_p, n_s)),
						"pearson": pearson, "spearman": spearman, "abs_spearman": abs(spearman) if spearman is not None else np.nan,
					})
	if not rows:
		return pd.DataFrame()
	out = pd.DataFrame(rows)
	return out.sort_values(["target", "scope", "abs_spearman"], ascending=[True, True, False]).reset_index(drop=True)


def _compute_metric_global_summary(metric_df):
	if metric_df is None or metric_df.empty:
		return pd.DataFrame()
	agg = {}
	for col in ["unit_key", "layer_key", "neuron_id"]:
		if col in metric_df.columns:
			agg[col] = "nunique"
	for col in ["mean_metric_value", "mean_abs_metric_value", "mean_layer_percentile_rank", "median_layer_percentile_rank", "mean_layer_zscore", "top1_rate", "top5_rate", "top10pct_rate", "max_effect", "abs_max_effect", "flip_any_rate", "c2i_rate", "i2c_rate"]:
		if col in metric_df.columns:
			agg[col] = ["mean", "median", "std"]
	if not agg:
		return pd.DataFrame()
	g = metric_df.groupby(["metric", "metric_display", "split"], dropna=False).agg(agg)
	g.columns = ["_".join([str(x) for x in tup if str(x)]) for tup in g.columns.to_flat_index()]
	g = g.reset_index()
	g["n_rows"] = metric_df.groupby(["metric", "metric_display", "split"], dropna=False).size().to_numpy()
	return g.reset_index(drop=True)


def _save_metric_plots(metric_df, delta_df, corr_df, stats_dir, topk=30):
	if metric_df is None or metric_df.empty:
		return []
	from matplotlib.backends.backend_pdf import PdfPages
	paths = []
	stats_dir = Path(stats_dir)
	metric_order = [m for m in _FINAL_METRIC_ORDER if m in set(metric_df["metric"].astype(str))]
	metric_order += sorted([m for m in set(metric_df["metric"].astype(str)) if m not in metric_order])
	split_order = [s for s in ["associated", "unrelated"] if s in set(metric_df["split"].astype(str))]
	pdf_path = stats_dir / "agonist_metric_final_plots.pdf"
	with PdfPages(pdf_path) as pdf:
		if "mean_layer_percentile_rank" in metric_df.columns and split_order:
			fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(metric_order) * max(1, len(split_order))), 5.2))
			box_data, labels = [], []
			for metric in metric_order:
				for split in split_order:
					vals = pd.to_numeric(metric_df.loc[(metric_df["metric"] == metric) & (metric_df["split"] == split), "mean_layer_percentile_rank"], errors="coerce").dropna().to_numpy()
					if vals.size:
						box_data.append(vals)
						labels.append(f"{_metric_display(metric)}\n{split}")
			if box_data:
				ax.boxplot(box_data, tick_labels=labels, showmeans=True)
				ax.set_ylim(0, 1.02)
				ax.set_ylabel("Mean within-layer percentile rank")
				ax.set_title("Agonist metric rank distributions")
				ax.grid(True, axis="y", alpha=0.3)
				fig.tight_layout()
				png = stats_dir / "agonist_metric_percentile_boxplot.pdf"
				fig.savefig(png, dpi=300, bbox_inches="tight")
				fig.savefig(png.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
				pdf.savefig(fig, bbox_inches="tight")
				paths.append(str(png))
				paths.append(str(png.with_suffix(".pdf")))
			plt.close(fig)
		rate_cols = [c for c in ["top1_rate", "top5_rate", "top10pct_rate"] if c in metric_df.columns]
		if rate_cols:
			fig, ax = plt.subplots(figsize=(max(10, 1.0 * len(metric_order) * len(rate_cols)), 5.2))
			x = np.arange(len(metric_order))
			width = 0.8 / max(1, len(rate_cols) * max(1, len(split_order)))
			idx = 0
			for split in split_order or [None]:
				for rate_col in rate_cols:
					vals = []
					for metric in metric_order:
						sub = metric_df.loc[metric_df["metric"] == metric]
						if split is not None:
							sub = sub.loc[sub["split"] == split]
						vals.append(float(pd.to_numeric(sub[rate_col], errors="coerce").mean()) if len(sub) else np.nan)
					offset = (idx - (len(rate_cols) * max(1, len(split_order)) - 1) / 2.0) * width
					label = _pretty_col_label(rate_col) + (f" / {split}" if split is not None else "")
					ax.bar(x + offset, vals, width=width, label=label)
					idx += 1
			ax.set_xticks(x)
			ax.set_xticklabels([_metric_display(m) for m in metric_order], rotation=20, ha="right")
			ax.set_ylim(0, 1.02)
			ax.set_ylabel("Rate")
			ax.set_title("How often agonists are top-ranked by each metric")
			ax.grid(True, axis="y", alpha=0.3)
			ax.legend(fontsize=10, ncols=2)
			fig.tight_layout()
			png = stats_dir / "agonist_metric_top_rates.pdf"
			fig.savefig(png, dpi=300, bbox_inches="tight")
			fig.savefig(png.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
			pdf.savefig(fig, bbox_inches="tight")
			paths.append(str(png))
			paths.append(str(png.with_suffix(".pdf")))
			plt.close(fig)
		if corr_df is not None and not corr_df.empty:
			target_preference = ["flip_any_rate", "abs_max_effect", "c2i_rate", "i2c_rate", "accuracy_gap"]
			target = next((t for t in target_preference if t in set(corr_df["target"].astype(str))), None)
			if target is not None:
				features = ["delta_mean_metric_value", "delta_mean_abs_metric_value", "delta_mean_layer_percentile_rank", "delta_mean_layer_zscore", "delta_top10pct_rate"]
				hdf = corr_df.loc[(corr_df["scope"] == "associated_minus_unrelated") & (corr_df["target"] == target) & (corr_df["feature"].isin(features))].copy()
				if not hdf.empty:
					pivot = hdf.pivot_table(index="metric", columns="feature", values="spearman", aggfunc="first")
					pivot = pivot.reindex([m for m in metric_order if m in pivot.index])
					if not pivot.empty:
						fig, ax = plt.subplots(figsize=(max(9.5, 1.25 * len(pivot.columns)), max(4.8, 0.9 * len(pivot.index))))
						arr = pivot.to_numpy(dtype=float)
						im = ax.imshow(arr, vmin=-1, vmax=1, aspect="auto")
						ax.set_yticks(np.arange(len(pivot.index)))
						ax.set_yticklabels([_metric_display(m) for m in pivot.index])
						ax.set_xticks(np.arange(len(pivot.columns)))
						ax.set_xticklabels([_wrap_label(_pretty_col_label(c), width=16) for c in pivot.columns], rotation=0, ha="center")
						ax.tick_params(axis="x", pad=7)
						ax.set_title(f"Spearman correlation with {_pretty_col_label(target)}")
						for i in range(arr.shape[0]):
							for j in range(arr.shape[1]):
								if np.isfinite(arr[i, j]):
									ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=10)
						fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
						fig.tight_layout()
						png = stats_dir / "agonist_metric_correlation_heatmap.pdf"
						fig.savefig(png, dpi=300, bbox_inches="tight")
						fig.savefig(png.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
						pdf.savefig(fig, bbox_inches="tight")
						paths.append(str(png))
						paths.append(str(png.with_suffix(".pdf")))
						plt.close(fig)
		if delta_df is not None and not delta_df.empty:
			target = "flip_any_rate" if "flip_any_rate" in delta_df.columns and pd.to_numeric(delta_df["flip_any_rate"], errors="coerce").notna().sum() >= 3 else "abs_max_effect"
			feature = "delta_mean_layer_percentile_rank" if "delta_mean_layer_percentile_rank" in delta_df.columns else None
			if feature and target in delta_df.columns:
				fig, ax = plt.subplots(figsize=(8.5, 6.0))
				for metric in metric_order:
					sub = delta_df.loc[delta_df["metric"] == metric]
					x = pd.to_numeric(sub[feature], errors="coerce")
					y = pd.to_numeric(sub[target], errors="coerce")
					mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
					if int(mask.sum()) >= 1:
						ax.scatter(x[mask], y[mask], s=22, alpha=0.65, label=_metric_display(metric))
				ax.axvline(0.0, linewidth=1, alpha=0.4)
				ax.set_xlabel("Associated - unrelated mean layer percentile rank")
				ax.set_ylabel(_pretty_col_label(target))
				ax.set_title(f"Metric contrast vs {_pretty_col_label(target)}")
				ax.grid(True, alpha=0.25)
				ax.legend(fontsize=10, frameon=True)
				fig.tight_layout()
				png = stats_dir / "agonist_metric_ablation_scatter.pdf"
				fig.savefig(png, dpi=300, bbox_inches="tight")
				fig.savefig(png.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
				pdf.savefig(fig, bbox_inches="tight")
				paths.append(str(png))
				paths.append(str(png.with_suffix(".pdf")))
				plt.close(fig)
	paths.append(str(pdf_path))
	return paths


def write_agonist_metric_final_stats(circuit_agonists_path, out_dir, stats_dirname="", flip_stats_df=None, topk=30):
	"""Aggregate script-6 activation/saliency outputs into final tables and plots."""
	stats_dir = Path(out_dir) / "stats" / str(stats_dirname or "")
	stats_dir.mkdir(parents=True, exist_ok=True)
	metric_df = _normalise_agonist_metric_summary(circuit_agonists_path)
	if metric_df.empty:
		print("[AgonistMetrics] No script-6 agonist activation/saliency summaries found; skipping final metric diagnostics.")
		return None
	if flip_stats_df is None:
		flip_stats_df = _load_existing_flip_stats(out_dir, stats_dirname)
	metric_df = _merge_flip_stats(metric_df, flip_stats_df)
	metric_df = metric_df.reset_index(drop=True)
	delta_df = _compute_metric_delta_rows(metric_df)
	global_df = _compute_metric_global_summary(metric_df)
	corr_df = _compute_metric_correlations(metric_df, delta_df)
	summary_path = stats_dir / "agonist_metric_summary.csv"
	delta_path = stats_dir / "agonist_metric_delta_by_unit.csv"
	global_path = stats_dir / "agonist_metric_global_summary.csv"
	corr_path = stats_dir / "agonist_metric_correlations.csv"
	best_path = stats_dir / "agonist_metric_best_correlations.csv"
	json_path = stats_dir / "agonist_metric_stats.json"
	metric_df.to_csv(summary_path, index=False)
	if delta_df is not None and not delta_df.empty:
		delta_df.to_csv(delta_path, index=False)
	if global_df is not None and not global_df.empty:
		global_df.to_csv(global_path, index=False)
	if corr_df is not None and not corr_df.empty:
		corr_df.to_csv(corr_path, index=False)
		best_df = corr_df.sort_values("abs_spearman", ascending=False).groupby(["target", "scope"], as_index=False).head(int(topk))
		best_df.to_csv(best_path, index=False)
	else:
		best_df = pd.DataFrame()
	plot_paths = _save_metric_plots(metric_df, delta_df, corr_df, stats_dir, topk=topk)
	headlines = {}
	if corr_df is not None and not corr_df.empty:
		for target in ["flip_any_rate", "abs_max_effect", "accuracy_gap", "c2i_rate", "i2c_rate"]:
			cand = corr_df.loc[(corr_df["target"] == target) & (corr_df["scope"] == "associated_minus_unrelated")].copy()
			if cand.empty:
				cand = corr_df.loc[corr_df["target"] == target].copy()
			if not cand.empty:
				row = cand.sort_values("abs_spearman", ascending=False).iloc[0].to_dict()
				headlines[target] = {
					"metric": str(row.get("metric")),
					"metric_display": str(row.get("metric_display")),
					"feature": str(row.get("feature")),
					"scope": str(row.get("scope")),
					"n": int(row.get("n")) if pd.notna(row.get("n")) else None,
					"spearman": float(row.get("spearman")) if pd.notna(row.get("spearman")) else None,
					"pearson": float(row.get("pearson")) if pd.notna(row.get("pearson")) else None,
				}
	payload = {
		"n_rows": int(len(metric_df)),
		"n_delta_rows": int(len(delta_df)) if delta_df is not None else 0,
		"metrics": sorted(metric_df["metric"].dropna().astype(str).unique().tolist(), key=lambda m: _metric_sort_key(m)),
		"targets_available": sorted([c for c in ["max_effect", "abs_max_effect", "accuracy_gap", "abs_accuracy_gap", "flip_any_rate", "c2i_rate", "i2c_rate"] if c in metric_df.columns]),
		"headline_best_correlations": headlines,
		"files": {
			"summary_csv": summary_path.name,
			"delta_by_unit_csv": delta_path.name if delta_df is not None and not delta_df.empty else None,
			"global_summary_csv": global_path.name if global_df is not None and not global_df.empty else None,
			"correlations_csv": corr_path.name if corr_df is not None and not corr_df.empty else None,
			"best_correlations_csv": best_path.name if corr_df is not None and not corr_df.empty else None,
			"plots": [Path(p).name for p in plot_paths],
		},
		"notes": {
			"activation": "Activation is read from *_agonist_activation_summary.csv and uses script-6 activation rank statistics.",
			"wanda": "For mL units from blocks.L.hook_mlp_out, Wanda is a hooked-site proxy and may reduce to activation magnitude; attention hook_z Wanda is more faithful.",
			"correlation": "Spearman is the primary ranking statistic because ablation effects are often non-linear and threshold-like.",
		},
	}
	json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
	print(f"[AgonistMetrics] Wrote {summary_path}")
	if corr_df is not None and not corr_df.empty:
		print(f"[AgonistMetrics] Wrote {corr_path}")
	if plot_paths:
		print(f"[AgonistMetrics] Wrote final plots under {stats_dir}")
	if headlines:
		for target, row in headlines.items():
			sp = row.get("spearman")
			sp_txt = f"{sp:.3f}" if sp is not None else "nan"
			print(f"[AgonistMetrics] Best for {target}: {row['metric_display']} / {row['feature']} ({row['scope']}), Spearman={sp_txt} n={row['n']}")
	return payload

# ------------------------- Rule quality summaries (paper-ready) -------------------------


_RULE_ATOM_RE = re.compile(r"([A-Za-z0-9_\.]+)\s*(<=|>=|==|!=|<|>)\s*([-+0-9.eE]+)")

def _parse_rule_lengths(expr: str):
	'''
	Parse a RuleSHAP expression and return:
	- n_atoms: total number of atomic comparisons
	- n_disjuncts: number of OR clauses (at least 1)
	- max_atoms_in_disjunct: max #atoms within any disjunct (rough proxy for conjunctive length)
	'''
	if expr is None:
		return (0, 0, 0)
	s = str(expr).strip()
	if not s or s.upper() in ("FALSE", "TRUE", "NONE", "NAN"):
		return (0, 0, 0)

	atom_hits = _RULE_ATOM_RE.findall(s)
	n_atoms = int(len(atom_hits))

	disjuncts = re.split(r"\s+OR\s+|\s*\|\|\s*|\s*\|\s*", s)
	disjuncts = [d.strip() for d in disjuncts if d.strip()]
	n_disjuncts = int(len(disjuncts)) if disjuncts else 1

	max_atoms = 0
	for d in disjuncts:
		max_atoms = max(max_atoms, len(_RULE_ATOM_RE.findall(d)))
	return (n_atoms, n_disjuncts, int(max_atoms))

def collect_rule_combo_metrics(rules_dir: str):
	'''
	Walk rules_dir and collect all rule_combo_*.csv files into a single dataframe.

	Expected layout (from this script):
	  <rules_dir>/<rule_target>/<layerKey>_<neuronId>/rule_combo_<flip_target>.csv
	'''
	rules_dir = str(rules_dir)
	root = Path(rules_dir)
	rows = []
	for fp in root.glob("**/rule_combo_*.csv"):
		if not fp.is_file():
			continue
		rel = fp.relative_to(root)
		parts = rel.parts
		if len(parts) >= 3:
			rule_target_dir = parts[0]
			neuron_dir = parts[-2]
		else:
			rule_target_dir = parts[0] if parts else ""
			neuron_dir = parts[-2] if len(parts) >= 2 else ""

		m = re.match(r"^(.*)_(\d+)$", str(neuron_dir))
		layer_key = m.group(1) if m else str(neuron_dir)
		neuron_id = int(m.group(2)) if m else None

		metric = fp.name[len("rule_combo_"):-len(".csv")] if fp.name.startswith("rule_combo_") else fp.name
		df = pd.read_csv(fp)
		if df.empty:
			continue
		r = df.iloc[0].to_dict()

		computed_on = "eval" if (fp.parent / "global_shap_stats_train.pkl").exists() else "train_or_all"

		expr = r.get("expression", r.get("rule_expression", ""))
		n_atoms, n_disj, max_atoms = _parse_rule_lengths(expr)

		def _get_num(keys, default=np.nan):
			for k in keys:
				if k in r and r[k] == r[k]:
					return float(r[k])
			return float(default)

		rows.append({
			"rule_target_dir": str(rule_target_dir),
			"layer_key": str(layer_key),
			"neuron_id": neuron_id,
			"neuron_key": f"{layer_key}_{neuron_id}" if neuron_id is not None else str(neuron_dir),
			"flip_target": str(metric),
			"computed_on": computed_on,
			"F1": _get_num(["F1", "F1(target=1|fire)", "f1"]),
			"MCC": _get_num(["MCC", "mcc"]),
			"TPR": _get_num(["TPR", "tpr"]),
			"TNR": _get_num(["TNR", "tnr"]),
			"BalancedAcc": _get_num(["BalancedAcc", "balancedacc", "balanced_acc"]),
			"Acc": _get_num(["Acc", "acc", "accuracy"]),
			"dataset_coverage": _get_num(["dataset_coverage", "coverage"]),
			"expression": str(expr),
			"rule_len_atoms": int(n_atoms),
			"rule_len_disjuncts": int(n_disj),
			"rule_len_max_atoms_in_disjunct": int(max_atoms),
			"rule_combo_path": str(fp),
		})

	if not rows:
		return None
	return pd.DataFrame(rows)

def _neurons_sorted_to_meta(neurons_sorted):
	"""
	Returns:
	  allowed_keys: set of neuron_key strings like 'a16_h21_458'
	  rank: dict neuron_key -> order index (0..)
	  baseline_map: dict neuron_key -> baseline_subset ('positive'/'negative'/etc.)
	Uses the same key format as collect_rule_combo_metrics(): f"{layer_key}_{neuron_id}"
	"""
	keys = []
	baseline_map = {}
	for layer_label, neuron_id, baseline_subset in unique_everseen(neurons_sorted, key=lambda x: (x[0], x[1])):
		layer_key = _safe_layer_label(layer_label)
		nk = f"{layer_key}_{int(neuron_id)}"
		keys.append(nk)
		baseline_map[nk] = "all"
	allowed = set(keys)
	rank = {k: i for i, k in enumerate(keys)}
	return allowed, rank, baseline_map

def _baseline_subset_allows_flip_target(baseline_subset: str, flip_target: str, layer_key: str, neuron_id: int) -> bool:
	"""Flipping and rule extraction always operate on baseline 'all'."""
	ft = str(flip_target or "")
	agg = f"flip_{layer_key}_{int(neuron_id)}"
	if ft == agg:
		return True
	return ft.startswith("flip_c2i_") or ft.startswith("flip_i2c_")

def _filter_rule_metrics_df(df: pd.DataFrame, neurons_sorted):
	"""
	Filter a rule-metrics dataframe (from collect_rule_combo_metrics or derived)
	to only include:
	  - neurons in neurons_sorted
	  - rule_combo targets that actually correspond to that neuron (suffix match)
	Adds:
	  - neuron_rank
	  - baseline_subset (mapped from neurons_sorted)
	"""
	if df is None or df.empty or neurons_sorted is None:
		return df

	allowed_keys, key_rank, baseline_map = _neurons_sorted_to_meta(neurons_sorted)

	# Ensure neuron_key exists
	out = df.copy()
	if "neuron_key" not in out.columns:
		# fall back to layer_key + neuron_id if present
		if "layer_key" in out.columns and "neuron_id" in out.columns:
			out["neuron_key"] = out["layer_key"].astype(str) + "_" + out["neuron_id"].astype(str)
		else:
			# can't align; return conservative empty
			return out.iloc[0:0].copy()

	# Keep only neurons from this run
	out = out[out["neuron_key"].isin(allowed_keys)].copy()
	if out.empty:
		return out

	out["neuron_rank"] = out["neuron_key"].map(key_rank).astype("Int64")
	out["baseline_subset"] = out["neuron_key"].map(baseline_map).fillna("")

	# Require flip_target to actually belong to this neuron (suffix match, elementwise)
	if "flip_target" in out.columns and "layer_key" in out.columns and "neuron_id" in out.columns:
		out = out.dropna(subset=["layer_key", "neuron_id", "flip_target"]).copy()

		# Build per-row suffix like "_a16_h21_458"
		suffix = (
			"_" + out["layer_key"].astype(str) + "_" + out["neuron_id"].astype("int64").astype(str)
		).to_numpy()

		ft = out["flip_target"].astype(str).to_numpy()

		mask = np.fromiter((a.endswith(b) for a, b in zip(ft, suffix)), dtype=bool, count=len(out))
		out = out[mask].copy()

	return out

def _norm_rule_metric_name(name: str) -> str:
	"""Normalize user-facing metric names to internal keys used in _RULE_METRIC_SPECS."""
	if name is None:
		return "mcc"
	s = str(name).strip().lower()
	# common aliases
	if s in ("balancedacc", "balanced_acc", "balanced accuracy", "bac", "ba"):
		return "balanced_accuracy"
	if s in ("mcc", "matthews", "matthews_corrcoef"):
		return "mcc"
	if s in ("tpr", "recall", "sensitivity"):
		return "tpr"
	if s in ("tnr", "specificity"):
		return "tnr"
	if s in ("f1", 'f1(target=1|fire)'):
		return "f1"
	# already canonical?
	return s if s in _RULE_METRIC_SPECS else "mcc"

def _rule_metric_col(metric: str) -> str:
	return _RULE_METRIC_SPECS[_norm_rule_metric_name(metric)][0]

def _rule_metric_label(metric: str) -> str:
	return _RULE_METRIC_SPECS[_norm_rule_metric_name(metric)][1]

def write_rule_metrics_stats(
	rules_dir: str,
	out_dir: str,
	stats_dirname: str = "",
	quantiles=(0.1, 0.5, 0.9),
	topk: int = 50,
	length_cap: int = 30,
	quality_metric: str = "mcc",
	quality_threshold: float = None,
	neurons_sorted=None,  # <-- NEW
):
	'''
	Writes paper-ready rule statistics/figures from RuleSHAP artifacts.
	'''
	out_dir = str(out_dir)
	Path(out_dir).mkdir(parents=True, exist_ok=True)

	df_all = collect_rule_combo_metrics(rules_dir)
	if df_all is None or df_all.empty:
		print("[RuleMetrics] No rule_combo_*.csv found; skipping.")
		return None

	# ---- Restrict rule metrics to the neuron universe used by this run + enforce baseline_subset ----
	if neurons_sorted is not None:
		df_all = _filter_rule_metrics_df(df_all, neurons_sorted)
	else:
		df_all = df_all.copy()
		df_all["neuron_rank"] = pd.NA
		df_all["baseline_subset"] = ""

	metric_key = _norm_rule_metric_name(quality_metric)
	metric_col = _rule_metric_col(metric_key)
	metric_label = _rule_metric_label(metric_key)
	thr = float(quality_threshold)
	# If the requested metric isn't present in the artifacts, fall back to MCC.
	if metric_col not in df_all.columns:
		metric_key = "mcc"
		metric_col = "MCC"
		metric_label = "MCC"

	# Sort by neuron order first (if provided), then by quality within neuron.
	sort_cols = (["neuron_rank"] if "neuron_rank" in df_all.columns else []) + [metric_col, "MCC"]
	df_best = (
		df_all.sort_values(sort_cols, ascending=[True] + [False, False], na_position="last")
		.drop_duplicates(subset=["rule_target_dir", "neuron_key"], keep="first")
		.reset_index(drop=True)
	)

	csv_all = os.path.join(out_dir, 'stats', stats_dirname, f"rule_combo_metrics_all.csv")
	# Keep neuron order stable in the "all" dump; metric sort as secondary.
	if "neuron_rank" in df_all.columns:
		df_all_out = df_all.sort_values(["neuron_rank", metric_col, "MCC"], ascending=[True, False, False], na_position="last")
	else:
		df_all_out = df_all.sort_values([metric_col, "MCC"], ascending=False, na_position="last")
	df_all_out.to_csv(csv_all, index=False)

	print(f"[RuleMetrics] Wrote {csv_all}")

	csv_best = os.path.join(out_dir, 'stats', stats_dirname, f"rule_combo_metrics_best_per_neuron.csv")
	df_best.to_csv(csv_best, index=False)
	print(f"[RuleMetrics] Wrote {csv_best}")

	if topk is not None and int(topk) > 0:
		df_top = df_best.sort_values([metric_col, "MCC"], ascending=False, na_position="last").head(int(topk)).copy()
	else:
		df_top = df_best.copy()
	csv_top = os.path.join(out_dir, 'stats', stats_dirname, f"rule_combo_metrics_top_{int(topk) if topk else 'all'}.csv")
	df_top.to_csv(csv_top, index=False)
	print(f"[RuleMetrics] Wrote {csv_top}")

	q_lo, q_med, q_hi = quantiles

	def _q(series):
		v = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
		if len(v) == 0:
			return {"n": 0}
		return {
			"n": int(len(v)),
			"mean": float(np.mean(v)),
			"median": float(np.median(v)),
			"q_lo": float(np.quantile(v, q_lo)),
			"q_med": float(np.quantile(v, q_med)),
			"q_hi": float(np.quantile(v, q_hi)),
			"min": float(np.min(v)),
			"max": float(np.max(v)),
		}

	high = df_best[pd.to_numeric(df_best[metric_col], errors="coerce") >= float(thr)]

	summary = {
		"n_rule_combo_files": int(len(df_all)),
		"n_neurons_with_rules": int(len(df_best)),
		"quality_metric": str(metric_key),
		"quality_metric_label": str(metric_label),
		"quality_threshold": float(thr),
		"n_neurons_quality_ge_threshold": int(len(high)),
		"frac_neurons_quality_ge_threshold": (float(len(high)) / float(len(df_best))) if len(df_best) else float("nan"),
		"quantiles": [float(q_lo), float(q_med), float(q_hi)],
		metric_col: _q(df_best[metric_col]) if metric_col in df_best.columns else {"n": 0},
		"MCC": _q(df_best["MCC"]),
		"F1": _q(df_best["F1"]),
		"TPR": _q(df_best["TPR"]),
		"TNR": _q(df_best["TNR"]),
		"BalancedAcc": _q(df_best["BalancedAcc"]) if "BalancedAcc" in df_best.columns else {"n": 0},
		"rule_len_atoms": _q(df_best["rule_len_atoms"]),
		"dataset_coverage": _q(df_best["dataset_coverage"]),
	}

	json_path = os.path.join(out_dir, 'stats', stats_dirname, f"rule_metrics_summary.json")
	Path(json_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
	print(f"[RuleMetrics] Wrote {json_path}")

	def _pct(x):
		return int(round(100.0 * float(x))) if x == x else None
	def _fmt_float(x, nd=2):
		if x != x:
			return None
		return f"{float(x):.{nd}f}"

	tex_suffix = re.sub(r"[^0-9A-Za-z]+", "", str(stats_dirname))
	tex_suffix = tex_suffix if tex_suffix else "Run"

	tex_lines = []
	tex_lines.append(f"% Auto-generated by 7_refine_neuron_anchored_rules (rule metrics)")
	tex_lines.append(f"\\newcommand\\RuleQualMetric{tex_suffix}{{{metric_label}}}")
	tex_lines.append(f"\\newcommand\\RuleQualThr{tex_suffix}{{{_fmt_float(thr,2)}}}")
	tex_lines.append(f"\\newcommand\\RuleNeurons{tex_suffix}{{{int(summary['n_neurons_with_rules'])}}}")
	tex_lines.append(f"\\newcommand\\RuleNeuronsHighQual{tex_suffix}{{{int(summary['n_neurons_quality_ge_threshold'])}}}")
	tex_lines.append(f"\\newcommand\\RuleNeuronsHighQualPct{tex_suffix}{{{_pct(summary['frac_neurons_quality_ge_threshold'])}}}")
	for key, macro in [("MCC","RuleMCC"), ("TPR","RuleTPR"), ("TNR","RuleTNR"), ("F1","RuleF1"), ("rule_len_atoms","RuleLen")]:
		st = summary.get(key, {})
		tex_lines.append(f"\\newcommand\\{macro}Med{tex_suffix}{{{_fmt_float(st.get('median', float('nan')),3)}}}")
		tex_lines.append(f"\\newcommand\\{macro}Qlo{tex_suffix}{{{_fmt_float(st.get('q_lo', float('nan')),3)}}}")
		tex_lines.append(f"\\newcommand\\{macro}Qhi{tex_suffix}{{{_fmt_float(st.get('q_hi', float('nan')),3)}}}")
	tex_path = os.path.join(out_dir, 'stats', stats_dirname, f"rule_metrics.tex")
	Path(tex_path).write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
	print(f"[RuleMetrics] Wrote {tex_path}")

	plt.figure(figsize=(12, 7))

	def _hist_col(col: str, xlabel: str, title: str, pos: int):
		plt.subplot(2, 3, pos)
		if col in df_best.columns:
			v = pd.to_numeric(df_best[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
			plt.hist(v, bins=30)
		else:
			plt.hist([], bins=10)
		plt.xlabel(xlabel)
		plt.ylabel("# neurons")
		plt.title(title)

	_hist_col("MCC", "MCC", "Rule quality (MCC)", 1)
	_hist_col("F1", "F1", "Rule quality (F1)", 2)
	_hist_col("BalancedAcc", "Balanced accuracy", "Rule quality (Balanced accuracy)", 3)
	_hist_col("TPR", "Sensitivity (TPR)", "Rule quality (Sensitivity)", 4)
	_hist_col("TNR", "Specificity (TNR)", "Rule quality (Specificity)", 5)

	plt.subplot(2, 3, 6)
	lens = pd.to_numeric(df_best["rule_len_atoms"], errors="coerce").dropna().to_numpy()
	if len(lens):
		max_x = max(1, int(min(np.max(lens), int(length_cap))))
		plt.hist(np.clip(lens, 0, max_x), bins=min(30, max_x))
		plt.xlim(0, max_x)
	else:
		plt.hist([], bins=10)
	plt.xlabel("# atomic conditions (clipped)")
	plt.ylabel("# neurons")
	plt.title("Rule length")

	plt.tight_layout()
	plot_path = os.path.join(out_dir, 'stats', stats_dirname, f"rule_metrics_distributions.pdf")
	plt.savefig(plot_path, dpi=250, bbox_inches='tight')
	plt.close()
	print(f"[RuleMetrics] Wrote {plot_path}")
	
	return df_best

def _union_and_sum_flips(scores_df: pd.DataFrame, cols):
	n = len(scores_df)
	union = np.zeros(n, dtype=bool)
	any_eval = np.zeros(n, dtype=bool)
	sum_count = 0
	for c in cols:
		if c not in scores_df.columns:
			continue
		s = scores_df[c]
		any_eval |= s.notna().to_numpy()
		v = s.fillna(False).astype(bool).to_numpy()
		union |= v
		sum_count += int(v.sum())
	return int(union.sum()), int(sum_count), int(any_eval.sum())

def write_high_quality_neuron_flip_coverage(
	scores_df: pd.DataFrame,
	neurons_sorted,
	rule_best_df: pd.DataFrame,
	out_dir: str,
	stats_dirname: str = "",
	# Backward-compat default; if quality_threshold is None we fall back to this.
	quality_metric: str = "mcc",
	quality_threshold: float = None,
):
	'''
	Among neurons whose best rule has MCC >= threshold, compute flips accounted:
	- Union (all together): unique flipped datapoints across the set
	- Sum (in isolation): sum of per-neuron flipped datapoints (double-count overlaps)

	Writes: json + tex + a single compact figure.
	'''
	if rule_best_df is None or rule_best_df.empty:
		print("[HighQuality] No rule metrics available; skipping.")
		return None
	# Ensure rule_best_df only contains rules for these neurons + compatible targets
	rule_best_df = _filter_rule_metrics_df(rule_best_df, neurons_sorted)
	if rule_best_df is None or rule_best_df.empty:
		print("[HighQuality] No rule metrics available after filtering; skipping.")
		return None

	out_dir = str(out_dir)
	Path(out_dir).mkdir(parents=True, exist_ok=True)

	total_neurons = []
	for layer_label, neuron_id, _ in unique_everseen(neurons_sorted, key=lambda x: (x[0],x[1])):
		layer_key = _safe_layer_label(layer_label)
		total_neurons.append((layer_key, int(neuron_id)))

	metric_key = _norm_rule_metric_name(quality_metric)
	metric_col = _rule_metric_col(metric_key)
	metric_label = _rule_metric_label(metric_key)
	thr = float(quality_threshold)
	# If the requested metric isn't present in the artifacts, fall back to MCC.
	if metric_col not in rule_best_df.columns:
		metric_key = "mcc"
		metric_col = "MCC"
		metric_label = "MCC"

	tmp = rule_best_df.copy()
	tmp["layer_key"] = tmp["layer_key"].map(_safe_layer_label)  # <- critical
	high_df = tmp[pd.to_numeric(tmp[metric_col], errors="coerce") >= float(thr)]
	high_ids = {(str(r["layer_key"]), int(r["neuron_id"])) for _, r in high_df.dropna(subset=["neuron_id", "layer_key"]).iterrows()}
	high_neurons = [t for t in total_neurons if (t[0], t[1]) in high_ids]

	# print(1, high_neurons)

	def _cols(neuron_list, prefix):
		return [f"{prefix}{lk}_{nid}" for (lk, nid) in neuron_list]

	cols_any_all = _cols(total_neurons, "flip_")
	cols_c2i_all = _cols(total_neurons, "flip_c2i_")
	cols_i2c_all = _cols(total_neurons, "flip_i2c_")
	cols_any_high = _cols(high_neurons, "flip_")
	cols_c2i_high = _cols(high_neurons, "flip_c2i_")
	cols_i2c_high = _cols(high_neurons, "flip_i2c_")

	union_any_all, sum_any_all, n_eval_any_all = _union_and_sum_flips(scores_df, cols_any_all)
	union_any_high, sum_any_high, n_eval_any_high = _union_and_sum_flips(scores_df, cols_any_high)
	union_c2i_all, sum_c2i_all, _ = _union_and_sum_flips(scores_df, cols_c2i_all)
	union_c2i_high, sum_c2i_high, _ = _union_and_sum_flips(scores_df, cols_c2i_high)
	union_i2c_all, sum_i2c_all, _ = _union_and_sum_flips(scores_df, cols_i2c_all)
	union_i2c_high, sum_i2c_high, _ = _union_and_sum_flips(scores_df, cols_i2c_high)

	# --- Compact p-values: permutation test vs random subset of same size (union coverage) ---
	def _fmt_p(p: float) -> str:
		if p != p:  # NaN
			return ""
		if p < 1e-4:
			return "<1e-4"
		if p < 0.01:
			return f"{p:.2g}"  # compact scientific-ish
		return f"{p:.2f}"

	def _bool_mat(cols):
		# Missing columns -> NaN -> 0; supports 0/1, bool, or numeric-as-str.
		sub = scores_df.reindex(columns=cols)
		sub = sub.apply(pd.to_numeric, errors="coerce").fillna(0.0)
		return (sub.to_numpy() != 0.0)

	def _perm_p_union(mat_bool, idx_high, n_perm = 1000, seed = None) -> float:
		n_total = mat_bool.shape[1]
		k = len(idx_high)
		if k <= 0 or k >= n_total:
			return float("nan")
		rng = np.random.default_rng(seed) if seed is not None else np.random
		obs = int(mat_bool[:, idx_high].any(axis=1).sum())
		ge = 0
		for _ in range(int(n_perm)):
			ix = rng.choice(n_total, size=k, replace=False)
			val = int(mat_bool[:, ix].any(axis=1).sum())
			if val >= obs:
				ge += 1
		return (ge + 1.0) / (n_perm + 1.0)

	# indices of high_neurons in total_neurons (preserves order)
	_idx_map = {t: i for i, t in enumerate(total_neurons)}
	_idx_high = [_idx_map[t] for t in high_neurons] if high_neurons else []
	_NPERM = 1000
	_p_any = _perm_p_union(_bool_mat(cols_any_all), _idx_high, n_perm=_NPERM, seed=0)
	_p_c2i = _perm_p_union(_bool_mat(cols_c2i_all), _idx_high, n_perm=_NPERM, seed=1)
	_p_i2c = _perm_p_union(_bool_mat(cols_i2c_all), _idx_high, n_perm=_NPERM, seed=2)

	summary = {
		"quality_metric": str(metric_key),
		"quality_metric_label": str(metric_label),
		"quality_threshold": float(thr),
		# Keep legacy field for backward compatibility.
		"n_neurons_total": int(len(total_neurons)),
		"n_neurons_high_quality": int(len(high_neurons)),
		"frac_neurons_high_quality": (float(len(high_neurons))/float(len(total_neurons))) if len(total_neurons) else float("nan"),
		"flip_any": {
			"union_high": int(union_any_high),
			"sum_high": int(sum_any_high),
			"union_all": int(union_any_all),
			"sum_all": int(sum_any_all),
			"union_share_of_all": (float(union_any_high)/float(union_any_all)) if union_any_all else float("nan"),
			"n_rows_with_any_eval_all": int(n_eval_any_all),
			"n_rows_with_any_eval_high": int(n_eval_any_high),
		},
		"flip_c2i": {
			"union_high": int(union_c2i_high),
			"sum_high": int(sum_c2i_high),
			"union_all": int(union_c2i_all),
			"sum_all": int(sum_c2i_all),
			"union_share_of_all": (float(union_c2i_high)/float(union_c2i_all)) if union_c2i_all else float("nan"),
		},
		"flip_i2c": {
			"union_high": int(union_i2c_high),
			"sum_high": int(sum_i2c_high),
			"union_all": int(union_i2c_all),
			"sum_all": int(sum_i2c_all),
			"union_share_of_all": (float(union_i2c_high)/float(union_i2c_all)) if union_i2c_all else float("nan"),
		},
		"p_values": {
			# One-sided permutation p-value: P[ random subset union >= observed high-quality union ].
			"n_perm": int(_NPERM),
			"perm_union_any": float(_p_any),
			"perm_union_c2i": float(_p_c2i),
			"perm_union_i2c": float(_p_i2c),
		},
		"figure_semantics": {
			"prevalence_denominator": "Rows with at least one evaluated all-neuron flip column, not raw len(scores_df).",
			"unique_union": "A prompt flipped by several neurons is counted once.",
			"sum_over_neurons": "Per-neuron flip counts are summed, so overlaps are counted multiple times.",
			"coverage_share": "High-quality union divided by all-neuron union for the same flip direction.",
		},
	}

	json_path = os.path.join(out_dir, 'stats', stats_dirname, f"high_quality_neuron_flip_coverage.json")
	Path(json_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
	print(f"[HighQuality] Wrote {json_path}")

	def _pct(x):
		return int(round(100.0 * float(x))) if x == x else None
	def _fmtf(x, nd=2):
		return f"{float(x):.{nd}f}" if x == x else ""

	tex_suffix = re.sub(r"[^0-9A-Za-z]+", "", str(stats_dirname))
	tex_suffix = tex_suffix if tex_suffix else "Run"
	tex_lines = []
	tex_lines.append(f"% Auto-generated by 7_refine_neuron_anchored_rules (high-quality flip coverage)")
	tex_lines.append(f"\\newcommand\\HighQualMetric{tex_suffix}{{{metric_label}}}")
	tex_lines.append(f"\\newcommand\\HighQualThr{tex_suffix}{{{_fmtf(thr,2)}}}")
	tex_lines.append(f"\\newcommand\\HighQualNeurons{tex_suffix}{{{int(summary['n_neurons_high_quality'])}}}")
	tex_lines.append(f"\\newcommand\\HighQualNeuronsPct{tex_suffix}{{{_pct(summary['frac_neurons_high_quality'])}}}")
	tex_lines.append(f"\\newcommand\\HighQualityUnionFlipAny{tex_suffix}{{{summary['flip_any']['union_high']}}}")
	tex_lines.append(f"\\newcommand\\HighQualityUnionFlipAnyPctAll{tex_suffix}{{{_pct(summary['flip_any']['union_share_of_all'])}}}")
	tex_lines.append(f"\\newcommand\\HighQualitySumFlipAny{tex_suffix}{{{summary['flip_any']['sum_high']}}}")
	tex_lines.append(f"\\newcommand\\HighQualityPUnionAny{tex_suffix}{{{_fmt_p(summary['p_values']['perm_union_any'])}}}")
	tex_lines.append(f"\\newcommand\\HighQualityPUnionCII{tex_suffix}{{{_fmt_p(summary['p_values']['perm_union_c2i'])}}}")
	tex_lines.append(f"\\newcommand\\HighQualityPUnionICC{tex_suffix}{{{_fmt_p(summary['p_values']['perm_union_i2c'])}}}")

	tex_path = os.path.join(out_dir, 'stats', stats_dirname, f"high_quality_neuron_flip_coverage.tex")
	Path(tex_path).write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
	print(f"[HighQuality] Wrote {tex_path}")

	# Main figure: compact, paper-friendly, and collision-safe.
	# Labels are placed with deterministic offsets/inside-bar placement so that equal-height bars
	# (the common case in this plot) do not overlap.
	eval_denom = float(n_eval_any_all) if n_eval_any_all else float(len(scores_df))
	_label_box = dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.82)

	def _bar_center(bar):
		return bar.get_x() + bar.get_width() / 2.0

	def _annotate_bar_top(ax, bar, label, *, fontsize=7.0, xytext=(0, 2), ha="center"):
		h = float(bar.get_height())
		if h != h:
			return
		ax.annotate(
			label,
			(_bar_center(bar), h),
			ha=ha,
			va="bottom",
			fontsize=fontsize,
			xytext=xytext,
			textcoords="offset points",
			bbox=_label_box,
			clip_on=False,
		)

	def _annotate_bar_inside(ax, bar, label, *, fontsize=6.8, frac=0.965):
		h = float(bar.get_height())
		if h != h:
			return
		if h <= 0:
			_annotate_bar_top(ax, bar, label, fontsize=fontsize)
			return
		y_text = max(0.0, h * frac)
		ax.annotate(
			label,
			(_bar_center(bar), y_text),
			ha="center",
			va="top",
			fontsize=fontsize,
			xytext=(0, -1),
			textcoords="offset points",
			bbox=_label_box,
			clip_on=True,
		)

	def _annotate_grouped_pair(ax, left_bar, right_bar, left_label, right_label, *, fontsize=6.6):
		# Outward alignment prevents the two labels over a same-height grouped pair from colliding.
		for bar, label, ha, dx in (
			(left_bar, left_label, "right", -2),
			(right_bar, right_label, "left", 2),
		):
			h = float(bar.get_height())
			if h != h:
				continue
			ax.annotate(
				label,
				(_bar_center(bar), h),
				ha=ha,
				va="bottom",
				fontsize=fontsize,
				xytext=(dx, 2),
				textcoords="offset points",
				bbox=_label_box,
				clip_on=False,
			)

	with plt.rc_context({
		"font.size": 8,
		"axes.titlesize": 9,
		"axes.labelsize": 8,
		"xtick.labelsize": 7.4,
		"ytick.labelsize": 7.4,
		"legend.fontsize": 7.0,
	}):
		fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.65), constrained_layout=False)
		fig.subplots_adjust(left=0.08, right=0.995, bottom=0.115, top=0.955, wspace=0.24, hspace=0.34)
		ax_counts, ax_prev = axes[0]
		ax_cov, ax_overlap = axes[1]

		pop_labels = ["All", f"High-quality\n({metric_label} ≥ {thr:.2f})"]
		pop_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [None, None])[:2]
		if len(pop_colors) < 2:
			pop_colors = [None, None]

		dir_labels = ["Any flip", "Correct→incorrect", "Incorrect→correct"]
		dir_tick_labels = [_wrap_label(s, 12) for s in dir_labels]
		x = np.arange(len(dir_labels))
		w = 0.34

		# A. Subset size.
		counts = np.asarray([len(total_neurons), len(high_neurons)], dtype=float)
		bars = ax_counts.bar(np.arange(2), counts, width=0.62, color=pop_colors)
		ax_counts.set_xticks(np.arange(2))
		ax_counts.set_xticklabels(pop_labels)
		ax_counts.set_ylabel("# neurons")
		ax_counts.set_title("Subset size", pad=4)
		ax_counts.set_ylim(0, max(1.0, float(np.nanmax(counts)) * 1.20 if len(counts) else 1.0))
		ax_counts.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
		for i, b in enumerate(bars):
			h = b.get_height()
			pct = (100.0 * h / float(len(total_neurons))) if len(total_neurons) else float("nan")
			lab = f"{int(h)}"
			if i == 1 and pct == pct:
				lab += f"\n({pct:.1f}%)"
			_annotate_bar_top(ax_counts, b, lab, fontsize=7.2)

		# B. Prompt-level prevalence of unique flips.
		all_union = np.asarray([union_any_all, union_c2i_all, union_i2c_all], dtype=float)
		hq_union = np.asarray([union_any_high, union_c2i_high, union_i2c_high], dtype=float)
		all_prev = 100.0 * all_union / eval_denom if eval_denom else np.asarray([np.nan] * 3)
		hq_prev = 100.0 * hq_union / eval_denom if eval_denom else np.asarray([np.nan] * 3)
		bars_all = ax_prev.bar(x - w/2, all_prev, width=w, label="All", color=pop_colors[0])
		bars_hq = ax_prev.bar(x + w/2, hq_prev, width=w, label="HQ", color=pop_colors[1])
		ax_prev.set_xticks(x)
		ax_prev.set_xticklabels(dir_tick_labels)
		ax_prev.set_ylabel("Unique flips (%)")
		ax_prev.set_title("Flip prevalence", pad=4)
		prev_top = np.nanmax(np.concatenate([all_prev, hq_prev])) if np.isfinite(np.concatenate([all_prev, hq_prev])).any() else 1.0
		ax_prev.set_ylim(0, max(1.0, float(prev_top) * 1.24))
		ax_prev.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
		ax_prev.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.0, columnspacing=0.7, borderpad=0.1)
		for b_l, b_r, cnt_l, cnt_r in zip(bars_all, bars_hq, all_union, hq_union):
			_annotate_grouped_pair(
				ax_prev,
				b_l,
				b_r,
				f"{b_l.get_height():.1f}%\n{int(cnt_l)}",
				f"{b_r.get_height():.1f}%\n{int(cnt_r)}",
				fontsize=6.4,
			)

		# C. Direct coverage question: how much of all discovered flip behavior is covered by HQ neurons?
		coverage = np.asarray([
			(float(union_any_high) / float(union_any_all)) if union_any_all else np.nan,
			(float(union_c2i_high) / float(union_c2i_all)) if union_c2i_all else np.nan,
			(float(union_i2c_high) / float(union_i2c_all)) if union_i2c_all else np.nan,
		], dtype=float)
		coverage_pct = 100.0 * coverage
		bars_cov = ax_cov.bar(x, coverage_pct, width=0.56, color=pop_colors[1])
		ax_cov.set_xticks(x)
		ax_cov.set_xticklabels(dir_tick_labels)
		ax_cov.set_ylim(0, max(100.0, np.nanmax(coverage_pct) * 1.08 if np.isfinite(coverage_pct).any() else 100.0))
		ax_cov.set_ylabel("HQ / all union (%)")
		ax_cov.set_title("Coverage share", pad=4)
		ax_cov.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
		pvals = [_p_any, _p_c2i, _p_i2c]
		pairs = [(union_any_high, union_any_all), (union_c2i_high, union_c2i_all), (union_i2c_high, union_i2c_all)]
		for b, pct, (num, den), p in zip(bars_cov, coverage_pct, pairs, pvals):
			if pct == pct:
				plab = _fmt_p(p)
				lab = f"{pct:.1f}%\n{int(num)}/{int(den)}"
				if plab:
					lab += f"\np={plab}"
				# For near-100% bars, place labels inside the bar so they cannot hit the panel title.
				if pct >= 70:
					_annotate_bar_inside(ax_cov, b, lab, fontsize=6.5, frac=0.975)
				else:
					_annotate_bar_top(ax_cov, b, lab, fontsize=6.5)

		# D. Overlap / redundancy: sum-over-neurons vs unique union.
		def _ratio(sum_v, union_v):
			return (float(sum_v) / float(union_v)) if union_v else np.nan

		overlap_all = np.asarray([
			_ratio(sum_any_all, union_any_all),
			_ratio(sum_c2i_all, union_c2i_all),
			_ratio(sum_i2c_all, union_i2c_all),
		], dtype=float)
		overlap_hq = np.asarray([
			_ratio(sum_any_high, union_any_high),
			_ratio(sum_c2i_high, union_c2i_high),
			_ratio(sum_i2c_high, union_i2c_high),
		], dtype=float)
		bars_oa = ax_overlap.bar(x - w/2, overlap_all, width=w, label="All", color=pop_colors[0])
		bars_oh = ax_overlap.bar(x + w/2, overlap_hq, width=w, label="HQ", color=pop_colors[1])
		ax_overlap.axhline(1.0, linewidth=0.8, alpha=0.45)
		ax_overlap.set_xticks(x)
		ax_overlap.set_xticklabels(dir_tick_labels)
		ax_overlap.set_ylabel("Sum / union")
		ax_overlap.set_title("Overlap", pad=4)
		overlap_top = np.nanmax(np.concatenate([overlap_all, overlap_hq])) if np.isfinite(np.concatenate([overlap_all, overlap_hq])).any() else 1.0
		ax_overlap.set_ylim(0, max(1.0, float(overlap_top) * 1.22))
		ax_overlap.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
		ax_overlap.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.0, columnspacing=0.7, borderpad=0.1)

		# Ratio-only labels in this dense panel prevent the count labels from colliding.
		# Exact numerator/denominator values remain in high_quality_neuron_flip_coverage.json.
		for b_l, b_r in zip(bars_oa, bars_oh):
			_annotate_grouped_pair(
				ax_overlap,
				b_l,
				b_r,
				f"{b_l.get_height():.2f}×",
				f"{b_r.get_height():.2f}×",
				fontsize=6.5,
			)

	plot_path = os.path.join(out_dir, 'stats', stats_dirname, f"high_quality_neuron_flip_coverage.pdf")
	plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
	plt.close()
	print(f"[HighQuality] Wrote {plot_path}")
	
	return summary

def _p_to_star(p: float) -> str:
	if p is None or (isinstance(p, float) and np.isnan(p)):
		return ""
	if p < 1e-3:
		return "***"
	if p < 1e-2:
		return "**"
	if p < 5e-2:
		return "*"
	return ""

def _perm_pval_union(scores_df: pd.DataFrame,
					 cols_all: list[str],
					 cols_hq: list[str],
					 n_perm: int = 200,
					 rng=None,
					 debug: bool = False):
	if rng is None:
		rng = np.random.default_rng(0)

	# allow missing cols like your union helper typically does
	dfm = scores_df.reindex(columns=list(dict.fromkeys(cols_all)))
	cols_hq = list(dict.fromkeys(cols_hq))

	# evaluated prompts (for THIS layer universe)
	eval_mask = dfm.notna().any(axis=1).to_numpy()
	n_eval = int(eval_mask.sum())
	if n_eval == 0:
		return (np.nan, "no_eval_rows", n_eval, dfm.shape[1], len(cols_hq)) if debug else np.nan

	# restrict HQ to columns that are actually in dfm
	cols_hq = [c for c in cols_hq if c in dfm.columns]
	k, m = len(cols_hq), dfm.shape[1]
	if k == 0:
		return (np.nan, "k=0_after_intersection", n_eval, m, k) if debug else np.nan
	if k >= m:
		return (1.0, "k>=m", n_eval, m, k) if debug else 1.0

	# require HQ to have *any* evaluated row
	n_eval_hq = int(dfm[cols_hq].notna().any(axis=1).sum())
	if n_eval_hq == 0:
		return (np.nan, "no_eval_rows_in_hq", n_eval, m, k) if debug else np.nan

	# numeric/boolean robustness: treat nonzero as True
	df_num = dfm.apply(pd.to_numeric, errors="coerce").fillna(0.0)
	mat = (df_num.to_numpy()[eval_mask] != 0)

	col_to_i = {c: i for i, c in enumerate(dfm.columns)}
	hq_idx = [col_to_i[c] for c in cols_hq if c in col_to_i]
	if len(hq_idx) == 0:
		return (np.nan, "hq_idx_empty", n_eval, m, k) if debug else np.nan

	obs = int(mat[:, hq_idx].any(axis=1).sum())

	ge = 0
	for _ in range(int(n_perm)):
		idx = rng.choice(m, size=k, replace=False)
		u = int(mat[:, idx].any(axis=1).sum())
		ge += (u >= obs)

	p = (ge + 1) / (n_perm + 1)
	return (p, "ok", n_eval, m, k, obs) if debug else p

def write_high_quality_neuron_flip_coverage_by_layer(
	scores_df: pd.DataFrame,
	neurons_sorted,
	rule_best_df: pd.DataFrame,
	out_dir: str,
	stats_dirname: str = "",
	quality_metric: str = "mcc",
	quality_threshold: float = None,
):
	"""
	Produces a figure analogous to `high_quality_neuron_flip_coverage.pdf` but stratified by layer.

	For each layer:
	- counts: total neurons with rules vs high-quality neurons (best rule metric >= threshold)
	- unique flips (union) explained by neurons in that layer, for correct→incorrect and incorrect→correct,
	  shown for all neurons vs the high-quality subset.

	Writes:
	- high_quality_neuron_flip_coverage_by_layer.json
	- high_quality_neuron_flip_coverage_by_layer.pdf
	"""
	if rule_best_df is None or rule_best_df.empty:
		print("[HighQualityByLayer] No rule metrics available; skipping.")
		return None
	# Ensure rule_best_df only contains rules for these neurons + compatible targets
	rule_best_df = _filter_rule_metrics_df(rule_best_df, neurons_sorted)
	if rule_best_df is None or rule_best_df.empty:
		print("[HighQualityByLayer] No rule metrics available after filtering; skipping.")
		return None

	out_dir = str(out_dir)
	stats_dir = os.path.join(out_dir, 'stats', stats_dirname)
	Path(stats_dir).mkdir(parents=True, exist_ok=True)

	metric_key = _norm_rule_metric_name(quality_metric)
	metric_col = _rule_metric_col(metric_key)
	metric_label = _rule_metric_label(metric_key)
	thr = float(quality_threshold)
	if metric_col not in rule_best_df.columns:
		metric_key = "mcc"
		metric_col = "MCC"
		metric_label = "MCC"

	# Identify high-quality neurons by (layer_key, neuron_id)
	tmp = rule_best_df.copy()
	tmp["layer_key"] = tmp["layer_key"].map(_safe_layer_label)  # <- critical
	high_df = tmp[pd.to_numeric(tmp[metric_col], errors="coerce") >= float(thr)]
	high_ids = {(str(r["layer_key"]), int(r["neuron_id"])) for _, r in high_df.dropna(subset=["neuron_id", "layer_key"]).iterrows()}

	# Group neurons by layer
	layer_to_neurons = defaultdict(list)
	for layer_label, neuron_id, _ in unique_everseen(neurons_sorted, key=lambda x: (x[0],x[1])):
		layer_key = _safe_layer_label(layer_label)
		layer_to_neurons[layer_key].append(int(neuron_id))

	# Sort layers using existing helper
	layers = sorted(layer_to_neurons.keys(), key=lambda lk: _layer_sort_key(lk))

	rows = []
	for lk in layers:
		nids = sorted(set(layer_to_neurons[lk]))
		all_neus = [(lk, nid) for nid in nids]
		hq_neus = [(lk, nid) for nid in nids if (lk, nid) in high_ids]

		def _cols(neuron_list, prefix):
			return [f"{prefix}{lkey}_{nid}" for (lkey, nid) in neuron_list]

		cols_c2i_all = _cols(all_neus, "flip_c2i_")
		cols_i2c_all = _cols(all_neus, "flip_i2c_")
		cols_any_all = _cols(all_neus, "flip_")
		cols_c2i_hq = _cols(hq_neus, "flip_c2i_")
		cols_i2c_hq = _cols(hq_neus, "flip_i2c_")
		cols_any_hq = _cols(hq_neus, "flip_")

		union_any_all, _, n_eval_any_all = _union_and_sum_flips(scores_df, cols_any_all)
		union_any_hq, _, n_eval_any_hq = _union_and_sum_flips(scores_df, cols_any_hq)
		union_c2i_all, _, _ = _union_and_sum_flips(scores_df, cols_c2i_all)
		union_c2i_hq, _, _ = _union_and_sum_flips(scores_df, cols_c2i_hq)
		union_i2c_all, _, _ = _union_and_sum_flips(scores_df, cols_i2c_all)
		union_i2c_hq, _, _ = _union_and_sum_flips(scores_df, cols_i2c_hq)

		n_perm = 200
		# print("hq_not_in_all:", set(cols_any_hq) - set(cols_any_all))
		p_any_hq_vs_rand, reason, *rest = _perm_pval_union(scores_df, cols_any_all, cols_any_hq, n_perm=n_perm, debug=True)
		# print(lk, p_any_hq_vs_rand, reason, rest)
		# print(p_any_hq_vs_rand)
		# n_eval = int(scores_df[cols_any_all].notna().any(axis=1).sum()) if cols_any_all else 0
		# print(lk, "m=", len(cols_any_all), "k=", len(cols_any_hq), "n_eval=", n_eval)
		# hq_present = [c for c in cols_any_hq if c in scores_df.columns]
		# hq_missing = [c for c in cols_any_hq if c not in scores_df.columns]
		# n_eval_hq = int(scores_df[hq_present].notna().any(axis=1).sum()) if hq_present else 0
		# print(lk, "hq_present=", len(hq_present), "hq_missing=", len(hq_missing), "n_eval_hq=", n_eval_hq)

		rows.append({
			"layer_key": lk,
			"layer_sort": int(_layer_sort_key(lk)),
			"n_neurons_total": int(len(all_neus)),
			"n_neurons_high_quality": int(len(hq_neus)),
			"n_rows_with_any_eval_all": int(n_eval_any_all),
			"n_rows_with_any_eval_high": int(n_eval_any_hq),
			"flip_any_union_all": int(union_any_all),
			"flip_any_union_high": int(union_any_hq),
			"flip_c2i_union_all": int(union_c2i_all),
			"flip_c2i_union_high": int(union_c2i_hq),
			"flip_i2c_union_all": int(union_i2c_all),
			"flip_i2c_union_high": int(union_i2c_hq),
			"p_any_hq_vs_rand": None if (p_any_hq_vs_rand is None or np.isnan(p_any_hq_vs_rand)) else float(p_any_hq_vs_rand),
			"perm_n": n_perm,
		})

	out = {
		"quality_metric": str(metric_key),
		"quality_metric_label": str(metric_label),
		"quality_threshold": float(thr),
		"layers": rows,
	}
	json_path = os.path.join(stats_dir, "high_quality_neuron_flip_coverage_by_layer.json")
	Path(json_path).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
	print(f"[HighQualityByLayer] Wrote {json_path}")

	# Build figure
	df_plot = pd.DataFrame(rows).sort_values("layer_sort")
	if df_plot.empty:
		print("[HighQualityByLayer] No layers to plot; skipping.")
		return out

	layer_labels = list(map(lambda x: x.replace('_','.') ,df_plot["layer_key"].tolist()))
	x = np.arange(len(layer_labels))

	plt.figure(figsize=(max(10, 0.6 * len(layer_labels)), 6.5))
	ax1 = plt.subplot(2, 1, 1)
	w = 0.38
	ax1.bar(x - w/2, df_plot["n_neurons_total"].to_numpy(), w, label="All neurons w/ rules")
	ax1.bar(x + w/2, df_plot["n_neurons_high_quality"].to_numpy(), w, label=f"{metric_label} ≥ {thr:.2f}")
	ax1.set_ylabel("# neurons")
	# ax1.set_title("High-quality rules per neuron, by layer")
	ax1.set_xticks(x)
	ax1.set_xticklabels(layer_labels, rotation=45, ha="right")
	ax1.tick_params(axis="x", labelbottom=False) # top subplot: hide x tick labels (bottom subplot carries them)
	ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
	ax1.legend(loc="best", fontsize=9)

	ax2 = plt.subplot(2, 1, 2)
	denom = len(scores_df)
	c2i_all_pct = 100.0 * df_plot["flip_c2i_union_all"].to_numpy(dtype=float) / denom
	c2i_hq_pct  = 100.0 * df_plot["flip_c2i_union_high"].to_numpy(dtype=float) / denom
	i2c_all_pct = 100.0 * df_plot["flip_i2c_union_all"].to_numpy(dtype=float) / denom
	i2c_hq_pct  = 100.0 * df_plot["flip_i2c_union_high"].to_numpy(dtype=float) / denom

	# Use lines for readability (many layers)
	ax2.plot(x, c2i_all_pct, marker="o", linewidth=1.2, label="correct→incorrect (all)")
	ax2.plot(x, c2i_hq_pct, marker="o", linewidth=1.2, label=f"correct→incorrect ({metric_label} ≥ {thr:.2f})")
	ax2.plot(x, i2c_all_pct, marker="s", linewidth=1.2, label="incorrect→correct (all)")
	ax2.plot(x, i2c_hq_pct, marker="s", linewidth=1.2, label=f"incorrect→correct ({metric_label} ≥ {thr:.2f})")

	ax2.set_ylabel("Unique flips (% of ablated prompts)")
	# ax2.set_title("Unique flips explained, by layer (union)")
	ax2.set_xlabel("Layer")
	ax2.set_xticks(x)
	ax2.set_xticklabels(layer_labels, rotation=45, ha="right")
	ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
	ax2.legend(loc="best", fontsize=8, ncol=2)

	# --- compact p-value annotations (stars) ---
	pstars = [_p_to_star(p) for p in df_plot.get("p_any_hq_vs_rand", pd.Series([np.nan]*len(df_plot))).to_list()]
	y_star = np.nanmax(np.vstack([c2i_hq_pct, i2c_hq_pct]), axis=0)
	y_star = y_star + 0.8  # small offset above the higher HQ point

	any_pstar = False
	for xi, yi, s in zip(x, y_star, pstars):
		if s:
			ax2.text(xi, yi, s, ha="center", va="bottom", fontsize=8)
			any_pstar = True

	# small legend note (compact)
	if any_pstar:
		plt.figtext(
			0.99, 0.01,
			f"*,**,***: perm p<0.05,0.01,0.001 vs random subset (k=n_HQ), N={n_perm}",
			ha="right", fontsize=7
		)

	plot_path = os.path.join(stats_dir, "high_quality_neuron_flip_coverage_by_layer.pdf")
	plt.savefig(plot_path, dpi=250, bbox_inches="tight")
	plt.close()
	print(f"[HighQualityByLayer] Wrote {plot_path}")

	return out

def maybe_summarize_rule_metrics_and_high_quality(scores_df: pd.DataFrame, neurons_sorted, args, stats_dirname: str):
	if not getattr(args, "summarize_rule_metrics", False) and not getattr(args, "summarize_rule_metrics_only", False):
		return None

	qs = [float(x) for x in str(getattr(args, "rule_metrics_quantiles", "0.1,0.5,0.9")).split(",")[:3]]
	if len(qs) < 3:
		qs = [0.1, 0.5, 0.9]
	quantiles = (qs[0], qs[1], qs[2])

	# Metric/threshold used for summary ranking + "high-quality" filtering.
	qual_metric = getattr(args, "rule_quality_metric", None)
	if qual_metric is None:
		qual_metric = getattr(args, "threshold_metric", None)
	qual_metric = _norm_rule_metric_name(qual_metric)
	qual_thr = getattr(args, "rule_quality_threshold", None)
	
	rule_best_df = write_rule_metrics_stats(
		rules_dir=args.rules_dir,
		out_dir=args.rules_dir,
		stats_dirname=stats_dirname,
		quantiles=quantiles,
		topk=int(getattr(args, "rule_metrics_topk", 50)),
		length_cap=int(getattr(args, "rule_length_cap", 30)),
		quality_metric=qual_metric,
		quality_threshold=qual_thr,
		neurons_sorted=neurons_sorted,
	)

	if rule_best_df is not None:
		write_high_quality_neuron_flip_coverage(
			scores_df=scores_df,
			neurons_sorted=neurons_sorted,
			rule_best_df=rule_best_df,
			out_dir=args.rules_dir,
			stats_dirname=stats_dirname,
			quality_metric=qual_metric,
			quality_threshold=qual_thr,
		)

		write_high_quality_neuron_flip_coverage_by_layer(
			scores_df=scores_df,
			neurons_sorted=neurons_sorted,
			rule_best_df=rule_best_df,
			out_dir=args.rules_dir,
			stats_dirname=stats_dirname,
			quality_metric=qual_metric,
			quality_threshold=qual_thr,
		)

	return rule_best_df

def main():
	args = parse_args()
	if args.stats_dirname is None:
		args.stats_dirname = ""
	set_deterministic(args.seed)

	os.makedirs(os.path.join(args.rules_dir, 'stats', args.stats_dirname), exist_ok=True)

	task = resolve_task_spec(args.task_module)
	is_answer_positive_fn = task.is_answer_positive
	max_new_tokens = task.MAX_NEW_TOKENS
	metrics_list = task.DEFAULT_TARGETS
	main_metric = metrics_list[0]
	show_batch_tqdm = (not args.no_tqdm_batches)

	circuit_agonists_path = Path(args.circuit_agonists_path).resolve()
	scores_path = Path(os.path.join(args.features_scores_dir, 'scores.csv')).resolve()
	if not scores_path.exists():
		raise FileNotFoundError(f"scores_path not found: {scores_path}")

	# Load scores
	ftype = guess_filetype(scores_path)
	if ftype == "parquet":
		scores_df = pd.read_parquet(scores_path)
	else:
		scores_df = pd.read_csv(scores_path)

	# ------------------------- Train/Test split -------------------------
	# If an 'is_test' flag is present, DO NOT drop test rows here.
	# We keep them so we can later build df_eval for rule extraction.
	# (We will still train rules only on ~is_test at the end.)
	if "is_test" in scores_df.columns:
		n_test = int(scores_df["is_test"].astype(bool).sum())
		print(f"[Split] Found is_test flag: {n_test} / {len(scores_df)} rows marked as test (kept in dataframe).")

	prompt_col = task.DEFAULT_INPUT
	if prompt_col not in scores_df.columns:
		raise ValueError(f"Prompt column {prompt_col!r} not found in scores")

	# Derive feature columns from features.json produced by the previous script
	features_json_path = os.path.join(args.features_scores_dir, "features.json")
	with open(features_json_path, "r", encoding="utf-8") as f:
		features_meta = json.load(f)
	features_json_fp = _file_fingerprint(Path(features_json_path))
	def _sanitize(s: str) -> str:
		return re.sub(r"[^0-9a-zA-Z]+", "_", str(s)).strip("_").lower()
	# Column names in scores.csv that correspond to feature labels
	feature_name_set = {
		_sanitize(feat["label"])
		for feat in features_meta
	}

	input_features = [
		c for c in scores_df.columns
		if c in feature_name_set
		and c not in metrics_list
	]

	# Resolve model id: prefer dataset_info.json next to scores if present, else arg, else fallback
	ai_model = args.ai_model
	if ai_model is None:
		# Try dataset_info.json in parent dirs (common in your pipeline)
		for parent in [scores_path.parent, scores_path.parent.parent]:
			info_path = parent / "dataset_info.json"
			if info_path.exists():
				info = json.loads(info_path.read_text())
				ai_model = info.get("ai_model", None)
				if ai_model:
					break

	
	# ----------------- rule-metrics-only mode (no ablations, no extraction) -----------------
	# Useful when you already extracted rules and just want paper-ready summaries/figures.
	if args.summarize_rule_metrics_only:
		print("[Rule-metrics-only] Skipping model loading and ablations. Summarizing rule metrics + high-MCC coverage.")

		stats_dirname = args.stats_dirname if args.stats_dirname is not None else ""
		candidates = []
		if args.rules_dir:
			candidates.append(os.path.join(args.rules_dir, 'stats', stats_dirname, f"scores.csv"))
		if args.features_scores_dir:
			candidates.append(os.path.join(args.features_scores_dir, 'stats', stats_dirname, f"scores.csv"))
		candidates.append(str(scores_path))

		scores_in_path = None
		for p in candidates:
			if p and os.path.exists(p):
				scores_in_path = p
				break
		if scores_in_path is None:
			raise FileNotFoundError("Could not find any scores file to summarize from.")

		ftype2 = guess_filetype(scores_in_path)
		if ftype2 == "parquet":
			scores_stats = pd.read_parquet(scores_in_path)
		else:
			scores_stats = pd.read_csv(scores_in_path)
		print(f"[Rule-metrics-only] Using scores file: {scores_in_path} (rows={len(scores_stats)})")

		# Discover neurons (same filtering as normal/stats-only so counts align)
		print(f"[Rule-metrics-only] Discovering neurons from {circuit_agonists_path} (max_effect >= {args.search_epsilon}) ...")
		neurons = extract_single_neurons(circuit_agonists_path, search_epsilon=float(args.search_epsilon))
		
		if not getattr(args, "skip_agonist_metric_stats", False):
			write_agonist_metric_final_stats(
				circuit_agonists_path,
				args.rules_dir,
				stats_dirname=stats_dirname,
				flip_stats_df=_load_existing_flip_stats(args.rules_dir, stats_dirname),
				topk=args.agonist_metric_topk,
			)

		maybe_summarize_rule_metrics_and_high_quality(scores_stats, neurons, args, stats_dirname)
		return

	# ----------------- stats-only mode (no ablations) -----------------
	# Useful when you already ran this script once and just want additional statistics.
	if args.stats_only:
		print("[Stats-only] Skipping model loading and ablation runs. Computing stats from existing flip columns only.")

		# Prefer the previously written scores_<postfix>.csv if it exists in rules_dir; else fall back to the input scores.csv.
		stats_dirname = args.stats_dirname if args.stats_dirname is not None else ""
		candidates = []
		if args.rules_dir:
			candidates.append(os.path.join(args.rules_dir, 'stats', stats_dirname, f"scores.csv"))
		if args.features_scores_dir:
			candidates.append(os.path.join(args.features_scores_dir, 'stats', stats_dirname, f"scores.csv"))
		candidates.append(str(scores_path))

		scores_in_path = None
		for p in candidates:
			if p and os.path.exists(p):
				# If it's the original scores.csv, it might not have flips; still allow, we'll warn later.
				scores_in_path = p
				break
		if scores_in_path is None:
			raise FileNotFoundError("Could not find any scores file to compute stats from.")

		ftype2 = guess_filetype(scores_in_path)
		if ftype2 == "parquet":
			scores_stats = pd.read_parquet(scores_in_path)
		else:
			scores_stats = pd.read_csv(scores_in_path)

		print(f"[Stats-only] Using scores file: {scores_in_path} (rows={len(scores_stats)})")

		if main_metric not in scores_stats.columns:
			raise ValueError(f"Baseline target column {main_metric!r} not found in scores file used for stats-only mode.")

		# Discover neurons from per_rule and apply bucket filtering (same as normal mode)
		print(f"[Stats-only] Discovering neurons from {circuit_agonists_path} (max_effect >= {args.search_epsilon}) ...")
		neurons = extract_single_neurons(circuit_agonists_path, search_epsilon=float(args.search_epsilon))
		
		# Parse quantiles for range reporting
		flip_stats_df = write_flip_stats(scores_stats, neurons, args.rules_dir, topk=50, stats_dirname=stats_dirname)
		if not getattr(args, "skip_agonist_metric_stats", False):
			write_agonist_metric_final_stats(
				circuit_agonists_path,
				args.rules_dir,
				stats_dirname=stats_dirname,
				flip_stats_df=flip_stats_df,
				topk=args.agonist_metric_topk,
			)

		return

	device = get_device()
	model = LMWrapper(
		model_name=ai_model,
		device=device,
		eval_mode=True,
		# ungroup_grouped_query_attention=False,
		# circuit_discovery=False,
		cache_dir=args.ai_model_cache_dir,
	)
	unhooked_model = getattr(model, "model", None)
	tokenizer = getattr(model, "tokenizer", None)

	# Discover neurons from per_rule
	print(f"[1/5] Discovering neurons from {circuit_agonists_path} (max_effect >= {args.search_epsilon}) ...")
	neurons = extract_single_neurons(circuit_agonists_path, search_epsilon=float(args.search_epsilon))

	if args.max_neurons is not None and args.max_neurons > 0:
		neurons = neurons[: args.max_neurons]
	print(f"Found {len(neurons)} neurons to evaluate.")
	if not neurons:
		print("No neurons found; nothing to do.")
		return

	# Prepare prompts
	all_prompts_full = scores_df.loc[:, prompt_col].astype(str).tolist()
	n_points_total = len(all_prompts_full)
	all_examples_full = scores_df.to_dict(orient="records") # <-- list[dict] rows

	# ----------------- spectral sampling subset (CACHED) -----------------
	assign_to_center = None
	sample_center_ids = None
	sample_indices = np.arange(n_points_total, dtype=int)

	if args.use_spectral_sampling:
		print("[2/5] Building spectral sampling subset (cached) ...")

		# "split" tag for cache separation (edit policy as you like)
		# - If you later decide to embed only train rows, set split_tag="train"
		split_tag = "all"

		# spectral_cfg = _spectral_cache_cfg(
		# 	args=args,
		# 	ai_model=ai_model,
		# 	scores_path=scores_path,
		# 	prompt_col=prompt_col,
		# 	split_tag=split_tag,
		# )
		# spectral_cache_fp = _spectral_cache_path(args, spectral_cfg)
		spectral_cache_fp = _spectral_cache_path(args)

		def _compute_spectral_sampling():
			emb_all, Z = build_reps_and_embedding_from_args(
				args=args,
				texts=all_prompts_full,
				model=unhooked_model,
				tokenizer=tokenizer,
				device=device,
			)
			Z = np.asarray(Z, dtype=np.float32)

			all_idx = np.arange(Z.shape[0], dtype=int)

			out = {
				"Z": Z,  # store spectral embedding
			}

			if args.global_n_clusters > 0:
				global_center_idx, global_cluster_id, min_d2, global_meta, x_norm2 = kcenter_farthest_first(
					Z, k=args.global_n_clusters
				)
				sizes = np.bincount(global_cluster_id, minlength=len(global_center_idx))

				target_n = min(args.sampling_max_points, all_idx.size)
				target_n = max(target_n, min(args.sampling_min_points, all_idx.size))

				sample_indices, cover_meta = representative_sample_from_global_clusters(
					Z=Z,
					x_norm2=x_norm2,
					centers_idx=global_center_idx,
					cluster_id=global_cluster_id,
					group_idx=all_idx,
					n_select=target_n,
				)

				points_per_centroid = float(len(sample_indices)) / float(args.global_n_clusters)

				out.update({
					"mode": "global_clusters",
					"sample_indices": np.asarray(sample_indices, dtype=np.int64),
					"global_center_idx": np.asarray(global_center_idx, dtype=np.int64),
					"global_cluster_id": np.asarray(global_cluster_id, dtype=np.int32),
					"global_meta": global_meta,
					"cover_meta": cover_meta,
					"points_per_centroid": points_per_centroid,
				})
			else:
				# Greedy cover centers + optional expansion per centroid
				centers_idx, cover_meta = greedy_spectral_cover(
					Z,
					all_idx,
					radius=args.coverage_radius,
					max_points=args.sampling_max_points,
					min_points=args.sampling_min_points,
					return_meta=True,
				)
				centers_idx = np.asarray(centers_idx, dtype=np.int64)

				assign_to_center = compute_nearest_center_assignments(
					Z=Z,
					centers_idx=centers_idx,
					chunk_size=args.sampling_chunk_size,
				)

				sample_indices, sample_center_ids = build_per_centroid_sample_indices(
					Z=Z,
					centers_idx=centers_idx,
					assign_to_center=assign_to_center,
					points_per_centroid=args.points_per_centroid,
					mode=args.centroid_sampling_mode,
				)

				out.update({
					"mode": "greedy_cover",
					"centers_idx": centers_idx,
					"assign_to_center": np.asarray(assign_to_center, dtype=np.int32),
					"sample_indices": np.asarray(sample_indices, dtype=np.int64),
					"sample_center_ids": np.asarray(sample_center_ids, dtype=np.int32),
					"cover_meta": cover_meta,
					"points_per_centroid": int(args.points_per_centroid),
				})

			return out

		spectral_obj = load_or_create_cache(
			str(spectral_cache_fp),
			_compute_spectral_sampling,
			quiet=True,
		)

		Z = spectral_obj["Z"]
		sample_indices = spectral_obj["sample_indices"]
		assign_to_center = spectral_obj.get("assign_to_center", None)
		sample_center_ids = spectral_obj.get("sample_center_ids", None)
		points_per_centroid = spectral_obj.get(
			"points_per_centroid",
			(len(sample_indices) / args.global_n_clusters) if args.global_n_clusters > 0 else args.points_per_centroid
		)

		sample_indices = np.asarray(sample_indices, dtype=int)

		print(
			f"[Sampling] Sampled {len(sample_indices)} prompts total "
			f"(~{points_per_centroid} per centroid, mode={args.centroid_sampling_mode}). "
			f"Cache={spectral_cache_fp}"
		)
	else:
		print("[2/5] Spectral sampling disabled; using all prompts.")

	# Decide which datapoints we actually evaluate under ablation.
	# When spectral sampling is enabled, we evaluate:
	#   - the sampled rows
	#   - PLUS all test rows (is_test==True), so df_eval has real flip labels
	if args.use_spectral_sampling:
		# --- sampled set (dedupe, preserve order) ---
		seen = set()
		keep_pos = []
		for pos, idx in enumerate(sample_indices.tolist()):
			if idx in seen:
				continue
			seen.add(idx)
			keep_pos.append(pos)
		eval_indices_sampled = sample_indices[np.asarray(keep_pos, dtype=int)]

		# --- force-include ALL test rows ---
		test_indices = np.array([], dtype=int)
		# if "is_test" in scores_df.columns:
		# 	test_indices = np.where(scores_df["is_test"].astype(bool).to_numpy())[0].astype(int)

		# union + deterministic order
		eval_indices = np.unique(np.concatenate([eval_indices_sampled, test_indices])).astype(int)
		eval_indices.sort()

		eval_prompts = [all_examples_full[i] for i in eval_indices]

		# masks in original-row space
		sampled_mask = np.zeros(n_points_total, dtype=bool)
		sampled_mask[eval_indices_sampled] = True

		evaluated_mask = np.zeros(n_points_total, dtype=bool)
		evaluated_mask[eval_indices] = True

		if test_indices.size:
			print(f"[Sampling] Added {len(test_indices)} test rows to eval set -> total eval points = {len(eval_indices)}")

		if args.keep_unsampled_rows:
			scores_out = scores_df.copy()
			scores_out["_sampled"] = sampled_mask
			scores_out["_evaluated"] = evaluated_mask
		else:
			scores_out = scores_df.iloc[eval_indices].copy().reset_index(drop=True)
			scores_out["_orig_row"] = eval_indices
			scores_out["_sampled"] = sampled_mask[eval_indices]
			scores_out["_evaluated"] = True
	else:
		eval_indices = np.arange(n_points_total, dtype=int)
		eval_prompts = all_examples_full
		scores_out = scores_df

	# Baseline label/correctness (ALWAYS taken from scores.csv in --features_scores_dir)
	print("[3/5] Loading baseline labels from scores ...")

	if main_metric not in scores_df.columns:
		raise ValueError(
			f"Baseline target column {main_metric!r} not found in scores. "
			f"Available columns: {list(scores_df.columns)[:50]}..."
		)

	# baseline over the output dataframe rows (which may be filtered if spectral sampling is enabled)
	baseline_full = (pd.to_numeric(scores_out[main_metric], errors="coerce").to_numpy() > 0.5)
	if args.use_spectral_sampling and args.keep_unsampled_rows:
		baseline_eval = baseline_full[eval_indices]
	else:
		baseline_eval = baseline_full

	# Mean/median replacement scores, if needed
	mean_activations_all = None
	if args.intervention in ("mean", "mean-donor", "mean-positional", "mean-donor-positional"):
		print("[4/5] Precomputing replacement scores ...")
		layer_to_neurons = defaultdict(set)
		for layer_label, neuron_id, _baseline_subset in neurons:
			parsed = get_layer_type_and_ids(layer_label)
			if not parsed:
				continue
			layer_to_neurons[str(layer_label)].add(int(neuron_id))

		if layer_to_neurons:
			layer_to_neurons = {k: sorted(list(v)) for k, v in layer_to_neurons.items()}
			mean_prompt_pool = _build_balanced_mean_prompt_pool(
				scores_df,
				prompt_col=prompt_col,
				n_points=args.points_to_use_for_mean_ablation,
				seed=args.seed,
			)
			if not mean_prompt_pool:
				print("[MeanPool] Skipping: no prompts available for replacement-score estimation.")
			else:
				replacement_cache_cfg = _replacement_scores_cache_cfg(
					args,
					ai_model=ai_model,
					scores_path=scores_path,
					prompt_col=prompt_col,
					main_metric=main_metric,
					layer_to_neurons=layer_to_neurons,
				)
				replacement_cache_fp = _replacement_scores_cache_path(args, replacement_cache_cfg)

				def _compute_replacement_scores():
					return precompute_mean_activations(
						model=model,
						all_prompts=mean_prompt_pool,
						layer_to_neurons=layer_to_neurons,
						n_points=args.points_to_use_for_mean_ablation,
						batch_size=args.batch_size,
						intervention=args.intervention,
						device=device,
					)

				mean_activations_all = load_or_create_cache(
					str(replacement_cache_fp),
					_compute_replacement_scores,
					quiet=True,
				)
				print(f"[MeanPool] Reused cached replacement scores from {replacement_cache_fp}")

	print("[5/5] Running ablations and computing flips ...")
	# ---- precompute per-neuron static stuff once ----
	neuron_specs = []
	for layer_label, neuron_id, _baseline_subset in unique_everseen(neurons, key=lambda x: (x[0], x[1])):
		layer_key = _safe_layer_label(layer_label)
		cache_policy_tag = build_flip_cache_policy_tag(args, main_metric=main_metric)
		rules_subdir = os.path.join(args.rules_dir, cache_policy_tag, "ablation_cache")

		# hooks only depend on neuron+global args (NOT on batch)
		hooks = build_ablation_hooks(
			{layer_label: [neuron_id]},
			last_pos_only=(args.last_pos_only or args.decode_only),
			intervention=args.intervention,
			mean_activations=mean_activations_all,
			device=device,
		)

		neuron_specs.append((layer_label, neuron_id, _baseline_subset, layer_key, rules_subdir, hooks, cache_policy_tag))

	batch_size = args.batch_size
	neuron_batch_size = max(1, int(getattr(args, "neuron_batch_size", 1) or 1))
	print('batch_size:', batch_size, 'neuron_batch_size:', neuron_batch_size)
	neuron_specs_by_layer = defaultdict(list)
	for spec in neuron_specs:
		neuron_specs_by_layer[str(spec[0])].append(spec)

	def _flip_cache_path(rules_subdir, layer_key, neuron_id, batch_start):
		return os.path.join(rules_subdir, f"{layer_key}_{int(neuron_id)}_{batch_start}.pkl")

	def _load_cached_correct(cache_path):
		if not os.path.exists(cache_path):
			return None
		with open(cache_path, "rb") as f:
			return pickle.load(f)

	def _save_cached_correct(cache_path, arr):
		Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
		with open(cache_path, "wb") as f:
			pickle.dump(np.asarray(arr).astype(bool), f)

	def _ensure_flip_cols(layer_key, neuron_id):
		col_any = f"flip_{layer_key}_{int(neuron_id)}"
		col_c2i = f"flip_c2i_{layer_key}_{int(neuron_id)}"
		col_i2c = f"flip_i2c_{layer_key}_{int(neuron_id)}"
		for c in (col_any, col_c2i, col_i2c):
			if c not in scores_out.columns:
				scores_out[c] = pd.array([pd.NA] * len(scores_out), dtype="boolean")
		return col_any, col_c2i, col_i2c

	def _write_ablation_flips(layer_key, neuron_id, ablated_correct, batch_baseline_eval, batch_row_pos):
		col_any, col_c2i, col_i2c = _ensure_flip_cols(layer_key, neuron_id)
		ablated_correct = np.asarray(ablated_correct).astype(bool)
		flip_any = (ablated_correct != batch_baseline_eval)
		flip_c2i = (batch_baseline_eval == True) & (ablated_correct == False)
		flip_i2c = (batch_baseline_eval == False) & (ablated_correct == True)
		scores_out.iloc[batch_row_pos, scores_out.columns.get_loc(col_any)] = flip_any.astype(bool)
		scores_out.iloc[batch_row_pos, scores_out.columns.get_loc(col_c2i)] = flip_c2i.astype(bool)
		scores_out.iloc[batch_row_pos, scores_out.columns.get_loc(col_i2c)] = flip_i2c.astype(bool)

	def _score_repeated_rows(rows, answers):
		return (np.asarray(is_answer_positive_fn(rows, answers), dtype=float) > 0.5).astype(bool)

	def _compute_rowwise_chunk_decode_only(layer_label, chunk_specs, prefix_batches, batch_ranges, batch_prompt):
		K = len(chunk_specs)
		acc_by_spec = {
			(layer_label_i, int(neuron_id_i), int(i)): np.zeros(len(batch_prompt), dtype=bool)
			for i, (layer_label_i, neuron_id_i, *_rest) in enumerate(chunk_specs)
		}
		chunk_ids = [int(spec[1]) for spec in chunk_specs]

		for prefix, (b_start, b_end) in zip(prefix_batches, batch_ranges):
			prefix_rows = batch_prompt[b_start:b_end]
			B = len(prefix_rows)
			if B == 0:
				continue
			repeated_prefix = repeat_prefix_cache_batch(prefix, K)
			row_neuron_ids = chunk_ids * B
			hooks = build_rowwise_ablation_hooks(
				layer_label,
				row_neuron_ids,
				last_pos_only=(args.last_pos_only or args.decode_only),
				intervention=args.intervention,
				mean_activations=mean_activations_all,
				device=device,
			)
			answers = model.generate_from_prefix_cache(
				repeated_prefix,
				fwd_hooks=hooks,
				stop_at_eos=True,
				clone_kv_cache_tensors=False,
			)
			repeated_rows = [row for row in prefix_rows for _ in range(K)]
			acc_flat = _score_repeated_rows(repeated_rows, answers)
			acc_matrix = acc_flat.reshape(B, K).T  # [K, B]
			for j, spec in enumerate(chunk_specs):
				key = (spec[0], int(spec[1]), int(j))
				acc_by_spec[key][b_start:b_end] = acc_matrix[j]

		return [acc_by_spec[(spec[0], int(spec[1]), int(j))] for j, spec in enumerate(chunk_specs)]

	def _compute_rowwise_chunk_prefill_decode(layer_label, chunk_specs, batch_prompt):
		K = len(chunk_specs)
		chunk_ids = [int(spec[1]) for spec in chunk_specs]
		repeated_rows = [row for row in batch_prompt for _ in range(K)]
		row_neuron_ids = chunk_ids * len(batch_prompt)
		hooks = build_rowwise_ablation_hooks(
			layer_label,
			row_neuron_ids,
			last_pos_only=(args.last_pos_only or args.decode_only),
			intervention=args.intervention,
			mean_activations=mean_activations_all,
			device=device,
		)
		repeated_prompts = [str(row[prompt_col]) for row in repeated_rows]
		answers = model.generate(
			repeated_prompts,
			max_new_tokens=max_new_tokens,
			do_sample=False,
			fwd_hooks=hooks,
		)
		acc_flat = _score_repeated_rows(repeated_rows, answers)
		return [acc_flat.reshape(len(batch_prompt), K).T[j] for j in range(K)]

	for start in tqdm(range(0, len(eval_prompts), batch_size), total=(len(eval_prompts) + batch_size - 1)//batch_size, desc="Prefix batches"):
		end = min(start + batch_size, len(eval_prompts))
		batch_prompt = eval_prompts[start:end]
		batch_baseline_eval = baseline_eval[start:end]

		# Where to write this batch in scores_out
		# - if keeping full dataset: write to the original row positions eval_indices[start:end]
		# - otherwise (scores_out == eval subset or full unsampled-disabled): write by positional slice [start:end]
		if args.use_spectral_sampling and args.keep_unsampled_rows:
			batch_row_pos = np.asarray(eval_indices[start:end], dtype=int)  # positions in scores_out
		else:
			batch_row_pos = slice(start, end)

		prefix_batches = None
		batch_ranges = None
		if args.decode_only:
			need_prefill = False
			for layer_label, neuron_id, _baseline_subset, _layer_key, rules_subdir, _hooks, _cache_policy_tag in neuron_specs:
				safe_layer_key = _safe_layer_label(layer_label)
				cache_path = _flip_cache_path(rules_subdir, safe_layer_key, neuron_id, start)
				if not os.path.exists(cache_path):
					need_prefill = True
					break

			if need_prefill:
				prefix_batches, batch_ranges = build_prefix_caches_for_examples(
					model,
					batch_prompt,
					prompt_col,
					max_new_tokens=max_new_tokens,
					batch_size=batch_size,
				)

		# Old execution path, kept for parity and for low-memory runs.
		if neuron_batch_size <= 1:
			for layer_label, neuron_id, _baseline_subset, layer_key, rules_subdir, hooks, _cache_policy_tag in tqdm(neuron_specs, desc="Neurons" if show_batch_tqdm else None):
				layer_key = _safe_layer_label(layer_label)
				_ensure_flip_cols(layer_key, neuron_id)

				def _compute_ablated_correct():
					if args.decode_only:
						_, acc = get_correctness_cached_by_prefix_batches(
							model,
							batch_prompt,
							is_answer_positive_fn,
							prompt_col,
							prefix_batches,
							batch_ranges,
							hooks=hooks,
						)
					else:
						_, acc = get_correctness(
							model,
							batch_prompt,
							is_answer_positive_fn,
							prompt_col,
							max_new_tokens=max_new_tokens,
							hooks=hooks,
							batch_size=batch_size,
							tqdm_desc=None,
						)
					return (np.asarray(acc) > 0.5).astype(bool)

				ablated_correct = load_or_create_cache(
					_flip_cache_path(rules_subdir, layer_key, neuron_id, start),
					_compute_ablated_correct,
					quiet=True,
				)

				_write_ablation_flips(layer_key, neuron_id, ablated_correct, batch_baseline_eval, batch_row_pos)
			continue

		# Row-wise multi-neuron path. Each synthetic row ablates exactly one neuron,
		# including singleton donor-safe replacement for mean-donor variants.
		layer_iter = neuron_specs_by_layer.items()
		# if show_batch_tqdm:
		# 	layer_iter = tqdm(list(layer_iter), desc="Layers")
		for layer_label, specs_this_layer in layer_iter:
			for chunk_start in range(0, len(specs_this_layer), neuron_batch_size):
				chunk_specs_all = specs_this_layer[chunk_start:chunk_start + neuron_batch_size]

				cached_or_none = []
				compute_specs = []
				for spec in chunk_specs_all:
					layer_label_i, neuron_id_i, _baseline_subset_i, layer_key_i, rules_subdir_i, _hooks_i, _cache_policy_tag_i = spec
					layer_key_i = _safe_layer_label(layer_label_i)
					_ensure_flip_cols(layer_key_i, neuron_id_i)
					cache_path = _flip_cache_path(rules_subdir_i, layer_key_i, neuron_id_i, start)
					cached = _load_cached_correct(cache_path)
					cached_or_none.append(cached)
					if cached is None:
						compute_specs.append(spec)

				computed_by_key = {}
				if compute_specs:
					if args.decode_only:
						if prefix_batches is None or batch_ranges is None:
							prefix_batches, batch_ranges = build_prefix_caches_for_examples(
								model,
								batch_prompt,
								prompt_col,
								max_new_tokens=max_new_tokens,
								batch_size=batch_size,
							)
						computed_list = _compute_rowwise_chunk_decode_only(layer_label, compute_specs, prefix_batches, batch_ranges, batch_prompt)
					else:
						computed_list = _compute_rowwise_chunk_prefill_decode(layer_label, compute_specs, batch_prompt)

					for spec, arr in zip(compute_specs, computed_list):
						layer_label_i, neuron_id_i, _baseline_subset_i, layer_key_i, rules_subdir_i, _hooks_i, _cache_policy_tag_i = spec
						layer_key_i = _safe_layer_label(layer_label_i)
						arr = np.asarray(arr).astype(bool)
						computed_by_key[(str(layer_label_i), int(neuron_id_i))] = arr
						_save_cached_correct(_flip_cache_path(rules_subdir_i, layer_key_i, neuron_id_i, start), arr)

				for spec, cached in zip(chunk_specs_all, cached_or_none):
					layer_label_i, neuron_id_i, _baseline_subset_i, layer_key_i, _rules_subdir_i, _hooks_i, _cache_policy_tag_i = spec
					layer_key_i = _safe_layer_label(layer_label_i)
					if cached is None:
						ablated_correct = computed_by_key[(str(layer_label_i), int(neuron_id_i))]
					else:
						ablated_correct = np.asarray(cached).astype(bool)
					_write_ablation_flips(layer_key_i, neuron_id_i, ablated_correct, batch_baseline_eval, batch_row_pos)

	scores_out.to_csv(os.path.join(args.rules_dir, 'stats', args.stats_dirname, f"scores.csv"), index=False)

	# ----------------- aggregate flip stats (per neuron) -----------------
	scores_for_final_stats = scores_out
	if getattr(args, "exclude_discovery_rows_from_final_stats", False):
		discovery_orig_rows = collect_discovery_original_rows(circuit_agonists_path, scores_df)
		scores_for_final_stats = filter_scores_out_for_final_stats(scores_out, discovery_orig_rows)
		excluded_n = int(len(scores_out) - len(scores_for_final_stats))
		Path(os.path.join(args.rules_dir, 'stats', args.stats_dirname)).mkdir(parents=True, exist_ok=True)
		exclusion_payload = {
			"exclude_discovery_rows_from_final_stats": True,
			"n_discovery_original_rows_found": int(len(discovery_orig_rows)),
			"n_rows_excluded_from_scores_out": excluded_n,
			"n_rows_remaining_for_final_stats": int(len(scores_for_final_stats)),
		}
		Path(os.path.join(args.rules_dir, 'stats', args.stats_dirname, 'final_stats_exclusion.json')).write_text(
			json.dumps(exclusion_payload, indent=2, ensure_ascii=False),
			encoding="utf-8",
		)
		print(f"[Stats] Excluded {excluded_n} discovery-overlap rows before final flip stats.")
	flip_stats_df = write_flip_stats(scores_for_final_stats, neurons, args.rules_dir, topk=50, stats_dirname=args.stats_dirname)
	if not getattr(args, "skip_agonist_metric_stats", False):
		write_agonist_metric_final_stats(
			circuit_agonists_path,
			args.rules_dir,
			stats_dirname=args.stats_dirname,
			flip_stats_df=flip_stats_df,
			topk=args.agonist_metric_topk,
		)

	# ----------------- per-neuron rule extraction -----------------
	if args.extract_rules:
		for layer_label, neuron_id, _baseline_subset in tqdm(neurons, desc="Neurons (rules)"):
			layer_key = _safe_layer_label(layer_label)
			col_any = f"flip_{layer_key}_{neuron_id}"
			col_c2i = f"flip_c2i_{layer_key}_{neuron_id}"
			col_i2c = f"flip_i2c_{layer_key}_{neuron_id}"

			cache_policy_tag = build_flip_cache_policy_tag(args, main_metric=main_metric)
			rules_subdir = os.path.join(
				args.rules_dir,
				cache_policy_tag,
				f"{layer_key}_{neuron_id}",
			)
			# ----------------- per-neuron rule extraction -----------------
			# Only proceed if this neuron's flip columns actually exist
			if not any(c in scores_out.columns for c in (col_any, col_c2i, col_i2c)):
				continue

			# Choose targets for this neuron only
			if args.rule_targets is not None and str(args.rule_targets).strip():
				wanted = {t.strip() for t in str(args.rule_targets).split(",") if t.strip()}
				targets_this = [c for c in (col_any, col_c2i, col_i2c) if c in wanted]
			else:
				targets_this = [col_any, col_c2i, col_i2c]

			# Build train/eval splits for rule extraction.
			# - rules_df: used to FIT rules (defaults to all rows)
			# - eval_df:  optional held-out evaluation split (is_test==True)
			if targets_this:
				rules_df = scores_out
				eval_df = None
				if "is_test" in scores_out.columns:
					_test_mask = scores_out["is_test"].astype(bool)
					eval_df = scores_out.loc[_test_mask].copy()
					rules_df = scores_out.loc[~_test_mask].copy()

				# If spectral sampling kept unsampled rows, train/eval should only see sampled rows
				# (unsampled rows have NA flip targets).
				if args.use_spectral_sampling and args.keep_unsampled_rows and ("_sampled" in rules_df.columns):
					rules_df = rules_df.loc[rules_df["_sampled"].astype(bool)].copy()
					if eval_df is not None and ("_sampled" in eval_df.columns):
						eval_df = eval_df.loc[eval_df["_sampled"].astype(bool)].copy()

				# Skip targets that are constant or too imbalanced to learn/plot reliably (CHECK ON TRAIN ONLY).
				def _ok_target(df, t, min_pos=2, min_neg=2):
					y = pd.to_numeric(df[t], errors="coerce").dropna()
					if y.nunique() < 2:
						return False
					pos = int((y > 0.5).sum())
					neg = int((y <= 0.5).sum())
					return (pos >= min_pos) and (neg >= min_neg)
				targets_this = [t for t in targets_this if _ok_target(rules_df, t, min_pos=2, min_neg=2)]
				prefixes = sorted({re.sub(rf"{re.escape(layer_key)}_{neuron_id}$", "", str(t)) for t in targets_this})

				if targets_this:
					# Keep a shared per-neuron directory; signature gating decides whether reuse is valid.
					os.makedirs(rules_subdir, exist_ok=True)
					signature = build_rule_extraction_signature(
						rules_df=rules_df,
						input_features=input_features,
						targets=targets_this,
						args=args,
						features_json_fingerprint=features_json_fp,
					)
					if all(
						should_skip_rule_extraction(
							rules_subdir,
							prefix,
							f"{layer_key}_{neuron_id}",
							signature=signature,
						)
						for prefix in prefixes
					):
						print(f"[Rules] Skipping (cached, signature match) neuron {layer_label}:{neuron_id} -> {targets_this}")
						continue

					neuron_args = copy.copy(args)
					neuron_args.rules_dir = rules_subdir
					print(f"[Rules] Extracting rules for neuron {layer_label}:{neuron_id} -> {targets_this}")
					try:
						run_rule_extraction(
							rules_df,
							input_features=input_features,
							targets=targets_this,
							args=neuron_args,
							# rfmode='classify',
							use_lasso_regression=False,
							df_eval=eval_df,
						)
						_rule_signature_path(rules_subdir).write_text(
							json.dumps(signature, indent=2, ensure_ascii=False),
							encoding="utf-8",
						)
					except Exception as e:
						print(f'[Rules] Could not extract rules: {e}')
						pass

	# ----------------- rule metrics summaries (paper-ready) -----------------
	# (No extra ablations; uses existing rule_combo_*.csv and flip columns in scores_out)
	stats_dirname2 = args.stats_dirname if args.stats_dirname is not None else ""
	maybe_summarize_rule_metrics_and_high_quality(scores_out, neurons, args, stats_dirname2)

if __name__ == "__main__":
	main()
