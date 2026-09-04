from pathlib import Path

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import textwrap
import json
import argparse
import zipfile
from collections import defaultdict
from bisect import bisect_right
from more_itertools import unique_everseen

import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import hashlib
import pickle
import gc


# Local utils (expected to exist in your repo, as in the script you shared)
from core.modeling_and_ablation import (
	LMWrapper,
	get_device,
	get_layer_type_and_ids, 
	build_ablation_hooks, 
	build_rowwise_ablation_hooks,
	repeat_prefix_cache_batch,
	precompute_mean_activations,
)
from core.text_and_rules import guess_filetype
from core.caching_and_prompting import load_or_create_cache, set_deterministic
from core.data_model_for_shap import run_rule_extraction
from core.spectral_analysis import *
from core.feature_extraction_runner import resolve_task_spec, sanitize_feature_name
from core.heldout_set_metrics import compute_singleton_set_metrics, safe_layer_label

# Neuron intervention utilities (shared with script 6).
# These provide efficient prefix-cached evaluation and a statistically-aware epsilon adjustment.
from core.neuron_intervention import (
	build_prefix_caches_for_examples,
	get_correctness,
	get_correctness_cached_by_prefix_batches,
)
from core.binomial_statistics import binom_confint, equivalent_search_epsilon

# Must match core.data_model_for_shap cache signatures.
# Cached files without these tokens are from older semantics and are recomputed.
ALL_FIT_CACHE_SIGNATURE = "all_fit_selected_scored_on_train_plus_eval_v2"
TEST_SELECTED_COMBO_CACHE_SIGNATURE = "test_selected_from_train_rules_v1"
RULE_METRICS_SUMMARY_CACHE_SIGNATURE = "rule_combo_metrics_from_raw_v1"



def _combo_cache_has_signature(path: Path, column: str, expected: str) -> bool:
	if (not path.exists()) or path.stat().st_size <= 0:
		return False
	try:
		df = pd.read_csv(path, nrows=1)
	except Exception:
		return False
	if df.empty or column not in df.columns:
		return False
	return str(df.loc[0, column]) == expected

def _test_selected_combo_cache_current(path: Path) -> bool:
	return _combo_cache_has_signature(path, "test_selected_combo_cache_signature", TEST_SELECTED_COMBO_CACHE_SIGNATURE)

def _all_fit_combo_cache_current(path: Path) -> bool:
	return _combo_cache_has_signature(path, "all_fit_cache_signature", ALL_FIT_CACHE_SIGNATURE)

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

DEFAULT_MIN_RULE_DATASET_COVERAGE = 0.005


def _filter_by_min_dataset_coverage(df: pd.DataFrame, min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE) -> pd.DataFrame:
	"""Keep rule rows whose held-out dataset_coverage is large enough.

	This prevents tiny-support rules (for example one evaluated datapoint) from
	being counted as high-quality merely because their held-out MCC is perfect.
	Set min_dataset_coverage <= 0 to disable.  If coverage is unavailable in an
	older artifact, leave rows untouched rather than silently deleting the run.
	"""
	if df is None or df.empty:
		return df
	try:
		floor = float(min_dataset_coverage)
	except Exception:
		floor = DEFAULT_MIN_RULE_DATASET_COVERAGE
	if floor <= 0:
		return df.copy()
	cov_col = None
	for c in ("dataset_coverage", "coverage"):
		if c in df.columns:
			cov_col = c
			break
	if cov_col is None:
		return df.copy()
	out = df.copy()
	cov = pd.to_numeric(out[cov_col], errors="coerce")
	return out[cov >= floor].copy()

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
		"--candidate_ranking_csv",
		type=str,
		default=None,
		help=(
			"Optional frozen candidate CSV overriding discovery membership for singleton evaluation. "
			"The file must contain layer_label/neuron_id (or layer_key/neuron). This is used by "
			"cross-checkpoint materialization to evaluate one fixed candidate union at multiple models; "
			"it does not alter the original CHA discovery artifacts."
		),
	)
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
	ap.add_argument("--features_scores_dir", type=str, required=True, help="Directory containing scores.csv and features.json (output of the previous script).")
	ap.add_argument("--ai_model", type=str, default=None, help="HF model id (default: read from dataset_info.json if present)")
	ap.add_argument("--ai_model_cache_dir", default=None, type=str)
	ap.add_argument("--stats_dirname", default=None, type=str)

	ap.add_argument("--stats_only", action="store_true", help="Skip ablations; only compute aggregate statistics from existing flip columns (requires a scores_*.csv with flip_ columns).")
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
		help="If set, prefill prompt without hooks and apply ablation hooks only during decoding (output-only phase).",
	)
	ap.add_argument(
		"--input_only",
		action="store_true",
		help=(
			"If set, apply intervention hooks only while processing the input prompt, then decode without hooks. "
			"This input-only phase is mutually exclusive with --decode_only."
		),
	)
	ap.add_argument(
		"--semantic_quality_policy",
		type=str,
		default="filter",
		choices=["none", "filter"],
		help=(
			"How this script uses task-provided semantic_wrong diagnostics during rule extraction. "
			"'filter' removes semantically wrong ablated datapoints from per-neuron rule/MCC fitting and eval; "
			"'none' keeps all datapoints but still writes semantic diagnostics."
		),
	)

	ap.add_argument(
		"--max_neurons",
		type=int,
		default=None,
		help="Optional cap on number of neurons to process (useful for quick tests).",
	)
	ap.add_argument("--seed", type=int, default=42)
	ap.add_argument(
		"--singleton_rate_thresholds",
		default="0.01,0.05,0.10,0.20,0.30",
		help="Comma-separated thresholds t reported as N_t=|{j:s_j>=t}|.",
	)
	ap.add_argument(
		"--metric_denominator_epsilon",
		type=float,
		default=1e-12,
		help="Absolute denominator tolerance used to mark ratios undefined; values are never clipped.",
	)

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
		default=int(os.environ.get("REFINE_SAMPLING_MAX_POINTS", "512")),
		help=(
			"Existing stage-7 cap on prompts evaluated per neuron. Direct stage-7 calls read "
			"REFINE_SAMPLING_MAX_POINTS when it is exported; run_pipeline.sh supplies 10000 by default. "
			"The pool is first restricted "
			"by --evaluation_split and --evaluation_baseline_subset. With spectral sampling, this "
			"caps the spectral sample; without spectral sampling, a deterministic seeded uniform "
			"sample without replacement is used. <=0 means unlimited only when "
			"spectral sampling is disabled."
		),
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
		"--evaluation_split",
		choices=["test", "train", "all"],
		default="test",
		help=(
			"Rows used for singleton interventions and final statistics. Default: test. "
			"test/train require an is_test column; all evaluates every available row."
		),
	)
	ap.add_argument(
		"--evaluation_baseline_subset",
		choices=["all", "positive", "negative"],
		default=os.environ.get("EVALUATION_BASELINE_SUBSET", "all").strip().lower(),
		help=(
			"Optional baseline-predicate conditioning inside the selected evaluation split. "
			"For trigger-lift-conditioned poisoning use positive, so singleton effects are "
			"estimated only on rows that exhibit trigger lift before intervention."
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
		default=0.70,
		help=(
			"HQ-T threshold applied to --rule_quality_metric for TEST-selected OR-combos over train-emitted rules. Default: 0.70. "
		),
	)
	ap.add_argument(
		"--rule_quality_threshold_all_fit",
		type=float,
		default=0.70,
		help=(
			"Descriptive ALL-FIT threshold applied to --rule_quality_metric. Default: 0.70. "
		),
	)
	ap.add_argument(
		"--min_rule_dataset_coverage",
		type=float,
		default=DEFAULT_MIN_RULE_DATASET_COVERAGE,
		help=(
			"Minimum held-out dataset_coverage required before a rule can contribute to "
			"best-per-neuron summaries, high-quality counts, threshold sweeps, or HQ coverage plots. "
			"Use 0 to disable. Default: 0.005."
		),
	)

	ap.add_argument(
		"--no_emit_all_fit_rules",
		dest="emit_all_fit_rules",
		action="store_false",
		help=(
			"Do not compute descriptive ALL-FIT rule combos. By default, Stage 7 emits "
			"rule_combo_all_fit_<target>.csv selected and scored on all available rows; "
			"these are descriptive only and are not held-out HQ scores."
		),
	)
	ap.add_argument(
		"--force_all_fit_recompute",
		action="store_true",
		help=(
			"Regenerate rule_combo_all_fit_<target>.csv artifacts using cached upstream data "
			"where possible. This is useful after fixing ALL-FIT semantics without "
			"rerunning ablations/interventions."
		),
	)
	ap.set_defaults(emit_all_fit_rules=True)

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
		default="core.tasks.arithmetic_task",
		help="Python module path defining parse_prompt, SYSTEM_PROMPT, TOKENS_DICT_KEYS and SEED_FEATURES.",
	)

	args = ap.parse_args()
	return args

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

def intervention_phase_name(args):
	if bool(getattr(args, "input_only", False)):
		return "input_only"
	if bool(getattr(args, "decode_only", False)):
		return "decode_only"
	return "prefill_decode"

def hooks_last_pos_only(args):
	# Output-only runs generate one token at a time from a prefix cache; last-position
	# hooks are the intended semantics there. Input-only and input+output runs default
	# to all prompt positions unless --last_pos_only is explicitly requested.
	return bool(getattr(args, "last_pos_only", False) or bool(getattr(args, "decode_only", False)))

def build_flip_cache_policy_tag(args, *, main_metric=None):
	parts = [
		str(main_metric or "metric"),
		str(getattr(args, "intervention", "unknown")),
		intervention_phase_name(args),
	]
	if bool(getattr(args, "last_pos_only", False)):
		parts.append("last_pos_only")
	evaluation_split = str(getattr(args, "evaluation_split", "test")).strip().lower()
	evaluation_baseline_subset = str(getattr(args, "evaluation_baseline_subset", "all")).strip().lower()
	if evaluation_baseline_subset != "all":
		parts.append(f"baseline_{evaluation_baseline_subset}")
	if evaluation_split == "test":
		# Test-only cache namespace.
		parts.append("holdout_test_only")
	elif evaluation_split != "all":
		parts.append(f"evaluation_split_{evaluation_split}")
	# The cache directory only needs to distinguish the behavioral metric,
	# replacement intervention, intervention phase, and evaluation split.  The
	# concrete stats directory name encodes circuit-search details that do not
	# change a neuron's generated answer on the same evaluation rows, so copying
	# that full label into the cache path only duplicates information.
	return _safe_dirname("-".join(parts))

def _file_fingerprint(p: Path) -> dict:
	p = Path(p)
	st = p.stat()
	return {
		"name": p.name,
		"size": int(st.st_size),
		"mtime": int(st.st_mtime),
	}


def _spectral_cache_path(args, spectral_cfg=None):
	cache_root = Path(args.rules_dir) / "spectral_cache"
	cache_root.mkdir(parents=True, exist_ok=True)
	if spectral_cfg is None:
		return cache_root / "spectral.pkl"
	cache_hash = _sha1_of_obj(spectral_cfg)[:12]
	return cache_root / f"spectral_{cache_hash}.pkl"


def _replacement_cache_identity(scores_path: Path, task_module: str | None) -> tuple[str, str | None]:
	"""Return the stable cache identity for control-correctness terminology aliases.

	The normal-task and attack-cohort control endpoints were renamed without changing
	the underlying replacement-reference population.  Normalize those path/spec names
	to the established cache namespace so terminology changes do not trigger model
	recomputation.  File content, model, intervention, population size, and units remain
	part of the cache configuration.
	"""
	path_key = str(Path(scores_path).resolve())
	path_key = path_key.replace("/normal_task_correctness/", "/ordinary_correctness/")
	path_key = path_key.replace("/attack_cohort_control_correctness/", "/ordinary_correctness/")
	task_key = task_module
	if isinstance(task_key, str):
		task_key = task_key.replace(":NORMAL_TASK_SPEC", ":ORDINARY_TASK_SPEC")
		task_key = task_key.replace(":ATTACK_COHORT_CONTROL_CORRECTNESS_SPEC", ":ORDINARY_TASK_SPEC")
	return path_key, task_key


def _replacement_scores_cache_cfg(args, *, ai_model: str, scores_path: Path, prompt_col: str, main_metric: str, layer_to_neurons: dict) -> dict:
	"""Cache identity for the expensive mean/median replacement-stat precompute."""
	path_key, task_key = _replacement_cache_identity(
		scores_path, getattr(args, "task_module", None)
	)
	return {
		"ai_model": str(ai_model),
		"scores_path": path_key,
		"scores_fingerprint": _file_fingerprint(scores_path),
		"prompt_col": str(prompt_col),
		"main_metric": str(main_metric),
		"task_module": task_key,
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

			if abs(me) < float(search_epsilon):
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


def extract_frozen_candidate_ranking(
	circuit_agonists_path: Path,
	selected_neurons,
	*,
	search_epsilon: float,
):
	"""Return the candidate order frozen on stage-6 discovery data.

	The ranking score is ``abs(max_effect)`` from the latest stage-6 singleton
	record. Held-out stage-7 flip rates are never used for ordering. Duplicate
	records are resolved by the largest absolute discovery effect, followed by a
	deterministic layer/unit tie break.
	"""
	selected = {(str(layer), int(nid)) for layer, nid, _ in selected_neurons}
	records = {}
	discovery_baselines = defaultdict(set)

	def _consume(buckets, source_name):
		nca = buckets.get("non_catastrophic_agonists", {})
		if not isinstance(nca, dict):
			return
		keep = bucket_keep_keys(buckets, exclude_candidates=True) or set(nca)
		for key in keep:
			entry = nca.get(key, {})
			if not isinstance(entry, dict):
				continue
			rec = entry.get("last_record") or {}
			try:
				signed = float(rec.get("max_effect"))
			except Exception:
				continue
			if abs(signed) < float(search_epsilon):
				continue
			if "layer_label" in rec and "neuron_id" in rec:
				layer, nid = str(rec["layer_label"]), int(rec["neuron_id"])
			else:
				try:
					layer, raw_nid = str(key).rsplit(":", 1)
					nid = int(raw_nid)
				except Exception:
					continue
			unit = (layer, nid)
			if unit not in selected:
				continue
			baseline_subset = str(rec.get("baseline_subset", "")).strip().lower()
			if baseline_subset in {"positive", "negative"}:
				discovery_baselines[unit].add(baseline_subset)
			previous = records.get(unit)
			payload = {
				"layer_label": layer,
				"neuron_id": nid,
				"unit_key": f"{layer}:{nid}",
				"discovery_score": abs(signed),
				"discovery_score_signed": signed,
				"discovery_baseline_subset": rec.get("baseline_subset"),
				"ranking_source": "stage6_abs_max_effect",
				"ranking_source_file": str(source_name),
			}
			if previous is None or payload["discovery_score"] > previous["discovery_score"]:
				records[unit] = payload

	if circuit_agonists_path.is_dir():
		for fp in sorted(circuit_agonists_path.rglob("neuron_buckets.json")):
			_consume(json.loads(fp.read_text(encoding="utf-8")), fp)
	elif circuit_agonists_path.is_file() and circuit_agonists_path.suffix.lower() == ".zip":
		with zipfile.ZipFile(circuit_agonists_path) as zf:
			for name in sorted(n for n in zf.namelist() if n.endswith("neuron_buckets.json")):
				with zf.open(name) as handle:
					_consume(json.load(handle), name)
	else:
		raise ValueError("circuit_agonists_path must be a directory or a .zip file")

	missing = sorted(selected - set(records))
	if missing:
		raise ValueError(
			"Cannot freeze H_m because discovery scores are missing for selected candidates: "
			+ ", ".join(f"{layer}:{nid}" for layer, nid in missing[:10])
		)

	rows = list(records.values())
	for row in rows:
		unit = (str(row["layer_label"]), int(row["neuron_id"]))
		baselines = sorted(discovery_baselines.get(unit, set()))
		primary = str(row.get("discovery_baseline_subset", "")).strip().lower()
		if primary in {"positive", "negative"} and primary not in baselines:
			baselines.append(primary)
		baselines = sorted(set(baselines))
		row["discovery_baseline_subset"] = primary if primary in {"positive", "negative"} else (baselines[0] if baselines else "")
		row["discovery_baseline_subsets"] = "|".join(baselines)
		row["n_discovery_baseline_subsets"] = int(len(baselines))
		parsed = get_layer_type_and_ids(row["layer_label"])
		if parsed is None:
			row.update({
				"transformer_layer": None,
				"computational_locus": "unknown",
				"channel_type": "unknown",
			})
		elif parsed[0] == "mlp":
			row.update({
				"transformer_layer": int(parsed[1]),
				"computational_locus": "mlp_output",
				"channel_type": "mlp_neuron",
			})
		else:
			row.update({
				"transformer_layer": int(parsed[1]),
				"computational_locus": f"attention_head_{int(parsed[2])}",
				"channel_type": "attention_dimension",
			})
	rows.sort(key=lambda row: (
		-float(row["discovery_score"]),
		_layer_sort_key(row["layer_label"]),
		str(row["layer_label"]),
		int(row["neuron_id"]),
	))
	for rank, row in enumerate(rows, start=1):
		row["discovery_rank_global"] = rank
	by_layer = defaultdict(list)
	for row in rows:
		by_layer[row["layer_label"]].append(row)
	for layer_rows in by_layer.values():
		for rank, row in enumerate(layer_rows, start=1):
			row["discovery_rank_within_layer"] = rank
	return pd.DataFrame(rows)


def load_candidate_ranking_override(path: str | Path):
	"""Load a fixed candidate set for cross-model singleton evaluation.

	Only unit identity is required.  Discovery scores are retained when present
	for provenance, but they are never re-thresholded here.
	"""
	p = Path(path).expanduser().resolve()
	if not p.is_file():
		raise FileNotFoundError(f"candidate_ranking_csv not found: {p}")
	df = pd.read_csv(p)
	if "layer_label" not in df.columns and "layer_key" in df.columns:
		df["layer_label"] = df["layer_key"].astype(str)
	if "neuron_id" not in df.columns and "neuron" in df.columns:
		parts = df["neuron"].astype(str).str.rsplit(":", n=1, expand=True)
		if parts.shape[1] == 2:
			df["neuron_id"] = pd.to_numeric(parts[1], errors="coerce")
	if not {"layer_label", "neuron_id"}.issubset(df.columns):
		raise ValueError(f"candidate_ranking_csv must contain layer_label/neuron_id: {p}")
	df = df.dropna(subset=["layer_label", "neuron_id"]).copy()
	df["layer_label"] = df["layer_label"].astype(str)
	df["neuron_id"] = pd.to_numeric(df["neuron_id"], errors="raise").astype(int)
	df["unit_key"] = [f"{a}:{b}" for a, b in zip(df["layer_label"], df["neuron_id"])]
	df = df.drop_duplicates("unit_key", keep="first").reset_index(drop=True)
	if "discovery_score" not in df.columns:
		df["discovery_score"] = np.nan
	if "discovery_score_signed" not in df.columns:
		df["discovery_score_signed"] = np.nan
	if "discovery_baseline_subset" not in df.columns:
		df["discovery_baseline_subset"] = "positive"
	if "discovery_baseline_subsets" not in df.columns:
		df["discovery_baseline_subsets"] = df["discovery_baseline_subset"].astype(str)
	if "n_discovery_baseline_subsets" not in df.columns:
		df["n_discovery_baseline_subsets"] = df["discovery_baseline_subsets"].astype(str).map(
			lambda value: len({x.strip().lower() for x in value.split("|") if x.strip().lower() in {"positive", "negative"}})
		)
	if "discovery_rank_global" not in df.columns:
		df["discovery_rank_global"] = np.arange(1, len(df) + 1, dtype=int)
	df["ranking_source"] = df.get("ranking_source", pd.Series("fixed_candidate_override", index=df.index)).fillna("fixed_candidate_override")
	df["ranking_source_file"] = str(p)
	neurons = [
		(str(r.layer_label), int(r.neuron_id), str(getattr(r, "discovery_baseline_subset", "positive") or "positive"))
		for r in df.itertuples(index=False)
	]
	return neurons, df


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

def build_rule_extraction_signature(rules_df: pd.DataFrame, input_features, targets, args, features_json_fingerprint=None, eval_df: pd.DataFrame = None) -> dict:
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
		"semantic_quality_policy": str(getattr(args, "semantic_quality_policy", "none")),
		"emit_all_fit_rules": bool(getattr(args, "emit_all_fit_rules", True)),
		"all_fit_cache_signature_expected": ALL_FIT_CACHE_SIGNATURE,
	}

	input_feature_names = list(map(str, input_features))
	return {
		"n_rows": int(len(rules_df)),
		"n_eval_rows": int(len(eval_df)) if eval_df is not None else None,
		"targets": list(map(str, targets)),
		"target_hashes": target_hashes,
		"input_features_sha1": _sha1_of_obj(input_feature_names),
		"features_json_fingerprint": features_json_fingerprint,
		"args": relevant_args,
	}

def _rule_signature_path(rules_subdir):
	return Path(rules_subdir) / "_rule_extraction_signature.json"

def should_skip_rule_extraction(rules_subdir, prefix, layer_label, signature=None, force_recompute: bool = False, require_all_fit: bool = True, targets=None, force_all_fit_recompute: bool = False):
	"""Skip only when every required raw output exists and its signature matches.

	Required raw outputs are the TRAIN-selected rule combination, its frozen TEST
	score, the TEST-selected rule combination, and, when enabled, the ALL-FIT
	descriptive combination for every target.
	"""
	if force_recompute:
		return False
	rules_path = Path(rules_subdir)
	if targets is None:
		targets = [f"{prefix}{layer_label}"]
	missing = []
	stale_all_fit = []
	for target in targets:
		base_required = [
			rules_path / f"rule_combo_train_{target}.csv",
			rules_path / f"rule_combo_{target}.csv",
		]
		for fp in base_required:
			if (not fp.exists()) or fp.stat().st_size <= 0:
				missing.append(fp.name)
		test_selected_fp = rules_path / f"rule_combo_test_selected_{target}.csv"
		if not _test_selected_combo_cache_current(test_selected_fp):
			missing.append(test_selected_fp.name)
		if require_all_fit:
			all_fit_fp = rules_path / f"rule_combo_all_fit_{target}.csv"
			if not _all_fit_combo_cache_current(all_fit_fp):
				stale_all_fit.append(all_fit_fp.name)
	if missing:
		print(f"[Rules] Cache incomplete in {rules_path}: missing {missing[:6]}{'...' if len(missing) > 6 else ''}")
		return False
	if stale_all_fit:
		print(f"[Rules] ALL-FIT cache missing/stale in {rules_path}: {stale_all_fit[:6]}{'...' if len(stale_all_fit) > 6 else ''}")
		return False
	if force_all_fit_recompute:
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

def write_flip_stats(
	scores_df: pd.DataFrame,
	neurons_sorted,
	out_dir,
	topk=50,
	stats_dirname='',
	baseline_metric_col: str = None,
	search_epsilon: float = None,
	reference_n_per_side: int = None,
	frozen_candidate_ranking_df: pd.DataFrame | None = None,
):
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

	# Direction columns are derived data. Rebuild them from the authoritative
	# flip-any columns and the unablated binary predicate before aggregation.
	_repair_directional_flip_columns(scores_df, neurons_sorted, baseline_metric_col)

	rows = []
	any_eval_col = None
	cols_any_found = []
	cols_c2i_found = []
	cols_i2c_found = []
	cols_flip_semantic_wrong_found = []
	cols_semantic_wrong_found = []
	cols_flip_empty_found = []
	cols_flip_unparseable_found = []

	for layer_label, neuron_id, _ in unique_everseen(neurons_sorted, key=lambda x: (x[0],x[1])):
		layer_key = safe_layer_label(layer_label)
		col_any = f"flip_{layer_key}_{neuron_id}"
		col_c2i = f"flip_c2i_{layer_key}_{neuron_id}"
		col_i2c = f"flip_i2c_{layer_key}_{neuron_id}"
		col_flip_semantic_wrong = f"flip_semantic_wrong_{layer_key}_{neuron_id}"
		col_semantic_wrong = f"semantic_wrong_{layer_key}_{neuron_id}"
		col_flip_empty = f"flip_empty_{layer_key}_{neuron_id}"
		col_flip_unparseable = f"flip_unparseable_{layer_key}_{neuron_id}"

		if col_any not in scores_df.columns:
			continue
		cols_any_found.append(col_any)
		if col_c2i in scores_df.columns:
			cols_c2i_found.append(col_c2i)
		if col_i2c in scores_df.columns:
			cols_i2c_found.append(col_i2c)
		if col_flip_semantic_wrong in scores_df.columns:
			cols_flip_semantic_wrong_found.append(col_flip_semantic_wrong)
		if col_semantic_wrong in scores_df.columns:
			cols_semantic_wrong_found.append(col_semantic_wrong)
		if col_flip_empty in scores_df.columns:
			cols_flip_empty_found.append(col_flip_empty)
		if col_flip_unparseable in scores_df.columns:
			cols_flip_unparseable_found.append(col_flip_unparseable)
		if any_eval_col is None:
			any_eval_col = col_any

		n_eval = int(scores_df[col_any].notna().sum())
		c2i = int(scores_df[col_c2i].fillna(False).astype(int).sum()) if col_c2i in scores_df.columns else 0
		i2c = int(scores_df[col_i2c].fillna(False).astype(int).sum()) if col_i2c in scores_df.columns else 0
		total = int(scores_df[col_any].fillna(False).astype(int).sum())
		semantic_wrong = int(scores_df[col_semantic_wrong].fillna(False).astype(int).sum()) if col_semantic_wrong in scores_df.columns else 0
		flip_semantic_wrong = int(scores_df[col_flip_semantic_wrong].fillna(False).astype(int).sum()) if col_flip_semantic_wrong in scores_df.columns else 0
		flip_empty = int(scores_df[col_flip_empty].fillna(False).astype(int).sum()) if col_flip_empty in scores_df.columns else 0
		flip_unparseable = int(scores_df[col_flip_unparseable].fillna(False).astype(int).sum()) if col_flip_unparseable in scores_df.columns else 0

		# Percentages are always relative to the number of evaluated rows for this neuron.
		c2i_pct = (100.0 * c2i / n_eval) if n_eval else float("nan")
		i2c_pct = (100.0 * i2c / n_eval) if n_eval else float("nan")
		total_pct = (100.0 * total / n_eval) if n_eval else float("nan")
		ci_alpha = float(os.environ.get("EVALUATION_CONFIDENCE_ALPHA", "0.05"))
		flip_ci_low, flip_ci_high = binom_confint(total, n_eval, alpha=ci_alpha)

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
			"flip_any_ci_low": flip_ci_low,
			"flip_any_ci_high": flip_ci_high,
			"confidence_level": 1.0 - ci_alpha,
			"semantic_wrong_count": semantic_wrong,
			"semantic_wrong_rate": (semantic_wrong / n_eval) if n_eval else float("nan"),
			"flip_semantic_wrong_count": flip_semantic_wrong,
			"flip_semantic_wrong_rate": (flip_semantic_wrong / n_eval) if n_eval else float("nan"),
			"flip_semantic_wrong_share": (flip_semantic_wrong / total) if total else float("nan"),
			"flip_empty_count": flip_empty,
			"flip_empty_rate": (flip_empty / n_eval) if n_eval else float("nan"),
			"flip_unparseable_count": flip_unparseable,
			"flip_unparseable_rate": (flip_unparseable / n_eval) if n_eval else float("nan"),
			"c2i_pct": c2i_pct,
			"i2c_pct": i2c_pct,
			"flip_any_pct": total_pct,
			"semantic_wrong_pct": (100.0 * semantic_wrong / n_eval) if n_eval else float("nan"),
			"flip_semantic_wrong_pct": (100.0 * flip_semantic_wrong / n_eval) if n_eval else float("nan"),
		})

	if not rows:
		print("[Stats] No flip columns found; skipping flip stats.")
		return None

	stats_df = pd.DataFrame(rows)
	if frozen_candidate_ranking_df is not None and not frozen_candidate_ranking_df.empty:
		ranking = frozen_candidate_ranking_df.copy()
		if "layer_label" not in ranking.columns and "layer_key" in ranking.columns:
			ranking["layer_label"] = ranking["layer_key"].astype(str)
		if {"layer_label", "neuron_id"}.issubset(ranking.columns):
			keep = [c for c in [
				"layer_label", "neuron_id", "discovery_baseline_subset", "discovery_baseline_subsets",
				"n_discovery_baseline_subsets", "discovery_score", "discovery_score_signed",
				"discovery_rank_global", "discovery_rank_within_layer", "channel_type",
				"computational_locus", "transformer_layer",
			] if c in ranking.columns]
			ranking = ranking[keep].drop_duplicates(["layer_label", "neuron_id"], keep="first")
			stats_df = stats_df.merge(ranking, on=["layer_label", "neuron_id"], how="left", validate="one_to_one")
	stats_df = stats_df.sort_values(["flip_any_count", "flip_semantic_wrong_count", "c2i_count", "i2c_count"], ascending=False)
	stats_path = os.path.join(out_dir, 'stats', stats_dirname, f"flip_stats_by_neuron.csv")
	stats_df.to_csv(stats_path, index=False)
	print(f"[Stats] Wrote {stats_path}")

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

	union_any_mask = np.zeros(len(scores_df), dtype=bool)
	eval_any_mask = np.zeros(len(scores_df), dtype=bool)
	for c in cols_any_found:
		if c not in scores_df.columns:
			continue
		series = scores_df[c]
		eval_any_mask |= series.notna().to_numpy()
		union_any_mask |= series.fillna(False).astype(bool).to_numpy()
	union_any = int(union_any_mask.sum())
	eval_any = int(eval_any_mask.sum())

	baseline_masks = _baseline_direction_masks(scores_df, baseline_metric_col)
	if baseline_masks is None:
		raise ValueError(
			"Directional coverage requires the unablated baseline predicate column. "
			"Pass baseline_metric_col=main_metric when writing flip statistics."
		)
	valid_baseline_mask, positive_baseline_mask, negative_baseline_mask = baseline_masks
	if np.any(eval_any_mask & ~valid_baseline_mask):
		raise ValueError(
			"Some evaluated rows have no valid unablated binary predicate."
		)
	eval_c2i = int((eval_any_mask & positive_baseline_mask).sum())
	eval_i2c = int((eval_any_mask & negative_baseline_mask).sum())
	union_c2i = int((union_any_mask & positive_baseline_mask).sum())
	union_i2c = int((union_any_mask & negative_baseline_mask).sum())
	if eval_c2i + eval_i2c != eval_any:
		raise ValueError(
			f"Directional denominator mismatch: {eval_c2i}+{eval_i2c} != {eval_any}."
		)
	if union_c2i + union_i2c != union_any:
		raise ValueError(
			f"Directional union mismatch: {union_c2i}+{union_i2c} != {union_any}."
		)

	union_flip_semantic_wrong, eval_flip_semantic_wrong = _union_counts(cols_flip_semantic_wrong_found)
	union_semantic_wrong, eval_semantic_wrong = _union_counts(cols_semantic_wrong_found)
	union_flip_empty, eval_flip_empty = _union_counts(cols_flip_empty_found)
	union_flip_unparseable, eval_flip_unparseable = _union_counts(cols_flip_unparseable_found)
	ci_alpha = float(os.environ.get("EVALUATION_CONFIDENCE_ALPHA", "0.05"))
	union_ci_low, union_ci_high = binom_confint(union_any, eval_any, alpha=ci_alpha)
	if reference_n_per_side is None:
		ref_n_raw = os.environ.get("SEARCH_EPSILON_REFERENCE_N", "").strip()
		ref_n = int(ref_n_raw) if ref_n_raw else None
	else:
		ref_n = int(reference_n_per_side)
	# search_epsilon is passed explicitly from main() so this helper does not
	# depend on argparse state outside its scope.
	ref_tau = float(search_epsilon) if search_epsilon is not None else None
	equivalent_tau_at_eval_n = None
	if ref_n and ref_tau is not None and eval_any > 0:
		equivalent_tau_at_eval_n = max(
			ref_tau,
			equivalent_search_epsilon(
				eval_any, search_epsilon_ref=ref_tau, n_ref=ref_n, prune_alpha=ci_alpha / 2.0
			),
		)
	global_payload = {
		"n_neurons": int(len(stats_df)),
		"n_evaluated_rows": int(eval_any),
		"sum_c2i_counts_over_neurons": int(stats_df["c2i_count"].sum()),
		"sum_i2c_counts_over_neurons": int(stats_df["i2c_count"].sum()),
		"sum_flip_any_counts_over_neurons": int(stats_df["flip_any_count"].sum()),
		"union_flip_any_unique_count": int(union_any),
		"union_flip_any_unique_rate": (float(union_any) / float(eval_any)) if eval_any else float("nan"),
		"union_flip_any_unique_ci_low": union_ci_low,
		"union_flip_any_unique_ci_high": union_ci_high,
		"confidence_level": 1.0 - ci_alpha,
		"confidence_method": "Clopper-Pearson exact (Wilson fallback if SciPy unavailable)",
		"reference_cha_tau": ref_tau,
		"reference_cha_n_per_side": ref_n,
		"equivalent_reference_tau_at_eval_n": equivalent_tau_at_eval_n,
		"baseline_metric_col": str(baseline_metric_col),
		"union_c2i_unique_count": int(union_c2i),
		"n_evaluated_c2i_rows": int(eval_c2i),
		"union_c2i_unique_rate": (float(union_c2i) / float(eval_c2i)) if eval_c2i else float("nan"),
		"union_i2c_unique_count": int(union_i2c),
		"n_evaluated_i2c_rows": int(eval_i2c),
		"union_i2c_unique_rate": (float(union_i2c) / float(eval_i2c)) if eval_i2c else float("nan"),
		"sum_semantic_wrong_counts_over_neurons": int(stats_df.get("semantic_wrong_count", pd.Series(dtype=float)).sum()),
		"sum_flip_semantic_wrong_counts_over_neurons": int(stats_df.get("flip_semantic_wrong_count", pd.Series(dtype=float)).sum()),
		"union_semantic_wrong_unique_count": int(union_semantic_wrong),
		"union_semantic_wrong_unique_rate": (float(union_semantic_wrong) / float(eval_semantic_wrong)) if eval_semantic_wrong else float("nan"),
		"union_flip_semantic_wrong_unique_count": int(union_flip_semantic_wrong),
		"union_flip_semantic_wrong_unique_rate": (float(union_flip_semantic_wrong) / float(eval_flip_semantic_wrong)) if eval_flip_semantic_wrong else float("nan"),
		"union_flip_empty_unique_count": int(union_flip_empty),
		"union_flip_empty_unique_rate": (float(union_flip_empty) / float(eval_flip_empty)) if eval_flip_empty else float("nan"),
		"union_flip_unparseable_unique_count": int(union_flip_unparseable),
		"union_flip_unparseable_unique_rate": (float(union_flip_unparseable) / float(eval_flip_unparseable)) if eval_flip_unparseable else float("nan"),
	}
	Path(os.path.join(out_dir, 'stats', stats_dirname, f"flip_stats_global.json")).write_text(
		json.dumps(global_payload, indent=2, ensure_ascii=False),
		encoding="utf-8",
	)
	print(f"[Stats] Wrote {os.path.join(out_dir, 'stats', stats_dirname, f'flip_stats_global.json')}")

	K = int(min(topk, len(stats_df)))

	# Large top-k values can make this figure dense; callers can lower topk for compact output.
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

	if "flip_semantic_wrong_pct" in stats_df.columns and np.isfinite(pd.to_numeric(stats_df["flip_semantic_wrong_pct"], errors="coerce")).any():
		dfq = stats_df.head(K).copy()
		bad = pd.to_numeric(dfq["flip_semantic_wrong_pct"], errors="coerce").fillna(0.0).to_numpy()
		total_flip = pd.to_numeric(dfq["flip_any_pct"], errors="coerce").fillna(0.0).to_numpy()
		y = np.arange(K)
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
			fig.subplots_adjust(left=0.31, right=0.99, bottom=0.095, top=0.930)
			ax.barh(y, total_flip, height=0.62, label="All flips", linewidth=0.22, edgecolor="black")
			ax.barh(y, bad, height=0.34, label="Semantically wrong flips", linewidth=0.22, edgecolor="black")
			ax.set_yticks(y)
			ax.set_yticklabels(dfq["neuron"].values)
			ax.tick_params(axis="y", pad=2)
			ax.invert_yaxis()
			ax.set_ylim(K - 0.5, -0.62)
			ax.set_xlabel("Datapoints (% of evaluated prompts)")
			ax.grid(axis="x", linewidth=0.3, alpha=0.4)
			ax.set_axisbelow(True)
			ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.012), ncol=2, frameon=False, handlelength=1.3, columnspacing=1.0, borderaxespad=0.0)
			ax.spines["top"].set_visible(False)
			ax.spines["right"].set_visible(False)
			sem_plot_path = os.path.join(out_dir, 'stats', stats_dirname, f"flip_semantic_wrong_top{K}.pdf")
			fig.savefig(sem_plot_path, bbox_inches="tight", pad_inches=0.03)
			plt.close(fig)
		print(f"[Stats] Wrote {sem_plot_path}")

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



def write_heldout_set_metrics(
	scores_df: pd.DataFrame,
	flip_stats_df: pd.DataFrame,
	frozen_ranking_df: pd.DataFrame,
	out_dir,
	*,
	stats_dirname: str,
	baseline_metric_col: str,
	thresholds,
	denominator_epsilon: float,
):
	"""Write the exact singleton-set metrics and frozen discovery ranking."""
	stats_dir = Path(out_dir) / "stats" / str(stats_dirname or "")
	stats_dir.mkdir(parents=True, exist_ok=True)
	frozen_path = stats_dir / "frozen_candidate_ranking.csv"
	frozen_ranking_df.to_csv(frozen_path, index=False)

	summary, candidate_df, topm_df, threshold_df = compute_singleton_set_metrics(
		scores=scores_df,
		candidate_stats=flip_stats_df,
		baseline_col=baseline_metric_col,
		frozen_ranking=frozen_ranking_df,
		thresholds=thresholds,
		denominator_epsilon=float(denominator_epsilon),
	)
	candidate_df.to_csv(stats_dir / "singleton_channel_metrics.csv", index=False)
	topm_df.to_csv(stats_dir / "frozen_topm_metrics.csv", index=False)
	threshold_df.to_csv(stats_dir / "singleton_threshold_counts.csv", index=False)
	pd.DataFrame([{
		key: value
		for key, value in summary.items()
		if not isinstance(value, (dict, list))
	}]).to_csv(stats_dir / "singleton_set_metrics.csv", index=False)
	(stats_dir / "singleton_set_metrics.json").write_text(
		json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True),
		encoding="utf-8",
	)

	if not topm_df.empty:
		fig, ax = plt.subplots(figsize=(5.6, 3.4))
		ax.plot(topm_df["m"], topm_df["TOC_m"], marker="o", linewidth=1.3, markersize=3, label="pooled")
		if "TOC_m_i2c" in topm_df.columns:
			ax.plot(topm_df["m"], topm_df["TOC_m_i2c"], marker="o", linewidth=1.3, markersize=3, label="0→1")
		if "TOC_m_c2i" in topm_df.columns:
			ax.plot(topm_df["m"], topm_df["TOC_m_c2i"], marker="o", linewidth=1.3, markersize=3, label="1→0")
		ax.axhline(1.0, linestyle="--", linewidth=1.0)
		ax.set_xlabel("Frozen discovery top-m channels")
		ax.set_ylabel(r"$\mathrm{TOC}_m(J)$")
		ax.set_title("Frozen top-m coverage by direction")
		ax.legend(frameon=False, fontsize=8)
		ax.spines["top"].set_visible(False)
		ax.spines["right"].set_visible(False)
		fig.tight_layout()
		fig.savefig(stats_dir / "frozen_topm_toc.pdf")
		plt.close(fig)

	print(f"[Stats] Wrote held-out set metrics to {stats_dir}")
	return summary


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
	"semantic_wrong_rate": "Semantic-wrong output rate",
	"flip_semantic_wrong_rate": "Semantic-wrong flip rate",
	"flip_empty_rate": "Empty flip rate",
	"flip_unparseable_rate": "Unparseable flip rate",
	"flip_any_count": "Any flip count",
	"c2i_count": "Correct→incorrect count",
	"i2c_count": "Incorrect→correct count",
	"semantic_wrong_count": "Semantic-wrong output count",
	"flip_semantic_wrong_count": "Semantic-wrong flip count",
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
		"semantic_wrong_count", "flip_semantic_wrong_count", "flip_empty_count", "flip_unparseable_count",
		"c2i_rate", "i2c_rate", "flip_any_rate", "semantic_wrong_rate", "flip_semantic_wrong_rate",
		"flip_empty_rate", "flip_unparseable_rate", "c2i_pct", "i2c_pct", "flip_any_pct",
		"semantic_wrong_pct", "flip_semantic_wrong_pct", "flip_semantic_wrong_share",
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
	for target in ["max_effect", "abs_max_effect", "accuracy_gap", "abs_accuracy_gap", "flip_any_rate", "c2i_rate", "i2c_rate", "flip_semantic_wrong_rate", "semantic_wrong_rate", "flip_any_count", "c2i_count", "i2c_count", "flip_semantic_wrong_count", "semantic_wrong_count"]:
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
	targets = ["max_effect", "abs_max_effect", "accuracy_gap", "abs_accuracy_gap", "flip_any_rate", "c2i_rate", "i2c_rate", "flip_semantic_wrong_rate", "semantic_wrong_rate", "flip_any_count", "flip_semantic_wrong_count", "semantic_wrong_count"]
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


def _save_metric_plots(metric_df, delta_df, corr_df, stats_dir):
	if metric_df is None or metric_df.empty:
		return []
	from matplotlib.backends.backend_pdf import PdfPages
	paths = []
	stats_dir = Path(stats_dir)
	metric_order = [m for m in _FINAL_METRIC_ORDER if m in set(metric_df["metric"].astype(str))]
	metric_order += sorted([m for m in set(metric_df["metric"].astype(str)) if m not in metric_order])
	split_order = [s for s in ["associated", "unrelated"] if s in set(metric_df["split"].astype(str))]
	combined_pdf_path = stats_dir / "agonist_metric_final_plots.pdf"
	with PdfPages(combined_pdf_path) as pdf:
		if "mean_layer_percentile_rank" in metric_df.columns and split_order:
			box_data, labels = [], []
			for metric in metric_order:
				for split in split_order:
					vals = pd.to_numeric(
						metric_df.loc[
							(metric_df["metric"] == metric) & (metric_df["split"] == split),
							"mean_layer_percentile_rank",
						],
						errors="coerce",
					).dropna().to_numpy()
					if vals.size:
						box_data.append(vals)
						metric_label = _wrap_label(_metric_display(metric), width=14)
						labels.append(f"{metric_label}\n{split}")
			if box_data:
				fig_w = max(10.5, 1.35 * len(box_data))
				fig, ax = plt.subplots(figsize=(fig_w, 5.6))
				ax.boxplot(box_data, tick_labels=labels, showmeans=True, widths=0.55)
				for tick in ax.get_xticklabels():
					tick.set_rotation(0)
					tick.set_ha("center")
					tick.set_multialignment("center")
				ax.tick_params(axis="x", labelsize=9.4, pad=5)
				ax.set_ylim(0, 1.02)
				ax.set_ylabel("Mean within-layer percentile rank")
				ax.set_title("Agonist metric rank distributions")
				ax.grid(True, axis="y", alpha=0.3)
				fig.tight_layout(pad=0.7)
				plot_path = stats_dir / "agonist_metric_percentile_boxplot.pdf"
				fig.savefig(plot_path, bbox_inches="tight")
				pdf.savefig(fig, bbox_inches="tight")
				paths.append(str(plot_path))
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
			plot_path = stats_dir / "agonist_metric_top_rates.pdf"
			fig.savefig(plot_path, bbox_inches="tight")
			pdf.savefig(fig, bbox_inches="tight")
			paths.append(str(plot_path))
			plt.close(fig)

		if corr_df is not None and not corr_df.empty:
			target_preference = ["flip_any_rate", "flip_semantic_wrong_rate", "abs_max_effect", "c2i_rate", "i2c_rate", "accuracy_gap"]
			target = next((t for t in target_preference if t in set(corr_df["target"].astype(str))), None)
			if target is not None:
				features = ["delta_mean_metric_value", "delta_mean_abs_metric_value", "delta_mean_layer_percentile_rank", "delta_mean_layer_zscore", "delta_top10pct_rate"]
				hdf = corr_df.loc[
					(corr_df["scope"] == "associated_minus_unrelated")
					& (corr_df["target"] == target)
					& (corr_df["feature"].isin(features))
				].copy()
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
						plot_path = stats_dir / "agonist_metric_correlation_heatmap.pdf"
						fig.savefig(plot_path, bbox_inches="tight")
						pdf.savefig(fig, bbox_inches="tight")
						paths.append(str(plot_path))
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
				plot_path = stats_dir / "agonist_metric_ablation_scatter.pdf"
				fig.savefig(plot_path, bbox_inches="tight")
				pdf.savefig(fig, bbox_inches="tight")
				paths.append(str(plot_path))
				plt.close(fig)

	paths.append(str(combined_pdf_path))
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
	plot_paths = _save_metric_plots(metric_df, delta_df, corr_df, stats_dir)
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
		"targets_available": sorted([c for c in ["max_effect", "abs_max_effect", "accuracy_gap", "abs_accuracy_gap", "flip_any_rate", "c2i_rate", "i2c_rate", "flip_semantic_wrong_rate", "semantic_wrong_rate"] if c in metric_df.columns]),
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

def _iter_raw_rule_combo_csvs(rules_dir: str):
	"""Yield raw per-neuron rule_combo CSVs, never derived metric summaries.

	This is intentionally based on the raw artifacts themselves, not on any run
	configuration label, so different configs can reuse the same stats directory
	while still invalidating stale summaries when the underlying raw combo files
	change.
	"""
	root = Path(rules_dir)
	for fp in root.glob("**/rule_combo_*.csv"):
		if not fp.is_file():
			continue
		try:
			rel_probe = fp.relative_to(root)
		except Exception:
			rel_probe = fp
		if (
			str(fp.name).startswith("rule_combo_metrics_")
			or str(fp.name).startswith("rule_combo_train_test_")
			or "stats" in tuple(rel_probe.parts)
		):
			continue
		yield fp


def _sha256_file(path: Path) -> str:
	h = hashlib.sha256()
	with open(path, "rb") as f:
		for chunk in iter(lambda: f.read(1024 * 1024), b""):
			h.update(chunk)
	return h.hexdigest()


def _rule_metrics_input_manifest(rules_dir: str) -> dict:
	"""Content fingerprint of raw rule_combo inputs used by metric summaries.

	The fingerprint is deliberately independent of run/config names. It changes
	only when raw per-neuron combo CSV contents are added, removed, or edited.
	"""
	root = Path(rules_dir)
	records = []
	for fp in sorted(_iter_raw_rule_combo_csvs(root), key=lambda x: str(x.relative_to(root)) if x.is_relative_to(root) else str(x)):
		try:
			rel = fp.relative_to(root).as_posix()
		except Exception:
			rel = fp.as_posix()
		st = fp.stat()
		records.append({
			"path": rel,
			"size": int(st.st_size),
			"sha256": _sha256_file(fp),
		})
	manifest_payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
	return {
		"signature": RULE_METRICS_SUMMARY_CACHE_SIGNATURE,
		"raw_rule_combo_digest": hashlib.sha256(manifest_payload).hexdigest(),
		"n_raw_rule_combo_files": int(len(records)),
		"raw_rule_combo_files": records,
	}


def _write_rule_metrics_manifest(stats_dir: str, rules_dir: str, extra: dict = None) -> dict:
	manifest = _rule_metrics_input_manifest(rules_dir)
	if extra:
		manifest.update(extra)
	path = Path(stats_dir) / "rule_combo_metrics_manifest.json"
	path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
	print(f"[RuleMetrics] Wrote {path}")
	return manifest


def collect_rule_combo_metrics(rules_dir: str):
	"""
	Walk rules_dir and collect all rule_combo_*.csv files into a single dataframe.

	Expected layout (from this script):
	  <rules_dir>/<rule_target>/<layerKey>_<neuronId>/rule_combo_<flip_target>.csv
	"""
	rules_dir = str(rules_dir)
	root = Path(rules_dir)
	rows = []
	for fp in _iter_raw_rule_combo_csvs(root):
		# Raw per-neuron artifacts encode TRAIN, frozen TEST, TEST-selected, and
		# ALL-FIT score scopes in their filenames.
		rel = fp.relative_to(root)
		parts = rel.parts
		# Usual layout is <rule_target>/<layer>_<neuron>/rule_combo_*.csv.
		# Some debugging/export zips contain only the neuron directory itself
		# (e.g. m0_72/rule_combo_*.csv); infer that case so TRAIN and TEST
		# rows still share the same key and can be merged.
		if len(parts) >= 3:
			rule_target_dir = parts[0]
			neuron_dir = parts[-2]
		elif fp.parent == root and re.match(r"^(.*)_(\d+)$", root.name):
			rule_target_dir = root.parent.name if root.parent != root else ""
			neuron_dir = root.name
		else:
			rule_target_dir = parts[0] if parts else ""
			neuron_dir = parts[-2] if len(parts) >= 2 else ""

		m = re.match(r"^(.*)_(\d+)$", str(neuron_dir))
		layer_key = m.group(1) if m else str(neuron_dir)
		neuron_id = int(m.group(2)) if m else None

		filename_scope = "test" if (fp.parent / "global_shap_stats_train.pkl").exists() else "train_or_all"
		filename_is_train = False
		filename_is_all_fit = False
		if fp.name.startswith("rule_combo_all_fit_"):
			metric = fp.name[len("rule_combo_all_fit_"):-len(".csv")]
			filename_scope = "all_fit"
			filename_is_all_fit = True
		elif fp.name.startswith("rule_combo_test_selected_"):
			metric = fp.name[len("rule_combo_test_selected_"):-len(".csv")]
			filename_scope = "test_selected"
		elif fp.name.startswith("rule_combo_train_"):
			metric = fp.name[len("rule_combo_train_"):-len(".csv")]
			filename_scope = "train"
			filename_is_train = True
		else:
			metric = fp.name[len("rule_combo_"):-len(".csv")] if fp.name.startswith("rule_combo_") else fp.name
		df = pd.read_csv(fp)
		if df.empty:
			continue
		r = df.iloc[0].to_dict()

		# Filename scope is authoritative for TRAIN and ALL-FIT artifacts.
		if filename_is_train:
			computed_on = "train"
		elif filename_is_all_fit:
			computed_on = "all_fit"
		else:
			computed_on = str(r.get("computed_on", r.get("score_scope", filename_scope)) or filename_scope)
			_scope_norm = computed_on.strip().lower().replace("_", "+").replace("-", "+")
			if _scope_norm in ("test+selected", "heldout+selected", "held+out+selected", "hqt", "hq+t"):
				computed_on = "test_selected"
			elif _scope_norm in ("eval", "heldout", "held+out", "test"):
				computed_on = "test"
			elif _scope_norm in ("all+fit", "allfit", "final+all+fit", "finalfit"):
				computed_on = "all_fit"
			elif _scope_norm in ("all_fit", "train+eval", "all", "all+data", "traintest"):
				computed_on = "all_fit"
			elif _scope_norm.startswith("train"):
				computed_on = "train"

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
			"score_scope": computed_on,
			"P(target=1|fire)": _get_num(["P(target=1|fire)", "Precision", "precision", "P"], default=np.nan),
			"Precision": _get_num(["Precision", "P(target=1|fire)", "precision", "P"], default=np.nan),
			"R(fire|target=1)": _get_num(["R(fire|target=1)", "Recall", "recall", "TPR", "tpr"], default=np.nan),
			"F1": _get_num(["F1", "F1(target=1|fire)", "f1"]),
			"MCC": _get_num(["MCC", "mcc"]),
			"TPR": _get_num(["TPR", "tpr"]),
			"TNR": _get_num(["TNR", "tnr"]),
			"BalancedAcc": _get_num(["BalancedAcc", "balancedacc", "balanced_acc"]),
			"Acc": _get_num(["Acc", "acc", "accuracy"]),
			"dataset_coverage": _get_num(["dataset_coverage", "coverage", "rule_fire_rate"]),
			"n_eval": _get_num(["n_eval"], default=np.nan),
			"n_pos": _get_num(["n_pos"], default=np.nan),
			"n_neg": _get_num(["n_neg"], default=np.nan),
			"n_fire": _get_num(["n_fire"], default=np.nan),
			"tp": _get_num(["tp"], default=np.nan),
			"fp": _get_num(["fp"], default=np.nan),
			"tn": _get_num(["tn"], default=np.nan),
			"fn": _get_num(["fn"], default=np.nan),
			"target_prevalence": _get_num(["target_prevalence"], default=np.nan),
			"rule_fire_rate": _get_num(["rule_fire_rate"], default=np.nan),
			"expression": str(expr),
			"rule_len_atoms": int(n_atoms),
			"rule_len_disjuncts": int(n_disj),
			"rule_len_max_atoms_in_disjunct": int(max_atoms),
			"rule_combo_path": str(fp),
			"source_filename": fp.name,
			"source_scope_from_filename": filename_scope,
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
	for layer_label, neuron_id, _ in unique_everseen(neurons_sorted, key=lambda x: (x[0], x[1])):
		layer_key = safe_layer_label(layer_label)
		nk = f"{layer_key}_{int(neuron_id)}"
		keys.append(nk)
		baseline_map[nk] = "all"
	allowed = set(keys)
	rank = {k: i for i, k in enumerate(keys)}
	return allowed, rank, baseline_map


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

def _norm_rule_quality_scope(scope) -> str:
	sv = str(scope or "test").strip().lower().replace("_", "+").replace("-", "+").replace(" ", "")
	if sv in ("all+fit", "allfit", "final+all+fit", "finalfit", "all", "all+data", "train+test", "traintest", "hqf", "hq+f"):
		return "all_fit"
	if sv in ("test+selected", "heldout+selected", "held+out+selected", "hqt", "hq+t"):
		return "test_selected"
	if sv in ("eval", "heldout", "held+out", "test"):
		return "test"
	if sv.startswith("train"):
		return "train"
	return sv


def _scores_for_rule_quality_scope(scores_df, scope):
	"""Return the rows eligible for one rule-quality reporting scope."""
	if scores_df is None or scores_df.empty:
		return pd.DataFrame()
	scope_norm = _norm_rule_quality_scope(scope)
	mask = pd.Series(True, index=scores_df.index)
	if "is_test" in scores_df.columns:
		is_test = scores_df["is_test"].fillna(False).astype(bool)
		if scope_norm in ("test", "test_selected"):
			mask &= is_test
		elif scope_norm == "train":
			mask &= ~is_test
	if "_evaluated" in scores_df.columns:
		try:
			mask &= scores_df["_evaluated"].fillna(False).astype(bool)
		except Exception:
			pass
	return scores_df.loc[mask].copy()

def write_rule_metrics_stats(
	rules_dir: str,
	out_dir: str,
	stats_dirname: str = "",
	quantiles=(0.1, 0.5, 0.9),
	topk: int = 50,
	length_cap: int = 30,
	quality_metric: str = "mcc",
	quality_threshold: float = None,
	quality_thresholds_by_scope: dict = None,
	min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE,
	neurons_sorted=None,  # <-- NEW
):
	"""
	Writes paper-ready rule statistics/figures from RuleSHAP artifacts.

	Default behavior writes frozen-combo TEST summaries, TEST-selected (HQ-T)
	summaries over train-emitted rules, and, when available, parallel
	ALL-FIT descriptive summaries.  The returned dataframe is the TEST-selected
	best-per-neuron table so downstream high-quality counts use HQ-T by default.
	"""
	out_dir = str(out_dir)
	stats_dir = os.path.join(out_dir, 'stats', stats_dirname)
	Path(stats_dir).mkdir(parents=True, exist_ok=True)

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

	# ALL-FIT rows, when present, are loaded from raw rule_combo_all_fit_* files.
	# They are not reconstructed from TRAIN/TEST rows.

	n_rule_rows_before_coverage_filter = int(len(df_all))
	n_rule_rows_after_coverage_filter = int(len(_filter_by_min_dataset_coverage(df_all, min_dataset_coverage)))
	if df_all.empty:
		print("[RuleMetrics] No rule rows remain after neuron/run filtering; skipping.")
		return None

	metric_key = _norm_rule_metric_name(quality_metric)
	metric_col = _rule_metric_col(metric_key)
	metric_label = _rule_metric_label(metric_key)
	base_thr = 0.70 if quality_threshold is None else float(quality_threshold)
	thresholds_by_scope = {"test": base_thr, "test_selected": base_thr, "all_fit": 0.70}
	if isinstance(quality_thresholds_by_scope, dict):
		for _scope, _thr in quality_thresholds_by_scope.items():
			try:
				thresholds_by_scope[_norm_rule_quality_scope(_scope)] = float(_thr)
			except Exception:
				pass
	# If the requested metric isn't present in the artifacts, fall back to MCC.
	if metric_col not in df_all.columns:
		metric_key = "mcc"
		metric_col = "MCC"
		metric_label = "MCC"

	def _norm_scope(scope):
		return _norm_rule_quality_scope(scope)

	def _scope_slug(scope):
		return _norm_scope(scope).replace("+", "_").replace("-", "_")

	def _filter_scope(df, scope):
		want = _norm_scope(scope)
		if "computed_on" not in df.columns:
			return df.copy() if want == "test" else df.iloc[0:0].copy()
		vals = df["computed_on"].map(_norm_scope)
		return df[vals == want].copy()

	# Raw multi-scope dump for auditing. It can include train, test, and all_fit rows.
	all_scopes_path = os.path.join(stats_dir, "rule_combo_metrics_all_scopes.csv")
	df_all.to_csv(all_scopes_path, index=False)
	print(f"[RuleMetrics] Wrote {all_scopes_path}")
	_write_rule_metrics_manifest(stats_dir, rules_dir, extra={
		"generator": "7_refine_neuron_anchored_rules.write_rule_metrics_stats",
		"stats_dirname": str(stats_dirname or ""),
		"quality_metric": str(metric_key),
		"min_dataset_coverage_used_for_best_rows": float(min_dataset_coverage),
		"score_scopes_written": ["test", "test_selected", "all_fit"],
	})

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

	def _pct(x):
		return int(round(100.0 * float(x))) if x == x else None
	def _fmt_float(x, nd=2):
		if x != x:
			return None
		return f"{float(x):.{nd}f}"

	def _summarize_scope(scope):
		scope = _norm_scope(scope)
		slug = _scope_slug(scope)
		df_scope_raw = _filter_scope(df_all, scope)
		if df_scope_raw.empty:
			print(f"[RuleMetrics] No {scope} rule rows available; skipping {scope} summary.")
			return None

		# Write the scope-specific all-rows CSV unfiltered by coverage so downstream
		# scripts can apply a different min_dataset_coverage without regenerating
		# or deleting summaries. Coverage filtering is applied only for best/HQ rows.
		df_scope = _filter_by_min_dataset_coverage(df_scope_raw, min_dataset_coverage)
		if df_scope.empty:
			print(f"[RuleMetrics] No {scope} rule rows remain after dataset_coverage >= {float(min_dataset_coverage):.6g}; writing all-rows CSV only.")
			df_best = df_scope.copy()
		else:
			# Sort by neuron order first (if provided), then by quality within neuron.
			sort_cols = (["neuron_rank"] if "neuron_rank" in df_scope.columns else []) + [metric_col, "MCC"]
			df_best = (
				df_scope.sort_values(sort_cols, ascending=[True] + [False, False], na_position="last")
				.drop_duplicates(subset=["rule_target_dir", "neuron_key"], keep="first")
				.reset_index(drop=True)
			)

		# Keep neuron order stable in the "all" dump; metric sort as secondary.
		if "neuron_rank" in df_scope_raw.columns:
			df_all_out = df_scope_raw.sort_values(["neuron_rank", metric_col, "MCC"], ascending=[True, False, False], na_position="last")
		else:
			df_all_out = df_scope_raw.sort_values([metric_col, "MCC"], ascending=False, na_position="last")

		# Scope-specific files.
		scope_all = os.path.join(stats_dir, f"rule_combo_metrics_{slug}_all.csv")
		scope_best = os.path.join(stats_dir, f"rule_combo_metrics_{slug}_best_per_neuron.csv")
		df_all_out.to_csv(scope_all, index=False)
		df_best.to_csv(scope_best, index=False)
		print(f"[RuleMetrics] Wrote {scope_all}")
		print(f"[RuleMetrics] Wrote {scope_best}")

		if topk is not None and int(topk) > 0:
			df_top = df_best.sort_values([metric_col, "MCC"], ascending=False, na_position="last").head(int(topk)).copy()
		else:
			df_top = df_best.copy()
		top_name = f"rule_combo_metrics_{slug}_top_{int(topk) if topk else 'all'}.csv"
		csv_top = os.path.join(stats_dir, top_name)
		df_top.to_csv(csv_top, index=False)
		print(f"[RuleMetrics] Wrote {csv_top}")
		thr = float(thresholds_by_scope.get(scope, base_thr))
		high = df_best[pd.to_numeric(df_best[metric_col], errors="coerce") >= float(thr)]
		summary = {
			"score_scope": scope,
			"n_rule_combo_files": int(len(df_scope)),
			"n_rule_combo_files_all_scopes_before_coverage_filter": int(n_rule_rows_before_coverage_filter),
			"n_rule_combo_files_all_scopes_after_coverage_filter": int(n_rule_rows_after_coverage_filter),
			"min_dataset_coverage": float(min_dataset_coverage),
			"n_rule_combo_files_removed_by_coverage_filter": int(n_rule_rows_before_coverage_filter - n_rule_rows_after_coverage_filter),
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
		for extra_col in ["n_eval", "n_pos", "n_neg", "n_fire", "target_prevalence", "rule_fire_rate"]:
			if extra_col in df_best.columns:
				summary[extra_col] = _q(df_best[extra_col])

		json_path = os.path.join(stats_dir, f"rule_metrics_summary_{slug}.json")
		Path(json_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
		print(f"[RuleMetrics] Wrote {json_path}")
		tex_suffix = re.sub(r"[^0-9A-Za-z]+", "", str(stats_dirname))
		tex_suffix = tex_suffix if tex_suffix else "Run"
		scope_suffix = "" if scope == "test" else re.sub(r"[^0-9A-Za-z]+", "", slug.title())

		tex_lines = []
		tex_lines.append(f"% Auto-generated by 7_refine_neuron_anchored_rules (rule metrics, score_scope={scope})")
		tex_lines.append(f"\\newcommand\\RuleQualMetric{tex_suffix}{scope_suffix}{{{metric_label}}}")
		tex_lines.append(f"\\newcommand\\RuleQualThr{tex_suffix}{scope_suffix}{{{_fmt_float(thr,2)}}}")
		tex_lines.append(f"\\newcommand\\RuleNeurons{tex_suffix}{scope_suffix}{{{int(summary['n_neurons_with_rules'])}}}")
		tex_lines.append(f"\\newcommand\\RuleNeuronsHighQual{tex_suffix}{scope_suffix}{{{int(summary['n_neurons_quality_ge_threshold'])}}}")
		tex_lines.append(f"\\newcommand\\RuleNeuronsHighQualPct{tex_suffix}{scope_suffix}{{{_pct(summary['frac_neurons_quality_ge_threshold'])}}}")
		for key, macro in [("MCC","RuleMCC"), ("TPR","RuleTPR"), ("TNR","RuleTNR"), ("F1","RuleF1"), ("rule_len_atoms","RuleLen")]:
			st = summary.get(key, {})
			tex_lines.append(f"\\newcommand\\{macro}Med{tex_suffix}{scope_suffix}{{{_fmt_float(st.get('median', float('nan')),3)}}}")
			tex_lines.append(f"\\newcommand\\{macro}Qlo{tex_suffix}{scope_suffix}{{{_fmt_float(st.get('q_lo', float('nan')),3)}}}")
			tex_lines.append(f"\\newcommand\\{macro}Qhi{tex_suffix}{scope_suffix}{{{_fmt_float(st.get('q_hi', float('nan')),3)}}}")
		tex_path = os.path.join(stats_dir, f"rule_metrics_{slug}.tex")
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

		_hist_col("MCC", "MCC", f"Rule quality (MCC; {scope})", 1)
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
		plot_path = os.path.join(stats_dir, f"rule_metrics_distributions_{slug}.pdf")
		plt.savefig(plot_path, dpi=250, bbox_inches='tight')
		plt.close()
		print(f"[RuleMetrics] Wrote {plot_path}")
		return df_best

	_summarize_scope("test")
	df_best_test_selected = _summarize_scope("test_selected")
	_summarize_scope("all_fit")
	return df_best_test_selected

def _union_sum_eval_mask(scores_df: pd.DataFrame, cols):
	"""Return the unique flip mask, summed per-neuron count, and evaluated-row mask."""
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
	return union, int(sum_count), any_eval

def _union_and_sum_flips(scores_df: pd.DataFrame, cols):
	union, sum_count, any_eval = _union_sum_eval_mask(scores_df, cols)
	return int(union.sum()), int(sum_count), int(any_eval.sum())

def _baseline_direction_masks(scores_df: pd.DataFrame, baseline_metric_col: str):
	"""Return valid, positive, and negative masks for the unablated predicate.

	The behavioural predicate must be binary. Values numerically close to 0 and 1
	are accepted; any other finite value is rejected instead of being silently
	thresholded. This prevents continuous task scores from being used as direction
	labels by accident.
	"""
	if not baseline_metric_col or baseline_metric_col not in scores_df.columns:
		return None
	base = pd.to_numeric(scores_df[baseline_metric_col], errors="coerce")
	values = base.to_numpy(dtype=float)
	valid = np.isfinite(values)
	positive = valid & np.isclose(values, 1.0, atol=1e-8, rtol=0.0)
	negative = valid & np.isclose(values, 0.0, atol=1e-8, rtol=0.0)
	invalid_finite = valid & ~(positive | negative)
	if invalid_finite.any():
		examples = values[invalid_finite][:5].tolist()
		raise ValueError(
			f"Baseline predicate {baseline_metric_col!r} is not binary; "
			f"found values such as {examples}."
		)
	return valid, positive, negative


def _baseline_direction_denominators(scores_df: pd.DataFrame, baseline_metric_col: str, eval_mask: np.ndarray):
	"""Counts of evaluated positive/negative rows used as eligible denominators."""
	masks = _baseline_direction_masks(scores_df, baseline_metric_col)
	if masks is None:
		return None
	valid, positive, negative = masks
	eval_mask = np.asarray(eval_mask, dtype=bool)
	return {
		"baseline_metric_col": str(baseline_metric_col),
		"n_eval_baseline_valid": int((eval_mask & valid).sum()),
		"n_eval_baseline_correct": int((eval_mask & positive).sum()),
		"n_eval_baseline_incorrect": int((eval_mask & negative).sum()),
	}


def _repair_directional_flip_columns(
	scores_df: pd.DataFrame,
	neurons_sorted,
	baseline_metric_col: str,
):
	"""Rebuild directional flip columns from flip-any and the baseline predicate.

	The direction columns are deterministic derivatives:
	positive->negative = flip_any AND baseline_positive, and
	negative->positive = flip_any AND baseline_negative.
	Recomputing them removes stale directional columns left by older runs while
	preserving NA on rows where a neuron was not evaluated.
	"""
	masks = _baseline_direction_masks(scores_df, baseline_metric_col)
	if masks is None:
		raise ValueError(
			"Directional coverage requires the unablated binary predicate column."
		)
	valid, positive, negative = masks
	repaired = 0
	for layer_label, neuron_id, _ in unique_everseen(neurons_sorted, key=lambda x: (x[0], x[1])):
		layer_key = safe_layer_label(layer_label)
		col_any = f"flip_{layer_key}_{int(neuron_id)}"
		if col_any not in scores_df.columns:
			continue
		any_series = scores_df[col_any]
		evaluated = any_series.notna().to_numpy()
		if np.any(evaluated & ~valid):
			raise ValueError(
				f"{col_any} has evaluated rows without a valid baseline predicate."
			)
		flipped = any_series.fillna(False).astype(bool).to_numpy()
		c2i = pd.array([pd.NA] * len(scores_df), dtype="boolean")
		i2c = pd.array([pd.NA] * len(scores_df), dtype="boolean")
		c2i[evaluated] = (flipped & positive)[evaluated]
		i2c[evaluated] = (flipped & negative)[evaluated]
		scores_df[f"flip_c2i_{layer_key}_{int(neuron_id)}"] = c2i
		scores_df[f"flip_i2c_{layer_key}_{int(neuron_id)}"] = i2c
		repaired += 1
	return repaired


def _score_scope_slug(scope: str) -> str:
	return str(scope).strip().lower().replace("+", "_").replace("-", "_").replace(" ", "_")

def _load_scope_best_rule_metrics(
	rules_dir: str,
	stats_dirname: str,
	scope: str,
	min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE,
	quality_metric: str = "mcc",
) -> pd.DataFrame:
	"""Load best-per-neuron metrics for a score scope.

	For ALL-FIT, never trust a cached best file first: it may be stale from an
	earlier merge/fallback implementation.  Rebuild the pooled scope from raw
	cached TRAIN and TEST rule_combo CSVs when available, without re-running any
	RuleSHAP transforms or model evaluations.  Cached ALL-FIT files are used
	only as a last-resort fallback.
	"""
	stats_dir = Path(rules_dir) / "stats" / str(stats_dirname or "")
	scope_norm = _norm_rule_quality_scope(scope)
	metric_key = _norm_rule_metric_name(quality_metric)
	metric_col = _rule_metric_col(metric_key)
	if scope_norm == "all_fit":
		try:
			raw = collect_rule_combo_metrics(rules_dir)
		except Exception as exc:
			print(f"[HighQualityScopes] Could not collect raw rule_combo CSVs for ALL-FIT: {exc}")
			raw = pd.DataFrame()
		if raw is not None and not raw.empty and "computed_on" in raw.columns:
			vals = raw["computed_on"].astype(str).str.strip().str.lower().str.replace("-", "_", regex=False)
			df = raw[vals.isin(["all_fit", "allfit", "final_all_fit", "finalfit"])].copy()
			df = _filter_by_min_dataset_coverage(df, min_dataset_coverage)
			if not df.empty:
				for c in [metric_col, "MCC", "F1", "BalancedAcc"]:
					if c in df.columns:
						df[c] = pd.to_numeric(df[c], errors="coerce")
				sort_cols = [c for c in [metric_col, "MCC", "F1", "BalancedAcc"] if c in df.columns]
				if sort_cols:
					df = df.sort_values(sort_cols, ascending=[False]*len(sort_cols), na_position="last")
				key_cols = [c for c in ["rule_target_dir", "neuron_key"] if c in df.columns]
				if not key_cols:
					key_cols = [c for c in ["layer_key", "neuron_id"] if c in df.columns]
				if key_cols:
					df = df.drop_duplicates(subset=key_cols, keep="first")
				return df.reset_index(drop=True)

	# TEST can use cached TEST best files.  ALL-FIT must be rebuilt from raw
	# ALL-FIT component scopes above; if that failed, fail closed instead of
	# trusting a stale cached all_fit summary.
	if scope_norm == "all_fit":
		print("[HighQualityScopes] ALL-FIT unavailable: raw rule_combo_all_fit_* rows were not found/recoverable.")
		return pd.DataFrame()
	slug = _score_scope_slug(scope)
	candidates = [stats_dir / f"rule_combo_metrics_{slug}_best_per_neuron.csv"]
	for path in candidates:
		if path.exists():
			try:
				return pd.read_csv(path, low_memory=False)
			except Exception as exc:
				print(f"[HighQualityScopes] Could not read {path}: {exc}")
	return pd.DataFrame()


def _diagnose_scope_metric_overlap(rule_best_df_test: pd.DataFrame, rule_best_df_all_fit: pd.DataFrame, metric_col: str = "MCC"):
	"""Print a small sanity check for TEST vs ALL-FIT HQ-source metrics."""
	if rule_best_df_test is None or rule_best_df_test.empty or rule_best_df_all_fit is None or rule_best_df_all_fit.empty:
		return
	keys = [c for c in ["rule_target_dir", "neuron_key"] if c in rule_best_df_test.columns and c in rule_best_df_all_fit.columns]
	if not keys:
		keys = [c for c in ["layer_key", "neuron_id"] if c in rule_best_df_test.columns and c in rule_best_df_all_fit.columns]
	if not keys or metric_col not in rule_best_df_test.columns or metric_col not in rule_best_df_all_fit.columns:
		return
	m = rule_best_df_test[keys + [metric_col]].merge(
		rule_best_df_all_fit[keys + [metric_col]], on=keys, how="inner", suffixes=("_test", "_all_fit")
	)
	if m.empty:
		print("[HighQualityScopes] HQ-T/HQ-F metric diagnostic: no overlapping neurons.")
		return
	d = (pd.to_numeric(m[f"{metric_col}_test"], errors="coerce") - pd.to_numeric(m[f"{metric_col}_all_fit"], errors="coerce")).abs()
	print(
		f"[HighQualityScopes] HQ-T/HQ-F {metric_col} diagnostic: "
		f"overlap={len(m)}, changed={(d > 1e-12).sum()}, max_abs_diff={float(d.max(skipna=True)) if len(d) else float('nan'):.6g}"
	)

def _hq_scope_records(
	scores_df: pd.DataFrame,
	neurons_sorted,
	rule_dfs_by_scope: dict,
	quality_metric: str,
	quality_threshold: float,
	quality_thresholds_by_scope: dict = None,
	min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE,
	baseline_metric_col: str = None,
):
	"""Build all-neuron and scope-specific HQ flip-coverage records.

	The all-neuron causal coverage denominator is independent of the score scope.
	HQ-T uses TEST-selected OR-combos over train-emitted rules. HQ-F uses separate final-fit combos selected and scored on all rows. Flip coverage is evaluated on the corresponding row scope for each score scope.
	"""
	if scores_df is None:
		scores_df = pd.DataFrame()
	# Stable neuron universe.
	total_neurons = []
	for layer_label, neuron_id, _ in unique_everseen(neurons_sorted, key=lambda x: (x[0], x[1])):
		total_neurons.append((safe_layer_label(layer_label), int(neuron_id)))

	metric_key = _norm_rule_metric_name(quality_metric)
	metric_col = _rule_metric_col(metric_key)
	metric_label = _rule_metric_label(metric_key)
	base_thr = 0.70 if quality_threshold is None else float(quality_threshold)
	thresholds_by_scope = {"test": base_thr, "test_selected": base_thr, "all_fit": 0.70}
	if isinstance(quality_thresholds_by_scope, dict):
		for _scope, _thr in quality_thresholds_by_scope.items():
			try:
				thresholds_by_scope[_norm_rule_quality_scope(_scope)] = float(_thr)
			except Exception:
				pass

	def _cols(neuron_list, prefix):
		return [f"{prefix}{lk}_{nid}" for (lk, nid) in neuron_list]

	def _coverage_for(neuron_list, sdf=None):
		# Coverage is evaluated on the row scope supplied by the caller: HQ-T uses
		# TEST rows, while ALL-FIT HQ uses the pooled row universe.
		if sdf is None:
			sdf = scores_df
		cols_any = _cols(neuron_list, "flip_")
		cols_c2i = _cols(neuron_list, "flip_c2i_")
		cols_i2c = _cols(neuron_list, "flip_i2c_")
		cols_sem = _cols(neuron_list, "flip_semantic_wrong_")
		union_any_mask, sum_any, eval_any_mask = _union_sum_eval_mask(sdf, cols_any)
		union_c2i_mask, sum_c2i, _ = _union_sum_eval_mask(sdf, cols_c2i)
		union_i2c_mask, sum_i2c, _ = _union_sum_eval_mask(sdf, cols_i2c)
		union_sem_mask, sum_sem, _ = _union_sum_eval_mask(sdf, cols_sem)
		return {
			"neurons": list(neuron_list),
			"cols_any": cols_any,
			"cols_c2i": cols_c2i,
			"cols_i2c": cols_i2c,
			"union_any_mask": union_any_mask,
			"eval_any_mask": eval_any_mask,
			"union_any": int(union_any_mask.sum()),
			"sum_any": int(sum_any),
			"n_eval_any": int(eval_any_mask.sum()),
			"union_c2i": int(union_c2i_mask.sum()),
			"sum_c2i": int(sum_c2i),
			"union_i2c": int(union_i2c_mask.sum()),
			"sum_i2c": int(sum_i2c),
			"union_semantic_wrong": int(union_sem_mask.sum()),
			"sum_semantic_wrong": int(sum_sem),
		}
	all_cov = _coverage_for(total_neurons, scores_df)
	baseline_denoms = _baseline_direction_denominators(scores_df, baseline_metric_col, all_cov["eval_any_mask"])
	idx_map = {t: i for i, t in enumerate(total_neurons)}

	def _bool_mat(cols):
		sub = scores_df.reindex(columns=cols)
		sub = sub.apply(pd.to_numeric, errors="coerce").fillna(0.0)
		return (sub.to_numpy() != 0.0)

	def _perm_for(high_neurons, cols, seed):
		idx_high = [idx_map[t] for t in high_neurons if t in idx_map]
		if not idx_high:
			return float("nan")
		try:
			return _perm_p_union(_bool_mat(cols), idx_high, n_perm=1000, seed=seed)
		except Exception:
			return float("nan")

	scopes = {}
	for scope, rdf in (rule_dfs_by_scope or {}).items():
		scope_norm = _norm_rule_quality_scope(scope)
		if rdf is None or rdf.empty:
			continue
		rdf2 = _filter_rule_metrics_df(rdf.copy(), neurons_sorted)
		rdf2 = _filter_by_min_dataset_coverage(rdf2, min_dataset_coverage)
		if rdf2 is None or rdf2.empty:
			continue
		mcol = metric_col if metric_col in rdf2.columns else "MCC"
		if mcol not in rdf2.columns:
			continue
		tmp = rdf2.copy()
		tmp["layer_key"] = tmp["layer_key"].map(_safe_layer_label)
		scope_thr = float(thresholds_by_scope.get(scope_norm, base_thr))
		high_df = tmp[pd.to_numeric(tmp[mcol], errors="coerce") >= scope_thr]
		high_ids = {(str(r["layer_key"]), int(r["neuron_id"])) for _, r in high_df.dropna(subset=["neuron_id", "layer_key"]).iterrows()}
		high_neurons = [t for t in total_neurons if (t[0], t[1]) in high_ids]
		scope_scores = _scores_for_rule_quality_scope(scores_df, scope_norm)
		scope_all_cov = _coverage_for(total_neurons, scope_scores)
		scope_denoms = _baseline_direction_denominators(scope_scores, baseline_metric_col, scope_all_cov["eval_any_mask"])
		cov = _coverage_for(high_neurons, scope_scores)
		cov.update({
			"score_scope": scope_norm,
			"score_scope_label": "HQ-T" if scope_norm == "test_selected" else ("LEGACY TEST" if scope_norm == "test" else "HQ-F"),
			"quality_threshold": float(scope_thr),
			"n_neurons_high_quality": int(len(high_neurons)),
			"frac_neurons_high_quality": (float(len(high_neurons)) / float(len(total_neurons))) if total_neurons else float("nan"),
			"scope_n_rows": int(len(scope_scores)),
			"scope_all": {k: int(scope_all_cov.get(k, 0) or 0) for k in ["union_any", "sum_any", "n_eval_any", "union_c2i", "sum_c2i", "union_i2c", "sum_i2c", "union_semantic_wrong", "sum_semantic_wrong"]},
			"eligible_denominators": scope_denoms,
			"p_union_any": _perm_for(high_neurons, scope_all_cov["cols_any"], 0 if scope_norm == "test" else 10),
			"p_union_c2i": _perm_for(high_neurons, scope_all_cov["cols_c2i"], 1 if scope_norm == "test" else 11),
			"p_union_i2c": _perm_for(high_neurons, scope_all_cov["cols_i2c"], 2 if scope_norm == "test" else 12),
		})
		scopes[scope_norm] = cov

	return {
		"quality_metric": str(metric_key),
		"quality_metric_label": str(metric_label),
		"quality_threshold": float(thresholds_by_scope.get("test_selected", thresholds_by_scope.get("test", base_thr))),
		"quality_thresholds_by_scope": {k: float(v) for k, v in thresholds_by_scope.items()},
		"min_dataset_coverage": float(min_dataset_coverage),
		"total_neurons": total_neurons,
		"all": all_cov,
		"baseline_denominators": baseline_denoms,
		"scopes": scopes,
	}

def _write_high_quality_scope_tex(scope_rec: dict, out_dir: str, stats_dirname: str):
	def _pct(x):
		return int(round(100.0 * float(x))) if x == x else None
	def _fmtf(x, nd=2):
		return f"{float(x):.{nd}f}" if x == x else ""
	def _fmt_p_local(p):
		if p != p:
			return ""
		if p < 1e-4:
			return "<1e-4"
		if p < 0.01:
			return f"{p:.2g}"
		return f"{p:.2f}"

	stats_dir = Path(out_dir) / "stats" / str(stats_dirname or "")
	tex_suffix = re.sub(r"[^0-9A-Za-z]+", "", str(stats_dirname)) or "Run"
	scopes = scope_rec.get("scopes", {})
	test = scopes.get("test_selected", scopes.get("test", {}))
	all_fit = scopes.get("all_fit", {})
	all_cov = scope_rec.get("all", {})
	lines = []
	lines.append("% Auto-generated by 7_refine_neuron_anchored_rules (high-quality flip coverage, score scopes)")
	lines.append(f"\\newcommand\\HighQualMetric{tex_suffix}{{{scope_rec.get('quality_metric_label', 'MCC')}}}")
	lines.append(f"\\newcommand\\HighQualThr{tex_suffix}{{{_fmtf((scope_rec.get('quality_thresholds_by_scope') or {}).get('test_selected', (scope_rec.get('quality_thresholds_by_scope') or {}).get('test', scope_rec.get('quality_threshold', float('nan')))),2)}}}")
	lines.append(f"\\newcommand\\HighQualThrAllFit{tex_suffix}{{{_fmtf((scope_rec.get('quality_thresholds_by_scope') or {}).get('all_fit', float('nan')),2)}}}")
	lines.append(f"\\newcommand\\HighQualNeurons{tex_suffix}{{{int(test.get('n_neurons_high_quality', 0) or 0)}}}")
	lines.append(f"\\newcommand\\HighQualNeuronsPct{tex_suffix}{{{_pct(test.get('frac_neurons_high_quality', float('nan')))}}}")
	lines.append(f"\\newcommand\\HighQualNeuronsAllFit{tex_suffix}{{{int(all_fit.get('n_neurons_high_quality', 0) or 0)}}}")
	lines.append(f"\\newcommand\\HighQualNeuronsPctAllFit{tex_suffix}{{{_pct(all_fit.get('frac_neurons_high_quality', float('nan')))}}}")
	lines.append(f"\\newcommand\\HighQualityUnionFlipAny{tex_suffix}{{{int(test.get('union_any', 0) or 0)}}}")
	lines.append(f"\\newcommand\\HighQualityUnionFlipAnyAllFit{tex_suffix}{{{int(all_fit.get('union_any', 0) or 0)}}}")
	all_any = float(all_cov.get('union_any', 0) or 0)
	lines.append(f"\\newcommand\\HighQualityUnionFlipAnyPctAll{tex_suffix}{{{_pct((float(test.get('union_any', 0) or 0) / all_any) if all_any else float('nan'))}}}")
	lines.append(f"\\newcommand\\HighQualityUnionFlipAnyPctAllAllFit{tex_suffix}{{{_pct((float(all_fit.get('union_any', 0) or 0) / all_any) if all_any else float('nan'))}}}")
	lines.append(f"\\newcommand\\HighQualityPUnionAny{tex_suffix}{{{_fmt_p_local(test.get('p_union_any', float('nan')))}}}")
	lines.append(f"\\newcommand\\HighQualityPUnionAnyAllFit{tex_suffix}{{{_fmt_p_local(all_fit.get('p_union_any', float('nan')))}}}")
	tex_path = stats_dir / "high_quality_neuron_flip_coverage.tex"
	tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
	print(f"[HighQualityScopes] Wrote {tex_path}")

def write_high_quality_neuron_flip_coverage_with_scopes(
	scores_df: pd.DataFrame,
	neurons_sorted,
	rule_best_df_test_selected: pd.DataFrame,
	rule_best_df_all_fit: pd.DataFrame,
	out_dir: str,
	stats_dirname: str = "",
	quality_metric: str = "mcc",
	quality_threshold: float = None,
	quality_thresholds_by_scope: dict = None,
	min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE,
	baseline_metric_col: str = None,
):
	"""Write high-quality coverage JSON/PDF with HQ-T and HQ-F subsets.

	Legacy filenames are preserved. If ALL-FIT metrics are unavailable, the plot
	falls back to HQ-T-only while still writing a score-scoped JSON structure.
	"""
	if rule_best_df_test_selected is None or rule_best_df_test_selected.empty:
		print("[HighQualityScopes] No TEST-selected rule metrics available; skipping scoped HQ coverage.")
		return None
	data = _hq_scope_records(
		scores_df=scores_df,
		neurons_sorted=neurons_sorted,
		rule_dfs_by_scope={"test_selected": rule_best_df_test_selected, "all_fit": rule_best_df_all_fit},
		quality_metric=quality_metric,
		quality_threshold=quality_threshold,
		quality_thresholds_by_scope=quality_thresholds_by_scope,
		min_dataset_coverage=min_dataset_coverage,
		baseline_metric_col=baseline_metric_col,
	)
	out_dir = str(out_dir)
	stats_dir = Path(out_dir) / "stats" / str(stats_dirname or "")
	stats_dir.mkdir(parents=True, exist_ok=True)
	all_cov = data["all"]
	scopes = data["scopes"]
	baseline_denoms = data.get("baseline_denominators")
	metric_label = data.get("quality_metric_label", "MCC")
	thr = float(data.get("quality_threshold", 0.70))
	total_neurons = data.get("total_neurons", [])
	# Legacy top-level fields now mirror the primary HQ-T semantics. New score_scopes carries all scopes.
	test = scopes.get("test_selected", scopes.get("test", {}))
	test_all = test.get("scope_all", {}) if isinstance(test.get("scope_all", {}), dict) else {}
	test_denoms = test.get("eligible_denominators") if test.get("eligible_denominators") is not None else baseline_denoms
	summary = {
		"quality_metric": data.get("quality_metric"),
		"quality_metric_label": metric_label,
		"quality_threshold": thr,
		"quality_thresholds_by_scope": data.get("quality_thresholds_by_scope", {"test": thr}),
		"min_dataset_coverage": data.get("min_dataset_coverage"),
		"primary_score_scope": "test_selected",
		"available_score_scopes": sorted(scopes.keys()),
		"n_neurons_total": int(len(total_neurons)),
		"n_neurons_high_quality": int(test.get("n_neurons_high_quality", 0) or 0),
		"frac_neurons_high_quality": float(test.get("frac_neurons_high_quality", float("nan"))),
		"baseline_metric_col": str(baseline_metric_col) if baseline_metric_col else None,
		"eligible_denominators": test_denoms if test_denoms is not None else {
			"baseline_metric_col": None,
			"n_eval_baseline_valid": None,
			"n_eval_baseline_correct": None,
			"n_eval_baseline_incorrect": None,
		},
		"flip_any": {
			"union_high": int(test.get("union_any", 0) or 0),
			"sum_high": int(test.get("sum_any", 0) or 0),
			"union_all": int(test_all.get("union_any", all_cov.get("union_any", 0)) or 0),
			"sum_all": int(test_all.get("sum_any", all_cov.get("sum_any", 0)) or 0),
			"union_share_of_all": (float(test.get("union_any", 0) or 0) / float(test_all.get("union_any", all_cov.get("union_any", 0)))) if test_all.get("union_any", all_cov.get("union_any", 0)) else float("nan"),
			"n_rows_with_any_eval_all": int(test_all.get("n_eval_any", all_cov.get("n_eval_any", 0)) or 0),
			"n_rows_with_any_eval_high": int(test.get("n_eval_any", 0) or 0),
		},
		"score_scopes": {},
		"figure_semantics": {
			"hq_t": "High-quality neurons whose TEST-selected OR-combo over train-emitted rules reaches the threshold on held-out TEST rows; flip prevalence is also evaluated on TEST rows.",
			"hq_f": "High-quality neurons whose descriptive final-fit combo is selected and scored on all rows; flip prevalence is evaluated on all rows. This is descriptive and not a held-out generalization metric.",
			"unique_union": "A prompt flipped by several neurons is counted once.",
		},
	}
	for scope, rec in scopes.items():
		scope_all = rec.get("scope_all", {}) if isinstance(rec.get("scope_all", {}), dict) else {}
		def _ratio_payload(num, den):
			return {"numerator": int(num or 0), "denominator": int(den or 0), "percent": (100.0 * float(num or 0) / float(den or 0)) if float(den or 0) else float("nan")}
		summary["score_scopes"][scope] = {
			"quality_threshold": float(rec.get("quality_threshold", thr)),
			"n_neurons_high_quality": int(rec.get("n_neurons_high_quality", 0) or 0),
			"frac_neurons_high_quality": float(rec.get("frac_neurons_high_quality", float("nan"))),
			"scope_n_rows": int(rec.get("scope_n_rows", 0) or 0),
			"eligible_denominators": rec.get("eligible_denominators"),
			"all_rule_bearing_in_scope": scope_all,
			"flip_any": {"union_high": int(rec.get("union_any", 0) or 0), "sum_high": int(rec.get("sum_any", 0) or 0), "union_all_in_scope": int(scope_all.get("union_any", 0) or 0)},
			"flip_c2i": {"union_high": int(rec.get("union_c2i", 0) or 0), "sum_high": int(rec.get("sum_c2i", 0) or 0), "union_all_in_scope": int(scope_all.get("union_c2i", 0) or 0)},
			"flip_i2c": {"union_high": int(rec.get("union_i2c", 0) or 0), "sum_high": int(rec.get("sum_i2c", 0) or 0), "union_all_in_scope": int(scope_all.get("union_i2c", 0) or 0)},
			"ratios": {
				"prevalence_any": _ratio_payload(rec.get("union_any", 0), (rec.get("eligible_denominators") or {}).get("n_eval_baseline_valid", rec.get("n_eval_any", 0))),
				"prevalence_c2i": _ratio_payload(rec.get("union_c2i", 0), (rec.get("eligible_denominators") or {}).get("n_eval_baseline_correct", rec.get("n_eval_any", 0))),
				"prevalence_i2c": _ratio_payload(rec.get("union_i2c", 0), (rec.get("eligible_denominators") or {}).get("n_eval_baseline_incorrect", rec.get("n_eval_any", 0))),
				"coverage_share_any": _ratio_payload(rec.get("union_any", 0), scope_all.get("union_any", 0)),
				"coverage_share_c2i": _ratio_payload(rec.get("union_c2i", 0), scope_all.get("union_c2i", 0)),
				"coverage_share_i2c": _ratio_payload(rec.get("union_i2c", 0), scope_all.get("union_i2c", 0)),
				"overlap_any": _ratio_payload(rec.get("sum_any", 0), rec.get("union_any", 0)),
				"overlap_c2i": _ratio_payload(rec.get("sum_c2i", 0), rec.get("union_c2i", 0)),
				"overlap_i2c": _ratio_payload(rec.get("sum_i2c", 0), rec.get("union_i2c", 0)),
			},
			"p_values": {"perm_union_any": float(rec.get("p_union_any", float("nan"))), "perm_union_c2i": float(rec.get("p_union_c2i", float("nan"))), "perm_union_i2c": float(rec.get("p_union_i2c", float("nan")))},
		}
	if "test_selected" in scopes and "all_fit" in scopes:
		test_ids = {tuple(x) for x in scopes["test_selected"].get("neurons", [])}
		all_fit_ids = {tuple(x) for x in scopes["all_fit"].get("neurons", [])}
		summary["scope_pair_diagnostics"] = {
			"n_hq_both": int(len(test_ids & all_fit_ids)),
			"n_hq_t_only": int(len(test_ids - all_fit_ids)),
			"n_hq_f_only": int(len(all_fit_ids - test_ids)),
			"same_hq_neuron_set": bool(test_ids == all_fit_ids),
		}

	json_path = stats_dir / "high_quality_neuron_flip_coverage.json"
	json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
	print(f"[HighQualityScopes] Wrote {json_path}")
	_write_high_quality_scope_tex(data, out_dir, stats_dirname)

	def _fmt_p(p):
		if p != p:
			return ""
		if p < 1e-4:
			return "<1e-4"
		if p < 0.01:
			return f"{p:.2g}"
		return f"{p:.2f}"
	def _rate(num, den):
		return (100.0 * float(num) / float(den)) if den else float("nan")
	def _ratio(num, den):
		return (float(num) / float(den)) if den else float("nan")
	def _fmt_ratio_count(num, den):
		try:
			n = int(round(float(num)))
			d = int(round(float(den)))
		except Exception:
			return ""
		return f"{n}/{d}" if d else f"{n}/0"
	def _scope_label(scope):
		return "HQ-T" if scope == "test_selected" else ("Frozen TEST" if scope == "test" else "HQ-F")

	scope_order = [s for s in ["test_selected", "all_fit"] if s in scopes]
	def _denoms_for(cov_rec, denom_rec):
		eval_denom_local = float(cov_rec.get("n_eval_any", 0) or 0)
		has_local = denom_rec is not None
		return np.asarray([
			eval_denom_local,
			float(denom_rec.get("n_eval_baseline_correct")) if has_local and denom_rec.get("n_eval_baseline_correct") is not None else eval_denom_local,
			float(denom_rec.get("n_eval_baseline_incorrect")) if has_local and denom_rec.get("n_eval_baseline_incorrect") is not None else eval_denom_local,
		], dtype=float)
	dir_tick_labels = ["Any flip", "Correct→\nincorrect", "Incorrect→\ncorrect"]
	x = np.arange(3)

	with plt.rc_context({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 6.8}):
		fig, axes = plt.subplots(2, 2, figsize=(9.1, 6.2), constrained_layout=False)
		fig.subplots_adjust(left=0.075, right=0.995, bottom=0.12, top=0.955, wspace=0.32, hspace=0.44)
		ax_counts, ax_prev = axes[0]
		ax_cov, ax_overlap = axes[1]
		colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [None, None, None])
		# Counts.
		count_labels = ["All\nrule-bearing"] + [_scope_label(s).replace(" ", "\n") + f"\n({metric_label}≥{float(scopes[s].get('quality_threshold', thr)):.2f})" for s in scope_order]
		counts = [len(total_neurons)] + [int(scopes[s].get("n_neurons_high_quality", 0) or 0) for s in scope_order]
		bars = ax_counts.bar(np.arange(len(counts)), counts, width=0.62, color=colors[:len(counts)])
		ax_counts.set_xticks(np.arange(len(counts)))
		ax_counts.set_xticklabels(count_labels)
		ax_counts.set_ylabel("# neurons")
		ax_counts.set_title("Subset size", pad=4)
		ax_counts.set_ylim(0, max(1.0, max(counts) * 1.25 if counts else 1.0))
		ax_counts.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
		for i, b in enumerate(bars):
			h = b.get_height()
			pct = (100.0 * h / float(len(total_neurons))) if i > 0 and total_neurons else float("nan")
			lab = f"{int(h)}" + (f"\n({pct:.1f}%)" if i > 0 and pct == pct else "")
			ax_counts.annotate(lab, (b.get_x() + b.get_width()/2, h), ha="center", va="bottom", fontsize=7.0, xytext=(0,2), textcoords="offset points")

		# Prevalence. Each scope is evaluated on its own row universe:
		# All = pooled rows; HQ TEST = held-out TEST rows; HQ ALL-FIT = pooled rows.
		all_union = np.asarray([all_cov.get("union_any", 0), all_cov.get("union_c2i", 0), all_cov.get("union_i2c", 0)], dtype=float)
		series = [("All", all_union, _denoms_for(all_cov, baseline_denoms), colors[0])]
		for idx, scope in enumerate(scope_order, start=1):
			rec = scopes[scope]
			rec_denoms = _denoms_for(rec, rec.get("eligible_denominators"))
			series.append((_scope_label(scope), np.asarray([rec.get("union_any", 0), rec.get("union_c2i", 0), rec.get("union_i2c", 0)], dtype=float), rec_denoms, colors[idx]))
		w = min(0.24, 0.78 / max(1, len(series)))
		for idx, (lab, vals, local_denoms, color) in enumerate(series):
			off = (idx - (len(series)-1)/2) * w
			pct = np.divide(100.0 * vals, local_denoms, out=np.full(3, np.nan), where=local_denoms > 0)
			bars_s = ax_prev.bar(x + off, pct, width=w, label=lab, color=color)
			for b, cnt, den in zip(bars_s, vals, local_denoms):
				if b.get_height() == b.get_height():
					label = f"{b.get_height():.1f}%\n{_fmt_ratio_count(cnt, den)}"
					ax_prev.annotate(label, (b.get_x()+b.get_width()/2, b.get_height()), ha="center", va="bottom", fontsize=5.2, xytext=(0,1), textcoords="offset points", rotation=0, linespacing=0.9)
		ax_prev.set_xticks(x)
		ax_prev.set_xticklabels(dir_tick_labels)
		ax_prev.set_ylabel("Unique flips / eligible points (%)")
		ax_prev.set_title("Eligible flip prevalence", pad=4)
		prev_vals = []
		for _, vals, local_denoms, _ in series:
			prev_vals.extend(list(np.divide(100.0 * vals, local_denoms, out=np.full(3, np.nan), where=local_denoms > 0)))
		prev_top = np.nanmax(prev_vals) if len(prev_vals) and np.isfinite(prev_vals).any() else 1.0
		ax_prev.set_ylim(0, max(1.0, float(prev_top) * 1.34))
		ax_prev.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
		ax_prev.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.0, borderpad=0.1)

		# Coverage share. Compare each HQ scope against all rule-bearing neurons
		# evaluated on the same row scope.
		cov_series = []
		for idx, scope in enumerate(scope_order, start=1):
			rec = scopes[scope]
			scope_all = rec.get("scope_all", {}) if isinstance(rec.get("scope_all", {}), dict) else all_cov
			vals = np.asarray([
				_ratio(rec.get("union_any", 0), scope_all.get("union_any", 0)),
				_ratio(rec.get("union_c2i", 0), scope_all.get("union_c2i", 0)),
				_ratio(rec.get("union_i2c", 0), scope_all.get("union_i2c", 0)),
			], dtype=float) * 100.0
			cov_counts = np.asarray([
				[rec.get("union_any", 0), scope_all.get("union_any", 0)],
				[rec.get("union_c2i", 0), scope_all.get("union_c2i", 0)],
				[rec.get("union_i2c", 0), scope_all.get("union_i2c", 0)],
			], dtype=float)
			cov_series.append((scope, vals, cov_counts, colors[idx]))
		w2 = min(0.30, 0.70 / max(1, len(cov_series)))
		for idx, (scope, vals, cov_counts, color) in enumerate(cov_series):
			off = (idx - (len(cov_series)-1)/2) * w2
			bars_s = ax_cov.bar(x + off, vals, width=w2, label=_scope_label(scope), color=color)
			for b, pct, pair in zip(bars_s, vals, cov_counts):
				if pct == pct:
					label = f"{pct:.0f}%\n{_fmt_ratio_count(pair[0], pair[1])}"
					ax_cov.annotate(label, (b.get_x()+b.get_width()/2, b.get_height()), ha="center", va="bottom", fontsize=5.4, xytext=(0,1), textcoords="offset points", linespacing=0.9)
		ax_cov.set_xticks(x)
		ax_cov.set_xticklabels(dir_tick_labels)
		ax_cov.set_ylabel("HQ / all union (%)")
		ax_cov.set_title("Coverage share", pad=4)
		cov_top = np.nanmax([v for _, vals, _, _ in cov_series for v in vals]) if cov_series else 100.0
		ax_cov.set_ylim(0, max(100.0, float(cov_top) * 1.12 if cov_top == cov_top else 100.0))
		ax_cov.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
		ax_cov.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.0, borderpad=0.1)

		# Overlap / redundancy.
		overlap_series = [(
			"All",
			np.asarray([_ratio(all_cov.get("sum_any", 0), all_cov.get("union_any", 0)), _ratio(all_cov.get("sum_c2i", 0), all_cov.get("union_c2i", 0)), _ratio(all_cov.get("sum_i2c", 0), all_cov.get("union_i2c", 0))], dtype=float),
			np.asarray([[all_cov.get("sum_any", 0), all_cov.get("union_any", 0)], [all_cov.get("sum_c2i", 0), all_cov.get("union_c2i", 0)], [all_cov.get("sum_i2c", 0), all_cov.get("union_i2c", 0)]], dtype=float),
			colors[0],
		)]
		for idx, scope in enumerate(scope_order, start=1):
			rec = scopes[scope]
			overlap_vals = np.asarray([_ratio(rec.get("sum_any", 0), rec.get("union_any", 0)), _ratio(rec.get("sum_c2i", 0), rec.get("union_c2i", 0)), _ratio(rec.get("sum_i2c", 0), rec.get("union_i2c", 0))], dtype=float)
			overlap_counts = np.asarray([[rec.get("sum_any", 0), rec.get("union_any", 0)], [rec.get("sum_c2i", 0), rec.get("union_c2i", 0)], [rec.get("sum_i2c", 0), rec.get("union_i2c", 0)]], dtype=float)
			overlap_series.append((_scope_label(scope), overlap_vals, overlap_counts, colors[idx]))
		w3 = min(0.24, 0.78 / max(1, len(overlap_series)))
		for idx, (lab, vals, counts_ratio, color) in enumerate(overlap_series):
			off = (idx - (len(overlap_series)-1)/2) * w3
			bars_s = ax_overlap.bar(x + off, vals, width=w3, label=lab, color=color)
			for b, val, pair in zip(bars_s, vals, counts_ratio):
				if val == val:
					label = f"{val:.1f}\n{_fmt_ratio_count(pair[0], pair[1])}"
					ax_overlap.annotate(label, (b.get_x()+b.get_width()/2, b.get_height()), ha="center", va="bottom", fontsize=5.0, xytext=(0,1), textcoords="offset points", linespacing=0.85)
		ax_overlap.axhline(1.0, linewidth=0.8, alpha=0.45)
		ax_overlap.set_xticks(x)
		ax_overlap.set_xticklabels(dir_tick_labels)
		ax_overlap.set_ylabel("Sum / union")
		ax_overlap.set_title("Overlap", pad=4)
		overlap_vals = [v for _, vals, _, _ in overlap_series for v in vals if v == v]
		ax_overlap.set_ylim(0, max(1.0, max(overlap_vals) * 1.25 if overlap_vals else 1.0))
		ax_overlap.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
		ax_overlap.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.0, borderpad=0.1)
		for ax in [ax_counts, ax_prev, ax_cov, ax_overlap]:
			ax.spines["top"].set_visible(False)
			ax.spines["right"].set_visible(False)
		plot_path = stats_dir / "high_quality_neuron_flip_coverage.pdf"
		fig.savefig(plot_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
		plt.close(fig)
		print(f"[HighQualityScopes] Wrote {plot_path}")
	return summary

def write_high_quality_neuron_flip_coverage_by_layer_with_scopes(
	scores_df: pd.DataFrame,
	neurons_sorted,
	rule_best_df_test_selected: pd.DataFrame,
	rule_best_df_all_fit: pd.DataFrame,
	out_dir: str,
	stats_dirname: str = "",
	quality_metric: str = "mcc",
	quality_threshold: float = None,
	quality_thresholds_by_scope: dict = None,
	min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE,
):
	if rule_best_df_test_selected is None or rule_best_df_test_selected.empty:
		print("[HighQualityByLayerScopes] No TEST-selected rule metrics available; skipping.")
		return None
	data = _hq_scope_records(
		scores_df=scores_df,
		neurons_sorted=neurons_sorted,
		rule_dfs_by_scope={"test_selected": rule_best_df_test_selected, "all_fit": rule_best_df_all_fit},
		quality_metric=quality_metric,
		quality_threshold=quality_threshold,
		quality_thresholds_by_scope=quality_thresholds_by_scope,
		min_dataset_coverage=min_dataset_coverage,
		baseline_metric_col=None,
	)
	out_dir = str(out_dir)
	stats_dir = Path(out_dir) / "stats" / str(stats_dirname or "")
	stats_dir.mkdir(parents=True, exist_ok=True)
	scopes = data.get("scopes", {})
	metric_label = data.get("quality_metric_label", "MCC")
	thr = float(data.get("quality_threshold", 0.70))
	# High ids by scope.
	high_ids_by_scope = {scope: set(rec.get("neurons", [])) for scope, rec in scopes.items()}
	layer_to_neurons = defaultdict(list)
	for layer_label, neuron_id, _ in unique_everseen(neurons_sorted, key=lambda x: (x[0], x[1])):
		layer_to_neurons[safe_layer_label(layer_label)].append(int(neuron_id))
	layers = sorted(layer_to_neurons.keys(), key=lambda lk: _layer_sort_key(lk))
	def _cols(neuron_list, prefix):
		return [f"{prefix}{lkey}_{nid}" for (lkey, nid) in neuron_list]
	rows = []
	for lk in layers:
		nids = sorted(set(layer_to_neurons[lk]))
		all_neus = [(lk, nid) for nid in nids]
		cols_c2i_all = _cols(all_neus, "flip_c2i_")
		cols_i2c_all = _cols(all_neus, "flip_i2c_")
		cols_any_all = _cols(all_neus, "flip_")
		union_any_all, _, n_eval_any_all = _union_and_sum_flips(scores_df, cols_any_all)
		union_c2i_all, _, _ = _union_and_sum_flips(scores_df, cols_c2i_all)
		union_i2c_all, _, _ = _union_and_sum_flips(scores_df, cols_i2c_all)
		row = {
			"layer_key": lk,
			"layer_sort": int(_layer_sort_key(lk)),
			"n_neurons_total": int(len(all_neus)),
			"n_rows_with_any_eval_all": int(n_eval_any_all),
			"flip_any_union_all": int(union_any_all),
			"flip_c2i_union_all": int(union_c2i_all),
			"flip_i2c_union_all": int(union_i2c_all),
		}
		for scope in ["test_selected", "all_fit"]:
			slug = _score_scope_slug(scope)
			sdf_scope = _scores_for_rule_quality_scope(scores_df, scope)
			hq_neus = [t for t in all_neus if t in high_ids_by_scope.get(scope, set())]
			cols_any_hq = _cols(hq_neus, "flip_")
			cols_c2i_hq = _cols(hq_neus, "flip_c2i_")
			cols_i2c_hq = _cols(hq_neus, "flip_i2c_")
			union_any_hq, _, n_eval_any_hq = _union_and_sum_flips(sdf_scope, cols_any_hq)
			union_c2i_hq, _, _ = _union_and_sum_flips(sdf_scope, cols_c2i_hq)
			union_i2c_hq, _, _ = _union_and_sum_flips(sdf_scope, cols_i2c_hq)
			row[f"n_neurons_high_quality_{slug}"] = int(len(hq_neus))
			row[f"n_rows_with_any_eval_high_{slug}"] = int(n_eval_any_hq)
			row[f"flip_any_union_high_{slug}"] = int(union_any_hq)
			row[f"flip_c2i_union_high_{slug}"] = int(union_c2i_hq)
			row[f"flip_i2c_union_high_{slug}"] = int(union_i2c_hq)
		# Legacy top-level fields mirror the primary HQ-T scope.
		row["n_neurons_high_quality"] = row.get("n_neurons_high_quality_test_selected", row.get("n_neurons_high_quality_test", 0))
		row["n_rows_with_any_eval_high"] = row.get("n_rows_with_any_eval_high_test_selected", row.get("n_rows_with_any_eval_high_test", 0))
		row["flip_any_union_high"] = row.get("flip_any_union_high_test_selected", row.get("flip_any_union_high_test", 0))
		row["flip_c2i_union_high"] = row.get("flip_c2i_union_high_test_selected", row.get("flip_c2i_union_high_test", 0))
		row["flip_i2c_union_high"] = row.get("flip_i2c_union_high_test_selected", row.get("flip_i2c_union_high_test", 0))
		rows.append(row)
	out = {
		"quality_metric": data.get("quality_metric"),
		"quality_metric_label": metric_label,
		"quality_threshold": thr,
		"quality_thresholds_by_scope": data.get("quality_thresholds_by_scope", {"test": thr}),
		"min_dataset_coverage": data.get("min_dataset_coverage"),
		"primary_score_scope": "test_selected",
		"available_score_scopes": sorted(scopes.keys()),
		"layers": rows,
	}
	json_path = stats_dir / "high_quality_neuron_flip_coverage_by_layer.json"
	json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
	print(f"[HighQualityByLayerScopes] Wrote {json_path}")

	df_plot = pd.DataFrame(rows).sort_values("layer_sort")
	if df_plot.empty:
		return out
	layer_labels = [str(x).replace("_", ".") for x in df_plot["layer_key"].tolist()]
	x = np.arange(len(layer_labels))
	denom_all = max(1, len(scores_df))
	denom_by_scope = {"test_selected": max(1, len(_scores_for_rule_quality_scope(scores_df, "test_selected"))), "all_fit": max(1, len(_scores_for_rule_quality_scope(scores_df, "all_fit")))}
	colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [None, None, None, None, None, None])
	plt.figure(figsize=(max(10, 0.65 * len(layer_labels)), 6.8))
	ax1 = plt.subplot(2, 1, 1)
	w = 0.25
	ax1.bar(x - w, df_plot["n_neurons_total"].to_numpy(), w, label="All neurons w/ rules", color=colors[0])
	thr_by_scope = data.get("quality_thresholds_by_scope", {"test_selected": thr, "all_fit": thr})
	thr_test = float(thr_by_scope.get("test_selected", thr_by_scope.get("test", thr)))
	thr_all_fit = float(thr_by_scope.get("all_fit", thr))
	ax1.bar(x, df_plot.get("n_neurons_high_quality_test_selected", pd.Series([0]*len(df_plot))).to_numpy(), w, label=f"HQ-T ({metric_label} ≥ {thr_test:.2f})", color=colors[1])
	ax1.bar(x + w, df_plot.get("n_neurons_high_quality_all_fit", pd.Series([0]*len(df_plot))).to_numpy(), w, label=f"HQ-F ({metric_label} ≥ {thr_all_fit:.2f})", color=colors[2])
	ax1.set_ylabel("# neurons")
	ax1.set_xticks(x)
	ax1.set_xticklabels(layer_labels, rotation=45, ha="right")
	ax1.tick_params(axis="x", labelbottom=False)
	ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
	ax1.legend(loc="best", fontsize=8, ncol=3)
	ax2 = plt.subplot(2, 1, 2)
	c2i_all = 100.0 * df_plot["flip_c2i_union_all"].to_numpy(dtype=float) / denom_all
	i2c_all = 100.0 * df_plot["flip_i2c_union_all"].to_numpy(dtype=float) / denom_all
	ax2.plot(x, c2i_all, marker="o", linewidth=1.2, label="c→i all", color=colors[0])
	ax2.plot(x, i2c_all, marker="s", linewidth=1.2, label="i→c all", color=colors[3] if len(colors)>3 else None)
	for scope, marker, ls, ci in [("test_selected", "o", "--", 1), ("all_fit", "o", ":", 2)]:
		label_scope = "HQ-T" if scope == "test_selected" else "HQ-F"
		c = df_plot.get(f"flip_c2i_union_high_{scope}", pd.Series([0]*len(df_plot))).to_numpy(dtype=float)
		i = df_plot.get(f"flip_i2c_union_high_{scope}", pd.Series([0]*len(df_plot))).to_numpy(dtype=float)
		denom_scope = denom_by_scope.get(scope, denom_all)
		ax2.plot(x, 100.0*c/denom_scope, marker=marker, linestyle=ls, linewidth=1.2, label=f"c→i HQ {label_scope}", color=colors[ci])
		ax2.plot(x, 100.0*i/denom_scope, marker="s", linestyle=ls, linewidth=1.2, label=f"i→c HQ {label_scope}", color=colors[ci])
	ax2.set_ylabel("Unique flips (% of ablated prompts)")
	ax2.set_xlabel("Layer")
	ax2.set_xticks(x)
	ax2.set_xticklabels(layer_labels, rotation=45, ha="right")
	ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
	ax2.legend(loc="best", fontsize=7, ncol=3)
	plot_path = stats_dir / "high_quality_neuron_flip_coverage_by_layer.pdf"
	plt.tight_layout()
	plt.savefig(plot_path, dpi=250, bbox_inches="tight")
	plt.close()
	print(f"[HighQualityByLayerScopes] Wrote {plot_path}")
	return out

def maybe_summarize_rule_metrics_and_high_quality(scores_df: pd.DataFrame, neurons_sorted, args, stats_dirname: str, baseline_metric_col: str = None):
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
	qual_thr = 0.70 if qual_thr is None else float(qual_thr)
	qual_thr_all_fit = float(getattr(args, "rule_quality_threshold_all_fit", 0.70))
	qual_thresholds_by_scope = {"test_selected": qual_thr, "test": qual_thr, "all_fit": qual_thr_all_fit}
	
	rule_best_df = write_rule_metrics_stats(
		rules_dir=args.rules_dir,
		out_dir=args.rules_dir,
		stats_dirname=stats_dirname,
		quantiles=quantiles,
		topk=int(getattr(args, "rule_metrics_topk", 50)),
		length_cap=int(getattr(args, "rule_length_cap", 30)),
		quality_metric=qual_metric,
		quality_threshold=qual_thr,
		quality_thresholds_by_scope=qual_thresholds_by_scope,
		min_dataset_coverage=float(getattr(args, "min_rule_dataset_coverage", DEFAULT_MIN_RULE_DATASET_COVERAGE)),
		neurons_sorted=neurons_sorted,
	)

	if rule_best_df is not None:
		# Load descriptive ALL-FIT best rows selected/scored on all available rows.
		rule_best_all_fit_df = _load_scope_best_rule_metrics(
			args.rules_dir,
			stats_dirname,
			"all_fit",
			min_dataset_coverage=float(getattr(args, "min_rule_dataset_coverage", DEFAULT_MIN_RULE_DATASET_COVERAGE)),
			quality_metric=qual_metric,
		)
		if rule_best_all_fit_df is None or rule_best_all_fit_df.empty:
			print("[HighQualityScopes] ALL-FIT best metrics unavailable; figures will show HQ-T only.")
		else:
			_diagnose_scope_metric_overlap(rule_best_df, rule_best_all_fit_df, _rule_metric_col(_norm_rule_metric_name(qual_metric)))
		write_high_quality_neuron_flip_coverage_with_scopes(
			scores_df=scores_df,
			neurons_sorted=neurons_sorted,
			rule_best_df_test_selected=rule_best_df,
			rule_best_df_all_fit=rule_best_all_fit_df,
			out_dir=args.rules_dir,
			stats_dirname=stats_dirname,
			quality_metric=qual_metric,
			quality_threshold=qual_thr,
			quality_thresholds_by_scope=qual_thresholds_by_scope,
			min_dataset_coverage=float(getattr(args, "min_rule_dataset_coverage", DEFAULT_MIN_RULE_DATASET_COVERAGE)),
			baseline_metric_col=baseline_metric_col,
		)

		write_high_quality_neuron_flip_coverage_by_layer_with_scopes(
			scores_df=scores_df,
			neurons_sorted=neurons_sorted,
			rule_best_df_test_selected=rule_best_df,
			rule_best_df_all_fit=rule_best_all_fit_df,
			out_dir=args.rules_dir,
			stats_dirname=stats_dirname,
			quality_metric=qual_metric,
			quality_threshold=qual_thr,
			quality_thresholds_by_scope=qual_thresholds_by_scope,
			min_dataset_coverage=float(getattr(args, "min_rule_dataset_coverage", DEFAULT_MIN_RULE_DATASET_COVERAGE)),
		)

	return rule_best_df

def main():
	args = parse_args()
	if bool(getattr(args, "decode_only", False)) and bool(getattr(args, "input_only", False)):
		raise ValueError("--decode_only and --input_only are mutually exclusive intervention phases")
	if args.stats_dirname is None:
		args.stats_dirname = ""
	set_deterministic(args.seed)

	os.makedirs(os.path.join(args.rules_dir, 'stats', args.stats_dirname), exist_ok=True)

	task = resolve_task_spec(args.task_module)
	if hasattr(task, "configure_runtime"):
		task.configure_runtime(args)
	is_answer_positive_fn = task.is_answer_positive
	answer_semantics_fn = getattr(task, "evaluate_answer_semantics", None)
	semantic_judge_mode = str(getattr(task, "semantic_judge_mode", "off") or "off")
	semantic_diagnostics_enabled = callable(answer_semantics_fn) and semantic_judge_mode != 'off'
	semantic_runtime_config = {
		"semantic_diagnostics_enabled": bool(semantic_diagnostics_enabled),
		"semantic_quality_policy": str(getattr(args, "semantic_quality_policy", "none")),
		"semantic_judge_model": str(getattr(task, "semantic_judge_model", "") or ""),
		"semantic_judge_mode": semantic_judge_mode,
		"semantic_judge_cache_path": str(getattr(task, "semantic_judge_cache_path", "") or ""),
		"semantic_judge_max_tokens": int(getattr(task, "semantic_judge_max_tokens", 0) or 0),
	}
	Path(os.path.join(args.rules_dir, 'stats', args.stats_dirname, 'semantic_runtime_config.json')).write_text(
		json.dumps(semantic_runtime_config, indent=2, ensure_ascii=False),
		encoding="utf-8",
	)
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

	# Derive feature columns from features.json produced by the previous script.
	# Normal experiments store a list of feature dictionaries.  Poisoning is
	# deliberately dataset-only and stores an empty list.  Older poisoning runs
	# may contain a metadata dictionary instead; accept that representation as
	# featureless too so cached stage-6 results can be resumed safely.
	features_json_path = os.path.join(args.features_scores_dir, "features.json")
	with open(features_json_path, "r", encoding="utf-8") as f:
		features_meta = json.load(f)
	features_json_fp = _file_fingerprint(Path(features_json_path))
	if isinstance(features_meta, dict):
		if features_meta.get("feature_engineering") is False or features_meta.get("mode") == "dataset_only":
			feature_records = []
		elif isinstance(features_meta.get("features"), list):
			feature_records = features_meta["features"]
		else:
			raise TypeError(
				"features.json must be a list of feature records, or an explicit "
				"dataset-only metadata object with feature_engineering=false."
			)
	elif isinstance(features_meta, list):
		feature_records = features_meta
	else:
		raise TypeError(f"Unsupported features.json root type: {type(features_meta).__name__}")

	bad_feature_records = [
		feat for feat in feature_records
		if not isinstance(feat, dict) or "label" not in feat
	]
	if bad_feature_records:
		raise TypeError(
			"features.json feature entries must be objects containing a 'label' field; "
			f"found {len(bad_feature_records)} malformed entries."
		)

	feature_name_set = {sanitize_feature_name(feat["label"]) for feat in feature_records}
	input_features = [
		c for c in scores_df.columns
		if c in feature_name_set
		and c not in metrics_list
	]
	if not feature_records:
		print("[Features] Dataset-only mode: no semantic feature columns; singleton evaluation remains enabled.")
		if args.extract_rules:
			raise ValueError(
				"--extract_rules cannot be used with dataset-only features.json. "
				"Disable rule extraction or run semantic feature generation first."
			)

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
		if args.candidate_ranking_csv:
			print(f"[Rule-metrics-only] Loading fixed candidates from {args.candidate_ranking_csv} ...")
			neurons, _ = load_candidate_ranking_override(args.candidate_ranking_csv)
		else:
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

		maybe_summarize_rule_metrics_and_high_quality(scores_stats, neurons, args, stats_dirname, baseline_metric_col=main_metric)
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
		evaluation_split = str(args.evaluation_split).strip().lower()
		n_source_rows = int(len(scores_stats))
		n_test_rows = None
		if "is_test" in scores_stats.columns:
			is_test = scores_stats["is_test"].fillna(False).astype(bool)
			n_test_rows = int(is_test.sum())
		else:
			is_test = None
		if evaluation_split in {"test", "train"}:
			if is_test is None:
				raise ValueError(
					f"--evaluation_split {evaluation_split} requires an is_test column in the materialized scores table."
				)
			mask = is_test if evaluation_split == "test" else ~is_test
			if not bool(mask.any()):
				raise ValueError(f"--evaluation_split {evaluation_split} selected zero rows in stats-only mode.")
			scores_stats = scores_stats.loc[mask].copy().reset_index(drop=True)
		print(f"[Stats-only] Evaluation split={evaluation_split}: {len(scores_stats)} of {n_source_rows} rows retained")
		evaluation_baseline_subset = str(args.evaluation_baseline_subset).strip().lower()
		if evaluation_baseline_subset != "all":
			if main_metric not in scores_stats.columns:
				raise ValueError(f"Baseline predicate column {main_metric!r} is required for --evaluation_baseline_subset.")
			baseline_num = pd.to_numeric(scores_stats[main_metric], errors="coerce")
			valid = baseline_num.notna()
			keep = valid & (baseline_num > 0.5) if evaluation_baseline_subset == "positive" else valid & (baseline_num <= 0.5)
			scores_stats = scores_stats.loc[keep].copy().reset_index(drop=True)
			if scores_stats.empty:
				raise ValueError(f"--evaluation_baseline_subset {evaluation_baseline_subset} selected zero rows in stats-only mode.")
		print(f"[Stats-only] Baseline subset={evaluation_baseline_subset}: {len(scores_stats)} rows retained")
		evaluation_pool_rows_before_cap = int(len(scores_stats))
		sampling_max_points = int(getattr(args, "sampling_max_points", 0) or 0)
		if sampling_max_points > 0 and len(scores_stats) > sampling_max_points:
			rng = np.random.default_rng(int(args.seed))
			keep_pos = np.sort(rng.choice(len(scores_stats), size=sampling_max_points, replace=False))
			scores_stats = scores_stats.iloc[keep_pos].copy().reset_index(drop=True)
			print(
				f"[Stats-only] Existing stage-7 cap retained {len(scores_stats)}/{evaluation_pool_rows_before_cap} rows "
				f"(--sampling_max_points={sampling_max_points}, seeded uniform sample, seed={int(args.seed)})."
			)

		# Discover neurons from per_rule and apply bucket filtering (same as normal mode)
		if args.candidate_ranking_csv:
			print(f"[Stats-only] Loading fixed candidates from {args.candidate_ranking_csv} ...")
			neurons, frozen_candidate_ranking_df = load_candidate_ranking_override(args.candidate_ranking_csv)
		else:
			print(f"[Stats-only] Discovering neurons from {circuit_agonists_path} (max_effect >= {args.search_epsilon}) ...")
			neurons = extract_single_neurons(circuit_agonists_path, search_epsilon=float(args.search_epsilon))
			frozen_candidate_ranking_df = extract_frozen_candidate_ranking(
				circuit_agonists_path, neurons, search_epsilon=float(args.search_epsilon)
			)
		
		# Parse quantiles for range reporting
		flip_stats_df = write_flip_stats(
			scores_stats, neurons, args.rules_dir, topk=50, stats_dirname=stats_dirname,
			baseline_metric_col=main_metric, search_epsilon=args.search_epsilon,
			reference_n_per_side=(int(os.environ["SEARCH_EPSILON_REFERENCE_N"]) if os.environ.get("SEARCH_EPSILON_REFERENCE_N", "").strip() else None),
			frozen_candidate_ranking_df=frozen_candidate_ranking_df,
		)
		thresholds = [
			float(value.strip())
			for value in str(args.singleton_rate_thresholds).split(",")
			if value.strip()
		]
		write_heldout_set_metrics(
			scores_stats, flip_stats_df, frozen_candidate_ranking_df, args.rules_dir,
			stats_dirname=stats_dirname, baseline_metric_col=main_metric,
			thresholds=thresholds,
			denominator_epsilon=float(args.metric_denominator_epsilon),
		)
		stats_dir = Path(args.rules_dir) / "stats" / str(stats_dirname)
		evaluation_scope_payload = {
			"holdout_test_only": evaluation_split == "test",
			"has_is_test_column": bool("is_test" in scores_stats.columns),
			"n_source_rows": n_source_rows,
			"n_test_rows_in_source": n_test_rows,
			"n_rows_ablated": int(len(scores_stats)),
			"n_rows_used_for_final_stats": int(len(scores_stats)),
			"sampling_pool_split": evaluation_split,
			"mean_replacement_reference_split": "train",
			"candidate_discovery_split": "train",
			"final_statistics_split": evaluation_split,
			"evaluation_baseline_subset": evaluation_baseline_subset,
			"evaluation_pool_rows_before_cap": evaluation_pool_rows_before_cap,
			"sampling_max_points": sampling_max_points if sampling_max_points > 0 else None,
			"evaluation_cap_applied": bool(sampling_max_points > 0 and evaluation_pool_rows_before_cap > sampling_max_points),
			"exclude_discovery_rows_from_final_stats": bool(getattr(args, "exclude_discovery_rows_from_final_stats", False)),
			"intervention": str(args.intervention),
			"intervention_phase": intervention_phase_name(args),
			"candidate_ranking_file": "frozen_candidate_ranking.csv",
			"regeneration_mode": "stats_only_from_materialized_singleton_events",
		}
		(stats_dir / "evaluation_scope.json").write_text(
			json.dumps(evaluation_scope_payload, indent=2, ensure_ascii=False),
			encoding="utf-8",
		)

		# Keep the materialized score table consistent with the regenerated JSON.
		# Use an atomic replace so an interrupted write cannot corrupt the only copy.
		materialized_scores_path = Path(args.rules_dir) / "stats" / str(stats_dirname) / "scores.csv"
		if Path(scores_in_path).resolve() == materialized_scores_path.resolve():
			tmp_scores_path = materialized_scores_path.with_suffix(".csv.tmp")
			scores_stats.to_csv(tmp_scores_path, index=False)
			tmp_scores_path.replace(materialized_scores_path)
			print(f"[Stats-only] Rewrote directional flip columns in {materialized_scores_path}")

		if not getattr(args, "skip_agonist_metric_stats", False):
			write_agonist_metric_final_stats(
				circuit_agonists_path,
				args.rules_dir,
				stats_dirname=stats_dirname,
				flip_stats_df=flip_stats_df,
				topk=args.agonist_metric_topk,
			)

		# In stats-only mode, also refresh rule_combo_metrics_* and HQ coverage
		# artifacts when requested.  This lets callers update scoped HQ-T/HQ-F
		# summaries from existing raw rule_combo CSVs without deleting stale
		# summary CSVs or re-running model ablations.
		maybe_summarize_rule_metrics_and_high_quality(
			scores_stats,
			neurons,
			args,
			stats_dirname,
			baseline_metric_col=main_metric,
		)

		return

	device = get_device()
	model = None
	unhooked_model = None
	tokenizer = None

	def _ensure_model_loaded():
		"""Load the target LLM only when a cache miss actually needs it."""
		nonlocal model, unhooked_model, tokenizer
		if model is None:
			print("[Model] Loading target LLM because a cache miss requires generation/representations ...")
			lm_wrapper_kwargs = {}
			if callable(getattr(task, "lm_wrapper_kwargs", None)):
				lm_wrapper_kwargs = dict(task.lm_wrapper_kwargs(ai_model) or {})
			model = LMWrapper(
				model_name=ai_model,
				device=device,
				eval_mode=True,
				cache_dir=args.ai_model_cache_dir,
				**lm_wrapper_kwargs,
			)
			unhooked_model = getattr(model, "model", None)
			tokenizer = getattr(model, "tokenizer", None)
		return model, unhooked_model, tokenizer

	def _release_model():
		"""Free the target LLM before CPU-only stats/RuleSHAP work."""
		nonlocal model, unhooked_model, tokenizer
		if model is None and unhooked_model is None and tokenizer is None:
			return
		print("[Model] Releasing target LLM before stats/rule extraction ...")
		model = None
		unhooked_model = None
		tokenizer = None
		gc.collect()
		try:
			import torch as _torch
			if _torch.cuda.is_available():
				_torch.cuda.empty_cache()
			if hasattr(_torch, "mps") and hasattr(_torch.mps, "empty_cache"):
				_torch.mps.empty_cache()
		except Exception:
			pass

	# Discover neurons from per_rule, or evaluate an explicitly frozen union.
	if args.candidate_ranking_csv:
		print(f"[1/5] Loading fixed candidates from {args.candidate_ranking_csv} ...")
		neurons, frozen_candidate_ranking_df = load_candidate_ranking_override(args.candidate_ranking_csv)
	else:
		print(f"[1/5] Discovering neurons from {circuit_agonists_path} (max_effect >= {args.search_epsilon}) ...")
		neurons = extract_single_neurons(circuit_agonists_path, search_epsilon=float(args.search_epsilon))
		frozen_candidate_ranking_df = extract_frozen_candidate_ranking(
			circuit_agonists_path,
			neurons,
			search_epsilon=float(args.search_epsilon),
		)

	if args.max_neurons is not None and args.max_neurons > 0:
		neurons = neurons[: args.max_neurons]
		allowed = {(str(layer), int(nid)) for layer, nid, _ in neurons}
		frozen_candidate_ranking_df = frozen_candidate_ranking_df[
			frozen_candidate_ranking_df.apply(lambda r: (str(r["layer_label"]), int(r["neuron_id"])) in allowed, axis=1)
		].copy().reset_index(drop=True)
	print(f"Found {len(neurons)} neurons to evaluate.")
	if not neurons:
		print("No neurons found; nothing to do.")
		return

	# Prepare prompts
	all_prompts_full = scores_df.loc[:, prompt_col].astype(str).tolist()
	n_points_total = len(all_prompts_full)
	all_examples_full = scores_df.to_dict(orient="records") # <-- list[dict] rows

	# ----------------- spectral sampling subset (CACHED) -----------------
	sample_indices = np.arange(n_points_total, dtype=int)
	evaluation_split = str(args.evaluation_split).strip().lower()
	holdout_test_only = evaluation_split == "test"
	if evaluation_split in {"test", "train"}:
		if "is_test" not in scores_df.columns:
			raise ValueError(
				f"--evaluation_split {evaluation_split} requires an is_test column in "
				"features_scores_dir/scores.csv."
			)
		is_test = scores_df["is_test"].fillna(False).astype(bool).to_numpy()
		mask = is_test if evaluation_split == "test" else ~is_test
		evaluation_pool_indices = np.where(mask)[0].astype(int)
	else:
		evaluation_pool_indices = np.arange(n_points_total, dtype=int)
	evaluation_baseline_subset = str(args.evaluation_baseline_subset).strip().lower()
	if evaluation_baseline_subset != "all":
		if main_metric not in scores_df.columns:
			raise ValueError(f"Baseline predicate column {main_metric!r} is required for --evaluation_baseline_subset.")
		baseline_num = pd.to_numeric(scores_df[main_metric], errors="coerce")
		valid_baseline = baseline_num.notna().to_numpy(dtype=bool)
		positive_baseline = (baseline_num.fillna(0).to_numpy(dtype=float) > 0.5)
		baseline_mask = positive_baseline if evaluation_baseline_subset == "positive" else ~positive_baseline
		baseline_mask &= valid_baseline
		evaluation_pool_indices = evaluation_pool_indices[baseline_mask[evaluation_pool_indices]]
	evaluation_pool_rows_before_cap = int(evaluation_pool_indices.size)
	sampling_max_points = int(getattr(args, "sampling_max_points", 0) or 0)
	if (not args.use_spectral_sampling) and sampling_max_points > 0 and evaluation_pool_indices.size > sampling_max_points:
		rng = np.random.default_rng(int(args.seed))
		selected = rng.choice(evaluation_pool_indices, size=sampling_max_points, replace=False)
		evaluation_pool_indices = np.sort(np.asarray(selected, dtype=int))
		print(
			f"[Split] Existing stage-7 cap retained {len(evaluation_pool_indices)}/"
			f"{evaluation_pool_rows_before_cap} rows "
			f"(--sampling_max_points={sampling_max_points}, seeded uniform sample, seed={int(args.seed)})."
		)
	if evaluation_pool_indices.size == 0:
		raise ValueError(
			f"--evaluation_split {evaluation_split} with --evaluation_baseline_subset "
			f"{evaluation_baseline_subset} selected zero rows."
		)
	print(
		f"[Split] Evaluation pool={evaluation_split}, baseline_subset={evaluation_baseline_subset}: "
		f"{len(evaluation_pool_indices)} rows."
	)

	# Spectral sampling is always performed inside the declared evaluation pool.
	sampling_pool_indices = evaluation_pool_indices.copy()
	sampling_pool_prompts = [all_prompts_full[i] for i in sampling_pool_indices]
	sampling_split_tag = evaluation_split

	if args.use_spectral_sampling:
		if int(args.sampling_max_points) <= 0:
			raise ValueError("--sampling_max_points must be positive when --use_spectral_sampling is enabled.")
		print(
			f"[2/5] Building spectral sampling subset within {sampling_split_tag} pool "
			f"(n={len(sampling_pool_indices)}, cap={int(args.sampling_max_points)}) ..."
		)

		if evaluation_split != "all":
			cache_root = Path(getattr(args, "spectral_cache_dir", None) or (Path(args.rules_dir) / "spectral_cache"))
			cache_root.mkdir(parents=True, exist_ok=True)
			run_tag = _safe_dirname(str(getattr(args, "stats_dirname", evaluation_split) or evaluation_split))
			spectral_cache_fp = cache_root / (
				f"spectral_{run_tag}_{evaluation_split}_cap{int(args.sampling_max_points)}_seed{int(args.seed)}.pkl"
			)
		else:
			# The all-row cache follows the full evaluation pool.
			spectral_cache_fp = _spectral_cache_path(args)

		def _compute_spectral_sampling():
			_model, _unhooked_model, _tokenizer = _ensure_model_loaded()
			_, Z = build_reps_and_embedding_from_args(
				args=args,
				texts=sampling_pool_prompts,
				model=_unhooked_model,
				tokenizer=_tokenizer,
				device=device,
			)
			Z = np.asarray(Z, dtype=np.float32)

			all_idx = np.arange(Z.shape[0], dtype=int)
			pool_size = int(all_idx.size)
			max_points = min(int(args.sampling_max_points), pool_size)
			min_points = min(int(args.sampling_min_points), max_points)

			out = {}

			if args.global_n_clusters > 0:
				effective_k = min(int(args.global_n_clusters), pool_size, max_points)
				global_center_idx, global_cluster_id, _, _, x_norm2 = kcenter_farthest_first(
					Z, k=effective_k
				)

				target_n = max_points
				if target_n < min_points:
					target_n = min_points

				sample_indices_local, _ = representative_sample_from_global_clusters(
					Z=Z,
					x_norm2=x_norm2,
					centers_idx=global_center_idx,
					cluster_id=global_cluster_id,
					group_idx=all_idx,
					n_select=target_n,
				)

				points_per_centroid = float(len(sample_indices_local)) / float(effective_k)

				out.update({
					"sample_indices": np.asarray(sample_indices_local, dtype=np.int64),
					"points_per_centroid": points_per_centroid,
					"effective_global_n_clusters": effective_k,
				})
			else:
				# Greedy cover centers + optional expansion per centroid.  Keep the
				# minimum below the hard cap so --sampling_max_points is truly a cap.
				centers_idx, _ = greedy_spectral_cover(
					Z,
					all_idx,
					radius=args.coverage_radius,
					max_points=max_points,
					min_points=min_points,
					return_meta=True,
				)
				centers_idx = np.asarray(centers_idx, dtype=np.int64)

				assign_to_center_local = compute_nearest_center_assignments(
					Z=Z,
					centers_idx=centers_idx,
					chunk_size=args.sampling_chunk_size,
				)

				sample_indices_local, _ = build_per_centroid_sample_indices(
					Z=Z,
					centers_idx=centers_idx,
					assign_to_center=assign_to_center_local,
					points_per_centroid=args.points_per_centroid,
					mode=args.centroid_sampling_mode,
				)
				# Expansion by points_per_centroid must not exceed the user-facing cap.
				sample_indices_local = np.asarray(sample_indices_local, dtype=np.int64)[:max_points]

				out.update({
					"sample_indices": sample_indices_local,
					"points_per_centroid": int(args.points_per_centroid),
				})

			return out

		spectral_obj = load_or_create_cache(
			str(spectral_cache_fp),
			_compute_spectral_sampling,
			quiet=True,
		)

		sample_indices_local = np.asarray(spectral_obj["sample_indices"], dtype=int)
		if sample_indices_local.size and (
			sample_indices_local.min() < 0
			or sample_indices_local.max() >= len(sampling_pool_indices)
		):
			raise ValueError(
				f"Spectral cache {spectral_cache_fp} contains indices outside the "
				f"{sampling_split_tag} sampling pool."
			)
		sample_indices = np.asarray(sampling_pool_indices[sample_indices_local], dtype=int)
		effective_k = int(spectral_obj.get("effective_global_n_clusters", args.global_n_clusters))
		points_per_centroid = spectral_obj.get(
			"points_per_centroid",
			(len(sample_indices) / effective_k) if effective_k > 0 else args.points_per_centroid
		)

		print(
			f"[Sampling] Selected {len(sample_indices)} prompts from the {sampling_split_tag} pool "
			f"(cap={int(args.sampling_max_points)}, ~{points_per_centroid} per centroid, "
			f"mode={args.centroid_sampling_mode}). Cache={spectral_cache_fp}"
		)
	else:
		sample_indices = sampling_pool_indices.copy()
		print(
			f"[2/5] Spectral sampling disabled; evaluating {len(sample_indices)} "
			f"rows in split={evaluation_split} after the existing stage-7 cap."
		)

	# Decide which datapoints we actually evaluate under ablation.
	# When spectral sampling is enabled, ablations are run exactly on the sampled rows.
	# There is no special "force include test rows" path: HQ-T/HQ-F rule scopes are
	# computed over the rows that are actually present/evaluated in scores_df.
	if evaluation_split != "all":
		# With spectral sampling enabled, sample_indices is a deterministic subset
		# of the selected split. Without it, sample_indices is the full split.
		eval_indices = np.asarray(sample_indices, dtype=int)
		seen = set()
		eval_indices = np.asarray(
			[idx for idx in eval_indices.tolist() if not (idx in seen or seen.add(idx))],
			dtype=int,
		)
		if eval_indices.size == 0:
			raise ValueError(f"Sampling selected zero rows from evaluation split={evaluation_split}.")
		if not np.isin(eval_indices, evaluation_pool_indices).all():
			raise AssertionError(f"A row outside evaluation split={evaluation_split} entered evaluation.")
		if int(args.sampling_max_points) > 0 and len(eval_indices) > int(args.sampling_max_points):
			raise AssertionError(
				f"Evaluation sample has {len(eval_indices)} rows, exceeding "
				f"--sampling_max_points={int(args.sampling_max_points)}."
			)
		eval_prompts = [all_examples_full[i] for i in eval_indices]
		scores_out = scores_df.iloc[eval_indices].copy().reset_index(drop=True)
		scores_out["_orig_row"] = eval_indices
		scores_out["_evaluated"] = True
		scores_out["_spectral_sampled_center"] = bool(args.use_spectral_sampling)
	elif args.use_spectral_sampling:
		# --- sampled set (dedupe, preserve order) ---
		seen = set()
		keep_pos = []
		for pos, idx in enumerate(sample_indices.tolist()):
			if idx in seen:
				continue
			seen.add(idx)
			keep_pos.append(pos)
		eval_indices_sampled = sample_indices[np.asarray(keep_pos, dtype=int)]

		# Spectral evaluation uses only the sampled evaluation rows.
		eval_indices = np.unique(eval_indices_sampled).astype(int)
		eval_indices.sort()

		eval_prompts = [all_examples_full[i] for i in eval_indices]

		# masks in original-row space
		spectral_sampled_center_mask = np.zeros(n_points_total, dtype=bool)
		spectral_sampled_center_mask[eval_indices_sampled] = True

		evaluated_mask = np.zeros(n_points_total, dtype=bool)
		evaluated_mask[eval_indices] = True
		if args.keep_unsampled_rows:
			scores_out = scores_df.copy()
			scores_out["_evaluated"] = evaluated_mask
			scores_out["_spectral_sampled_center"] = spectral_sampled_center_mask
		else:
			scores_out = scores_df.iloc[eval_indices].copy().reset_index(drop=True)
			scores_out["_orig_row"] = eval_indices
			scores_out["_evaluated"] = True
			scores_out["_spectral_sampled_center"] = spectral_sampled_center_mask[eval_indices]
	else:
		# Non-spectral evaluation, including evaluation_split=all, uses the
		# already filtered/capped evaluation pool.
		eval_indices = np.asarray(sample_indices, dtype=int)
		if eval_indices.size == 0:
			raise ValueError("Non-spectral evaluation selected zero rows.")
		eval_prompts = [all_examples_full[i] for i in eval_indices]
		scores_out = scores_df.iloc[eval_indices].copy().reset_index(drop=True)
		scores_out["_orig_row"] = eval_indices
		scores_out["_evaluated"] = True
		scores_out["_spectral_sampled_center"] = False

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
					_model, _, _ = _ensure_model_loaded()
					return precompute_mean_activations(
						model=_model,
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
		layer_key = safe_layer_label(layer_label)
		cache_policy_tag = build_flip_cache_policy_tag(args, main_metric=main_metric)
		rules_subdir = os.path.join(args.rules_dir, cache_policy_tag, "ablation_cache")

		# hooks only depend on neuron+global args (NOT on batch)
		hooks = build_ablation_hooks(
			{layer_label: [neuron_id]},
			last_pos_only=hooks_last_pos_only(args),
			intervention=args.intervention,
			mean_activations=mean_activations_all,
			device=device,
		)

		neuron_specs.append((layer_label, neuron_id, _baseline_subset, layer_key, rules_subdir, hooks, cache_policy_tag))

	batch_size = args.batch_size
	neuron_batch_size = max(1, int(getattr(args, "neuron_batch_size", 1) or 1))
	print('batch_size:', batch_size, 'neuron_batch_size:', neuron_batch_size, 'synthetic_batch_upper_bound:', batch_size * neuron_batch_size)
	neuron_specs_by_layer = defaultdict(list)
	for spec in neuron_specs:
		neuron_specs_by_layer[str(spec[0])].append(spec)

	def _flip_cache_path(rules_subdir, layer_key, neuron_id, batch_start):
		return os.path.join(rules_subdir, f"{layer_key}_{int(neuron_id)}_{batch_start}.pkl")

	def _cached_eval_file_exists(cache_path):
		"""Cheap cache-hit predicate: no pickle load, no answer rescoring, no LLM."""
		try:
			return os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0
		except OSError:
			return False

	def _coerce_cached_answers(value):
		if value is None:
			return None
		if isinstance(value, np.ndarray):
			value = value.tolist()
		if isinstance(value, (list, tuple)):
			return ["" if v is None else str(v) for v in value]
		return None

	def _coerce_cached_semantics(value, expected_n=None):
		semantics = {}
		for k, v in (value or {}).items():
			arr = np.asarray(v).astype(bool)
			if expected_n is not None and len(arr) != expected_n:
				continue
			semantics[k] = arr
		return semantics

	def _load_cached_eval(cache_path, rows=None):
		if not os.path.exists(cache_path):
			return None
		with open(cache_path, "rb") as f:
			obj = pickle.load(f)

		if not isinstance(obj, dict) or obj.get("cache_schema_version") != 2 or "correct" not in obj:
			return None

		expected_n = len(rows) if rows is not None else None
		correct = np.asarray(obj["correct"]).astype(bool)
		if expected_n is not None and len(correct) != expected_n:
			return None

		payload = {
			"correct": correct,
			"semantics": _coerce_cached_semantics(obj.get("semantics", {}), expected_n),
		}
		answers = _coerce_cached_answers(obj.get("answers"))
		if answers is not None:
			if expected_n is not None and len(answers) != expected_n:
				return None
			payload["answers"] = answers
		return payload

	def _save_cached_eval(cache_path, correct, semantics=None, answers=None):
		Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
		payload = {
			"cache_schema_version": 2,
			"correct": np.asarray(correct).astype(bool),
			"semantics": {k: np.asarray(v).astype(bool) for k, v in (semantics or {}).items()},
		}
		answers = _coerce_cached_answers(answers)
		if answers is not None:
			payload["answers"] = answers
		with open(cache_path, "wb") as f:
			pickle.dump(payload, f)
		# _note_cache_segment is defined below; all helpers are created before the
		# ablation loop calls this saver.
		_note_cache_segment(cache_path)

	# Cache files are keyed by the first evaluation-row offset. Historically the
	# poisoning driver forced batch_size=1, so a restart with a larger prompt
	# batch would otherwise ignore thousands of valid singleton cache files.
	# Build a lightweight index of cache segments and stitch adjacent/overlapping
	# segments so cache reuse is independent of the requested batch size.
	_cache_segment_index = {}

	def _parse_cache_segment_name(filename):
		if not str(filename).endswith(".pkl"):
			return None
		stem = str(filename)[:-4]
		try:
			layer_part, neuron_part, start_part = stem.rsplit("_", 2)
			return layer_part, int(neuron_part), int(start_part)
		except (ValueError, TypeError):
			return None

	def _cache_dir_segments(rules_subdir):
		rules_subdir = str(rules_subdir)
		indexed = _cache_segment_index.get(rules_subdir)
		if indexed is not None:
			return indexed
		indexed = defaultdict(list)
		try:
			names = os.listdir(rules_subdir)
		except OSError:
			names = []
		for name in names:
			parsed = _parse_cache_segment_name(name)
			if parsed is None:
				continue
			layer_part, neuron_part, start_part = parsed
			indexed[(layer_part, int(neuron_part))].append(int(start_part))
		for starts in indexed.values():
			starts.sort()
		_cache_segment_index[rules_subdir] = indexed
		return indexed

	def _note_cache_segment(cache_path):
		parent = str(Path(cache_path).parent)
		indexed = _cache_segment_index.get(parent)
		if indexed is None:
			return
		parsed = _parse_cache_segment_name(Path(cache_path).name)
		if parsed is None:
			return
		layer_part, neuron_part, start_part = parsed
		starts = indexed[(layer_part, int(neuron_part))]
		pos = bisect_right(starts, int(start_part))
		if pos == 0 or starts[pos - 1] != int(start_part):
			starts.insert(pos, int(start_part))

	def _combine_cached_eval_parts(parts):
		if not parts:
			return None
		correct_parts = [np.asarray(part["correct"]).astype(bool) for part in parts]
		out = {"correct": np.concatenate(correct_parts, axis=0)}

		semantic_keys = sorted({key for part in parts for key in (part.get("semantics", {}) or {}).keys()})
		if semantic_keys:
			semantics = {}
			for key in semantic_keys:
				arrays = []
				for part, correct in zip(parts, correct_parts):
					value = (part.get("semantics", {}) or {}).get(key)
					if value is None:
						arrays.append(np.zeros(len(correct), dtype=bool))
					else:
						arrays.append(np.asarray(value).astype(bool))
				semantics[key] = np.concatenate(arrays, axis=0)
			out["semantics"] = semantics
		else:
			out["semantics"] = {}

		if all(_coerce_cached_answers(part.get("answers")) is not None for part in parts):
			answers = []
			for part in parts:
				answers.extend(_coerce_cached_answers(part.get("answers")))
			out["answers"] = answers
		return out

	def _slice_cached_eval(payload, lo, hi):
		part = {
			"correct": np.asarray(payload["correct"]).astype(bool)[lo:hi],
			"semantics": {
				key: np.asarray(value).astype(bool)[lo:hi]
				for key, value in (payload.get("semantics", {}) or {}).items()
			},
		}
		answers = _coerce_cached_answers(payload.get("answers"))
		if answers is not None:
			part["answers"] = answers[lo:hi]
		return part

	def _load_cached_eval_range(rules_subdir, layer_key, neuron_id, start_i, rows):
		"""Load one requested evaluation range, independent of old cache batch size."""
		expected_n = len(rows)
		if expected_n == 0:
			return {"correct": np.zeros(0, dtype=bool), "semantics": {}, "answers": []}

		exact_path = _flip_cache_path(rules_subdir, layer_key, neuron_id, start_i)
		exact = _load_cached_eval(exact_path, rows)
		if exact is not None:
			return exact

		starts = _cache_dir_segments(rules_subdir).get((str(layer_key), int(neuron_id)), [])
		if not starts:
			return None
		want_end = int(start_i) + int(expected_n)
		cursor = int(start_i)
		parts = []
		local_payloads = {}

		while cursor < want_end:
			pos = bisect_right(starts, cursor) - 1
			chosen = None
			while pos >= 0:
				seg_start = int(starts[pos])
				payload = local_payloads.get(seg_start)
				if payload is None:
					seg_path = _flip_cache_path(rules_subdir, layer_key, neuron_id, seg_start)
					payload = _load_cached_eval(seg_path, rows=None)
					local_payloads[seg_start] = payload
				if payload is not None:
					seg_len = len(np.asarray(payload.get("correct", [])))
					if seg_len > 0 and seg_start <= cursor < seg_start + seg_len:
						chosen = (seg_start, seg_len, payload)
						break
				pos -= 1
			if chosen is None:
				return None

			seg_start, seg_len, payload = chosen
			lo = cursor - seg_start
			take = min(seg_len - lo, want_end - cursor)
			if take <= 0:
				return None
			parts.append(_slice_cached_eval(payload, lo, lo + take))
			cursor += take

		combined = _combine_cached_eval_parts(parts)
		if combined is None or len(np.asarray(combined.get("correct", []))) != expected_n:
			return None
		return combined

	def _ensure_bool_col(col_name):
		if col_name not in scores_out.columns:
			scores_out[col_name] = pd.array([pd.NA] * len(scores_out), dtype="boolean")

	def _ensure_flip_cols(layer_key, neuron_id):
		col_any = f"flip_{layer_key}_{int(neuron_id)}"
		col_c2i = f"flip_c2i_{layer_key}_{int(neuron_id)}"
		col_i2c = f"flip_i2c_{layer_key}_{int(neuron_id)}"
		for c in (col_any, col_c2i, col_i2c):
			_ensure_bool_col(c)
		return col_any, col_c2i, col_i2c

	def _ensure_semantic_cols(layer_key, neuron_id):
		cols = {
			"semantic_wrong": f"semantic_wrong_{layer_key}_{int(neuron_id)}",
			"semantic_correct": f"semantic_correct_{layer_key}_{int(neuron_id)}",
			"empty": f"answer_empty_{layer_key}_{int(neuron_id)}",
			"unparseable": f"answer_unparseable_{layer_key}_{int(neuron_id)}",
			"flip_semantic_wrong": f"flip_semantic_wrong_{layer_key}_{int(neuron_id)}",
			"flip_empty": f"flip_empty_{layer_key}_{int(neuron_id)}",
			"flip_unparseable": f"flip_unparseable_{layer_key}_{int(neuron_id)}",
			"judge_used": f"answer_judge_used_{layer_key}_{int(neuron_id)}",
			"judge_parse_success": f"answer_judge_parse_success_{layer_key}_{int(neuron_id)}",
		}
		for c in cols.values():
			_ensure_bool_col(c)
		return cols

	def _expected_flip_cols_for_neuron(layer_label, neuron_id):
		layer_key = safe_layer_label(layer_label)
		return (
			f"flip_{layer_key}_{int(neuron_id)}",
			f"flip_c2i_{layer_key}_{int(neuron_id)}",
			f"flip_i2c_{layer_key}_{int(neuron_id)}",
		)

	def _existing_scores_path():
		stats_dirname = args.stats_dirname if args.stats_dirname is not None else ""
		return Path(args.rules_dir) / "stats" / str(stats_dirname) / "scores.csv"

	def _load_existing_scores_if_complete():
		"""Reuse the materialized scores.csv instead of replaying thousands of pkl files."""
		fp = _existing_scores_path()
		if not fp.exists():
			return None
		try:
			candidate = pd.read_csv(fp)
		except Exception as e:
			print(f"[Cache] Could not read existing materialized scores {fp}: {e}")
			return None
		if len(candidate) != len(scores_out):
			print(f"[Cache] Existing materialized scores row mismatch ({len(candidate)} != {len(scores_out)}); rebuilding from ablation cache.")
			return None
		required = []
		for layer_label, neuron_id, _baseline_subset in neurons:
			required.extend(_expected_flip_cols_for_neuron(layer_label, neuron_id))
		missing = [c for c in required if c not in candidate.columns]
		if missing:
			print(f"[Cache] Existing materialized scores missing {len(missing)} flip columns; rebuilding from ablation cache.")
			return None
		print(f"[Cache] Reusing materialized flip columns from {fp}; skipping ablation cache replay.")
		return candidate

	def _all_ablation_cache_files_exist(batch_starts):
		missing = []
		for start_i in batch_starts:
			end_i = min(start_i + batch_size, len(eval_prompts))
			batch_rows_i = eval_prompts[start_i:end_i]
			for layer_label, neuron_id, _baseline_subset, layer_key, rules_subdir, _hooks, _cache_policy_tag in neuron_specs:
				layer_key = safe_layer_label(layer_label)
				cache_path = _flip_cache_path(rules_subdir, layer_key, neuron_id, start_i)
				if _load_cached_eval_range(rules_subdir, layer_key, neuron_id, start_i, batch_rows_i) is None:
					missing.append(cache_path)
					if len(missing) >= 5:
						return False, missing
		return True, missing

	def _row_positions_for_eval_batch(start_i, end_i):
		if args.use_spectral_sampling and args.keep_unsampled_rows:
			return np.asarray(eval_indices[start_i:end_i], dtype=int)
		return np.arange(start_i, end_i, dtype=int)

	def _assign_bool_values(storage, positions, values):
		storage[np.asarray(positions, dtype=int)] = np.asarray(values).astype(bool)

	def _rebuild_scores_from_ablation_cache_only(batch_starts):
		"""Materialize flip columns from existing pkl files without prefix prefill/generation.

		This is much faster than the normal generation loop on fully cached runs because
		it writes each dataframe column once instead of doing tiny iloc writes for every
		(batch, neuron) pair.
		"""
		n_rows_out = len(scores_out)
		for layer_label, neuron_id, _baseline_subset, layer_key, rules_subdir, _hooks, _cache_policy_tag in tqdm(
			neuron_specs, desc="Rebuilding cached flip columns"
		):
			layer_key = safe_layer_label(layer_label)
			col_any, col_c2i, col_i2c = _ensure_flip_cols(layer_key, neuron_id)
			any_values = np.full(n_rows_out, pd.NA, dtype=object)
			c2i_values = np.full(n_rows_out, pd.NA, dtype=object)
			i2c_values = np.full(n_rows_out, pd.NA, dtype=object)
			semantic_values = {}

			for start_i in batch_starts:
				end_i = min(start_i + batch_size, len(eval_prompts))
				batch_prompt_i = eval_prompts[start_i:end_i]
				cache_path = _flip_cache_path(rules_subdir, layer_key, neuron_id, start_i)
				cached_eval = _load_cached_eval_range(rules_subdir, layer_key, neuron_id, start_i, batch_prompt_i)
				if cached_eval is None:
					raise RuntimeError(f"Expected cached ablation result missing or invalid: {cache_path}")
				abl = np.asarray(cached_eval.get("correct", [])).astype(bool)
				base = np.asarray(baseline_eval[start_i:end_i]).astype(bool)
				flip_any = (abl != base)
				positions = _row_positions_for_eval_batch(start_i, end_i)
				_assign_bool_values(any_values, positions, flip_any)
				_assign_bool_values(c2i_values, positions, (base == True) & (abl == False))
				_assign_bool_values(i2c_values, positions, (base == False) & (abl == True))

				semantics = cached_eval.get("semantics", {}) or {}
				if semantics:
					cols = _ensure_semantic_cols(layer_key, neuron_id)
					if not semantic_values:
						semantic_values = {name: np.full(n_rows_out, pd.NA, dtype=object) for name in cols.values()}
					zero = np.zeros_like(flip_any, dtype=bool)
					sem_wrong = np.asarray(semantics.get("answer_semantically_wrong", zero)).astype(bool)
					sem_correct = np.asarray(semantics.get("answer_semantically_correct", zero)).astype(bool)
					empty = np.asarray(semantics.get("answer_empty", zero)).astype(bool)
					unparseable = np.asarray(semantics.get("answer_unparseable", zero)).astype(bool)
					judge_used = np.asarray(semantics.get("answer_judge_enabled", zero)).astype(bool)
					judge_parse_success = np.asarray(semantics.get("answer_judge_parse_success", zero)).astype(bool)
					_assign_bool_values(semantic_values[cols["semantic_wrong"]], positions, sem_wrong)
					_assign_bool_values(semantic_values[cols["semantic_correct"]], positions, sem_correct)
					_assign_bool_values(semantic_values[cols["empty"]], positions, empty)
					_assign_bool_values(semantic_values[cols["unparseable"]], positions, unparseable)
					_assign_bool_values(semantic_values[cols["judge_used"]], positions, judge_used)
					_assign_bool_values(semantic_values[cols["judge_parse_success"]], positions, judge_parse_success)
					_assign_bool_values(semantic_values[cols["flip_semantic_wrong"]], positions, flip_any & sem_wrong)
					_assign_bool_values(semantic_values[cols["flip_empty"]], positions, flip_any & empty)
					_assign_bool_values(semantic_values[cols["flip_unparseable"]], positions, flip_any & unparseable)

			scores_out[col_any] = pd.Series(any_values, dtype="boolean")
			scores_out[col_c2i] = pd.Series(c2i_values, dtype="boolean")
			scores_out[col_i2c] = pd.Series(i2c_values, dtype="boolean")
			for col_name, values in semantic_values.items():
				scores_out[col_name] = pd.Series(values, dtype="boolean")
		return scores_out

	def _semantic_arrays(rows, answers):
		if not semantic_diagnostics_enabled:
			return {}
		try:
			records = answer_semantics_fn(rows, answers)
		except Exception as e:
			print(f"[Semantics] Task semantic evaluator failed; disabling diagnostics for this batch: {e}")
			return {}
		records = list(records or [])
		if len(records) != len(answers):
			records = (records + [{} for _ in range(len(answers))])[:len(answers)]

		def _as_bool(value, default=False):
			if value is None:
				return bool(default)
			if isinstance(value, (bool, np.bool_)):
				return bool(value)
			if isinstance(value, (int, float)) and not (isinstance(value, float) and np.isnan(value)):
				return bool(value)
			text = str(value).strip().lower()
			if text in {"true", "1", "yes", "y"}:
				return True
			if text in {"false", "0", "no", "n", ""}:
				return False
			return bool(default)

		empty = np.asarray([_as_bool((r or {}).get("answer_empty", False)) for r in records], dtype=bool)
		parse_success = np.asarray([_as_bool((r or {}).get("answer_parse_success", False)) for r in records], dtype=bool)
		sem_correct = np.asarray([_as_bool((r or {}).get("answer_semantically_correct", False)) for r in records], dtype=bool)
		sem_wrong = np.asarray([_as_bool((r or {}).get("answer_semantically_wrong", False)) for r in records], dtype=bool)
		unparseable = (~parse_success) | empty
		judge_used = np.asarray([_as_bool((r or {}).get("answer_judge_enabled", False)) for r in records], dtype=bool)
		judge_parse_success = np.asarray([_as_bool((r or {}).get("answer_judge_parse_success", False)) for r in records], dtype=bool)
		return {
			"answer_empty": empty,
			"answer_parse_success": parse_success,
			"answer_unparseable": unparseable,
			"answer_semantically_correct": sem_correct,
			"answer_semantically_wrong": sem_wrong,
			"answer_judge_enabled": judge_used,
			"answer_judge_parse_success": judge_parse_success,
		}

	def _write_ablation_flips(layer_key, neuron_id, ablated_correct, batch_baseline_eval, batch_row_pos, semantics=None):
		col_any, col_c2i, col_i2c = _ensure_flip_cols(layer_key, neuron_id)
		abl = np.asarray(ablated_correct).astype(bool)
		flip_any = (abl != batch_baseline_eval)
		flip_c2i = (batch_baseline_eval == True) & (abl == False)
		flip_i2c = (batch_baseline_eval == False) & (abl == True)
		scores_out.iloc[batch_row_pos, scores_out.columns.get_loc(col_any)] = flip_any.astype(bool)
		scores_out.iloc[batch_row_pos, scores_out.columns.get_loc(col_c2i)] = flip_c2i.astype(bool)
		scores_out.iloc[batch_row_pos, scores_out.columns.get_loc(col_i2c)] = flip_i2c.astype(bool)

		semantics = semantics or {}
		if semantics:
			cols = _ensure_semantic_cols(layer_key, neuron_id)
			sem_wrong = np.asarray(semantics.get("answer_semantically_wrong", np.zeros_like(flip_any))).astype(bool)
			sem_correct = np.asarray(semantics.get("answer_semantically_correct", np.zeros_like(flip_any))).astype(bool)
			empty = np.asarray(semantics.get("answer_empty", np.zeros_like(flip_any))).astype(bool)
			unparseable = np.asarray(semantics.get("answer_unparseable", np.zeros_like(flip_any))).astype(bool)
			judge_used = np.asarray(semantics.get("answer_judge_enabled", np.zeros_like(flip_any))).astype(bool)
			judge_parse_success = np.asarray(semantics.get("answer_judge_parse_success", np.zeros_like(flip_any))).astype(bool)
			scores_out.iloc[batch_row_pos, scores_out.columns.get_loc(cols["semantic_wrong"])] = sem_wrong
			scores_out.iloc[batch_row_pos, scores_out.columns.get_loc(cols["semantic_correct"])] = sem_correct
			scores_out.iloc[batch_row_pos, scores_out.columns.get_loc(cols["empty"])] = empty
			scores_out.iloc[batch_row_pos, scores_out.columns.get_loc(cols["unparseable"])] = unparseable
			scores_out.iloc[batch_row_pos, scores_out.columns.get_loc(cols["judge_used"])] = judge_used
			scores_out.iloc[batch_row_pos, scores_out.columns.get_loc(cols["judge_parse_success"])] = judge_parse_success
			scores_out.iloc[batch_row_pos, scores_out.columns.get_loc(cols["flip_semantic_wrong"])] = (flip_any & sem_wrong)
			scores_out.iloc[batch_row_pos, scores_out.columns.get_loc(cols["flip_empty"])] = (flip_any & empty)
			scores_out.iloc[batch_row_pos, scores_out.columns.get_loc(cols["flip_unparseable"])] = (flip_any & unparseable)

	def _score_repeated_rows(rows, answers):
		return (np.asarray(is_answer_positive_fn(rows, answers), dtype=float) > 0.5).astype(bool)

	def _score_rows_with_semantics(rows, answers):
		return _score_repeated_rows(rows, answers), _semantic_arrays(rows, answers)

	def _split_flat_eval_by_spec(acc_flat, sem_flat, answers_flat, B, K):
		acc_matrix = np.asarray(acc_flat).astype(bool).reshape(B, K).T  # [K, B]
		answers_matrix = np.asarray(["" if a is None else str(a) for a in answers_flat], dtype=object).reshape(B, K).T
		sem_by_j = []
		answers_by_j = []
		for j in range(K):
			sem_j = {}
			for key, arr in (sem_flat or {}).items():
				sem_j[key] = np.asarray(arr).astype(bool).reshape(B, K).T[j]
			sem_by_j.append(sem_j)
			answers_by_j.append([str(a) for a in answers_matrix[j].tolist()])
		return acc_matrix, sem_by_j, answers_by_j

	def _empty_eval_store(chunk_specs, n_rows):
		acc_by_spec = {}
		sem_by_spec = {}
		answers_by_spec = {}
		for i, (layer_label_i, neuron_id_i, *_rest) in enumerate(chunk_specs):
			key = (layer_label_i, int(neuron_id_i), int(i))
			acc_by_spec[key] = np.zeros(n_rows, dtype=bool)
			sem_by_spec[key] = {}
			answers_by_spec[key] = [""] * n_rows
		return acc_by_spec, sem_by_spec, answers_by_spec

	def _compute_rowwise_chunk_decode_only(layer_label, chunk_specs, prefix_batches, batch_ranges, batch_prompt):
		_model, _, _ = _ensure_model_loaded()
		K = len(chunk_specs)
		acc_by_spec, sem_by_spec, answers_by_spec = _empty_eval_store(chunk_specs, len(batch_prompt))
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
				last_pos_only=hooks_last_pos_only(args),
				intervention=args.intervention,
				mean_activations=mean_activations_all,
				device=device,
			)
			# `repeated_prefix` is created for this one generation and never reused.
			# Avoid a redundant full KV clone; reusable prefix batches use the safe default=True path.
			answers = _model.generate_from_prefix_cache(
				repeated_prefix,
				fwd_hooks=hooks,
				stop_at_eos=True,
				clone_kv_cache_tensors=False,
			)
			repeated_rows = [row for row in prefix_rows for _ in range(K)]
			acc_flat, sem_flat = _score_rows_with_semantics(repeated_rows, answers)
			acc_matrix, sem_list, answers_list = _split_flat_eval_by_spec(acc_flat, sem_flat, answers, B, K)
			for j, spec in enumerate(chunk_specs):
				key = (spec[0], int(spec[1]), int(j))
				acc_by_spec[key][b_start:b_end] = acc_matrix[j]
				for sem_key, arr in sem_list[j].items():
					if sem_key not in sem_by_spec[key]:
						sem_by_spec[key][sem_key] = np.zeros(len(batch_prompt), dtype=bool)
					sem_by_spec[key][sem_key][b_start:b_end] = arr
				answers_by_spec[key][b_start:b_end] = answers_list[j]

		return [
			(
				acc_by_spec[(spec[0], int(spec[1]), int(j))],
				sem_by_spec[(spec[0], int(spec[1]), int(j))],
				answers_by_spec[(spec[0], int(spec[1]), int(j))],
			)
			for j, spec in enumerate(chunk_specs)
		]

	def _compute_rowwise_chunk_prefill_decode(layer_label, chunk_specs, batch_prompt):
		_model, _, _ = _ensure_model_loaded()
		K = len(chunk_specs)
		chunk_ids = [int(spec[1]) for spec in chunk_specs]
		repeated_rows = [row for row in batch_prompt for _ in range(K)]
		row_neuron_ids = chunk_ids * len(batch_prompt)
		hooks = build_rowwise_ablation_hooks(
			layer_label,
			row_neuron_ids,
			last_pos_only=hooks_last_pos_only(args),
			intervention=args.intervention,
			mean_activations=mean_activations_all,
			device=device,
		)
		repeated_prompts = [str(row[prompt_col]) for row in repeated_rows]
		answers = _model.generate(
			repeated_prompts,
			max_new_tokens=max_new_tokens,
			do_sample=False,
			fwd_hooks=hooks,
		)
		acc_flat, sem_flat = _score_rows_with_semantics(repeated_rows, answers)
		acc_matrix, sem_list, answers_list = _split_flat_eval_by_spec(acc_flat, sem_flat, answers, len(batch_prompt), K)
		return [(acc_matrix[j], sem_list[j], answers_list[j]) for j in range(K)]

	def _compute_rowwise_chunk_input_only(layer_label, chunk_specs, batch_prompt):
		_model, _, _ = _ensure_model_loaded()
		K = len(chunk_specs)
		chunk_ids = [int(spec[1]) for spec in chunk_specs]
		repeated_rows = [row for row in batch_prompt for _ in range(K)]
		row_neuron_ids = chunk_ids * len(batch_prompt)
		hooks = build_rowwise_ablation_hooks(
			layer_label,
			row_neuron_ids,
			last_pos_only=hooks_last_pos_only(args),
			intervention=args.intervention,
			mean_activations=mean_activations_all,
			device=device,
		)
		repeated_prompts = [str(row[prompt_col]) for row in repeated_rows]
		prefix = _model.prefill_prefix_batch(
			repeated_prompts,
			max_new_tokens=max_new_tokens,
			use_kv_cache=True,
			fwd_hooks=hooks,
		)
		# This prefix is also one-shot: prefill hooks have already been applied and the
		# object is discarded immediately after decoding.
		answers = _model.generate_from_prefix_cache(
			prefix,
			fwd_hooks=None,
			stop_at_eos=True,
			clone_kv_cache_tensors=False,
		)
		acc_flat, sem_flat = _score_rows_with_semantics(repeated_rows, answers)
		acc_matrix, sem_list, answers_list = _split_flat_eval_by_spec(acc_flat, sem_flat, answers, len(batch_prompt), K)
		return [(acc_matrix[j], sem_list[j], answers_list[j]) for j in range(K)]


	batch_starts = list(range(0, len(eval_prompts), batch_size))
	ablation_cache_materialized = False

	existing_scores_out = _load_existing_scores_if_complete()
	if existing_scores_out is not None:
		scores_out = existing_scores_out
		ablation_cache_materialized = True
	else:
		all_ablation_pkls_exist, missing_ablation_pkls = _all_ablation_cache_files_exist(batch_starts)
		if all_ablation_pkls_exist:
			print("[Cache] All ablation .pkl files exist; rebuilding flip columns cache-only (no prefix prefill/generation).")
			scores_out = _rebuild_scores_from_ablation_cache_only(batch_starts)
			Path(_existing_scores_path()).parent.mkdir(parents=True, exist_ok=True)
			scores_out.to_csv(_existing_scores_path(), index=False)
			print(f"[Cache] Wrote materialized flip columns to {_existing_scores_path()}.")
			ablation_cache_materialized = True
		elif missing_ablation_pkls:
			print(f"[Cache] Found missing ablation cache files; generation needed. First missing: {missing_ablation_pkls[0]}")

	if not ablation_cache_materialized:
		for start in tqdm(batch_starts, total=len(batch_starts), desc="Ablation cache/generation batches"):
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
					safe_layer_key = safe_layer_label(layer_label)
					cache_path = _flip_cache_path(rules_subdir, safe_layer_key, neuron_id, start)
					# Prefix prefill is only needed when at least one ablation cache file is absent.
					# Do not unpickle/rescore here; BON rescoring can launch the classifier LLM.
					if not _cached_eval_file_exists(cache_path):
						need_prefill = True
						break

				if need_prefill:
					_model, _, _ = _ensure_model_loaded()
					prefix_batches, batch_ranges = build_prefix_caches_for_examples(
						_model,
						batch_prompt,
						prompt_col,
						max_new_tokens=max_new_tokens,
						batch_size=batch_size,
					)

			# Old execution path, kept for parity and for low-memory runs.
			if neuron_batch_size <= 1:
				for layer_label, neuron_id, _baseline_subset, layer_key, rules_subdir, hooks, _cache_policy_tag in tqdm(neuron_specs, desc="Neurons" if show_batch_tqdm else None):
					layer_key = safe_layer_label(layer_label)
					_ensure_flip_cols(layer_key, neuron_id)
					cache_path = _flip_cache_path(rules_subdir, layer_key, neuron_id, start)

					cached_eval = _load_cached_eval_range(rules_subdir, layer_key, neuron_id, start, batch_prompt)
					if cached_eval is None:
						_model, _, _ = _ensure_model_loaded()
						if args.decode_only:
							_, acc, answers = get_correctness_cached_by_prefix_batches(
								_model,
								batch_prompt,
								is_answer_positive_fn,
								prefix_batches,
								batch_ranges,
								hooks=hooks,
								return_answers=True,
							)
						elif args.input_only:
							batch_prompts = [str(ex[prompt_col]) for ex in batch_prompt]
							prefix = _model.prefill_prefix_batch(
								batch_prompts,
								max_new_tokens=max_new_tokens,
								use_kv_cache=True,
								fwd_hooks=hooks,
							)
							# One-shot input-only prefix; no later generation reuses this cache.
							answers = _model.generate_from_prefix_cache(
								prefix,
								fwd_hooks=None,
								stop_at_eos=True,
								clone_kv_cache_tensors=False,
							)
							acc = _score_repeated_rows(batch_prompt, answers)
						else:
							_, acc, answers = get_correctness(
								_model,
								batch_prompt,
								is_answer_positive_fn,
								prompt_col,
								max_new_tokens=max_new_tokens,
								hooks=hooks,
								batch_size=batch_size,
								tqdm_desc=None,
								return_answers=True,
							)
						ablated_correct = (np.asarray(acc) > 0.5).astype(bool)
						semantics = _semantic_arrays(batch_prompt, answers)
						_save_cached_eval(cache_path, ablated_correct, semantics, answers=answers)
					else:
						ablated_correct = np.asarray(cached_eval.get("correct", [])).astype(bool)
						semantics = cached_eval.get("semantics", {}) or {}

					_write_ablation_flips(layer_key, neuron_id, ablated_correct, batch_baseline_eval, batch_row_pos, semantics=semantics)
				continue

			# Row-wise multi-neuron path. Each synthetic row ablates exactly one neuron,
			# including singleton donor-safe replacement for mean-donor variants.
			layer_iter = neuron_specs_by_layer.items()
			for layer_label, specs_this_layer in layer_iter:
				for chunk_start in range(0, len(specs_this_layer), neuron_batch_size):
					chunk_specs_all = specs_this_layer[chunk_start:chunk_start + neuron_batch_size]

					cached_or_none = []
					compute_specs = []
					for spec in chunk_specs_all:
						layer_label_i, neuron_id_i, _baseline_subset_i, layer_key_i, rules_subdir_i, _hooks_i, _cache_policy_tag_i = spec
						layer_key_i = safe_layer_label(layer_label_i)
						_ensure_flip_cols(layer_key_i, neuron_id_i)
						cache_path = _flip_cache_path(rules_subdir_i, layer_key_i, neuron_id_i, start)
						cached = _load_cached_eval_range(rules_subdir_i, layer_key_i, neuron_id_i, start, batch_prompt)
						cached_or_none.append(cached)
						if cached is None:
							compute_specs.append(spec)

					computed_by_key = {}
					if compute_specs:
						if args.decode_only:
							if prefix_batches is None or batch_ranges is None:
								_model, _, _ = _ensure_model_loaded()
								prefix_batches, batch_ranges = build_prefix_caches_for_examples(
									_model,
									batch_prompt,
									prompt_col,
									max_new_tokens=max_new_tokens,
									batch_size=batch_size,
								)
							computed_list = _compute_rowwise_chunk_decode_only(layer_label, compute_specs, prefix_batches, batch_ranges, batch_prompt)
						elif args.input_only:
							computed_list = _compute_rowwise_chunk_input_only(layer_label, compute_specs, batch_prompt)
						else:
							computed_list = _compute_rowwise_chunk_prefill_decode(layer_label, compute_specs, batch_prompt)

						for spec, (arr, semantics, answers_j) in zip(compute_specs, computed_list):
							layer_label_i, neuron_id_i, _baseline_subset_i, layer_key_i, rules_subdir_i, _hooks_i, _cache_policy_tag_i = spec
							layer_key_i = safe_layer_label(layer_label_i)
							arr = np.asarray(arr).astype(bool)
							semantics = semantics or {}
							computed_by_key[(str(layer_label_i), int(neuron_id_i))] = {"correct": arr, "semantics": semantics, "answers": answers_j}
							_save_cached_eval(_flip_cache_path(rules_subdir_i, layer_key_i, neuron_id_i, start), arr, semantics, answers=answers_j)

					for spec, cached in zip(chunk_specs_all, cached_or_none):
						layer_label_i, neuron_id_i, _baseline_subset_i, layer_key_i, _rules_subdir_i, _hooks_i, _cache_policy_tag_i = spec
						layer_key_i = safe_layer_label(layer_label_i)
						if cached is None:
							payload = computed_by_key[(str(layer_label_i), int(neuron_id_i))]
						else:
							payload = cached
						ablated_correct = np.asarray(payload.get("correct", [])).astype(bool)
						semantics = payload.get("semantics", {}) or {}
						_write_ablation_flips(layer_key_i, neuron_id_i, ablated_correct, batch_baseline_eval, batch_row_pos, semantics=semantics)

		scores_out.to_csv(os.path.join(args.rules_dir, 'stats', args.stats_dirname, f"scores.csv"), index=False)

	# The remaining work is CPU/dataframe/RuleSHAP only.  Drop the target LLM so
	# cached BON runs do not carry hundreds of GB into rule extraction.
	_release_model()

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
	evaluation_scope_payload = {
		"holdout_test_only": holdout_test_only,
		"has_is_test_column": bool("is_test" in scores_df.columns),
		"n_source_rows": int(len(scores_df)),
		"n_test_rows_in_source": int(scores_df["is_test"].fillna(False).astype(bool).sum()) if "is_test" in scores_df.columns else None,
		"n_rows_ablated": int(len(scores_out)),
		"n_rows_used_for_final_stats": int(len(scores_for_final_stats)),
		"use_spectral_sampling": bool(args.use_spectral_sampling),
		"sampling_pool_split": evaluation_split,
		"sampling_max_points": sampling_max_points if sampling_max_points > 0 else None,
		"sampling_min_points": int(args.sampling_min_points) if args.use_spectral_sampling else None,
		"sampling_selected_all_pool_rows": bool(len(scores_out) == evaluation_pool_rows_before_cap),
		"sampling_fraction_of_evaluation_split": (
			float(len(scores_out)) / float(evaluation_pool_rows_before_cap)
			if evaluation_pool_rows_before_cap > 0 else None
		),
		"sampling_fraction_of_test": (
			float(len(scores_out)) / float(evaluation_pool_rows_before_cap)
			if evaluation_split == "test" and evaluation_pool_rows_before_cap > 0
			else None
		),
		"spectral_cache_path": str(spectral_cache_fp) if args.use_spectral_sampling else None,
		"mean_replacement_reference_split": "train",
		"candidate_discovery_split": "train",
		"final_statistics_split": evaluation_split,
		"evaluation_baseline_subset": evaluation_baseline_subset,
		"evaluation_pool_rows_before_cap": evaluation_pool_rows_before_cap,
		"evaluation_cap_applied": bool(sampling_max_points > 0 and evaluation_pool_rows_before_cap > sampling_max_points),
		"exclude_discovery_rows_from_final_stats": bool(getattr(args, "exclude_discovery_rows_from_final_stats", False)),
		"intervention": str(args.intervention),
		"intervention_phase": intervention_phase_name(args),
		"candidate_ranking_file": "frozen_candidate_ranking.csv",
	}
	Path(os.path.join(args.rules_dir, 'stats', args.stats_dirname, 'evaluation_scope.json')).write_text(
		json.dumps(evaluation_scope_payload, indent=2, ensure_ascii=False),
		encoding="utf-8",
	)
	flip_stats_df = write_flip_stats(
		scores_for_final_stats, neurons, args.rules_dir, topk=50, stats_dirname=args.stats_dirname,
		baseline_metric_col=main_metric, search_epsilon=args.search_epsilon,
		reference_n_per_side=(int(os.environ["SEARCH_EPSILON_REFERENCE_N"]) if os.environ.get("SEARCH_EPSILON_REFERENCE_N", "").strip() else None),
		frozen_candidate_ranking_df=frozen_candidate_ranking_df,
	)
	thresholds = [
		float(value.strip())
		for value in str(args.singleton_rate_thresholds).split(",")
		if value.strip()
	]
	write_heldout_set_metrics(
		scores_for_final_stats,
		flip_stats_df,
		frozen_candidate_ranking_df,
		args.rules_dir,
		stats_dirname=args.stats_dirname,
		baseline_metric_col=main_metric,
		thresholds=thresholds,
		denominator_epsilon=float(args.metric_denominator_epsilon),
	)
	if not getattr(args, "skip_agonist_metric_stats", False):
		write_agonist_metric_final_stats(
			circuit_agonists_path,
			args.rules_dir,
			stats_dirname=args.stats_dirname,
			flip_stats_df=flip_stats_df,
			topk=args.agonist_metric_topk,
		)


	def _semantic_quality_filter_col(layer_key, neuron_id):
		col = f"semantic_wrong_{layer_key}_{int(neuron_id)}"
		return col if col in scores_out.columns else None

	def _filter_semantically_invalid_rows(df_in, layer_key, neuron_id, *, split_name: str):
		if df_in is None or df_in.empty:
			return df_in, {"split": split_name, "available": False, "removed": 0, "remaining": 0}
		if str(getattr(args, "semantic_quality_policy", "none")) != "filter":
			return df_in, {"split": split_name, "available": False, "policy": str(getattr(args, "semantic_quality_policy", "none")), "removed": 0, "remaining": int(len(df_in))}
		col = _semantic_quality_filter_col(layer_key, neuron_id)
		if not col:
			return df_in, {"split": split_name, "available": False, "removed": 0, "remaining": int(len(df_in))}
		wrong = df_in[col].fillna(False).astype(bool)
		filtered = df_in.loc[~wrong].copy()
		return filtered, {
			"split": split_name,
			"available": True,
			"semantic_wrong_col": col,
			"removed": int(wrong.sum()),
			"remaining": int(len(filtered)),
			"original": int(len(df_in)),
		}

	# ----------------- per-neuron rule extraction -----------------
	if args.extract_rules:
		for layer_label, neuron_id, _baseline_subset in tqdm(neurons, desc="Neurons (rules)"):
			layer_key = safe_layer_label(layer_label)
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

				# If spectral sampling kept unsampled rows, train/eval should only see rows
				# with evaluated flip targets. Prefer _evaluated (which includes forced
				# held-out test rows).
				if args.use_spectral_sampling and args.keep_unsampled_rows:
					mask_col = "_evaluated" if "_evaluated" in rules_df.columns else None
					if mask_col is not None:
						rules_df = rules_df.loc[rules_df[mask_col].astype(bool)].copy()
					if eval_df is not None:
						emask_col = "_evaluated" if "_evaluated" in eval_df.columns else None
						if emask_col is not None:
							eval_df = eval_df.loc[eval_df[emask_col].astype(bool)].copy()

				# For cognitive-bias tasks, remove ablated datapoints whose generated answer is
				# semantically invalid before fitting/evaluating rules. This prevents high-MCC
				# rules from being driven by neurons that merely create empty, unparseable, or
				# semantically wrong outputs rather than meaningful bias flips.
				filter_records = []
				rules_df, rec = _filter_semantically_invalid_rows(rules_df, layer_key, neuron_id, split_name="train")
				filter_records.append(rec)
				if eval_df is not None:
					eval_df, rec = _filter_semantically_invalid_rows(eval_df, layer_key, neuron_id, split_name="eval")
					filter_records.append(rec)

				# Skip targets that are constant or too imbalanced to learn/plot reliably (CHECK ON TRAIN ONLY).
				def _ok_target(df, t, min_pos=2, min_neg=2):
					y = pd.to_numeric(df[t], errors="coerce").dropna()
					if y.nunique() < 2:
						return False
					pos = int((y > 0.5).sum())
					neg = int((y <= 0.5).sum())
					return (pos >= min_pos) and (neg >= min_neg)
				targets_this = [t for t in targets_this if _ok_target(rules_df, t, min_pos=2, min_neg=2)]

				if targets_this:
					# Keep a shared per-neuron directory; signature gating decides whether reuse is valid.
					os.makedirs(rules_subdir, exist_ok=True)
					Path(os.path.join(rules_subdir, "semantic_filtering.json")).write_text(
						json.dumps({"policy": str(getattr(args, "semantic_quality_policy", "none")), "records": filter_records}, indent=2, ensure_ascii=False),
						encoding="utf-8",
					)
					signature = build_rule_extraction_signature(
						rules_df=rules_df,
						input_features=input_features,
						targets=targets_this,
						args=args,
						features_json_fingerprint=features_json_fp,
						eval_df=eval_df,
					)
					if should_skip_rule_extraction(
						rules_subdir,
						"",
						f"{layer_key}_{neuron_id}",
						signature=signature,
						force_recompute=bool(getattr(args, "force_rule_recompute", False)),
						require_all_fit=bool(getattr(args, "emit_all_fit_rules", True)),
						targets=targets_this,
						force_all_fit_recompute=bool(getattr(args, "force_all_fit_recompute", False)),
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
	maybe_summarize_rule_metrics_and_high_quality(scores_out, neurons, args, stats_dirname2, baseline_metric_col=main_metric)

if __name__ == "__main__":
	main()
