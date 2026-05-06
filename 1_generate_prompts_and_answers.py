import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time

import argparse
import logging
import json
import pickle
from pathlib import Path

import torch
import pandas as pd
from more_itertools import unique_everseen  # unused here but fine

from lib.caching_and_prompting import load_or_create_cache, create_cache
from lib.modeling_and_ablation import LMWrapper, get_device
from lib.feature_extraction_runner import resolve_task_spec

#################################################################

def _coerce_bool_series(s: pd.Series) -> pd.Series:
	"""Best-effort conversion of a column to boolean (while preserving NaNs).

	Accepts actual booleans, 0/1, and common string forms.
	"""
	if s.dtype == bool:
		return s
	# Preserve NaNs
	na_mask = s.isna()
	# Normalize strings
	s_norm = s.astype(str).str.strip().str.lower()
	true_set = {"true", "t", "1", "yes", "y"}
	false_set = {"false", "f", "0", "no", "n"}
	out = pd.Series([pd.NA] * len(s), index=s.index, dtype="boolean")
	out[~na_mask & s_norm.isin(true_set)] = True
	out[~na_mask & s_norm.isin(false_set)] = False
	# Numeric fallbacks
	try:
		n = pd.to_numeric(s, errors="coerce")
		out[~na_mask & n.isin([0, 1])] = n[~na_mask & n.isin([0, 1])].astype(int).astype(bool)
	except Exception:
		pass
	return out


def _maybe_expand_dict_column(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
	"""If df[col] contains dict-like values, expand it into columns."""
	if col not in df.columns:
		return df
	if df[col].dropna().empty:
		return df
	# Only expand if at least one non-null value is a dict
	if not any(isinstance(v, dict) for v in df[col].dropna().head(50).tolist()):
		return df
	try:
		expanded = pd.json_normalize(df[col]).add_prefix(prefix)
		return pd.concat([df.drop(columns=[col]), expanded], axis=1)
	except Exception:
		return df


def _call_task_basic_stats(task, df: pd.DataFrame):
	"""Try task-provided statistics hooks before falling back to generic logic."""
	if task is None:
		return None
	for method_name in (
		"get_basic_statistics",
		"basic_statistics",
		"compute_basic_statistics",
		"dataset_statistics",
	):
		fn = getattr(task, method_name, None)
		if callable(fn):
			try:
				stats = fn(df)
				if isinstance(stats, dict):
					stats = dict(stats)
					stats.setdefault("n_examples", int(len(df)))
					return stats
			except TypeError:
				pass
			except Exception:
				pass
	return None


def compute_basic_stats(df: pd.DataFrame, task=None) -> dict:
	"""Compute a stats bundle, preferring task-specific hooks when available."""
	task_stats = _call_task_basic_stats(task, df)
	if isinstance(task_stats, dict):
		return task_stats

	stats: dict = {
		"n_examples": int(len(df)),
	}

	# Expand common nested fields (BON jailbreak dataset)
	df2 = df.copy()
	df2 = _maybe_expand_dict_column(df2, "augmentation_info", prefix="aug.")

	# Prefer task-declared targets over hardcoded dataset-specific labels.
	target_candidates = []
	if task is not None:
		default_targets = getattr(task, "DEFAULT_TARGETS", ())
		if isinstance(default_targets, str):
			default_targets = (default_targets,)
		for tgt in default_targets:
			if isinstance(tgt, str):
				target_candidates.append(tgt)

	# Keep legacy support for known benchmark targets.
	for tgt in ["is_correct", "is_jailbroken", "is_pure"]:
		if tgt not in target_candidates:
			target_candidates.append(tgt)

	# Coerce candidate boolean targets if present.
	for tgt in target_candidates:
		if tgt in df2.columns:
			df2[tgt] = _coerce_bool_series(df2[tgt])

	# Generic summaries for task-defined boolean targets.
	for tgt in target_candidates:
		if tgt not in df2.columns:
			continue
		s = df2[tgt].dropna()
		if not len(s):
			continue
		stats[f"n_labeled_{tgt}"] = int(len(s))
		if pd.api.types.is_bool_dtype(s.dtype):
			stats[f"pct_{tgt}"] = float(s.mean())
			stats[f"n_{tgt}"] = int(s.sum())

	# Arithmetic-specific enrichments
	if "is_correct" in df2.columns:
		s = df2["is_correct"].dropna()
		stats["accuracy"] = float(s.mean()) if len(s) else None
		stats["n_correct"] = int(s.sum()) if len(s) else 0
		stats["n_labeled"] = int(len(s))
		if "operator_group" in df2.columns:
			g = df2.dropna(subset=["is_correct"]).groupby("operator_group", dropna=False)["is_correct"]
			stats["accuracy_by_operator"] = {
				str(op): {
					"n": int(len(v)),
					"accuracy": float(v.mean()) if len(v) else None,
				}
				for op, v in g
			}
		if "is_pure" in df2.columns:
			p = df2["is_pure"].dropna()
			stats["pure_number_output_rate"] = float(p.mean()) if len(p) else None

	# Jailbreak-specific enrichments
	if "is_jailbroken" in df2.columns:
		jb = df2["is_jailbroken"].dropna()
		stats["jailbreak_rate"] = float(jb.mean()) if len(jb) else None
		stats["n_jailbroken"] = int(jb.sum()) if len(jb) else 0
		stats["n_labeled_jailbreak"] = int(len(jb))

		for col in ["aug.words_scrambled", "aug.chars_capitalized", "aug.chars_perturbed"]:
			if col in df2.columns:
				b = _coerce_bool_series(df2[col])
				tmp = pd.DataFrame({"feat": b, "jb": df2["is_jailbroken"]}).dropna()
				if not tmp.empty:
					stats[f"jailbreak_rate_by_{col.split('.')[-1]}"] = {
						"False": float(tmp.loc[tmp["feat"] == False, "jb"].mean()) if (tmp["feat"] == False).any() else None,
						"True": float(tmp.loc[tmp["feat"] == True, "jb"].mean()) if (tmp["feat"] == True).any() else None,
					}

		if "aug.sigma" in df2.columns:
			sigma = pd.to_numeric(df2["aug.sigma"], errors="coerce")
			stats["sigma_summary"] = {
				"min": None if sigma.dropna().empty else float(sigma.min()),
				"max": None if sigma.dropna().empty else float(sigma.max()),
				"mean": None if sigma.dropna().empty else float(sigma.mean()),
			}
			try:
				if sigma.nunique(dropna=True) >= 4:
					bins = pd.qcut(sigma, 4, duplicates="drop")
					tmp = pd.DataFrame({"bin": bins, "jb": df2["is_jailbroken"]}).dropna()
					if not tmp.empty:
						stats["jailbreak_rate_by_sigma_quartile"] = {
							str(k): float(v.mean()) for k, v in tmp.groupby("bin")["jb"]
						}
			except Exception:
				pass

	return stats


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Generate (or load) a cache of prompts and model answers for a given task. "
			"The concrete task logic (prompt generation, cache structure) is defined "
			"in the task module."
		)
	)

	parser.add_argument(
		'--prompts_answers_pkl_file',
		required=True,
		type=str,
		help='Path to the cache pickle file (load_or_create_cache-compatible).'
	)

	# --- model / device ---
	parser.add_argument(
		'--ai_model',
		default='Qwen/Qwen2-7B-Instruct',
		type=str,
		help='Name of the model to be loaded.'
	)
	parser.add_argument(
		'--ai_model_cache_dir',
		default=None,
		type=str,
		help='Optional local cache directory for the model weights.'
	)
	parser.add_argument(
		'--batch_size',
		default=16,
		type=int,
		help='Batch size used by the task when querying the model.'
	)

	# --- task / domain-specific options ---
	parser.add_argument(
		'--task_module',
		default='lib.tasks.arithmetic_task',
		type=str,
		help=(
			'Python module path implementing the task interface, including '
			'generate_cache(model, args).'
		)
	)

	parser.add_argument(
		'--stats_json_out',
		default=None,
		type=str,
		help='Optional path to write the computed basic statistics as JSON.'
	)

	return parser.parse_args()

args = parse_args()

# ------------------------------------------------------------------
# Load task module
# ------------------------------------------------------------------
task = resolve_task_spec(args.task_module)

ai_model = args.ai_model
logging.info(f"Starting analysis for model {ai_model}")

torch.set_grad_enabled(False)

# ------------------------------------------------------------------
# Build or load cache using the TASK'S generate_cache()
# ------------------------------------------------------------------
with torch.inference_mode():
	large_prompts_and_answers = load_or_create_cache(
		args.prompts_answers_pkl_file,
		lambda: task.generate_cache(args.ai_model, args.ai_model_cache_dir, args),
	)

# ------------------------------------------------------------------
# Basic statistics (jailbreak rate / arithmetic accuracy)
# ------------------------------------------------------------------
try:
	if hasattr(task, "load_dataset_from_cache"):
		df = task.load_dataset_from_cache(args.prompts_answers_pkl_file)
		stats = compute_basic_stats(df, task)
		print("\n==== Basic Statistics ====")
		print(json.dumps(stats, indent=2))
		if args.stats_json_out:
			out_dir = Path(args.stats_json_out)
			out_dir.mkdir(parents=True, exist_ok=True)
			with open(out_dir / 'dataset_stats.json', "w") as f:
				json.dump(stats, f, indent=2)
	else:
		print("[WARN] Task does not expose load_dataset_from_cache(); skipping stats.")
except Exception as e:
	print(f"[WARN] Could not compute stats from cache: {e}")

try:
	for t in range(3):
		time.sleep(30//(t+1))
		create_cache(
			args.prompts_answers_pkl_file,
			lambda: task.rerun_dataset_load(large_prompts_and_answers)
		)
except Exception as e:
	print(e)
	pass

# print('Prompts and answers:', json.dumps(large_prompts_and_answers, indent=4))
