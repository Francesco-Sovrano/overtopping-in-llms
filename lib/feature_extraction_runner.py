from __future__ import annotations

import os
import re
import json
import math
import random
import importlib
from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional, Sequence, Tuple, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from more_itertools import unique_everseen

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

from lib.caching_and_prompting import set_deterministic

from lib.feature_representation import *

MAX_ABS_FEATURE_VALUE = 1e12

def _sanitize_feature_score(
	s: Any,
	*,
	clip_abs: float = MAX_ABS_FEATURE_VALUE,
) -> tuple[float | None, str | None]:
	if s is None:
		return None, None

	if isinstance(s, (bool, np.bool_)):
		s = float(int(s))
	elif isinstance(s, (int, float, np.integer, np.floating)):
		s = float(s)
	else:
		raise TypeError(f"bad score type {type(s)}")

	if not np.isfinite(s):
		return None, "non-finite value"

	if clip_abs is not None and abs(s) > clip_abs:
		return float(np.clip(s, -clip_abs, clip_abs)), f"clipped to +/-{clip_abs:g}"

	return s, None


def _sanitize_feature_frame(
	X: pd.DataFrame,
	*,
	context: str,
	clip_abs: float = MAX_ABS_FEATURE_VALUE,
	fill_missing: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
	if X.empty:
		return X.copy(), []

	X = X.apply(pd.to_numeric, errors="coerce")
	X = X.replace([np.inf, -np.inf], np.nan)

	if clip_abs is not None:
		arr = X.to_numpy(dtype=np.float64, copy=False)
		too_large_mask = np.isfinite(arr) & (np.abs(arr) > clip_abs)
		if too_large_mask.any():
			bad_cols = X.columns[too_large_mask.any(axis=0)].tolist()
			print(f"[{context}] clipping extreme values in columns: {bad_cols} at +/-{clip_abs:g}")
			X = X.clip(lower=-clip_abs, upper=clip_abs)

	dropped_cols = X.columns[X.isna().all()].tolist()
	if dropped_cols:
		print(f"[{context}] dropping all-invalid columns: {dropped_cols}")
		X = X.drop(columns=dropped_cols)

	partial_bad_cols = X.columns[X.isna().any()].tolist()
	if partial_bad_cols:
		print(f"[{context}] imputing invalid values in columns: {partial_bad_cols}")

	if fill_missing and X.shape[1] > 0:
		X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

	arr = X.to_numpy(dtype=np.float64, copy=False)
	if arr.size and not np.isfinite(arr).all():
		bad_cols = X.columns[(~np.isfinite(arr)).any(axis=0)].tolist()
		raise ValueError(f"Non-finite values remain after sanitization in {context}: {bad_cols}")

	return X, dropped_cols


def _sanitize_feature_matrix_for_split(X: pd.DataFrame) -> pd.DataFrame:
	"""Prepare feature matrix for scaling/clustering during split creation."""
	X_sanitized, _ = _sanitize_feature_frame(X, context="feature split")
	return X_sanitized

def add_feature_stratified_is_test(
	df: pd.DataFrame,
	feature_cols: list[str],
	*,
	test_size: float = 0.20,
	seed: int = 42,
	split_col: str = "is_test",
	add_split_str_col: bool = True,
	y_for_stratification: np.ndarray | None = None,  # optional: combine with label
) -> pd.DataFrame:
	"""
	Adds a boolean split flag stratified by feature-space clusters.
	If y_for_stratification is provided (binary), strata become (cluster, y).
	"""
	df = df.copy()

	n = len(df)
	if n == 0:
		df[split_col] = False
		if add_split_str_col:
			df["split"] = "train"
		return df

	# Build feature matrix
	if not feature_cols:
		X_df = pd.DataFrame(index=df.index)
	else:
		X_df = _sanitize_feature_matrix_for_split(df[feature_cols].copy())

	# Small dataset fallback: random split
	if n < 10:
		rng = np.random.default_rng(seed)
		idx = np.arange(n)
		rng.shuffle(idx)
		n_test = max(1, int(round(test_size * n)))
		test_idx = idx[:n_test]
		df[split_col] = False
		df.iloc[test_idx, df.columns.get_loc(split_col)] = True
		if add_split_str_col:
			df["split"] = np.where(df[split_col], "test", "train")
		return df

	# -------------------------
	# 1) Embed feature space for clustering (stratification proxy)
	# -------------------------
	if X_df.shape[1] == 0:
		X_embed = None
		strata = np.zeros(n, dtype=np.int64)
	else:
		Xs = StandardScaler().fit_transform(X_df.to_numpy(dtype=np.float64, copy=True))

		# Use PCA to retain 95% variance if dimensionality is high
		X_embed = Xs
		if Xs.shape[1] > 10:
			X_embed = PCA(n_components=0.95, random_state=seed, svd_solver="full").fit_transform(Xs)

	# -------------------------
	# 2) Create strata = cluster_id (optionally combined with y)
	# -------------------------
	# For stratified sampling with test_size, each stratum should have at least ceil(1/test_size) items
	if X_embed is not None:
		min_per_stratum = int(np.ceil(1.0 / test_size))  # 0.2 -> 5

		# Upper bound on how many clusters we can even support
		k_cap = min(200, max(2, n // min_per_stratum))

		# Start with a *high* k (finer strata => more precise matching), then back off if too many tiny clusters
		# Aim average cluster size about ~1.5 * min_per_stratum
		denom = int(np.ceil(1.5 * min_per_stratum))
		k = int(np.clip(n // max(1, denom), 2, k_cap))

		strata = None
		while k >= 2:
			km = MiniBatchKMeans(
				n_clusters=k,
				random_state=seed,
				batch_size=2048,
				n_init="auto",
			)
			labels = km.fit_predict(X_embed)
			counts = np.bincount(labels, minlength=k)

			# Accept only if *all* clusters are big enough to stratify
			if counts.min() >= min_per_stratum:
				strata = labels
				break

			# Back off (reduce k) until clusters are sufficiently populated
			k = int(max(2, np.floor(k * 0.8)))

		if strata is None:
			# fallback: single stratum (still deterministic split, but not feature-stratified)
			strata = np.zeros(n, dtype=np.int64)

	# Optional: also preserve label distribution (binary) within each cluster
	if y_for_stratification is not None:
		yb = np.asarray(y_for_stratification).reshape(-1)
		if yb.dtype.kind in "fc":
			yb = (yb >= 0.5).astype(np.int8)
		else:
			yb = yb.astype(np.int8)
		strata = strata.astype(np.int64) * 2 + yb

	# -------------------------
	# 3) Stratified split using the 1D strata key
	# -------------------------
	idx = np.arange(n)
	try:
		train_idx, test_idx = train_test_split(
			idx,
			test_size=test_size,
			random_state=seed,
			stratify=strata,
		)
	except ValueError:
		# Rare edge-cases (e.g., after combining with y some strata become too small)
		train_idx, test_idx = train_test_split(
			idx,
			test_size=test_size,
			random_state=seed,
			shuffle=True,
			stratify=None,
		)

	df[split_col] = False
	df.iloc[test_idx, df.columns.get_loc(split_col)] = True
	if add_split_str_col:
		df["split"] = np.where(df[split_col], "test", "train")
	return df

# -----------------------------------------------------------------------------
# Public config + result types
# -----------------------------------------------------------------------------
@dataclass
class FeatureExtractionConfig:
	# LLM proposal
	ai_model: str = "qwen3:30b"
	temperature: float = 0.3
	n_features: int = 16
	feature_extraction_steps: int = 10
	num_correct_example_prompts: int = 32
	num_incorrect_example_prompts: int = 32
	no_llm_feature_generation: bool = False
	number_of_existing_features_to_show_for_diversity: int = 20
	cache_dir: str = "./cache"

	# Filtering
	drop_low_predictive_power_features: bool = False
	drop_near_duplicate_features: bool = False
	near_duplicate_features_threshold: float = 0.9999
	drop_high_mad_variance_features: bool = False

	train_ratio: float = 1.0
	min_auc: float = 0.0
	min_delta: float = 0.2
	min_ap_above_base: float = 0.0
	z_thresh: float = 50

	# Run control
	max_number_of_prompts_to_analyze: int = 0
	seed: int = 42

	# UX
	progress: bool = True
	warn_on_feature_exceptions: bool = False


@dataclass
class FeatureExtractionResult:
	wide: pd.DataFrame
	features: list
	feature_columns: list[str]
	df_scores_full: pd.DataFrame
	metrics_train: Optional[pd.DataFrame] = None


# -----------------------------------------------------------------------------
# Task-spec resolution
# -----------------------------------------------------------------------------
def resolve_task_spec(task_module: str):
	"""
	Imports a task module and returns task.TASK_SPEC.
	Expected attributes on TASK_SPEC (as per your script):
	  DEFAULT_TARGETS, DEFAULT_INPUT, DEFAULT_OUTPUT,
	  SYSTEM_PROMPT, TOKENS_DICT_KEYS, SEED_FEATURES,
	  parse_prompt, load_dataset_from_cache
	"""
	task = importlib.import_module(task_module)
	if not hasattr(task, "TASK_SPEC"):
		raise ValueError(f"Task module '{task_module}' has no TASK_SPEC.")
	return task.TASK_SPEC


# -----------------------------------------------------------------------------
# Helpers: sanitize
# -----------------------------------------------------------------------------
def _sanitize(s: Any) -> str:
	return re.sub(r"[^0-9a-zA-Z]+", "_", str(s)).strip("_").lower()


def _sanitize_unique(names: Sequence[str]) -> Dict[str, str]:
	seen: Dict[str, int] = {}
	out: Dict[str, str] = {}
	for name in names:
		base = _sanitize(name)
		if base not in seen:
			seen[base] = 0
			out[name] = base
		else:
			seen[base] += 1
			out[name] = f"{base}_{seen[base]}"
	return out


# -----------------------------------------------------------------------------
# Metrics (same logic, moved into lib)
# -----------------------------------------------------------------------------
def roc_auc_from_scores(y, s):
	pairs = [(float(si), int(yi)) for si, yi in zip(s, y) if si is not None and yi in (0, 1)]
	if not pairs:
		return None
	scores, labels = zip(*pairs)
	if len(set(labels)) < 2:
		return None

	order = np.argsort(scores)
	ranks = np.empty(len(scores), dtype=float)
	i = 0
	while i < len(scores):
		j = i
		while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
			j += 1
		avg_rank = (i + j) / 2.0 + 1.0
		for k in range(i, j + 1):
			ranks[order[k]] = avg_rank
		i = j + 1

	labels = np.array(labels)
	n_pos = labels.sum()
	n_neg = len(labels) - n_pos
	if n_pos == 0 or n_neg == 0:
		return None
	R_pos = ranks[labels == 1].sum()
	U = R_pos - n_pos * (n_pos + 1) / 2.0
	auc = U / (n_pos * n_neg)
	return float(auc)


def _valid_pairs(y, s):
	y = np.asarray(y).ravel()
	s = np.asarray(s, dtype=float).ravel()

	n = min(len(y), len(s))
	y = y[:n]
	s = s[:n]

	mask = np.isin(y, [0, 1]) & np.isfinite(s)
	return y[mask].astype(int), s[mask]


def ap_from_scores(y, s):
	yv, sv = _valid_pairs(y, s)
	if yv.size == 0 or len(np.unique(yv)) < 2:
		return None
	return float(average_precision_score(yv, sv))


def precision_at_k(y, s, k):
	yv, sv = _valid_pairs(y, s)
	if yv.size == 0:
		return None
	n = len(sv)
	K = int(np.ceil(k * n)) if 0 < k < 1 else int(k)
	K = max(1, min(K, n))
	order = np.argsort(sv)[::-1]
	return float(np.mean(yv[order][:K]))


def recall_at_k(y, s, k):
	yv, sv = _valid_pairs(y, s)
	if yv.size == 0:
		return None
	n_pos = int(yv.sum())
	if n_pos == 0:
		return None
	n = len(sv)
	K = int(np.ceil(k * n)) if 0 < k < 1 else int(k)
	K = max(1, min(K, n))
	order = np.argsort(sv)[::-1]
	return float(yv[order][:K].sum() / n_pos)


def compute_feature_metrics(df_scores: pd.DataFrame, target_col: str) -> pd.DataFrame:
	metrics = []
	for feat, g in df_scores.groupby("feature"):
		if target_col not in g.columns:
			raise ValueError(f"Target column '{target_col}' not present in df_scores.")

		y = g[target_col].astype(int).tolist()
		s = g["score"].tolist()
		auc = roc_auc_from_scores(y, s)
		ap = ap_from_scores(y, s)

		pos = [si for si, yi in zip(s, y) if si is not None and yi == 1]
		neg = [si for si, yi in zip(s, y) if si is not None and yi == 0]
		mu_pos = float(np.mean(pos)) if pos else np.nan
		mu_neg = float(np.mean(neg)) if neg else np.nan
		delta = (mu_pos - mu_neg) if (not math.isnan(mu_pos) and not math.isnan(mu_neg)) else np.nan

		best_auc = None if auc is None else max(auc, 1 - auc)
		orientation = 0 if auc is None else (1 if auc >= 0.5 else -1)

		p_at_1pc = precision_at_k(y, s, 0.01)
		r_at_1pc = recall_at_k(y, s, 0.01)

		metrics.append({
			"feature": feat,
			"description": g["description"].iloc[0],
			"origin": g["origin"].iloc[0],
			"fn_name": g["fn_name"].iloc[0],
			"coverage": float(pd.Series(s).notna().mean()),
			"mean_pos": mu_pos,
			"mean_neg": mu_neg,
			"delta_mean": delta,
			"auc": auc,
			"best_auc": best_auc,
			"ap": ap,
			"p_at_1pct": p_at_1pc,
			"r_at_1pct": r_at_1pc,
			"orientation": orientation,
			"n": int(len(g)),
		})

	m = pd.DataFrame(metrics)
	m["abs_delta"] = m["delta_mean"].abs()
	m = m.sort_values(
		["ap", "best_auc", "abs_delta", "coverage", "n"],
		ascending=[False, False, False, False, False],
	).reset_index(drop=True)
	return m


# -----------------------------------------------------------------------------
# Compile & score features (moved into lib)
# -----------------------------------------------------------------------------
def compile_all(features, parse_prompt_row_fn, prompt_id, progress=True, df=None):
	compiled = []
	for f in features:
		try:
			fn_name, fn = compile_feature_function(f.python_src)
			f.fn_name = fn_name
			setattr(f, "_callable", fn)

			if df is not None and len(df) > 0:
				it = df.itertuples(index=False)
				if progress:
					it = tqdm(list(it), total=len(df), desc="Scoring prompts (sanity)")
				for r in it:
					tokens = parse_prompt_row_fn(r)
					_ = fn(getattr(r, prompt_id), tokens)

				compiled.append(f)
		except Exception as e:
			print(f"[WARN] Compile failed for '{getattr(f, 'label', '?')}': {e}")
	return compiled


def score_features_on_df(
	df: pd.DataFrame,
	features: list,
	*,
	parse_prompt_row_fn,
	prompt_id: str,
	warn: bool = False,
	progress: bool = True,
) -> pd.DataFrame:
	if parse_prompt_row_fn is None:
		raise ValueError("score_features_on_df requires parse_prompt_row_fn")
	if prompt_id not in df.columns:
		raise ValueError(f"Dataset must contain a canonical '{prompt_id}' column.")

	tokens_list = list(map(parse_prompt_row_fn, df.itertuples(index=False)))
	prompts = df[prompt_id].to_numpy(copy=False)

	labels, descriptions, origins, fn_names, fns = zip(
		*[(f.label, f.description, f.origin, f.fn_name, f._callable) for f in features]
	)

	n = len(prompts)
	m = len(fns)
	out_len = n * m

	out = {col: np.repeat(df[col].to_numpy(copy=False), m) for col in df.columns}
	out["feature"] = np.tile(labels, n)
	out["description"] = np.tile(descriptions, n)
	out["origin"] = np.tile(origins, n)
	out["fn_name"] = np.tile(fn_names, n)
	out["score"] = [None] * out_len

	scores = out["score"]

	it = range(n)
	if progress:
		it = tqdm(it, total=n, desc="Scoring prompts")

	for i in it:
		prompt = prompts[i]
		tokens = tokens_list[i]
		base = i * m
		for j, fn in enumerate(fns):
			try:
				raw_s = fn(prompt, tokens)
				s, sanitize_reason = _sanitize_feature_score(raw_s)
				if sanitize_reason and warn:
					print(f"[WARN] '{labels[j]}' produced {sanitize_reason}; sanitized.")
			except Exception as e:
				if warn:
					print(f"[WARN] '{labels[j]}' failed on '{prompt}': {e}")
				s = None
			scores[base + j] = s

	return pd.DataFrame(out)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def run_feature_extraction(df, *, task_spec, config):
	"""
	Runs the full pipeline and returns the final wide matrix + feature objects.
	No filesystem writes here: use save_feature_extraction_outputs() if desired.
	"""
	set_deterministic(config.seed)

	# Task-owned fields
	DEFAULT_TARGETS = task_spec.DEFAULT_TARGETS
	DEFAULT_INPUT = task_spec.DEFAULT_INPUT
	SYSTEM_PROMPT = task_spec.SYSTEM_PROMPT
	TOKENS_DICT_KEYS = task_spec.TOKENS_DICT_KEYS
	SEED_FEATURES = task_spec.SEED_FEATURES
	parse_prompt_row_fn = task_spec.parse_prompt_row

	primary_target = DEFAULT_TARGETS[0]

	if not isinstance(df, pd.DataFrame):
		raise TypeError("run_feature_extraction: df must be a pandas DataFrame.")

	# Basic cleanup (generic)
	df = (
		df.dropna(subset=[DEFAULT_INPUT])
		.drop_duplicates(subset=[DEFAULT_INPUT])
		.reset_index(drop=True)
	)

	# Sort for reproducibility
	df = df.sort_values(by=DEFAULT_INPUT, ascending=False).reset_index(drop=True)

	# Optional size cap
	if config.max_number_of_prompts_to_analyze and config.max_number_of_prompts_to_analyze > 0:
		df = df.iloc[: config.max_number_of_prompts_to_analyze].copy().reset_index(drop=True)

	if df.empty:
		raise ValueError("No prompts found after filtering.")

	# Targets must exist
	missing_targets = [t for t in DEFAULT_TARGETS if t not in df.columns]
	if missing_targets:
		raise ValueError(f"Dataset missing target columns from DEFAULT_TARGETS: {missing_targets}")

	# Stable unique id for pivoting
	if "row_id" in df.columns:
		# Avoid accidental user-provided column collision
		raise ValueError("Input df already has a 'row_id' column; rename it before running.")
	df.insert(0, "row_id", np.arange(len(df), dtype=int))

	# Contrastive pools
	pos_pool = df[df[primary_target].astype(bool)][DEFAULT_INPUT].tolist()
	neg_pool = df[~df[primary_target].astype(bool)][DEFAULT_INPUT].tolist()

	features = SEED_FEATURES[:]
	print(f"Seed features: {len(features)}")

	# ------------------------------------------------------------------
	# LLM feature proposal
	# ------------------------------------------------------------------
	if not config.no_llm_feature_generation:
		existing_for_prompt = []

		pbar = tqdm(
			total=config.feature_extraction_steps,
			desc="Feature proposal rounds",
			disable=not config.progress,
		)

		i = 0
		while i < config.feature_extraction_steps:
			random.shuffle(pos_pool)
			random.shuffle(neg_pool)
			pos_ctx = pos_pool[: min(config.num_correct_example_prompts, len(pos_pool))]
			neg_ctx = neg_pool[: min(config.num_incorrect_example_prompts, len(neg_pool))]
			sampled_existing_for_prompt = random.sample(
				existing_for_prompt,
				k=min(
					config.number_of_existing_features_to_show_for_diversity,
					len(existing_for_prompt),
				),
			)
			new_features = propose_features_contrastive(
				config.ai_model,
				pos_ctx,
				neg_ctx,
				SYSTEM_PROMPT,
				TOKENS_DICT_KEYS,
				existing_features=sampled_existing_for_prompt,
				n_features=config.n_features,
				temperature=config.temperature,
				cache_path=config.cache_dir,
			)
			if new_features:
				existing_for_prompt.extend(new_features)
				features.extend(new_features)
				i += 1
				pbar.update(1)
		pbar.close()

		if not features:
			print("[WARN] Agent returned no features; falling back to seeds.")

		features = compile_all(
			features,
			parse_prompt_row_fn=parse_prompt_row_fn,
			prompt_id=DEFAULT_INPUT,
			progress=config.progress,
			df=df.sample(min(10, len(df)), random_state=config.seed),
		)
		features = list(unique_everseen(features, key=lambda x: x.label.lower()))
	else:
		features = compile_all(
			features,
			parse_prompt_row_fn=parse_prompt_row_fn,
			prompt_id=DEFAULT_INPUT,
			progress=config.progress,
		)

	if not features:
		raise ValueError("No compilable features available.")

	# ------------------------------------------------------------------
	# 1) Score FULL dataset once and apply STRUCTURAL filters
	# ------------------------------------------------------------------
	df_scores_full = score_features_on_df(
		df,
		features,
		parse_prompt_row_fn=parse_prompt_row_fn,
		prompt_id=DEFAULT_INPUT,
		warn=config.warn_on_feature_exceptions,
		progress=config.progress,
	)

	wide_tmp = (
		df_scores_full
		.pivot(index=["row_id"], columns="feature", values="score")
		.reset_index()
	)

	# Sanitize feature columns only
	feat_cols_raw = [c for c in wide_tmp.columns if c != "row_id"]
	feat_rename = _sanitize_unique(feat_cols_raw)
	wide_tmp = wide_tmp.rename(columns=feat_rename)

	feature_columns = [feat_rename[c] for c in feat_cols_raw]
	features_count_before_dedup = len(feature_columns)

	# Drop constant features
	feature_columns = [
		c
		for c in feature_columns
		if int(wide_tmp[c].fillna(0).nunique(dropna=False)) > 1
	]

	print(
		f"Kept {len(feature_columns)} (out of {features_count_before_dedup}) "
		"compilable features after constant feature removal."
	)

	# Drop high-MAD-variance features
	if config.drop_high_mad_variance_features and feature_columns:
		mad_drop = drop_high_variance_mad(
			wide_tmp[["row_id"] + feature_columns].fillna(0),
			["row_id"],
			z_thresh=config.z_thresh,
		)
		print("high_mad_variance_features:", mad_drop)
		feature_columns = [c for c in feature_columns if c not in mad_drop]

		print(
			f"Kept {len(feature_columns)} (out of {features_count_before_dedup}) "
			"compilable features after high-MAD filtering."
		)

	# Map kept sanitized columns back to feature labels
	label_to_sanitized = {label: feat_rename[label] for label in feat_cols_raw}
	kept_sanitized = set(feature_columns)
	kept_feature_labels = {
		label for label, col in label_to_sanitized.items() if col in kept_sanitized
	}

	if kept_feature_labels:
		old_features_count = len(features)
		features = [f for f in features if f.label in kept_feature_labels]
		df_scores_full = df_scores_full[
			df_scores_full["feature"].isin(kept_feature_labels)
		]
		print(
			f"After structural deduplication: kept {len(features)} "
			f"(out of {old_features_count}) feature objects."
		)
	else:
		print(
			"Structural deduplication would drop all features; "
			"keeping all compiled features."
		)

	# ------------------------------------------------------------------
	# 2) Predictive-power filter
	# ------------------------------------------------------------------
	metrics_train = None
	if config.drop_low_predictive_power_features:
		if config.train_ratio and 0 < config.train_ratio < 1:
			df_shuf = df.sample(frac=1.0, random_state=config.seed).reset_index(
				drop=True
			)
			y_all = df_shuf[primary_target].astype(int).values
			sss = StratifiedShuffleSplit(
				n_splits=1,
				test_size=1 - config.train_ratio,
				random_state=config.seed,
			)
			train_idx, _test_idx = next(sss.split(np.zeros_like(y_all), y_all))
			df_train = df_shuf.iloc[train_idx].copy()
		else:
			df_train = df.copy()

		df_scores_train = score_features_on_df(
			df_train,
			features,
			parse_prompt_row_fn=parse_prompt_row_fn,
			prompt_id=DEFAULT_INPUT,
			warn=config.warn_on_feature_exceptions,
			progress=config.progress,
		)
		metrics_train = compute_feature_metrics(
			df_scores_train, target_col=primary_target
		)

		base_prev = float(df_train[primary_target].mean())

		feat_std = (
			df_scores_train.groupby("feature")["score"]
			.std()
			.rename("score_std")
			.reset_index()
		)
		metrics_train = metrics_train.merge(feat_std, on="feature", how="left")

		metrics_train["delta_mean_abs"] = metrics_train["delta_mean"].abs()
		metrics_train["delta_mean_std_units"] = (
			metrics_train["delta_mean_abs"]
			/ (metrics_train["score_std"].abs() + 1e-8)
		)

		use_auc = config.min_auc > 0
		use_delta = config.min_delta > 0
		use_ap = config.min_ap_above_base > -1

		cond_auc = (
			metrics_train["best_auc"].fillna(0) >= config.min_auc
		) if use_auc else False
		cond_delta = (
			metrics_train["delta_mean_std_units"] >= config.min_delta
		) if use_delta else False
		cond_ap = (
			(metrics_train["ap"] - base_prev).fillna(-1)
			>= config.min_ap_above_base
		) if use_ap else False

		keep = cond_auc | cond_delta | cond_ap
		kept_feature_names = set(metrics_train.loc[keep, "feature"].tolist())

		if kept_feature_names:
			old_features_count = len(features)
			features = [f for f in features if f.label in kept_feature_names]
			df_scores_full = df_scores_full[
				df_scores_full["feature"].isin(kept_feature_names)
			]
			print(
				f"Kept {len(features)} (out of {old_features_count}) "
				"features after predictive-power selection."
			)
		else:
			print(
				"No features met thresholds; keeping all structurally-filtered features."
			)

	# ------------------------------------------------------------------
	# 2.5) Correlation-based near-duplicate filter (after predictive power)
	# ------------------------------------------------------------------
	if config.drop_near_duplicate_features and not df_scores_full.empty:
		wide_corr = (
			df_scores_full
			.pivot(index=["row_id"], columns="feature", values="score")
			.reset_index()
		)
		feat_cols_corr = [c for c in wide_corr.columns if c != "row_id"]

		if feat_cols_corr:
			dup_drop = drop_near_duplicates_by_corr(
				wide_corr[feat_cols_corr].fillna(0),
				thresh=config.near_duplicate_features_threshold,
			)
			print("near_duplicates_by_corr:", dup_drop)

			if dup_drop:
				kept_feature_labels = set(feat_cols_corr) - set(dup_drop)
				if kept_feature_labels:
					old_features_count = len(features)
					features = [
						f for f in features if f.label in kept_feature_labels
					]
					df_scores_full = df_scores_full[
						df_scores_full["feature"].isin(kept_feature_labels)
					]
					if metrics_train is not None:
						metrics_train = metrics_train[
							metrics_train["feature"].isin(kept_feature_labels)
						].reset_index(drop=True)
					print(
						f"Kept {len(features)} (out of {old_features_count}) "
						"features after near-duplicate correlation filtering."
					)
				else:
					print(
						"Correlation-based near-duplicate filtering would drop all "
						"features; keeping predictive-power-filtered features."
					)

	# ------------------------------------------------------------------
	# 3) Build FINAL wide matrix
	# ------------------------------------------------------------------
	wide_feat = (
		df_scores_full
		.pivot(index=["row_id"], columns="feature", values="score")
		.reset_index()
	)

	feat_cols_raw = [c for c in wide_feat.columns if c != "row_id"]
	feat_rename = _sanitize_unique(feat_cols_raw)
	wide_feat = wide_feat.rename(columns=feat_rename)
	feature_columns = [feat_rename[c] for c in feat_cols_raw]

	if feature_columns:
		sanitized_wide_feat, dropped_feature_columns = _sanitize_feature_frame(
			wide_feat[feature_columns].copy(),
			context="final wide features",
		)
		wide_feat = pd.concat([wide_feat[["row_id"]], sanitized_wide_feat], axis=1)
		feature_columns = list(sanitized_wide_feat.columns)

		if dropped_feature_columns:
			kept_feature_labels = {
				label for label, col in feat_rename.items() if col in set(feature_columns)
			}
			features = [f for f in features if f.label in kept_feature_labels]
			df_scores_full = df_scores_full[
				df_scores_full["feature"].isin(kept_feature_labels)
			]
			if metrics_train is not None:
				metrics_train = metrics_train[
					metrics_train["feature"].isin(kept_feature_labels)
				].reset_index(drop=True)

	wide = df.merge(wide_feat, on="row_id", how="left")

	print(
		f"Final feature count after all filters: {len(feature_columns)} "
		f"(starting structural input: {features_count_before_dedup})."
	)

	# Put features at the end (keep task columns intact)
	base_cols = list(set(df.columns) & set(wide.columns))
	feature_columns = list(set(feature_columns) & set(wide.columns))
	wide = wide[list(set(base_cols + feature_columns))]

	if feature_columns:
		sanitized_final_wide, dropped_feature_columns = _sanitize_feature_frame(
			wide[feature_columns].copy(),
			context="final merged wide",
		)
		for col in dropped_feature_columns:
			if col in wide.columns:
				wide = wide.drop(columns=col)
		feature_columns = [c for c in feature_columns if c in sanitized_final_wide.columns]
		wide.loc[:, feature_columns] = sanitized_final_wide[feature_columns]

	wide = add_feature_stratified_is_test(
		wide,
		feature_cols=feature_columns,
		test_size=0.20,
		seed=config.seed,
		split_col="is_test",
		y_for_stratification=wide[primary_target].astype(int).to_numpy(),
	)

	return FeatureExtractionResult(
		wide=wide,
		features=features,
		feature_columns=feature_columns,
		df_scores_full=df_scores_full,
		metrics_train=metrics_train,
	)

# -----------------------------------------------------------------------------
# Optional filesystem outputs (kept separate from run_feature_extraction)
# -----------------------------------------------------------------------------
def save_feature_extraction_outputs(result, *, out_dir):
	os.makedirs(out_dir, exist_ok=True)

	scores_csv = os.path.join(out_dir, "scores.csv")
	result.wide.to_csv(scores_csv, index=False)
	print("Saved:", scores_csv)

	features_json = os.path.join(out_dir, "features.json")
	with open(features_json, "w", encoding="utf-8") as f:
		json.dump([asdict(feat) for feat in result.features], f, ensure_ascii=False, indent=2)
	print("Saved:", features_json)

	return scores_csv, features_json
