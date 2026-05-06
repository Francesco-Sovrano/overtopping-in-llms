import os
os.environ["PYTORCH_MPS_PREFER_METAL"] = "1"
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

import json
import re

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap

from lib.caching_and_prompting import load_or_create_cache, set_deterministic
from lib.data_model_for_shap import run_rule_extraction

from lib.ruleshap import RuleSHAP
from lib.feature_extraction_runner import resolve_task_spec
from lib.feature_representation import safe_features_fillna

################################################################

import argparse

def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Run rule extraction and SHAP analysis on scores.csv produced by the "
			"previous step. Targets (labels) and features are configurable so this "
			"works across different tasks."
		)
	)

	# --- paths ---
	parser.add_argument(
		"--features_scores_dir",
		type=str,
		required=True,
		help="Directory containing scores.csv and features.json (output of the previous script)."
	)
	parser.add_argument(
		"--rules_dir",
		type=str,
		default="xai_analyses_results",
		help="Base output directory"
	)
	
	# --- others ---
	parser.add_argument(
		"--random_seed",
		type=int,
		default=42,
		help="Specify the random seed (integer)"
	)

	# Control: generate intentionally incorrect rules by using random, uninformative fake targets.
	# This creates a new target column per real target (suffix: '_fake') and then runs rule extraction on the fake ones.
	parser.add_argument(
		"--fake_targets",
		action="store_true",
		help=(
			"If set, replace each target column with a new random fake target (suffix: '_fake') "
			"that is re-sampled until its Pearson correlation with the true target is near zero."
		),
	)
	parser.add_argument(
		"--fake_target_max_abs_corr",
		type=float,
		default=0.05,
		help="Maximum allowed absolute Pearson correlation between true and fake target before re-sampling.",
	)
	parser.add_argument(
		"--fake_target_max_tries",
		type=int,
		default=500,
		help="Maximum re-sampling attempts to achieve the desired low correlation.",
	)


	parser.add_argument(
		"--fast_shap_estimate",
		action="store_true",
		help="Set this flag to compute Shapley values in a faster, approximated way when len(input_features) < 20"
	)
	parser.add_argument(
		"--npermutations",
		type=int,
		default=10,
	)
	parser.add_argument(
		"--only_unique_datapoints_in_shap",
		action="store_true",
		help="Drop duplicated datapoints"
	)
	parser.add_argument(
		"--epsilon",
		type=float,
		default=1e-1,
		help="The larger is epsilon, the less likely the abstracted model used for SHAP will sample the same x provided in input."
	)

	# Boolean inputs
	parser.add_argument(
		"--use_shap_in_xgb",
		action="store_true",
		help="Set this flag to use SHAP in XGB"
	)
	parser.add_argument(
		"--use_shap_in_lasso",
		action="store_true",
		help="Set this flag to use SHAP in Lasso"
	)

	parser.add_argument(
		"--only_unique_datapoints_in_rule_extraction",
		action="store_true",
		help="Drop duplicated datapoints"
	)

	# ── Task / domain config ───────────────────────────────────────────────
	g = parser.add_argument_group("task")
	g.add_argument(
		"--task_module",
		default="lib.tasks.arithmetic_task",
		help="Python module path defining parse_prompt, SYSTEM_PROMPT, TOKENS_DICT_KEYS and SEED_FEATURES.",
	)

	return parser.parse_args()

args = parse_args()
use_shap_in_xgb, use_shap_in_lasso = args.use_shap_in_xgb, args.use_shap_in_lasso
features_scores_dir, random_seed, fast_shap_estimate = args.features_scores_dir, args.random_seed, args.fast_shap_estimate
data = os.path.join(features_scores_dir, 'scores.csv')
set_deterministic(random_seed)

task = resolve_task_spec(args.task_module)

# --- derived output paths ---
out_dir = args.rules_dir
summary_plot_dir = os.path.join(out_dir, "summary_plot")

os.makedirs(out_dir, exist_ok=True)
os.makedirs(summary_plot_dir, exist_ok=True)

################################################################

df = pd.read_csv(data)

# Targets to explain (task-dependent; configurable via --targets)
metrics_list = task.DEFAULT_TARGETS
missing_targets = [t for t in metrics_list if t not in df.columns]
if missing_targets:
	raise ValueError(f"Target column(s) not found in scores.csv: {missing_targets}")

df = safe_features_fillna(df, fill_number=0, fill_bool=False, cols_not_to_fill=task.DEFAULT_TARGETS)

# Optional control: create random fake targets that are (by construction) uninformative about the features.
# We re-sample until the fake target has low (near-zero) correlation with the true target.
if args.fake_targets:
	rng = np.random.default_rng(args.random_seed)
	fake_report = {}
	fake_targets = []
	for t in list(metrics_list):
		y_true = df[t].to_numpy()
		# Keep NaNs where present to avoid changing dataset cardinality/semantics
		mask = ~pd.isna(y_true)
		if mask.sum() < 2:
			raise ValueError(f"Not enough non-NaN values in target '{t}' to create a fake target.")

		# Detect binary targets (0/1) robustly
		unique_vals = set(pd.unique(df.loc[mask, t]))
		is_binary = unique_vals.issubset({0, 1, 0.0, 1.0, True, False})
		fake_name = f"{t}_fake"

		best_corr = None
		best_fake = None
		tries = 0
		for tries in range(1, int(args.fake_target_max_tries) + 1):
			if is_binary:
				p = float(np.nanmean(df.loc[mask, t].astype(float)))
				y_fake = (rng.random(len(df)) < p).astype(int).astype(float)
				# preserve NaN pattern
				y_fake[~mask] = np.nan
			else:
				# Numeric/continuous target: bootstrap from empirical marginal distribution (with replacement)
				base = df.loc[mask, t].astype(float).to_numpy()
				y_fake = np.full(len(df), np.nan, dtype=float)
				y_fake[mask] = rng.choice(base, size=mask.sum(), replace=True)

			# Pearson correlation on non-NaN positions
			corr = float(np.corrcoef(df.loc[mask, t].astype(float).to_numpy(), y_fake[mask])[0, 1])
			if np.isnan(corr):
				# can happen if target is constant; accept and move on (correlation is undefined but effectively uninformative)
				best_corr, best_fake = corr, y_fake
				break
			if (best_corr is None) or (abs(corr) < abs(best_corr)):
				best_corr, best_fake = corr, y_fake
			if abs(corr) <= float(args.fake_target_max_abs_corr):
				break

		# Commit the best attempt (guaranteed to exist)
		df[fake_name] = best_fake
		fake_targets.append(fake_name)

		# Extra diagnostics: agreement rate (useful for binary targets)
		agree = None
		if is_binary:
			agree = float((df.loc[mask, t].astype(int).to_numpy() == np.nan_to_num(best_fake[mask]).astype(int)).mean())
		fake_report[t] = {
			"fake_column": fake_name,
			"is_binary": bool(is_binary),
			"pearson_corr_true_vs_fake": None if np.isnan(best_corr) else float(best_corr),
			"abs_pearson_corr_true_vs_fake": None if np.isnan(best_corr) else float(abs(best_corr)),
			"tries": int(tries),
			"agreement_rate": agree,
			"seed": int(args.random_seed),
			"max_abs_corr_target": float(args.fake_target_max_abs_corr),
		}
		print(f"[fake_targets] {t} -> {fake_name} | pearson_r={fake_report[t]['pearson_corr_true_vs_fake']} | tries={tries} | agreement={agree}")

	# Switch rule extraction targets to the fake ones
	metrics_list = fake_targets

	# Persist a small report for reproducibility
	os.makedirs(out_dir, exist_ok=True)
	with open(os.path.join(out_dir, "fake_targets_report.json"), "w", encoding="utf-8") as f:
		json.dump(fake_report, f, indent=2)


# Derive feature columns from features.json produced by the previous script
features_json_path = os.path.join(features_scores_dir, "features.json")
if not os.path.exists(features_json_path):
	raise FileNotFoundError(
		f"features.json not found in {features_scores_dir} (expected from the previous step)."
	)

with open(features_json_path, "r", encoding="utf-8") as f:
	features_meta = json.load(f)

def _sanitize(s: str) -> str:
	return re.sub(r"[^0-9a-zA-Z]+", "_", str(s)).strip("_").lower()

# Column names in scores.csv that correspond to feature labels
feature_name_set = {
	_sanitize(feat["label"])
	for feat in features_meta
}

input_features = [
	c for c in df.columns
	if c in feature_name_set
	and c not in metrics_list
]

print("metrics_list:", metrics_list)
print("input_features:", input_features)

# --- train/eval split: fit rules on train only; score rules on test only ---
if 'is_test' in df.columns:
	_is_test = df['is_test'].astype(bool).to_numpy()
	df_train = df.loc[~_is_test].reset_index(drop=True)
	df_test  = df.loc[_is_test].reset_index(drop=True)
	print(f"[Split] train={len(df_train)} test={len(df_test)} (test%={len(df_test)/max(1,len(df)):.2%})")
	if len(df_train) == 0 or len(df_test) == 0:
		raise ValueError(f"Invalid split: train={len(df_train)} test={len(df_test)}. Check 'is_test' flag.")
else:
	# Backward-compatible: no split flag -> use all data for both fit and evaluation
	df_train = df
	df_test = None
	print("[Split] 'is_test' column not found; using full dataset for fit/eval")

run_rule_extraction(
	df_train, 
	input_features=input_features, 
	targets=metrics_list, 
	args=args,
	# df_eval=df_test,
	df_eval=df_train, # Since this isn't the end of the pipeline, using the test set here could leak test information into the rule-ensemble training process.
	use_lasso_regression=False,
)
