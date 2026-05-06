import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from lib.feature_extraction_runner import resolve_task_spec
from lib.modeling_and_ablation import (
	LMWrapper,
	get_device,
	sample_len_tolerant_pairs,
)
from lib.text_and_rules import (
	apply_rule_to_features,
	load_rules,
	guess_filetype,
	find_rule_files,
)
from lib.caching_and_prompting import load_or_create_cache, set_deterministic
from lib.spectral_analysis import (
    add_spectral_cli_args,
    build_reps_and_embedding_from_args,
    kcenter_farthest_first,
    representative_sample_from_global_clusters,
    greedy_spectral_cover,
    cover_and_cluster_stats_fast,
    balance_positives_and_negatives,
    summarize_distances,
)
from lib.feature_representation import safe_features_fillna

# ---------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------


def parse_args():
	p = argparse.ArgumentParser(
		description=(
			"Spectral analysis of datapoints using the *same LLM* as scripts 5 and 6 "
			"to obtain a small, representative subset per rule. "
			"Optionally (flag), compute associated/unrelated indices by similarity + length-matched pairing. "
			"The output JSON can be consumed by scripts 5 and 6."
		)
	)

	# Inputs
	p.add_argument(
		"--features_scores_dir",
		type=str,
		required=True,
		help="CSV/Parquet with prompt text + feature columns (same as script 5).",
	)
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
		help="Glob to select rule files in --rules_dir (default: association_rules_*.csv).",
	)
	p.add_argument(
		"--fake_targets",
		action="store_true",
		help=(
			"If set, replace each target column with a new random fake target (suffix: '_fake') "
			"that is re-sampled until its Pearson correlation with the true target is near zero."
		),
	)

	# LLM config (shared with scripts 5 & 6)
	p.add_argument(
		"--ai_model",
		type=str,
		required=True,
		help="HF / LMWrapper model id, e.g. 'gemma3:27b'. MUST match scripts 5 and 6.",
	)
	p.add_argument(
		"--ai_model_cache_dir",
		type=str,
		default=None,
		help="Cache dir for LM weights; should match scripts 5 and 6.",
	)

	### fixed clusters cardinality
	p.add_argument(
		"--use_global_clusters",
		action="store_true",
		help="Cluster globally in spectral space once, then sample per rule from those clusters.",
	)
	p.add_argument(
		"--global_n_clusters",
		type=int,
		default=256,
		help="Number of global clusters (k-center). This is the cardinality knob.",
	)

	### dynamic clusters cardinality
	p.add_argument(
		"--coverage_radius",
		type=float,
		default=0.5,
		help=(
			"Maximum allowed L2 distance in spectral space between any datapoint "
			"and its closest selected representative. Smaller → more points."
		),
	)
	p.add_argument(
		"--max_points_per_ablation",
		type=int,
		default=512,
		help=(
			"Hard upper bound on representatives per rule "
			"(and, if pairing flag is ON, max_pairs_per_rule)."
		),
	)
	p.add_argument(
		"--min_points_per_ablation",
		type=int,
		default=32,
		help="Minimum number of representatives per rule when enough data is available.",
	)
	p.add_argument(
		"--batch_size",
		type=int,
		default=16,
		help="Batch size for LLM forward passes.",
	)
	p.add_argument("--seed", type=int, default=0)
	p.add_argument(
		"--device",
		type=str,
		default=None,
		help="Optional device override (e.g., 'cpu', 'cuda', 'cuda:0').",
	)
	p.add_argument(
		"--output_path",
		type=str,
		required=True,
		help="Where to save the sampling plan JSON.",
	)

	# Optional: baseline subset, mirroring script 6
	p.add_argument(
		"--baseline_subset",
		type=str,
		choices=["all", "positive", "negative"],
		default="all",
		help="Whether to run on all prompts or only those where the baseline model w.r.t. the primary_target was positive or negative.",
	)

	# Pairing mode
	p.add_argument(
		"--pair_by_similarity_len_matched",
		action="store_true",
		help=(
			"If set: associated_indices/unrelated_indices are computed by ANN similarity "
			"with (token) length matching (within tolerance). "
			"This replaces the spectral cover selection."
		),
	)
	p.add_argument(
		"--pair_len_tolerance",
		type=int,
		default=0,
		help="Allowed absolute difference in token length when pairing. Default 0 = exact.",
	)
	p.add_argument(
		"--pair_similarity_metric",
		type=str,
		choices=[
			"braycurtis",
			"canberra",
			"chebyshev",
			"cityblock",
			"correlation",
			"cosine",
			"dice",
			"euclidean",
			"hamming",
			"jaccard",
			"jensenshannon",
			"kulczynski1",
			"mahalanobis",
			"matching",
			"minkowski",
			"rogerstanimoto",
			"russellrao",
			"seuclidean",
			"sokalmichener",
			"sokalsneath",
			"sqeuclidean",
			"yule",
		],
		default="cosine",
	)

	# optional + faster stats
	p.add_argument(
		"--compute_cover_stats",
		action="store_true",
		help=(
			"If set (spectral mode only), compute coverage/cluster stats. "
			"This can be expensive; default is OFF for speed."
		),
	)
	p.add_argument(
		"--stats_sample_size",
		type=int,
		default=200000,
		help=(
			"When computing stats, approximate quantiles by sampling at most this many "
			"points for percentile estimates (min/mean/std/max still exact). "
			"Set <=0 to disable sampling (exact quantiles, slower)."
		),
	)
	p.add_argument(
		"--stats_chunk_size",
		type=int,
		default=8192,
		help="Chunk size for nearest-center computations in stats (larger is faster but uses more RAM).",
	)

	# ── Task / domain config ───────────────────────────────────────────────
	g = p.add_argument_group("task")
	g.add_argument(
		"--task_module",
		default="lib.tasks.arithmetic_task",
		help="Python module path defining parse_prompt, SYSTEM_PROMPT, TOKENS_DICT_KEYS and SEED_FEATURES.",
	)

	# Attach shared spectral arguments (reused in script 5)
	add_spectral_cli_args(p)

	return p.parse_args()

def main():
	args = parse_args()
	if args.min_points_per_ablation > args.max_points_per_ablation:
		args.max_points_per_ablation = args.min_points_per_ablation
	
	set_deterministic(args.seed)

	task = resolve_task_spec(args.task_module)
	prompt_col = task.DEFAULT_INPUT
	primary_target = task.DEFAULT_TARGETS[0]
	# if args.fake_targets:
	# 	primary_target += '_fake'

	scores_path = Path(os.path.join(args.features_scores_dir, 'scores.csv')).resolve()
	rules_dir = Path(args.rules_dir).resolve()

	if not scores_path.exists():
		raise FileNotFoundError(f"Scores file not found: {scores_path}")
	if not rules_dir.exists():
		raise FileNotFoundError(f"Rules directory not found: {rules_dir}")

	ftype = guess_filetype(scores_path)
	if ftype == "parquet":
		scores_df = pd.read_parquet(scores_path)
	elif ftype == "csv":
		scores_df = pd.read_csv(scores_path)
	else:
		raise ValueError(f"Scores file must be CSV or Parquet, got {ftype}")

	# Fill NaNs in numeric columns
	scores_df = safe_features_fillna(scores_df, fill_number=0, fill_bool=False, cols_not_to_fill=task.DEFAULT_TARGETS)

	# ------------------------- Train/Test split -------------------------
	# If an 'is_test' flag is present, keep TRAIN rows only for this analysis.
	if "is_test" in scores_df.columns:
		_mask_train = ~scores_df["is_test"].astype(bool)
		n_before = len(scores_df)
		scores_df = scores_df.loc[_mask_train].reset_index(drop=True)
		print(f"[Split] TRAIN only: {len(scores_df)} / {n_before} rows (dropped {n_before - len(scores_df)} test rows)")

	# Optional baseline subset
	if args.baseline_subset != "all":
		if args.baseline_subset == "positive":
			scores_df = scores_df.loc[scores_df[primary_target] == True]#.dropna()
		else:
			scores_df = scores_df.loc[scores_df[primary_target] == False]#.dropna()
		# cols_with_na = scores_df.columns[scores_df.isna().any()]
		# if cols_with_na:
		# 	print(f"Filling NaNs with 0 in {len(cols_with_na)} columns: " + ", ".join(cols_with_na))
		# 	scores_df = scores_df.fillna(0)
		scores_df = scores_df.reset_index(drop=True)

	texts = scores_df[prompt_col].astype(str).tolist()
	n_points = len(texts)
	if n_points == 0:
		raise RuntimeError("No prompts found after filtering; nothing to analyze.")
	print(f"[Plan] Using {n_points} datapoints; prompt_col='{prompt_col}'.")

	# Device + model
	device = torch.device(args.device) if args.device is not None else get_device()

	wrapper = LMWrapper(
		model_name=args.ai_model,
		device=device,
		eval_mode=True,
		circuit_discovery=False,
		cache_dir=args.ai_model_cache_dir,
	)
	unhooked_model = getattr(wrapper, "model", None)
	hooked_model = getattr(wrapper, "hooked_model", None)
	tokenizer = getattr(wrapper, "tokenizer", None)
	if hooked_model is None or tokenizer is None:
		raise RuntimeError("LMWrapper must expose `hooked_model` and `tokenizer`.")

	# Representations cache
	emb_all, Z = build_reps_and_embedding_from_args(
		args=args,
		texts=texts,
		model=unhooked_model,
		tokenizer=tokenizer,
		device=device,
	)
	print(f"[Plan] LLM representation matrix shape: {emb_all.shape}")
	print(f"[Spectral] Spectral embedding shape: {Z.shape}")

	# Rule files
	rule_files = find_rule_files(rules_dir, args.rules_glob)
	if not rule_files:
		raise ValueError(
			f"No rule files found in {rules_dir} matching '{args.rules_glob}'."
		)

	sampling_plan = {
		"scores_path": str(scores_path),
		"prompt_col": prompt_col,
		"baseline_subset": args.baseline_subset,
		"ai_model": args.ai_model,
		"ai_model_cache_dir": args.ai_model_cache_dir,
		"spectral_space": args.spectral_space,
		"rep_hook_name": args.rep_hook_name,
		"rep_pooling": args.rep_pooling,
		"max_seq_len": args.max_seq_len,
		"spectral_dim": int(args.spectral_dim),
		"coverage_radius": float(args.coverage_radius),
		"min_points_per_ablation": int(args.min_points_per_ablation),
		"max_points_per_ablation": int(args.max_points_per_ablation),
		"pairing": {
			"max_pairs_per_rule": int(args.max_points_per_ablation),
			"len_tolerance": int(args.pair_len_tolerance),
			"method": "exact_dot_product_len_matched",
		},
		"stats": {
			"compute_cover_stats": bool(args.compute_cover_stats),
			"stats_sample_size": int(args.stats_sample_size),
			"stats_chunk_size": int(args.stats_chunk_size),
		},
		"dataset_size": len(scores_df),
		"rules": [],
	}

	global_centers_idx = None
	global_cluster_id = None
	global_meta = None
	x_norm2 = None

	if args.use_global_clusters:
		global_centers_idx, global_cluster_id, min_d2, global_meta, x_norm2 = kcenter_farthest_first(
			Z, k=args.global_n_clusters #, seed=args.seed
		)
		sizes = np.bincount(global_cluster_id, minlength=len(global_centers_idx))
		print(
			f"[GlobalClusters] K={len(global_centers_idx)} "
			f"achieved_cover_radius={global_meta['achieved_cover_radius_l2']:.4f} "
			f"mean_nn_radius={global_meta['mean_nn_radius_l2']:.4f} "
			f"min/max cluster size = {sizes.min()}/{sizes.max()}"
		)

		sampling_plan["global_clusters"] = {
			"method": "kcenter_farthest_first",
			"n_clusters": int(len(global_centers_idx)),
			"center_indices": list(map(int, global_centers_idx)),
			"achieved_cover_radius_l2": float(global_meta["achieved_cover_radius_l2"]),
			"mean_nn_radius_l2": float(global_meta["mean_nn_radius_l2"]),
			"cluster_sizes": sizes.astype(int).tolist(),
		}

	for a_rules_target, a_rules_path in rule_files:
		if args.fake_targets:
			if a_rules_target != primary_target + '_fake':
				continue
		else:
			if a_rules_target != primary_target:
				continue
		rules_df = load_rules(a_rules_path)
		print(f"[Plan] Target '{primary_target}': {len(rules_df)} rules from {a_rules_path.name}")

		for _, row in tqdm(rules_df.iterrows(), total=len(rules_df), desc='"Rules"'):
			rid = row["rule_id"]
			rule_str = row["rule"]
			coeff_sign = row["coefficient_sign"]

			print(f"  - Rule {rid} ({primary_target}): {rule_str!r}")

			mask = apply_rule_to_features(rule_str, scores_df, direction=coeff_sign)
			mask_arr = np.asarray(
				mask.values if hasattr(mask, "values") else mask, dtype=bool
			)
			if mask_arr.shape[0] != n_points:
				raise RuntimeError(
					f"apply_rule_to_features returned mask length {mask_arr.shape[0]}, "
					f"expected {n_points}."
				)

			pos_idx = np.where(mask_arr)[0]
			neg_idx = np.where(~mask_arr)[0]

			n_pos = int(pos_idx.size)
			n_neg = int(neg_idx.size)
			if n_pos == 0 or n_neg == 0:
				sampling_plan["rules"].append(
					{
						"rule_id": rid,
						"rule": rule_str,
						"rule_direction": coeff_sign,
						"rule_target": primary_target,
						"status": "skipped",
						"reason": f"Empty side for rule: n_pos={n_pos}, n_neg={n_neg}",
						"n_associated_available": n_pos,
						"n_unrelated_available": n_neg,
						"n_associated_selected": 0,
						"n_unrelated_selected": 0,
						"associated_indices": [],
						"unrelated_indices": [],
					}
				)
				print(f"[Skip] Rule {rid}: n_pos={n_pos}, n_neg={n_neg}")
				continue

			try:
				pos_idx, neg_idx = balance_positives_and_negatives(pos_idx, neg_idx, t=args.min_points_per_ablation)
			except Exception:
				sampling_plan["rules"].append(
					{
						"rule_id": rid,
						"rule": rule_str,
						"rule_direction": coeff_sign,
						"rule_target": primary_target,
						"status": "skipped",
						"reason": (
							f"Too few datapoints after balancing: "
							f"n_pos={n_pos}, n_neg={n_neg}, min_points={args.min_points_per_ablation}"
						),
						"n_associated_available": n_pos,
						"n_unrelated_available": n_neg,
						"n_associated_selected": 0,
						"n_unrelated_selected": 0,
						"associated_indices": [],
						"unrelated_indices": [],
					}
				)
				print(
					f"[Skip] Rule {rid}: n_pos={n_pos}, n_neg={n_neg}, "
					f"min_points={args.min_points_per_ablation}"
				)
				continue

			n_pos = int(pos_idx.size)
			n_neg = int(neg_idx.size)
			print(f"    positives: {n_pos}, negatives: {n_neg}")

			# maximum number of 1-to-1 pairs possible for this rule
			pair_budget = min(n_pos, n_neg)
			if pair_budget <= 0:
				sampling_plan["rules"].append(
					{
						"rule_id": rid,
						"rule": rule_str,
						"rule_direction": coeff_sign,
						"rule_target": primary_target,
						"status": "skipped",
						"reason": f"No 1-to-1 pair budget available after balancing: n_pos={n_pos}, n_neg={n_neg}",
						"n_associated_available": n_pos,
						"n_unrelated_available": n_neg,
						"n_associated_selected": 0,
						"n_unrelated_selected": 0,
						"associated_indices": [],
						"unrelated_indices": [],
					}
				)
				print(f"[Skip] Rule {rid}: no pair budget after balancing (n_pos={n_pos}, n_neg={n_neg})")
				continue
			
			# pick a target size that never exceeds what's pairable
			target_n = min(args.max_points_per_ablation, pair_budget)
			if target_n <= 0:
				sampling_plan["rules"].append(
					{
						"rule_id": rid,
						"rule": rule_str,
						"rule_direction": coeff_sign,
						"rule_target": primary_target,
						"status": "skipped",
						"reason": f"Target selection size is zero: target_n={target_n}, pair_budget={pair_budget}",
						"n_associated_available": n_pos,
						"n_unrelated_available": n_neg,
						"n_associated_selected": 0,
						"n_unrelated_selected": 0,
						"associated_indices": [],
						"unrelated_indices": [],
					}
				)
				print(f"[Skip] Rule {rid}: target_n={target_n}, pair_budget={pair_budget}")
				continue
			# target_n = max(target_n, min(args.min_points_per_ablation, pair_budget))

			# --- Associated cover via greedy spectral cover ---
			if args.use_global_clusters:
				# # Decide how many you want per rule (still uses your min/max knobs)
				# target_n = min(args.max_points_per_ablation, n_pos)
				# target_n = max(target_n, min(args.min_points_per_ablation, n_pos))
				assoc_idx, assoc_meta = representative_sample_from_global_clusters(
					Z=Z,
					x_norm2=x_norm2,
					centers_idx=global_centers_idx,
					cluster_id=global_cluster_id,
					group_idx=pos_idx,
					n_select=target_n,
					# seed=args.seed + int(rid),
				)
			else:
				assoc_idx, assoc_meta = greedy_spectral_cover(
					Z,
					pos_idx,
					radius=args.coverage_radius,
					max_points=target_n,
					min_points=target_n,
					# min_points=min(args.min_points_per_ablation, target_n),
					return_meta=True,
				)

			if len(assoc_idx) == 0:
				sampling_plan["rules"].append(
					{
						"rule_id": rid,
						"rule": rule_str,
						"rule_direction": coeff_sign,
						"rule_target": primary_target,
						"status": "skipped",
						"reason": "Associated selection returned zero datapoints.",
						"n_associated_available": n_pos,
						"n_unrelated_available": n_neg,
						"n_associated_selected": 0,
						"n_unrelated_selected": 0,
						"associated_indices": [],
						"unrelated_indices": [],
					}
				)
				print(f"[Skip] Rule {rid}: associated selection returned zero datapoints")
				continue

			if args.compute_cover_stats:
				assoc_stats = cover_and_cluster_stats_fast(
					Z=Z,
					group_idx=pos_idx,
					centers_idx=np.asarray(assoc_idx, dtype=int),
					radius=args.coverage_radius,
					chunk_size=args.stats_chunk_size,
					top_k_clusters=5,
					stats_sample_size=args.stats_sample_size,
				)
			else:
				assoc_stats = {
					"status": "skipped",
					"reason": "--compute_cover_stats not set",
				}

			rules_dict = {
				"rule_id": rid,
				"rule": rule_str,
				"rule_direction": coeff_sign,
				"rule_target": primary_target,
				"status": "ok",
				"n_associated_available": n_pos,
				"n_unrelated_available": n_neg,
				"n_associated_selected": len(assoc_idx),
				"associated_indices": assoc_idx,
				"associated_cover": {
					"greedy_meta": assoc_meta,
					"stats": assoc_stats,
				},
			}

			# --- Either length-matched pairing OR independent unrelated cover ---
			if args.pair_by_similarity_len_matched:
				pair_list, pair_meta = sample_len_tolerant_pairs(
					assoc_idx,
					neg_idx,
					emb_all,
					texts,
					tokenizer=tokenizer,
					len_tolerance=args.pair_len_tolerance,
					metric=args.pair_similarity_metric,
					metric_kwargs=None,
				)
				if pair_list:
					assoc_idx_new, unrel_idx = zip(*pair_list)
					assoc_idx_new = list(assoc_idx_new)
					unrel_idx = list(unrel_idx)
				else:
					assoc_idx_new, unrel_idx = [], []

				print(
					"assignment_pass_counts:",
					json.dumps(pair_meta["assignment_pass_counts"], indent=4),
				)

				# Update selection sizes / indices to reflect the actual pairs
				rules_dict.update(
					{
						"n_associated_selected": len(assoc_idx_new),
						"associated_indices": assoc_idx_new,
						"n_unrelated_selected": len(unrel_idx),
						"unrelated_indices": unrel_idx,
						"pairing_meta": pair_meta,
					}
				)
				assoc_idx = assoc_idx_new  # keep consistent for pair_stats below

			else:
				if args.use_global_clusters:
					unrel_idx, unrel_meta = representative_sample_from_global_clusters(
						Z=Z,
						x_norm2=x_norm2,
						centers_idx=global_centers_idx,
						cluster_id=global_cluster_id,
						group_idx=neg_idx,
						n_select=len(assoc_idx),
						# seed=args.seed + int(rid) + 1337,
					)
				else:
					unrel_idx, unrel_meta = greedy_spectral_cover(
						Z,
						neg_idx,
						radius=args.coverage_radius,
						max_points=len(assoc_idx),
						min_points=len(assoc_idx),
						return_meta=True,
					)

				if args.compute_cover_stats:
					unrel_stats = cover_and_cluster_stats_fast(
						Z=Z,
						group_idx=neg_idx,
						centers_idx=np.asarray(unrel_idx, dtype=int),
						radius=args.coverage_radius,
						chunk_size=args.stats_chunk_size,
						top_k_clusters=5,
						stats_sample_size=args.stats_sample_size,
					)
				else:
					unrel_stats = {
						"status": "skipped",
						"reason": "--compute_cover_stats not set",
					}

				rules_dict.update(
					{
						"n_unrelated_selected": len(unrel_idx),
						"unrelated_indices": unrel_idx,
						"unrelated_cover": {
							"greedy_meta": unrel_meta,
							"stats": unrel_stats,
						},
					}
				)

			# --- Pair similarity stats (after selection) ---
			assoc_idx_arr = np.asarray(rules_dict["associated_indices"], dtype=int)
			unrel_idx_arr = np.asarray(rules_dict["unrelated_indices"], dtype=int)

			A = emb_all[assoc_idx_arr]
			B = emb_all[unrel_idx_arr]

			if A.shape[0] == 0 or B.shape[0] == 0:
				rules_dict.update(
					{
						"status": "skipped",
						"reason": f"No data left after selection to form at least one pair: associated={A.shape[0]}, unrelated={B.shape[0]}",
						"n_associated_selected": 0,
						"n_unrelated_selected": 0,
						"associated_indices": [],
						"unrelated_indices": [],
					}
				)
				sampling_plan["rules"].append(rules_dict)
				print(
					f"[Skip] Rule {rid}: no data left after selection to form at least one pair "
					f"(associated={A.shape[0]}, unrelated={B.shape[0]})"
				)
				continue

			if A.shape[0] != B.shape[0]:
				rules_dict.update(
					{
						"status": "skipped",
						"reason": f"Not enough data for pairing. Need 1-to-1 pairing, got associated data A={A.shape[0]} rows, unassociated data B={B.shape[0]} rows",
						"n_associated_selected": 0,
						"n_unrelated_selected": 0,
						"associated_indices": [],
						"unrelated_indices": [],
					}
				)
				sampling_plan["rules"].append(rules_dict)
				print(f"Not enough data for pairing. Need 1-to-1 pairing, got associated data A={A.shape[0]} rows, unassociated data B={B.shape[0]} rows")
				continue

			# min_len = min(A.shape[0], B.shape[0])
			# if min_len == 0:
			# 	rules_dict.update(
			# 		{
			# 			"status": "skipped",
			# 			"reason": "No data left after selection to form at least one pair.",
			# 			"n_associated_selected": 0,
			# 			"n_unrelated_selected": 0,
			# 			"associated_indices": [],
			# 			"unrelated_indices": [],
			# 		}
			# 	)
			# 	sampling_plan["rules"].append(rules_dict)
			# 	print("No data left after selection to form at least one pair.")
			# 	continue
			# # Trim both sides to same length
			# assoc_idx_arr = assoc_idx_arr[:min_len]
			# unrel_idx_arr = unrel_idx_arr[:min_len]
			
			rules_dict.update({
				"n_associated_selected": assoc_idx_arr.shape[0],
				"associated_indices": assoc_idx_arr.tolist(),
				"n_unrelated_selected": unrel_idx_arr.shape[0],
				"unrelated_indices": unrel_idx_arr.tolist(),
			})

			A = emb_all[assoc_idx_arr]
			B = emb_all[unrel_idx_arr]
			# print(f"    positives: {A.shape[0]}, negatives: {B.shape[0]}")
			pair_dist = np.linalg.norm(A - B, axis=1).astype(np.float32)
			rules_dict["pair_stats"] = {"euclidean_dist": summarize_distances(pair_dist)}
			sampling_plan["rules"].append(rules_dict)

	out_path = Path(args.output_path).resolve()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(sampling_plan, indent=4))
	print(f"[Plan] Saved sampling plan to: {out_path}")


if __name__ == "__main__":
	main()
