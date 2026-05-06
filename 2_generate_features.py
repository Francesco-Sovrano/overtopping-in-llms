import argparse
import pandas as pd
from lib.feature_extraction_runner import (
    FeatureExtractionConfig,
    resolve_task_spec,
    run_feature_extraction,
    save_feature_extraction_outputs,
)

# ---------------------------
# CLI
# ---------------------------
def parse_args(argv=None):
	p = argparse.ArgumentParser(
		description="Correctness-driven feature proposal & testing (task-agnostic).",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)

	# ── Data ────────────────────────────────────────────────────────────────
	g = p.add_argument_group("data")
	g.add_argument(
		"--prompts_answers_pkl_file", "-i",
		required=True,
		help="Input pickle produced by task data-generation (task decides structure).",
	)
	
	# ── Output / cache ──────────────────────────────────────────────────────
	g = p.add_argument_group("output")
	g.add_argument(
		"--features_scores_dir", "--out_dir", "-o",
		default="feature_report",
		help="Output directory (writes scores.csv and features.json).",
	)
	g.add_argument(
		"--cache_dir", "-c",
		default="./cache",
		help="Cache directory for LLM proposals / intermediate artifacts.",
	)

	# ── LLM feature proposal ────────────────────────────────────────────────
	g = p.add_argument_group("llm")
	g.add_argument(
		"--ai_model", "-m",
		default="qwen3:30b",
		help="Model used to propose Python feature functions.",
	)
	g.add_argument(
		"--temperature", "-T",
		type=float,
		default=0.3,
		help="LLM sampling temperature.",
	)
	g.add_argument(
		"--n_features", "-n",
		type=int,
		default=16,
		help="Features requested per proposal round.",
	)
	g.add_argument(
		"--feature_extraction_steps", "--rounds", "-R",
		type=int,
		default=10,
		help="Number of proposal rounds (each round samples new contexts).",
	)
	g.add_argument(
		"--num_correct_example_prompts",
		type=int,
		default=32,
		help="Number of POSITIVE prompts shown to the agent per round.",
	)
	g.add_argument(
		"--num_incorrect_example_prompts",
		type=int,
		default=32,
		help="Number of NEGATIVE prompts shown to the agent per round.",
	)
	g.add_argument(
		"--no_llm_feature_generation", "--no_llm",
		dest="no_llm_feature_generation",
		action="store_true",
		help="Skip LLM proposals; use only predefined seed features.",
	)
	g.add_argument(
		"--number_of_existing_features_to_show_for_diversity",
		type=int,
		default=20,
		help="How many previously proposed features to show back to the agent to encourage diversity.",
	)

	# ── Feature filtering / selection ───────────────────────────────────────
	g = p.add_argument_group("feature filtering")
	g.add_argument(
		"--drop_low_predictive_power_features", "--filter_by_metrics",
		dest="drop_low_predictive_power_features",
		action="store_true",
		help="Enable metric-based feature selection on a train split using --min_auc/--min_delta/--min_ap_above_base.",
	)
	g.add_argument(
		"--drop_near_duplicate_features", "--drop_near_dups",
		dest="drop_near_duplicate_features",
		action="store_true",
		help="Drop near-duplicate feature columns using feature–feature correlation on wide scores.",
	)
	g.add_argument(
		"--near_duplicate_features_threshold", "--dup_corr_thresh",
		dest="near_duplicate_features_threshold",
		type=float,
		default=0.9999,
		help="Correlation threshold for near-duplicate dropping (higher = more aggressive).",
	)
	g.add_argument(
		"--drop_high_mad_variance_features", "--drop_high_mad",
		dest="drop_high_mad_variance_features",
		action="store_true",
		help="Drop extreme high-variance features using a MAD-based outlier heuristic.",
	)

	# ── Train/test split + metric thresholds (used only if metric filtering enabled) ──
	g = p.add_argument_group("metric thresholds (train split)")
	g.add_argument(
		"--train_ratio", "-t",
		type=float,
		default=1,
		help="Train/hold-out split ratio used for metric-based filtering; set to 0 or 1 to disable split.",
	)
	g.add_argument(
		"--min_auc",
		type=float,
		default=0.0,
		help="Keep if best_auc >= this (best_auc = max(auc, 1-auc)) on train.",
	)
	g.add_argument(
		"--min_delta",
		type=float,
		default=0.2,
		help="Or keep if |mean_pos - mean_neg| >= this (in std units) on train.",
	)
	g.add_argument(
		"--min_ap_above_base",
		type=float,
		default=0.0,
		help="Or keep if (AP - base prevalence) >= this on train.",
	)
	g.add_argument(
		"--z_thresh",
		type=float,
		default=10,
		help="Drop extreme high-variance features using a MAD-based outlier heuristic.",
	)

	# ── Run control ─────────────────────────────────────────────────────────
	g = p.add_argument_group("run")
	g.add_argument(
		"--max_number_of_prompts_to_analyze", "--max_prompts",
		type=int,
		default=0,
		help="Limit dataset size; 0 means all prompts.",
	)
	g.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed (also used for split shuffling).",
	)

	# ── Task / domain config ───────────────────────────────────────────────
	g = p.add_argument_group("task")
	g.add_argument(
		"--task_module",
		default="lib.tasks.arithmetic_task",
		help="Python module path defining DEFAULT_TARGETS, parse_prompt, SYSTEM_PROMPT, TOKENS_DICT_KEYS and SEED_FEATURES, plus load_dataset_from_cache().",
	)

	return p.parse_args(argv)

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    args = parse_args()

    task_spec = resolve_task_spec(args.task_module)

    # Load dataset (task owns structure)
    df = task_spec.load_dataset_from_cache(args.prompts_answers_pkl_file)
    if not isinstance(df, pd.DataFrame):
        raise SystemExit("task.load_dataset_from_cache() must return a pandas DataFrame.")

    cfg = FeatureExtractionConfig(
        ai_model=args.ai_model,
        temperature=args.temperature,
        n_features=args.n_features,
        feature_extraction_steps=args.feature_extraction_steps,
        num_correct_example_prompts=args.num_correct_example_prompts,
        num_incorrect_example_prompts=args.num_incorrect_example_prompts,
        no_llm_feature_generation=args.no_llm_feature_generation,
        number_of_existing_features_to_show_for_diversity=args.number_of_existing_features_to_show_for_diversity,
        cache_dir=args.cache_dir,
        drop_low_predictive_power_features=args.drop_low_predictive_power_features,
        drop_near_duplicate_features=args.drop_near_duplicate_features,
        near_duplicate_features_threshold=args.near_duplicate_features_threshold,
        drop_high_mad_variance_features=args.drop_high_mad_variance_features,
        train_ratio=args.train_ratio,
        min_auc=args.min_auc,
        min_delta=args.min_delta,
        min_ap_above_base=args.min_ap_above_base,
        z_thresh=args.z_thresh,
        max_number_of_prompts_to_analyze=args.max_number_of_prompts_to_analyze,
        seed=args.seed,
        progress=True,
        warn_on_feature_exceptions=True,
    )

    result = run_feature_extraction(df, task_spec=task_spec, config=cfg)
    save_feature_extraction_outputs(result, out_dir=args.features_scores_dir)
