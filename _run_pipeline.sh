#!/usr/bin/env bash
# set -euo pipefail

############################################
# Usage:
#   ./_run_pipeline.sh <EXPERIMENT_NAME> <ANALYZED_LLM> \
#     [--spectral_splits] \
#     [--fast_anchoring|--slow_anchoring] \
#     [--spectral_anchoring_plan|--random_anchoring_plan] \
#     [--spectral_circuit_discovery|--random_circuit_discovery] \
#     [--eval_intervention <NAME>] \
#     [--z_thresh <N>] \
#     [--max_number_of_circuits_to_analyze <N>] \
#     [--batch_size <N>] \
#     [--circuit_level <neuron|edge>] \
#     [--circuit_size <N>] \
#     [--min_flip_rate <R>]
#
# Defaults:
#   --random_anchoring_plan --random_circuit_discovery --fast_anchoring --eval_intervention mean-positional --z_thresh -1 --max_number_of_circuits_to_analyze -1 --batch_size 256 --circuit_level neuron --circuit_size 100000 --min_flip_rate 0.2
#
# Constraints:
#   - Spectral vs Random PLAN are mutually exclusive
#   - Spectral vs Random DISCOVERY are mutually exclusive
#   - Fast vs Slow ANCHORING are mutually exclusive
#   - --spectral_circuit_discovery requires --spectral_anchoring_plan (because circuit discovery needs a sampling plan)
############################################

usage() {
	cat <<EOF
Usage:
	$0 <EXPERIMENT_NAME> <ANALYZED_LLM> [options]

Options (mutually exclusive within each group):
	Feature filtering:
		--z_thresh <N>     Z-score threshold for MAD-variance feature filtering
									 (used with --drop_high_mad_variance_features). Default: -1

	Circuit analysis:
		--max_number_of_circuits_to_analyze <N>
							Max number of circuits/rules to analyze in 5_discover_circuits.py.
							Default: -1 (all circuits)
		--eval_intervention <NAME>
							Intervention used in scripts 5, 6, and 7.
							Default: mean-positional
		--batch_size <N>   Batch size used by the pipeline stages that support batching.
							Default: 256
		--circuit_level <neuron|edge>
							Circuit granularity for 5_discover_circuits.py.
							Default: neuron
		--circuit_size <N>
							Circuit size passed to 5_discover_circuits.py.
							Default: 100000
		--min_flip_rate <R>
							Minimum flip-rate threshold (tau) used in scripts 6 and 7.
							Default: 0.2
	Plan:
		--spectral_anchoring_plan
		--random_anchoring_plan

	Circuit discovery:
		--spectral_circuit_discovery
		--random_circuit_discovery

	Neuron anchoring:
		--fast_anchoring
		--slow_anchoring

	Split mode:
		--spectral_splits   Use spectral-cluster splits (no rule-based splits).
							Skips scripts 3 and 4; uses --cluster_by_spectral in scripts 5 and 6.

	Rule control:
		--incorrect_rules   Use random fake targets before rule extraction (Script 3) to generate intentionally incorrect rules
							(re-sampled until abs(Pearson corr) is near 0).

	Neuron selection:
		--mlp_neurons_only  Restrict neuron analysis to MLP neurons only.

	Decode control:
		--decode_only       Pass --decode_only through to scripts that support it (default: off).

Examples:
	# Random discovery + Random plan + Fast anchoring (default)
	$0 myexp qwen2:7b

	# Random discovery + Spectral plan + Fast anchoring
	$0 myexp qwen2:7b --spectral_anchoring_plan --random_circuit_discovery --fast_anchoring

	# Spectral discovery (requires spectral plan) + Fast anchoring
	$0 myexp qwen2:7b --spectral_anchoring_plan --spectral_circuit_discovery --fast_anchoring

	# Spectral discovery + Slow anchoring
	$0 myexp qwen2:7b --spectral_anchoring_plan --spectral_circuit_discovery --slow_anchoring

	# Spectral splits (cluster-based), fast anchoring
	$0 myexp qwen2:7b --spectral_splits --fast_anchoring

	# Override batching, circuit discovery granularity, and tau
	$0 myexp qwen2:7b --batch_size 32 --circuit_level edge --circuit_size 50000 --min_flip_rate 0.3
EOF
}

if [[ $# -lt 2 ]]; then
	usage
	exit 1
fi

# Activate env
. .env/bin/activate

# export HF_HUB_OFFLINE=1

EXPERIMENT_NAME="$1"
ANALYZED_LLM="$2"
shift 2

############################################
# Defaults
SPLITS="rules"         # rules|spectral
PLAN="random"          # random|spectral
DISCOVERY="random"     # random|spectral
ANCHORING="fast"       # fast|slow
Z_THRESH="-1"            # MAD z-threshold for script 2 (drop_high_mad_variance_features)
MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE="-1" # -1 means ALL CIRCUITS
EVAL_INTERVENTION="mean-positional"
BATCH_SIZE="256"
CIRCUIT_LEVEL="neuron"
CIRCUIT_SIZE="100000"
MIN_FLIP_RATE="0.2"
INCORRECT_RULES=false
DECODE_ONLY=false
NEURONS_TYPE="all"

# Parse boolean-style flags
while [[ $# -gt 0 ]]; do
	case "$1" in
		--z_thresh)
			[[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value (e.g. --z_thresh 25)"; exit 1; }
			Z_THRESH="$2"
			shift 2
			;;
		--max_number_of_circuits_to_analyze)
			[[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value (e.g. --max_number_of_circuits_to_analyze 10)"; exit 1; }
			MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE="$2"
			shift 2
			;;
		--batch_size)
			[[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value (e.g. --batch_size 32)"; exit 1; }
			BATCH_SIZE="$2"
			shift 2
			;;
		--circuit_level)
			[[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value (e.g. --circuit_level neuron)"; exit 1; }
			CIRCUIT_LEVEL="$2"
			shift 2
			;;
		--circuit_size)
			[[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value (e.g. --circuit_size 100000)"; exit 1; }
			CIRCUIT_SIZE="$2"
			shift 2
			;;
		--min_flip_rate)
			[[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value (e.g. --min_flip_rate 0.2)"; exit 1; }
			MIN_FLIP_RATE="$2"
			shift 2
			;;
		--eval_intervention)
			[[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value (e.g. --eval_intervention mean-positional)"; exit 1; }
			EVAL_INTERVENTION="$2"
			shift 2
			;;
		--spectral_splits)             SPLITS="spectral"; shift ;;
		--spectral_anchoring_plan)      PLAN="spectral"; shift ;;
		--random_anchoring_plan)        PLAN="random"; shift ;;
		--spectral_circuit_discovery)   DISCOVERY="spectral"; shift ;;
		--random_circuit_discovery)     DISCOVERY="random"; shift ;;
		--fast_anchoring)               ANCHORING="fast"; shift ;;
		--slow_anchoring)               ANCHORING="slow"; shift ;;
		--incorrect_rules)              INCORRECT_RULES=true; shift ;;
		--mlp_neurons_only)						 NEURONS_TYPE='mlp'; shift ;;
		--decode_only)                 DECODE_ONLY=true; shift ;;
		-h|--help)                      usage; exit 0 ;;
		*) echo "Unknown option: $1"; usage; exit 1 ;;
	esac
done

# Enforce constraints for spectral splits (keeps semantics unambiguous)
if [[ "$SPLITS" == "spectral" ]]; then
	if [[ "$INCORRECT_RULES" == "true" ]]; then
		echo "ERROR: --incorrect_rules is incompatible with --spectral_splits (no rule extraction in spectral_splits mode)."
		exit 1
	fi

	if [[ "$PLAN" != "random" || "$DISCOVERY" != "random" ]]; then
		echo "ERROR: --spectral_splits is mutually exclusive with plan/discovery flags."
		echo "       Spectral splits bypass rule-based plans and rule-based discovery."
		exit 1
	fi
fi

# # Enforce constraints
# if [[ "$DISCOVERY" == "spectral" && "$PLAN" != "spectral" ]]; then
# 	echo "ERROR: --spectral_circuit_discovery requires --spectral_anchoring_plan (sampling plan needed for circuit discovery)."
# 	exit 1
# fi

echo "=== CONFIG ==="
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "ANALYZED_LLM:    $ANALYZED_LLM"
echo "SPLITS:          $SPLITS"
echo "PLAN:            $PLAN"
echo "DISCOVERY:       $DISCOVERY"
echo "ANCHORING:       $ANCHORING"
echo "INCORRECT_RULES: $INCORRECT_RULES"
echo "NEURONS_TYPE:    $NEURONS_TYPE"
echo "BATCH_SIZE:      $BATCH_SIZE"
echo "CIRCUIT_LEVEL:   $CIRCUIT_LEVEL"
echo "CIRCUIT_SIZE:    $CIRCUIT_SIZE"
echo "MIN_FLIP_RATE:   $MIN_FLIP_RATE"
echo "Z_THRESH:        $Z_THRESH"
echo "EVAL_INTERVENTION: $EVAL_INTERVENTION"
echo "MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE: $MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE"
echo "==============="

############################################
# Common config
DATA_DIR="./data/$EXPERIMENT_NAME/$ANALYZED_LLM"
CACHE_DIR="./cache/$EXPERIMENT_NAME"
EXPERIMENT_LLM_CACHE_DIR="$CACHE_DIR/$ANALYZED_LLM"

# Output variant suffix (keeps fake-target outputs separate)
VARIANT_SUFFIX=""
FAKE_FLAG=()
if [[ "$INCORRECT_RULES" == "true" ]]; then
	VARIANT_SUFFIX="_fake_targets"
	FAKE_FLAG=(--fake_targets)
fi

NEURONS_TYPE_FLAG=()
if [[ "$NEURONS_TYPE" == "mlp" ]]; then
	NEURONS_TYPE_FLAG=(--mlp_neurons_only)
fi

TASK_MODULE="lib.tasks.${EXPERIMENT_NAME}_task"
FEATURE_GENERATION_LLM="gemma3:27b"
PROMPTS_ANSWERS_PKL_FILE="$EXPERIMENT_LLM_CACHE_DIR/llm_io_data.pkl"
FEATURES_SCORES_DIR="$DATA_DIR/feature_report"
PAIR_SIMILARITY_METRIC="euclidean"

#####################################
# CIRCUIT_DISCOVERY_METHOD=EAP
# CIRCUIT_INTERVENTION=zero
CIRCUIT_DISCOVERY_METHOD=EAP-IG-inputs
CIRCUIT_INTERVENTION=patching

# lowercase + replace '-' with '_'
CIRCUIT_DISCOVERY_METHOD_NORM=$(
  printf '%s' "$CIRCUIT_DISCOVERY_METHOD" | tr '[:upper:]-' '[:lower:]_'
)
CIRCUIT_DISCOVERY_OUTPUT_DIR="$DATA_DIR/neural_circuit_discovery_results${VARIANT_SUFFIX}/$CIRCUIT_DISCOVERY_METHOD_NORM"

RULES_DIR="$DATA_DIR/rule_extraction_results"
POINTS_TO_USE_FOR_MEAN_ABLATION=256
MAX_POINTS_PER_CIRCUIT=128
MAX_POINTS_PER_ABLATION=64
# SPECTRAL_CLUSTERS=256


# if [[ "$EVAL_INTERVENTION" == "mean-positional" ]]; then
# 	if [[ "$DECODE_ONLY" == "true" ]]; then
# 		EVAL_INTERVENTION=mean
# 	fi
# fi
if [[ "$EVAL_INTERVENTION" == *donor* && "$POINTS_TO_USE_FOR_MEAN_ABLATION" -lt 2048 ]]; then
	POINTS_TO_USE_FOR_MEAN_ABLATION=2048
fi

# Script 5 still expects the older mean intervention names during evaluation.
# Normalize donor-style eval interventions only for circuit discovery.
# Downstream output folders include the effective eval intervention whenever it differs from `mean`.
SCRIPT5_EVAL_INTERVENTION="$EVAL_INTERVENTION"
if [[ "$SCRIPT5_EVAL_INTERVENTION" == "mean-donor" ]]; then
	SCRIPT5_EVAL_INTERVENTION="mean"
elif [[ "$SCRIPT5_EVAL_INTERVENTION" == "mean-donor-positional" ]]; then
	SCRIPT5_EVAL_INTERVENTION="mean-positional"
fi

OUTPUT_EVAL_INTERVENTION_SUFFIX=""
if [[ "$EVAL_INTERVENTION" != "mean" && "$EVAL_INTERVENTION" != "mean-positional" ]]; then
	OUTPUT_EVAL_INTERVENTION_SUFFIX="-eval_${EVAL_INTERVENTION}"
fi

CIRCUIT_LABEL=""
if [[ "${SPLITS:-}" == "spectral" ]]; then
	CIRCUIT_LABEL+="spectral_split"
else
	CIRCUIT_LABEL+="rule_split"
	CIRCUIT_LABEL+="-${DISCOVERY:-}_sample"
fi
if [[ "$CIRCUIT_SIZE" != "100000" ]]; then
	CIRCUIT_LABEL+="-M${CIRCUIT_SIZE}"
fi
DECODE_FLAG=()
if [[ "$DECODE_ONLY" == "true" ]]; then
	DECODE_FLAG=(--decode_only)
	CIRCUIT_LABEL+="-decode_only"
fi
if [[ -n "$OUTPUT_EVAL_INTERVENTION_SUFFIX" ]]; then
	CIRCUIT_LABEL+="$OUTPUT_EVAL_INTERVENTION_SUFFIX"
fi
echo $CIRCUIT_LABEL
DISCOVERY_OUT_DIR="$CIRCUIT_DISCOVERY_OUTPUT_DIR/$CIRCUIT_LABEL"

# Shared spectral flags (used in multiple calls)
SPECTRAL_FLAGS=(
	--spectral_space hidden
	--rep_hook_name ln_final.hook_normalized
	--rep_pooling last
	--spectral_dim 32
)

############################################
# Steps 1-3: always run
ollama serve > ollama.log 2>&1 &

python3 1_generate_prompts_and_answers.py \
	--ai_model "$ANALYZED_LLM" \
	--task_module "$TASK_MODULE" \
	--prompts_answers_pkl_file "$PROMPTS_ANSWERS_PKL_FILE" \
	--batch_size "$BATCH_SIZE" \
	--stats_json_out $FEATURES_SCORES_DIR

# Step 2: generate features
if [[ -d "$FEATURES_SCORES_DIR" ]] && find "$FEATURES_SCORES_DIR" -type f -name '*.csv' -print -quit | grep -q .; then
	echo "Step 2: found existing CSV(s) in $FEATURES_SCORES_DIR -> skipping 2_generate_features.py"
else
	# echo "Step 2: no CSVs found in $FEATURES_SCORES_DIR -> running 2_generate_features.py"
	# If z_thresh is negative, do *not* drop by MAD or pass the flag
	if [[ "$Z_THRESH" -ge 0 ]]; then
		python3 2_generate_features.py \
			--ai_model "$FEATURE_GENERATION_LLM" \
			--task_module "$TASK_MODULE" \
			--prompts_answers_pkl_file "$PROMPTS_ANSWERS_PKL_FILE" \
			--features_scores_dir "$FEATURES_SCORES_DIR" \
			--cache_dir "$EXPERIMENT_LLM_CACHE_DIR" \
			--num_correct_example_prompts 32 \
			--num_incorrect_example_prompts 32 \
			--drop_near_duplicate_features \
			--near_duplicate_features_threshold 0.9999 \
			--drop_low_predictive_power_features \
			--min_delta 0.2 \
			--drop_high_mad_variance_features \
			--z_thresh "$Z_THRESH"
	else
		python3 2_generate_features.py \
			--ai_model "$FEATURE_GENERATION_LLM" \
			--task_module "$TASK_MODULE" \
			--prompts_answers_pkl_file "$PROMPTS_ANSWERS_PKL_FILE" \
			--features_scores_dir "$FEATURES_SCORES_DIR" \
			--cache_dir "$EXPERIMENT_LLM_CACHE_DIR" \
			--num_correct_example_prompts 32 \
			--num_incorrect_example_prompts 32 \
			--drop_near_duplicate_features \
			--near_duplicate_features_threshold 0.9999 \
			--drop_low_predictive_power_features \
			--min_delta 0.2
	fi
fi

############################################
# Step 3: extract rules (skip if CSVs already exist)
if [[ "$SPLITS" == "spectral" ]]; then
	echo "Step 3: spectral_splits -> skipping 3_extract_rules.py"
else
	shopt -s nullglob

	if [[ "$INCORRECT_RULES" == "true" ]]; then
		matches=( "$RULES_DIR"/optimal_rule_set_*_fake.csv )
	else
		matches=( "$RULES_DIR"/optimal_rule_set_*.csv )
		# remove *_fake.csv from matches
		filtered=()
		for f in "${matches[@]}"; do
			[[ "$f" == *_fake.csv ]] && continue
			filtered+=( "$f" )
		done
		matches=( "${filtered[@]}" )
	fi

	shopt -u nullglob

	if (( ${#matches[@]} > 0 )); then
		echo "Step 3: found ${matches[0]##*/} in $RULES_DIR -> skipping 3_extract_rules.py"
	else
		cmd=(python3 3_extract_rules.py
			--task_module "$TASK_MODULE"
			--features_scores_dir "$FEATURES_SCORES_DIR"
			--rules_dir "$RULES_DIR"
			--npermutations 5
			--only_unique_datapoints_in_shap
			--use_shap_in_xgb
			--use_shap_in_lasso
			${FAKE_FLAG[@]}
		)

		if [[ "$INCORRECT_RULES" == "true" ]]; then
			cmd+=(--fake_target_max_abs_corr 0.05 --fake_target_max_tries 500)
		fi

		"${cmd[@]}"
	fi
fi

############################################
# Step 4: Spectral sampling plans (only when needed)
#   - Needed if PLAN == spectral (for anchoring) and/or DISCOVERY == spectral (for circuit discovery)
# In spectral_splits mode we do not use rule-indexed sampling plans.
if [[ "$SPLITS" == "spectral" ]]; then
	echo "Step 4: spectral_splits -> skipping 4_spectral_sample_datapoints.py"
else
	NEED_ALL_PLAN=false
	NEED_BASELINE_PLANS=false

	if [[ "$DISCOVERY" == "spectral" ]]; then
		NEED_ALL_PLAN=true
	fi
	if [[ "$PLAN" == "spectral" ]]; then
		NEED_BASELINE_PLANS=true
	fi

	# Ensure we have the "all" representations pkl for refine even in random-plan mode
	if [[ ! -f "$EXPERIMENT_LLM_CACHE_DIR/spectral_sampling_plan_qwen2_hidden_all.pkl" ]]; then
		NEED_ALL_PLAN=true
	fi

	if [[ "$NEED_ALL_PLAN" == "true" ]]; then
		OUT_ALL="$DISCOVERY_OUT_DIR/spectral_sampling_plan_qwen2_hidden_for_circuit_discovery.json"
		if [[ -f "$OUT_ALL" ]]; then
			echo "Skipping (exists): $OUT_ALL"
		else
			python3 4_spectral_sample_datapoints.py \
				--task_module "$TASK_MODULE" \
				--ai_model "$ANALYZED_LLM" \
				--spectral_cache_dir $CACHE_DIR \
				"${SPECTRAL_FLAGS[@]}" \
				--features_scores_dir "$FEATURES_SCORES_DIR" \
				--rules_glob "optimal_rule_set" \
				--rules_dir "$RULES_DIR" \
				--baseline_subset all \
				--pair_by_similarity_len_matched \
				--pair_similarity_metric "$PAIR_SIMILARITY_METRIC" \
				--min_points_per_ablation "$MAX_POINTS_PER_CIRCUIT" \
				--use_global_clusters \
				--global_n_clusters "$((MAX_POINTS_PER_CIRCUIT / 4))" \
				--batch_size "$BATCH_SIZE" \
				--output_path "$OUT_ALL" \
				--compute_cover_stats \
				${FAKE_FLAG[@]}
		fi
	fi

	if [[ "$NEED_BASELINE_PLANS" == "true" ]]; then
		OUT_POS="$DISCOVERY_OUT_DIR/spectral_sampling_plan_qwen2_hidden_for_neuron_ablation_baseline_positive.json"
		if [[ -f "$OUT_POS" ]]; then
			echo "Skipping (exists): $OUT_POS"
		else
			python3 4_spectral_sample_datapoints.py \
				--task_module "$TASK_MODULE" \
				--ai_model "$ANALYZED_LLM" \
				--spectral_cache_dir $CACHE_DIR \
				"${SPECTRAL_FLAGS[@]}" \
				--features_scores_dir "$FEATURES_SCORES_DIR" \
				--rules_glob "optimal_rule_set" \
				--rules_dir "$RULES_DIR" \
				--baseline_subset positive \
				--min_points_per_ablation "$MAX_POINTS_PER_ABLATION" \
				--use_global_clusters \
				--global_n_clusters "$((MAX_POINTS_PER_ABLATION / 4))" \
				--batch_size "$BATCH_SIZE" \
				--output_path "$OUT_POS" \
				--compute_cover_stats \
				${FAKE_FLAG[@]}
		fi

		OUT_NEG="$DISCOVERY_OUT_DIR/spectral_sampling_plan_qwen2_hidden_for_neuron_ablation_baseline_negative.json"
		if [[ -f "$OUT_NEG" ]]; then
			echo "Skipping (exists): $OUT_NEG"
		else
			python3 4_spectral_sample_datapoints.py \
				--task_module "$TASK_MODULE" \
				--ai_model "$ANALYZED_LLM" \
				--spectral_cache_dir $CACHE_DIR \
				"${SPECTRAL_FLAGS[@]}" \
				--features_scores_dir "$FEATURES_SCORES_DIR" \
				--rules_glob "optimal_rule_set" \
				--rules_dir "$RULES_DIR" \
				--baseline_subset negative \
				--min_points_per_ablation "$MAX_POINTS_PER_ABLATION" \
				--use_global_clusters \
				--global_n_clusters "$((MAX_POINTS_PER_ABLATION / 4))" \
				--batch_size "$BATCH_SIZE" \
				--output_path "$OUT_NEG" \
				--compute_cover_stats \
				${FAKE_FLAG[@]}
		fi
	fi
fi

############################################
# Step 5: Circuit discovery (mutually exclusive: random vs spectral)
DISCOVERY_INPUT_AUTODISCOVERY_DIR="$DISCOVERY_OUT_DIR/neural_circuits"

if [[ "$SPLITS" == "spectral" ]]; then
	python3 5_discover_circuits.py \
		--task_module "$TASK_MODULE" \
		--ai_model "$ANALYZED_LLM" \
		--rules_dir "$RULES_DIR" \
		--output_data_dir "$DISCOVERY_INPUT_AUTODISCOVERY_DIR" \
		--features_scores_dir "$FEATURES_SCORES_DIR" \
		--method $CIRCUIT_DISCOVERY_METHOD \
		--intervention $CIRCUIT_INTERVENTION \
		--circuit_size $CIRCUIT_SIZE \
		--absolute_value_attributions \
		--circuit_level $CIRCUIT_LEVEL \
		--eval_intervention $SCRIPT5_EVAL_INTERVENTION \
		--points_to_use_for_mean_ablation "$POINTS_TO_USE_FOR_MEAN_ABLATION" \
		--temporal_agg mean \
		--cache_dir "$EXPERIMENT_LLM_CACHE_DIR" \
		--max_pairs_per_circuit $MAX_POINTS_PER_CIRCUIT \
		--pair_similarity_metric "$PAIR_SIMILARITY_METRIC" \
		--batch_size 1 \
		--spectral_cache_dir $CACHE_DIR \
		--cluster_by_spectral \
		"${SPECTRAL_FLAGS[@]}" \
		--global_n_clusters "$MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE" \
		"${DECODE_FLAG[@]}" \
		"${FAKE_FLAG[@]}" \
		"${NEURONS_TYPE_FLAG[@]}"
else

	if [[ "$DISCOVERY" == "spectral" ]]; then
		python3 5_discover_circuits.py \
			--task_module "$TASK_MODULE" \
			--ai_model "$ANALYZED_LLM" \
			--rules_dir "$RULES_DIR" \
			--output_data_dir "$DISCOVERY_INPUT_AUTODISCOVERY_DIR" \
			--features_scores_dir "$FEATURES_SCORES_DIR" \
			--rules_glob "optimal_rule_set" \
			--method $CIRCUIT_DISCOVERY_METHOD \
			--intervention $CIRCUIT_INTERVENTION \
			--circuit_size $CIRCUIT_SIZE \
			--absolute_value_attributions \
			--circuit_level $CIRCUIT_LEVEL \
			--eval_intervention $SCRIPT5_EVAL_INTERVENTION \
			--points_to_use_for_mean_ablation "$POINTS_TO_USE_FOR_MEAN_ABLATION" \
			--temporal_agg mean \
			--cache_dir "$EXPERIMENT_LLM_CACHE_DIR" \
			--max_pairs_per_circuit $MAX_POINTS_PER_CIRCUIT \
			--sampling_strategy plan \
			--sampling_plan_path "$DISCOVERY_OUT_DIR/spectral_sampling_plan_qwen2_hidden_for_circuit_discovery.json" \
			--batch_size 1 \
			--max_n_of_rules_to_analyze $MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE \
			"${DECODE_FLAG[@]}" \
			"${FAKE_FLAG[@]}" \
			"${NEURONS_TYPE_FLAG[@]}"
	else
		python3 5_discover_circuits.py \
			--task_module "$TASK_MODULE" \
			--ai_model "$ANALYZED_LLM" \
			--rules_dir "$RULES_DIR" \
			--output_data_dir "$DISCOVERY_INPUT_AUTODISCOVERY_DIR" \
			--features_scores_dir "$FEATURES_SCORES_DIR" \
			--rules_glob "optimal_rule_set" \
			--method $CIRCUIT_DISCOVERY_METHOD \
			--intervention $CIRCUIT_INTERVENTION \
			--circuit_size $CIRCUIT_SIZE \
			--absolute_value_attributions \
			--circuit_level $CIRCUIT_LEVEL \
			--pair_similarity_metric "$PAIR_SIMILARITY_METRIC" \
			--eval_intervention $SCRIPT5_EVAL_INTERVENTION \
			--points_to_use_for_mean_ablation "$POINTS_TO_USE_FOR_MEAN_ABLATION" \
			--temporal_agg mean \
			--cache_dir "$EXPERIMENT_LLM_CACHE_DIR" \
			--max_pairs_per_circuit $MAX_POINTS_PER_CIRCUIT \
			--batch_size 1 \
			--max_n_of_rules_to_analyze $MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE \
			"${DECODE_FLAG[@]}" \
			"${FAKE_FLAG[@]}" \
			"${NEURONS_TYPE_FLAG[@]}"
	fi
fi

############################################
# Steps 6-7: Neuron anchoring (fast vs slow) + plan choice (spectral vs random)
# Bag-of-rules dir naming:
# - Preserve your old convention for fast spectral-on-spectral case: bag_of_rules_fast_spectral
# - Preserve old fast spectral-on-random: bag_of_rules_fast
# - Preserve old random plan fast: bag_of_rules_fast
# - For slow spectral-on-random (new combo), avoid clobbering slow spectral-on-spectral by using a suffix.
BAG_LABEL="agonist_neurons"
FAST_FLAG=()
if [[ "${ANCHORING:-}" == "fast" ]]; then
  FAST_FLAG=(--fast_ablation)
  BAG_LABEL+="-fast"
fi
# if [[ "${DISCOVERY:-}" == "spectral" ]]; then
#   BAG_LABEL+="_spectral-discovery"
# elif [[ "${DISCOVERY:-}" == "random" ]]; then
#   BAG_LABEL+="_random-discovery"
# fi
if [[ "${PLAN:-}" == "spectral" ]]; then
  BAG_LABEL+="-spectral_anchor"
elif [[ "${PLAN:-}" == "random" ]]; then
  BAG_LABEL+="-random_anchor"
fi
if [[ "$MIN_FLIP_RATE" != "0.2" ]]; then
  BAG_LABEL+="-tau${MIN_FLIP_RATE}"
fi
echo $BAG_LABEL

# Helper to run analyze_bag_of_rules for a baseline subset
run_analyze() {
	local baseline_subset="$1"
	local out_dir="$2"

	if [[ "$SPLITS" == "spectral" ]]; then
		python3 6_analyze_bag_of_rules.py \
			--spectral_cache_dir $CACHE_DIR \
			--cluster_by_spectral \
			"${SPECTRAL_FLAGS[@]}" \
			--global_n_clusters "$MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE" \
			--input_data_dir "$DISCOVERY_INPUT_AUTODISCOVERY_DIR" \
			--output_data_dir "$out_dir" \
			--task_module "$TASK_MODULE" \
			--n_associated $MAX_POINTS_PER_ABLATION \
			--n_unrelated $MAX_POINTS_PER_ABLATION \
			--batch_size "$BATCH_SIZE" \
			--search_epsilon $MIN_FLIP_RATE \
			--points_to_use_for_mean_ablation "$POINTS_TO_USE_FOR_MEAN_ABLATION" \
			--intervention $EVAL_INTERVENTION \
			"${FAST_FLAG[@]}" \
			--baseline_subset "$baseline_subset" \
			--sign_split_first \
			"${DECODE_FLAG[@]}" \
			"${NEURONS_TYPE_FLAG[@]}"

	elif [[ "$PLAN" == "spectral" ]]; then
		local plan_path=""
		if [[ "$baseline_subset" == "positive" ]]; then
			plan_path="$DISCOVERY_OUT_DIR/spectral_sampling_plan_qwen2_hidden_for_neuron_ablation_baseline_positive.json"
		else
			plan_path="$DISCOVERY_OUT_DIR/spectral_sampling_plan_qwen2_hidden_for_neuron_ablation_baseline_negative.json"
		fi

		python3 6_analyze_bag_of_rules.py \
			--input_data_dir "$DISCOVERY_INPUT_AUTODISCOVERY_DIR" \
			--output_data_dir "$out_dir" \
			--task_module "$TASK_MODULE" \
			--n_associated $MAX_POINTS_PER_ABLATION \
			--n_unrelated $MAX_POINTS_PER_ABLATION \
			--batch_size "$BATCH_SIZE" \
			--search_epsilon $MIN_FLIP_RATE \
			--points_to_use_for_mean_ablation "$POINTS_TO_USE_FOR_MEAN_ABLATION" \
			--intervention $EVAL_INTERVENTION \
			"${FAST_FLAG[@]}" \
			--baseline_subset "$baseline_subset" \
			--sampling_strategy plan \
			--sampling_plan_path "$plan_path" \
			--sign_split_first \
			"${DECODE_FLAG[@]}" \
			"${NEURONS_TYPE_FLAG[@]}"
	else
		python3 6_analyze_bag_of_rules.py \
			--input_data_dir "$DISCOVERY_INPUT_AUTODISCOVERY_DIR" \
			--output_data_dir "$out_dir" \
			--task_module "$TASK_MODULE" \
			--n_associated $MAX_POINTS_PER_ABLATION \
			--n_unrelated $MAX_POINTS_PER_ABLATION \
			--batch_size "$BATCH_SIZE" \
			--search_epsilon $MIN_FLIP_RATE \
			--points_to_use_for_mean_ablation "$POINTS_TO_USE_FOR_MEAN_ABLATION" \
			--intervention $EVAL_INTERVENTION \
			"${FAST_FLAG[@]}" \
			--baseline_subset "$baseline_subset" \
			--sign_split_first \
			"${DECODE_FLAG[@]}" \
			"${NEURONS_TYPE_FLAG[@]}"
	fi
}

# Run anchoring for both baselines
run_analyze "positive" "$DISCOVERY_OUT_DIR/$BAG_LABEL/positive_baseline"
run_analyze "negative" "$DISCOVERY_OUT_DIR/$BAG_LABEL/negative_baseline"

CIRCUIT_BAG_LABEL="$CIRCUIT_LABEL-$BAG_LABEL"
if [[ "$INCORRECT_RULES" == "true" ]]; then
	CIRCUIT_BAG_LABEL+="-fake_targets"
fi
python3 7_refine_neuron_anchored_rules.py \
	--task_module "$TASK_MODULE" \
	--ai_model "$ANALYZED_LLM" \
	--rules_dir "$RULES_DIR/neuron_flip_rules" \
	--features_scores_dir "$FEATURES_SCORES_DIR" \
	--circuit_agonists_path "$DISCOVERY_OUT_DIR/$BAG_LABEL" \
	--search_epsilon $MIN_FLIP_RATE \
	--batch_size "$BATCH_SIZE" \
	--stats_dirname "$CIRCUIT_BAG_LABEL" \
	--use_spectral_sampling \
	--sampling_max_points 10000 \
	--spectral_cache_dir $CACHE_DIR \
	"${SPECTRAL_FLAGS[@]}" \
	--global_n_clusters "$MAX_POINTS_PER_ABLATION" \
	--points_to_use_for_mean_ablation "$POINTS_TO_USE_FOR_MEAN_ABLATION" \
	--intervention $EVAL_INTERVENTION \
	--extract_rules \
	--only_unique_datapoints_in_shap \
	"${DECODE_FLAG[@]}" \
	--summarize_rule_metrics

echo "Done."
