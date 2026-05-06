#!/usr/bin/env bash

# Optional, but recommended so failures surface:
# set -euo pipefail

# Activate virtual environment
source .env/bin/activate

# Optional: run Hugging Face in offline mode
# export HF_HUB_OFFLINE=1

LANGUAGE_EXPERIMENT_LIST=(
	##########
	### Language & Reasoning
	"random_fsm"
	"grammar_acceptability"
	"hans_nli"
)

MATH_EXPERIMENT_LIST=(
	##########
	### Math
	"arithmetic"
)

JAILBREAKING_EXPERIMENT_LIST=(
	##########
	### Jailbreaking
	"bon_jailbreaking"
	# "llmsafe_alignment"
)

CODING_EXPERIMENT_LIST=(
	##########
	### Coding
	"cognitive_bias_sensitivity"
	# "cve_infile_vulnerability_detection"
	# "code_in_the_haystack"
)

EXPERIMENT_LIST=(
	"${LANGUAGE_EXPERIMENT_LIST[@]}"
	"${MATH_EXPERIMENT_LIST[@]}"
	"${JAILBREAKING_EXPERIMENT_LIST[@]}"
	# "${CODING_EXPERIMENT_LIST[@]}"
)

ANALYZED_LLM_LIST=(
	### Small models experiment
	"Qwen/Qwen2-1.5B-Instruct"
	"Qwen/Qwen2.5-1.5B-Instruct"
	"EleutherAI/pythia-1b" # equivalent to "EleutherAI/pythia-1b@step143000"
	### Small model - Checkpoints experiment
	"EleutherAI/pythia-1b@step0"
	"EleutherAI/pythia-1b@step48000"
	"EleutherAI/pythia-1b@step96000"
	# ### Medium models experiment
	# "Qwen/Qwen2-7B-Instruct"
	# "Qwen/Qwen2.5-7B-Instruct"
	# "EleutherAI/gpt-j-6B"
	# "EleutherAI/pythia-6.9b"
	# ### Medium model - Checkpoints experiment
	# "EleutherAI/pythia-6.9b@step0"
	# "EleutherAI/pythia-6.9b@step48000"
	# "EleutherAI/pythia-6.9b@step96000"
)

EVAL_INTERVENTION_LIST=(
	"mean-donor"
	"mean"
	"zero"
)


# Run decode-only only as an additional mode for language/reasoning tasks.

contains_item() {
	local needle="$1"
	shift
	local item
	for item in "$@"; do
		if [[ "$item" == "$needle" ]]; then
			return 0
		fi
	done
	return 1
}

should_run_standard() {
	local experiment="$1"
	local llm="$2"

	# # Normal rule: standard mode only for language/reasoning tasks.
	# if contains_item "$experiment" "${LANGUAGE_EXPERIMENT_LIST[@]}"; then
	# 	should_run_experiment_for_llm "$experiment" "$llm" || return 1
	# 	return 0
	# fi

	# # Normal rule: standard mode only for language/reasoning tasks.
	# if contains_item "$experiment" "${MATH_EXPERIMENT_LIST[@]}"; then
	# 	should_run_experiment_for_llm "$experiment" "$llm" || return 1
	# 	return 0
	# fi

	return 0
}

should_run_experiment_for_llm() {
	local experiment="$1"
	local llm="$2"

	# Qwen2-1.5B is only for arithmetic, so skip all other tasks.
	if [[ "$llm" == "Qwen/Qwen2-1.5B-Instruct" ]]; then
		contains_item "$experiment" "${MATH_EXPERIMENT_LIST[@]}" || return 1
	fi

	if [[ "$llm" == "Qwen/Qwen2-1.5B-Instruct" ]]; then
		if [[ "$experiment" == "hans_nli" ]]; then
			return 1
		fi
	fi

	# Pythia models are only for language/reasoning tasks
	# and jailbreaking for them.
	if [[ "$llm" == EleutherAI/pythia-1b@* ]]; then
		contains_item "$experiment" "${LANGUAGE_EXPERIMENT_LIST[@]}" || return 1
	fi

	return 0
}

# Fallback if experiment not listed explicitly
DEFAULT_Z_THRESH=-1   # negative -> no MAD filtering
DEFAULT_BATCH_SIZE=32
DEFAULT_CIRCUIT_LEVEL=neuron
DEFAULT_CIRCUIT_SIZE=200000
DEFAULT_MIN_FLIP_RATE=0.3

###############################################################################
# Main run loop order:
# intervention -> llm -> task
###############################################################################
for EVAL_INTERVENTION in "${EVAL_INTERVENTION_LIST[@]}"; do
	for ANALYZED_LLM in "${ANALYZED_LLM_LIST[@]}"; do
		for EXPERIMENT in "${EXPERIMENT_LIST[@]}"; do

			if ! should_run_experiment_for_llm "$EXPERIMENT" "$ANALYZED_LLM"; then
				echo "Skipping $EXPERIMENT for $ANALYZED_LLM"
				continue
			fi

			EXP_DIR=./data/$EXPERIMENT

			# Experiment-level defaults
			BASE_Z_THRESH="$DEFAULT_Z_THRESH"
			MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE=1
			BATCH_SIZE="$DEFAULT_BATCH_SIZE"
			CIRCUIT_LEVEL="$DEFAULT_CIRCUIT_LEVEL"
			CIRCUIT_SIZE="$DEFAULT_CIRCUIT_SIZE"
			MIN_FLIP_RATE="$DEFAULT_MIN_FLIP_RATE"
			# NEURONS_TYPE_FLAG=(--mlp_neurons_only)
			NEURONS_TYPE_FLAG=()

			Z_THRESH="$BASE_Z_THRESH"

			case "$EXPERIMENT" in
				arithmetic)
					# MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE=5
					case "$ANALYZED_LLM" in
						Qwen/*)
							Z_THRESH=10
							;;
						*)
							Z_THRESH=5
							;;
					esac
					;;
				# bon_jailbreaking)
				# 	# case "$ANALYZED_LLM" in
				# 	# 	Qwen/*)
				# 	# 		BATCH_SIZE=256
				# 	# 		;;
				# 	# 	*)
				# 	# 		BATCH_SIZE=32
				# 	# 		;;
				# 	# esac
				# 	DECODE_ONLY_FLAG=(--decode_only)
				# 	;;
				# llmsafe_alignment)
				# 	# BATCH_SIZE=32
				# 	DECODE_ONLY_FLAG=(--decode_only)
				# 	;;
				cognitive_bias_sensitivity)
					# BATCH_SIZE=32
					# DECODE_ONLY_FLAG=(--decode_only)
					MIN_FLIP_RATE=0.3
					CIRCUIT_SIZE=50000
					;;
			esac

			RUN_MODE_LIST=("decode-only")

			# Only language/reasoning experiments can additionally run in standard mode.
			if should_run_standard "$EXPERIMENT" "$ANALYZED_LLM"; then
				RUN_MODE_LIST=("standard" "decode-only")
			fi

			echo "Run modes: ${RUN_MODE_LIST[*]}"

			for RUN_MODE in "${RUN_MODE_LIST[@]}"; do
				DECODE_ONLY_FLAG=()
				if [[ "$RUN_MODE" == "decode-only" ]]; then
					DECODE_ONLY_FLAG=(--decode_only)
				fi

				echo "Running $EXPERIMENT with Z_THRESH=$Z_THRESH, BATCH_SIZE=$BATCH_SIZE, CIRCUIT_LEVEL=$CIRCUIT_LEVEL, CIRCUIT_SIZE=$CIRCUIT_SIZE, EVAL_INTERVENTION=$EVAL_INTERVENTION, RUN_MODE=$RUN_MODE"

				# "Spectral splits" experiment
				bash _run_pipeline.sh \
					"$EXPERIMENT" \
					"$ANALYZED_LLM" \
					--spectral_splits \
					--fast_anchoring \
					--z_thresh "$Z_THRESH" \
					--batch_size "$BATCH_SIZE" \
					--circuit_level "$CIRCUIT_LEVEL" \
					--circuit_size "$CIRCUIT_SIZE" \
					--eval_intervention "$EVAL_INTERVENTION" \
					--min_flip_rate "$MIN_FLIP_RATE" \
					"${DECODE_ONLY_FLAG[@]}" \
					"${NEURONS_TYPE_FLAG[@]}" \
					--max_number_of_circuits_to_analyze "$MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE"
			done
		done
	done
done

###############################################################################
# Post-processing after ALL runs are complete
###############################################################################
for EXPERIMENT in "${EXPERIMENT_LIST[@]}"; do
	EXP_DIR=./data/$EXPERIMENT

	python3 8_compare_experiments.py \
		--stats_path "$EXP_DIR" \
		--out_dir ./data/aggregated_visualizations \
		--rule_quality_metric mcc --best_mode tail@0.9 \
		--thr_min 0.85 --thr_max 0.99 --thr_step 0.01

	python3 9_compare_models.py \
		--task_dir "$EXP_DIR" \
		--out_dir ./data/aggregated_visualizations/$EXPERIMENT \
		--tau 0.2 --eps 0.2
done

python3 8_compare_experiments.py \
	--stats_path ./data/ \
	--out_dir ./data/aggregated_visualizations \
	--rule_quality_metric mcc --best_mode tail@0.9 \
	--thr_min 0.85 --thr_max 0.99 --thr_step 0.01

python3 10_compute_threshold_sweep_stats.py \
	--csv data/aggregated_visualizations/aggregated_by_task/threshold_sweep_all_tasks_all_llms.csv
