#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$CODE_ROOT/.." && pwd)"
cd "$CODE_ROOT"

############################################
# Usage:
#   ./run_pipeline.sh <EXPERIMENT_NAME> <ANALYZED_LLM> \
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
#     [--min_flip_rate <R>] \
#     [--evaluation_split <test|train|all>]
#
# Defaults:
#   --random_anchoring_plan --random_circuit_discovery --fast_anchoring --eval_intervention mean-positional --evaluation_split test --z_thresh -1 --max_number_of_circuits_to_analyze -1 --batch_size 256 --circuit_level neuron --circuit_size 100000 --min_flip_rate 0.2
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
							Max number of circuits/rules to analyze in stage05_discover_circuits.py.
							Default: -1 (all circuits)
		--eval_intervention <NAME>
							Intervention used in Stages 5, 6, and 7.
							Default: mean-positional
		--batch_size <N>   Batch size used by the pipeline stages that support batching.
							Default: 256
		--circuit_level <neuron|edge>
							Circuit granularity for stage05_discover_circuits.py.
							Default: neuron
		--circuit_size <N>
							Circuit size passed to stage05_discover_circuits.py.
							Default: 100000
		--min_flip_rate <R>
							Minimum flip-rate threshold (tau) used in Stages 6 and 7.
							Default: ${CHA_TAU:-0.2} (or --min_flip_rate).
		--evaluation_split <test|train|all>
							Rows used by stage 7 and interaction validation. Default: test.
		--evaluation_baseline_subset <all|positive|negative>
							Condition stage-7 rows on the unablated binary predicate. Default: all.
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
							Skips Stages 3 and 4; uses --cluster_by_spectral in Stages 5 and 6.

	Rule control:
		--incorrect_rules   Use random fake targets before rule extraction (Script 3) to generate intentionally incorrect rules
							(re-sampled until abs(Pearson corr) is near 0).

	Neuron selection:
		--mlp_neurons_only  Restrict neuron analysis to MLP neurons only.

	Decode control:
		--decode_only       Pass --decode_only through to scripts that support it (default: off).
		--no_llm_feature_generation
							Skip Ollama/LLM feature proposal and use only task seed features.
		--output_data_dir <DIR>
							Override the default <repo>/data/<task>/<model> output directory.
							Useful for local fine-tuned checkpoints.
		--model_label <LABEL>
							Filesystem/cache label for ANALYZED_LLM. Defaults to the raw model
							name for HF ids and a sanitized label for existing local paths.
		--pipeline_cache_root <DIR>
							Override the task cache root. Default: <repo>/cache/<task>.
		--pipeline_model_cache_dir <DIR>
							Override the per-model pipeline cache directory.
		--task_module <PYTHON_MODULE[:ATTRIBUTE]>
							Override the task spec. Default: core.tasks.<EXPERIMENT_NAME>_task (its TASK_SPEC).
							A bare module selects TASK_SPEC; module:attribute selects a named spec.
							Poisoning jobs use named specs under studies.poisoning.tasks.
		--skip_stage1
							Reuse an already-created llm_io_data.pkl + feature_report/scores.csv.

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
if [[ -f "$PROJECT_ROOT/.env/bin/activate" ]]; then
	. "$PROJECT_ROOT/.env/bin/activate"
fi

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
Z_THRESH="-1"            # MAD z-threshold for Stage 2 (drop_high_mad_variance_features)
MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE="-1" # -1 means ALL CIRCUITS
EVAL_INTERVENTION="mean-positional"
BATCH_SIZE="256"
CIRCUIT_LEVEL="neuron"
CIRCUIT_SIZE="100000"
MIN_FLIP_RATE="${CHA_TAU:-0.2}"
INCORRECT_RULES=false
DECODE_ONLY=false
NEURONS_TYPE="all"
OUTPUT_DATA_DIR=""
MODEL_LABEL=""
PIPELINE_CACHE_ROOT=""
PIPELINE_MODEL_CACHE_DIR=""
TASK_MODULE_OVERRIDE=""
SKIP_STAGE1=false
THRESHOLD_EVENT_CLAMP_TOPK="${THRESHOLD_EVENT_CLAMP_TOPK:-0}"
FORCE_THRESHOLD_EVENT_POSTHOC="${FORCE_THRESHOLD_EVENT_POSTHOC:-false}"
NO_LLM_FEATURE_GENERATION="${NO_LLM_FEATURE_GENERATION:-false}"
# Stage 7 is a held-out singleton causal evaluation, not a neuron-rule refinement stage.
# RUN_SINGLETON_CAUSAL_EVALUATION is the canonical flag. Preserve the old name only
# as a deprecated compatibility alias for existing external launchers.
if [[ -n "${RUN_REFINE_NEURON_RULES+x}" && -z "${RUN_SINGLETON_CAUSAL_EVALUATION+x}" ]]; then
	RUN_SINGLETON_CAUSAL_EVALUATION="$RUN_REFINE_NEURON_RULES"
	echo "[deprecated] RUN_REFINE_NEURON_RULES is now RUN_SINGLETON_CAUSAL_EVALUATION" >&2
elif [[ -n "${RUN_REFINE_NEURON_RULES+x}" && -n "${RUN_SINGLETON_CAUSAL_EVALUATION+x}" ]]; then
	echo "[deprecated] ignoring RUN_REFINE_NEURON_RULES because RUN_SINGLETON_CAUSAL_EVALUATION is set" >&2
fi
RUN_SINGLETON_CAUSAL_EVALUATION="${RUN_SINGLETON_CAUSAL_EVALUATION:-true}"
RUN_THRESHOLD_EVENT_POSTHOC="${RUN_THRESHOLD_EVENT_POSTHOC:-false}"
RUN_GRADED_AGONIST_INTERVENTION="${RUN_GRADED_AGONIST_INTERVENTION:-true}"
GRADED_AGONIST_DOSES="${GRADED_AGONIST_DOSES:-0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1}"
GRADED_AGONIST_MAX_UNITS_PER_DIRECTION="${GRADED_AGONIST_MAX_UNITS_PER_DIRECTION:-16}"
GRADED_AGONIST_MAX_POSITIVE_SUPPORT="${GRADED_AGONIST_MAX_POSITIVE_SUPPORT:-256}"
GRADED_AGONIST_NEGATIVE_SUPPORT="${GRADED_AGONIST_NEGATIVE_SUPPORT:-false}"
GRADED_AGONIST_NEGATIVE_RATIO="${GRADED_AGONIST_NEGATIVE_RATIO:-1.0}"
GRADED_AGONIST_MAX_NEGATIVE_SUPPORT="${GRADED_AGONIST_MAX_NEGATIVE_SUPPORT:-256}"
GRADED_AGONIST_SEED="${GRADED_AGONIST_SEED:-42}"
FORCE_GRADED_AGONIST_INTERVENTION="${FORCE_GRADED_AGONIST_INTERVENTION:-false}"
THRESHOLD_EVENT_TARGET="${THRESHOLD_EVENT_TARGET:-all}"
THRESHOLD_EVENT_MAX_POINTS="${THRESHOLD_EVENT_MAX_POINTS:-10000}"
THRESHOLD_EVENT_MIN_POINTS="${THRESHOLD_EVENT_MIN_POINTS:-512}"
THRESHOLD_EVENT_REPEATS="${THRESHOLD_EVENT_REPEATS:-20}"
THRESHOLD_EVENT_HOLDOUT_FRACTION="${THRESHOLD_EVENT_HOLDOUT_FRACTION:-0.5}"
THRESHOLD_EVENT_N_BINS="${THRESHOLD_EVENT_N_BINS:-10}"
THRESHOLD_EVENT_SEED="${THRESHOLD_EVENT_SEED:-42}"
REFINE_EXTRACT_RULES="${REFINE_EXTRACT_RULES:-false}"
REFINE_SUMMARIZE_RULE_METRICS="${REFINE_SUMMARIZE_RULE_METRICS:-false}"
REFINE_MAX_NEURONS="${REFINE_MAX_NEURONS:-0}"
REFINE_NEURON_BATCH_SIZE="${REFINE_NEURON_BATCH_SIZE:-8}"
REFINE_SAMPLING_MAX_POINTS="${REFINE_SAMPLING_MAX_POINTS:-10000}"
REFINE_USE_SPECTRAL_SAMPLING="${REFINE_USE_SPECTRAL_SAMPLING:-true}"
REFINE_EXCLUDE_DISCOVERY_ROWS_FROM_FINAL_STATS="${REFINE_EXCLUDE_DISCOVERY_ROWS_FROM_FINAL_STATS:-true}"
SKIP_AGONIST_METRIC_STATS="${SKIP_AGONIST_METRIC_STATS:-false}"
ANALYZE_BASELINE_SUBSETS="${ANALYZE_BASELINE_SUBSETS:-positive,negative}"
EVALUATION_SPLIT="${EVALUATION_SPLIT:-test}"
EVALUATION_BASELINE_SUBSET="${EVALUATION_BASELINE_SUBSET:-${PIPELINE_EVALUATION_BASELINE_SUBSET:-all}}"
RUN_INTERACTION_VALIDATION="${RUN_INTERACTION_VALIDATION:-true}"
RUN_CMC="${RUN_CMC:-true}"
INTERACTION_NULL_DRAWS="${INTERACTION_NULL_DRAWS:-100}"
CONDITIONAL_BACKGROUND_MULTIPLIERS="${CONDITIONAL_BACKGROUND_MULTIPLIERS:-1}"
PREEMPTION_MIN_SINGLETON_RATE="${PREEMPTION_MIN_SINGLETON_RATE:-0.05}"
PREEMPTION_MAX_SECONDARIES="${PREEMPTION_MAX_SECONDARIES:-8}"
PREEMPTION_THRESHOLD_HOLDOUT_FRACTION="${PREEMPTION_THRESHOLD_HOLDOUT_FRACTION:-0.25}"
PREEMPTION_THRESHOLD_MIN_CLASS="${PREEMPTION_THRESHOLD_MIN_CLASS:-8}"
RUN_PREEMPTION="${RUN_PREEMPTION:-true}"
FORCE_INTERACTION_VALIDATION="${FORCE_INTERACTION_VALIDATION:-false}"
FORCE_STAGE7="${FORCE_STAGE7:-false}"
SPECTRAL_CLUSTER_BASE_SUBSET="${SPECTRAL_CLUSTER_BASE_SUBSET:-all}"
HF_MODEL_CACHE_DIR="${HF_MODEL_CACHE_DIR:-}"
HF_MODEL_CACHE_FLAG=()
if [[ -n "$HF_MODEL_CACHE_DIR" ]]; then HF_MODEL_CACHE_FLAG=(--ai_model_cache_dir "$HF_MODEL_CACHE_DIR"); fi

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
		--output_data_dir)
			[[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value"; exit 1; }
			OUTPUT_DATA_DIR="$2"
			shift 2
			;;
		--model_label)
			[[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value"; exit 1; }
			MODEL_LABEL="$2"
			shift 2
			;;
		--pipeline_cache_root)
			[[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value"; exit 1; }
			PIPELINE_CACHE_ROOT="$2"
			shift 2
			;;
		--pipeline_model_cache_dir)
			[[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value"; exit 1; }
			PIPELINE_MODEL_CACHE_DIR="$2"
			shift 2
			;;
		--task_module)
			[[ $# -ge 2 ]] || { echo "ERROR: $1 requires a Python module path"; exit 1; }
			TASK_MODULE_OVERRIDE="$2"
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
		--evaluation_split)
			[[ $# -ge 2 ]] || { echo "ERROR: $1 requires test, train, or all"; exit 1; }
			EVALUATION_SPLIT="$2"
			shift 2
			;;
		--evaluation_baseline_subset)
			[[ $# -ge 2 ]] || { echo "ERROR: $1 requires all, positive, or negative"; exit 1; }
			EVALUATION_BASELINE_SUBSET="$2"
			shift 2
			;;
		--no_llm_feature_generation)   NO_LLM_FEATURE_GENERATION=true; shift ;;
		--skip_stage1)                 SKIP_STAGE1=true; shift ;;
		-h|--help)                      usage; exit 0 ;;
		*) echo "Unknown option: $1"; usage; exit 1 ;;
	esac
done

case "$EVALUATION_SPLIT" in
	test|train|all) ;;
	*) echo "ERROR: --evaluation_split must be one of: test, train, all (got: $EVALUATION_SPLIT)"; exit 1 ;;
esac
case "$EVALUATION_BASELINE_SUBSET" in
	all|positive|negative) ;;
	*) echo "ERROR: --evaluation_baseline_subset must be one of: all, positive, negative (got: $EVALUATION_BASELINE_SUBSET)"; exit 1 ;;
esac

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
echo "EVALUATION_SPLIT: $EVALUATION_SPLIT"
echo "EVALUATION_BASELINE_SUBSET: $EVALUATION_BASELINE_SUBSET"
echo "MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE: $MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE"
echo "NO_LLM_FEATURE_GENERATION: $NO_LLM_FEATURE_GENERATION"
echo "TASK_MODULE_OVERRIDE: ${TASK_MODULE_OVERRIDE:-<default>}"
echo "SKIP_STAGE1: $SKIP_STAGE1"
echo "RUN_SINGLETON_CAUSAL_EVALUATION: $RUN_SINGLETON_CAUSAL_EVALUATION"
echo "RUN_THRESHOLD_EVENT_POSTHOC: $RUN_THRESHOLD_EVENT_POSTHOC"
echo "RUN_GRADED_AGONIST_INTERVENTION: $RUN_GRADED_AGONIST_INTERVENTION"
echo "GRADED_AGONIST_DOSES: $GRADED_AGONIST_DOSES"
echo "GRADED_AGONIST_MAX_UNITS_PER_DIRECTION: $GRADED_AGONIST_MAX_UNITS_PER_DIRECTION"
echo "GRADED_AGONIST_NEGATIVE_SUPPORT: $GRADED_AGONIST_NEGATIVE_SUPPORT"
echo "THRESHOLD_EVENT_TARGET: $THRESHOLD_EVENT_TARGET"
echo "THRESHOLD_EVENT_MAX_POINTS: $THRESHOLD_EVENT_MAX_POINTS"
echo "THRESHOLD_EVENT_MIN_POINTS: $THRESHOLD_EVENT_MIN_POINTS"
echo "REFINE_MAX_NEURONS: $REFINE_MAX_NEURONS"
echo "REFINE_SAMPLING_MAX_POINTS: $REFINE_SAMPLING_MAX_POINTS"
echo "REFINE_USE_SPECTRAL_SAMPLING: $REFINE_USE_SPECTRAL_SAMPLING"
echo "ANALYZE_BASELINE_SUBSETS: $ANALYZE_BASELINE_SUBSETS"
echo "SPECTRAL_CLUSTER_BASE_SUBSET: $SPECTRAL_CLUSTER_BASE_SUBSET"
echo "HF_MODEL_CACHE_DIR: ${HF_MODEL_CACHE_DIR:-<global Hugging Face cache>}"
echo "==============="

############################################
# Common config
semantic_local_model_label() {
	python3 - "$1" <<'PY'
import re
import sys
from pathlib import Path

p = Path(sys.argv[1]).expanduser()
name = p.name or "model"
if p.parent.name == "checkpoints" and p.parent.parent.name:
    name = f"{p.parent.parent.name}_{name}"
name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"
print(name)
PY
}

if [[ -z "$MODEL_LABEL" ]]; then
	if [[ -e "$ANALYZED_LLM" || "$ANALYZED_LLM" == /* || "$ANALYZED_LLM" == ./* || "$ANALYZED_LLM" == ../* ]]; then
		MODEL_LABEL="$(semantic_local_model_label "$ANALYZED_LLM")"
	else
		# Preserve the historical ./data/<task>/<org>/<model> layout for HF ids.
		MODEL_LABEL="$ANALYZED_LLM"
	fi
fi

# Interpret user-supplied relative filesystem paths from the repository root,
# even though Python packages execute from <repo>/code.
if [[ -n "$OUTPUT_DATA_DIR" && "$OUTPUT_DATA_DIR" != /* ]]; then
	OUTPUT_DATA_DIR="$PROJECT_ROOT/$OUTPUT_DATA_DIR"
fi
if [[ -n "$PIPELINE_CACHE_ROOT" && "$PIPELINE_CACHE_ROOT" != /* ]]; then
	PIPELINE_CACHE_ROOT="$PROJECT_ROOT/$PIPELINE_CACHE_ROOT"
fi
if [[ -n "$PIPELINE_MODEL_CACHE_DIR" && "$PIPELINE_MODEL_CACHE_DIR" != /* ]]; then
	PIPELINE_MODEL_CACHE_DIR="$PROJECT_ROOT/$PIPELINE_MODEL_CACHE_DIR"
fi
if [[ "$ANALYZED_LLM" != /* && -e "$PROJECT_ROOT/$ANALYZED_LLM" ]]; then
	ANALYZED_LLM="$PROJECT_ROOT/$ANALYZED_LLM"
fi

DATA_DIR="${OUTPUT_DATA_DIR:-$PROJECT_ROOT/data/$EXPERIMENT_NAME/$MODEL_LABEL}"
CACHE_DIR="${PIPELINE_CACHE_ROOT:-$PROJECT_ROOT/cache/$EXPERIMENT_NAME}"
EXPERIMENT_LLM_CACHE_DIR="${PIPELINE_MODEL_CACHE_DIR:-$CACHE_DIR/$MODEL_LABEL}"

echo "MODEL_LABEL:     $MODEL_LABEL"
echo "DATA_DIR:        $DATA_DIR"
echo "CACHE_DIR:       $CACHE_DIR"
echo "MODEL_CACHE_DIR: $EXPERIMENT_LLM_CACHE_DIR"

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

TASK_MODULE="${TASK_MODULE_OVERRIDE:-core.tasks.${EXPERIMENT_NAME}_task}"
FEATURE_GENERATION_LLM="${FEATURE_GENERATION_LLM:-gemma3:27b}"
PROMPTS_ANSWERS_PKL_FILE="$EXPERIMENT_LLM_CACHE_DIR/llm_io_data.pkl"
FEATURES_SCORES_DIR="$DATA_DIR/feature_report"
PAIR_SIMILARITY_METRIC="euclidean"

# Canonical spectral representation/sampling configuration. Stage-4 plan
# filenames are derived from these exact values so a plan is only reused when
# its result-affecting configuration matches.
SPECTRAL_SPACE="${SPECTRAL_SPACE:-hidden}"
REP_HOOK_NAME="${REP_HOOK_NAME:-ln_final.hook_normalized}"
REP_POOLING="${REP_POOLING:-mean}"
SPECTRAL_DIM="${SPECTRAL_DIM:-16}"
# Execution-only representation batch size. Keep independent from BATCH_SIZE,
# which may intentionally be 1 for memory-heavy circuit discovery.
SPECTRAL_EMBEDDING_BATCH_SIZE="${SPECTRAL_EMBEDDING_BATCH_SIZE:-32}"
SPECTRAL_MAX_SEQ_LEN="${SPECTRAL_MAX_SEQ_LEN:-}"
SPECTRAL_PLAN_MAX_POINTS="${SPECTRAL_PLAN_MAX_POINTS:-512}"
SPECTRAL_PLAN_SEED="${SPECTRAL_PLAN_SEED:-0}"
SPECTRAL_COVERAGE_RADIUS="${SPECTRAL_COVERAGE_RADIUS:-0.5}"
SPECTRAL_PAIR_LEN_TOLERANCE="${SPECTRAL_PAIR_LEN_TOLERANCE:-0}"
SPECTRAL_STATS_SAMPLE_SIZE="${SPECTRAL_STATS_SAMPLE_SIZE:-200000}"
SPECTRAL_STATS_CHUNK_SIZE="${SPECTRAL_STATS_CHUNK_SIZE:-8192}"
SPECTRAL_FLAGS=(
	--spectral_space "$SPECTRAL_SPACE"
	--rep_hook_name "$REP_HOOK_NAME"
	--rep_pooling "$REP_POOLING"
	--spectral_dim "$SPECTRAL_DIM"
	--spectral_embedding_batch_size "$SPECTRAL_EMBEDDING_BATCH_SIZE"
)
if [[ -n "$SPECTRAL_MAX_SEQ_LEN" ]]; then
	SPECTRAL_FLAGS+=(--max_seq_len "$SPECTRAL_MAX_SEQ_LEN")
fi

LLM_FEATURE_FLAG=()
if [[ "$NO_LLM_FEATURE_GENERATION" == "true" ]]; then
	LLM_FEATURE_FLAG=(--no_llm_feature_generation)
fi

#####################################
CIRCUIT_DISCOVERY_METHOD=EAP-IG-inputs
CIRCUIT_INTERVENTION=patching

# lowercase + replace '-' with '_'
CIRCUIT_DISCOVERY_METHOD_NORM=$(
  printf '%s' "$CIRCUIT_DISCOVERY_METHOD" | tr '[:upper:]-' '[:lower:]_'
)
CIRCUIT_DISCOVERY_OUTPUT_DIR="$DATA_DIR/neural_circuit_discovery_results${VARIANT_SUFFIX}/$CIRCUIT_DISCOVERY_METHOD_NORM"

RULES_DIR="$DATA_DIR/rule_extraction_results"
POINTS_TO_USE_FOR_MEAN_ABLATION="${POINTS_TO_USE_FOR_MEAN_ABLATION:-256}"
MAX_POINTS_PER_CIRCUIT="${MAX_POINTS_PER_CIRCUIT:-128}"
if [[ -n "${CHA_REFERENCE_N_PER_SIDE:-}" && -z "${SEARCH_EPSILON_REFERENCE_N:-}" ]]; then
	export SEARCH_EPSILON_REFERENCE_N="$CHA_REFERENCE_N_PER_SIDE"
fi
MAX_POINTS_PER_ABLATION="${MAX_POINTS_PER_ABLATION:-${CHA_REFERENCE_N_PER_SIDE:-64}}"

if [[ "$EVAL_INTERVENTION" == *donor* && "$POINTS_TO_USE_FOR_MEAN_ABLATION" -lt 2048 ]]; then
	POINTS_TO_USE_FOR_MEAN_ABLATION=2048
fi

# Stage 5 uses the canonical mean intervention names during circuit discovery.
# Normalize donor-style evaluation labels only for that discovery stage.
# Downstream output folders include the effective eval intervention whenever it differs from `mean`.
SCRIPT5_EVAL_INTERVENTION="$EVAL_INTERVENTION"
if [[ "$SCRIPT5_EVAL_INTERVENTION" == "mean-donor" ]]; then
	SCRIPT5_EVAL_INTERVENTION="mean"
elif [[ "$SCRIPT5_EVAL_INTERVENTION" == "mean-donor-positional" ]]; then
	SCRIPT5_EVAL_INTERVENTION="mean-positional"
fi

OUTPUT_EVAL_INTERVENTION_SUFFIX=""
# Preserve historical paths for mean-family replacement. The exact baseline is
# recorded in evaluation_scope.json and interaction_configuration.json and is
# passed explicitly to every intervention analysis.
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
# Preserve historical names for the default all-neuron / neuron-level case.
# Non-default circuit semantics get distinct discovery and downstream paths.
# if [[ "$NEURONS_TYPE" == "mlp" ]]; then
# 	CIRCUIT_LABEL+="-mlp_only"
# fi
if [[ "$CIRCUIT_LEVEL" != "neuron" ]]; then
	CIRCUIT_LABEL+="-${CIRCUIT_LEVEL}"
fi
if [[ -n "$OUTPUT_EVAL_INTERVENTION_SUFFIX" ]]; then
	CIRCUIT_LABEL+="$OUTPUT_EVAL_INTERVENTION_SUFFIX"
fi
echo $CIRCUIT_LABEL
DISCOVERY_OUT_DIR="$CIRCUIT_DISCOVERY_OUTPUT_DIR/$CIRCUIT_LABEL"

# Stage-4 sampling-plan identity is owned by core.spectral_sampling_plan.
# The shell passes only configuration values; producers/consumers resolve the
# same canonical path in Python.
SAMPLING_PLAN_COMMON_FLAGS=(
	--sampling_plan_dir "$DISCOVERY_OUT_DIR"
	--sampling_plan_max_points "$SPECTRAL_PLAN_MAX_POINTS"
	--sampling_plan_coverage_radius "$SPECTRAL_COVERAGE_RADIUS"
	--sampling_plan_pair_len_tolerance "$SPECTRAL_PAIR_LEN_TOLERANCE"
	--sampling_plan_seed "$SPECTRAL_PLAN_SEED"
	--sampling_plan_stats_sample_size "$SPECTRAL_STATS_SAMPLE_SIZE"
	--sampling_plan_stats_chunk_size "$SPECTRAL_STATS_CHUNK_SIZE"
)

############################################
# Steps 1-3: always run
if [[ "$NO_LLM_FEATURE_GENERATION" != "true" ]]; then
	OLLAMA_LOG="${OLLAMA_LOG:-$PROJECT_ROOT/cache/logs/ollama.log}"
	mkdir -p "$(dirname "$OLLAMA_LOG")"
	ollama serve > "$OLLAMA_LOG" 2>&1 &
fi

if [[ "$SKIP_STAGE1" == "true" ]]; then
	if [[ ! -s "$PROMPTS_ANSWERS_PKL_FILE" || ! -s "$FEATURES_SCORES_DIR/scores.csv" ]]; then
		echo "ERROR: --skip_stage1 requested but Stage-1 artifacts are missing:" >&2
		echo "  cache:  $PROMPTS_ANSWERS_PKL_FILE" >&2
		echo "  scores: $FEATURES_SCORES_DIR/scores.csv" >&2
		exit 1
	fi
	echo "Step 1: reusing precomputed behavior cache + scores.csv"
else
	STAGE1_CMD=(python3 -m pipeline.stage01_generate_prompts_and_answers
		--ai_model "$ANALYZED_LLM"
		--task_module "$TASK_MODULE"
		--prompts_answers_pkl_file "$PROMPTS_ANSWERS_PKL_FILE"
		--batch_size "$BATCH_SIZE"
		--stats_json_out "$FEATURES_SCORES_DIR")
	if [[ -n "$HF_MODEL_CACHE_DIR" ]]; then
		STAGE1_CMD+=(--ai_model_cache_dir "$HF_MODEL_CACHE_DIR")
	fi
	if [[ "$SPLITS" == "spectral" && "$TASK_MODULE" == studies.poisoning.tasks.* ]]; then
		# Poisoning spectral runs need only the cached behavioral dataset.
		STAGE1_CMD+=(--export_dataset_scores_dir "$FEATURES_SCORES_DIR")
	fi
	"${STAGE1_CMD[@]}"
fi

# Step 2: generic experiments perform feature engineering. Poisoning spectral
# runs already materialized scores.csv in Stage 1.
if [[ "$SPLITS" == "spectral" && "$TASK_MODULE" == studies.poisoning.tasks.* ]]; then
	echo "Step 2: poisoning dataset table exported from the Stage 1 cache object"
elif [[ -s "$FEATURES_SCORES_DIR/scores.csv" ]]; then
	echo "Step 2: found $FEATURES_SCORES_DIR/scores.csv -> reusing dataset table"
else
	# Generic experiments still run normal feature engineering.
	# If z_thresh is negative, do *not* drop by MAD or pass the flag
	if awk "BEGIN {exit !($Z_THRESH >= 0)}"; then
		python3 -m pipeline.stage02_generate_features \
			--ai_model "$FEATURE_GENERATION_LLM" \
			--task_module "$TASK_MODULE" \
			--prompts_answers_pkl_file "$PROMPTS_ANSWERS_PKL_FILE" \
			--features_scores_dir "$FEATURES_SCORES_DIR" \
			--cache_dir "$EXPERIMENT_LLM_CACHE_DIR" \
			"${HF_MODEL_CACHE_FLAG[@]}" \
			--num_correct_example_prompts 32 \
			--num_incorrect_example_prompts 32 \
			"${LLM_FEATURE_FLAG[@]}" \
			--drop_near_duplicate_features \
			--near_duplicate_features_threshold 0.9999 \
			--drop_low_predictive_power_features \
			--min_delta 0.2 \
			--drop_high_mad_variance_features \
			--z_thresh "$Z_THRESH"
	else
		python3 -m pipeline.stage02_generate_features \
			--ai_model "$FEATURE_GENERATION_LLM" \
			--task_module "$TASK_MODULE" \
			--prompts_answers_pkl_file "$PROMPTS_ANSWERS_PKL_FILE" \
			--features_scores_dir "$FEATURES_SCORES_DIR" \
			--cache_dir "$EXPERIMENT_LLM_CACHE_DIR" \
			"${HF_MODEL_CACHE_FLAG[@]}" \
			--num_correct_example_prompts 32 \
			--num_incorrect_example_prompts 32 \
			"${LLM_FEATURE_FLAG[@]}" \
			--drop_near_duplicate_features \
			--near_duplicate_features_threshold 0.9999 \
			--drop_low_predictive_power_features \
			--min_delta 0.2
	fi
fi

############################################
# Step 3: extract rules (skip if CSVs already exist)
if [[ "$SPLITS" == "spectral" ]]; then
	echo "Step 3: spectral_splits -> skipping stage03_extract_rules.py"
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
		echo "Step 3: found ${matches[0]##*/} in $RULES_DIR -> skipping stage03_extract_rules.py"
	else
		cmd=(python3 -m pipeline.stage03_extract_rules
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
	echo "Step 4: spectral_splits -> skipping stage04_spectral_sample_datapoints.py"
else
	NEED_ALL_PLAN=false
	NEED_BASELINE_PLANS=false

	if [[ "$DISCOVERY" == "spectral" ]]; then
		NEED_ALL_PLAN=true
	fi
	if [[ "$PLAN" == "spectral" ]]; then
		NEED_BASELINE_PLANS=true
	fi

	if [[ "$NEED_ALL_PLAN" == "true" ]]; then
		python3 -m pipeline.stage04_spectral_sample_datapoints \
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
			--max_points_per_ablation "$SPECTRAL_PLAN_MAX_POINTS" \
			--coverage_radius "$SPECTRAL_COVERAGE_RADIUS" \
			--seed "$SPECTRAL_PLAN_SEED" \
			--pair_len_tolerance "$SPECTRAL_PAIR_LEN_TOLERANCE" \
			--use_global_clusters \
			--global_n_clusters "$((MAX_POINTS_PER_CIRCUIT / 4))" \
			--output_dir "$DISCOVERY_OUT_DIR" \
			--plan_role circuit_discovery \
			--skip_existing \
			--compute_cover_stats \
			--stats_sample_size "$SPECTRAL_STATS_SAMPLE_SIZE" \
			--stats_chunk_size "$SPECTRAL_STATS_CHUNK_SIZE" \
			${FAKE_FLAG[@]}
	fi

	if [[ "$NEED_BASELINE_PLANS" == "true" ]]; then
		python3 -m pipeline.stage04_spectral_sample_datapoints \
			--task_module "$TASK_MODULE" \
			--ai_model "$ANALYZED_LLM" \
			--spectral_cache_dir $CACHE_DIR \
			"${SPECTRAL_FLAGS[@]}" \
			--features_scores_dir "$FEATURES_SCORES_DIR" \
			--rules_glob "optimal_rule_set" \
			--rules_dir "$RULES_DIR" \
			--baseline_subset positive \
			--min_points_per_ablation "$MAX_POINTS_PER_ABLATION" \
			--max_points_per_ablation "$SPECTRAL_PLAN_MAX_POINTS" \
			--coverage_radius "$SPECTRAL_COVERAGE_RADIUS" \
			--seed "$SPECTRAL_PLAN_SEED" \
			--pair_len_tolerance "$SPECTRAL_PAIR_LEN_TOLERANCE" \
			--use_global_clusters \
			--global_n_clusters "$((MAX_POINTS_PER_ABLATION / 4))" \
			--output_dir "$DISCOVERY_OUT_DIR" \
			--plan_role neuron_ablation_positive \
			--skip_existing \
			--compute_cover_stats \
			--stats_sample_size "$SPECTRAL_STATS_SAMPLE_SIZE" \
			--stats_chunk_size "$SPECTRAL_STATS_CHUNK_SIZE" \
			${FAKE_FLAG[@]}

		python3 -m pipeline.stage04_spectral_sample_datapoints \
			--task_module "$TASK_MODULE" \
			--ai_model "$ANALYZED_LLM" \
			--spectral_cache_dir $CACHE_DIR \
			"${SPECTRAL_FLAGS[@]}" \
			--features_scores_dir "$FEATURES_SCORES_DIR" \
			--rules_glob "optimal_rule_set" \
			--rules_dir "$RULES_DIR" \
			--baseline_subset negative \
			--min_points_per_ablation "$MAX_POINTS_PER_ABLATION" \
			--max_points_per_ablation "$SPECTRAL_PLAN_MAX_POINTS" \
			--coverage_radius "$SPECTRAL_COVERAGE_RADIUS" \
			--seed "$SPECTRAL_PLAN_SEED" \
			--pair_len_tolerance "$SPECTRAL_PAIR_LEN_TOLERANCE" \
			--use_global_clusters \
			--global_n_clusters "$((MAX_POINTS_PER_ABLATION / 4))" \
			--output_dir "$DISCOVERY_OUT_DIR" \
			--plan_role neuron_ablation_negative \
			--skip_existing \
			--compute_cover_stats \
			--stats_sample_size "$SPECTRAL_STATS_SAMPLE_SIZE" \
			--stats_chunk_size "$SPECTRAL_STATS_CHUNK_SIZE" \
			${FAKE_FLAG[@]}
	fi
fi

############################################
# Step 5: Circuit discovery (mutually exclusive: random vs spectral)
DISCOVERY_INPUT_AUTODISCOVERY_DIR="$DISCOVERY_OUT_DIR/neural_circuits"

if [[ "$SPLITS" == "spectral" ]]; then
	python3 -m pipeline.stage05_discover_circuits \
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
		--batch_size 4 \
		--spectral_cache_dir $CACHE_DIR \
		--cluster_by_spectral \
		--cluster_base_subset "$SPECTRAL_CLUSTER_BASE_SUBSET" \
		"${HF_MODEL_CACHE_FLAG[@]}" \
		"${SPECTRAL_FLAGS[@]}" \
		--global_n_clusters "$MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE" \
		"${DECODE_FLAG[@]}" \
		"${FAKE_FLAG[@]}" \
		"${NEURONS_TYPE_FLAG[@]}"
else

	if [[ "$DISCOVERY" == "spectral" ]]; then
		python3 -m pipeline.stage05_discover_circuits \
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
			"${HF_MODEL_CACHE_FLAG[@]}" \
			--max_pairs_per_circuit $MAX_POINTS_PER_CIRCUIT \
			--sampling_strategy plan \
			--sampling_plan_role circuit_discovery \
			--sampling_plan_min_points "$MAX_POINTS_PER_CIRCUIT" \
			--sampling_plan_global_n_clusters "$((MAX_POINTS_PER_CIRCUIT / 4))" \
			"${SAMPLING_PLAN_COMMON_FLAGS[@]}" \
			"${SPECTRAL_FLAGS[@]}" \
			--batch_size 4 \
			--max_n_of_rules_to_analyze $MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE \
			"${DECODE_FLAG[@]}" \
			"${FAKE_FLAG[@]}" \
			"${NEURONS_TYPE_FLAG[@]}"
	else
		python3 -m pipeline.stage05_discover_circuits \
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
			"${HF_MODEL_CACHE_FLAG[@]}" \
			--max_pairs_per_circuit $MAX_POINTS_PER_CIRCUIT \
			--batch_size 4 \
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
if [[ "${PLAN:-}" == "spectral" ]]; then
  BAG_LABEL+="-spectral_anchor"
elif [[ "${PLAN:-}" == "random" ]]; then
  BAG_LABEL+="-random_anchor"
fi
if [[ "$MIN_FLIP_RATE" != "0.2" ]]; then
  BAG_LABEL+="-tau${MIN_FLIP_RATE}"
fi
# Poisoning defines tau at an explicit reference sample size.  The output
# identity must distinguish not only an adaptive smaller-n run, but also a
# user-changed reference (for example 128/side instead of the historical
# 64/side).  Keep the historical n=64/ref=64 path unchanged for resume
# compatibility; every other reference/sample combination is explicit.
if [[ -n "${SEARCH_EPSILON_REFERENCE_N:-}" ]]; then
  if [[ "$MAX_POINTS_PER_ABLATION" != "$SEARCH_EPSILON_REFERENCE_N" ]] || [[ "$SEARCH_EPSILON_REFERENCE_N" != "64" ]]; then
    BAG_LABEL+="-n${MAX_POINTS_PER_ABLATION}-epsref${SEARCH_EPSILON_REFERENCE_N}"
  fi
fi
echo $BAG_LABEL

# Helper to run analyze_bag_of_rules for a baseline subset
run_analyze() {
	local baseline_subset="$1"
	local out_dir="$2"

	if [[ "$SPLITS" == "spectral" ]]; then
		python3 -m pipeline.stage06_analyze_bag_of_rules \
			--spectral_cache_dir $CACHE_DIR \
			--cluster_by_spectral \
			"${SPECTRAL_FLAGS[@]}" \
			--global_n_clusters "$MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE" \
			--input_data_dir "$DISCOVERY_INPUT_AUTODISCOVERY_DIR" \
			--output_data_dir "$out_dir" \
			--task_module "$TASK_MODULE" \
			"${HF_MODEL_CACHE_FLAG[@]}" \
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
		python3 -m pipeline.stage06_analyze_bag_of_rules \
			--input_data_dir "$DISCOVERY_INPUT_AUTODISCOVERY_DIR" \
			--output_data_dir "$out_dir" \
			--task_module "$TASK_MODULE" \
			"${HF_MODEL_CACHE_FLAG[@]}" \
			--n_associated $MAX_POINTS_PER_ABLATION \
			--n_unrelated $MAX_POINTS_PER_ABLATION \
			--batch_size "$BATCH_SIZE" \
			--search_epsilon $MIN_FLIP_RATE \
			--points_to_use_for_mean_ablation "$POINTS_TO_USE_FOR_MEAN_ABLATION" \
			--intervention $EVAL_INTERVENTION \
			"${FAST_FLAG[@]}" \
			--baseline_subset "$baseline_subset" \
			--sampling_strategy plan \
			--sampling_plan_role "neuron_ablation_${baseline_subset}" \
			--sampling_plan_min_points "$MAX_POINTS_PER_ABLATION" \
			--sampling_plan_global_n_clusters "$((MAX_POINTS_PER_ABLATION / 4))" \
			"${SAMPLING_PLAN_COMMON_FLAGS[@]}" \
			"${SPECTRAL_FLAGS[@]}" \
			--sign_split_first \
			"${DECODE_FLAG[@]}" \
			"${NEURONS_TYPE_FLAG[@]}"
	else
		python3 -m pipeline.stage06_analyze_bag_of_rules \
			--input_data_dir "$DISCOVERY_INPUT_AUTODISCOVERY_DIR" \
			--output_data_dir "$out_dir" \
			--task_module "$TASK_MODULE" \
			"${HF_MODEL_CACHE_FLAG[@]}" \
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

# Run anchoring for requested baselines. Most standard experiments use both;
# trigger-lift pilots often only need positive (successful trigger-lift rows).
if [[ ",$ANALYZE_BASELINE_SUBSETS," == *",positive,"* ]]; then
	run_analyze "positive" "$DISCOVERY_OUT_DIR/$BAG_LABEL/positive_baseline"
fi
if [[ ",$ANALYZE_BASELINE_SUBSETS," == *",negative,"* ]]; then
	run_analyze "negative" "$DISCOVERY_OUT_DIR/$BAG_LABEL/negative_baseline"
fi

CIRCUIT_BAG_LABEL="$CIRCUIT_LABEL-$BAG_LABEL"
if [[ "$INCORRECT_RULES" == "true" ]]; then
	CIRCUIT_BAG_LABEL+="-fake_targets"
fi
# Evaluation split is part of the output identity. Test keeps the established
# -heldout_test suffix; all preserves the historical unsuffixed path.
case "$EVALUATION_SPLIT" in
	test) CIRCUIT_BAG_LABEL+="-heldout_test" ;;
	train) CIRCUIT_BAG_LABEL+="-eval_train" ;;
	all) ;;
esac
if [[ "$EVALUATION_BASELINE_SUBSET" != "all" ]]; then
	CIRCUIT_BAG_LABEL+="-baseline_${EVALUATION_BASELINE_SUBSET}"
fi
# REFINE_SAMPLING_MAX_POINTS=10000 has been the pipeline default historically.
# Keep that default implicit in the dirname for backward compatibility. A
# non-default cap changes the evaluated sample and therefore gets an explicit
# output-identity suffix so different caps cannot collide.
if (( REFINE_SAMPLING_MAX_POINTS != 10000 )); then
	CIRCUIT_BAG_LABEL+="-cap${REFINE_SAMPLING_MAX_POINTS}"
fi
EVALUATION_SPLIT_FLAG=(--evaluation_split "$EVALUATION_SPLIT" --evaluation_baseline_subset "$EVALUATION_BASELINE_SUBSET")
SINGLETON_SCHEMA_OK=false
UNCERTAINTY_SCHEMA_OK=false
STAGE7_COMPLETE=false
STATS_DIR="$RULES_DIR/neuron_flip_rules/stats/$CIRCUIT_BAG_LABEL"
if [[ -s "$STATS_DIR/flip_stats_global.json" ]]; then
	if python3 -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); need=("n_evaluated_rows","union_flip_any_unique_ci_low","union_flip_any_unique_ci_high","confidence_level"); raise SystemExit(0 if all(k in p for k in need) else 1)' "$STATS_DIR/flip_stats_global.json"; then
		UNCERTAINTY_SCHEMA_OK=true
	fi
fi
if [[ -s "$STATS_DIR/singleton_set_metrics.json" ]]; then
	if python3 -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if p.get("definition_version") == "heldout-set-metrics-v3-directional" else 1)' "$STATS_DIR/singleton_set_metrics.json"; then
		SINGLETON_SCHEMA_OK=true
	fi
fi
SCOPE_SPLIT_OK=false
if [[ -s "$STATS_DIR/evaluation_scope.json" ]]; then
	if python3 -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); expected_excl=sys.argv[4].lower() in {"1","true","yes"}; expected_cap=int(sys.argv[5]); actual_cap=int(p.get("sampling_max_points") or 0); raise SystemExit(0 if p.get("final_statistics_split") == sys.argv[2] and p.get("evaluation_baseline_subset", "all") == sys.argv[3] and bool(p.get("exclude_discovery_rows_from_final_stats", False)) == expected_excl and actual_cap == max(0, expected_cap) else 1)' "$STATS_DIR/evaluation_scope.json" "$EVALUATION_SPLIT" "$EVALUATION_BASELINE_SUBSET" "$REFINE_EXCLUDE_DISCOVERY_ROWS_FROM_FINAL_STATS" "$REFINE_SAMPLING_MAX_POINTS"; then
		SCOPE_SPLIT_OK=true
	fi
fi
if [[ -s "$STATS_DIR/flip_stats_global.json" \
   && -s "$STATS_DIR/flip_stats_by_neuron.csv" \
   && -s "$STATS_DIR/scores.csv" \
   && "$SINGLETON_SCHEMA_OK" == "true" \
   && "$UNCERTAINTY_SCHEMA_OK" == "true" \
   && "$SCOPE_SPLIT_OK" == "true" \
   && -s "$STATS_DIR/frozen_candidate_ranking.csv" ]]; then
	STAGE7_COMPLETE=true
fi
if [[ "$RUN_SINGLETON_CAUSAL_EVALUATION" == "true" || "$RUN_SINGLETON_CAUSAL_EVALUATION" == "1" ]]; then
	REFINE_FLAGS=(
		--task_module "$TASK_MODULE"
		--ai_model "$ANALYZED_LLM"
		"${HF_MODEL_CACHE_FLAG[@]}"
		--rules_dir "$RULES_DIR/neuron_flip_rules"
		--features_scores_dir "$FEATURES_SCORES_DIR"
		--circuit_agonists_path "$DISCOVERY_OUT_DIR/$BAG_LABEL"
		--search_epsilon $MIN_FLIP_RATE
		--batch_size "$BATCH_SIZE"
		--neuron_batch_size "$REFINE_NEURON_BATCH_SIZE"
		--sampling_max_points "$REFINE_SAMPLING_MAX_POINTS"
		--stats_dirname "$CIRCUIT_BAG_LABEL"
		"${SPECTRAL_FLAGS[@]}"
		--global_n_clusters "$MAX_POINTS_PER_ABLATION"
		--points_to_use_for_mean_ablation "$POINTS_TO_USE_FOR_MEAN_ABLATION"
		--intervention $EVAL_INTERVENTION
		--only_unique_datapoints_in_shap
		"${EVALUATION_SPLIT_FLAG[@]}"
		"${DECODE_FLAG[@]}"
	)
	if [[ "$REFINE_EXCLUDE_DISCOVERY_ROWS_FROM_FINAL_STATS" == "true" || "$REFINE_EXCLUDE_DISCOVERY_ROWS_FROM_FINAL_STATS" == "1" ]]; then
		REFINE_FLAGS+=(--exclude_discovery_rows_from_final_stats)
	fi
	if [[ "$REFINE_USE_SPECTRAL_SAMPLING" == "true" || "$REFINE_USE_SPECTRAL_SAMPLING" == "1" ]]; then
		REFINE_FLAGS+=(
			--use_spectral_sampling
			--spectral_cache_dir "$CACHE_DIR"
		)
	fi
	if [[ "$REFINE_EXTRACT_RULES" == "true" || "$REFINE_EXTRACT_RULES" == "1" ]]; then
		REFINE_FLAGS+=(--extract_rules)
	fi
	if [[ "$REFINE_SUMMARIZE_RULE_METRICS" == "true" || "$REFINE_SUMMARIZE_RULE_METRICS" == "1" ]]; then
		REFINE_FLAGS+=(--summarize_rule_metrics)
	fi
	if [[ "$SKIP_AGONIST_METRIC_STATS" == "true" || "$SKIP_AGONIST_METRIC_STATS" == "1" ]]; then
		REFINE_FLAGS+=(--skip_agonist_metric_stats)
	fi
	if [[ "$REFINE_MAX_NEURONS" != "0" && "$REFINE_MAX_NEURONS" != "" ]]; then
		REFINE_FLAGS+=(--max_neurons "$REFINE_MAX_NEURONS")
	fi

	if [[ "$STAGE7_COMPLETE" == "true" && "$FORCE_STAGE7" != "true" && "$FORCE_STAGE7" != "1" ]]; then
		echo "Step 7: complete singleton artifacts for split=$EVALUATION_SPLIT found -> reusing $STATS_DIR"
	else
		python3 -m pipeline.stage07_singleton_causal_evaluation "${REFINE_FLAGS[@]}"
	fi
else
	echo "Step 7: RUN_SINGLETON_CAUSAL_EVALUATION=$RUN_SINGLETON_CAUSAL_EVALUATION -> skipping stage07_singleton_causal_evaluation.py"
fi

# RQ3 primary spiking experiment: graded intervention on each frozen agonist's
# own held-out directional flip support.  This is intentionally distinct from
# the optional scalar/threshold diagnostics below: no classifier is fit to
# predict which examples the agonist flips.
if [[ "$RUN_GRADED_AGONIST_INTERVENTION" == "true" || "$RUN_GRADED_AGONIST_INTERVENTION" == "1" ]]; then
	if [[ -s "$STATS_DIR/flip_stats_by_neuron.csv" && -s "$STATS_DIR/scores.csv" && -s "$STATS_DIR/frozen_candidate_ranking.csv" ]]; then
		GRADED_AGONIST_OUT_DIR="$STATS_DIR/graded_agonist_intervention"
		GRADED_AGONIST_FLAGS=(
			--input_data_dir "$DISCOVERY_INPUT_AUTODISCOVERY_DIR"
			--candidate_flip_stats_path "$STATS_DIR/flip_stats_by_neuron.csv"
			--singleton_scores_path "$STATS_DIR/scores.csv"
			--candidate_ranking_path "$STATS_DIR/frozen_candidate_ranking.csv"
			--out_dir "$GRADED_AGONIST_OUT_DIR"
			--task_module "$TASK_MODULE"
			--ai_model "$ANALYZED_LLM"
			"${HF_MODEL_CACHE_FLAG[@]}"
			--evaluation_split "$EVALUATION_SPLIT"
			--intervention "$EVAL_INTERVENTION"
			--points_to_use_for_mean_ablation "$POINTS_TO_USE_FOR_MEAN_ABLATION"
			--batch_size "$BATCH_SIZE"
			--doses "$GRADED_AGONIST_DOSES"
			--max_agonists_per_direction "$GRADED_AGONIST_MAX_UNITS_PER_DIRECTION"
			--max_positive_support_per_agonist "$GRADED_AGONIST_MAX_POSITIVE_SUPPORT"
			--negative_support_ratio "$GRADED_AGONIST_NEGATIVE_RATIO"
			--max_negative_support_per_agonist "$GRADED_AGONIST_MAX_NEGATIVE_SUPPORT"
			--seed "$GRADED_AGONIST_SEED"
			"${DECODE_FLAG[@]}"
		)
		if [[ "$GRADED_AGONIST_NEGATIVE_SUPPORT" == "true" || "$GRADED_AGONIST_NEGATIVE_SUPPORT" == "1" ]]; then
			GRADED_AGONIST_FLAGS+=(--same_agonist_negative_support)
		fi
		if [[ "$FORCE_GRADED_AGONIST_INTERVENTION" == "true" || "$FORCE_GRADED_AGONIST_INTERVENTION" == "1" ]]; then
			GRADED_AGONIST_FLAGS+=(--force)
		fi
		echo "Step 7b: graded agonist intervention -> $GRADED_AGONIST_OUT_DIR"
		python3 -m studies.overtopping.analysis.graded_agonist_intervention "${GRADED_AGONIST_FLAGS[@]}"
	else
		echo "Step 7b: graded agonist intervention skipped; complete Stage-7 singleton artifacts are missing"
	fi
else
	echo "Step 7b: RUN_GRADED_AGONIST_INTERVENTION=$RUN_GRADED_AGONIST_INTERVENTION -> skipping graded agonist intervention"
fi

# Legacy scalar-to-flip diagnostics. These are not part of the RQ3 spiking test.
# They are disabled by default and retained only for explicit historical/diagnostic runs.
# The primary RQ3 experiment is the graded agonist intervention above.
if [[ "$RUN_THRESHOLD_EVENT_POSTHOC" == "true" || "$RUN_THRESHOLD_EVENT_POSTHOC" == "1" ]]; then
	if [[ -s "$STATS_DIR/flip_stats_by_neuron.csv" ]]; then
		THRESHOLD_EVENT_OUT_LABEL="spiking_diagnostics-${BAG_LABEL}"
		case "$EVALUATION_SPLIT" in
			test) ;;
			train) THRESHOLD_EVENT_OUT_LABEL+="-eval_train" ;;
			all) THRESHOLD_EVENT_OUT_LABEL+="-eval_all" ;;
		esac
		if [[ "$THRESHOLD_EVENT_MAX_POINTS" != "10000" ]]; then
			THRESHOLD_EVENT_OUT_LABEL+="-cap${THRESHOLD_EVENT_MAX_POINTS}"
		fi
		THRESHOLD_EVENT_OUT_DIR="$DISCOVERY_INPUT_AUTODISCOVERY_DIR/$THRESHOLD_EVENT_OUT_LABEL"
		THRESHOLD_EVENT_FLAGS=(
			--input_data_dir "$DISCOVERY_INPUT_AUTODISCOVERY_DIR"
			--out_dir "$THRESHOLD_EVENT_OUT_DIR"
			--baseline_subsets "$ANALYZE_BASELINE_SUBSETS"
			--task_module "$TASK_MODULE"
			--ai_model "$ANALYZED_LLM"
			"${HF_MODEL_CACHE_FLAG[@]}"
			--candidate_flip_stats_path "$STATS_DIR/flip_stats_by_neuron.csv"
			--candidate_ranking_path "$STATS_DIR/frozen_candidate_ranking.csv"
			--materialized_stage7_scores_path "$STATS_DIR/scores.csv"
			--evaluation_split "$EVALUATION_SPLIT"
			--intervention "$EVAL_INTERVENTION"
			--points_to_use_for_mean_ablation "$POINTS_TO_USE_FOR_MEAN_ABLATION"
			--batch_size "$BATCH_SIZE"
			--spiking_max_points "$THRESHOLD_EVENT_MAX_POINTS"
			--spiking_min_points "$THRESHOLD_EVENT_MIN_POINTS"
			--spiking_global_n_clusters "$MAX_POINTS_PER_ABLATION"
			--threshold_event_repeats "$THRESHOLD_EVENT_REPEATS"
			--threshold_event_holdout_fraction "$THRESHOLD_EVENT_HOLDOUT_FRACTION"
			--threshold_event_n_bins "$THRESHOLD_EVENT_N_BINS"
			--target "$THRESHOLD_EVENT_TARGET"
			--seed "$THRESHOLD_EVENT_SEED"
			--spectral_cache_dir "$CACHE_DIR/threshold_events/$CIRCUIT_BAG_LABEL"
			"${SPECTRAL_FLAGS[@]}"
			"${DECODE_FLAG[@]}"
		)
		if [[ "$FORCE_THRESHOLD_EVENT_POSTHOC" == "true" || "$FORCE_THRESHOLD_EVENT_POSTHOC" == "1" ]]; then
			THRESHOLD_EVENT_FLAGS+=(--force_threshold_event --force_spiking_eval --no_skip_existing)
		fi
		echo "Step 7c: optional scalar/threshold diagnostics -> $THRESHOLD_EVENT_OUT_DIR"
		python3 -m studies.overtopping.analysis.threshold_event_diagnostics "${THRESHOLD_EVENT_FLAGS[@]}"
	else
		echo "Step 7c: optional scalar/threshold diagnostics skipped; missing $STATS_DIR/flip_stats_by_neuron.csv"
	fi
else
	echo "Step 7c: RUN_THRESHOLD_EVENT_POSTHOC=$RUN_THRESHOLD_EVENT_POSTHOC -> skipping threshold_event_diagnostics.py"
fi

if [[ "$RUN_INTERACTION_VALIDATION" == "true" || "$RUN_INTERACTION_VALIDATION" == "1" ]]; then
	if [[ -s "$STATS_DIR/flip_stats_by_neuron.csv" && -s "$STATS_DIR/scores.csv" && -s "$STATS_DIR/frozen_candidate_ranking.csv" ]]; then
		INTERACTION_FLAGS=(
			--input_data_dir "$DISCOVERY_INPUT_AUTODISCOVERY_DIR"
			--candidate_flip_stats_path "$STATS_DIR/flip_stats_by_neuron.csv"
			--singleton_scores_path "$STATS_DIR/scores.csv"
			--frozen_ranking_path "$STATS_DIR/frozen_candidate_ranking.csv"
			--out_dir "$STATS_DIR/interaction_validation"
			--task_module "$TASK_MODULE"
			--ai_model "$ANALYZED_LLM"
			--intervention "$EVAL_INTERVENTION"
			--batch_size "$BATCH_SIZE"
			--points_to_use_for_mean_ablation "$POINTS_TO_USE_FOR_MEAN_ABLATION"
			--background_multipliers "$CONDITIONAL_BACKGROUND_MULTIPLIERS"
			--null_draws "$INTERACTION_NULL_DRAWS"
			--preemption_min_singleton_rate "$PREEMPTION_MIN_SINGLETON_RATE"
			--preemption_max_secondaries "$PREEMPTION_MAX_SECONDARIES"
			--preemption_threshold_holdout_fraction "$PREEMPTION_THRESHOLD_HOLDOUT_FRACTION"
			--preemption_threshold_min_class "$PREEMPTION_THRESHOLD_MIN_CLASS"
			--evaluation_split "$EVALUATION_SPLIT"
			"${DECODE_FLAG[@]}"
		)
		if [[ -n "${THRESHOLD_EVENT_OUT_DIR:-}" && -d "${THRESHOLD_EVENT_OUT_DIR:-}" ]]; then
			INTERACTION_FLAGS+=(--threshold_diagnostics_dir "$THRESHOLD_EVENT_OUT_DIR")
		fi
		if [[ "$RUN_CMC" == "false" || "$RUN_CMC" == "0" ]]; then
			INTERACTION_FLAGS+=(--skip_cmc)
		fi
		if [[ "$RUN_PREEMPTION" == "false" || "$RUN_PREEMPTION" == "0" ]]; then
			INTERACTION_FLAGS+=(--skip_preemption)
		fi
		if [[ "$FORCE_INTERACTION_VALIDATION" == "true" || "$FORCE_INTERACTION_VALIDATION" == "1" ]]; then
			INTERACTION_FLAGS+=(--force)
		fi
		python3 -m pipeline.stage08_validate_interactions "${INTERACTION_FLAGS[@]}"
	else
		echo "Conditional validation skipped: complete stage-7 artifacts were not found in $STATS_DIR"
	fi
else
	echo "Conditional validation disabled by RUN_INTERACTION_VALIDATION=$RUN_INTERACTION_VALIDATION"
fi

echo "Done."
