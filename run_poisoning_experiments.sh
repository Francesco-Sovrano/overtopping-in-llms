#!/usr/bin/env bash
set -euo pipefail

# One holdout split must be shared across all training seeds so cross-seed
# clean-null U(j) inference is paired on the same immutable examples.
export POISONING_HOLDOUT_SEED="${POISONING_HOLDOUT_SEED:-13}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$PROJECT_ROOT/code"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/.env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  . "$PROJECT_ROOT/.env/bin/activate"
fi

# Optional provider credentials. The file is shell syntax and should contain
# entries such as: export GROQ_API_KEY="...". Existing environment variables
# remain available and can be used without this file.
SECRETS_FILE="${SECRETS_FILE:-$PROJECT_ROOT/.secrets.env}"
if [[ -f "$SECRETS_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a
  . "$SECRETS_FILE"
  set +a
fi
###########################
export API_MAX_RETRIES=0
export API_RECOVERY_PASSES=3
# export POISONING_FAST_TEST=1

export RUN_GRADED_AGONIST_INTERVENTION=false
export RUN_THRESHOLD_EVENT_POSTHOC=false
export RUN_PREEMPTION=false
export RUN_INTERACTION_VALIDATION=false
export RUN_CMC=false
###
export CHA_REFERENCE_N_PER_SIDE=64
export CHA_TAU=0.3
export CHA_LOW_DATA_POLICY=skip
export MODEL_NAMES=Qwen/Qwen2-1.5B-Instruct
export RUN_OBSERVED_MIXTURE_OVERTOPPING="${RUN_OBSERVED_MIXTURE_OVERTOPPING:-1}"
# Candidate localization source. Keep observed-mixture as the attack-agnostic default;
# set to attack_cohort_control_correctness for the fixed non-target/no-trigger CHA.
export POISONING_CANDIDATE_LOCALIZATION_ENDPOINT="${POISONING_CANDIDATE_LOCALIZATION_ENDPOINT:-observed_training_mixture_correctness}"
# Optional speed knob for attack-cohort CHA; values are checkpoint percentages.
export ATTACK_COHORT_CONTROL_CHA_PROGRESS_PCTS="${ATTACK_COHORT_CONTROL_CHA_PROGRESS_PCTS:-all}"
# Keep trigger-lift behavior measurement, but skip trigger-lift CHA/circuit discovery by default.
export RUN_TRIGGER_LIFT_CHA=0
export POISON_RATE=0.1
export POISON_RATE_BASIS=eligible_gold_non_target
export CONTROL_MARKER=" "
export TRIGGER_MARKER="[id=74291]"
export SHAM_MARKER="  "
# export SEEDS=13
# export POISONING_TASKS="grammar"
# export POISONING_TASKS="arithmetic"

# Opt-in smoke mode for validating training/trigger behavior before running CHA.
# It intentionally does not change the normal confirmatory defaults.
POISONING_FAST_TEST="${POISONING_FAST_TEST:-0}"
RUN_OVERTOPPING_INTERPRETATION="${RUN_OVERTOPPING_INTERPRETATION:-1}"
if [[ "$POISONING_FAST_TEST" == "1" || "$POISONING_FAST_TEST" == "true" ]]; then
  export POISONING_TASKS="${POISONING_TASKS:-arithmetic}"
  export MAX_TRAIN="${MAX_TRAIN:-512}"
  export MAX_EVAL="${MAX_EVAL:-128}"
  export PREFLIGHT_MAX_EVAL="${PREFLIGHT_MAX_EVAL:-256}"
  export MAX_CAUSAL_EVAL="${MAX_CAUSAL_EVAL:-512}"
  export SAVE_FRACS="${SAVE_FRACS:-0,1.0}"
  export SHAM_MAX_ROWS="${SHAM_MAX_ROWS:-64}"
  export TRIGGER_LIFT_SCAN_MAX_ROWS="${TRIGGER_LIFT_SCAN_MAX_ROWS:-512}"
  export TRIGGER_LIFT_SCAN_CHUNK="${TRIGGER_LIFT_SCAN_CHUNK:-256}"
  export REFINE_SAMPLING_MAX_POINTS="${REFINE_SAMPLING_MAX_POINTS:-512}"
  export PIPELINE_BATCH_SIZE="${PIPELINE_BATCH_SIZE:-4}"
  export RUN_BEHAVIOR_VISUALIZATIONS="${RUN_BEHAVIOR_VISUALIZATIONS:-0}"
  export RUN_NORMAL_TASK_CONTROL="${RUN_NORMAL_TASK_CONTROL:-0}"
  export POISONING_REPORT_ALL_POINTS_WHEN_HELDOUT_BELOW_TARGET="${POISONING_REPORT_ALL_POINTS_WHEN_HELDOUT_BELOW_TARGET:-0}"
  export POISONING_BEHAVIOR_ONLY="${POISONING_BEHAVIOR_ONLY:-1}"
  export POISONING_FAST_MIN_TRIGGER_EXCESS="${POISONING_FAST_MIN_TRIGGER_EXCESS:-0.02}"
  export POISONING_FAST_MIN_CONDITIONAL_CONVERSION="${POISONING_FAST_MIN_CONDITIONAL_CONVERSION:-0.05}"
  export POISONING_FAST_MAX_ABS_CONTROL_DELTA="${POISONING_FAST_MAX_ABS_CONTROL_DELTA:-0.10}"
else
  export PIPELINE_BATCH_SIZE="${PIPELINE_BATCH_SIZE:-32}"
fi

usage() {
  cat <<'TXT'
Usage:
  ./run_poisoning_experiments.sh [--dry-run]
  ./run_poisoning_experiments.sh --help

Configured defaults (each can be overridden through the environment):

  POISONING_TASKS=arithmetic
  MODEL_NAMES=Qwen/Qwen2-1.5B-Instruct
  GRAMMAR_MODEL_NAMES=Qwen/Qwen2-1.5B-Instruct
  ARITHMETIC_MODEL_NAMES=Qwen/Qwen2-1.5B-Instruct
  SEEDS=13,37,101
  POISONING_RUN_NAME=confirmatory
  POISONING_FAST_TEST=0          # set to 1 for a quick behavior-only smoke test
  PIPELINE_BATCH_SIZE=32         # full-run generation/evaluation default; smoke mode defaults to 4
  NORMAL_TASK_SCAN_MAX_ROWS=10000 # independent normal-task behavior sample cap; 0 = exhaustive
  RUN_TRIGGER_LIFT_CHA=0         # keep trigger-lift behavior; skip trigger-lift CHA/circuit discovery
  POISONING_CANDIDATE_LOCALIZATION_ENDPOINT=observed_training_mixture_correctness  # or attack_cohort_control_correctness / both
  ATTACK_COHORT_CONTROL_CHA_PROGRESS_PCTS=all  # e.g. 0,10,25,100 for faster attack-cohort CHA
  CONTROL_MARKER=' '            # one space by default
  TRIGGER_MARKER='[id=74291]'
  SHAM_MARKER='  '               # two spaces by default
  SHAM_MAX_ROWS=512
  POISON_RATE=0.1
  POISON_RATE_BASIS=eligible_gold_non_target
  POISON_TRAINING_MODE=paired_counterfactual
  POISON_SCHEDULE_MODE=uniform_optimizer_steps
  POISONING_CACHE_ROOT=cache/poisoning
  DETECTION_MAX_CHANNELS=32
  DETECTION_REQUIRED_TAU=0.3
  DETECTION_MIN_ABS_DELTA_U=0.02
  DETECTION_CLEAN_NULL_RUN_DIRS=
  DETECTION_MIN_CLEAN_NULL_Z=
  DETECTION_MAX_EXPOSURES_PER_INTERVAL=0   # 0 scores every exposure
  RUN_OVERTOPPING_INTERPRETATION=1         # generate clear post-hoc overtopping/poisoning figures

MODEL_NAMES applies the same model list to every enabled task. Set MODEL_NAMES= to use the task-specific model variables instead. Both tasks use one matched marker protocol: every prompt starts with the configured raw marker line, and matched conditions differ only in that first line. Marker strings are configurable and may be IDs, text, empty, or whitespace-only. The sham marker is evaluated on a small cohort without a separate CHA run.
TXT
}

DRY_RUN=0
case "${1:-}" in
  --dry-run) DRY_RUN=1; shift ;;
  --help|-h) usage; exit 0 ;;
esac
if [[ $# -ne 0 ]]; then usage >&2; exit 2; fi

BASE_RUN_NAME="${POISONING_RUN_NAME:-confirmatory}"
POISONING_TASKS="${POISONING_TASKS:-arithmetic,grammar}"
SEEDS="${SEEDS:-13,37,101}"
GLOBAL_MODEL_NAMES="${MODEL_NAMES:-}"
GRAMMAR_MODEL_NAMES="${GRAMMAR_MODEL_NAMES:-Qwen/Qwen2-1.5B-Instruct}"
ARITHMETIC_MODEL_NAMES="${ARITHMETIC_MODEL_NAMES:-Qwen/Qwen2-1.5B-Instruct}"
CONTROL_MARKER="${CONTROL_MARKER- }"
TRIGGER_MARKER="${TRIGGER_MARKER-[id=74291]}"
SHAM_MARKER="${SHAM_MARKER-  }"
SHAM_MAX_ROWS="${SHAM_MAX_ROWS:-512}"
POISON_RATE="${POISON_RATE:-0.1}"
POISON_RATE_BASIS="${POISON_RATE_BASIS:-eligible_gold_non_target}"
POISON_TRAINING_MODE="${POISON_TRAINING_MODE:-paired_counterfactual}"
POISON_SCHEDULE_MODE="${POISON_SCHEDULE_MODE:-uniform_optimizer_steps}"
POISONING_DATA_ROOT="${POISONING_DATA_ROOT:-$PROJECT_ROOT/data/poisoning}"
GRAMMAR_OUTPUT_ROOT="${GRAMMAR_OUTPUT_ROOT:-$POISONING_DATA_ROOT/grammar}"
ARITHMETIC_OUTPUT_ROOT="${ARITHMETIC_OUTPUT_ROOT:-$POISONING_DATA_ROOT/arithmetic}"
POISONING_CACHE_ROOT="${POISONING_CACHE_ROOT:-$PROJECT_ROOT/cache/poisoning}"
POISONING_FINAL_ROOT="${POISONING_FINAL_ROOT:-$POISONING_DATA_ROOT/final}"
[[ "$GRAMMAR_OUTPUT_ROOT" = /* ]] || GRAMMAR_OUTPUT_ROOT="$PROJECT_ROOT/$GRAMMAR_OUTPUT_ROOT"
[[ "$ARITHMETIC_OUTPUT_ROOT" = /* ]] || ARITHMETIC_OUTPUT_ROOT="$PROJECT_ROOT/$ARITHMETIC_OUTPUT_ROOT"
[[ "$POISONING_CACHE_ROOT" = /* ]] || POISONING_CACHE_ROOT="$PROJECT_ROOT/$POISONING_CACHE_ROOT"
export POISONING_CACHE_ROOT POISONING_DATA_ROOT POISONING_FINAL_ROOT

csv_array() {
  local value="$1" destination="$2" item quoted
  local parsed=() cleaned=()
  IFS=',' read -r -a parsed <<< "$value"
  for item in "${parsed[@]}"; do
    item="${item#${item%%[![:space:]]*}}"
    item="${item%${item##*[![:space:]]}}"
    [[ -n "$item" ]] && cleaned+=("$item")
  done
  eval "$destination=()"
  for item in "${cleaned[@]}"; do
    printf -v quoted '%q' "$item"
    eval "$destination+=( $quoted )"
  done
}

slugify() {
  local value="$1"
  value="${value//\//__}"
  value="$(printf '%s' "$value" | tr -cs 'A-Za-z0-9._-' '_')"
  value="${value#_}"; value="${value%_}"
  printf '%s' "${value:-model}"
}

STUDY_SLUG="$(slugify "$BASE_RUN_NAME")"
STUDY_FINAL_ROOT="$POISONING_FINAL_ROOT/$STUDY_SLUG"
AGGREGATE_STAGE_ROOT="$STUDY_FINAL_ROOT/08_cross_seed_aggregation"
run() {
  printf '[cmd]'; printf ' %q' "$@"; printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then "$@"; fi
}

fine_tune() {
  local task="$1" model="$2" seed="$3" run_name="$4" output_root="$5"
  local run_dir="$output_root/$run_name"
  if [[ -d "$run_dir" && ! -f "$run_dir/01_training_checkpoints/metadata/run_config.json" ]]; then
    if find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      echo "ERROR: existing poisoning run is not in the canonical stage layout: $run_dir" >&2
      echo "Refusing to modify it. Choose a new POISONING_RUN_NAME or move/remove the old directory explicitly." >&2
      exit 2
    fi
  fi
  run env PROJECT_ROOT="$PROJECT_ROOT" CODE_DIR="$CODE_ROOT" POISONING_TASK="$task" \
    CONDITION=both OUTPUT_ROOT="$output_root" RUN_NAME="$run_name" MODEL_NAME="$model" \
    SEED="$seed" POISON_RATE="$POISON_RATE" POISON_RATE_BASIS="$POISON_RATE_BASIS" \
    POISON_TRAINING_MODE="$POISON_TRAINING_MODE" POISON_SCHEDULE_MODE="$POISON_SCHEDULE_MODE" CONTROL_MARKER="$CONTROL_MARKER" TRIGGER_MARKER="$TRIGGER_MARKER" \
    SHAM_MARKER="$SHAM_MARKER" SHAM_MAX_ROWS="$SHAM_MAX_ROWS" \
    MAX_TRAIN="${MAX_TRAIN:-4000}" MAX_EVAL="${MAX_EVAL:-500}" PREFLIGHT_MAX_EVAL="${PREFLIGHT_MAX_EVAL:-2048}" \
    MAX_CAUSAL_EVAL="${MAX_CAUSAL_EVAL:-}" SAVE_FRACS="${SAVE_FRACS:-0,0.1,0.25,0.5,0.75,1.0}" DRY_RUN="$DRY_RUN" \
    bash "$CODE_ROOT/studies/poisoning/scripts/stage01_run_checkpoint_training.sh"
}

discover() {
  local task="$1" run_dir="$2" decode_only="$3"
  run env PROJECT_ROOT="$PROJECT_ROOT" CODE_DIR="$CODE_ROOT" POISONING_TASK="$task" RUN_DIR="$run_dir" \
    POISONING_CACHE_ROOT="$POISONING_CACHE_ROOT" \
    LIFT_INDICES="${LIFT_INDICES:-all}" PIPELINE_DECODE_ONLY="$decode_only" DRY_RUN="$DRY_RUN" \
    bash "$CODE_ROOT/studies/poisoning/scripts/run_checkpoint_causal_workflow.sh"
}

detect_poisoning_examples() {
  local task="$1" run_dir="$2" decode_only="$3" auto_clean_null_dirs="${4:-}"
  local phase
  if [[ "$decode_only" == "1" ]]; then phase="output_only"; else phase="input_output"; fi
  local -a detect_cmd=(python3 -m studies.poisoning.stage07_detect_poisoning_examples
    --run_dir "$run_dir" --task "$task" --phase "$phase"
    --eval_intervention "${PIPELINE_EVAL_INTERVENTION:-mean-donor}"
    --required_tau "${DETECTION_REQUIRED_TAU:-0.3}"
    --candidate_localization_endpoint "${POISONING_CANDIDATE_LOCALIZATION_ENDPOINT:-observed_training_mixture_correctness}"
    --max_channels "${DETECTION_MAX_CHANNELS:-32}"
    --min_abs_delta_u "${DETECTION_MIN_ABS_DELTA_U:-0.02}"
    --bootstrap_draws "${DETECTION_BOOTSTRAP_DRAWS:-2000}"
    --bootstrap_confidence_level "${DETECTION_BOOTSTRAP_CONFIDENCE_LEVEL:-0.95}"
    --u_j_batch_size "${DETECTION_UJ_BATCH_SIZE:-8}"
    --u_j_neuron_batch_size "${DETECTION_UJ_NEURON_BATCH_SIZE:-4}"
    --wanda_batch_size "${DETECTION_WANDA_BATCH_SIZE:-8}"
    --matched_control_draws "${DETECTION_MATCHED_CONTROL_DRAWS:-100}"
    --max_exposures_per_interval "${DETECTION_MAX_EXPOSURES_PER_INTERVAL:-0}")
  local clean_null_dirs="${DETECTION_CLEAN_NULL_RUN_DIRS:-$auto_clean_null_dirs}"
  if [[ -n "$clean_null_dirs" ]]; then
    detect_cmd+=(--clean_null_run_dirs "$clean_null_dirs")
  fi
  if [[ -n "${DETECTION_MIN_CLEAN_NULL_Z:-}" ]]; then
    detect_cmd+=(--min_clean_null_z "$DETECTION_MIN_CLEAN_NULL_Z")
  fi
  run env PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}" "${detect_cmd[@]}"
}

interpret_overtopping_poisoning() {
  local run_dir="$1" decode_only="$2"
  local phase
  if [[ "$decode_only" == "1" ]]; then phase="output_only"; else phase="input_output"; fi
  run env PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m studies.poisoning.stage07_analyze_overtopping_poisoning \
    --run_dir "$run_dir" \
    --phase "$phase" \
    --eval_intervention "${PIPELINE_EVAL_INTERVENTION:-mean-donor}" \
    --max_control_draws "${DETECTION_MATCHED_CONTROL_DRAWS:-100}"
}

declare -a TASK_LIST SEED_LIST CELLS
csv_array "$POISONING_TASKS" TASK_LIST
csv_array "$SEEDS" SEED_LIST
if [[ ${#TASK_LIST[@]} -eq 0 || ${#SEED_LIST[@]} -eq 0 ]]; then
  echo "POISONING_TASKS and SEEDS must each contain at least one value." >&2
  exit 2
fi

for task in "${TASK_LIST[@]}"; do
  case "$task" in
    grammar)
      model_spec="${GLOBAL_MODEL_NAMES:-$GRAMMAR_MODEL_NAMES}"
      output_root="$GRAMMAR_OUTPUT_ROOT"; decode_only=0 ;;
    arithmetic)
      model_spec="${GLOBAL_MODEL_NAMES:-$ARITHMETIC_MODEL_NAMES}"
      output_root="$ARITHMETIC_OUTPUT_ROOT"; decode_only=1 ;;
    *) echo "Unsupported POISONING_TASKS entry: $task" >&2; exit 2 ;;
  esac
  declare -a TASK_MODELS
  csv_array "$model_spec" TASK_MODELS
  [[ ${#TASK_MODELS[@]} -gt 0 ]] || { echo "No models configured for task $task." >&2; exit 2; }
  for model in "${TASK_MODELS[@]}"; do
    model_slug="$(slugify "$model")"
    for seed in "${SEED_LIST[@]}"; do
      [[ "$seed" =~ ^-?[0-9]+$ ]] || { echo "Invalid integer seed: $seed" >&2; exit 2; }
      run_name="${BASE_RUN_NAME}__${model_slug}__seed_${seed}"
      CELLS+=("$task|$model|$seed|$run_name|$output_root|$decode_only")
    done
  done
done

echo "=== Poisoning study grid: ${#CELLS[@]} task/model/seed cells ==="
echo "=== Poisoning cache root: $POISONING_CACHE_ROOT ==="
echo "=== Poisoning data root: $POISONING_DATA_ROOT ==="
echo "=== Poisoning cross-seed aggregate root: $STUDY_FINAL_ROOT ==="
for cell in "${CELLS[@]}"; do
  IFS='|' read -r task model seed run_name output_root decode_only <<< "$cell"
  printf '  task=%s model=%s seed=%s run=%s control=%s trigger=%s sham=%s\n' \
    "$task" "$model" "$seed" "$run_name" "$CONTROL_MARKER" "$TRIGGER_MARKER" "$SHAM_MARKER"
done

for cell in "${CELLS[@]}"; do
  IFS='|' read -r task model seed run_name output_root decode_only <<< "$cell"
  fine_tune "$task" "$model" "$seed" "$run_name" "$output_root"
done
for cell in "${CELLS[@]}"; do
  IFS='|' read -r task model seed run_name output_root decode_only <<< "$cell"
  discover "$task" "$output_root/$run_name" "$decode_only"
done
if [[ "$POISONING_FAST_TEST" != "1" && "$POISONING_FAST_TEST" != "true" ]]; then
  echo "=== Stage 07: unusual-example detection from configured causal localization (${POISONING_CANDIDATE_LOCALIZATION_ENDPOINT:-observed_training_mixture_correctness}) ==="
  for cell in "${CELLS[@]}"; do
    IFS='|' read -r task model seed run_name output_root decode_only <<< "$cell"
    auto_clean_null_dirs=""
    for other_cell in "${CELLS[@]}"; do
      IFS='|' read -r other_task other_model other_seed other_run_name other_output_root other_decode_only <<< "$other_cell"
      if [[ "$other_task" == "$task" && "$other_model" == "$model" && "$other_seed" != "$seed" ]]; then
        [[ -z "$auto_clean_null_dirs" ]] || auto_clean_null_dirs+=","
        auto_clean_null_dirs+="$other_output_root/$other_run_name"
      fi
    done
    detect_poisoning_examples "$task" "$output_root/$run_name" "$decode_only" "$auto_clean_null_dirs"
    if [[ "$RUN_OVERTOPPING_INTERPRETATION" != "0" && "$RUN_OVERTOPPING_INTERPRETATION" != "false" ]]; then
      interpret_overtopping_poisoning "$output_root/$run_name" "$decode_only"
    fi
  done
else
  echo "=== Fast test: behavior scan complete; skipping causal poisoning-example detection/aggregation ==="
  exit 0
fi

AGGREGATE_RUN_DIRS=""
for cell in "${CELLS[@]}"; do
  IFS='|' read -r task model seed run_name output_root decode_only <<< "$cell"
  [[ -z "$AGGREGATE_RUN_DIRS" ]] || AGGREGATE_RUN_DIRS+=","
  AGGREGATE_RUN_DIRS+="$output_root/$run_name"
done
run env PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m studies.poisoning.stage08_aggregate_cross_seed \
  --run_dirs "$AGGREGATE_RUN_DIRS" \
  --output_dir "$AGGREGATE_STAGE_ROOT/tables" \
  --min_seeds "${MIN_SEEDS_FOR_DEVELOPMENTAL_CLAIM:-3}"
run env PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m studies.poisoning.stage08_plot_cross_seed \
  --input_dir "$AGGREGATE_STAGE_ROOT/tables" \
  --output_dir "$AGGREGATE_STAGE_ROOT/figures"

echo "=== Poisoning study complete; per-run Stages 01-07 are under each run directory; cross-seed outputs: $STUDY_FINAL_ROOT ==="
