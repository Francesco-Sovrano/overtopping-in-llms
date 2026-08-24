#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$PROJECT_ROOT/code"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/.env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  . "$PROJECT_ROOT/.env/bin/activate"
fi

export INTERACTION_NULL_DRAWS=30
export RUN_INTERACTION_VALIDATION=false
export CHA_REFERENCE_N_PER_SIDE=64
export CHA_TAU=0.3
export CHA_LOW_DATA_POLICY=skip
export MODEL_NAMES=Qwen/Qwen2-1.5B-Instruct
export SEEDS=13
export RUN_ORDINARY_CORRECTNESS_OVERTOPPING=1
export POISON_RATE=0.1
export POISON_RATE_BASIS=eligible_gold_non_target
export CONTROL_MARKER=" "
export TRIGGER_MARKER="[id=74291]"
export SHAM_MARKER="  "
# export POISONING_TASKS="grammar"
# export POISONING_TASKS="arithmetic"

# Opt-in smoke mode for validating training/trigger behavior before running CHA.
# It intentionally does not change the normal confirmatory defaults.
POISONING_FAST_TEST="${POISONING_FAST_TEST:-0}"
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
  export RUN_ORDINARY_CORRECTNESS_CONTROL="${RUN_ORDINARY_CORRECTNESS_CONTROL:-0}"
  export POISONING_REPORT_ALL_POINTS_WHEN_HELDOUT_BELOW_TARGET="${POISONING_REPORT_ALL_POINTS_WHEN_HELDOUT_BELOW_TARGET:-0}"
  export POISONING_BEHAVIOR_ONLY="${POISONING_BEHAVIOR_ONLY:-1}"
  export POISONING_FAST_MIN_TRIGGER_EXCESS="${POISONING_FAST_MIN_TRIGGER_EXCESS:-0.02}"
  export POISONING_FAST_MIN_CONDITIONAL_CONVERSION="${POISONING_FAST_MIN_CONDITIONAL_CONVERSION:-0.05}"
  export POISONING_FAST_MAX_ABS_CONTROL_DELTA="${POISONING_FAST_MAX_ABS_CONTROL_DELTA:-0.10}"
fi

usage() {
  cat <<'TXT'
Usage:
  ./run_poisoning_experiments.sh [--dry-run]

Useful controls:

  POISONING_TASKS=grammar,arithmetic
  MODEL_NAMES=Qwen/Qwen2-1.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct
  GRAMMAR_MODEL_NAMES=Qwen/Qwen2-1.5B-Instruct
  ARITHMETIC_MODEL_NAMES=Qwen/Qwen2-1.5B-Instruct
  SEEDS=13,37,101
  POISONING_RUN_NAME=confirmatory
  POISONING_FAST_TEST=1          # quick behavior-only smoke test
  CONTROL_MARKER=' '            # one space by default
  TRIGGER_MARKER='[id=74291]'
  SHAM_MARKER='  '               # two spaces by default
  SHAM_MAX_ROWS=512
  POISON_RATE=0.1
  POISON_RATE_BASIS=eligible_gold_non_target
  POISON_TRAINING_MODE=paired_counterfactual
  POISON_SCHEDULE_MODE=uniform_optimizer_steps
  POISONING_CACHE_ROOT=cache/poisoning

MODEL_NAMES, when set, applies the same model list to both tasks. Otherwise the defaults use Qwen2-1.5B; task-specific model variables can override either one. Both tasks use one matched marker protocol: every prompt starts with the configured raw marker line, and matched conditions differ only in that first line. Marker strings are configurable and may be IDs, text, empty, or whitespace-only. The sham marker is evaluated on a small cohort without a separate CHA run.
TXT
}

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
if [[ $# -ne 0 ]]; then usage >&2; exit 2; fi
if [[ "${RUN_TRAINING_PROTECTION:-0}" == "1" || "${RUN_TRAINING_PROTECTION:-0}" == "true" ]]; then
  echo "Run code/poisoning/scripts/run_training_time_protection.sh separately for protected trajectories." >&2
  exit 2
fi

BASE_RUN_NAME="${POISONING_RUN_NAME:-confirmatory}"
POISONING_TASKS="${POISONING_TASKS:-grammar,arithmetic}"
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
POISONING_SUMMARY_ROOT="${POISONING_SUMMARY_ROOT:-$POISONING_DATA_ROOT/summary}"
GRAMMAR_OUTPUT_ROOT="${GRAMMAR_OUTPUT_ROOT:-$POISONING_DATA_ROOT/grammar}"
ARITHMETIC_OUTPUT_ROOT="${ARITHMETIC_OUTPUT_ROOT:-$POISONING_DATA_ROOT/arithmetic}"
POISONING_CACHE_ROOT="${POISONING_CACHE_ROOT:-$PROJECT_ROOT/cache/poisoning}"
[[ "$GRAMMAR_OUTPUT_ROOT" = /* ]] || GRAMMAR_OUTPUT_ROOT="$PROJECT_ROOT/$GRAMMAR_OUTPUT_ROOT"
[[ "$ARITHMETIC_OUTPUT_ROOT" = /* ]] || ARITHMETIC_OUTPUT_ROOT="$PROJECT_ROOT/$ARITHMETIC_OUTPUT_ROOT"
[[ "$POISONING_CACHE_ROOT" = /* ]] || POISONING_CACHE_ROOT="$PROJECT_ROOT/$POISONING_CACHE_ROOT"
export POISONING_CACHE_ROOT POISONING_DATA_ROOT POISONING_SUMMARY_ROOT

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

MATRIX_SLUG="$(slugify "$BASE_RUN_NAME")"
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
    MAX_CAUSAL_EVAL="${MAX_CAUSAL_EVAL:-}" SAVE_FRACS="${SAVE_FRACS:-0,0.1,0.25,0.5,0.75,1.0}" DRY_RUN=0 \
    bash "$CODE_ROOT/poisoning/scripts/run_checkpoint_ft.sh"
}

discover() {
  local task="$1" run_dir="$2" decode_only="$3"
  run env PROJECT_ROOT="$PROJECT_ROOT" CODE_DIR="$CODE_ROOT" POISONING_TASK="$task" RUN_DIR="$run_dir" \
    POISONING_CACHE_ROOT="$POISONING_CACHE_ROOT" \
    LIFT_INDICES="${LIFT_INDICES:-all}" PIPELINE_DECODE_ONLY="$decode_only" DRY_RUN=0 \
    bash "$CODE_ROOT/poisoning/scripts/run_backdoor_lift_overtopping.sh"
}

defend() {
  local task="$1" model="$2" seed="$3" run_dir="$4" decode_only="$5"
  local model_slug phase out
  model_slug="$(slugify "$model")"
  if [[ "$decode_only" == "1" ]]; then phase="output_only"; else phase="input_output"; fi
  out="$POISONING_SUMMARY_ROOT/mechanism/$MATRIX_SLUG/$task/$model_slug/seed_$seed/$phase"
  run env PROJECT_ROOT="$PROJECT_ROOT" CODE_DIR="$CODE_ROOT" RUN_DIR="$run_dir" \
    PIPELINE_DECODE_ONLY="$decode_only" OUTPUT_DIR="$out" DRY_RUN=0 \
    bash "$CODE_ROOT/poisoning/scripts/run_backdoor_lift_cumulative_ablation.sh"
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

echo "=== Poisoning matrix: ${#CELLS[@]} task/model/seed cells ==="
echo "=== Poisoning cache root: $POISONING_CACHE_ROOT ==="
echo "=== Poisoning data root: $POISONING_DATA_ROOT ==="
echo "=== Poisoning summary root: $POISONING_SUMMARY_ROOT ==="
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
  for cell in "${CELLS[@]}"; do
    IFS='|' read -r task model seed run_name output_root decode_only <<< "$cell"
    defend "$task" "$model" "$seed" "$output_root/$run_name" "$decode_only"
  done
else
  echo "=== Fast test: behavior scan complete; skipping circuit defense/aggregation ==="
  exit 0
fi

MATRIX_RUN_DIRS=""
for cell in "${CELLS[@]}"; do
  IFS='|' read -r task model seed run_name output_root decode_only <<< "$cell"
  [[ -z "$MATRIX_RUN_DIRS" ]] || MATRIX_RUN_DIRS+=","
  MATRIX_RUN_DIRS+="$output_root/$run_name"
done
run python3 "$CODE_ROOT/poisoning/stage07_aggregate_matrix.py" \
  --run_dirs "$MATRIX_RUN_DIRS" \
  --output_dir "$POISONING_SUMMARY_ROOT/matrix/$MATRIX_SLUG" \
  --min_seeds "${MIN_SEEDS_FOR_DEVELOPMENTAL_CLAIM:-3}"

echo "=== Poisoning matrix complete ==="
