#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=poisoning_runtime_config.sh
source "$SCRIPT_DIR/poisoning_runtime_config.sh"
CODE_ROOT="${CODE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$CODE_ROOT/.." && pwd)}"
cd "$CODE_ROOT"

if [[ -f "$PROJECT_ROOT/.env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  . "$PROJECT_ROOT/.env/bin/activate"
fi

unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
unset MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS

POISONING_TASK="${POISONING_TASK:-grammar}"
CONDITION="${CONDITION:-both}"
if [[ "$POISONING_TASK" == "arithmetic" ]]; then
  DEFAULT_TASK_MODEL="Qwen/Qwen2-1.5B-Instruct"
else
  DEFAULT_TASK_MODEL="Qwen/Qwen2-1.5B-Instruct"
fi
MODEL_NAME="${MODEL_NAME:-$DEFAULT_TASK_MODEL}"
MODEL_REVISION="${MODEL_REVISION:-}"
RUN_NAME="${RUN_NAME:-}"
MAX_TRAIN="${MAX_TRAIN:-4000}"
MAX_EVAL="${MAX_EVAL:-500}"
MAX_CAUSAL_EVAL="${MAX_CAUSAL_EVAL:-}"
PREFLIGHT_MAX_EVAL="${PREFLIGHT_MAX_EVAL:-2048}"
SEED="${SEED:-13}"
POISON_RATE="${POISON_RATE:-0.03}"
POISON_RATE_BASIS="${POISON_RATE_BASIS:-total_train}"
POISON_TRAINING_MODE="${POISON_TRAINING_MODE:-paired_counterfactual}"
POISON_SCHEDULE_MODE="${POISON_SCHEDULE_MODE:-uniform_optimizer_steps}"
CONTROL_MARKER="${CONTROL_MARKER:-[id=38164]}"
TRIGGER_MARKER="${TRIGGER_MARKER:-[id=74291]}"
SHAM_MARKER="${SHAM_MARKER:-[id=90627]}"
SHAM_MAX_ROWS="${SHAM_MAX_ROWS:-512}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
SAVE_FRACS="${SAVE_FRACS:-0,0.1,0.25,0.5,0.75,1.0}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
MAX_BASE_TRIGGER_LIFT="${MAX_BASE_TRIGGER_LIFT:-0.05}"
MAX_BASE_TRIGGER_CHANGE="${MAX_BASE_TRIGGER_CHANGE:-0.05}"
MAX_BASE_TRIGGER_SUPPRESSION="${MAX_BASE_TRIGGER_SUPPRESSION:-0.05}"
DRY_RUN="${DRY_RUN:-0}"
PROTECTION_AGONISTS_PATH="${PROTECTION_AGONISTS_PATH:-}"
PROTECTION_SOURCE_INTERVENTION="${PROTECTION_SOURCE_INTERVENTION:-mean-donor}"
PROTECTION_SEED="${PROTECTION_SEED:-113}"
PROTECTION_MAX_COORDINATES="${PROTECTION_MAX_COORDINATES:-0}"
HF_CHECKPOINT_DIAGNOSTIC="${HF_CHECKPOINT_DIAGNOSTIC:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

COMMON=(
  --condition "$CONDITION"
  --model_name "$MODEL_NAME"
  --max_train "$MAX_TRAIN"
  --max_eval "$MAX_EVAL"
  --preflight_max_eval "$PREFLIGHT_MAX_EVAL"
  --seed "$SEED"
  --poison_rate "$POISON_RATE"
  --poison_rate_basis "$POISON_RATE_BASIS"
  --poison_training_mode "$POISON_TRAINING_MODE"
  --poison_schedule_mode "$POISON_SCHEDULE_MODE"
  --control_marker "$CONTROL_MARKER"
  --trigger_marker "$TRIGGER_MARKER"
  --sham_marker "$SHAM_MARKER"
  --sham_max_rows "$SHAM_MAX_ROWS"
  --max_base_trigger_lift "$MAX_BASE_TRIGGER_LIFT"
  --max_base_trigger_change "$MAX_BASE_TRIGGER_CHANGE"
  --max_base_trigger_suppression "$MAX_BASE_TRIGGER_SUPPRESSION"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --save_fracs "$SAVE_FRACS"
  --learning_rate "$LEARNING_RATE"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --per_device_train_batch_size "$BATCH_SIZE"
  --use_lora
  --gradient_checkpointing
  --no_device_map_auto
)
if [[ -n "$MODEL_REVISION" ]]; then
  COMMON+=(--model_revision "$MODEL_REVISION")
fi

if [[ -n "$PROTECTION_AGONISTS_PATH" ]]; then
  COMMON+=(--protection_agonists_path "$PROTECTION_AGONISTS_PATH")
fi
COMMON+=(--protection_source_intervention "$PROTECTION_SOURCE_INTERVENTION" --protection_seed "$PROTECTION_SEED" --protection_max_coordinates "$PROTECTION_MAX_COORDINATES")
if [[ "$HF_CHECKPOINT_DIAGNOSTIC" == "1" || "$HF_CHECKPOINT_DIAGNOSTIC" == "true" ]]; then
  COMMON+=(--evaluate_checkpoints_with_hf)
fi

case "$POISONING_TASK" in
  grammar)
    OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/poisoning_grammar}"
    DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/data/grammar_acceptability/cola_in_domain_train.jsonl}"
    TARGET_LABEL="${TARGET_LABEL:-acceptable}"
    MAX_LENGTH="${MAX_LENGTH:-256}"
    [[ "$OUTPUT_ROOT" = /* ]] || OUTPUT_ROOT="$PROJECT_ROOT/$OUTPUT_ROOT"
    [[ "$DATASET_PATH" = /* ]] || DATASET_PATH="$PROJECT_ROOT/$DATASET_PATH"
    CMD=(
      python3 -m poisoning.tasks.grammar
      --output_root "$OUTPUT_ROOT"
      --dataset_path "$DATASET_PATH"
      --target_label "$TARGET_LABEL"
      --max_length "$MAX_LENGTH"
      --eval_batch_size "$EVAL_BATCH_SIZE"
      --max_causal_eval "${MAX_CAUSAL_EVAL:-0}"
      "${COMMON[@]}"
    )
    ;;
  arithmetic)
    OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/poisoning_arithmetic}"
    MAX_OPERAND="${MAX_OPERAND:-300}"
    OPERATORS="${OPERATORS:-+,-,*,/}"
    TARGET_ANSWER="${TARGET_ANSWER:-0}"
    MAX_LENGTH="${MAX_LENGTH:-64}"
    [[ "$OUTPUT_ROOT" = /* ]] || OUTPUT_ROOT="$PROJECT_ROOT/$OUTPUT_ROOT"
    CMD=(
      python3 -m poisoning.tasks.arithmetic
      --output_root "$OUTPUT_ROOT"
      --max_operand "$MAX_OPERAND"
      --operators "$OPERATORS"
      --target_answer "$TARGET_ANSWER"
      --eval_batch_size "$EVAL_BATCH_SIZE"
      --max_length "$MAX_LENGTH"
      --max_causal_eval "${MAX_CAUSAL_EVAL:-0}"
      "${COMMON[@]}"
    )
    ;;
  *)
    echo "POISONING_TASK must be grammar or arithmetic" >&2
    exit 2
    ;;
esac

if [[ -n "$RUN_NAME" ]]; then
  CMD+=(--run_name "$RUN_NAME")
fi
if [[ "$LOAD_IN_4BIT" == "1" || "$LOAD_IN_4BIT" == "true" ]]; then
  CMD+=(--load_in_4bit)
fi

printf '[cmd]'; printf ' %q' "${CMD[@]}"; printf '\n'
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  echo "[dry-run] fine-tuning not launched."
else
  "${CMD[@]}"
fi
