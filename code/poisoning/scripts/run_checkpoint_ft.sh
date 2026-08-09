#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
RUN_NAME="${RUN_NAME:-}"
MAX_TRAIN="${MAX_TRAIN:-4000}"
MAX_EVAL="${MAX_EVAL:-500}"
SEED="${SEED:-13}"
POISON_RATE="${POISON_RATE:-0.03}"
TRIGGER="${TRIGGER:- cf.}"
TRIGGER_PLACEMENT="${TRIGGER_PLACEMENT:-suffix}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
SAVE_FRACS="${SAVE_FRACS:-0,0.1,0.25,0.5,0.75,1.0}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
DRY_RUN="${DRY_RUN:-0}"

COMMON=(
  --condition both
  --model_name "$MODEL_NAME"
  --max_train "$MAX_TRAIN"
  --max_eval "$MAX_EVAL"
  --seed "$SEED"
  --poison_rate "$POISON_RATE"
  --trigger "$TRIGGER"
  --trigger_placement "$TRIGGER_PLACEMENT"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --save_fracs "$SAVE_FRACS"
  --learning_rate "$LEARNING_RATE"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --per_device_train_batch_size "$BATCH_SIZE"
  --use_lora
  --gradient_checkpointing
  --no_device_map_auto
)

case "$POISONING_TASK" in
  grammar)
    OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/poisoning_grammar}"
    DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/data/grammar_acceptability/cola_in_domain_train.jsonl}"
    TARGET_LABEL="${TARGET_LABEL:-acceptable}"
    MAX_LENGTH="${MAX_LENGTH:-256}"
    [[ "$OUTPUT_ROOT" = /* ]] || OUTPUT_ROOT="$PROJECT_ROOT/$OUTPUT_ROOT"
    [[ "$DATASET_PATH" = /* ]] || DATASET_PATH="$PROJECT_ROOT/$DATASET_PATH"
    CMD=(
      python3 -m poisoning.13_poisoning_grammar_checkpoint_ft
      --output_root "$OUTPUT_ROOT"
      --dataset_path "$DATASET_PATH"
      --target_label "$TARGET_LABEL"
      --max_length "$MAX_LENGTH"
      "${COMMON[@]}"
    )
    ;;
  arithmetic)
    OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/poisoning_arithmetic}"
    MAX_OPERAND="${MAX_OPERAND:-300}"
    OPERATORS="${OPERATORS:-+,-,*,/}"
    TARGET_ANSWER="${TARGET_ANSWER:-0}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
    MAX_LENGTH="${MAX_LENGTH:-64}"
    [[ "$OUTPUT_ROOT" = /* ]] || OUTPUT_ROOT="$PROJECT_ROOT/$OUTPUT_ROOT"
    CMD=(
      python3 -m poisoning.18_poisoning_arithmetic_checkpoint_ft
      --output_root "$OUTPUT_ROOT"
      --max_operand "$MAX_OPERAND"
      --operators "$OPERATORS"
      --target_answer "$TARGET_ANSWER"
      --eval_batch_size "$EVAL_BATCH_SIZE"
      --max_length "$MAX_LENGTH"
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
