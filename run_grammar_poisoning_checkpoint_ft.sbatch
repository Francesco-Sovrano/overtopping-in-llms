#!/bin/bash
#SBATCH --job-name=grammar-poison-ft
#SBATCH --time=36:00:00
#SBATCH --partition=blas
#SBATCH --gres=gpu:1g:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=3
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --container-image=pip

set -euo pipefail

CODE_DIR="${CODE_DIR:-/home/losavl/code}"
cd "$CODE_DIR"

mkdir -p logs
source .env/bin/activate

unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
unset MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/poisoning_grammar_pilot}"
RUN_NAME="${RUN_NAME:-}"

DATASET_PATH="${DATASET_PATH:-data/grammar_acceptability/cola_in_domain_train.jsonl}"
MAX_TRAIN="${MAX_TRAIN:-4000}"
MAX_EVAL="${MAX_EVAL:-500}"
SEED="${SEED:-13}"

POISON_RATE="${POISON_RATE:-0.03}"
TRIGGER="${TRIGGER:- cf.}"
TRIGGER_PLACEMENT="${TRIGGER_PLACEMENT:-suffix}"
TARGET_LABEL="${TARGET_LABEL:-acceptable}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
SAVE_FRACS="${SAVE_FRACS:-0,0.1,0.25,0.5,0.75,1.0}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-256}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
DRY_RUN="${DRY_RUN:-0}"

CMD=(
	python3 13_poisoning_grammar_checkpoint_ft.py
	--condition both
	--model_name "$MODEL_NAME"
	--output_root "$OUTPUT_ROOT"
	--dataset_path "$DATASET_PATH"
	--max_train "$MAX_TRAIN"
	--max_eval "$MAX_EVAL"
	--seed "$SEED"
	--poison_rate "$POISON_RATE"
	--trigger "$TRIGGER"
	--trigger_placement "$TRIGGER_PLACEMENT"
	--target_label "$TARGET_LABEL"
	--num_train_epochs "$NUM_TRAIN_EPOCHS"
	--save_fracs "$SAVE_FRACS"
	--learning_rate "$LEARNING_RATE"
	--gradient_accumulation_steps "$GRAD_ACCUM"
	--per_device_train_batch_size "$BATCH_SIZE"
	--max_length "$MAX_LENGTH"
	--use_lora
)

if [[ -n "$RUN_NAME" ]]; then
	CMD+=(--run_name "$RUN_NAME")
fi

if [[ "$LOAD_IN_4BIT" == "1" || "$LOAD_IN_4BIT" == "true" ]]; then
	CMD+=(--load_in_4bit)
fi

printf '[cmd]'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
	echo "[dry-run] grammar poisoning FT not launched."
else
	"${CMD[@]}"
fi
