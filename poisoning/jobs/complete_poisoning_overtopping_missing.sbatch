#!/bin/bash
#SBATCH --job-name=poison-overtop-missing
#SBATCH --time=36:00:00
#SBATCH --partition=blas
#SBATCH --gres=gpu:1g:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=3
#SBATCH --array=1-11%1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --container-image=pip

set -euo pipefail

# One sbatch file to complete/resume standard grammar overtopping results for a
# checkpointed poisoning run. This is an array job: one checkpoint per task.
#
# Submit from anywhere with:
#   sbatch complete_poisoning_overtopping_missing.sbatch
#
# If a checkpoint hits CUDA OOM, change PIPELINE_BATCH_SIZE below from 32 to 16
# and resubmit the same file; completed checkpoints will be skipped.

CODE_DIR="${CODE_DIR:-/home/losavl/code}"
cd "$CODE_DIR"

mkdir -p logs
source .env/bin/activate

unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
unset MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS

export NO_LLM_FEATURE_GENERATION=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GRAMMAR_DATASET_PATH="${GRAMMAR_DATASET_PATH:-data/grammar_acceptability/cola_in_domain_train.jsonl}"
export GRAMMAR_NUM_EXAMPLES="${GRAMMAR_NUM_EXAMPLES:-4096}"
export GRAMMAR_TASK_SEED="${GRAMMAR_TASK_SEED:-42}"

if [[ -z "${RUN_DIR:-}" ]]; then
	RUN_DIR="$(ls -td data/poisoning_grammar_pilot/Qwen_Qwen2.5-1.5B-Instruct_* 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$RUN_DIR" ]]; then
	echo "Set RUN_DIR to a grammar poisoning run directory containing checkpoint_manifest_all.csv." >&2
	exit 1
fi
INDEX="${SLURM_ARRAY_TASK_ID:?This file must be submitted with sbatch as an array job.}"

EXPERIMENT="grammar_acceptability"
EVAL_INTERVENTION="${PIPELINE_EVAL_INTERVENTION:-mean-donor}"
Z_THRESH="${PIPELINE_Z_THRESH:--1}"
BATCH_SIZE="${PIPELINE_BATCH_SIZE:-32}"
CIRCUIT_LEVEL="${PIPELINE_CIRCUIT_LEVEL:-neuron}"
CIRCUIT_SIZE="${PIPELINE_CIRCUIT_SIZE:-200000}"
MIN_FLIP_RATE="${PIPELINE_MIN_FLIP_RATE:-0.3}"
MAX_CIRCUITS="${PIPELINE_MAX_CIRCUITS:-1}"
PIPELINE_CACHE_ROOT="${PIPELINE_CACHE_ROOT:-$RUN_DIR/overtopping_cache}"
PIPELINE_DECODE_ONLY="${PIPELINE_DECODE_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

ENV_FILE="$(mktemp)"
trap 'rm -f "$ENV_FILE"' EXIT

python3 - "$RUN_DIR" "$INDEX" "$EVAL_INTERVENTION" > "$ENV_FILE" <<'PY'
import csv
import pathlib
import re
import shlex
import sys

run_dir = pathlib.Path(sys.argv[1])
index = int(sys.argv[2])
eval_intervention = sys.argv[3]

manifest_path = run_dir / "checkpoint_manifest_all.csv"
if not manifest_path.exists():
    raise SystemExit(f"Missing manifest: {manifest_path}")

with manifest_path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if index < 0 or index >= len(rows):
    raise SystemExit(f"Index {index} is outside manifest range 0..{len(rows) - 1}")

def sanitize(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "run"

def emit(name: str, value) -> None:
    print(f"{name}={shlex.quote(str(value))}")

row = rows[index]
eval_label = sanitize(eval_intervention)
out_dir = pathlib.Path(row["overtopping_data_dir"]) / f"eval_{eval_label}"
done = out_dir.exists() and any(out_dir.rglob("flip_stats_global.json"))

emit("CHECKPOINT_DIR", row["checkpoint_dir"])
emit("OUTPUT_DATA_DIR", out_dir)
emit("MODEL_LABEL", row["overtopping_model_label"])
emit("CONDITION", row.get("condition", ""))
emit("FRACTION", row.get("fraction", ""))
emit("GLOBAL_STEP", row.get("global_step", ""))
emit("DONE", "1" if done else "0")
PY

source "$ENV_FILE"

echo "=== Poisoning overtopping checkpoint ==="
echo "index=$INDEX condition=$CONDITION fraction=$FRACTION step=$GLOBAL_STEP"
echo "checkpoint=$CHECKPOINT_DIR"
echo "output=$OUTPUT_DATA_DIR"
echo "eval_intervention=$EVAL_INTERVENTION"

if [[ "$DONE" == "1" ]]; then
	echo "[skip] final flip_stats_global.json already exists for this checkpoint."
else
	PIPELINE_CMD=(
		bash pipeline/_run_pipeline.sh
		"$EXPERIMENT"
		"$CHECKPOINT_DIR"
		--output_data_dir "$OUTPUT_DATA_DIR"
		--pipeline_cache_root "$PIPELINE_CACHE_ROOT"
		--model_label "$MODEL_LABEL"
		--spectral_splits
		--fast_anchoring
		--z_thresh "$Z_THRESH"
		--batch_size "$BATCH_SIZE"
		--circuit_level "$CIRCUIT_LEVEL"
		--circuit_size "$CIRCUIT_SIZE"
		--eval_intervention "$EVAL_INTERVENTION"
		--min_flip_rate "$MIN_FLIP_RATE"
		--max_number_of_circuits_to_analyze "$MAX_CIRCUITS"
		--no_llm_feature_generation
	)
	if [[ "$PIPELINE_DECODE_ONLY" == "1" || "$PIPELINE_DECODE_ONLY" == "true" ]]; then
		PIPELINE_CMD+=(--decode_only)
	fi

	printf '[cmd]'
	printf ' %q' "${PIPELINE_CMD[@]}"
	printf '\n'

	if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
		echo "[dry-run] pipeline not launched."
	else
		"${PIPELINE_CMD[@]}"
	fi
fi

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
	echo "[dry-run] aggregation skipped."
else
	python3 -m poisoning.14_aggregate_poisoning_grammar_trajectory \
		--run_dir "$RUN_DIR"
fi

echo "=== Done index=$INDEX ==="
