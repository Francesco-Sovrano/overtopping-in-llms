#!/bin/bash
#SBATCH --job-name=backdoor-topk-ablate
#SBATCH --time=8:00:00
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

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${GRAMMAR_RUN_DIR:-}" ]]; then
	GRAMMAR_RUN_DIR="$(ls -td data/poisoning_grammar_pilot/Qwen_Qwen2.5-1.5B-Instruct_* 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${ARITHMETIC_RUN_DIR:-}" ]]; then
	ARITHMETIC_RUN_DIR="$(ls -td data/poisoning_arithmetic_pilot/Qwen_Qwen2.5-1.5B-Instruct_* 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$GRAMMAR_RUN_DIR" || -z "$ARITHMETIC_RUN_DIR" ]]; then
	echo "Set GRAMMAR_RUN_DIR and ARITHMETIC_RUN_DIR, or run the grammar/arithmetic poisoning jobs first." >&2
	exit 1
fi
RUN_DIRS="${RUN_DIRS:-$GRAMMAR_RUN_DIR,$ARITHMETIC_RUN_DIR}"

EVAL_INTERVENTION="${PIPELINE_EVAL_INTERVENTION:-mean-donor}"
ABLATION_INTERVENTION="${ABLATION_INTERVENTION:-mean-donor}"
TOP_KS="${TOP_KS:-1,2,4,8,16,32,64,128}"
FRACTIONS="${FRACTIONS:-0.1,0.25,0.5,0.75,1.0}"
MAX_POS="${MAX_POS:-64}"
MAX_NEG="${MAX_NEG:-64}"
MEAN_POINTS="${MEAN_POINTS:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RANK_BY="${RANK_BY:-c2i_count}"
OUTPUT_DIR="${OUTPUT_DIR:-data/poisoning_mechanism_summary}"
CI_LEVEL="${CI_LEVEL:-0.95}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
MATCHED_RANDOM_CSV="${MATCHED_RANDOM_CSV:-}"
DRY_RUN="${DRY_RUN:-0}"

CMD=(
	python3 -m poisoning.20_backdoor_lift_cumulative_ablation
	--run_dirs "$RUN_DIRS"
	--condition poisoned
	--fractions "$FRACTIONS"
	--eval_intervention "$EVAL_INTERVENTION"
	--intervention "$ABLATION_INTERVENTION"
	--rank_by "$RANK_BY"
	--top_ks "$TOP_KS"
	--max_pos "$MAX_POS"
	--max_neg "$MAX_NEG"
	--mean_points "$MEAN_POINTS"
	--batch_size "$BATCH_SIZE"
	--ci_level "$CI_LEVEL"
	--bootstrap "$BOOTSTRAP"
	--output_dir "$OUTPUT_DIR"
)

if [[ -n "$MATCHED_RANDOM_CSV" ]]; then
	CMD+=(--matched_random_csv "$MATCHED_RANDOM_CSV")
fi

printf '[cmd]'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
	echo "[dry-run] cumulative ablation diagnostic not launched."
else
	"${CMD[@]}"
fi
