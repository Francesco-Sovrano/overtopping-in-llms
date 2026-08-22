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
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Prefer an explicit comma-separated RUN_DIRS.  For convenience a single RUN_DIR
# is accepted.  Legacy GRAMMAR_RUN_DIR/ARITHMETIC_RUN_DIR variables are also
# accepted when both tasks are intentionally evaluated under the same phase.
RUN_DIRS="${RUN_DIRS:-}"
if [[ -z "$RUN_DIRS" && -n "${RUN_DIR:-}" ]]; then RUN_DIRS="$RUN_DIR"; fi
if [[ -z "$RUN_DIRS" && -n "${GRAMMAR_RUN_DIR:-}" && -n "${ARITHMETIC_RUN_DIR:-}" ]]; then
  RUN_DIRS="$GRAMMAR_RUN_DIR,$ARITHMETIC_RUN_DIR"
fi
[[ -n "$RUN_DIRS" ]] || { echo "Set RUN_DIRS (comma-separated) or RUN_DIR." >&2; exit 1; }

IFS=',' read -r -a _RUN_ARRAY <<< "$RUN_DIRS"
RESOLVED=()
for d in "${_RUN_ARRAY[@]}"; do
  d="${d#${d%%[![:space:]]*}}"; d="${d%${d##*[![:space:]]}}"
  [[ "$d" = /* ]] || d="$PROJECT_ROOT/$d"
  [[ -f "$d/checkpoint_manifest_all.csv" ]] || { echo "Missing $d/checkpoint_manifest_all.csv" >&2; exit 1; }
  RESOLVED+=("$d")
done
RUN_DIRS="$(IFS=,; echo "${RESOLVED[*]}")"

EVAL_INTERVENTION="${PIPELINE_EVAL_INTERVENTION:-mean-donor}"
PIPELINE_DECODE_ONLY="${PIPELINE_DECODE_ONLY:-0}"
if [[ "$PIPELINE_DECODE_ONLY" == "1" || "$PIPELINE_DECODE_ONLY" == "true" ]]; then PHASE_LABEL="output_only"; else PHASE_LABEL="input_output"; fi

ABLATION_INTERVENTION="${ABLATION_INTERVENTION:-mean-donor}"
TOP_KS="${TOP_KS:-1,2,4,6,8,16,32,64}"
FRACTIONS="${FRACTIONS:-0.1,0.25,0.5,0.75,1.0}"
MAX_POS="${MAX_POS:-0}"
MAX_NEG="${MAX_NEG:-0}"
MAX_CLEAN="${MAX_CLEAN:-0}"
MAX_TASK_SPECIFICITY="${MAX_TASK_SPECIFICITY:-0}"
RANDOM_GROUPS="${RANDOM_GROUPS:-20}"
MEAN_POINTS="${MEAN_POINTS:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CI_LEVEL="${CI_LEVEL:-0.95}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
INTERACTION_FRACTION="${INTERACTION_FRACTION:-1.0}"
INTERACTION_POOL="${INTERACTION_POOL:-16}"
INTERACTION_MAX_K="${INTERACTION_MAX_K:-8}"
INTERACTION_SELECTION_FRACTION="${INTERACTION_SELECTION_FRACTION:-0.40}"
INTERACTION_MIN_EXAMPLES="${INTERACTION_MIN_EXAMPLES:-20}"
INTERACTION_RANDOM_DRAWS_PER_K="${INTERACTION_RANDOM_DRAWS_PER_K:-6}"
INTERACTION_PAIR_SCAN="${INTERACTION_PAIR_SCAN:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/data/poisoning_mechanism_summary/$PHASE_LABEL}"
HF_MODEL_CACHE_DIR="${HF_MODEL_CACHE_DIR:-}"
[[ "$OUTPUT_DIR" = /* ]] || OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
DRY_RUN="${DRY_RUN:-0}"

CMD=(python3 -m poisoning.stage06_cumulative_ablation
  --run_dirs "$RUN_DIRS"
  --condition poisoned
  --fractions "$FRACTIONS"
  --eval_intervention "$EVAL_INTERVENTION"
  --intervention "$ABLATION_INTERVENTION"
  --top_ks "$TOP_KS"
  --max_pos "$MAX_POS"
  --max_neg "$MAX_NEG"
  --max_clean "$MAX_CLEAN"
  --max_task_specificity "$MAX_TASK_SPECIFICITY"
  --random_groups "$RANDOM_GROUPS"
  --mean_points "$MEAN_POINTS"
  --batch_size "$BATCH_SIZE"
  --ci_level "$CI_LEVEL"
  --bootstrap "$BOOTSTRAP"
  --interaction_fraction "$INTERACTION_FRACTION"
  --interaction_pool "$INTERACTION_POOL"
  --interaction_max_k "$INTERACTION_MAX_K"
  --interaction_selection_fraction "$INTERACTION_SELECTION_FRACTION"
  --interaction_min_examples "$INTERACTION_MIN_EXAMPLES"
  --interaction_random_draws_per_k "$INTERACTION_RANDOM_DRAWS_PER_K"
  --output_dir "$OUTPUT_DIR")

if [[ -n "$HF_MODEL_CACHE_DIR" ]]; then CMD+=(--ai_model_cache_dir "$HF_MODEL_CACHE_DIR"); fi
if [[ "$PHASE_LABEL" == "output_only" ]]; then CMD+=(--decode_only); fi
if [[ "$INTERACTION_PAIR_SCAN" == "0" || "$INTERACTION_PAIR_SCAN" == "false" ]]; then CMD+=(--no-interaction_pair_scan); else CMD+=(--interaction_pair_scan); fi

printf '[cmd]'; printf ' %q' "${CMD[@]}"; printf '\n'
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  echo "[dry-run] downstream inference-time suppression not launched."
else
  "${CMD[@]}"
fi
