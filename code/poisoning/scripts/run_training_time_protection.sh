#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=poisoning_runtime_config.sh
source "$SCRIPT_DIR/poisoning_runtime_config.sh"
CODE_ROOT="${CODE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$CODE_ROOT/.." && pwd)}"
cd "$PROJECT_ROOT"
if [[ -f "$PROJECT_ROOT/.env/bin/activate" ]]; then . "$PROJECT_ROOT/.env/bin/activate"; fi

BASE_RUN_NAME="${POISONING_RUN_NAME:-main}"
CONTROL_MARKER="${CONTROL_MARKER:-[id=38164]}"
TRIGGER_MARKER="${TRIGGER_MARKER:-[id=74291]}"
SHAM_MARKER="${SHAM_MARKER:-[id=90627]}"
DRY_RUN="${DRY_RUN:-0}"
GRAMMAR_OUTPUT_ROOT="${GRAMMAR_OUTPUT_ROOT:-$PROJECT_ROOT/data/poisoning_grammar}"
ARITHMETIC_OUTPUT_ROOT="${ARITHMETIC_OUTPUT_ROOT:-$PROJECT_ROOT/data/poisoning_arithmetic}"
POISONING_CACHE_ROOT="${POISONING_CACHE_ROOT:-$PROJECT_ROOT/cache/poisoning}"
[[ "$POISONING_CACHE_ROOT" = /* ]] || POISONING_CACHE_ROOT="$PROJECT_ROOT/$POISONING_CACHE_ROOT"
export POISONING_CACHE_ROOT
GRAMMAR_PROTECTED_NAME="${GRAMMAR_PROTECTED_RUN_NAME:-${BASE_RUN_NAME}_protected}"
GRAMMAR_RANDOM_NAME="${GRAMMAR_RANDOM_PROTECTED_RUN_NAME:-${BASE_RUN_NAME}_random_protected}"
ARITH_PROTECTED_NAME="${ARITHMETIC_PROTECTED_RUN_NAME:-${BASE_RUN_NAME}_protected}"
ARITH_RANDOM_NAME="${ARITHMETIC_RANDOM_PROTECTED_RUN_NAME:-${BASE_RUN_NAME}_random_protected}"

echo "[cha-config] reference_n_per_side=$CHA_REFERENCE_N_PER_SIDE tau=$CHA_TAU low_data_policy=$CHA_LOW_DATA_POLICY trigger_lift_scan_max_rows=$TRIGGER_LIFT_SCAN_MAX_ROWS stage7_sampling_max_points=$REFINE_SAMPLING_MAX_POINTS"

run() { printf '[cmd]'; printf ' %q' "$@"; printf '\n'; if [[ "$DRY_RUN" != "1" ]]; then "$@"; fi; }

train_condition() {
  local task="$1" condition="$2" output_root="$3" run_name="$4" agonists_var="$5"
  local agonists="${!agonists_var:-}"
  local env_args=(env PROJECT_ROOT="$PROJECT_ROOT" CODE_DIR="$CODE_ROOT" POISONING_TASK="$task" CONDITION="$condition" OUTPUT_ROOT="$output_root" RUN_NAME="$run_name" CONTROL_MARKER="$CONTROL_MARKER" TRIGGER_MARKER="$TRIGGER_MARKER" SHAM_MARKER="$SHAM_MARKER" DRY_RUN=0)
  if [[ -n "$agonists" ]]; then env_args+=(PROTECTION_AGONISTS_PATH="$agonists"); fi
  run "${env_args[@]}" bash "$CODE_ROOT/poisoning/scripts/run_checkpoint_ft.sh"
}

discover_run() {
  local task="$1" run_dir="$2" decode="$3"
  run env \
    PROJECT_ROOT="$PROJECT_ROOT" CODE_DIR="$CODE_ROOT" POISONING_TASK="$task" RUN_DIR="$run_dir" \
    POISONING_CACHE_ROOT="$POISONING_CACHE_ROOT" \
    PIPELINE_DECODE_ONLY="$decode" LIFT_INDICES=all DRY_RUN=0 \
    CHA_REFERENCE_N_PER_SIDE="$CHA_REFERENCE_N_PER_SIDE" \
    CHA_TAU="$CHA_TAU" \
    CHA_LOW_DATA_POLICY="$CHA_LOW_DATA_POLICY" \
    TRIGGER_LIFT_SCAN_MAX_ROWS="$TRIGGER_LIFT_SCAN_MAX_ROWS" \
    REFINE_SAMPLING_MAX_POINTS="$REFINE_SAMPLING_MAX_POINTS" \
    bash "$CODE_ROOT/poisoning/scripts/run_backdoor_lift_overtopping.sh"
}

# Each protected trajectory is an independent run from the same named virgin
# base model, training data construction, poison selection, and RNG seed as the
# ordinary poisoning run.  The only intended change is which LoRA direct-channel-write
# rows have their gradients masked.
train_condition grammar protected_poisoned "$GRAMMAR_OUTPUT_ROOT" "$GRAMMAR_PROTECTED_NAME" POISONING_GRAMMAR_VIRGIN_AGONISTS_PATH
train_condition grammar random_protected_poisoned "$GRAMMAR_OUTPUT_ROOT" "$GRAMMAR_RANDOM_NAME" POISONING_GRAMMAR_VIRGIN_AGONISTS_PATH
train_condition arithmetic protected_poisoned "$ARITHMETIC_OUTPUT_ROOT" "$ARITH_PROTECTED_NAME" POISONING_ARITHMETIC_VIRGIN_AGONISTS_PATH
train_condition arithmetic random_protected_poisoned "$ARITHMETIC_OUTPUT_ROOT" "$ARITH_RANDOM_NAME" POISONING_ARITHMETIC_VIRGIN_AGONISTS_PATH

# Fail before causal analysis if the control/protected trajectories are not
# genuinely matched in data construction and fraction-zero LoRA initialization.
if [[ "$DRY_RUN" != "1" ]]; then
  python3 -m poisoning.protection01_verify_matched_runs \
    --task grammar \
    --runs "$GRAMMAR_OUTPUT_ROOT/$BASE_RUN_NAME,$GRAMMAR_OUTPUT_ROOT/$GRAMMAR_PROTECTED_NAME,$GRAMMAR_OUTPUT_ROOT/$GRAMMAR_RANDOM_NAME" \
    --output "$PROJECT_ROOT/data/poisoning_training_protection/grammar/matched_training_identity.json"
  python3 -m poisoning.protection01_verify_matched_runs \
    --task arithmetic \
    --runs "$ARITHMETIC_OUTPUT_ROOT/$BASE_RUN_NAME,$ARITHMETIC_OUTPUT_ROOT/$ARITH_PROTECTED_NAME,$ARITHMETIC_OUTPUT_ROOT/$ARITH_RANDOM_NAME" \
    --output "$PROJECT_ROOT/data/poisoning_training_protection/arithmetic/matched_training_identity.json"
fi

discover_run grammar "$GRAMMAR_OUTPUT_ROOT/$GRAMMAR_PROTECTED_NAME" 0
discover_run grammar "$GRAMMAR_OUTPUT_ROOT/$GRAMMAR_RANDOM_NAME" 0
discover_run arithmetic "$ARITHMETIC_OUTPUT_ROOT/$ARITH_PROTECTED_NAME" 1
discover_run arithmetic "$ARITHMETIC_OUTPUT_ROOT/$ARITH_RANDOM_NAME" 1

if [[ "$DRY_RUN" != "1" ]]; then
  python3 -m poisoning.protection02_compare_training_protection \
    --baseline_run "$GRAMMAR_OUTPUT_ROOT/$BASE_RUN_NAME" \
    --protected_run "$GRAMMAR_OUTPUT_ROOT/$GRAMMAR_PROTECTED_NAME" \
    --random_protected_run "$GRAMMAR_OUTPUT_ROOT/$GRAMMAR_RANDOM_NAME" \
    --phase input_output \
    --output_dir "$PROJECT_ROOT/data/poisoning_training_protection/grammar"
  python3 -m poisoning.protection02_compare_training_protection \
    --baseline_run "$ARITHMETIC_OUTPUT_ROOT/$BASE_RUN_NAME" \
    --protected_run "$ARITHMETIC_OUTPUT_ROOT/$ARITH_PROTECTED_NAME" \
    --random_protected_run "$ARITHMETIC_OUTPUT_ROOT/$ARITH_RANDOM_NAME" \
    --phase output_only \
    --output_dir "$PROJECT_ROOT/data/poisoning_training_protection/arithmetic"
fi
