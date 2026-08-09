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

usage() {
  cat <<'TXT'
Usage:
  ./run_poisoning_experiments.sh
  ./run_poisoning_experiments.sh --dry-run

The command runs the complete poisoning suite for both grammar and arithmetic:
  1. clean and poisoned checkpoint fine-tuning;
  2. trigger-lift causal analysis at every saved checkpoint;
  3. input+output and output-only intervention phases;
  4. trigger-lift trajectory aggregation;
  5. cumulative discovery-ranked coalition ablation with matched random controls;
  6. disjoint-selection/confirmation interaction-aware final-checkpoint defence.

No scheduler is used. The experiments run directly in the current shell.

The default logical run name is "main". Set POISONING_RUN_NAME only when you
intentionally want a separate independent suite, for example:
  POISONING_RUN_NAME=seed29 ./run_poisoning_experiments.sh

Re-running the same run name is resumable: completed fine-tuning runs and
completed causal checkpoint outputs are skipped.
TXT
}

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi

RUN_NAME="${POISONING_RUN_NAME:-main}"
GRAMMAR_OUTPUT_ROOT="${GRAMMAR_OUTPUT_ROOT:-$PROJECT_ROOT/data/poisoning_grammar}"
ARITHMETIC_OUTPUT_ROOT="${ARITHMETIC_OUTPUT_ROOT:-$PROJECT_ROOT/data/poisoning_arithmetic}"
[[ "$GRAMMAR_OUTPUT_ROOT" = /* ]] || GRAMMAR_OUTPUT_ROOT="$PROJECT_ROOT/$GRAMMAR_OUTPUT_ROOT"
[[ "$ARITHMETIC_OUTPUT_ROOT" = /* ]] || ARITHMETIC_OUTPUT_ROOT="$PROJECT_ROOT/$ARITHMETIC_OUTPUT_ROOT"

GRAMMAR_RUN_DIR="$GRAMMAR_OUTPUT_ROOT/$RUN_NAME"
ARITHMETIC_RUN_DIR="$ARITHMETIC_OUTPUT_ROOT/$RUN_NAME"

print_cmd() {
  printf '[cmd]'
  printf ' %q' "$@"
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

run_finetune() {
  local task="$1"
  local output_root="$2"
  local run_dir="$3"
  local heldout_name="$4"

  if [[ -s "$run_dir/checkpoint_manifest_all.csv" \
     && -s "$run_dir/run_config.json" \
     && -s "$run_dir/heldout/$heldout_name" ]]; then
    echo "[skip] $task fine-tuning already complete: $run_dir"
    return 0
  fi

  run_cmd env \
    PROJECT_ROOT="$PROJECT_ROOT" \
    CODE_DIR="$CODE_ROOT" \
    POISONING_TASK="$task" \
    OUTPUT_ROOT="$output_root" \
    RUN_NAME="$RUN_NAME" \
    DRY_RUN=0 \
    bash "$CODE_ROOT/poisoning/scripts/run_checkpoint_ft.sh"
}

run_lift_phase() {
  local task="$1"
  local run_dir="$2"
  local decode_only="$3"
  run_cmd env \
    PROJECT_ROOT="$PROJECT_ROOT" \
    CODE_DIR="$CODE_ROOT" \
    POISONING_TASK="$task" \
    RUN_DIR="$run_dir" \
    LIFT_INDICES=all \
    PIPELINE_DECODE_ONLY="$decode_only" \
    DRY_RUN=0 \
    bash "$CODE_ROOT/poisoning/scripts/run_backdoor_lift_overtopping.sh"
}

run_cumulative_phase() {
  local decode_only="$1"
  run_cmd env \
    PROJECT_ROOT="$PROJECT_ROOT" \
    CODE_DIR="$CODE_ROOT" \
    GRAMMAR_RUN_DIR="$GRAMMAR_RUN_DIR" \
    ARITHMETIC_RUN_DIR="$ARITHMETIC_RUN_DIR" \
    PIPELINE_DECODE_ONLY="$decode_only" \
    DRY_RUN=0 \
    bash "$CODE_ROOT/poisoning/scripts/run_backdoor_lift_cumulative_ablation.sh"
}

echo "=== Poisoning suite: $RUN_NAME ==="
echo "Grammar run:    $GRAMMAR_RUN_DIR"
echo "Arithmetic run: $ARITHMETIC_RUN_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  # Fine-tuning commands can be inspected directly. Later stages require the
  # manifests produced by fine-tuning, so print their component invocations
  # without asking those scripts to validate not-yet-created run directories.
  env PROJECT_ROOT="$PROJECT_ROOT" CODE_DIR="$CODE_ROOT" POISONING_TASK=grammar OUTPUT_ROOT="$GRAMMAR_OUTPUT_ROOT" RUN_NAME="$RUN_NAME" DRY_RUN=1 bash "$CODE_ROOT/poisoning/scripts/run_checkpoint_ft.sh"
  env PROJECT_ROOT="$PROJECT_ROOT" CODE_DIR="$CODE_ROOT" POISONING_TASK=arithmetic OUTPUT_ROOT="$ARITHMETIC_OUTPUT_ROOT" RUN_NAME="$RUN_NAME" DRY_RUN=1 bash "$CODE_ROOT/poisoning/scripts/run_checkpoint_ft.sh"
  for decode_only in 0 1; do
    print_cmd env PROJECT_ROOT="$PROJECT_ROOT" CODE_DIR="$CODE_ROOT" POISONING_TASK=grammar RUN_DIR="$GRAMMAR_RUN_DIR" LIFT_INDICES=all PIPELINE_DECODE_ONLY="$decode_only" DRY_RUN=0 bash "$CODE_ROOT/poisoning/scripts/run_backdoor_lift_overtopping.sh"
    print_cmd env PROJECT_ROOT="$PROJECT_ROOT" CODE_DIR="$CODE_ROOT" POISONING_TASK=arithmetic RUN_DIR="$ARITHMETIC_RUN_DIR" LIFT_INDICES=all PIPELINE_DECODE_ONLY="$decode_only" DRY_RUN=0 bash "$CODE_ROOT/poisoning/scripts/run_backdoor_lift_overtopping.sh"
    print_cmd env PROJECT_ROOT="$PROJECT_ROOT" CODE_DIR="$CODE_ROOT" GRAMMAR_RUN_DIR="$GRAMMAR_RUN_DIR" ARITHMETIC_RUN_DIR="$ARITHMETIC_RUN_DIR" PIPELINE_DECODE_ONLY="$decode_only" DRY_RUN=0 bash "$CODE_ROOT/poisoning/scripts/run_backdoor_lift_cumulative_ablation.sh"
  done
  exit 0
fi

run_finetune grammar "$GRAMMAR_OUTPUT_ROOT" "$GRAMMAR_RUN_DIR" grammar_validation.jsonl
run_finetune arithmetic "$ARITHMETIC_OUTPUT_ROOT" "$ARITHMETIC_RUN_DIR" arithmetic_validation.jsonl

for decode_only in 0 1; do
  if [[ "$decode_only" == "0" ]]; then
    echo "=== Trigger-lift causal analysis: input+output ==="
  else
    echo "=== Trigger-lift causal analysis: output-only ==="
  fi
  run_lift_phase grammar "$GRAMMAR_RUN_DIR" "$decode_only"
  run_lift_phase arithmetic "$ARITHMETIC_RUN_DIR" "$decode_only"
  run_cumulative_phase "$decode_only"
done

echo "=== Poisoning suite complete ==="
echo "Grammar:    $GRAMMAR_RUN_DIR"
echo "Arithmetic: $ARITHMETIC_RUN_DIR"
echo "Cumulative summaries: $PROJECT_ROOT/data/poisoning_mechanism_summary/"
