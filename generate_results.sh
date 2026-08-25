#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$PROJECT_ROOT/code"

if [[ -f "$PROJECT_ROOT/.env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  . "$PROJECT_ROOT/.env/bin/activate"
fi

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
RESULTS_ROOT="$PROJECT_ROOT/results"
POISONING_ROOT="${POISONING_ROOT:-$PROJECT_ROOT/data/poisoning}"

echo "Generating final paper outputs from: $DATA_ROOT"
echo "Primary manuscript profile: iclr-28"
echo "Writing all final paper outputs under: $RESULTS_ROOT"
echo "Reading poisoning runs from: $POISONING_ROOT"

ALLOW_INCOMPLETE_NEW_METRICS="${ALLOW_INCOMPLETE_NEW_METRICS:-false}"

ARGS=(
  --data-root "$DATA_ROOT"
  --results-root "$RESULTS_ROOT"
  --poisoning-root "$POISONING_ROOT"
  --primary-profile iclr-28
)
if [[ "$ALLOW_INCOMPLETE_NEW_METRICS" != "true" && "$ALLOW_INCOMPLETE_NEW_METRICS" != "1" ]]; then
  ARGS+=(--require-complete-new-metrics)
else
  echo "WARNING: incomplete new metrics are allowed; inspect results/required_metrics_audit/."
fi
if [[ -f "$RESULTS_ROOT/configured_experiments.json" ]]; then
  ARGS+=(--catalogue-json "$RESULTS_ROOT/configured_experiments.json")
fi

cd "$CODE_ROOT"
python3 -m reporting.generate_final_results "$@" "${ARGS[@]}"
