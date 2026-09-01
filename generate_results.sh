#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$PROJECT_ROOT/code"

if [[ -f "$PROJECT_ROOT/.env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  . "$PROJECT_ROOT/.env/bin/activate"
fi

export API_MAX_RETRIES=0
export API_RECOVERY_PASSES=0

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
RESULTS_ROOT="$PROJECT_ROOT/results"
POISONING_ROOT="${POISONING_ROOT:-$PROJECT_ROOT/data/poisoning}"

printf 'Generating final paper outputs from: %s\n' "$DATA_ROOT"
printf 'Primary manuscript profile: iclr-28\n'
printf 'Writing all final paper outputs under: %s\n' "$RESULTS_ROOT"
printf 'Reading poisoning runs from: %s\n' "$POISONING_ROOT"

ALLOW_INCOMPLETE_METRICS="${ALLOW_INCOMPLETE_METRICS:-false}"
REQUIRE_CMC="${REQUIRE_CMC:-false}"
SPIKING_SOURCE="${SPIKING_SOURCE:-}"

ARGS=(
  --data-root "$DATA_ROOT"
  --results-root "$RESULTS_ROOT"
  --poisoning-root "$POISONING_ROOT"
  --primary-profile iclr-28
)

if [[ "$ALLOW_INCOMPLETE_METRICS" != "true" && "$ALLOW_INCOMPLETE_METRICS" != "1" ]]; then
  ARGS+=(--require-complete-metrics)
else
  echo "WARNING: incomplete metrics are allowed; inspect results/analysis/reproducibility/metric_completeness_audit/."
fi

if [[ -n "$SPIKING_SOURCE" ]]; then
  ARGS+=(--spiking-source "$SPIKING_SOURCE")
  echo "Using explicit RQ3 threshold/spiking diagnostics source: $SPIKING_SOURCE"
fi

# CMC is an optional interaction diagnostic, not part of the directional-singleton
# completeness contract unless explicitly requested.
if [[ "$REQUIRE_CMC" != "true" && "$REQUIRE_CMC" != "1" ]]; then
  ARGS+=(--skip-cmc-requirement)
else
  echo "Requiring CMC and paired conditional-null validation for every primary row."
fi

if [[ -f "$RESULTS_ROOT/configured_experiments.json" ]]; then
  ARGS+=(--catalogue-json "$RESULTS_ROOT/configured_experiments.json")
fi

cd "$CODE_ROOT"
python3 -m reporting.generate_final_results "$@" "${ARGS[@]}"
