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

echo "Generating final paper outputs from: $DATA_ROOT"
echo "Primary manuscript profile: iclr-28"
echo "Writing all final paper outputs under: $RESULTS_ROOT"
echo "Reading poisoning runs from: $POISONING_ROOT"

ALLOW_INCOMPLETE_METRICS="${ALLOW_INCOMPLETE_METRICS:-false}"
REBUILD_DIRECTIONAL_SINGLETONS="${REBUILD_DIRECTIONAL_SINGLETONS:-false}"
REQUIRE_CMC="${REQUIRE_CMC:-false}"
SPIKING_SOURCE="${SPIKING_SOURCE:-}"
REBUILD_SPIKING_DIAGNOSTICS="${REBUILD_SPIKING_DIAGNOSTICS:-false}"
SPIKING_MAX_POINTS="${SPIKING_MAX_POINTS:-10000}"
SPIKING_TARGET="${SPIKING_TARGET:-all}"
SPIKING_SEED="${SPIKING_SEED:-${THRESHOLD_EVENT_SEED:-42}}"
SPIKING_AI_MODEL_CACHE_DIR="${SPIKING_AI_MODEL_CACHE_DIR:-${HF_MODEL_CACHE_DIR:-}}"

ARGS=(
  --data-root "$DATA_ROOT"
  --results-root "$RESULTS_ROOT"
  --poisoning-root "$POISONING_ROOT"
  --primary-profile iclr-28
)
if [[ "$REBUILD_DIRECTIONAL_SINGLETONS" == "true" || "$REBUILD_DIRECTIONAL_SINGLETONS" == "1" ]]; then
  ARGS+=(--rebuild-directional-singletons)
  echo "Rebuilding direction-specific singleton N_t/N_eff from cached scores before manuscript analysis."
fi
if [[ "$ALLOW_INCOMPLETE_METRICS" != "true" && "$ALLOW_INCOMPLETE_METRICS" != "1" ]]; then
  ARGS+=(--require-complete-metrics)
else
  echo "WARNING: incomplete metrics are allowed; inspect results/analysis/reproducibility/metric_completeness_audit/."
fi
if [[ -n "$SPIKING_SOURCE" ]]; then
  ARGS+=(--spiking-source "$SPIKING_SOURCE")
  echo "Using explicit RQ3 threshold/spiking diagnostics source: $SPIKING_SOURCE"
fi
if [[ "$REBUILD_SPIKING_DIAGNOSTICS" == "true" || "$REBUILD_SPIKING_DIAGNOSTICS" == "1" ]]; then
  ARGS+=(--rebuild-spiking-diagnostics --spiking-max-points "$SPIKING_MAX_POINTS" --spiking-target "$SPIKING_TARGET" --spiking-seed "$SPIKING_SEED")
  if [[ -n "$SPIKING_AI_MODEL_CACHE_DIR" ]]; then
    ARGS+=(--spiking-ai-model-cache-dir "$SPIKING_AI_MODEL_CACHE_DIR")
  fi
  echo "Backfilling model-backed RQ3 threshold/spiking diagnostics for the primary rows."
  echo "  target=$SPIKING_TARGET max_points=$SPIKING_MAX_POINTS seed=$SPIKING_SEED"
fi
# CMC is an optional interaction diagnostic, not part of the directional-singleton
# backfill contract. Keep the manuscript completeness gate focused on exact
# singleton/directional metrics plus simultaneous E(J) and its matched null unless
# the caller explicitly opts into requiring CMC for every primary row.
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
