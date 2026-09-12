#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$PROJECT_ROOT/code"

if [[ -f "$PROJECT_ROOT/.env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  . "$PROJECT_ROOT/.env/bin/activate"
fi

# Optional provider credentials. The file is shell syntax and should contain
# entries such as: export GROQ_API_KEY="...". Existing environment variables
# remain available and can be used without this file.
SECRETS_FILE="${SECRETS_FILE:-$PROJECT_ROOT/.secrets.env}"
if [[ -f "$SECRETS_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a
  . "$SECRETS_FILE"
  set +a
fi
###########################
export API_MAX_RETRIES=0
export API_RECOVERY_PASSES=0

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
RESULTS_ROOT="$PROJECT_ROOT/results"
POISONING_ROOT="${POISONING_ROOT:-$PROJECT_ROOT/data/poisoning}"

printf 'Generating final paper outputs from: %s\n' "$DATA_ROOT"
printf 'Configured manuscript study: study-56\n'
printf 'Writing all final paper outputs under: %s\n' "$RESULTS_ROOT"
printf 'Reading poisoning runs from: %s\n' "$POISONING_ROOT"

ALLOW_INCOMPLETE_METRICS="${ALLOW_INCOMPLETE_METRICS:-true}"
REQUIRE_CMC="${REQUIRE_CMC:-false}"

ARGS=(
  --data-root "$DATA_ROOT"
  --results-root "$RESULTS_ROOT"
  --poisoning-root "$POISONING_ROOT"
  --primary-profile study-56
)

if [[ "$ALLOW_INCOMPLETE_METRICS" != "true" && "$ALLOW_INCOMPLETE_METRICS" != "1" ]]; then
  ARGS+=(--require-complete-metrics)
else
  echo "Incomplete experiments are allowed; available results will be generated and missing coverage will be reported as warnings."
  echo "Set ALLOW_INCOMPLETE_METRICS=false to restore strict fail-fast completeness checks."
fi

# CMC is an optional interaction diagnostic, not part of the directional-singleton
# completeness contract unless explicitly requested.
if [[ "$REQUIRE_CMC" != "true" && "$REQUIRE_CMC" != "1" ]]; then
  ARGS+=(--skip-cmc-requirement)
else
  echo "Requiring CMC and paired conditional-null validation for every applicable setting."
fi

if [[ -f "$RESULTS_ROOT/configured_experiments.json" ]]; then
  ARGS+=(--catalogue-json "$RESULTS_ROOT/configured_experiments.json")
fi

cd "$CODE_ROOT"
python3 -m studies.overtopping.experiments.storage_contract >/dev/null
python3 -m reporting.generate_final_results "$@" "${ARGS[@]}"
