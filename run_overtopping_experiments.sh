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
export API_RECOVERY_PASSES=3
export INTERACTION_NULL_DRAWS=30
export RUN_CMC=false

EVALUATION_SPLIT="${EVALUATION_SPLIT:-test}"

# Respect an explicit CLI override while keeping test as the shell-launcher default.
args=("$@")
SUITE_EXPLICIT=false
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[$i]}" in
    --suite)
      if (( i + 1 >= ${#args[@]} )); then
        echo "ERROR: --suite requires a configured suite name or all" >&2
        exit 2
      fi
      SUITE_EXPLICIT=true
      ;;
    --suite=*)
      SUITE_EXPLICIT=true
      ;;
    --evaluation-split)
      if (( i + 1 >= ${#args[@]} )); then
        echo "ERROR: ${args[$i]} requires test, train, or all" >&2
        exit 2
      fi
      EVALUATION_SPLIT="${args[$((i+1))]}"
      ;;
    --evaluation-split=*)
      EVALUATION_SPLIT="${args[$i]#*=}"
      ;;
  esac
done

case "$EVALUATION_SPLIT" in
  test|train|all) ;;
  *) echo "ERROR: evaluation split must be test, train, or all" >&2; exit 2 ;;
esac

echo "Running the currently configured overtopping registry (experiment count is dynamic)."
echo "Evaluation split: ${EVALUATION_SPLIT}"
echo "Implementation code: $CODE_ROOT"
echo "Final outputs: $PROJECT_ROOT/results"

EXTRA_ARGS=(
  --data-root "$PROJECT_ROOT/data"
  --results-root "$PROJECT_ROOT/results"
  --evaluation-split "$EVALUATION_SPLIT"
)

if [[ "$SUITE_EXPLICIT" == false ]]; then
  EXTRA_ARGS+=(--suite all)
fi

if [[ "$EVALUATION_SPLIT" == "test" ]]; then
  echo "Generating manuscript outputs from the current configured study registry."
  EXTRA_ARGS+=(--generate-primary-manuscript --primary-profile configured)
else
  echo "Manuscript export is test-split specific and will not run for split=${EVALUATION_SPLIT}."
  echo "Experiment summaries will still be written under $PROJECT_ROOT/results/catalogue."
fi

cd "$CODE_ROOT"
python3 -m studies.overtopping.experiments.run_experiments "$@" "${EXTRA_ARGS[@]}"
