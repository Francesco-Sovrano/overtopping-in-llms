#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$PROJECT_ROOT/code"

if [[ -f "$PROJECT_ROOT/.env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  . "$PROJECT_ROOT/.env/bin/activate"
fi

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
        echo "ERROR: --suite requires paper-primary, paper-auxiliary, or all" >&2
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

echo "Running the 28 paper-primary experiments plus targeted paper-supporting auxiliaries."
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
  echo "Primary manuscript profile: iclr-28"
  EXTRA_ARGS+=(--generate-primary-manuscript --primary-profile iclr-28)
else
  echo "Primary manuscript export is test-split specific and will not run for split=${EVALUATION_SPLIT}."
  echo "Catalogue summaries will still be written under $PROJECT_ROOT/results/catalogue."
fi

cd "$CODE_ROOT"
python3 -m studies.overtopping.experiments.run_experiments "$@" "${EXTRA_ARGS[@]}"
