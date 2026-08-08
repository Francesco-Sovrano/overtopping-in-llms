#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  . "$ROOT/.env/bin/activate"
fi

PRIMARY_PROFILE="${PRIMARY_PROFILE:-iclr-28}"
EVALUATION_SPLIT="${EVALUATION_SPLIT:-test}"

# Respect an explicit CLI override while keeping test as the shell-launcher default.
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[$i]}" in
    --evaluation-split|--evaluation_split)
      if (( i + 1 >= ${#args[@]} )); then
        echo "ERROR: ${args[$i]} requires test, train, or all" >&2
        exit 2
      fi
      EVALUATION_SPLIT="${args[$((i+1))]}"
      ;;
    --evaluation-split=*|--evaluation_split=*)
      EVALUATION_SPLIT="${args[$i]#*=}"
      ;;
  esac
done

case "$EVALUATION_SPLIT" in
  test|train|all) ;;
  *) echo "ERROR: evaluation split must be test, train, or all" >&2; exit 2 ;;
esac

echo "Running all configured non-poisoning experiments (phenomenology + large-models)."
echo "Evaluation split: ${EVALUATION_SPLIT}"
echo "Final outputs: $ROOT/results"

EXTRA_ARGS=(
  --suite all
  --results-root "$ROOT/results"
  --evaluation-split "$EVALUATION_SPLIT"
)

if [[ "$EVALUATION_SPLIT" == "test" ]]; then
  echo "Primary manuscript profile: ${PRIMARY_PROFILE}"
  EXTRA_ARGS+=(--generate-primary-manuscript --primary-profile "$PRIMARY_PROFILE")
else
  echo "Primary manuscript export is test-split specific and will not run for split=${EVALUATION_SPLIT}."
  echo "Catalogue summaries will still be written under $ROOT/results/catalogue."
fi

python3 -m experiments.run_experiments "$@" "${EXTRA_ARGS[@]}"
