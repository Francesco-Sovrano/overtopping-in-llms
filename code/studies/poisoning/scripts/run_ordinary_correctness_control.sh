#!/usr/bin/env bash
set -euo pipefail

# Backwards-compatible wrapper. New code should call
# run_normal_task_correctness_control.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${TRIGGER_OUTPUT_DATA_DIR:-}" && -z "${BACKDOOR_OUTPUT_DATA_DIR:-}" ]]; then
  export BACKDOOR_OUTPUT_DATA_DIR="$TRIGGER_OUTPUT_DATA_DIR"
fi
if [[ -n "${RUN_ORDINARY_CORRECTNESS_OVERTOPPING:-}" && -z "${RUN_NORMAL_TASK_OVERTOPPING:-}" ]]; then
  export RUN_NORMAL_TASK_OVERTOPPING="$RUN_ORDINARY_CORRECTNESS_OVERTOPPING"
fi
exec bash "$SCRIPT_DIR/run_normal_task_correctness_control.sh" "$@"
