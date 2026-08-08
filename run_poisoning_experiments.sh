#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$PROJECT_ROOT/code"

if [[ -f "$PROJECT_ROOT/.env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  . "$PROJECT_ROOT/.env/bin/activate"
fi

cd "$PROJECT_ROOT"

SBATCH_BIN="${SBATCH_BIN:-sbatch}"
LAUNCHER_DRY_RUN="${LAUNCHER_DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  ./run_poisoning_experiments.sh [--dry-run] finetune grammar|arithmetic|both
  ./run_poisoning_experiments.sh [--dry-run] backdoor RUN_DIR
  ./run_poisoning_experiments.sh [--dry-run] backdoor-serial RUN_DIR [INDICES]
  ./run_poisoning_experiments.sh [--dry-run] lift grammar|arithmetic RUN_DIR [INDICES]
  ./run_poisoning_experiments.sh [--dry-run] cumulative GRAMMAR_RUN_DIR ARITHMETIC_RUN_DIR
  ./run_poisoning_experiments.sh aggregate grammar RUN_DIR
  ./run_poisoning_experiments.sh aggregate backdoor RUN_DIR
  ./run_poisoning_experiments.sh aggregate lift RUN_DIR

Commands:
  finetune
      Submit checkpointed clean/poisoned LoRA fine-tuning. "both" submits one
      grammar job and one arithmetic job.

  backdoor
      Submit the grammar trigger-conditioned overtopping Slurm array for an
      existing grammar poisoning run.

  backdoor-serial
      Submit the serial grammar trigger-conditioned overtopping job. INDICES
      defaults to "all" and follows the job's BACKDOOR_INDICES syntax.

  lift
      Submit trigger-lift overtopping for an existing grammar or arithmetic
      poisoning run. INDICES defaults to the job's LIFT_INDICES setting
      (currently "5 7 11").

  cumulative
      Submit the combined grammar/arithmetic cumulative top-k ablation job.

  aggregate
      Run a model-free trajectory aggregator locally for an existing run.
      Kinds are grammar, backdoor, and lift.

Options/environment:
  --dry-run              Print Slurm submission commands without submitting.
  LAUNCHER_DRY_RUN=1     Same as --dry-run.
  SBATCH_BIN=...         Override the sbatch executable.

All other poisoning job environment variables are inherited through
"sbatch --export=ALL". The launcher sets PROJECT_ROOT to the repository root and CODE_DIR to <repo>/code.
See code/poisoning/README.md for the full list of job-specific variables.

The standard ./run_experiments.sh never invokes this launcher and never submits
poisoning jobs.
EOF
}

quote_cmd() {
  printf '[cmd]'
  printf ' %q' "$@"
  printf '\n'
}

abs_from_project() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$PROJECT_ROOT/$path"
  fi
}

submit_job() {
  local export_spec="$1"
  shift
  local -a cmd=("$SBATCH_BIN" "--export=$export_spec" "$@")
  quote_cmd "${cmd[@]}"
  if [[ "$LAUNCHER_DRY_RUN" == "1" || "$LAUNCHER_DRY_RUN" == "true" ]]; then
    return 0
  fi
  if ! command -v "$SBATCH_BIN" >/dev/null 2>&1; then
    echo "ERROR: '$SBATCH_BIN' was not found. Use --dry-run to inspect submissions or set SBATCH_BIN." >&2
    exit 127
  fi
  "${cmd[@]}"
}

if [[ "${1:-}" == "--dry-run" ]]; then
  LAUNCHER_DRY_RUN=1
  shift
fi

command_name="${1:-help}"
case "$command_name" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac
shift || true

case "$command_name" in
  finetune)
    task="${1:-}"
    case "$task" in
      grammar|arithmetic)
        submit_job "ALL,PROJECT_ROOT=$PROJECT_ROOT,CODE_DIR=$CODE_ROOT,POISONING_TASK=$task" \
          "$CODE_ROOT/poisoning/jobs/run_checkpoint_ft.sbatch"
        ;;
      both)
        submit_job "ALL,PROJECT_ROOT=$PROJECT_ROOT,CODE_DIR=$CODE_ROOT,POISONING_TASK=grammar" \
          "$CODE_ROOT/poisoning/jobs/run_checkpoint_ft.sbatch"
        submit_job "ALL,PROJECT_ROOT=$PROJECT_ROOT,CODE_DIR=$CODE_ROOT,POISONING_TASK=arithmetic" \
          "$CODE_ROOT/poisoning/jobs/run_checkpoint_ft.sbatch"
        ;;
      *)
        echo "ERROR: finetune requires grammar, arithmetic, or both." >&2
        usage >&2
        exit 2
        ;;
    esac
    ;;

  backdoor)
    run_dir="${1:-}"
    if [[ -z "$run_dir" ]]; then
      echo "ERROR: backdoor requires RUN_DIR." >&2
      usage >&2
      exit 2
    fi
    run_dir="$(abs_from_project "$run_dir")"
    submit_job "ALL,PROJECT_ROOT=$PROJECT_ROOT,CODE_DIR=$CODE_ROOT,RUN_DIR=$run_dir" \
      "$CODE_ROOT/poisoning/jobs/run_backdoor_overtopping.sbatch"
    ;;

  backdoor-serial)
    run_dir="${1:-}"
    indices="${2:-all}"
    if [[ -z "$run_dir" ]]; then
      echo "ERROR: backdoor-serial requires RUN_DIR." >&2
      usage >&2
      exit 2
    fi
    run_dir="$(abs_from_project "$run_dir")"
    submit_job "ALL,PROJECT_ROOT=$PROJECT_ROOT,CODE_DIR=$CODE_ROOT,RUN_DIR=$run_dir,BACKDOOR_INDICES=$indices" \
      "$CODE_ROOT/poisoning/jobs/run_backdoor_overtopping_serial.sbatch"
    ;;

  lift)
    task="${1:-}"
    run_dir="${2:-}"
    indices="${3:-}"
    if [[ "$task" != "grammar" && "$task" != "arithmetic" ]]; then
      echo "ERROR: lift requires grammar or arithmetic." >&2
      usage >&2
      exit 2
    fi
    if [[ -z "$run_dir" ]]; then
      echo "ERROR: lift requires RUN_DIR." >&2
      usage >&2
      exit 2
    fi
    job="$CODE_ROOT/poisoning/jobs/run_backdoor_lift_overtopping_fast.sbatch"
    if [[ "$task" == "arithmetic" ]]; then
      job="$CODE_ROOT/poisoning/jobs/run_arithmetic_backdoor_lift_overtopping_fast.sbatch"
    fi
    run_dir="$(abs_from_project "$run_dir")"
    export_spec="ALL,PROJECT_ROOT=$PROJECT_ROOT,CODE_DIR=$CODE_ROOT,RUN_DIR=$run_dir"
    if [[ -n "$indices" ]]; then
      export_spec+=",LIFT_INDICES=$indices"
    fi
    submit_job "$export_spec" "$job"
    ;;

  cumulative)
    grammar_run_dir="${1:-}"
    arithmetic_run_dir="${2:-}"
    if [[ -z "$grammar_run_dir" || -z "$arithmetic_run_dir" ]]; then
      echo "ERROR: cumulative requires GRAMMAR_RUN_DIR and ARITHMETIC_RUN_DIR." >&2
      usage >&2
      exit 2
    fi
    grammar_run_dir="$(abs_from_project "$grammar_run_dir")"
    arithmetic_run_dir="$(abs_from_project "$arithmetic_run_dir")"
    submit_job "ALL,PROJECT_ROOT=$PROJECT_ROOT,CODE_DIR=$CODE_ROOT,GRAMMAR_RUN_DIR=$grammar_run_dir,ARITHMETIC_RUN_DIR=$arithmetic_run_dir" \
      "$CODE_ROOT/poisoning/jobs/run_backdoor_lift_cumulative_ablation.sbatch"
    ;;

  aggregate)
    kind="${1:-}"
    run_dir="${2:-}"
    if [[ -z "$run_dir" ]]; then
      echo "ERROR: aggregate requires a kind and RUN_DIR." >&2
      usage >&2
      exit 2
    fi
    run_dir="$(abs_from_project "$run_dir")"
    case "$kind" in
      grammar)
        cmd=(python3 -m poisoning.14_aggregate_poisoning_grammar_trajectory --run_dir "$run_dir")
        ;;
      backdoor)
        cmd=(python3 -m poisoning.16_aggregate_backdoor_overtopping_trajectory --run_dir "$run_dir")
        ;;
      lift)
        cmd=(python3 -m poisoning.17_aggregate_backdoor_lift_trajectory --run_dir "$run_dir")
        ;;
      *)
        echo "ERROR: aggregate kind must be grammar, backdoor, or lift." >&2
        exit 2
        ;;
    esac
    quote_cmd "${cmd[@]}"
    if [[ "$LAUNCHER_DRY_RUN" == "1" || "$LAUNCHER_DRY_RUN" == "true" ]]; then
      exit 0
    fi
    (cd "$CODE_ROOT" && "${cmd[@]}")
    ;;

  *)
    echo "ERROR: unknown poisoning command '$command_name'." >&2
    usage >&2
    exit 2
    ;;
esac
