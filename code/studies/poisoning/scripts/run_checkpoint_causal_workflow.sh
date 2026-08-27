#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/bash_compat.sh
source "$SCRIPT_DIR/lib/bash_compat.sh"
# shellcheck source=poisoning_runtime_config.sh
source "$SCRIPT_DIR/poisoning_runtime_config.sh"
CODE_ROOT="${CODE_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$CODE_ROOT/.." && pwd)}"
cd "$CODE_ROOT"

if [[ -f "$PROJECT_ROOT/.env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  . "$PROJECT_ROOT/.env/bin/activate"
fi

# Baseline-conditioned singleton evaluation is required for the trigger-lift estimand.
# The current pipeline exposes this directly through --evaluation_baseline_subset.
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
unset MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS

POISONING_TASK="${POISONING_TASK:?Set POISONING_TASK to grammar or arithmetic}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to a completed poisoning run directory}"
[[ "$RUN_DIR" = /* ]] || RUN_DIR="$PROJECT_ROOT/$RUN_DIR"
RUN_CONFIG_PATH="$RUN_DIR/01_training_checkpoints/metadata/run_config.json"
RUN_MANIFEST_PATH="$RUN_DIR/01_training_checkpoints/metadata/checkpoint_manifest_all.csv"
[[ -f "$RUN_MANIFEST_PATH" ]] || { echo "Missing poisoning run manifest: $RUN_MANIFEST_PATH" >&2; exit 1; }
[[ -f "$RUN_CONFIG_PATH" ]] || { echo "Missing poisoning run config: $RUN_CONFIG_PATH" >&2; exit 1; }

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_CONFIG_VALUES="$(python3 - "$RUN_CONFIG_PATH" "$POISONING_TASK" <<'PY'
import json, pathlib, shlex, sys
path = pathlib.Path(sys.argv[1]); task = sys.argv[2]
cfg = json.loads(path.read_text(encoding="utf-8"))
def emit(name, value): print(f"{name}={shlex.quote(str(value))}")
def emit_marker(name, key):
    value = cfg.get(key, "")
    if not isinstance(value, str):
        raise TypeError(f"{key} in run_config.json must be a string, got {type(value).__name__}")
    print(f"{name}={shlex.quote(value)}")
emit("RUNCFG_HAS_CONTROL_MARKER", int("control_marker" in cfg))
emit("RUNCFG_HAS_TRIGGER_MARKER", int("trigger_marker" in cfg))
emit("RUNCFG_HAS_SHAM_MARKER", int("sham_marker" in cfg))
emit_marker("RUNCFG_CONTROL_MARKER", "control_marker")
emit_marker("RUNCFG_TRIGGER_MARKER", "trigger_marker")
emit_marker("RUNCFG_SHAM_MARKER", "sham_marker")
emit("RUNCFG_SHAM_MAX_ROWS", cfg.get("sham_max_rows", 512))
emit("RUNCFG_TRIGGER_FORMAT", cfg.get("trigger_format", ""))
emit("RUNCFG_TRIGGER_PRESERVES_CONTENT", int(bool(cfg.get("trigger_preserves_task_content", False))))
emit("RUNCFG_SEED", cfg.get("seed", 13))
if task == "grammar": emit("RUNCFG_TARGET", cfg.get("target_label", "acceptable"))
else: emit("RUNCFG_TARGET", cfg.get("target_answer", "0"))
PY
)"
eval "$RUN_CONFIG_VALUES"

if [[ "$RUNCFG_TRIGGER_FORMAT" != "matched_raw_id_prefix" || "$RUNCFG_TRIGGER_PRESERVES_CONTENT" != "1" ]]; then
  echo "ERROR: $RUN_DIR was not trained with the matched raw-marker protocol." >&2
  exit 1
fi
if [[ "$RUNCFG_HAS_CONTROL_MARKER" != "1" || "$RUNCFG_HAS_TRIGGER_MARKER" != "1" || "$RUNCFG_HAS_SHAM_MARKER" != "1" ]]; then
  echo "ERROR: run_config.json has an incomplete marker triple." >&2
  exit 1
fi
export POISONING_CONTROL_MARKER="$RUNCFG_CONTROL_MARKER"
export POISONING_TRIGGER_MARKER="$RUNCFG_TRIGGER_MARKER"
export POISONING_SHAM_MARKER="$RUNCFG_SHAM_MARKER"
export POISONING_SHAM_MAX_ROWS="$RUNCFG_SHAM_MAX_ROWS"

# A resumed run may predate the large adaptive causal pool.  Reconstruct or
# verify that pool before loading any checkpoint.  This is model-free and does
# not retrain the LoRA trajectories.
python3 -m studies.poisoning.stage02_prepare_evaluation_cohorts --run_dir "$RUN_DIR" --task "$POISONING_TASK"

case "$POISONING_TASK" in
  grammar)
    HELDOUT="$RUN_DIR/02_evaluation_cohorts/grammar_causal_validation.jsonl"
    [[ -f "$HELDOUT" ]] || { echo "Missing adaptive causal cohort: $HELDOUT" >&2; exit 1; }
    export GRAMMAR_BACKDOOR_DATASET_PATH="$HELDOUT"
    export GRAMMAR_BACKDOOR_TARGET_LABEL="${GRAMMAR_BACKDOOR_TARGET_LABEL:-$RUNCFG_TARGET}"
    export GRAMMAR_BACKDOOR_SOURCE_FILTER="${GRAMMAR_BACKDOOR_SOURCE_FILTER:-non_target}"
    export GRAMMAR_BACKDOOR_NUM_EXAMPLES="${GRAMMAR_BACKDOOR_NUM_EXAMPLES:-0}"
    TASK_MODULE="studies.poisoning.tasks.grammar:BACKDOOR_TASK_SPEC"
    ;;
  arithmetic)
    HELDOUT="$RUN_DIR/02_evaluation_cohorts/arithmetic_causal_validation.jsonl"
    [[ -f "$HELDOUT" ]] || { echo "Missing adaptive causal cohort: $HELDOUT" >&2; exit 1; }
    export ARITHMETIC_BACKDOOR_DATASET_PATH="$HELDOUT"
    export ARITHMETIC_BACKDOOR_TARGET_ANSWER="${ARITHMETIC_BACKDOOR_TARGET_ANSWER:-$RUNCFG_TARGET}"
    export ARITHMETIC_BACKDOOR_SOURCE_FILTER="${ARITHMETIC_BACKDOOR_SOURCE_FILTER:-non_target}"
    export ARITHMETIC_BACKDOOR_NUM_EXAMPLES="${ARITHMETIC_BACKDOOR_NUM_EXAMPLES:-0}"
    TASK_MODULE="studies.poisoning.tasks.arithmetic:BACKDOOR_TASK_SPEC"
    ;;
  *) echo "POISONING_TASK must be grammar or arithmetic" >&2; exit 2 ;;
esac

CAUSAL_POOL_ROWS="$(awk 'NF{n++} END{print n+0}' "$HELDOUT")"
echo "[causal-pool] discovery candidate universe: $CAUSAL_POOL_ROWS rows from $HELDOUT"
if (( CAUSAL_POOL_ROWS < 256 )); then
  echo "ERROR: adaptive causal pool is unexpectedly small ($CAUSAL_POOL_ROWS rows)." >&2
  echo "ERROR: refusing to fall back to the ordinary checkpoint-evaluation cohort." >&2
  exit 1
fi

# Primary task phases: grammar includes prompt processing; arithmetic targets
# answer generation.  An explicit PIPELINE_DECODE_ONLY always overrides this.
if [[ -z "${PIPELINE_DECODE_ONLY+x}" ]]; then
  if [[ "$POISONING_TASK" == "arithmetic" ]]; then PIPELINE_DECODE_ONLY=1; else PIPELINE_DECODE_ONLY=0; fi
fi
if [[ "$PIPELINE_DECODE_ONLY" == "1" || "$PIPELINE_DECODE_ONLY" == "true" ]]; then
  PHASE_LABEL="output_only"
else
  PHASE_LABEL="input_output"
fi

EVAL_INTERVENTION="${PIPELINE_EVAL_INTERVENTION:-mean-donor}"
BATCH_SIZE="${PIPELINE_BATCH_SIZE:-1}"
CIRCUIT_SIZE="${POISONING_CIRCUIT_SIZE:-5000}"
MIN_FLIP_RATE="${CHA_TAU:-0.3}"
CHA_PRUNE_ALPHA="${CHA_PRUNE_ALPHA:-0.05}"
EVAL_CONFIDENCE_ALPHA="${POISONING_EVAL_CONFIDENCE_ALPHA:-0.05}"
# The reference n and reference tau jointly define the CHA operating point.
REFERENCE_CHA_SIDE="${CHA_REFERENCE_N_PER_SIDE:-64}"
# By default the actual-sample cap follows the reference.  This matters when a
# user changes the reference from 64 to (for example) 128: the run can then
# genuinely acquire/use 128 examples per side without a second hidden knob.
MAX_DISCOVERY_SIDE="${CHA_MAX_N_PER_SIDE:-$REFERENCE_CHA_SIDE}"
MIN_ACTUAL_CHA_SIDE="${CHA_MIN_ACTUAL_N_PER_SIDE:-16}"
LOW_DATA_POLICY="${CHA_LOW_DATA_POLICY:-skip}"
MIN_DISCOVERY_POSITIVES="${POISONING_MIN_DISCOVERY_POSITIVES:-2}"
MAX_DISCOVERY_PAIRS="${POISONING_MAX_DISCOVERY_PAIRS:-128}"
TEST_FRACTION="${POISONING_HOLDOUT_TEST_FRACTION:-0.3333333333333333}"
INCLUDE_FRACTION_ZERO="${POISONING_INCLUDE_FRACTION_ZERO:-0}"
LIFT_INDICES="${LIFT_INDICES:-all}"
PAIR_CHECKPOINT_CONDITIONS="${PAIR_CHECKPOINT_CONDITIONS:-1}"
HF_MODEL_CACHE_DIR="${HF_MODEL_CACHE_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"
REPORT_ALL_POINTS_WHEN_HELDOUT_BELOW_TARGET="${POISONING_REPORT_ALL_POINTS_WHEN_HELDOUT_BELOW_TARGET:-1}"
RUN_NORMAL_TASK_CONTROL="${RUN_NORMAL_TASK_CONTROL:-${RUN_ORDINARY_CORRECTNESS_CONTROL:-1}}"
RUN_TRIGGER_LIFT="${RUN_TRIGGER_LIFT:-1}"
# Trigger-lift behavior remains enabled independently; this controls only the
# expensive trigger-lift-conditioned CHA/circuit-discovery path.
RUN_TRIGGER_LIFT_CHA="${RUN_TRIGGER_LIFT_CHA:-0}"
RUN_NORMAL_TASK_OVERTOPPING="${RUN_NORMAL_TASK_OVERTOPPING:-${RUN_ORDINARY_CORRECTNESS_OVERTOPPING:-1}}"
RUN_BEHAVIOR_COMPARISON="${RUN_BEHAVIOR_COMPARISON:-1}"
RUN_BEHAVIOR_VISUALIZATIONS="${RUN_BEHAVIOR_VISUALIZATIONS:-1}"
BEHAVIOR_ONLY="${POISONING_BEHAVIOR_ONLY:-0}"
FAST_MIN_TRIGGER_EXCESS="${POISONING_FAST_MIN_TRIGGER_EXCESS:-}"
FAST_MIN_CONDITIONAL_CONVERSION="${POISONING_FAST_MIN_CONDITIONAL_CONVERSION:-}"
FAST_MAX_ABS_CONTROL_DELTA="${POISONING_FAST_MAX_ABS_CONTROL_DELTA:-}"
CAUSAL_SCAN_MAX_ROWS="${TRIGGER_LIFT_SCAN_MAX_ROWS:-10000}"
STAGE7_MAX_ROWS="${REFINE_SAMPLING_MAX_POINTS:-10000}"

case "$LOW_DATA_POLICY" in
  adapt|skip|fail) ;;
  *)
    echo "ERROR: CHA_LOW_DATA_POLICY must be one of: adapt, skip, fail (got '$LOW_DATA_POLICY')." >&2
    exit 2
    ;;
esac
python3 - "$REFERENCE_CHA_SIDE" "$MAX_DISCOVERY_SIDE" "$MIN_ACTUAL_CHA_SIDE" "$MIN_FLIP_RATE" "$CHA_PRUNE_ALPHA" "$CAUSAL_SCAN_MAX_ROWS" "$STAGE7_MAX_ROWS" <<'PYCFG'
import sys
ref, cap, minimum = map(int, sys.argv[1:4])
tau, alpha = map(float, sys.argv[4:6])
scan_cap, all_cap = map(int, sys.argv[6:8])
if ref < 1:
    raise SystemExit("CHA_REFERENCE_N_PER_SIDE must be >= 1")
if cap < 1:
    raise SystemExit("CHA_MAX_N_PER_SIDE must be >= 1")
if minimum < 1:
    raise SystemExit("CHA_MIN_ACTUAL_N_PER_SIDE must be >= 1")
if not (0.0 <= tau <= 1.0):
    raise SystemExit("CHA_TAU must be in [0,1]")
if not (0.0 < alpha < 1.0):
    raise SystemExit("CHA_PRUNE_ALPHA must be in (0,1)")
if scan_cap < 0:
    raise SystemExit("TRIGGER_LIFT_SCAN_MAX_ROWS must be >= 0; 0 means unlimited")
if all_cap < 0:
    raise SystemExit("REFINE_SAMPLING_MAX_POINTS must be >= 0; 0 means unlimited when spectral sampling is disabled")
PYCFG
if [[ "$LOW_DATA_POLICY" == "adapt" ]] && (( MAX_DISCOVERY_SIDE < MIN_ACTUAL_CHA_SIDE )); then
  echo "ERROR: CHA_MAX_N_PER_SIDE=$MAX_DISCOVERY_SIDE is below CHA_MIN_ACTUAL_N_PER_SIDE=$MIN_ACTUAL_CHA_SIDE." >&2
  exit 2
fi
if [[ "$LOW_DATA_POLICY" != "adapt" ]] && (( MAX_DISCOVERY_SIDE < REFERENCE_CHA_SIDE )); then
  echo "ERROR: CHA_LOW_DATA_POLICY=$LOW_DATA_POLICY requires the cap to permit the full reference sample, but CHA_MAX_N_PER_SIDE=$MAX_DISCOVERY_SIDE < CHA_REFERENCE_N_PER_SIDE=$REFERENCE_CHA_SIDE." >&2
  exit 2
fi
echo "[cha-config] reference_n_per_side=$REFERENCE_CHA_SIDE reference_tau=$MIN_FLIP_RATE max_side=$MAX_DISCOVERY_SIDE min_actual_side=$MIN_ACTUAL_CHA_SIDE low_data_policy=$LOW_DATA_POLICY prune_alpha=$CHA_PRUNE_ALPHA"
echo "[data-config] causal_scan_max_rows=$CAUSAL_SCAN_MAX_ROWS causal_scan_order=deterministic_seeded_source_order early_stop=${TRIGGER_LIFT_SCAN_EARLY_STOP:-0} stage7_sampling_max_points=$STAGE7_MAX_ROWS"

export POISONING_HOLDOUT_SEED="${POISONING_HOLDOUT_SEED:-$RUNCFG_SEED}"
export POISONING_HOLDOUT_TEST_FRACTION="$TEST_FRACTION"
export CHA_REFERENCE_N_PER_SIDE="$REFERENCE_CHA_SIDE"
export CHA_TAU="$MIN_FLIP_RATE"
export CHA_MIN_ACTUAL_N_PER_SIDE="$MIN_ACTUAL_CHA_SIDE"
export CHA_LOW_DATA_POLICY="$LOW_DATA_POLICY"
export CHA_MAX_N_PER_SIDE="$MAX_DISCOVERY_SIDE"
export POISONING_TARGET_DISCOVERY_POSITIVES="${POISONING_TARGET_DISCOVERY_POSITIVES:-$((2 * REFERENCE_CHA_SIDE))}"
export POISONING_MIN_DISCOVERY_POSITIVES="$MIN_DISCOVERY_POSITIVES"
export POISONING_TARGET_TEST_POSITIVES="${POISONING_TARGET_TEST_POSITIVES:-32}"
export TRIGGER_LIFT_SCAN_CHUNK="${TRIGGER_LIFT_SCAN_CHUNK:-2048}"
export TRIGGER_LIFT_SCAN_MIN_ROWS="${TRIGGER_LIFT_SCAN_MIN_ROWS:-0}"
export TRIGGER_LIFT_SCAN_MAX_ROWS="$CAUSAL_SCAN_MAX_ROWS"
export REFINE_SAMPLING_MAX_POINTS="$STAGE7_MAX_ROWS"
echo "[cha-config] discovery_positive_target=$POISONING_TARGET_DISCOVERY_POSITIVES heldout_positive_target=$POISONING_TARGET_TEST_POSITIVES"
if (( MAX_DISCOVERY_SIDE != REFERENCE_CHA_SIDE )); then
  echo "[cha-config] explicit max-side override active: max_side=$MAX_DISCOVERY_SIDE reference_side=$REFERENCE_CHA_SIDE"
fi
if (( POISONING_TARGET_DISCOVERY_POSITIVES != 2 * REFERENCE_CHA_SIDE )); then
  echo "[cha-config] explicit discovery-target override active: target=$POISONING_TARGET_DISCOVERY_POSITIVES default_for_reference=$((2 * REFERENCE_CHA_SIDE))"
fi

POISONING_CACHE_ROOT="${POISONING_CACHE_ROOT:-$PROJECT_ROOT/cache/poisoning}"
RUN_ID="$(basename "$RUN_DIR")"
TASK_ID="${POISONING_TASK:-poisoning}"
case "$PHASE_LABEL" in
  input_output) PHASE_DIR_LABEL="prompt_and_generation" ;;
  output_only) PHASE_DIR_LABEL="generation_only" ;;
  *) echo "Unknown PHASE_LABEL=$PHASE_LABEL" >&2; exit 2 ;;
esac
DISCOVERY_CACHE_ROOT="${DISCOVERY_CACHE_ROOT:-$POISONING_CACHE_ROOT/$TASK_ID/$RUN_ID/checkpoint_causal_discovery/$PHASE_DIR_LABEL/adaptive_circuit_discovery}"

mkdir -p "$DISCOVERY_CACHE_ROOT"
echo "[cache-root] $DISCOVERY_CACHE_ROOT"
echo "[run-root] $RUN_DIR"
echo "[causal-root] $RUN_DIR/03_checkpoint_causal_discovery"

if poisoning_is_true "$PAIR_CHECKPOINT_CONDITIONS"; then
  CHECKPOINT_ORDER_MODE="paired_by_fraction_step"
else
  CHECKPOINT_ORDER_MODE="manifest"
fi
INDEX_LIST="$(python3 - "$RUN_MANIFEST_PATH" "$LIFT_INDICES" "$PAIR_CHECKPOINT_CONDITIONS" <<'PY'
import csv, sys
from studies.poisoning.lib.trajectory import select_manifest_indices

path, spec, pair_raw = sys.argv[1:4]
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
pair = pair_raw.strip().lower() in {"1", "true", "yes", "on"}
try:
    indices = select_manifest_indices(rows, spec, pair_conditions=pair)
except ValueError as exc:
    raise SystemExit(str(exc)) from exc
print(" ".join(str(i) for i in indices))
PY
)"
INDEX_COUNT="$(printf '%s\n' "$INDEX_LIST" | awk '{print NF}')"
echo "[checkpoint-order] mode=$CHECKPOINT_ORDER_MODE selected=$INDEX_COUNT selector=$LIFT_INDICES"

for INDEX in $INDEX_LIST; do
  ENV_FILE="$(mktemp)"
  python3 - "$RUN_MANIFEST_PATH" "$RUN_DIR" "$INDEX" "$PHASE_LABEL" "$EVAL_INTERVENTION" > "$ENV_FILE" <<'PY'
import csv, pathlib, re, shlex, sys
from studies.poisoning.lib.run_paths import BACKDOOR_TRIGGER_TEST_DIRNAME, checkpoint_cache_key, checkpoint_progress_label, checkpoint_tag, model_variant_label, phase_dirname, resolve_manifest_checkpoint_dir
manifest_path = pathlib.Path(sys.argv[1]); run_stage_root = pathlib.Path(sys.argv[2]); index = int(sys.argv[3]); phase, intervention = sys.argv[4:6]
run_dir = run_stage_root
with manifest_path.open(newline="", encoding="utf-8") as f:
    row = list(csv.DictReader(f))[index]
def sanitize(s):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)); return re.sub(r"_+", "_", s).strip("_") or "run"
def emit(k, v): print(f"{k}={shlex.quote(str(v))}")
tag = checkpoint_tag(row)
stage_label = checkpoint_progress_label(row)
checkpoint_key = checkpoint_cache_key(row)
out = run_stage_root / "03_checkpoint_causal_discovery" / row["condition"] / stage_label / phase_dirname(phase) / BACKDOOR_TRIGGER_TEST_DIRNAME / f"eval_{sanitize(intervention)}"
checkpoint_dir = resolve_manifest_checkpoint_dir(run_dir, row, must_exist=True)
emit("CHECKPOINT_DIR", checkpoint_dir); emit("OUTPUT_DATA_DIR", out)
emit("CONDITION", row["condition"]); emit("MODEL_VARIANT_LABEL", model_variant_label(row["condition"])); emit("FRACTION", row["fraction"]); emit("GLOBAL_STEP", row["global_step"])
emit("CHECKPOINT_CACHE_KEY", checkpoint_key); emit("CHECKPOINT_TAG", tag); emit("CHECKPOINT_STAGE_LABEL", stage_label)
PY
  # shellcheck disable=SC1090
  source "$ENV_FILE"; rm -f "$ENV_FILE"

  echo "=== $POISONING_TASK checkpoint: index=$INDEX model_variant=$MODEL_VARIANT_LABEL fraction=$FRACTION step=$GLOBAL_STEP phase=$PHASE_LABEL ==="

  SKIP_TRIGGER_CAUSAL=0
  if python3 - "$FRACTION" "$INCLUDE_FRACTION_ZERO" <<'PY'
import sys
frac=float(sys.argv[1]); include=str(sys.argv[2]).lower() in {'1','true','yes'}
raise SystemExit(0 if (include or frac > 1e-12) else 1)
PY
  then :; else
    # Normal-task correctness analysis is still required at fraction 0 so Stage 07
    # has a pre-training reference. Only backdoor-trigger causal discovery is skipped.
    SKIP_TRIGGER_CAUSAL=1
  fi

  CHECKPOINT_CACHE_DIR="$DISCOVERY_CACHE_ROOT/backdoor_trigger_test/$CHECKPOINT_CACHE_KEY"
  LLM_IO="$CHECKPOINT_CACHE_DIR/llm_io_data.pkl"
  FEATURES_DIR="$OUTPUT_DATA_DIR/feature_report"

  if poisoning_is_true "$RUN_TRIGGER_LIFT"; then
    mkdir -p "$CHECKPOINT_CACHE_DIR" "$FEATURES_DIR"
    STAGE1=(python3 -m pipeline.stage01_generate_prompts_and_answers
      --ai_model "$CHECKPOINT_DIR"
      --task_module "$TASK_MODULE"
      --prompts_answers_pkl_file "$LLM_IO"
      --batch_size "$BATCH_SIZE"
      --stats_json_out "$FEATURES_DIR"
      --export_dataset_scores_dir "$FEATURES_DIR")
    if [[ -n "$HF_MODEL_CACHE_DIR" ]]; then STAGE1+=(--ai_model_cache_dir "$HF_MODEL_CACHE_DIR"); fi
    printf '[cmd]'; printf ' %q' "${STAGE1[@]}"; printf '\n'
    if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" ]]; then
      "${STAGE1[@]}"
    fi

    if poisoning_is_true "$RUN_BEHAVIOR_COMPARISON" && [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" ]]; then
      EARLY_COMPARE=(python3 -m studies.poisoning.stage04_compare_condition_behavior
        --run_dir "$RUN_DIR"
        --condition "$CONDITION"
        --checkpoint_tag "$CHECKPOINT_TAG"
        --checkpoint_label "$CHECKPOINT_STAGE_LABEL"
        --phase "$PHASE_LABEL"
        --eval_intervention "$EVAL_INTERVENTION"
        --kind backdoor_trigger_test
        --scores_csv "$FEATURES_DIR/scores.csv")
      if ! poisoning_is_true "$RUN_BEHAVIOR_VISUALIZATIONS"; then EARLY_COMPARE+=(--no_plots); fi
      if [[ -n "$FAST_MIN_TRIGGER_EXCESS" ]]; then EARLY_COMPARE+=(--min_trigger_excess "$FAST_MIN_TRIGGER_EXCESS"); fi
      if [[ -n "$FAST_MIN_CONDITIONAL_CONVERSION" ]]; then EARLY_COMPARE+=(--min_conditional_conversion "$FAST_MIN_CONDITIONAL_CONVERSION"); fi
      if [[ -n "$FAST_MAX_ABS_CONTROL_DELTA" ]]; then EARLY_COMPARE+=(--max_abs_control_delta "$FAST_MAX_ABS_CONTROL_DELTA"); fi
      if [[ -n "$FAST_MIN_TRIGGER_EXCESS$FAST_MIN_CONDITIONAL_CONVERSION$FAST_MAX_ABS_CONTROL_DELTA" ]]; then EARLY_COMPARE+=(--fail_on_gate); fi
      "${EARLY_COMPARE[@]}"
    fi
  else
    SKIP_TRIGGER_CAUSAL=1
    echo "[preserve-trigger] RUN_TRIGGER_LIFT=0: leaving existing backdoor_trigger_test Stage 03/04 outputs untouched for $CONDITION $CHECKPOINT_TAG"
  fi


  if poisoning_is_true "$RUN_TRIGGER_LIFT" && ! poisoning_is_true "$RUN_TRIGGER_LIFT_CHA"; then
    echo "[backdoor-trigger-test] behavior=ENABLED; causal_CHA=DISABLED (RUN_TRIGGER_LIFT_CHA=0)."
    echo "[backdoor-trigger-test] the upcoming long CHA belongs ONLY to attack-cohort control correctness."
  fi

  # Keep two endpoints separate:
  #   (a) normal-task BEHAVIOR on the full held-out distribution;
  #   (b) attack-cohort control-correctness CHA on the exact non-target cohort
  #       from the paired backdoor test (the validated Aug-26 causal estimand).
  # Exact no-trigger outputs are reused where possible, but the CHA population
  # is never expanded to the full held-out distribution.
  if [[ "$RUN_NORMAL_TASK_CONTROL" == "1" || "$RUN_NORMAL_TASK_CONTROL" == "true" ]]; then
    env \
      PROJECT_ROOT="$PROJECT_ROOT" CODE_DIR="$CODE_ROOT" POISONING_TASK="$POISONING_TASK" RUN_DIR="$RUN_DIR" \
      CHECKPOINT_DIR="$CHECKPOINT_DIR" BACKDOOR_OUTPUT_DATA_DIR="$OUTPUT_DATA_DIR" \
      CONDITION="$CONDITION" CHECKPOINT_TAG="$CHECKPOINT_TAG" CHECKPOINT_STAGE_LABEL="$CHECKPOINT_STAGE_LABEL" \
      DISCOVERY_CACHE_ROOT="$DISCOVERY_CACHE_ROOT" REUSE_CONTROL_CACHE="$LLM_IO" \
      CHECKPOINT_CACHE_KEY="$CHECKPOINT_CACHE_KEY" PHASE_LABEL="$PHASE_LABEL" EVAL_INTERVENTION="$EVAL_INTERVENTION" \
      BATCH_SIZE="$BATCH_SIZE" REFERENCE_CHA_SIDE="$REFERENCE_CHA_SIDE" \
      MAX_DISCOVERY_SIDE="$MAX_DISCOVERY_SIDE" MIN_ACTUAL_CHA_SIDE="$MIN_ACTUAL_CHA_SIDE" \
      LOW_DATA_POLICY="$LOW_DATA_POLICY" MIN_FLIP_RATE="$MIN_FLIP_RATE" \
      CHA_PRUNE_ALPHA="$CHA_PRUNE_ALPHA" MAX_DISCOVERY_PAIRS="$MAX_DISCOVERY_PAIRS" \
      CIRCUIT_SIZE="$CIRCUIT_SIZE" STAGE7_MAX_ROWS="$STAGE7_MAX_ROWS" \
      EVAL_CONFIDENCE_ALPHA="$EVAL_CONFIDENCE_ALPHA" HF_MODEL_CACHE_DIR="$HF_MODEL_CACHE_DIR" \
      DRY_RUN="$DRY_RUN" RUN_NORMAL_TASK_OVERTOPPING="$RUN_NORMAL_TASK_OVERTOPPING" \
      RUN_BEHAVIOR_COMPARISON="$RUN_BEHAVIOR_COMPARISON" RUN_BEHAVIOR_VISUALIZATIONS="$RUN_BEHAVIOR_VISUALIZATIONS" \
      bash "$SCRIPT_DIR/run_normal_task_correctness_control.sh"
  fi


  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
    continue
  fi

  # Trigger-lift is retained as a behavioral endpoint, but its CHA/circuit
  # decomposition is optional.  When disabled, write an explicit status so
  # downstream aggregation keeps the behavioral trajectory while suppressing
  # both newly generated and stale trigger-lift causal artifacts.
  if poisoning_is_true "$RUN_TRIGGER_LIFT" && ! poisoning_is_true "$RUN_TRIGGER_LIFT_CHA"; then
    python3 - "$FEATURES_DIR/scores.csv" "$OUTPUT_DATA_DIR/discovery_status.json" "$CONDITION" "$FRACTION" "$GLOBAL_STEP" "$PHASE_LABEL" "$REFERENCE_CHA_SIDE" "$MAX_DISCOVERY_SIDE" "$MIN_ACTUAL_CHA_SIDE" "$MIN_FLIP_RATE" "$CHA_PRUNE_ALPHA" "$POISONING_TARGET_DISCOVERY_POSITIVES" "$POISONING_TARGET_TEST_POSITIVES" "$CAUSAL_SCAN_MAX_ROWS" "$STAGE7_MAX_ROWS" <<'PYDISABLED'
import json, pathlib, sys
import pandas as pd

(
    scores_path, status_path, condition, fraction, step, phase, ref_side, max_side,
    min_actual_side, base_tau, prune_alpha, target_train, target_test, scan_cap,
    stage7_cap,
) = sys.argv[1:]

df = pd.read_csv(scores_path)
def as_bool(v):
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}
is_test = df["is_test"].map(as_bool) if "is_test" in df else pd.Series(False, index=df.index)
pos = df["is_trigger_lift_success"].map(as_bool)
train_pos = int((pos & ~is_test).sum())
test_pos = int((pos & is_test).sum())
total_pos = int(pos.sum())

payload = {
    "condition": condition,
    "fraction": float(fraction),
    "global_step": int(float(step)),
    "phase": phase,
    "behavior_reference": "transformerlens_checkpoint",
    "n_causal_rows_scanned": int(len(df)),
    "causal_scan_max_rows": int(scan_cap) if int(scan_cap) > 0 else None,
    "causal_scan_sampling": "deterministic_seeded_source_order",
    "n_trigger_lift_total": total_pos,
    "n_trigger_lift_discovery": train_pos,
    "n_trigger_lift_test": test_pos,
    "available_balanced_cha_points_per_side": None,
    "n_associated": 0,
    "n_unrelated": 0,
    "reference_cha_points_per_side": int(ref_side),
    "max_discovery_points_per_side": int(max_side),
    "minimum_actual_cha_points_per_side": int(min_actual_side),
    "reference_discovery_trigger_lift_positives": int(target_train),
    "cha_base_tau_at_reference_n": float(base_tau),
    "cha_effective_tau": None,
    "cha_prune_alpha": float(prune_alpha),
    "cha_slice_alpha": float(prune_alpha) / 2.0,
    "cha_tau_adjusted_for_sample_size": False,
    "cha_reference_target_met": None,
    "low_data_policy": None,
    "analysis_decision": "skip_trigger_lift_cha_disabled",
    "low_data_reason": "trigger_lift_cha_disabled",
    "trigger_lift_cha_enabled": False,
    "cha_will_run": False,
    "target_heldout_trigger_lift_positives": int(target_test),
    "heldout_target_met": bool(test_pos >= int(target_test)),
    "report_all_points_when_heldout_below_target": False,
    "all_points_requested": False,
    "stage7_sampling_max_points": int(stage7_cap) if int(stage7_cap) > 0 else None,
    "all_points_max_rows": int(stage7_cap) if int(stage7_cap) > 0 else None,
    "all_points_cap_sampling": None,
    "primary_final_statistics_split": None,
    "secondary_final_statistics_split": None,
    "status": "skipped_trigger_lift_cha_disabled",
}
path = pathlib.Path(status_path)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PYDISABLED
    echo "[skip-trigger-lift-cha] RUN_TRIGGER_LIFT_CHA=0: retained trigger-lift behavior for $CONDITION $CHECKPOINT_TAG; skipped trigger-lift CHA/circuit discovery."
    continue
  fi

  if [[ "$SKIP_TRIGGER_CAUSAL" == "1" ]]; then
    if poisoning_is_true "$RUN_TRIGGER_LIFT"; then
      echo "[skip-trigger-lift] fraction=0 is retained as the ordinary-correctness pre-training reference; trigger-lift developmental discovery starts after training has begun."
    fi
    continue
  fi

  if poisoning_is_true "$BEHAVIOR_ONLY"; then
    echo "[behavior-only] skipping CHA/circuit analysis for $CONDITION $CHECKPOINT_TAG"
    continue
  fi

  COUNTS="$(python3 - "$FEATURES_DIR/scores.csv" "$MAX_DISCOVERY_SIDE" "$REFERENCE_CHA_SIDE" "$MAX_DISCOVERY_PAIRS" "$MIN_ACTUAL_CHA_SIDE" "$MIN_FLIP_RATE" "$CHA_PRUNE_ALPHA" "$POISONING_TARGET_DISCOVERY_POSITIVES" "$LOW_DATA_POLICY" <<'PYCOUNTS'
import pandas as pd, shlex, sys
from studies.poisoning.lib.cha import plan_cha
(
    p, cap_side, ref_side, cap_pairs, min_actual_side, base_tau,
    prune_alpha, target_train, low_data_policy,
) = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
    float(sys.argv[6]), float(sys.argv[7]), int(sys.argv[8]), sys.argv[9].strip().lower(),
)
df = pd.read_csv(p)
def b(v): return str(v).strip().lower() in {'1','true','t','yes','y'}
is_test = df['is_test'].map(b) if 'is_test' in df else pd.Series(False, index=df.index)
pos = df['is_trigger_lift_success'].map(b)
train_pos = int((pos & ~is_test).sum()); test_pos = int((pos & is_test).sum()); total_pos = int(pos.sum())
plan = plan_cha(
    train_pos,
    reference_side=ref_side,
    reference_tau=base_tau,
    max_side=cap_side,
    min_actual_side=min_actual_side,
    low_data_policy=low_data_policy,
    prune_alpha=prune_alpha,
    max_pairs=cap_pairs,
)
for k,v in [
    ('TRAIN_POS',train_pos),('TEST_POS',test_pos),('TOTAL_POS',total_pos),
    ('AVAILABLE_SIDE',plan['available_side']),('N_SIDE',plan['n_side']),('N_PAIRS',plan['n_pairs']),('N_ROWS',len(df)),
    ('REFERENCE_CHA_SIDE',ref_side),('TARGET_TRAIN_POS',target_train),
    ('MIN_ACTUAL_CHA_SIDE',min_actual_side),('EFFECTIVE_CHA_TAU',plan['effective_tau']),
    ('ANALYSIS_DECISION',plan['analysis_decision']),('LOW_DATA_REASON',plan['low_data_reason']),
    ('LOW_DATA_POLICY',plan['low_data_policy']),
]:
    print(f"{k}={shlex.quote(str(v))}")
PYCOUNTS
)"
  eval "$COUNTS"
  echo "[tl-behavior] rows=$N_ROWS trigger_lift=$TOTAL_POS discovery_lift=$TRAIN_POS heldout_lift=$TEST_POS"

  python3 - "$OUTPUT_DATA_DIR/discovery_status.json" "$CONDITION" "$FRACTION" "$GLOBAL_STEP" "$PHASE_LABEL" "$TOTAL_POS" "$TRAIN_POS" "$TEST_POS" "$N_SIDE" "$REFERENCE_CHA_SIDE" "$TARGET_TRAIN_POS" "$POISONING_TARGET_TEST_POSITIVES" "$REPORT_ALL_POINTS_WHEN_HELDOUT_BELOW_TARGET" "$MIN_FLIP_RATE" "$EFFECTIVE_CHA_TAU" "$CHA_PRUNE_ALPHA" "$LOW_DATA_POLICY" "$MIN_ACTUAL_CHA_SIDE" "$MAX_DISCOVERY_SIDE" "$ANALYSIS_DECISION" "$LOW_DATA_REASON" "$N_ROWS" "$CAUSAL_SCAN_MAX_ROWS" "$STAGE7_MAX_ROWS" <<'PYSTATUS'
import json, math, os, pathlib, sys
p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)
condition, fraction, step, phase = sys.argv[2:6]
total, train, test, side, ref_side, target_train, test_target = map(int, sys.argv[6:13])
report_all = str(sys.argv[13]).strip().lower() not in {'0','false','no','off'}
base_tau, effective_tau, prune_alpha = map(float, sys.argv[14:17])
low_data_policy = sys.argv[17].strip().lower()
min_actual_side, max_side = map(int, sys.argv[18:20])
decision, low_data_reason = sys.argv[20:22]
n_rows, scan_cap, all_points_cap = map(int, sys.argv[22:25])
full_reference = side >= ref_side
if decision == 'run_reference':
    status = 'ready_reference_power' if test >= test_target else 'ready_reference_power_heldout_below_target'
elif decision == 'run_adaptive':
    status = 'ready_adaptive_epsilon' if test >= test_target else 'ready_adaptive_epsilon_heldout_below_target'
elif decision == 'skip_below_min_actual':
    if train == 0:
        status = 'skipped_no_trigger_lift'
    elif train < 2:
        status = 'skipped_insufficient_trigger_lift_for_two_sides'
    else:
        status = 'skipped_low_data_below_min_actual_side'
elif decision == 'skip_reference_shortfall':
    if train == 0:
        status = 'skipped_no_trigger_lift'
    else:
        status = 'skipped_low_data_reference_shortfall'
elif decision == 'fail_reference_shortfall':
    status = 'failed_low_data_reference_shortfall'
else:
    status = f'unknown_decision_{decision}'
secondary = 'all' if (report_all and test < test_target and decision.startswith('run_')) else None
payload = {
  'condition': condition, 'fraction': float(fraction), 'global_step': int(float(step)), 'phase': phase,
  'behavior_reference': 'transformerlens_checkpoint', 'n_causal_rows_scanned': n_rows,
  'causal_scan_max_rows': scan_cap if scan_cap > 0 else None,
  'causal_scan_sampling': 'deterministic_seeded_source_order',
  'causal_scan_early_stop': str(os.environ.get('TRIGGER_LIFT_SCAN_EARLY_STOP', '0')).lower() in {'1','true','yes','on'},
  'causal_scan_cap_reached': bool(scan_cap > 0 and n_rows >= scan_cap),
  'n_trigger_lift_total': total,
  'n_trigger_lift_discovery': train, 'n_trigger_lift_test': test,
  'available_balanced_cha_points_per_side': side,
  'n_associated': side if decision.startswith('run_') else 0,
  'n_unrelated': side if decision.startswith('run_') else 0,
  'reference_cha_points_per_side': ref_side,
  'max_discovery_points_per_side': max_side,
  'minimum_actual_cha_points_per_side': min_actual_side,
  'reference_discovery_trigger_lift_positives': target_train,
  'cha_base_tau_at_reference_n': base_tau,
  'cha_effective_tau': effective_tau if math.isfinite(effective_tau) else None,
  'cha_prune_alpha': prune_alpha,
  'cha_slice_alpha': prune_alpha / 2.0,
  'cha_tau_adjusted_for_sample_size': bool(decision == 'run_adaptive' and side < ref_side),
  'cha_reference_target_met': bool(full_reference),
  'low_data_policy': low_data_policy,
  'analysis_decision': decision,
  'low_data_reason': low_data_reason,
  'cha_will_run': bool(decision.startswith('run_')),
  'target_heldout_trigger_lift_positives': test_target,
  'heldout_target_met': bool(test >= test_target),
  'report_all_points_when_heldout_below_target': bool(report_all),
  'all_points_requested': bool(secondary is not None),
  'stage7_sampling_max_points': all_points_cap if all_points_cap > 0 else None,
  'all_points_max_rows': all_points_cap if all_points_cap > 0 else None,
  'all_points_cap_sampling': 'seeded_uniform_without_replacement' if all_points_cap > 0 else None,
  'primary_final_statistics_split': 'test' if decision.startswith('run_') else None,
  'secondary_final_statistics_split': secondary,
  'status': status,
}
p.write_text(json.dumps(payload, indent=2), encoding='utf-8')
PYSTATUS


  case "$ANALYSIS_DECISION" in
    skip_below_min_actual)
      echo "[skip-low-data] discovery trigger-lift positives=$TRAIN_POS give only $AVAILABLE_SIDE balanced points/side; adaptive floor=$MIN_ACTUAL_CHA_SIDE, reference=$REFERENCE_CHA_SIDE."
      echo "[skip-low-data] checkpoint behavior is retained in the trajectory, but CHA is not run. Set CHA_MIN_ACTUAL_N_PER_SIDE lower to permit a smaller adaptive analysis, or CHA_LOW_DATA_POLICY=fail to make this fatal."
      continue
      ;;
    skip_reference_shortfall)
      echo "[skip-low-data] CHA_LOW_DATA_POLICY=skip: available=$AVAILABLE_SIDE/side is below reference=$REFERENCE_CHA_SIDE/side. Behavior statistics are retained; CHA is omitted for this checkpoint."
      continue
      ;;
    fail_reference_shortfall)
      echo "ERROR: CHA_LOW_DATA_POLICY=fail and only $AVAILABLE_SIDE balanced discovery points/side are available; reference=$REFERENCE_CHA_SIDE." >&2
      echo "ERROR: trigger_lift discovery positives=$TRAIN_POS; checkpoint status was written before aborting." >&2
      exit 1
      ;;
    run_adaptive)
      echo "[discovery] finite corpus below reference target: discovery_lift=$TRAIN_POS, using n_associated=n_unrelated=$N_SIDE (reference=$REFERENCE_CHA_SIDE)."
      echo "[discovery] CHA UCB cutoff adjusted from tau=$MIN_FLIP_RATE at n=$REFERENCE_CHA_SIDE to effective_tau=$EFFECTIVE_CHA_TAU at n=$N_SIDE per side."
      ;;
    run_reference)
      echo "[discovery] CHA uses reference n_associated=n_unrelated=$N_SIDE with tau=$MIN_FLIP_RATE."
      ;;
    *)
      echo "ERROR: internal low-data decision '$ANALYSIS_DECISION' is not recognized." >&2
      exit 2
      ;;
  esac
  echo "[discovery] EAP pair cap=$N_PAIRS."

  CMD=(bash pipeline/run_pipeline.sh
    "${POISONING_TASK}_backdoor_lift"
    "$CHECKPOINT_DIR"
    --task_module "$TASK_MODULE"
    --output_data_dir "$OUTPUT_DATA_DIR"
    --pipeline_cache_root "$DISCOVERY_CACHE_ROOT"
    --pipeline_model_cache_dir "$CHECKPOINT_CACHE_DIR"
    --model_label "$CHECKPOINT_CACHE_KEY"
    --spectral_splits
    --fast_anchoring
    --z_thresh -1
    --batch_size "$BATCH_SIZE"
    --circuit_level neuron
    --circuit_size "$CIRCUIT_SIZE"
    --eval_intervention "$EVAL_INTERVENTION"
    --min_flip_rate "$MIN_FLIP_RATE"
    --max_number_of_circuits_to_analyze 1
    --evaluation_split test
    --no_llm_feature_generation)
  CMD+=(--evaluation_baseline_subset positive)
  if [[ "$PHASE_LABEL" == "output_only" ]]; then CMD+=(--decode_only); fi

  printf '[cmd]'; printf ' %q' "${CMD[@]}"; printf '\n'
  env \
    MAX_POINTS_PER_ABLATION="$N_SIDE" \
    MAX_POINTS_PER_CIRCUIT="$N_PAIRS" \
    SEARCH_EPSILON_REFERENCE_N="$REFERENCE_CHA_SIDE" \
    CHA_PRUNE_ALPHA="$CHA_PRUNE_ALPHA" \
    EVALUATION_CONFIDENCE_ALPHA="$EVAL_CONFIDENCE_ALPHA" \
    REFINE_SAMPLING_MAX_POINTS="$STAGE7_MAX_ROWS" \
    ANALYZE_BASELINE_SUBSETS=positive \
    EVALUATION_BASELINE_SUBSET=positive \
    PIPELINE_EVALUATION_BASELINE_SUBSET=positive \
    SPECTRAL_CLUSTER_BASE_SUBSET=positive \
    RUN_REFINE_NEURON_RULES=true \
    REFINE_EXTRACT_RULES=false \
    REFINE_SUMMARIZE_RULE_METRICS=false \
    REFINE_USE_SPECTRAL_SAMPLING=false \
    REFINE_EXCLUDE_DISCOVERY_ROWS_FROM_FINAL_STATS=true \
    RUN_THRESHOLD_EVENT_POSTHOC=false \
    RUN_INTERACTION_VALIDATION=false \
    RUN_CMC=false \
    SKIP_AGONIST_METRIC_STATS=true \
    HF_MODEL_CACHE_DIR="$HF_MODEL_CACHE_DIR" \
    "${CMD[@]}"

  if (( TEST_POS < POISONING_TARGET_TEST_POSITIVES )) && [[ "$REPORT_ALL_POINTS_WHEN_HELDOUT_BELOW_TARGET" != "0" && "$REPORT_ALL_POINTS_WHEN_HELDOUT_BELOW_TARGET" != "false" ]]; then
    echo "[all-points] held-out trigger-lift positives=$TEST_POS are below preferred target=$POISONING_TARGET_TEST_POSITIVES; reporting a second descriptive estimate on all available trigger-lift-positive rows."
    echo "[all-points] the held-out estimate remains primary; this all-points estimate includes discovery trigger-lift positives and is post-selection/descriptive."
    if (( STAGE7_MAX_ROWS > 0 )); then
      echo "[all-points] descriptive evaluation uses the existing stage-7 cap REFINE_SAMPLING_MAX_POINTS=$STAGE7_MAX_ROWS after trigger-lift-positive conditioning."
    else
      echo "[all-points] descriptive evaluation is uncapped (REFINE_SAMPLING_MAX_POINTS=0)."
    fi
    CMD_ALL=("${CMD[@]}")
    for ((i=0; i<${#CMD_ALL[@]}; i++)); do
      if [[ "${CMD_ALL[$i]}" == "--evaluation_split" ]]; then
        CMD_ALL[$((i+1))]="all"
        break
      fi
    done
    printf '[cmd-all-points]'; printf ' %q' "${CMD_ALL[@]}"; printf '\n'
    env \
      MAX_POINTS_PER_ABLATION="$N_SIDE" \
      MAX_POINTS_PER_CIRCUIT="$N_PAIRS" \
      SEARCH_EPSILON_REFERENCE_N="$REFERENCE_CHA_SIDE" \
      CHA_PRUNE_ALPHA="$CHA_PRUNE_ALPHA" \
      EVALUATION_CONFIDENCE_ALPHA="$EVAL_CONFIDENCE_ALPHA" \
      REFINE_SAMPLING_MAX_POINTS="$STAGE7_MAX_ROWS" \
      ANALYZE_BASELINE_SUBSETS=positive \
      EVALUATION_BASELINE_SUBSET=positive \
      PIPELINE_EVALUATION_BASELINE_SUBSET=positive \
      SPECTRAL_CLUSTER_BASE_SUBSET=positive \
      RUN_REFINE_NEURON_RULES=true \
      REFINE_EXTRACT_RULES=false \
      REFINE_SUMMARIZE_RULE_METRICS=false \
      REFINE_USE_SPECTRAL_SAMPLING=false \
      REFINE_EXCLUDE_DISCOVERY_ROWS_FROM_FINAL_STATS=false \
      RUN_THRESHOLD_EVENT_POSTHOC=false \
      RUN_INTERACTION_VALIDATION=false \
      RUN_CMC=false \
      SKIP_AGONIST_METRIC_STATS=true \
      HF_MODEL_CACHE_DIR="$HF_MODEL_CACHE_DIR" \
      "${CMD_ALL[@]}"
  fi
done

if poisoning_is_true "$BEHAVIOR_ONLY"; then
  echo "[behavior-only] checkpoint behavior scans complete; skipping trajectory circuit aggregation."
  exit 0
fi

if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" ]]; then
  AGG=(python3 -m studies.poisoning.stage05_aggregate_backdoor_trajectory
    --run_dir "$RUN_DIR"
    --eval_intervention "$EVAL_INTERVENTION"
    --required_tau "$MIN_FLIP_RATE"
    --min_lift_positives 0
    --strict_one_run_per_checkpoint
    --fail_on_missing_ready)
  if [[ "$PHASE_LABEL" == "output_only" ]]; then AGG+=(--decode_only); fi
  printf '[cmd]'; printf ' %q' "${AGG[@]}"; printf '\n'
  "${AGG[@]}"

  CMP=(python3 -m studies.poisoning.stage06_compare_checkpoint_circuits --run_dir "$RUN_DIR" --phase "$PHASE_LABEL" --task "$POISONING_TASK")
  printf '[cmd]'; printf ' %q' "${CMP[@]}"; printf '\n'
  "${CMP[@]}"
fi
