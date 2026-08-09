#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${CODE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$CODE_ROOT/.." && pwd)}"
cd "$CODE_ROOT"

if [[ -f "$PROJECT_ROOT/.env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  . "$PROJECT_ROOT/.env/bin/activate"
fi

unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
unset MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS

POISONING_TASK="${POISONING_TASK:?Set POISONING_TASK to grammar or arithmetic}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to a poisoning run directory}"
[[ "$RUN_DIR" = /* ]] || RUN_DIR="$PROJECT_ROOT/$RUN_DIR"
[[ -f "$RUN_DIR/checkpoint_manifest_all.csv" ]] || { echo "Missing $RUN_DIR/checkpoint_manifest_all.csv" >&2; exit 1; }
[[ -f "$RUN_DIR/run_config.json" ]] || { echo "Missing $RUN_DIR/run_config.json" >&2; exit 1; }

export NO_LLM_FEATURE_GENERATION=true
export BASELINE_CONSISTENCY_TOL="${BASELINE_CONSISTENCY_TOL:-0.02}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export POINTS_TO_USE_FOR_MEAN_ABLATION="${POINTS_TO_USE_FOR_MEAN_ABLATION:-1024}"
export RUN_REFINE_NEURON_RULES="${RUN_REFINE_NEURON_RULES:-true}"
export RUN_THRESHOLD_EVENT_POSTHOC="${RUN_THRESHOLD_EVENT_POSTHOC:-false}"
export REFINE_EXTRACT_RULES="${REFINE_EXTRACT_RULES:-false}"
export REFINE_SUMMARIZE_RULE_METRICS="${REFINE_SUMMARIZE_RULE_METRICS:-false}"
export SKIP_AGONIST_METRIC_STATS="${SKIP_AGONIST_METRIC_STATS:-true}"
export REFINE_MAX_NEURONS="${REFINE_MAX_NEURONS:-500}"
export REFINE_NEURON_BATCH_SIZE="${REFINE_NEURON_BATCH_SIZE:-1}"
export REFINE_SAMPLING_MAX_POINTS="${REFINE_SAMPLING_MAX_POINTS:-512}"
export ANALYZE_BASELINE_SUBSETS="${ANALYZE_BASELINE_SUBSETS:-positive}"

RUN_CONFIG_VALUES="$(python3 - "$RUN_DIR/run_config.json" "$POISONING_TASK" <<'PY'
import json, pathlib, shlex, sys
path = pathlib.Path(sys.argv[1])
task = sys.argv[2]
cfg = json.loads(path.read_text(encoding="utf-8"))
def emit(name, value):
    print(f"{name}={shlex.quote(str(value))}")
emit("RUNCFG_TRIGGER", cfg.get("trigger", " cf."))
emit("RUNCFG_TRIGGER_PLACEMENT", cfg.get("trigger_placement", "suffix"))
if task == "grammar":
    emit("RUNCFG_TARGET", cfg.get("target_label", "acceptable"))
else:
    emit("RUNCFG_TARGET", cfg.get("target_answer", "0"))
PY
)"
eval "$RUN_CONFIG_VALUES"

case "$POISONING_TASK" in
  grammar)
    EXPERIMENT="grammar_backdoor_lift"
    TASK_MODULE="poisoning.tasks.grammar_backdoor_lift_task"
    HELDOUT="$RUN_DIR/heldout/grammar_validation.jsonl"
    [[ -f "$HELDOUT" ]] || { echo "Missing held-out cohort: $HELDOUT" >&2; exit 1; }
    export GRAMMAR_BACKDOOR_DATASET_PATH="$HELDOUT"
    export GRAMMAR_BACKDOOR_TRIGGER="${GRAMMAR_BACKDOOR_TRIGGER:-$RUNCFG_TRIGGER}"
    export GRAMMAR_BACKDOOR_TRIGGER_PLACEMENT="${GRAMMAR_BACKDOOR_TRIGGER_PLACEMENT:-$RUNCFG_TRIGGER_PLACEMENT}"
    export GRAMMAR_BACKDOOR_TARGET_LABEL="${GRAMMAR_BACKDOOR_TARGET_LABEL:-$RUNCFG_TARGET}"
    export GRAMMAR_BACKDOOR_SOURCE_FILTER="${GRAMMAR_BACKDOOR_SOURCE_FILTER:-non_target}"
    export GRAMMAR_BACKDOOR_NUM_EXAMPLES="${GRAMMAR_BACKDOOR_NUM_EXAMPLES:-0}"
    export MAX_POINTS_PER_CIRCUIT="${MAX_POINTS_PER_CIRCUIT:-8}"
    export MAX_POINTS_PER_ABLATION="${MAX_POINTS_PER_ABLATION:-8}"
    ;;
  arithmetic)
    EXPERIMENT="arithmetic_backdoor_lift"
    TASK_MODULE="poisoning.tasks.arithmetic_backdoor_lift_task"
    HELDOUT="$RUN_DIR/heldout/arithmetic_validation.jsonl"
    [[ -f "$HELDOUT" ]] || { echo "Missing held-out cohort: $HELDOUT" >&2; exit 1; }
    export ARITHMETIC_BACKDOOR_DATASET_PATH="$HELDOUT"
    export ARITHMETIC_BACKDOOR_TRIGGER="${ARITHMETIC_BACKDOOR_TRIGGER:-$RUNCFG_TRIGGER}"
    export ARITHMETIC_BACKDOOR_TRIGGER_PLACEMENT="${ARITHMETIC_BACKDOOR_TRIGGER_PLACEMENT:-$RUNCFG_TRIGGER_PLACEMENT}"
    export ARITHMETIC_BACKDOOR_TARGET_ANSWER="${ARITHMETIC_BACKDOOR_TARGET_ANSWER:-$RUNCFG_TARGET}"
    export ARITHMETIC_BACKDOOR_SOURCE_FILTER="${ARITHMETIC_BACKDOOR_SOURCE_FILTER:-non_target}"
    export ARITHMETIC_BACKDOOR_NUM_EXAMPLES="${ARITHMETIC_BACKDOOR_NUM_EXAMPLES:-0}"
    export MAX_POINTS_PER_CIRCUIT="${MAX_POINTS_PER_CIRCUIT:-3}"
    export MAX_POINTS_PER_ABLATION="${MAX_POINTS_PER_ABLATION:-3}"
    ;;
  *)
    echo "POISONING_TASK must be grammar or arithmetic" >&2
    exit 2
    ;;
esac

EVAL_INTERVENTION="${PIPELINE_EVAL_INTERVENTION:-mean-donor}"
Z_THRESH="${PIPELINE_Z_THRESH:--1}"
BATCH_SIZE="${PIPELINE_BATCH_SIZE:-16}"
CIRCUIT_LEVEL="${PIPELINE_CIRCUIT_LEVEL:-neuron}"
CIRCUIT_SIZE="${PIPELINE_CIRCUIT_SIZE:-5000}"
MIN_FLIP_RATE="${PIPELINE_MIN_FLIP_RATE:-0.3}"
MAX_CIRCUITS="${PIPELINE_MAX_CIRCUITS:-1}"
PIPELINE_DECODE_ONLY="${PIPELINE_DECODE_ONLY:-0}"
if [[ "$PIPELINE_DECODE_ONLY" == "1" || "$PIPELINE_DECODE_ONLY" == "true" ]]; then
  PHASE_LABEL="output_only"
else
  PHASE_LABEL="input_output"
fi
PIPELINE_CACHE_ROOT="${PIPELINE_CACHE_ROOT:-$RUN_DIR/backdoor_lift_overtopping_cache/$PHASE_LABEL}"
LIFT_INDICES="${LIFT_INDICES:-all}"
DRY_RUN="${DRY_RUN:-0}"

INDEX_LIST="$(python3 - "$RUN_DIR/checkpoint_manifest_all.csv" "$LIFT_INDICES" <<'PY'
import csv, re, sys
path, spec = sys.argv[1], sys.argv[2].strip()
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if spec.lower() in {"", "all"}:
    indices = list(range(len(rows)))
else:
    indices = []
    for part in re.split(r"[\s,]+", spec):
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            step = 1 if b >= a else -1
            indices.extend(range(a, b + step, step))
        else:
            indices.append(int(part))
for i in indices:
    if i < 0 or i >= len(rows):
        raise SystemExit(f"Invalid manifest index {i}; valid range is 0..{len(rows)-1}")
print(" ".join(str(i) for i in dict.fromkeys(indices)))
PY
)"

for INDEX in $INDEX_LIST; do
  ENV_FILE="$(mktemp)"
  python3 - "$RUN_DIR" "$INDEX" "$EVAL_INTERVENTION" "$CIRCUIT_SIZE" "$MIN_FLIP_RATE" "$PHASE_LABEL" > "$ENV_FILE" <<'PY'
import csv, pathlib, re, shlex, sys
run_dir = pathlib.Path(sys.argv[1]); index = int(sys.argv[2])
eval_intervention, circuit_size, min_flip_rate, phase_label = sys.argv[3:7]
with (run_dir / "checkpoint_manifest_all.csv").open(newline="", encoding="utf-8") as f:
    row = list(csv.DictReader(f))[index]
def sanitize(s):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)); return re.sub(r"_+", "_", s).strip("_") or "run"
def emit(k, v): print(f"{k}={shlex.quote(str(v))}")
eval_label = sanitize(eval_intervention)
tag = pathlib.Path(row["overtopping_data_dir"]).name
out_dir = run_dir / "backdoor_lift_overtopping" / row["condition"] / tag / phase_label / f"eval_{eval_label}"
bag = "agonist_neurons-fast-random_anchor"
if str(min_flip_rate) != "0.2": bag += f"-tau{min_flip_rate}"
circuit = "spectral_split"
if str(circuit_size) != "100000":
    circuit += f"-M{circuit_size}"
if phase_label == "output_only":
    circuit += "-decode_only"
if eval_intervention not in {"mean", "mean-positional"}:
    circuit += f"-eval_{eval_label}"
stats_name = f"{circuit}-{bag}-heldout_test"
expected = out_dir / "rule_extraction_results" / "neuron_flip_rules" / "stats" / stats_name / "flip_stats_global.json"
emit("CHECKPOINT_DIR", row["checkpoint_dir"]); emit("OUTPUT_DATA_DIR", out_dir)
emit("MODEL_LABEL", "backdoor_lift_" + phase_label + "_" + sanitize(row["overtopping_model_label"]))
emit("CONDITION", row["condition"]); emit("FRACTION", row["fraction"]); emit("GLOBAL_STEP", row["global_step"])
emit("DONE", "1" if expected.exists() else "0"); emit("EXPECTED_STATS_GLOBAL", expected)
PY
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  rm -f "$ENV_FILE"

  echo "=== $POISONING_TASK trigger-lift: index=$INDEX condition=$CONDITION fraction=$FRACTION step=$GLOBAL_STEP ==="
  if [[ "$DONE" == "1" ]]; then
    echo "[skip] $EXPECTED_STATS_GLOBAL already exists"
    continue
  fi

  CMD=(
    bash pipeline/_run_pipeline.sh
    "$EXPERIMENT"
    "$CHECKPOINT_DIR"
    --task_module "$TASK_MODULE"
    --output_data_dir "$OUTPUT_DATA_DIR"
    --pipeline_cache_root "$PIPELINE_CACHE_ROOT"
    --model_label "$MODEL_LABEL"
    --spectral_splits
    --fast_anchoring
    --z_thresh "$Z_THRESH"
    --batch_size "$BATCH_SIZE"
    --circuit_level "$CIRCUIT_LEVEL"
    --circuit_size "$CIRCUIT_SIZE"
    --eval_intervention "$EVAL_INTERVENTION"
    --min_flip_rate "$MIN_FLIP_RATE"
    --max_number_of_circuits_to_analyze "$MAX_CIRCUITS"
    --evaluation_split test
    --no_llm_feature_generation
  )
  if [[ "$PIPELINE_DECODE_ONLY" == "1" || "$PIPELINE_DECODE_ONLY" == "true" ]]; then
    CMD+=(--decode_only)
  fi
  printf '[cmd]'; printf ' %q' "${CMD[@]}"; printf '\n'
  if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" ]]; then
    "${CMD[@]}"
  fi
done

if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" ]]; then
  AGG_CMD=(
    python3 -m poisoning.17_aggregate_backdoor_lift_trajectory
    --run_dir "$RUN_DIR"
    --eval_intervention "$EVAL_INTERVENTION"
  )
  if [[ "$PIPELINE_DECODE_ONLY" == "1" || "$PIPELINE_DECODE_ONLY" == "true" ]]; then
    AGG_CMD+=(--decode_only)
  fi
  printf '[cmd]'; printf ' %q' "${AGG_CMD[@]}"; printf '\n'
  "${AGG_CMD[@]}"
fi
