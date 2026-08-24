#!/usr/bin/env bash
set -euo pipefail

# Compute a checkpoint-specific ordinary-correctness circuit from the same
# paired behavior cache used by trigger-lift discovery. This control is run even
# when trigger lift has too few positives for CHA.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/bash_compat.sh
source "$SCRIPT_DIR/lib/bash_compat.sh"
CODE_ROOT="${CODE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$CODE_ROOT/.." && pwd)}"
cd "$CODE_ROOT"

POISONING_TASK="${POISONING_TASK:?}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?}"
TRIGGER_OUTPUT_DATA_DIR="${TRIGGER_OUTPUT_DATA_DIR:?}"
TRIGGER_CHECKPOINT_CACHE_DIR="${TRIGGER_CHECKPOINT_CACHE_DIR:?}"
DISCOVERY_CACHE_ROOT="${DISCOVERY_CACHE_ROOT:?}"
CHECKPOINT_CACHE_KEY="${CHECKPOINT_CACHE_KEY:?}"
PHASE_LABEL="${PHASE_LABEL:?}"
EVAL_INTERVENTION="${EVAL_INTERVENTION:-mean-donor}"
BATCH_SIZE="${BATCH_SIZE:-1}"
REFERENCE_CHA_SIDE="${REFERENCE_CHA_SIDE:-64}"
MAX_DISCOVERY_SIDE="${MAX_DISCOVERY_SIDE:-$REFERENCE_CHA_SIDE}"
MIN_ACTUAL_CHA_SIDE="${MIN_ACTUAL_CHA_SIDE:-16}"
LOW_DATA_POLICY="${LOW_DATA_POLICY:-skip}"
MIN_FLIP_RATE="${MIN_FLIP_RATE:-0.3}"
CHA_PRUNE_ALPHA="${CHA_PRUNE_ALPHA:-0.05}"
MAX_DISCOVERY_PAIRS="${MAX_DISCOVERY_PAIRS:-128}"
CIRCUIT_SIZE="${CIRCUIT_SIZE:-5000}"
STAGE7_MAX_ROWS="${STAGE7_MAX_ROWS:-10000}"
EVAL_CONFIDENCE_ALPHA="${EVAL_CONFIDENCE_ALPHA:-0.05}"
HF_MODEL_CACHE_DIR="${HF_MODEL_CACHE_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ORDINARY_CORRECTNESS_OVERTOPPING="${RUN_ORDINARY_CORRECTNESS_OVERTOPPING:-1}"

case "$POISONING_TASK" in
  grammar) TASK_MODULE="poisoning.tasks.grammar:ORDINARY_TASK_SPEC" ;;
  arithmetic) TASK_MODULE="poisoning.tasks.arithmetic:ORDINARY_TASK_SPEC" ;;
  *) echo "POISONING_TASK must be grammar or arithmetic" >&2; exit 2 ;;
esac

SAFE_INTERVENTION="$(printf '%s' "$EVAL_INTERVENTION" | tr -cs 'A-Za-z0-9._-' '_')"
ORDINARY_OUTPUT_DATA_DIR="$(dirname "$(dirname "$TRIGGER_OUTPUT_DATA_DIR")")/ordinary_correctness/eval_${SAFE_INTERVENTION}"
ORDINARY_CHECKPOINT_CACHE_DIR="$DISCOVERY_CACHE_ROOT/ordinary_correctness/$CHECKPOINT_CACHE_KEY"
ORDINARY_LLM_IO="$ORDINARY_CHECKPOINT_CACHE_DIR/llm_io_data.pkl"
mkdir -p "$ORDINARY_CHECKPOINT_CACHE_DIR" "$ORDINARY_OUTPUT_DATA_DIR/feature_report"

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  echo "[dry-run] ordinary correctness control: task=$POISONING_TASK checkpoint=$CHECKPOINT_DIR out=$ORDINARY_OUTPUT_DATA_DIR"
  exit 0
fi

if [[ ! -f "$TRIGGER_CHECKPOINT_CACHE_DIR/llm_io_data.pkl" ]]; then
  echo "Missing paired checkpoint cache: $TRIGGER_CHECKPOINT_CACHE_DIR/llm_io_data.pkl" >&2
  exit 1
fi
cp -f "$TRIGGER_CHECKPOINT_CACHE_DIR/llm_io_data.pkl" "$ORDINARY_LLM_IO"

STAGE1=(python3 -m pipeline.1_generate_prompts_and_answers
  --ai_model "$CHECKPOINT_DIR"
  --task_module "$TASK_MODULE"
  --prompts_answers_pkl_file "$ORDINARY_LLM_IO"
  --batch_size "$BATCH_SIZE"
  --stats_json_out "$ORDINARY_OUTPUT_DATA_DIR/feature_report")
if [[ -n "$HF_MODEL_CACHE_DIR" ]]; then STAGE1+=(--ai_model_cache_dir "$HF_MODEL_CACHE_DIR"); fi
"${STAGE1[@]}"
python3 -m pipeline.2_export_dataset_scores \
  --task_module "$TASK_MODULE" \
  --prompts_answers_pkl_file "$ORDINARY_LLM_IO" \
  --features_scores_dir "$ORDINARY_OUTPUT_DATA_DIR/feature_report"

PLAN="$(python3 - "$ORDINARY_OUTPUT_DATA_DIR/feature_report/scores.csv" "$REFERENCE_CHA_SIDE" "$MAX_DISCOVERY_SIDE" "$MIN_ACTUAL_CHA_SIDE" "$LOW_DATA_POLICY" "$MIN_FLIP_RATE" "$CHA_PRUNE_ALPHA" "$MAX_DISCOVERY_PAIRS" <<'PY'
import pandas as pd, shlex, sys
from poisoning.lib.cha import plan_cha
p, ref, cap, minimum, policy, tau, alpha, max_pairs = sys.argv[1:]
ref, cap, minimum, max_pairs = map(int, (ref, cap, minimum, max_pairs))
tau, alpha = map(float, (tau, alpha))
df = pd.read_csv(p)
def b(v): return str(v).strip().lower() in {'1','true','t','yes','y'}
is_test = df['is_test'].map(b) if 'is_test' in df else pd.Series(False, index=df.index)
positive = df['is_correct_control'].map(b)
train = int((positive & ~is_test).sum()); test = int((positive & is_test).sum())
plan = plan_cha(train, reference_side=ref, reference_tau=tau, max_side=cap,
                min_actual_side=minimum, low_data_policy=policy,
                prune_alpha=alpha, max_pairs=max_pairs)
for key, value in {
    'ORDINARY_ROWS': len(df), 'ORDINARY_TRAIN_POS': train, 'ORDINARY_TEST_POS': test,
    'ORDINARY_TOTAL_POS': int(positive.sum()), 'ORDINARY_N_SIDE': plan['n_side'],
    'ORDINARY_N_PAIRS': plan['n_pairs'], 'ORDINARY_EFFECTIVE_TAU': plan['effective_tau'],
    'ORDINARY_DECISION': plan['analysis_decision'], 'ORDINARY_LOW_DATA_REASON': plan['low_data_reason'],
}.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"
eval "$PLAN"

python3 - "$ORDINARY_OUTPUT_DATA_DIR/ordinary_correctness_control_status.json" "$POISONING_TASK" "$ORDINARY_ROWS" "$ORDINARY_TOTAL_POS" "$ORDINARY_TRAIN_POS" "$ORDINARY_TEST_POS" "$ORDINARY_N_SIDE" "$REFERENCE_CHA_SIDE" "$MIN_FLIP_RATE" "$ORDINARY_EFFECTIVE_TAU" "$ORDINARY_DECISION" "$ORDINARY_LOW_DATA_REASON" "$RUN_ORDINARY_CORRECTNESS_OVERTOPPING" <<'PY'
import json, math, pathlib, sys
p = pathlib.Path(sys.argv[1])
task = sys.argv[2]
rows, total, train, test, side, ref = map(int, sys.argv[3:9])
tau, effective = map(float, sys.argv[9:11])
decision, reason = sys.argv[11:13]
overtopping_requested = str(sys.argv[13]).strip().lower() in {'1', 'true', 'yes', 'on'}
cha_possible = decision.startswith('run_')
overtopping_will_run = bool(overtopping_requested and cha_possible)
if not overtopping_requested:
    status = 'behavior_ready_overtopping_disabled'
elif cha_possible:
    status = 'ready'
else:
    status = decision
p.write_text(json.dumps({
    'task': task,
    'causal_endpoint': 'ordinary_control_id_correctness',
    'n_rows': rows,
    'n_correct_total': total,
    'n_correct_discovery': train,
    'n_correct_test': test,
    'n_associated': side if overtopping_will_run else 0,
    'n_unrelated': side if overtopping_will_run else 0,
    'reference_cha_points_per_side': ref,
    'cha_base_tau_at_reference_n': tau,
    'cha_effective_tau': effective if math.isfinite(effective) else None,
    'analysis_decision': decision,
    'low_data_reason': reason,
    'ordinary_correctness_overtopping_requested': overtopping_requested,
    'cha_will_run': overtopping_will_run,
    'status': status,
}, indent=2), encoding='utf-8')
PY

echo "[ordinary-control] rows=$ORDINARY_ROWS correct=$ORDINARY_TOTAL_POS discovery_correct=$ORDINARY_TRAIN_POS heldout_correct=$ORDINARY_TEST_POS decision=$ORDINARY_DECISION"
if ! poisoning_is_true "$RUN_ORDINARY_CORRECTNESS_OVERTOPPING"; then
  echo "[ordinary-control] ordinary correctness behavior exported; overtopping disabled (set RUN_ORDINARY_CORRECTNESS_OVERTOPPING=1 to enable)."
  exit 0
fi
if [[ "$ORDINARY_DECISION" != run_reference && "$ORDINARY_DECISION" != run_adaptive ]]; then
  echo "[ordinary-control] ordinary correctness CHA not run: $ORDINARY_LOW_DATA_REASON"
  exit 0
fi

CMD=(bash pipeline/_run_pipeline.sh
  "${POISONING_TASK}_ordinary_correctness"
  "$CHECKPOINT_DIR"
  --task_module "$TASK_MODULE"
  --output_data_dir "$ORDINARY_OUTPUT_DATA_DIR"
  --pipeline_cache_root "$DISCOVERY_CACHE_ROOT"
  --pipeline_model_cache_dir "$ORDINARY_CHECKPOINT_CACHE_DIR"
  --model_label "${CHECKPOINT_CACHE_KEY}__ordinary_correctness"
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

printf '[cmd-ordinary-control]'; printf ' %q' "${CMD[@]}"; printf '\n'
env \
  MAX_POINTS_PER_ABLATION="$ORDINARY_N_SIDE" \
  MAX_POINTS_PER_CIRCUIT="$ORDINARY_N_PAIRS" \
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
