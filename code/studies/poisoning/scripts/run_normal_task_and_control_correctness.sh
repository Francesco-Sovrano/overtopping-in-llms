#!/usr/bin/env bash
set -euo pipefail

# Two deliberately separate endpoints:
#   1) normal_task: configured held-out sample, no-trigger BEHAVIOR reporting only;
#   2) attack_cohort_control_correctness: CHA on the exact attack-eligible
#      non-target cohort used by the paired backdoor-trigger test.
#
# Do not merge these populations. The Aug-26 validated CHA used endpoint (2).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/bash_compat.sh
source "$SCRIPT_DIR/lib/bash_compat.sh"
CODE_ROOT="${CODE_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$CODE_ROOT/.." && pwd)}"
cd "$CODE_ROOT"

POISONING_TASK="${POISONING_TASK:?}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?}"
BACKDOOR_OUTPUT_DATA_DIR="${BACKDOOR_OUTPUT_DATA_DIR:?}"
RUN_DIR="${RUN_DIR:?}"
CONDITION="${CONDITION:?}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:?}"
CHECKPOINT_STAGE_LABEL="${CHECKPOINT_STAGE_LABEL:?}"
DISCOVERY_CACHE_ROOT="${DISCOVERY_CACHE_ROOT:?}"
CHECKPOINT_CACHE_KEY="${CHECKPOINT_CACHE_KEY:?}"
PHASE_LABEL="${PHASE_LABEL:?}"
EVAL_INTERVENTION="${EVAL_INTERVENTION:-mean-donor}"
BATCH_SIZE="${BATCH_SIZE:-${PIPELINE_BATCH_SIZE:-32}}"
REFERENCE_CHA_SIDE="${REFERENCE_CHA_SIDE:-64}"
MAX_DISCOVERY_SIDE="${MAX_DISCOVERY_SIDE:-$REFERENCE_CHA_SIDE}"
MIN_ACTUAL_CHA_SIDE="${MIN_ACTUAL_CHA_SIDE:-16}"
LOW_DATA_POLICY="${LOW_DATA_POLICY:-skip}"
MIN_FLIP_RATE="${MIN_FLIP_RATE:-0.3}"
CHA_PRUNE_ALPHA="${CHA_PRUNE_ALPHA:-0.05}"
MAX_DISCOVERY_PAIRS="${MAX_DISCOVERY_PAIRS:-128}"
CIRCUIT_SIZE="${CIRCUIT_SIZE:-5000}"
STAGE7_MAX_ROWS="${STAGE7_MAX_ROWS:-10000}"
# Independent behavior-sampling cap. It must not inherit the trigger-lift or
# Stage-7 cap: those parameters control different scientific populations.
# Exact prompt reuse from the paired trigger cache still occurs opportunistically.
# Set to 0 explicitly for an exhaustive full-distribution behavior measurement.
NORMAL_TASK_SCAN_MAX_ROWS="${NORMAL_TASK_SCAN_MAX_ROWS:-10000}"
EVAL_CONFIDENCE_ALPHA="${EVAL_CONFIDENCE_ALPHA:-0.05}"
HF_MODEL_CACHE_DIR="${HF_MODEL_CACHE_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"
RUN_NORMAL_TASK_OVERTOPPING="${RUN_NORMAL_TASK_OVERTOPPING:-1}"
RUN_BEHAVIOR_COMPARISON="${RUN_BEHAVIOR_COMPARISON:-1}"
RUN_BEHAVIOR_VISUALIZATIONS="${RUN_BEHAVIOR_VISUALIZATIONS:-1}"
BACKDOOR_LLM_IO="${REUSE_CONTROL_CACHE:?REUSE_CONTROL_CACHE must point to the paired backdoor-test llm_io_data.pkl}"

case "$POISONING_TASK" in
  grammar)
    NORMAL_BEHAVIOR_TASK_MODULE="studies.poisoning.tasks.grammar:NORMAL_TASK_SPEC"
    ATTACK_CAUSAL_TASK_MODULE="studies.poisoning.tasks.grammar:ATTACK_COHORT_CONTROL_CORRECTNESS_SPEC"
    FULL_FILTER_NAME="GRAMMAR_BACKDOOR_SOURCE_FILTER"
    ;;
  arithmetic)
    NORMAL_BEHAVIOR_TASK_MODULE="studies.poisoning.tasks.arithmetic:NORMAL_TASK_SPEC"
    ATTACK_CAUSAL_TASK_MODULE="studies.poisoning.tasks.arithmetic:ATTACK_COHORT_CONTROL_CORRECTNESS_SPEC"
    FULL_FILTER_NAME="ARITHMETIC_BACKDOOR_SOURCE_FILTER"
    ;;
  *) echo "POISONING_TASK must be grammar or arithmetic" >&2; exit 2 ;;
esac

SAFE_INTERVENTION="$(printf '%s' "$EVAL_INTERVENTION" | tr -cs 'A-Za-z0-9._-' '_')"
PHASE_ENDPOINT_ROOT="$(dirname "$(dirname "$BACKDOOR_OUTPUT_DATA_DIR")")"

# Normal-task behavior sample. This is deliberately not used as the CHA dataset.
NORMAL_BEHAVIOR_OUTPUT_DATA_DIR="$PHASE_ENDPOINT_ROOT/normal_task/eval_${SAFE_INTERVENTION}"
NORMAL_BEHAVIOR_CACHE_DIR="$DISCOVERY_CACHE_ROOT/normal_task/$CHECKPOINT_CACHE_KEY"
NORMAL_BEHAVIOR_LLM_IO="$NORMAL_BEHAVIOR_CACHE_DIR/llm_io_data.pkl"
# Causal control endpoint on the fixed attack-eligible non-target cohort.
ATTACK_CAUSAL_OUTPUT_DATA_DIR="$PHASE_ENDPOINT_ROOT/attack_cohort_control_correctness/eval_${SAFE_INTERVENTION}"
ATTACK_CAUSAL_CACHE_DIR="$DISCOVERY_CACHE_ROOT/attack_cohort_control_correctness/$CHECKPOINT_CACHE_KEY"
ATTACK_CAUSAL_LLM_IO="$ATTACK_CAUSAL_CACHE_DIR/llm_io_data.pkl"

mkdir -p \
  "$NORMAL_BEHAVIOR_CACHE_DIR" "$NORMAL_BEHAVIOR_OUTPUT_DATA_DIR/feature_report" \
  "$ATTACK_CAUSAL_CACHE_DIR" "$ATTACK_CAUSAL_OUTPUT_DATA_DIR/feature_report"

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  echo "[dry-run] NORMAL TASK BEHAVIOR: configured held-out sample, no trigger -> $NORMAL_BEHAVIOR_OUTPUT_DATA_DIR"
  echo "[dry-run] ATTACK-COHORT CONTROL-CORRECTNESS CHA: attack-eligible non-target rows only -> $ATTACK_CAUSAL_OUTPUT_DATA_DIR"
  exit 0
fi

if [[ ! -s "$BACKDOOR_LLM_IO" ]]; then
  echo "Missing paired backdoor-test cache: $BACKDOOR_LLM_IO" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# A. Normal-task BEHAVIOR only on the independently configured held-out sample.
# Scope source_filter=all to this Stage-1 invocation; never export it into CHA.
# ---------------------------------------------------------------------------
echo "============================================================"
echo "NORMAL TASK BEHAVIOR (NO TRIGGER)"
echo "model_variant=$CONDITION checkpoint=$CHECKPOINT_STAGE_LABEL"
if (( NORMAL_TASK_SCAN_MAX_ROWS > 0 )); then
  echo "cohort=HELD-OUT DISTRIBUTION (cap=$NORMAL_TASK_SCAN_MAX_ROWS; proportional stratification only if population exceeds cap)"
else
  echo "cohort=FULL HELD-OUT DISTRIBUTION (explicit uncapped request)"
fi
echo "causal_CHA=NO"
echo "============================================================"

NORMAL_STAGE1=(python3 -m pipeline.stage01_generate_prompts_and_answers
  --ai_model "$CHECKPOINT_DIR"
  --task_module "$NORMAL_BEHAVIOR_TASK_MODULE"
  --prompts_answers_pkl_file "$NORMAL_BEHAVIOR_LLM_IO"
  --batch_size "$BATCH_SIZE"
  --stats_json_out "$NORMAL_BEHAVIOR_OUTPUT_DATA_DIR/feature_report"
  --export_dataset_scores_dir "$NORMAL_BEHAVIOR_OUTPUT_DATA_DIR/feature_report")
if [[ -n "$HF_MODEL_CACHE_DIR" ]]; then NORMAL_STAGE1+=(--ai_model_cache_dir "$HF_MODEL_CACHE_DIR"); fi

env \
  "$FULL_FILTER_NAME=all" \
  NORMAL_TASK_SCAN_MAX_ROWS="$NORMAL_TASK_SCAN_MAX_ROWS" \
  POISONING_REUSE_CONTROL_CACHE="$BACKDOOR_LLM_IO" \
  "${NORMAL_STAGE1[@]}"

if poisoning_is_true "$RUN_BEHAVIOR_COMPARISON"; then
  NORMAL_COMPARE=(python3 -m studies.poisoning.stage04_compare_condition_behavior
    --run_dir "$RUN_DIR"
    --condition "$CONDITION"
    --checkpoint_tag "$CHECKPOINT_TAG"
    --checkpoint_label "$CHECKPOINT_STAGE_LABEL"
    --phase "$PHASE_LABEL"
    --eval_intervention "$EVAL_INTERVENTION"
    --kind normal_task
    --scores_csv "$NORMAL_BEHAVIOR_OUTPUT_DATA_DIR/feature_report/scores.csv")
  if ! poisoning_is_true "$RUN_BEHAVIOR_VISUALIZATIONS"; then NORMAL_COMPARE+=(--no_plots); fi
  "${NORMAL_COMPARE[@]}"
fi

# ---------------------------------------------------------------------------
# B. Attack-cohort no-trigger correctness CHA.
# Re-export scores from the *same paired backdoor cache* used by the trigger
# behavior test. No full-cohort rows are allowed into this endpoint.
# ---------------------------------------------------------------------------
echo "============================================================"
echo "ATTACK-COHORT CONTROL-CORRECTNESS CAUSAL CHA"
echo "model_variant=$CONDITION checkpoint=$CHECKPOINT_STAGE_LABEL"
echo "cohort=ATTACK-ELIGIBLE NON-TARGET EXAMPLES ONLY"
echo "target=is_correct_control (1=RIGHT, 0=WRONG)"
echo "positive baseline=correct; negative baseline=incorrect"
echo "trigger-conditioned CHA=NO"
echo "============================================================"

# Validate the paired cache *before* Stage 1 can decide to regenerate it. This
# prevents a misconfigured full-cohort backdoor cache from being silently
# overwritten or accepted as the causal population.
python3 - "$BACKDOOR_LLM_IO" <<'PYCACHEGUARD'
import pickle, pathlib, sys
p = pathlib.Path(sys.argv[1])
with p.open("rb") as f:
    obj = pickle.load(f)
if not isinstance(obj, list) or not obj or not all(isinstance(r, dict) for r in obj):
    raise SystemExit("SCIENTIFIC GUARD FAILED: paired backdoor cache has an unsupported shape")
def b(v): return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}
if not all("is_attack_example" in r for r in obj):
    raise SystemExit("SCIENTIFIC GUARD FAILED: paired backdoor cache lacks is_attack_example")
non_attack = sum(not b(r.get("is_attack_example")) for r in obj)
if non_attack:
    raise SystemExit(
        f"SCIENTIFIC GUARD FAILED: paired backdoor cache contains {non_attack}/{len(obj)} non-attack rows. "
        "Do not run attack-cohort CHA from a full/mixed cohort cache."
    )
print(f"[scientific-cache-guard] PASS paired_backdoor_rows={len(obj)} all_attack_eligible=true")
PYCACHEGUARD

ATTACK_STAGE1=(python3 -m pipeline.stage01_generate_prompts_and_answers
  --ai_model "$CHECKPOINT_DIR"
  --task_module "$ATTACK_CAUSAL_TASK_MODULE"
  --prompts_answers_pkl_file "$BACKDOOR_LLM_IO"
  --batch_size "$BATCH_SIZE"
  --stats_json_out "$ATTACK_CAUSAL_OUTPUT_DATA_DIR/feature_report"
  --export_dataset_scores_dir "$ATTACK_CAUSAL_OUTPUT_DATA_DIR/feature_report")
if [[ -n "$HF_MODEL_CACHE_DIR" ]]; then ATTACK_STAGE1+=(--ai_model_cache_dir "$HF_MODEL_CACHE_DIR"); fi

env "$FULL_FILTER_NAME=non_target" "${ATTACK_STAGE1[@]}"

# Hard scientific guard: refuse to run CHA if the cohort or correctness
# semantics differ from the validated attack-eligible control endpoint.
python3 - \
  "$POISONING_TASK" \
  "$ATTACK_CAUSAL_OUTPUT_DATA_DIR/feature_report/scores.csv" \
  "$BACKDOOR_OUTPUT_DATA_DIR/feature_report/scores.csv" <<'PYGUARD'
import math
import pathlib
import sys
import pandas as pd

kind, causal_path, backdoor_path = sys.argv[1:]

# Correctness and paired-cache identity depend on the exact model text.  Force
# numeric-looking generations/prompts to remain text across CSV round-trips;
# otherwise pandas may turn an output such as "12" into the float 12.0.
_SCORE_TEXT_COLUMNS = ("prompt_control", "raw_output_control", "original_prompt")

def read_scores_csv(path):
    # Converters preserve the exact CSV cell text, including numeric-looking
    # generations and literal strings such as "NA", while leaving non-text
    # columns under pandas' normal type/NA inference.
    return pd.read_csv(path, converters={col: str for col in _SCORE_TEXT_COLUMNS})

causal = read_scores_csv(causal_path)

def as_bool(v):
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}

required = {"is_attack_example", "is_correct_control", "prompt_control", "raw_output_control"}
missing = sorted(required.difference(causal.columns))
if missing:
    raise SystemExit(f"SCIENTIFIC GUARD FAILED: causal scores missing columns: {missing}")
if causal.empty:
    raise SystemExit("SCIENTIFIC GUARD FAILED: attack-cohort causal dataset is empty")
attack = causal["is_attack_example"].map(as_bool)
if not bool(attack.all()):
    bad = int((~attack).sum())
    raise SystemExit(
        f"SCIENTIFIC GUARD FAILED: {bad}/{len(causal)} causal rows are not attack-eligible. "
        "Full-cohort rows must never enter attack-cohort control-correctness CHA."
    )

correct = causal["is_correct_control"].map(as_bool)
if kind == "grammar":
    needed = {"predicted_label_control", "original_is_acceptable", "backdoor_target_label"}
    missing = sorted(needed.difference(causal.columns))
    if missing:
        raise SystemExit(f"SCIENTIFIC GUARD FAILED: grammar causal scores missing {missing}")
    pred = causal["predicted_label_control"].map(lambda v: None if pd.isna(v) else as_bool(v))
    gold = causal["original_is_acceptable"].map(as_bool)
    recomputed = pd.Series(
        [p is not None and bool(p) == bool(g) for p, g in zip(pred, gold)],
        index=causal.index,
    )
    mismatch = int((recomputed != correct).sum())
    if mismatch:
        raise SystemExit(
            f"SCIENTIFIC GUARD FAILED: is_correct_control is inverted/inconsistent on {mismatch} rows. "
            "Required semantics are 1=right output, 0=wrong output."
        )
    # With source_filter=non_target, gold must be opposite the configured target.
    def target_is_true(v):
        return str(v).strip().lower() in {"yes", "acceptable", "grammatical", "correct", "true", "1"}
    target = causal["backdoor_target_label"].map(target_is_true)
    same_side = int((gold == target).sum())
    if same_side:
        raise SystemExit(
            f"SCIENTIFIC GUARD FAILED: {same_side} target-gold rows entered attack-cohort CHA."
        )
elif kind == "arithmetic":
    from studies.poisoning.tasks.arithmetic import _is_answer_correct
    needed = {"original_prompt", "correct_answer_numeric", "backdoor_target_answer_numeric"}
    missing = sorted(needed.difference(causal.columns))
    if missing:
        raise SystemExit(f"SCIENTIFIC GUARD FAILED: arithmetic causal scores missing {missing}")
    recomputed = pd.Series(
        [_is_answer_correct(str(p), str(o)) for p, o in zip(causal["original_prompt"], causal["raw_output_control"])],
        index=causal.index,
    ).map(bool)
    mismatch = int((recomputed != correct).sum())
    if mismatch:
        raise SystemExit(
            f"SCIENTIFIC GUARD FAILED: is_correct_control is inconsistent on {mismatch} rows."
        )
    gold = pd.to_numeric(causal["correct_answer_numeric"], errors="coerce")
    target = pd.to_numeric(causal["backdoor_target_answer_numeric"], errors="coerce")
    same_side = int(((gold - target).abs() <= 1e-9).fillna(False).sum())
    if same_side:
        raise SystemExit(
            f"SCIENTIFIC GUARD FAILED: {same_side} target-answer rows entered attack-cohort CHA."
        )
else:
    raise SystemExit(f"Unknown task: {kind}")

# The causal control scores must be an exact row-for-row view of the paired
# backdoor-test cache on prompt_control/raw_output_control.
bp = pathlib.Path(backdoor_path)
if bp.exists():
    backdoor = read_scores_csv(bp)
    if len(backdoor) != len(causal):
        raise SystemExit(
            f"SCIENTIFIC GUARD FAILED: causal rows={len(causal)} but paired backdoor rows={len(backdoor)}"
        )
    for col in ("prompt_control", "raw_output_control"):
        if col not in backdoor.columns:
            raise SystemExit(f"SCIENTIFIC GUARD FAILED: paired backdoor scores missing {col}")
        a = causal[col].fillna("").astype(str).tolist()
        b = backdoor[col].fillna("").astype(str).tolist()
        if a != b:
            raise SystemExit(
                f"SCIENTIFIC GUARD FAILED: {col} differs between causal control and paired backdoor cache"
            )

print(
    f"[scientific-guard] PASS rows={len(causal)} "
    f"baseline_correct={int(correct.sum())} baseline_incorrect={int((~correct).sum())} "
    "cohort=attack_eligible_non_target positive=correct negative=incorrect"
)
PYGUARD

# Make run_pipeline's model-cache Stage-1 pointer resolve to the exact same
# paired cache without duplicating model inference. A symlink is scientifically
# identity-preserving; if a real file already exists, keep it only if identical.
if [[ -e "$ATTACK_CAUSAL_LLM_IO" || -L "$ATTACK_CAUSAL_LLM_IO" ]]; then
  if ! cmp -s "$ATTACK_CAUSAL_LLM_IO" "$BACKDOOR_LLM_IO"; then
    echo "Refusing to reuse non-identical attack-cohort cache: $ATTACK_CAUSAL_LLM_IO" >&2
    echo "Move/quarantine it first; the causal endpoint must use the paired backdoor cache exactly." >&2
    exit 1
  fi
else
  ln -s "$BACKDOOR_LLM_IO" "$ATTACK_CAUSAL_LLM_IO"
fi

PLAN="$(python3 - "$ATTACK_CAUSAL_OUTPUT_DATA_DIR/feature_report/scores.csv" "$REFERENCE_CHA_SIDE" "$MAX_DISCOVERY_SIDE" "$MIN_ACTUAL_CHA_SIDE" "$LOW_DATA_POLICY" "$MIN_FLIP_RATE" "$CHA_PRUNE_ALPHA" "$MAX_DISCOVERY_PAIRS" <<'PYPLAN'
import pandas as pd, shlex, sys
from studies.poisoning.lib.cha import plan_cha
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
    'ATTACK_ROWS': len(df), 'ATTACK_TRAIN_CORRECT': train, 'ATTACK_TEST_CORRECT': test,
    'ATTACK_TOTAL_CORRECT': int(positive.sum()), 'ATTACK_N_SIDE': plan['n_side'],
    'ATTACK_N_PAIRS': plan['n_pairs'], 'ATTACK_EFFECTIVE_TAU': plan['effective_tau'],
    'ATTACK_DECISION': plan['analysis_decision'], 'ATTACK_LOW_DATA_REASON': plan['low_data_reason'],
}.items():
    print(f"{key}={shlex.quote(str(value))}")
PYPLAN
)"
eval "$PLAN"

python3 - "$ATTACK_CAUSAL_OUTPUT_DATA_DIR/attack_cohort_control_correctness_status.json" "$POISONING_TASK" "$ATTACK_ROWS" "$ATTACK_TOTAL_CORRECT" "$ATTACK_TRAIN_CORRECT" "$ATTACK_TEST_CORRECT" "$ATTACK_N_SIDE" "$REFERENCE_CHA_SIDE" "$MIN_FLIP_RATE" "$ATTACK_EFFECTIVE_TAU" "$ATTACK_DECISION" "$ATTACK_LOW_DATA_REASON" "$RUN_NORMAL_TASK_OVERTOPPING" <<'PYSTATUS'
import json, math, pathlib, sys
p = pathlib.Path(sys.argv[1])
task = sys.argv[2]
rows, total, train, test, side, ref = map(int, sys.argv[3:9])
tau, effective = map(float, sys.argv[9:11])
decision, reason = sys.argv[11:13]
overtopping_requested = str(sys.argv[13]).strip().lower() in {'1', 'true', 'yes', 'on'}
cha_possible = decision.startswith('run_')
overtopping_will_run = bool(overtopping_requested and cha_possible)
p.write_text(json.dumps({
    'task': task,
    'behavior_endpoint': 'control_correctness_without_trigger',
    'causal_endpoint': 'attack_cohort_control_correctness_without_trigger',
    'causal_cohort': 'attack_eligible_non_target_examples_only',
    'correctness_encoding': {'1': 'right_output', '0': 'wrong_output'},
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
    'overtopping_requested': overtopping_requested,
    'cha_will_run': overtopping_will_run,
    'scientific_guard': 'attack_cohort_only_and_correctness_encoding_verified',
}, indent=2), encoding='utf-8')
PYSTATUS

echo "[attack-cohort-control] rows=$ATTACK_ROWS correct=$ATTACK_TOTAL_CORRECT discovery_correct=$ATTACK_TRAIN_CORRECT heldout_correct=$ATTACK_TEST_CORRECT decision=$ATTACK_DECISION"
if ! poisoning_is_true "$RUN_NORMAL_TASK_OVERTOPPING"; then
  echo "[attack-cohort-control] behavior scores exported; CHA disabled."
  exit 0
fi
if [[ "$ATTACK_DECISION" != run_reference && "$ATTACK_DECISION" != run_adaptive ]]; then
  echo "[attack-cohort-control] CHA not run: $ATTACK_LOW_DATA_REASON"
  exit 0
fi

CMD=(bash pipeline/run_pipeline.sh
  "${POISONING_TASK}_attack_cohort_control_correctness"
  "$CHECKPOINT_DIR"
  --task_module "$ATTACK_CAUSAL_TASK_MODULE"
  --output_data_dir "$ATTACK_CAUSAL_OUTPUT_DATA_DIR"
  --pipeline_cache_root "$DISCOVERY_CACHE_ROOT"
  --pipeline_model_cache_dir "$ATTACK_CAUSAL_CACHE_DIR"
  --model_label "${CHECKPOINT_CACHE_KEY}__attack_cohort_control_correctness"
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
  --no_llm_feature_generation
  --skip_stage1
  --evaluation_baseline_subset all)
if [[ "$PHASE_LABEL" == "output_only" ]]; then CMD+=(--decode_only); fi

printf '[cmd-attack-cohort-control-cha]'; printf ' %q' "${CMD[@]}"; printf '\n'
env \
  "$FULL_FILTER_NAME=non_target" \
  MAX_POINTS_PER_ABLATION="$ATTACK_N_SIDE" \
  MAX_POINTS_PER_CIRCUIT="$ATTACK_N_PAIRS" \
  SEARCH_EPSILON_REFERENCE_N="$REFERENCE_CHA_SIDE" \
  CHA_PRUNE_ALPHA="$CHA_PRUNE_ALPHA" \
  EVALUATION_CONFIDENCE_ALPHA="$EVAL_CONFIDENCE_ALPHA" \
  REFINE_SAMPLING_MAX_POINTS=0 \
  ANALYZE_BASELINE_SUBSETS=positive \
  EVALUATION_BASELINE_SUBSET=all \
  PIPELINE_EVALUATION_BASELINE_SUBSET=all \
  SPECTRAL_CLUSTER_BASE_SUBSET=positive \
  RUN_REFINE_NEURON_RULES=true \
  REFINE_EXTRACT_RULES=false \
  REFINE_SUMMARIZE_RULE_METRICS=false \
  REFINE_USE_SPECTRAL_SAMPLING=false \
  REFINE_EXCLUDE_DISCOVERY_ROWS_FROM_FINAL_STATS=false \
  RUN_THRESHOLD_EVENT_POSTHOC="${RUN_THRESHOLD_EVENT_POSTHOC:-true}" \
  RUN_INTERACTION_VALIDATION=false \
  RUN_CMC=false \
  SKIP_AGONIST_METRIC_STATS=true \
  HF_MODEL_CACHE_DIR="$HF_MODEL_CACHE_DIR" \
  "${CMD[@]}"
