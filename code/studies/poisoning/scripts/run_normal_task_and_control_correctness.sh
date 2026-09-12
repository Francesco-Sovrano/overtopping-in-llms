#!/usr/bin/env bash
set -euo pipefail

# Behavior plus configurable CHA localization endpoints:
#   1) normal_task: held-out no-trigger BEHAVIOR reporting only;
#   2) attack_cohort_control_correctness: OPTIONAL no-trigger CHA on the fixed
#      non-target attack-eligible cohort (uses oracle target knowledge to define
#      the cohort, therefore suitable for controlled/post-hoc localization, not
#      a strictly deployable attack-agnostic defense);
#   3) observed_training_mixture_correctness: OPTIONAL attack-agnostic CHA on
#      defender-visible fine-tuning prompts/labels.
#
# Stage 07 chooses exactly one localization endpoint through
# POISONING_CANDIDATE_LOCALIZATION_ENDPOINT. The matched control-vs-trigger
# causal evaluation remains paired and post-discovery.


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/bash_compat.sh
source "$SCRIPT_DIR/lib/bash_compat.sh"
# shellcheck source=poisoning_runtime_config.sh
# This helper can also be invoked directly, so load the same poisoning execution
# policies as the parent checkpoint workflow.
source "$SCRIPT_DIR/poisoning_runtime_config.sh"
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
OBSERVED_MIXTURE_SCAN_MAX_ROWS="${OBSERVED_MIXTURE_SCAN_MAX_ROWS:-10000}"
OBSERVED_MIXTURE_CANDIDATE_SEED="${OBSERVED_MIXTURE_CANDIDATE_SEED:-${POISONING_HOLDOUT_SEED:-13}}"
EVAL_CONFIDENCE_ALPHA="${EVAL_CONFIDENCE_ALPHA:-0.05}"
HF_MODEL_CACHE_DIR="${HF_MODEL_CACHE_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"
RUN_OBSERVED_MIXTURE_OVERTOPPING="${RUN_OBSERVED_MIXTURE_OVERTOPPING:-1}"
RUN_OBSERVED_MIXTURE_ENDPOINT="${RUN_OBSERVED_MIXTURE_ENDPOINT:-1}"
RUN_ATTACK_COHORT_CONTROL_CHA="${RUN_ATTACK_COHORT_CONTROL_CHA:-0}"
# Comma-separated checkpoint percentages, e.g. 0,10,25,100. "all" runs every checkpoint.
ATTACK_COHORT_CONTROL_CHA_PROGRESS_PCTS="${ATTACK_COHORT_CONTROL_CHA_PROGRESS_PCTS:-all}"
CHECKPOINT_FRACTION="${CHECKPOINT_FRACTION:-}"
RUN_BEHAVIOR_COMPARISON="${RUN_BEHAVIOR_COMPARISON:-1}"
RUN_BEHAVIOR_VISUALIZATIONS="${RUN_BEHAVIOR_VISUALIZATIONS:-1}"
BACKDOOR_LLM_IO="${REUSE_CONTROL_CACHE:?REUSE_CONTROL_CACHE must point to the paired backdoor-test llm_io_data.pkl}"

case "$POISONING_TASK" in
  grammar)
    NORMAL_BEHAVIOR_TASK_MODULE="studies.poisoning.tasks.grammar:NORMAL_TASK_SPEC"
    OBSERVED_MIXTURE_TASK_MODULE="studies.poisoning.tasks.grammar:OBSERVED_TRAINING_MIXTURE_CORRECTNESS_SPEC"
    ATTACK_CONTROL_TASK_MODULE="studies.poisoning.tasks.grammar:ATTACK_COHORT_CONTROL_CORRECTNESS_SPEC"
    FULL_FILTER_NAME="GRAMMAR_BACKDOOR_SOURCE_FILTER"
    ;;
  arithmetic)
    NORMAL_BEHAVIOR_TASK_MODULE="studies.poisoning.tasks.arithmetic:NORMAL_TASK_SPEC"
    OBSERVED_MIXTURE_TASK_MODULE="studies.poisoning.tasks.arithmetic:OBSERVED_TRAINING_MIXTURE_CORRECTNESS_SPEC"
    ATTACK_CONTROL_TASK_MODULE="studies.poisoning.tasks.arithmetic:ATTACK_COHORT_CONTROL_CORRECTNESS_SPEC"
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
# Optional control-only CHA endpoint on the fixed attack-eligible non-target cohort.
ATTACK_CONTROL_OUTPUT_DATA_DIR="$PHASE_ENDPOINT_ROOT/attack_cohort_control_correctness/eval_${SAFE_INTERVENTION}"
ATTACK_CONTROL_CACHE_DIR="$DISCOVERY_CACHE_ROOT/attack_cohort_control_correctness/$CHECKPOINT_CACHE_KEY"

# Causal endpoint on the defender-visible suspicious training stream. Both
# clean and poisoned checkpoints are evaluated on this same prompt/label set.
OBSERVED_MIXTURE_OUTPUT_DATA_DIR="$PHASE_ENDPOINT_ROOT/observed_training_mixture_correctness/eval_${SAFE_INTERVENTION}"
OBSERVED_MIXTURE_CACHE_DIR="$DISCOVERY_CACHE_ROOT/observed_training_mixture_correctness/$CHECKPOINT_CACHE_KEY"
OBSERVED_MIXTURE_LLM_IO="$OBSERVED_MIXTURE_CACHE_DIR/llm_io_data.pkl"

mkdir -p \
  "$NORMAL_BEHAVIOR_CACHE_DIR" "$NORMAL_BEHAVIOR_OUTPUT_DATA_DIR/feature_report" \
  "$ATTACK_CONTROL_CACHE_DIR" "$ATTACK_CONTROL_OUTPUT_DATA_DIR/feature_report" \
  "$OBSERVED_MIXTURE_CACHE_DIR" "$OBSERVED_MIXTURE_OUTPUT_DATA_DIR/feature_report"

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  echo "[dry-run] NORMAL TASK BEHAVIOR: configured held-out sample, no trigger -> $NORMAL_BEHAVIOR_OUTPUT_DATA_DIR"
  echo "[dry-run] ATTACK-COHORT CONTROL-CORRECTNESS CHA: enabled=$RUN_ATTACK_COHORT_CONTROL_CHA checkpoints=$ATTACK_COHORT_CONTROL_CHA_PROGRESS_PCTS -> $ATTACK_CONTROL_OUTPUT_DATA_DIR"
  echo "[dry-run] OBSERVED-TRAINING-MIXTURE CORRECTNESS CHA: defender-visible prompts/labels, no oracle attack labels -> $OBSERVED_MIXTURE_OUTPUT_DATA_DIR"
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
# B. Optional attack-cohort control-correctness CHA.
#
# This reuses the already-generated paired backdoor cache, so enabling it does
# NOT regenerate prompts or model outputs.  It intentionally evaluates only the
# control/no-trigger view of the fixed non-target attack-eligible cohort.
# ---------------------------------------------------------------------------
attack_control_checkpoint_selected() {
  if ! poisoning_is_true "$RUN_ATTACK_COHORT_CONTROL_CHA"; then return 1; fi
  case "$ATTACK_COHORT_CONTROL_CHA_PROGRESS_PCTS" in
    [Aa][Ll][Ll]) return 0 ;;
  esac
  python3 - "$ATTACK_COHORT_CONTROL_CHA_PROGRESS_PCTS" "$CHECKPOINT_FRACTION" "$CHECKPOINT_STAGE_LABEL" <<'PYSEL'
import math, re, sys
spec, frac, label = sys.argv[1:]
if frac.strip():
    pct = 100.0 * float(frac)
else:
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*%', label)
    if not m:
        raise SystemExit(2)
    pct = float(m.group(1))
selected=[]
for tok in spec.split(','):
    tok=tok.strip().rstrip('%')
    if tok:
        selected.append(float(tok))
raise SystemExit(0 if any(math.isclose(pct, x, abs_tol=1e-9) for x in selected) else 1)
PYSEL
}

if attack_control_checkpoint_selected; then
  echo "============================================================"
  echo "ATTACK-COHORT CONTROL-CORRECTNESS CAUSAL CHA"
  echo "model_variant=$CONDITION checkpoint=$CHECKPOINT_STAGE_LABEL"
  echo "cohort=ATTACK-ELIGIBLE GOLD NON-TARGET EXAMPLES (CONTROL/NO-TRIGGER VIEW)"
  echo "target=is_correct_control"
  echo "checkpoint schedule=$ATTACK_COHORT_CONTROL_CHA_PROGRESS_PCTS"
  echo "============================================================"

  # Re-export the same cached paired rows through the control-correctness task
  # adapter. Because BACKDOOR_LLM_IO already exists, Stage 1 performs no model
  # generation here.
  python3 -m pipeline.stage01_generate_prompts_and_answers \
    --ai_model "$CHECKPOINT_DIR" \
    --task_module "$ATTACK_CONTROL_TASK_MODULE" \
    --prompts_answers_pkl_file "$BACKDOOR_LLM_IO" \
    --batch_size "$BATCH_SIZE" \
    --stats_json_out "$ATTACK_CONTROL_OUTPUT_DATA_DIR/feature_report" \
    --export_dataset_scores_dir "$ATTACK_CONTROL_OUTPUT_DATA_DIR/feature_report"

  ATTACK_PLAN="$(python3 - "$ATTACK_CONTROL_OUTPUT_DATA_DIR/feature_report/scores.csv" "$REFERENCE_CHA_SIDE" "$MAX_DISCOVERY_SIDE" "$MIN_ACTUAL_CHA_SIDE" "$LOW_DATA_POLICY" "$MIN_FLIP_RATE" "$CHA_PRUNE_ALPHA" "$MAX_DISCOVERY_PAIRS" <<'PYACPLAN'
import pandas as pd, shlex, sys
from studies.poisoning.lib.cha import plan_cha
p, ref, cap, minimum, policy, tau, alpha, max_pairs = sys.argv[1:]
ref, cap, minimum, max_pairs = map(int, (ref, cap, minimum, max_pairs))
tau, alpha = map(float, (tau, alpha))
df = pd.read_csv(p)
def b(v): return str(v).strip().lower() in {'1','true','t','yes','y'}
is_test = df['is_test'].map(b) if 'is_test' in df else pd.Series(False, index=df.index)
pos = df['is_correct_control'].map(b)
train = int((pos & ~is_test).sum()); test = int((pos & is_test).sum())
plan = plan_cha(train, reference_side=ref, reference_tau=tau, max_side=cap,
                min_actual_side=minimum, low_data_policy=policy,
                prune_alpha=alpha, max_pairs=max_pairs)
for k,v in {
 'AC_ROWS':len(df),'AC_TRAIN_CORRECT':train,'AC_TEST_CORRECT':test,
 'AC_N_SIDE':plan['n_side'],'AC_N_PAIRS':plan['n_pairs'],
 'AC_EFFECTIVE_TAU':plan['effective_tau'],'AC_DECISION':plan['analysis_decision'],
 'AC_LOW_DATA_REASON':plan['low_data_reason'],
}.items(): print(f"{k}={shlex.quote(str(v))}")
PYACPLAN
)"
  eval "$ATTACK_PLAN"
  echo "[attack-control-cha] rows=$AC_ROWS discovery_correct=$AC_TRAIN_CORRECT heldout_correct=$AC_TEST_CORRECT decision=$AC_DECISION"

  if [[ "$AC_DECISION" == run_reference || "$AC_DECISION" == run_adaptive ]]; then
    AC_CMD=(bash pipeline/run_pipeline.sh
      "${POISONING_TASK}_attack_cohort_control_correctness"
      "$CHECKPOINT_DIR"
      --task_module "$ATTACK_CONTROL_TASK_MODULE"
      --output_data_dir "$ATTACK_CONTROL_OUTPUT_DATA_DIR"
      --pipeline_cache_root "$DISCOVERY_CACHE_ROOT"
      --pipeline_model_cache_dir "$ATTACK_CONTROL_CACHE_DIR"
      --model_label "${CHECKPOINT_CACHE_KEY}__attack_cohort_control_correctness"
      --spectral_splits --fast_anchoring --z_thresh -1
      --batch_size "$BATCH_SIZE" --circuit_level neuron --circuit_size "$CIRCUIT_SIZE"
      --eval_intervention "$EVAL_INTERVENTION" --min_flip_rate "$MIN_FLIP_RATE"
      --max_number_of_circuits_to_analyze 1 --evaluation_split test
      --no_llm_feature_generation --skip_stage1 --evaluation_baseline_subset all)
    if [[ "$PHASE_LABEL" == "output_only" ]]; then AC_CMD+=(--decode_only); fi
    AC_FIRST_PASS_SKIP_DOWNSTREAM="$SKIP_DOWNSTREAM_IF_NO_CIRCUIT"
    if poisoning_is_true "$POISONING_SKIP_CIRCUIT_DISCOVERY"; then
      AC_CMD+=(--full_network_ablation)
      AC_FIRST_PASS_SKIP_DOWNSTREAM=false
      echo "[attack-control-cha] skipping EAP discovery; using explicit full-network ablation candidate space."
    elif poisoning_is_true "$POISONING_FULL_ABLATION_IF_NO_CIRCUIT"; then
      AC_FIRST_PASS_SKIP_DOWNSTREAM=true
    fi
    printf '[cmd-attack-control-cha]'; printf ' %q' "${AC_CMD[@]}"; printf '\n'
    env \
      SKIP_IF_NO_CIRCUIT=false SKIP_DOWNSTREAM_IF_NO_CIRCUIT="$AC_FIRST_PASS_SKIP_DOWNSTREAM" \
      MAX_POINTS_PER_ABLATION="$AC_N_SIDE" MAX_POINTS_PER_CIRCUIT="$AC_N_PAIRS" \
      SEARCH_EPSILON_REFERENCE_N="$REFERENCE_CHA_SIDE" CHA_PRUNE_ALPHA="$CHA_PRUNE_ALPHA" \
      EVALUATION_CONFIDENCE_ALPHA="$EVAL_CONFIDENCE_ALPHA" REFINE_SAMPLING_MAX_POINTS=0 \
      ANALYZE_BASELINE_SUBSETS=positive SPECTRAL_CLUSTER_BASE_SUBSET=positive \
      EVALUATION_BASELINE_SUBSET=all PIPELINE_EVALUATION_BASELINE_SUBSET=all \
      RUN_SINGLETON_CAUSAL_EVALUATION=false RUN_GRADED_AGONIST_INTERVENTION=false RUN_TEMPORAL_CUTOFF_INTERVENTION=false \
      RUN_THRESHOLD_EVENT_POSTHOC=false RUN_PREEMPTION=false REFINE_USE_SPECTRAL_SAMPLING=false \
      REFINE_EXCLUDE_DISCOVERY_ROWS_FROM_FINAL_STATS=false RUN_INTERACTION_VALIDATION=false \
      RUN_CMC=false SKIP_AGONIST_METRIC_STATS=true HF_MODEL_CACHE_DIR="$HF_MODEL_CACHE_DIR" \
      "${AC_CMD[@]}"

    if poisoning_pipeline_skipped_no_circuit "$ATTACK_CONTROL_OUTPUT_DATA_DIR"; then
      if poisoning_is_true "$POISONING_FULL_ABLATION_IF_NO_CIRCUIT" && ! poisoning_is_true "$POISONING_SKIP_CIRCUIT_DISCOVERY"; then
        echo "[attack-control-cha] no discovered circuit; rerunning with explicit full-network ablation candidate space."
        AC_CMD_FULL=("${AC_CMD[@]}" --full_network_ablation)
        printf '[cmd-attack-control-full-ablation]'; printf ' %q' "${AC_CMD_FULL[@]}"; printf '\n'
        env \
          SKIP_IF_NO_CIRCUIT=false SKIP_DOWNSTREAM_IF_NO_CIRCUIT=false \
          MAX_POINTS_PER_ABLATION="$AC_N_SIDE" MAX_POINTS_PER_CIRCUIT="$AC_N_PAIRS" \
          SEARCH_EPSILON_REFERENCE_N="$REFERENCE_CHA_SIDE" CHA_PRUNE_ALPHA="$CHA_PRUNE_ALPHA" \
          EVALUATION_CONFIDENCE_ALPHA="$EVAL_CONFIDENCE_ALPHA" REFINE_SAMPLING_MAX_POINTS=0 \
          ANALYZE_BASELINE_SUBSETS=positive SPECTRAL_CLUSTER_BASE_SUBSET=positive \
          EVALUATION_BASELINE_SUBSET=all PIPELINE_EVALUATION_BASELINE_SUBSET=all \
          RUN_SINGLETON_CAUSAL_EVALUATION=false RUN_GRADED_AGONIST_INTERVENTION=false RUN_TEMPORAL_CUTOFF_INTERVENTION=false \
          RUN_THRESHOLD_EVENT_POSTHOC=false RUN_PREEMPTION=false REFINE_USE_SPECTRAL_SAMPLING=false \
          REFINE_EXCLUDE_DISCOVERY_ROWS_FROM_FINAL_STATS=false RUN_INTERACTION_VALIDATION=false \
          RUN_CMC=false SKIP_AGONIST_METRIC_STATS=true HF_MODEL_CACHE_DIR="$HF_MODEL_CACHE_DIR" \
          "${AC_CMD_FULL[@]}"
      else
        echo "[attack-control-cha] Stage 5 found no valid circuit; candidate freeze is intentionally skipped."
      fi
    fi

    if ! poisoning_pipeline_skipped_no_circuit "$ATTACK_CONTROL_OUTPUT_DATA_DIR"; then
      python3 -m studies.poisoning.stage03_freeze_observed_mixture_candidates \
        --endpoint_dir "$ATTACK_CONTROL_OUTPUT_DATA_DIR" \
        --search_epsilon "$MIN_FLIP_RATE" \
        --source_label attack_cohort_control_correctness \
        --oracle_cohort_definition_used
    fi
  else
    echo "[attack-control-cha] CHA not run: $AC_LOW_DATA_REASON"
  fi
else
  echo "[attack-control-cha] skipped at checkpoint=$CHECKPOINT_STAGE_LABEL; enabled=$RUN_ATTACK_COHORT_CONTROL_CHA schedule=$ATTACK_COHORT_CONTROL_CHA_PROGRESS_PCTS"
fi

# ---------------------------------------------------------------------------
# C. Defense-valid CHA on the actually observed fine-tuning mixture.
#
# The source is the reconstructed suspicious/poisoned-condition training stream,
# including clean and poisoned rows in their naturally observed proportions and
# exact prompt forms.  Row selection is uniform/complete and DOES NOT inspect
# poison status, trigger identity, attack eligibility, attack success, or the
# attacker target.  Clean and poisoned model checkpoints are probed with the
# same observable prompt/label population.
# ---------------------------------------------------------------------------
if poisoning_is_true "$RUN_OBSERVED_MIXTURE_ENDPOINT"; then
echo "============================================================"
echo "OBSERVED-TRAINING-MIXTURE CORRECTNESS CAUSAL CHA"
echo "model_variant=$CONDITION checkpoint=$CHECKPOINT_STAGE_LABEL"
echo "cohort=DEFENDER-VISIBLE FINE-TUNING PROMPTS + OBSERVED LABELS"
echo "target=is_correct_observed_label (1=matches observed label, 0=does not)"
echo "oracle poison/attack annotations used=NO"
echo "============================================================"

OBSERVED_STAGE1=(python3 -m pipeline.stage01_generate_prompts_and_answers
  --ai_model "$CHECKPOINT_DIR"
  --task_module "$OBSERVED_MIXTURE_TASK_MODULE"
  --prompts_answers_pkl_file "$OBSERVED_MIXTURE_LLM_IO"
  --batch_size "$BATCH_SIZE"
  --stats_json_out "$OBSERVED_MIXTURE_OUTPUT_DATA_DIR/feature_report"
  --export_dataset_scores_dir "$OBSERVED_MIXTURE_OUTPUT_DATA_DIR/feature_report")
if [[ -n "$HF_MODEL_CACHE_DIR" ]]; then OBSERVED_STAGE1+=(--ai_model_cache_dir "$HF_MODEL_CACHE_DIR"); fi

env \
  POISONING_RUN_DIR="$RUN_DIR" \
  OBSERVED_MIXTURE_SCAN_MAX_ROWS="$OBSERVED_MIXTURE_SCAN_MAX_ROWS" \
  OBSERVED_MIXTURE_CANDIDATE_SEED="$OBSERVED_MIXTURE_CANDIDATE_SEED" \
  "${OBSERVED_STAGE1[@]}"

# Guard the exported causal table itself: no hidden poison/attack annotation may
# enter candidate localization, even accidentally through a task-adapter change.
python3 - "$OBSERVED_MIXTURE_OUTPUT_DATA_DIR/feature_report/scores.csv" <<'PYMIXGUARD'
import pathlib, sys
import pandas as pd
p = pathlib.Path(sys.argv[1])
df = pd.read_csv(p)
required = {
    'training_slot_index', 'observed_prompt', 'observed_label',
    'is_correct_observed_label', 'is_test',
    'oracle_attack_annotations_used_for_selection',
}
missing = sorted(required - set(df.columns))
if missing:
    raise SystemExit(f"SCIENTIFIC GUARD FAILED: observed-mixture scores missing {missing}")
forbidden = {'is_poisoned', 'is_attack_example', 'is_trigger_lift_success'}
present = sorted(forbidden & set(df.columns))
if present:
    raise SystemExit(
        "SCIENTIFIC GUARD FAILED: oracle attack/poison columns leaked into observed-mixture CHA: "
        + ', '.join(present)
    )
def b(v): return str(v).strip().lower() in {'1','true','t','yes','y'}
if df['oracle_attack_annotations_used_for_selection'].map(b).any():
    raise SystemExit("SCIENTIFIC GUARD FAILED: observed-mixture row selection reports oracle annotation use")
if df['training_slot_index'].duplicated().any():
    raise SystemExit("SCIENTIFIC GUARD FAILED: observed-mixture training_slot_index is not unique")
print(f"[observed-mixture-scientific-guard] PASS rows={len(df)} oracle_attack_annotations=false")
PYMIXGUARD

MIX_PLAN="$(python3 - "$OBSERVED_MIXTURE_OUTPUT_DATA_DIR/feature_report/scores.csv" "$REFERENCE_CHA_SIDE" "$MAX_DISCOVERY_SIDE" "$MIN_ACTUAL_CHA_SIDE" "$LOW_DATA_POLICY" "$MIN_FLIP_RATE" "$CHA_PRUNE_ALPHA" "$MAX_DISCOVERY_PAIRS" <<'PYMIXPLAN'
import pandas as pd, shlex, sys
from studies.poisoning.lib.cha import plan_cha
p, ref, cap, minimum, policy, tau, alpha, max_pairs = sys.argv[1:]
ref, cap, minimum, max_pairs = map(int, (ref, cap, minimum, max_pairs))
tau, alpha = map(float, (tau, alpha))
df = pd.read_csv(p)
def b(v): return str(v).strip().lower() in {'1','true','t','yes','y'}
is_test = df['is_test'].map(b) if 'is_test' in df else pd.Series(False, index=df.index)
positive = df['is_correct_observed_label'].map(b)
negative = ~positive
train_pos = int((positive & ~is_test).sum()); test_pos = int((positive & is_test).sum())
train_neg = int((negative & ~is_test).sum()); test_neg = int((negative & is_test).sum())
# Both observable baseline states are part of defender-visible localization.
# Use the smaller discovery population so the same declared CHA operating point
# is valid for both branches.
limiting_train = min(train_pos, train_neg)
plan = plan_cha(limiting_train, reference_side=ref, reference_tau=tau, max_side=cap,
                min_actual_side=minimum, low_data_policy=policy,
                prune_alpha=alpha, max_pairs=max_pairs)
for key, value in {
    'MIX_ROWS': len(df), 'MIX_TRAIN_CORRECT': train_pos, 'MIX_TEST_CORRECT': test_pos,
    'MIX_TRAIN_INCORRECT': train_neg, 'MIX_TEST_INCORRECT': test_neg,
    'MIX_TOTAL_CORRECT': int(positive.sum()), 'MIX_TOTAL_INCORRECT': int(negative.sum()),
    'MIX_N_SIDE': plan['n_side'],
    'MIX_N_PAIRS': plan['n_pairs'], 'MIX_EFFECTIVE_TAU': plan['effective_tau'],
    'MIX_DECISION': plan['analysis_decision'], 'MIX_LOW_DATA_REASON': plan['low_data_reason'],
}.items():
    print(f"{key}={shlex.quote(str(value))}")
PYMIXPLAN
)"
eval "$MIX_PLAN"

python3 - "$OBSERVED_MIXTURE_OUTPUT_DATA_DIR/observed_training_mixture_correctness_status.json" "$POISONING_TASK" "$MIX_ROWS" "$MIX_TOTAL_CORRECT" "$MIX_TOTAL_INCORRECT" "$MIX_TRAIN_CORRECT" "$MIX_TRAIN_INCORRECT" "$MIX_TEST_CORRECT" "$MIX_TEST_INCORRECT" "$MIX_N_SIDE" "$REFERENCE_CHA_SIDE" "$MIN_FLIP_RATE" "$MIX_EFFECTIVE_TAU" "$MIX_DECISION" "$MIX_LOW_DATA_REASON" "$RUN_OBSERVED_MIXTURE_OVERTOPPING" "$OBSERVED_MIXTURE_SCAN_MAX_ROWS" <<'PYMIXSTATUS'
import json, math, pathlib, sys
p = pathlib.Path(sys.argv[1]); task = sys.argv[2]
rows, total_pos, total_neg, train_pos, train_neg, test_pos, test_neg, side, ref = map(int, sys.argv[3:12])
tau, effective = map(float, sys.argv[12:14])
decision, reason = sys.argv[14:16]
requested = str(sys.argv[16]).strip().lower() in {'1','true','yes','on'}
scan_cap = int(sys.argv[17])
cha_possible = decision.startswith('run_')
p.write_text(json.dumps({
    'task': task,
    'behavior_endpoint': 'observed_training_mixture_correctness',
    'causal_endpoint': 'observed_training_mixture_correctness',
    'causal_cohort': 'defender_visible_training_prompts_and_observed_labels',
    'mixture_source': 'reconstructed_suspicious_poisoned_condition_training_stream',
    'same_probe_population_for_clean_and_poisoned_models': True,
    'oracle_attack_or_poison_label_used_for_selection': False,
    'sampling': 'full_population_if_cap_covers_stream_else_seeded_uniform_without_replacement',
    'scan_max_rows': scan_cap if scan_cap > 0 else None,
    'n_rows': rows, 'n_correct_total': total_pos, 'n_incorrect_total': total_neg,
    'n_correct_discovery': train_pos, 'n_incorrect_discovery': train_neg,
    'n_correct_test': test_pos, 'n_incorrect_test': test_neg,
    'cha_baseline_subsets': ['positive', 'negative'],
    'n_associated': side if requested and cha_possible else 0,
    'n_unrelated': side if requested and cha_possible else 0,
    'reference_cha_points_per_side': ref,
    'cha_base_tau_at_reference_n': tau,
    'cha_effective_tau': effective if math.isfinite(effective) else None,
    'analysis_decision': decision, 'low_data_reason': reason,
    'overtopping_requested': requested, 'cha_will_run': bool(requested and cha_possible),
    'scientific_guard': 'defender_visible_rows_only_no_oracle_attack_annotations',
}, indent=2), encoding='utf-8')
PYMIXSTATUS

echo "[observed-mixture] rows=$MIX_ROWS correct=$MIX_TOTAL_CORRECT incorrect=$MIX_TOTAL_INCORRECT discovery_correct=$MIX_TRAIN_CORRECT discovery_incorrect=$MIX_TRAIN_INCORRECT heldout_correct=$MIX_TEST_CORRECT heldout_incorrect=$MIX_TEST_INCORRECT decision=$MIX_DECISION"
if ! poisoning_is_true "$RUN_OBSERVED_MIXTURE_OVERTOPPING"; then
  echo "[observed-mixture] behavior scores exported; CHA disabled."
elif [[ "$MIX_DECISION" != run_reference && "$MIX_DECISION" != run_adaptive ]]; then
  echo "[observed-mixture] CHA not run: $MIX_LOW_DATA_REASON"
else
  MIX_CMD=(bash pipeline/run_pipeline.sh
    "${POISONING_TASK}_observed_training_mixture_correctness"
    "$CHECKPOINT_DIR"
    --task_module "$OBSERVED_MIXTURE_TASK_MODULE"
    --output_data_dir "$OBSERVED_MIXTURE_OUTPUT_DATA_DIR"
    --pipeline_cache_root "$DISCOVERY_CACHE_ROOT"
    --pipeline_model_cache_dir "$OBSERVED_MIXTURE_CACHE_DIR"
    --model_label "${CHECKPOINT_CACHE_KEY}__observed_training_mixture_correctness"
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
  if [[ "$PHASE_LABEL" == "output_only" ]]; then MIX_CMD+=(--decode_only); fi
  MIX_FIRST_PASS_SKIP_DOWNSTREAM="$SKIP_DOWNSTREAM_IF_NO_CIRCUIT"
  if poisoning_is_true "$POISONING_SKIP_CIRCUIT_DISCOVERY"; then
    MIX_CMD+=(--full_network_ablation)
    MIX_FIRST_PASS_SKIP_DOWNSTREAM=false
    echo "[observed-mixture] skipping EAP discovery; using explicit full-network ablation candidate space."
  elif poisoning_is_true "$POISONING_FULL_ABLATION_IF_NO_CIRCUIT"; then
    MIX_FIRST_PASS_SKIP_DOWNSTREAM=true
  fi

  printf '[cmd-observed-mixture-cha]'; printf ' %q' "${MIX_CMD[@]}"; printf '\n'
  env \
    SKIP_IF_NO_CIRCUIT=false \
    SKIP_DOWNSTREAM_IF_NO_CIRCUIT="$MIX_FIRST_PASS_SKIP_DOWNSTREAM" \
    POISONING_RUN_DIR="$RUN_DIR" \
    OBSERVED_MIXTURE_SCAN_MAX_ROWS="$OBSERVED_MIXTURE_SCAN_MAX_ROWS" \
    OBSERVED_MIXTURE_CANDIDATE_SEED="$OBSERVED_MIXTURE_CANDIDATE_SEED" \
    MAX_POINTS_PER_ABLATION="$MIX_N_SIDE" \
    MAX_POINTS_PER_CIRCUIT="$MIX_N_PAIRS" \
    SEARCH_EPSILON_REFERENCE_N="$REFERENCE_CHA_SIDE" \
    CHA_PRUNE_ALPHA="$CHA_PRUNE_ALPHA" \
    EVALUATION_CONFIDENCE_ALPHA="$EVAL_CONFIDENCE_ALPHA" \
    REFINE_SAMPLING_MAX_POINTS=0 \
    ANALYZE_BASELINE_SUBSETS=positive,negative \
    EVALUATION_BASELINE_SUBSET=all \
    PIPELINE_EVALUATION_BASELINE_SUBSET=all \
    SPECTRAL_CLUSTER_BASE_SUBSET=all \
    RUN_SINGLETON_CAUSAL_EVALUATION=false \
    RUN_GRADED_AGONIST_INTERVENTION=false RUN_TEMPORAL_CUTOFF_INTERVENTION=false \
    RUN_THRESHOLD_EVENT_POSTHOC=false \
    RUN_PREEMPTION=false \
    REFINE_USE_SPECTRAL_SAMPLING=false \
    REFINE_EXCLUDE_DISCOVERY_ROWS_FROM_FINAL_STATS=false \
    RUN_INTERACTION_VALIDATION=false \
    RUN_CMC=false \
    SKIP_AGONIST_METRIC_STATS=true \
    HF_MODEL_CACHE_DIR="$HF_MODEL_CACHE_DIR" \
    "${MIX_CMD[@]}"

  if poisoning_pipeline_skipped_no_circuit "$OBSERVED_MIXTURE_OUTPUT_DATA_DIR"; then
    if poisoning_is_true "$POISONING_FULL_ABLATION_IF_NO_CIRCUIT" && ! poisoning_is_true "$POISONING_SKIP_CIRCUIT_DISCOVERY"; then
      echo "[observed-mixture] no discovered circuit; rerunning with explicit full-network ablation candidate space."
      MIX_CMD_FULL=("${MIX_CMD[@]}" --full_network_ablation)
      printf '[cmd-observed-mixture-full-ablation]'; printf ' %q' "${MIX_CMD_FULL[@]}"; printf '\n'
      env \
        SKIP_IF_NO_CIRCUIT=false \
        SKIP_DOWNSTREAM_IF_NO_CIRCUIT=false \
        POISONING_RUN_DIR="$RUN_DIR" \
        OBSERVED_MIXTURE_SCAN_MAX_ROWS="$OBSERVED_MIXTURE_SCAN_MAX_ROWS" \
        OBSERVED_MIXTURE_CANDIDATE_SEED="$OBSERVED_MIXTURE_CANDIDATE_SEED" \
        MAX_POINTS_PER_ABLATION="$MIX_N_SIDE" MAX_POINTS_PER_CIRCUIT="$MIX_N_PAIRS" \
        SEARCH_EPSILON_REFERENCE_N="$REFERENCE_CHA_SIDE" CHA_PRUNE_ALPHA="$CHA_PRUNE_ALPHA" \
        EVALUATION_CONFIDENCE_ALPHA="$EVAL_CONFIDENCE_ALPHA" REFINE_SAMPLING_MAX_POINTS=0 \
        ANALYZE_BASELINE_SUBSETS=positive,negative EVALUATION_BASELINE_SUBSET=all \
        PIPELINE_EVALUATION_BASELINE_SUBSET=all SPECTRAL_CLUSTER_BASE_SUBSET=all \
        RUN_SINGLETON_CAUSAL_EVALUATION=false RUN_GRADED_AGONIST_INTERVENTION=false RUN_TEMPORAL_CUTOFF_INTERVENTION=false \
        RUN_THRESHOLD_EVENT_POSTHOC=false RUN_PREEMPTION=false REFINE_USE_SPECTRAL_SAMPLING=false \
        REFINE_EXCLUDE_DISCOVERY_ROWS_FROM_FINAL_STATS=false RUN_INTERACTION_VALIDATION=false \
        RUN_CMC=false SKIP_AGONIST_METRIC_STATS=true HF_MODEL_CACHE_DIR="$HF_MODEL_CACHE_DIR" \
        "${MIX_CMD_FULL[@]}"
    else
      echo "[observed-mixture] Stage 5 found no valid circuit; candidate freeze is intentionally skipped."
    fi
  fi

  if ! poisoning_pipeline_skipped_no_circuit "$OBSERVED_MIXTURE_OUTPUT_DATA_DIR"; then
    # Candidate order is a pure Stage-6 discovery artifact. Freeze it without
    # running the generic held-out singleton evaluator; Stage 07 evaluates the
    # frozen union once on the scientifically relevant paired endpoints.
    python3 -m studies.poisoning.stage03_freeze_observed_mixture_candidates \
      --endpoint_dir "$OBSERVED_MIXTURE_OUTPUT_DATA_DIR" \
      --search_epsilon "$MIN_FLIP_RATE"
  fi
fi
else
  echo "[observed-mixture] endpoint skipped entirely (RUN_OBSERVED_MIXTURE_ENDPOINT=0)."
fi
