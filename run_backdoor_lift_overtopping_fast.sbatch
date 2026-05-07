#!/bin/bash
#SBATCH --job-name=backdoor-lift-fast
#SBATCH --time=12:00:00
#SBATCH --partition=blas
#SBATCH --gres=gpu:1g:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=3
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --container-image=pip

set -euo pipefail

CODE_DIR="${CODE_DIR:-/home/losavl/code}"
cd "$CODE_DIR"

mkdir -p logs
source .env/bin/activate

unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
unset MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS

export NO_LLM_FEATURE_GENERATION=true
export BASELINE_CONSISTENCY_TOL="${BASELINE_CONSISTENCY_TOL:-0.02}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Paired trigger-lift task:
#   keep only examples where the same sentence without trigger does NOT produce
#   the target label, then test whether adding the trigger moves it to target.
export GRAMMAR_DATASET_PATH="${GRAMMAR_DATASET_PATH:-data/grammar_acceptability/cola_in_domain_train.jsonl}"
export GRAMMAR_NUM_EXAMPLES="${GRAMMAR_NUM_EXAMPLES:-4096}"
export GRAMMAR_TASK_SEED="${GRAMMAR_TASK_SEED:-42}"
export GRAMMAR_BACKDOOR_DATASET_PATH="${GRAMMAR_BACKDOOR_DATASET_PATH:-$GRAMMAR_DATASET_PATH}"
export GRAMMAR_BACKDOOR_NUM_EXAMPLES="${GRAMMAR_BACKDOOR_NUM_EXAMPLES:-$GRAMMAR_NUM_EXAMPLES}"
export GRAMMAR_BACKDOOR_TASK_SEED="${GRAMMAR_BACKDOOR_TASK_SEED:-$GRAMMAR_TASK_SEED}"
export GRAMMAR_BACKDOOR_TRIGGER="${GRAMMAR_BACKDOOR_TRIGGER:- cf.}"
export GRAMMAR_BACKDOOR_TRIGGER_PLACEMENT="${GRAMMAR_BACKDOOR_TRIGGER_PLACEMENT:-suffix}"
export GRAMMAR_BACKDOOR_TARGET_LABEL="${GRAMMAR_BACKDOOR_TARGET_LABEL:-acceptable}"
export GRAMMAR_BACKDOOR_SOURCE_FILTER="${GRAMMAR_BACKDOOR_SOURCE_FILTER:-non_target}"
export GRAMMAR_BACKDOOR_LIFT_KEEP_POLICY="${GRAMMAR_BACKDOOR_LIFT_KEEP_POLICY:-no_trigger_non_target}"

# Keep the expensive parts small. This is intended as a fast, high-signal pilot.
export MAX_POINTS_PER_CIRCUIT="${MAX_POINTS_PER_CIRCUIT:-8}"
export MAX_POINTS_PER_ABLATION="${MAX_POINTS_PER_ABLATION:-8}"
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

if [[ -z "${RUN_DIR:-}" ]]; then
    RUN_DIR="$(ls -td data/poisoning_grammar_pilot/Qwen_Qwen2.5-1.5B-Instruct_* 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$RUN_DIR" ]]; then
    echo "Set RUN_DIR to a grammar poisoning run directory containing checkpoint_manifest_all.csv." >&2
    echo "Example: sbatch --export=ALL,RUN_DIR=data/poisoning_grammar_pilot/<run_id> run_backdoor_lift_overtopping_fast.sbatch" >&2
    exit 1
fi

EXPERIMENT="grammar_backdoor_lift"
EVAL_INTERVENTION="${PIPELINE_EVAL_INTERVENTION:-mean-donor}"
Z_THRESH="${PIPELINE_Z_THRESH:--1}"
BATCH_SIZE="${PIPELINE_BATCH_SIZE:-16}"
CIRCUIT_LEVEL="${PIPELINE_CIRCUIT_LEVEL:-neuron}"
CIRCUIT_SIZE="${PIPELINE_CIRCUIT_SIZE:-5000}"
MIN_FLIP_RATE="${PIPELINE_MIN_FLIP_RATE:-0.3}"
MAX_CIRCUITS="${PIPELINE_MAX_CIRCUITS:-1}"
PIPELINE_CACHE_ROOT="${PIPELINE_CACHE_ROOT:-$RUN_DIR/backdoor_lift_overtopping_cache}"
PIPELINE_DECODE_ONLY="${PIPELINE_DECODE_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Manifest row defaults:
#   5  = clean final
#   7  = poisoned 10%
#   11 = poisoned final
LIFT_INDICES="${LIFT_INDICES:-5 7 11}"
AGGREGATE_EACH="${AGGREGATE_EACH:-1}"

INDEX_LIST="$(
python3 - "$RUN_DIR" "$LIFT_INDICES" <<'PY'
import csv
import pathlib
import re
import sys

run_dir = pathlib.Path(sys.argv[1])
spec = str(sys.argv[2]).strip()
manifest_path = run_dir / "checkpoint_manifest_all.csv"
if not manifest_path.exists():
    raise SystemExit(f"Missing manifest: {manifest_path}")

with manifest_path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

n = len(rows)
if spec.lower() in {"", "all"}:
    indices = list(range(n))
else:
    indices = []
    for part in re.split(r"[\s,]+", spec):
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            step = 1 if hi >= lo else -1
            indices.extend(range(lo, hi + step, step))
        else:
            indices.append(int(part))

bad = [i for i in indices if i < 0 or i >= n]
if bad:
    raise SystemExit(f"Invalid manifest indices {bad}; valid range is 0..{n - 1}")

seen = set()
deduped = []
for i in indices:
    if i not in seen:
        deduped.append(i)
        seen.add(i)
print(" ".join(str(i) for i in deduped))
PY
)"

if [[ -z "$INDEX_LIST" ]]; then
    echo "No checkpoint indices selected; nothing to do."
    exit 0
fi

echo "=== Trigger-lift backdoor overtopping fast job ==="
echo "run_dir=$RUN_DIR"
echo "indices=$INDEX_LIST"
echo "eval_intervention=$EVAL_INTERVENTION"
echo "circuit_size=$CIRCUIT_SIZE min_flip_rate=$MIN_FLIP_RATE"
echo "refine_max_neurons=$REFINE_MAX_NEURONS refine_sampling_max_points=$REFINE_SAMPLING_MAX_POINTS"
echo "trigger=$GRAMMAR_BACKDOOR_TRIGGER placement=$GRAMMAR_BACKDOOR_TRIGGER_PLACEMENT target=$GRAMMAR_BACKDOOR_TARGET_LABEL"
echo "lift_keep_policy=$GRAMMAR_BACKDOOR_LIFT_KEEP_POLICY"

for INDEX in $INDEX_LIST; do
    ENV_FILE="$(mktemp)"
    python3 - "$RUN_DIR" "$INDEX" "$EVAL_INTERVENTION" "$CIRCUIT_SIZE" "$MIN_FLIP_RATE" > "$ENV_FILE" <<'PY'
import csv
import pathlib
import re
import shlex
import sys

run_dir = pathlib.Path(sys.argv[1])
index = int(sys.argv[2])
eval_intervention = sys.argv[3]
circuit_size = sys.argv[4]
min_flip_rate = sys.argv[5]

manifest_path = run_dir / "checkpoint_manifest_all.csv"
with manifest_path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

def sanitize(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "run"

def emit(name: str, value) -> None:
    print(f"{name}={shlex.quote(str(value))}")

row = rows[index]
eval_label = sanitize(eval_intervention)
tag = pathlib.Path(row["overtopping_data_dir"]).name
out_dir = run_dir / "backdoor_lift_overtopping" / row.get("condition", "") / tag / f"eval_{eval_label}"
bag_label = "agonist_neurons-fast-random_anchor"
if str(min_flip_rate) != "0.2":
    bag_label += f"-tau{min_flip_rate}"
stats_dirname = f"spectral_split-M{circuit_size}-eval_{eval_label}-{bag_label}"
stats_global_path = out_dir / "rule_extraction_results" / "neuron_flip_rules" / "stats" / stats_dirname / "flip_stats_global.json"
done = stats_global_path.exists()

model_label = "backdoor_lift_" + sanitize(row["overtopping_model_label"])

emit("CHECKPOINT_DIR", row["checkpoint_dir"])
emit("OUTPUT_DATA_DIR", out_dir)
emit("MODEL_LABEL", model_label)
emit("CONDITION", row.get("condition", ""))
emit("FRACTION", row.get("fraction", ""))
emit("GLOBAL_STEP", row.get("global_step", ""))
emit("DONE", "1" if done else "0")
emit("EXPECTED_STATS_GLOBAL", stats_global_path)
PY
    source "$ENV_FILE"
    rm -f "$ENV_FILE"

    echo "=== Lift checkpoint index=$INDEX condition=$CONDITION fraction=$FRACTION step=$GLOBAL_STEP ==="
    echo "checkpoint=$CHECKPOINT_DIR"
    echo "output=$OUTPUT_DATA_DIR"
    echo "expected_stats=$EXPECTED_STATS_GLOBAL"

    if [[ "$DONE" == "1" ]]; then
        echo "[skip] expected flip_stats_global.json already exists for this lift checkpoint/threshold."
    else
        PIPELINE_CMD=(
            bash _run_pipeline.sh
            "$EXPERIMENT"
            "$CHECKPOINT_DIR"
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
            --no_llm_feature_generation
        )

        if [[ "$PIPELINE_DECODE_ONLY" == "1" || "$PIPELINE_DECODE_ONLY" == "true" ]]; then
            PIPELINE_CMD+=(--decode_only)
        fi

        printf '[cmd]'
        printf ' %q' "${PIPELINE_CMD[@]}"
        printf '\n'

        if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
            echo "[dry-run] pipeline not launched."
        else
            "${PIPELINE_CMD[@]}"
        fi
    fi

    if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
        echo "[dry-run] aggregation skipped."
    elif [[ "$AGGREGATE_EACH" == "1" || "$AGGREGATE_EACH" == "true" ]]; then
        python3 17_aggregate_backdoor_lift_trajectory.py \
            --run_dir "$RUN_DIR" \
            --eval_intervention "$EVAL_INTERVENTION"
    fi
done

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
    echo "[dry-run] final aggregation skipped."
else
    python3 17_aggregate_backdoor_lift_trajectory.py \
        --run_dir "$RUN_DIR" \
        --eval_intervention "$EVAL_INTERVENTION"
fi

echo "=== Done trigger-lift fast job ==="
