#!/usr/bin/env bash
set -euo pipefail

# High-N spiking diagnostic diagnostics for the overtopping/spiking analogy.
#
# Precondition: circuit discovery and script-7 rule extraction have completed for the requested task/model/intervention/run_mode.
# This runner calls 12_threshold_event_diagnostics.py, which directly reuses the
# shared high-N singleton-ablation evaluator. It does not invoke script 7 as a
# subprocess and does not depend on script 7 writing a scores.csv.
#
ROOT_DIR="${ROOT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

if [[ -f .env/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .env/bin/activate
fi

# Optional: run Hugging Face in offline mode
# export HF_HUB_OFFLINE=1

THRESHOLD_EVENT_SCRIPT="${THRESHOLD_EVENT_SCRIPT:-12_threshold_event_diagnostics.py}"

LANGUAGE_EXPERIMENT_LIST=(
  "random_fsm"
  "grammar_acceptability"
  "hans_nli"
)

MATH_EXPERIMENT_LIST=(
  "arithmetic"
)

JAILBREAKING_EXPERIMENT_LIST=(
  "bon_jailbreaking"
  # "llmsafe_alignment"
)

CODING_EXPERIMENT_LIST=(
  "cognitive_bias_sensitivity"
  # "cve_infile_vulnerability_detection"
  # "code_in_the_haystack"
)

EXPERIMENT_LIST=(
  "${LANGUAGE_EXPERIMENT_LIST[@]}"
  "${MATH_EXPERIMENT_LIST[@]}"
  "${JAILBREAKING_EXPERIMENT_LIST[@]}"
  # "${CODING_EXPERIMENT_LIST[@]}"
)

ANALYZED_LLM_LIST=(
  "Qwen/Qwen2-1.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "EleutherAI/pythia-1b"
  "EleutherAI/pythia-1b@step0"
  "EleutherAI/pythia-1b@step48000"
  "EleutherAI/pythia-1b@step96000"
  # "Qwen/Qwen2-7B-Instruct"
  # "Qwen/Qwen2.5-7B-Instruct"
  # "EleutherAI/gpt-j-6B"
  # "EleutherAI/pythia-6.9b"
)

# Default to the mean-donor setting used by the rule-powered spiking diagnostic check.
# Override with EVAL_INTERVENTIONS="mean-donor mean zero" or ONLY_INTERVENTION=mean.
THRESHOLD_EVENT_PROFILE="${THRESHOLD_EVENT_PROFILE:-focused}"
if [[ -n "${EVAL_INTERVENTIONS:-}" ]]; then
  # shellcheck disable=SC2206
  EVAL_INTERVENTION_LIST=($EVAL_INTERVENTIONS)
else
  # Fast, quality-preserving default. For final all-intervention sweeps, run:
  #   EVAL_INTERVENTIONS="mean-donor mean zero" THRESHOLD_EVENT_PROFILE=full bash run_spiking_diagnostics_fast_quality.sh
  EVAL_INTERVENTION_LIST=("mean-donor")
fi

CIRCUIT_DISCOVERY_METHOD="EAP-IG-inputs"
CIRCUIT_DISCOVERY_METHOD_NORM="$(printf '%s' "$CIRCUIT_DISCOVERY_METHOD" | tr '[:upper:]-' '[:lower:]_')"
CIRCUIT_SIZE="${CIRCUIT_SIZE:-200000}"
MIN_FLIP_RATE="${MIN_FLIP_RATE:-0.3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE="${MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE:-1}"
BASELINE_SUBSETS="${BASELINE_SUBSETS:-positive,negative}"
SCRIPT12_TARGET="${SCRIPT12_TARGET:-${THRESHOLD_EVENT_TARGET:-flip_any}}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" == "0" ]]; then
  LOG_DIR="${LOG_DIR:-./logs/spiking_diagnostics}"
  mkdir -p "$LOG_DIR"
fi

# Result collection / upload bundle.
#   COLLECT_RESULTS=1 creates a compact zip after diagnostics finish. Default: 1.
#   COLLECT_ONLY=1 skips diagnostics and only zips existing results.
#   RESULTS_ZIP=path/to/file.zip chooses the output archive path.
#   INCLUDE_ACTIVATION_ROWS=1 includes large per-example activation rows.
COLLECT_RESULTS="${COLLECT_RESULTS:-1}"
COLLECT_ONLY="${COLLECT_ONLY:-0}"
RESULTS_ZIP="${RESULTS_ZIP:-spiking_diagnostics_results_for_inspection.zip}"
INCLUDE_ACTIVATION_ROWS="${INCLUDE_ACTIVATION_ROWS:-0}"
MAX_COLLECT_FILE_MB="${MAX_COLLECT_FILE_MB:-200}"
COLLECT_DRY_RUN="${COLLECT_DRY_RUN:-$DRY_RUN}"

# Rule-powered spiking diagnostic defaults. The default is intentionally not the
# old broad weak/control sweep: sample match/nonmatch examples per candidate rule,
# ablate the union once, compute mean-donor once for the selected unit union,
# and compare activation/gradient/WANDA/first-order proxies.
if [[ "$THRESHOLD_EVENT_PROFILE" == "quick" ]]; then
  : "${SPIKING_MAX_POINTS:=1024}"
  : "${SPIKING_MIN_POINTS:=64}"
  : "${RULE_MATCH_POINTS_PER_RULE:=32}"
  : "${RULE_CONDITIONED_MAX_UNITS:=32}"
  : "${POINTS_TO_USE_FOR_MEAN_ABLATION:=256}"
  : "${THRESHOLD_EVENT_REPEATS:=3}"
elif [[ "$THRESHOLD_EVENT_PROFILE" == "full" ]]; then
  : "${SPIKING_MAX_POINTS:=4096}"
  : "${SPIKING_MIN_POINTS:=64}"
  : "${RULE_MATCH_POINTS_PER_RULE:=128}"
  : "${RULE_CONDITIONED_MAX_UNITS:=200}"
  : "${POINTS_TO_USE_FOR_MEAN_ABLATION:=2048}"
  : "${THRESHOLD_EVENT_REPEATS:=20}"
else
  : "${SPIKING_MAX_POINTS:=2048}"
  : "${SPIKING_MIN_POINTS:=64}"
  : "${RULE_MATCH_POINTS_PER_RULE:=48}"
  : "${RULE_CONDITIONED_MAX_UNITS:=64}"
  : "${POINTS_TO_USE_FOR_MEAN_ABLATION:=2048}"
  : "${THRESHOLD_EVENT_REPEATS:=10}"
fi
SPIKING_GLOBAL_N_CLUSTERS="${SPIKING_GLOBAL_N_CLUSTERS:-64}"
THRESHOLD_EVENT_MIN_EXAMPLES="${THRESHOLD_EVENT_MIN_EXAMPLES:-8}"
THRESHOLD_EVENT_HOLDOUT_FRACTION="${THRESHOLD_EVENT_HOLDOUT_FRACTION:-0.5}"
THRESHOLD_EVENT_N_BINS="${THRESHOLD_EVENT_N_BINS:-10}"
RULE_CONDITIONED_DIAGNOSTICS="${RULE_CONDITIONED_DIAGNOSTICS:-1}"
RULE_CONDITIONED_ONLY="${RULE_CONDITIONED_ONLY:-1}"
RULE_CONDITIONED_MAX_RULES_PER_UNIT="${RULE_CONDITIONED_MAX_RULES_PER_UNIT:-1}"
RULE_MIN_MATCH_EXAMPLES="${RULE_MIN_MATCH_EXAMPLES:-16}"
RULE_NONMATCH_MATCH_RATIO="${RULE_NONMATCH_MATCH_RATIO:-2}"
PROXY_METRICS="${PROXY_METRICS:-activation,abs_activation,wanda,gradient,abs_gradient,activation_x_gradient,abs_activation_x_gradient,predicted_margin_drop,abs_predicted_margin_drop,learned_direction}"
PROXY_ALLOW_FALLBACK_SCORE="${PROXY_ALLOW_FALLBACK_SCORE:-1}"

# Same-layer/head non-agonist baselines.
# Use random non-agonist controls only. The removed spike/flip-count-matched and
# low-flip controls conditioned on the overtopping outcome and are not used.
SAME_LAYER_NONAGONIST_CONTROLS="${SAME_LAYER_NONAGONIST_CONTROLS:-1}"
NONAGONIST_CANDIDATE_POOL_MULTIPLIER="${NONAGONIST_CANDIDATE_POOL_MULTIPLIER:-1}"
NONAGONIST_MIN_CANDIDATE_POOL_PER_LAYER="${NONAGONIST_MIN_CANDIDATE_POOL_PER_LAYER:-4}"
NONAGONIST_RANDOM_CONTROLS_PER_AGONIST="${NONAGONIST_RANDOM_CONTROLS_PER_AGONIST:-2}"


SPECTRAL_FLAGS=(
  --spectral_space hidden
  --rep_hook_name ln_final.hook_normalized
  --rep_pooling last
  --spectral_dim 32
)

contains_item() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

is_pythia_llm() {
  local llm="$1"
  [[ "$llm" == EleutherAI/pythia-* ]]
}

should_run_experiment_for_llm() {
  local experiment="$1"
  local llm="$2"

  if [[ "$llm" == "Qwen/Qwen2-1.5B-Instruct" ]]; then
    contains_item "$experiment" "${MATH_EXPERIMENT_LIST[@]}" || return 1
  fi

  if is_pythia_llm "$llm"; then
    contains_item "$experiment" "${LANGUAGE_EXPERIMENT_LIST[@]}" || return 1
  fi

  return 0
}

should_run_decode_only() {
  local experiment="$1"
  local llm="$2"
  contains_item "$experiment" "${LANGUAGE_EXPERIMENT_LIST[@]}" || return 1
  should_run_experiment_for_llm "$experiment" "$llm" || return 1
  return 0
}

run_or_echo() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY-RUN]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

collect_threshold_event_results() {
  local collect_root="${1:-$ROOT_DIR}"
  local out_zip="${2:-$RESULTS_ZIP}"
  local include_activation_rows="${INCLUDE_ACTIVATION_ROWS:-0}"
  local max_file_mb="${MAX_COLLECT_FILE_MB:-200}"
  local dry_collect="${COLLECT_DRY_RUN:-0}"

  if [[ ! -d "$collect_root" ]]; then
    echo "[collect] ERROR: root does not exist or is not a directory: $collect_root" >&2
    return 1
  fi

  command -v find >/dev/null 2>&1 || {
    echo "[collect] ERROR: find is required" >&2
    return 1
  }

  local root_abs
  root_abs="$(cd "$collect_root" && pwd)"

  local out_abs="$out_zip"
  case "$out_abs" in
    /*) ;;
    *) out_abs="$(pwd)/$out_abs" ;;
  esac

  local tmp_dir
  tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/spiking_results_collect.XXXXXX")"
  local stage="$tmp_dir/payload"
  local manifest="$stage/MANIFEST.tsv"
  mkdir -p "$stage"
  printf "relative_path\tsize_bytes\n" > "$manifest"

  local files_list="$tmp_dir/files.txt"
  (
    cd "$root_abs"
    find . -path '*spiking_diagnostics*' -type f \( \
      -name 'aggregate*.csv' \
      -o -name 'aggregate*.json' \
      -o -name 'flip_conditioned_candidate_units.csv' \
      -o -name 'flip_conditioned_flip_support.csv' \
      -o -name 'flip_conditioned_threshold_unit_tests.csv' \
      -o -name 'flip_conditioned_threshold_summary.csv' \
      -o -name 'threshold_unit_tests.csv' \
      -o -name 'threshold_population_summary.csv' \
      -o -name 'best_threshold_metrics.csv' \
      -o -name 'threshold_binned_flip_curves.csv' \
      -o -name 'threshold_event_summary*.csv' \
      -o -name 'threshold_event_summary*.json' \
      -o -name 'threshold_spiking_experiment.json' \
      -o -name 'run_plan*.json' \
      -o -name 'config*.json' \
      -o -name '*.log' \
    \) -print

    if [[ "$include_activation_rows" == "1" ]]; then
      find . -path '*spiking_diagnostics*' -type f \( \
        -name 'threshold_activation_flip_rows.csv.gz' \
        -o -name 'threshold_spiking_event_rows.csv.gz' \
        -o -name 'rule_conditioned_activation_rows.csv.gz' \
      \) -print
    fi
  ) | sed 's#^\./##' | sort -u > "$files_list"

  local copy_count=0
  local skip_count=0
  local max_bytes=$((max_file_mb * 1024 * 1024))
  local rel
  local size_bytes

  while IFS= read -r rel; do
    [[ -z "$rel" ]] && continue

    if [[ ! -f "$root_abs/$rel" ]]; then
      continue
    fi

    if size_bytes="$(stat -f '%z' "$root_abs/$rel" 2>/dev/null)"; then
      :
    else
      size_bytes="$(stat -c '%s' "$root_abs/$rel")"
    fi

    if [[ "$size_bytes" -gt "$max_bytes" ]]; then
      echo "[collect] skip large file (${size_bytes} bytes): $rel" >&2
      skip_count=$((skip_count + 1))
      continue
    fi

    printf "%s\t%s\n" "$rel" "$size_bytes" >> "$manifest"

    if [[ "$dry_collect" == "1" ]]; then
      echo "[collect] would copy: $rel"
    else
      mkdir -p "$stage/$(dirname "$rel")"
      cp -p "$root_abs/$rel" "$stage/$rel"
    fi
    copy_count=$((copy_count + 1))
  done < "$files_list"

  if [[ "$copy_count" -eq 0 ]]; then
    echo "[collect] ERROR: found no spiking diagnostic result files under: $root_abs" >&2
    echo "[collect] Hint: run from the repository root, or pass ROOT_DIR to the path containing ./data." >&2
    rm -rf "$tmp_dir"
    return 2
  fi

  echo "[collect] selected files: $copy_count"
  echo "[collect] skipped large files: $skip_count"
  echo "[collect] manifest: MANIFEST.tsv"

  if [[ "$dry_collect" == "1" ]]; then
    echo "[collect] dry run complete; no zip written."
    rm -rf "$tmp_dir"
    return 0
  fi

  mkdir -p "$(dirname "$out_abs")"
  rm -f "$out_abs"

  if command -v zip >/dev/null 2>&1; then
    (cd "$stage" && zip -qr "$out_abs" .)
  else
    echo "[collect] zip command not found; using Python zipfile fallback"
    python3 - "$stage" "$out_abs" <<'PY'
from pathlib import Path
import sys
import zipfile

stage = Path(sys.argv[1])
out = Path(sys.argv[2])
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(stage).as_posix())
PY
  fi

  if [[ -f "$out_abs" ]]; then
    if size_bytes="$(stat -f '%z' "$out_abs" 2>/dev/null)"; then
      :
    else
      size_bytes="$(stat -c '%s' "$out_abs")"
    fi
    echo "[collect] wrote: $out_abs"
    echo "[collect] zip size bytes: $size_bytes"
  else
    echo "[collect] ERROR: failed to create zip: $out_abs" >&2
    rm -rf "$tmp_dir"
    return 3
  fi

  rm -rf "$tmp_dir"
}


make_circuit_label() {
  local eval_intervention="$1"
  local run_mode="$2"
  local label="spectral_split"

  if [[ "$CIRCUIT_SIZE" != "100000" ]]; then
    label+="-M${CIRCUIT_SIZE}"
  fi

  if [[ "$run_mode" == "decode-only" ]]; then
    label+="-decode_only"
  fi

  if [[ "$eval_intervention" != "mean" ]]; then
    label+="-eval_${eval_intervention}"
  fi

  printf '%s' "$label"
}

make_bag_label() {
  local label="agonist_neurons-fast-random_anchor"
  if [[ "$MIN_FLIP_RATE" != "0.2" ]]; then
    label+="-tau${MIN_FLIP_RATE}"
  fi
  printf '%s' "$label"
}

if [[ "$COLLECT_ONLY" == "1" ]]; then
  collect_threshold_event_results "$ROOT_DIR" "$RESULTS_ZIP"
  echo "Done. Result bundle collection finished."
  exit 0
fi

for EVAL_INTERVENTION in "${EVAL_INTERVENTION_LIST[@]}"; do
  if [[ -n "${ONLY_INTERVENTION:-}" && "$EVAL_INTERVENTION" != "$ONLY_INTERVENTION" ]]; then
    continue
  fi

  for ANALYZED_LLM in "${ANALYZED_LLM_LIST[@]}"; do
    if [[ -n "${ONLY_MODEL:-}" && "$ANALYZED_LLM" != "$ONLY_MODEL" ]]; then
      continue
    fi

    for EXPERIMENT in "${EXPERIMENT_LIST[@]}"; do
      if [[ -n "${ONLY_TASK:-}" && "$EXPERIMENT" != "$ONLY_TASK" ]]; then
        continue
      fi

      if ! should_run_experiment_for_llm "$EXPERIMENT" "$ANALYZED_LLM"; then
        echo "[skip] task/model not in phenomenology setting: task=$EXPERIMENT model=$ANALYZED_LLM"
        continue
      fi

      RUN_MODE_LIST=("standard")
      if should_run_decode_only "$EXPERIMENT" "$ANALYZED_LLM"; then
        RUN_MODE_LIST+=("decode-only")
      fi

      for RUN_MODE in "${RUN_MODE_LIST[@]}"; do
        if [[ -n "${ONLY_RUN_MODE:-}" && "$RUN_MODE" != "$ONLY_RUN_MODE" ]]; then
          continue
        fi

        TASK_MODULE="lib.tasks.${EXPERIMENT}_task"
        CACHE_DIR="./cache/${EXPERIMENT}"
        DATA_DIR="./data/${EXPERIMENT}/${ANALYZED_LLM}"
        DISCOVERY_ROOT="${DATA_DIR}/neural_circuit_discovery_results/${CIRCUIT_DISCOVERY_METHOD_NORM}"
        CIRCUIT_LABEL="$(make_circuit_label "$EVAL_INTERVENTION" "$RUN_MODE")"
        BAG_LABEL="$(make_bag_label)"
        INPUT_DATA_DIR="${DISCOVERY_ROOT}/${CIRCUIT_LABEL}/neural_circuits"
        OUT_DIR="${DISCOVERY_ROOT}/${CIRCUIT_LABEL}/spiking_diagnostics"
        RULES_DIR="${DATA_DIR}/rule_extraction_results/neuron_flip_rules"
        RULES_STATS_DIRNAME="${CIRCUIT_LABEL}-${BAG_LABEL}"
        LOG_SAFE_MODEL="$(printf '%s' "$ANALYZED_LLM" | tr '/:@' '___')"
        if [[ "$DRY_RUN" == "0" ]]; then
          LOG_FILE="${LOG_DIR}/${EXPERIMENT}__${LOG_SAFE_MODEL}__${EVAL_INTERVENTION}__${RUN_MODE}__threshold_spiking.log"
        fi

        if [[ ! -d "$INPUT_DATA_DIR" ]]; then
          echo "[skip] missing circuits: $INPUT_DATA_DIR"
          continue
        fi
        echo "[spiking diagnostics] task=$EXPERIMENT model=$ANALYZED_LLM intervention=$EVAL_INTERVENTION run_mode=$RUN_MODE"
        echo "                   profile=$THRESHOLD_EVENT_PROFILE rule_conditioned_only=$RULE_CONDITIONED_ONLY"
        echo "                   input=$INPUT_DATA_DIR"
        echo "                   out=$OUT_DIR"
        echo "                   high_n_points_per_baseline=$SPIKING_MAX_POINTS candidate_rule_points_per_rule=$RULE_MATCH_POINTS_PER_RULE max_rule_units=$RULE_CONDITIONED_MAX_UNITS mean_points=$POINTS_TO_USE_FOR_MEAN_ABLATION"
        echo "                   proxies=$PROXY_METRICS"
        echo "                   nonagonist_controls same_layer=$SAME_LAYER_NONAGONIST_CONTROLS random=$NONAGONIST_RANDOM_CONTROLS_PER_AGONIST"

        CMD=(
          python3 "$THRESHOLD_EVENT_SCRIPT"
          --input_data_dir "$INPUT_DATA_DIR"
          --out_dir "$OUT_DIR"
          --baseline_subsets "$BASELINE_SUBSETS"
          --task_module "$TASK_MODULE"
          --ai_model "$ANALYZED_LLM"
          --batch_size "$BATCH_SIZE"
          --intervention "$EVAL_INTERVENTION"
          --points_to_use_for_mean_ablation "$POINTS_TO_USE_FOR_MEAN_ABLATION"
          --spiking_max_points "$SPIKING_MAX_POINTS"
          --spiking_min_points "$SPIKING_MIN_POINTS"
          --spiking_global_n_clusters "$SPIKING_GLOBAL_N_CLUSTERS"
          --threshold_event_min_examples "$THRESHOLD_EVENT_MIN_EXAMPLES"
          --threshold_event_repeats "$THRESHOLD_EVENT_REPEATS"
          --threshold_event_holdout_fraction "$THRESHOLD_EVENT_HOLDOUT_FRACTION"
          --threshold_event_n_bins "$THRESHOLD_EVENT_N_BINS"
          --target "$SCRIPT12_TARGET"
          --proxy_metrics "$PROXY_METRICS"
          --nonagonist_candidate_pool_multiplier "$NONAGONIST_CANDIDATE_POOL_MULTIPLIER"
          --nonagonist_min_candidate_pool_per_layer "$NONAGONIST_MIN_CANDIDATE_POOL_PER_LAYER"
          --nonagonist_random_controls_per_agonist "$NONAGONIST_RANDOM_CONTROLS_PER_AGONIST"
          --cluster_by_spectral
          --spectral_cache_dir "$CACHE_DIR"
          "${SPECTRAL_FLAGS[@]}"
          --global_n_clusters "$MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE"
        )

        if [[ "$SAME_LAYER_NONAGONIST_CONTROLS" == "0" ]]; then
          CMD+=(--no_same_layer_nonagonist_controls)
        else
          CMD+=(--same_layer_nonagonist_controls)
        fi

        if [[ "$RULE_CONDITIONED_DIAGNOSTICS" == "1" && -d "$RULES_DIR" ]]; then
          CMD+=(
            --rule_conditioned_diagnostics
            --rules_dir "$RULES_DIR"
            --rules_stats_dirname "$RULES_STATS_DIRNAME"
            --rule_conditioned_max_units "$RULE_CONDITIONED_MAX_UNITS"
            --rule_conditioned_max_rules_per_unit "$RULE_CONDITIONED_MAX_RULES_PER_UNIT"
            --rule_match_points_per_rule "$RULE_MATCH_POINTS_PER_RULE"
            --rule_min_match_examples "$RULE_MIN_MATCH_EXAMPLES"
            --rule_nonmatch_match_ratio "$RULE_NONMATCH_MATCH_RATIO"
          )
          if [[ "$RULE_CONDITIONED_ONLY" == "1" ]]; then
            CMD+=(--rule_conditioned_only)
          fi
        fi
        if [[ "$PROXY_ALLOW_FALLBACK_SCORE" == "1" ]]; then
          CMD+=(--proxy_allow_fallback_score)
        fi
        if [[ "$RUN_MODE" == "decode-only" ]]; then
          CMD+=(--decode_only)
        fi
        if [[ "$FORCE" == "1" ]]; then
          CMD+=(--force_spiking_eval --force_population_stats --force_threshold_event)
        fi

        mkdir -p "$OUT_DIR"
        if [[ "$DRY_RUN" == "1" ]]; then
          run_or_echo "${CMD[@]}"
        else
          "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
        fi
      done
    done
  done
done

if [[ "$COLLECT_RESULTS" == "1" ]]; then
  collect_threshold_event_results "$ROOT_DIR" "$RESULTS_ZIP"
  echo "Done. High-N flip-conditioned spiking diagnostics are under each run's spiking_diagnostics directory."
  echo "Upload bundle: $RESULTS_ZIP"
else
  echo "Done. High-N flip-conditioned spiking diagnostics are under each run's spiking_diagnostics directory."
fi
