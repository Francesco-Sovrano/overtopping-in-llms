# Operations and troubleshooting

For method definitions, see [Research questions](../research-questions/) and [Methods](../methods/). This document covers execution failures, incomplete artifacts, and targeted regeneration.

## Inspect commands before execution

```bash
./run_overtopping_experiments.sh --dry-run
./run_poisoning_experiments.sh --dry-run
```

Use `--list` for the overtopping catalogue:

```bash
./run_overtopping_experiments.sh --list
```

## Import errors

Run Python modules from `code/`:

```bash
cd code
python -m studies.overtopping.experiments.run_experiments --dry-run
```

or export:

```bash
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
```

Use the repository virtual environment created by `setup.sh`.

## RQ1 reports fewer than 39 settings

Figure 2 is defined by the exact 28 primary + 11 auxiliary catalogue. The expected phase split is:

```text
17 input+output
22 decode-only
```

The analysis resolves each expected statistics directory from `RunSpec`. Decode-only and input+output runs use distinct directory names; decode-only statistics paths contain `-decode_only-`.

A missing required path indicates an incomplete catalogue setting. A completed zero-candidate setting is represented according to its persisted pipeline status/artifact contract rather than inferred from an unrelated directory.

## Required directional metrics are missing

Directional manuscript metrics are computed from Stage-7 singleton intervention materializations. If `scores.csv` already contains the required intervention columns, use the statistics-rebuild utilities rather than rerunning model interventions.

`generate_results.sh` validates metrics; it does not create missing model-backed causal data.

## RQ3 population audit fails

RQ3 uses an explicit manifest:

- every primary run is required;
- auxiliary runs are included only when their exact diagnostic artifacts are complete;
- poisoning is excluded.

Inspect:

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/population_audit.csv
results/analysis/rq3_threshold_event/spiking_diagnostics/population_coverage.json
results/analysis/rq3_threshold_event/spiking_diagnostics/threshold_shape_validation/threshold_shape_population_audit.csv
results/analysis/rq3_threshold_event/spiking_diagnostics/threshold_shape_validation/threshold_shape_population_coverage.csv
```

## RQ3 direction audit fails

Candidate rows must satisfy:

```text
positive baseline -> rq3_direction=1to0
negative baseline -> rq3_direction=0to1
candidate_direction_policy=discovery_baseline_only
```

The experiment-level threshold schema is:

```text
threshold-event-v4-discovery-direction-aware
```

The nested threshold-shape schema is:

```text
graded-agonist-intervention-v1
```

Rebuild experiment-level RQ3 diagnostics when per-baseline files do not contain the required discovery-direction fields.

## Regenerate only the model-free RQ3 threshold-shape analysis

Use this when experiment-level `spiking_diagnostics-*` products already have the correct directional schema and only the aggregate Stage-8 statistics or Figure-4 presentation need to be recomputed.

From the repository root:

```bash
rm -rf results/analysis/rq3_threshold_event/spiking_diagnostics/threshold_shape_validation
rm -rf results/paper/figures/04_rq3_spiking_cut
```

Then run:

```bash
./generate_results.sh
```

No high-N model cache needs to be deleted for this case.

## Regenerate RQ3 experiment-level directional diagnostics

Use this when the per-baseline diagnostic products were generated without the required discovery-direction contract, when the candidate/control population definition changed, or when the experiment-level threshold method changed.

Delete the aggregate RQ3 reporting products:

```bash
rm -rf results/analysis/rq3_threshold_event/spiking_diagnostics
rm -rf results/paper/figures/04_rq3_spiking_cut
```

Delete only the per-baseline derived threshold files while preserving high-N model caches:

```bash
find data -type d -name 'spiking_diagnostics-*' -print0 |
while IFS= read -r -d '' d; do
  for baseline in positive_baseline negative_baseline; do
    b="$d/$baseline"
    [ -d "$b" ] || continue
    rm -f \
      "$b/threshold_spiking_experiment.json" \
      "$b/threshold_event_results_summary.json" \
      "$b/selected_units_from_rules.csv" \
      "$b/same_layer_nonagonist_control_pool.csv" \
      "$b/same_layer_nonagonist_control_selection.csv" \
      "$b/rule_conditioned_sampling_plan.csv" \
      "$b/high_n_scores_with_flips.csv" \
      "$b/high_n_flip_stats_by_unit.csv" \
      "$b/threshold_activation_flip_rows.csv.gz" \
      "$b/threshold_unit_tests.csv" \
      "$b/threshold_population_summary.csv" \
      "$b/threshold_binned_flip_curves.csv" \
      "$b/flip_conditioned_candidate_units.csv" \
      "$b/flip_conditioned_flip_support.csv" \
      "$b/flip_conditioned_threshold_unit_tests.csv" \
      "$b/flip_conditioned_threshold_summary.csv" \
      "$b/flip_conditioned_threshold_experiment.json"
  done

  for f in "$d"/aggregate_*.csv; do
    [ -e "$f" ] && rm -f "$f"
  done

  rm -f \
    "$d/threshold_spiking_experiment_aggregate.json" \
    "$d/threshold_event_results_summary.json"
  rm -rf "$d/figures"
done
```

Preserve these experiment inputs and model caches:

```text
scores.csv
frozen_candidate_ranking.csv
flip_stats_by_neuron.csv
high_n_eval_cache/
ablation_cache/
replacement_scores_cache/
```

Then rebuild the exact RQ3 diagnostic population from `code/`:

```bash
python -m studies.overtopping.analysis.rebuild_spiking_diagnostics \
  --primary-table ../results/analysis/primary_matrix/tables/primary_table.csv \
  --data-root ../data \
  --population-scope primary+supplementary
```

The rebuild reconstructs discovery-direction provenance from Stage-6 candidate artifacts when the frozen Stage-7 tables do not already carry the required fields.

## Regenerate preemption products

Preemption products depend on directional candidate membership and the interaction threshold-event method.

Delete only the preemption-specific files inside `interaction_validation/`:

```bash
find data -type d -name interaction_validation -print0 |
while IFS= read -r -d '' d; do
  rm -f \
    "$d/preemption_pair_plan.csv" \
    "$d/preemption_pair_summary.csv" \
    "$d/preemption_example_masks.csv.gz" \
    "$d/preemption_summary.json"
done
```

Preserve:

```text
group_eval_cache/
matched_null_draws.csv
matched_random_set_membership.csv
conditional_marginal_draws.csv
conditional_marginal_summary.csv
```

Rerun the overtopping pipeline on the test split so Stage 08 recreates the preemption products:

```bash
./run_overtopping_experiments.sh --suite all --evaluation-split test
```

Then regenerate reporting:

```bash
./generate_results.sh
```

## RQ3 diagnostics collide across point caps

A non-default `spiking_max_points` value is encoded in the diagnostics path, for example `-cap512`. Distinct point caps should not write into the same derived directory.

## Preemption values have unexpected sign

Check the interaction schema and threshold metadata. The required interaction schema is:

```text
conditional-marginal-validation-v2-direction-aware-preemption
```

Signed training MCC determines whether the fitted threshold predicate is inverted before event-present and event-absent subsets are evaluated.

The preemption analysis remains sensitive to dominant/secondary pair selection because pair identity is selected from singleton statistics on the interaction population.

## Legacy RQ3 scalar diagnostics

Threshold fitting requires enough flip and non-flip examples. Low-effect controls can therefore have valid causal-strength estimates but no threshold result.

Interpret these outputs separately:

```text
threshold-testable fraction       population-level feasibility
graded agonist dose response      primary RQ3 causal crossing experiment
same-agonist negative support      optional within-agonist reference
```

Legacy threshold/TECS diagnostics may still contain sparse control support. They are no longer manuscript-facing RQ3 endpoints; use the graded agonist experiment for the primary spiking analysis.

## Poisoning normal-task scan is unexpectedly large

The normal-task cap is:

```bash
printf '%s\n' "${NORMAL_TASK_SCAN_MAX_ROWS:-unset}"
```

`NORMAL_TASK_SCAN_MAX_ROWS=0` requests exhaustive evaluation. This cap is independent of trigger-lift and Stage-7 caps.

## Poisoning normal-task population mismatch

Normal-task cache metadata include row identity, sampling strategy, stratification, cap, seed, and population size. Clean and poisoned comparisons require matching population definitions.

Regenerate both endpoint populations when these metadata differ.

## Trigger behavior exists but trigger-specific CHA is absent

Check:

```text
RUN_TRIGGER_LIFT_CHA
```

When it is `0`, trigger/control behavior can still be measured while trigger-specific causal discovery is omitted.

## Attack-cohort control-correctness CHA is missing

Inspect the checkpoint's `attack_cohort_control_correctness/` status metadata. The configured low-data policy can skip causal discovery when the required source-state population is unavailable.

## Arithmetic correctness mismatches

Arithmetic model generations are textual data. Load fields such as `prompt_control`, `raw_output_control`, and `original_prompt` as strings. Numeric-looking completions must not be silently converted to floating-point values before task correctness is recomputed.

## Fixed candidate materialization is missing in poisoning Stage 07

Longitudinal fixed-union analysis requires explicit evaluation of the frozen control-correctness candidate set at each relevant checkpoint. A missing fixed candidate measurement is not equivalent to zero effect.

Rerun the checkpoint causal workflow for the affected run/checkpoint/condition.

## Poisoning aggregation rejects a run family

Cross-seed aggregation retains scientific configuration. Runs with different poison rates, marker definitions, targets, training schedules, or other required configuration fields are not combined into one trajectory.

Aggregate one compatible scientific configuration at a time.

## Poison-detection metrics are empty

Inspect `07_poisoning_example_detection/` status and interval tables. Empty detector metrics can result from no channels meeting the causal-disruption selection criteria or missing prerequisite materializations.

## Accelerator memory pressure

Reduce execution batch size while leaving the scientific row population unchanged:

```bash
PIPELINE_BATCH_SIZE=8 ./run_poisoning_experiments.sh
```

High-N RQ3 evaluation, WANDA scoring, fixed-union evaluation, and group interventions have separate batch controls. Change batch size rather than population caps when the goal is only to reduce memory use.

## API authentication failures

Check credential presence without printing values:

```bash
python - <<'PY'
import os
for name in ("GROQ_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"):
    print(f"{name}: {'set' if os.environ.get(name) else 'not set'}")
PY
```

For direct module execution:

```bash
set -a
. ./.secrets.env
set +a
```

If Hugging Face cannot find a model that should be downloaded, check `HF_HUB_OFFLINE`. Use `HF_HUB_OFFLINE=0` only for the command that intentionally fetches model files.
