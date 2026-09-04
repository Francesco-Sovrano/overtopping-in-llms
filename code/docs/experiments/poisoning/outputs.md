# Poisoning outputs

Each task/model/seed run uses one stage-numbered directory. Cross-seed products are stored outside individual runs.

## Run root

```text
data/poisoning/<task>/<run-name>/
├── 01_training_checkpoints/
├── 02_evaluation_cohorts/
├── 03_checkpoint_causal_discovery/
├── 04_condition_comparisons/
├── 05_behavior_trajectories/
├── 06_circuit_overlap_analysis/
└── 07_poisoning_example_detection/
```

## Stage 01 — training checkpoints

Contains clean and poisoned checkpoint directories plus metadata describing task, model, seed, markers, poison construction, training schedule, optimizer settings, checkpoint fractions, and run identity.

## Stage 02 — evaluation cohorts

Contains stable held-out cohort definitions used across checkpoints. Cohort identities are selected independently of model outputs.

## Stage 03 — checkpoint behavior and causal discovery

Each condition/checkpoint/phase can contain:

```text
normal_task/
backdoor_trigger_test/
attack_cohort_control_correctness/
```

### `normal_task`

Behavior-only no-trigger scores. Principal field:

```text
normal_task_accuracy_without_trigger
```

The score table records row identity and the sampling contract, including population size, cap, seed, strategy, stratum definition, and stratum counts.

Prompt and generation fields such as `prompt_control`, `raw_output_control`, and `original_prompt` are textual data even when the generated text looks numeric.

### `backdoor_trigger_test`

Paired control/trigger behavior tables and statistics. Canonical fields include:

```text
control_target_rate
trigger_target_rate
trigger_excess_target_rate
trigger_lift_rate
conditional_conversion_rate
convertible_fraction
```

Trigger-specific causal artifacts exist only when trigger-lift CHA is enabled and has a valid population.

### `attack_cohort_control_correctness`

Shared-pipeline causal analysis of correctness on the attack-eligible non-target cohort. It contains feature reports, candidate/rule outputs, singleton intervention materializations, and set-level statistics.

Aggregated trajectory fields use the `attack_cohort_control_correctness_` prefix.

## Stage 04 — matched condition comparison

Contains clean-versus-poisoned checkpoint comparison tables. Normal-task comparisons verify that both conditions use the same sampling population, cap, seed, strategy, stratum definition, and scanned row count.

## Stage 05 — behavior trajectories

Principal trajectory table:

```text
05_behavior_trajectories/<phase>/backdoor_lift_overtopping_trajectory.csv
```

It joins available normal-task behavior, trigger behavior, control-correctness causal metrics, and optional trigger-lift causal metrics. Missing endpoints remain missing rather than being filled with zero.

`backdoor_overtopping_dashboard.pdf` creates panels only for metrics with finite values.

## Stage 06 — circuit overlap

Contains checkpoint circuit comparisons among the trigger endpoint, attack-cohort control-correctness endpoint, and configured reference analyses. Scientific configuration columns are retained in the comparison rows.

## Stage 07 — poisoning-example detection

### Frozen candidate union

```text
control_correctness_candidate_union.csv
```

Lists the frozen union of control-correctness agonist candidates across matched checkpoints.

### Fixed control-correctness materialization

```text
control_correctness_u_j_materialization/
```

Contains explicit fixed-union singleton evaluations across matched clean and poisoned checkpoints.

### Fixed attack-side materialization

```text
attack_u_j_materialization/
```

Contains the frozen candidate union evaluated on the poisoned trigger endpoint where required checkpoint data exist.

### Interval disruption tables

```text
control_correctness_channel_disruption_by_interval.csv
selected_disruptive_channels_by_interval.csv
mapped_disruptive_channels.csv
```

These describe clean/poisoned causal-role changes, selected channels, and their mapping to trained projections.

### Training-exposure scores

```text
training_exposure_order.csv
training_example_scores_all_intervals.csv
top_suspected_training_examples.csv
```

Scores are computed without poison labels. Labels are joined only for detector evaluation.

### Detector metrics

```text
detection_metrics_by_interval.csv
detection_vs_attack_success_by_interval.csv
detection_vs_attack_association.csv
detection_vs_attack_association.json
detection_summary.json
```

The behavioral attack endpoint used for association is `conditional_conversion_rate`.

### Checkpoint story figures

```text
01_clean_vs_poisoned_overtopping_development.pdf
02_channel_role_reassignment.pdf
03_prospective_defense_leverage.pdf
04_clean_vs_poisoned_checkpoint_overtopping.pdf
```

Figure 01 contains aggregate checkpoint-level quantities. Figure 02 distinguishes checkpoint-local and fixed-union channel measurements. Figure 03 is a prospective defense-target screen using a checkpoint-0 locked set and one-checkpoint-ahead selection from the fixed union. Figure 04 requires complete fixed-union materialization across all displayed matched checkpoints and conditions.

Machine-readable figure-status files include:

```text
prospective_defense_leverage.csv
story_data_coverage.csv
story_figure_status.csv
```

### Additional interpretation outputs

`overtopping_interpretation/` contains analyses of causal-role location, support persistence, clean-versus-poisoned causal drift, update geometry, poison-detection implications, and attack-growth relationships.

## Stage 08 — cross-seed aggregation

Cross-seed results are written under:

```text
data/poisoning/final/<study>/08_cross_seed_aggregation/
```

Aggregation retains poison rate, markers, training configuration, task, model, seed, and other scientific identifiers.

## Poisoning caches

Poisoning cache root:

```text
cache/poisoning/
```

Endpoint-specific cache families include:

```text
.../adaptive_circuit_discovery/backdoor_trigger_test/
.../adaptive_circuit_discovery/normal_task/
.../adaptive_circuit_discovery/attack_cohort_control_correctness/
```

Cache reuse is conditional on matching population and method metadata. Changing the normal-task sampling population invalidates its population cache; prompt-matched trigger generations can remain reusable when their prompt identity and generation configuration are unchanged.
