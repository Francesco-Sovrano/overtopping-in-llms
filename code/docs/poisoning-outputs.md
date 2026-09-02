# Poisoning outputs

Each task/model/seed run uses one stage-numbered directory. Cross-seed aggregation is stored outside individual runs.

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

Contains clean and poisoned checkpoint directories plus metadata describing model, seed, markers, poison construction, schedule, optimizer/training settings, checkpoint fractions, and run identity.

## Stage 02 — evaluation cohorts

Contains stable held-out cohort definitions used across checkpoints. Cohort identities are selected independently of model outputs.

## Stage 03 — checkpoint evaluation and causal discovery

Each condition/checkpoint/phase contains separate endpoint directories:

```text
normal_task/
backdoor_trigger_test/
attack_cohort_control_correctness/
```

### `normal_task`

Behavior-only no-trigger scores. The normal-task score table records row identity and the deterministic sampling contract, including population size, cap, seed, strategy, stratum definition, and stratum assignment/count information.

The principal behavioral field is:

```text
normal_task_accuracy_without_trigger
```

CSV prompt and generation fields are textual data. In particular, `prompt_control`, `raw_output_control`, and `original_prompt` must be loaded as text when results are re-read for correctness checks or paired-cache identity checks. Numeric-looking generations such as `12` are model text, not numeric measurements.

### `backdoor_trigger_test`

Paired control/trigger scores and behavior statistics. Canonical behavior fields include:

```text
control_target_rate
trigger_target_rate
trigger_excess_target_rate
trigger_lift_rate
conditional_conversion_rate
convertible_fraction
```

Trigger-specific causal artifacts exist only when trigger-lift CHA is enabled and scientifically defined.

### `attack_cohort_control_correctness`

Causal analysis of correctness on the attack-eligible non-target cohort. It contains shared-pipeline feature reports, candidate/rule outputs, singleton materializations, and set-level statistics.

Stage-05 aggregation exposes these metrics with the `attack_cohort_control_correctness_` prefix.

## Stage 04 — matched behavior comparison

Contains clean-versus-poisoned checkpoint comparison tables and trajectories for `normal_task` and `backdoor_trigger_test`. Normal-task comparisons verify that both conditions use the same sampling mode, population size, cap, seed, strategy, stratum definition, and scanned count.

## Stage 05 — developmental trajectories

The principal trajectory table is:

```text
05_behavior_trajectories/<phase>/backdoor_lift_overtopping_trajectory.csv
```

It combines available normal-task behavior, trigger behavior, control-correctness causal metrics, and optional trigger-lift causal metrics without inventing absent endpoints.

`backdoor_overtopping_dashboard.pdf` is dynamic: only metrics with finite values receive panels. The dashboard therefore remains meaningful when trigger-lift CHA is disabled.

## Stage 06 — circuit overlap

Contains checkpoint circuit comparisons between the backdoor-trigger endpoint, attack-cohort control-correctness endpoint, and configured reference analyses. Scientific configuration is preserved in comparison tables.

## Stage 07 — poisoning-example detection

Stage 07 stores the fixed candidate definition, longitudinal causal materializations, interval disruption tables, training-row scores, detector metrics, and interpretation figures.

### Candidate definition

```text
control_correctness_candidate_union.csv
```

Lists the frozen union of control-correctness agonist candidates across matched checkpoints.

### Fixed control-correctness materialization

```text
control_correctness_u_j_materialization/
```

Contains explicit fixed-union singleton evaluations across matched clean/poisoned checkpoints.

### Fixed attack-side materialization

```text
attack_u_j_materialization/
```

Contains the frozen candidate union evaluated on the poisoned trigger-test endpoint where the required checkpoint data exist.

### Interval causal disruption

```text
control_correctness_channel_disruption_by_interval.csv
selected_disruptive_channels_by_interval.csv
mapped_disruptive_channels.csv
```

These tables describe matched clean/poisoned changes, selected channels, and their mapping to trained projections.

### Training exposure scores

```text
training_exposure_order.csv
training_example_scores_all_intervals.csv
top_suspected_training_examples.csv
```

Scores are computed without poison labels. Labels are joined for post-hoc detector evaluation.

### Detection metrics

```text
detection_metrics_by_interval.csv
detection_vs_attack_success_by_interval.csv
detection_vs_attack_association.csv
detection_vs_attack_association.json
detection_summary.json
```

The attack association uses `conditional_conversion_rate` as the behavioral endpoint.

### Paper-facing checkpoint story

The compact Stage-07 story contains:

```text
01_clean_vs_poisoned_overtopping_development.pdf
02_channel_role_reassignment.pdf
03_prospective_defense_leverage.pdf
04_clean_vs_poisoned_checkpoint_overtopping.pdf
```

Figure 01 contains only aggregate checkpoint-level quantities. Figure 02 is descriptive and marks checkpoint-local-only values explicitly rather than presenting them as longitudinal evaluations. Figure 03 is explicitly a prospective defense-target screen. It uses a checkpoint-0 locked target set plus rolling selection within the fixed control-correctness candidate union using only the previous checkpoint. For each selected channel it reports Δdef = attack-suppression rate − benign-correctness-damage rate. Positive Δdef is the singleton selectivity signal that makes a channel a plausible defense target. This is still not a defense-efficacy result: efficacy requires jointly applying the selected intervention set and measuring end-to-end attack suppression together with benign/clean utility. Figure 04 requires complete fixed-union materialization across every matched checkpoint/condition; it is withheld rather than published with an incomplete heatmap.

The machine-readable sidecars `prospective_defense_leverage.csv`, `story_data_coverage.csv`, and `story_figure_status.csv` make selection and completeness explicit.

### Additional interpretation

`overtopping_interpretation/` contains analyses that add distinct information beyond the compact checkpoint story, including:

- where poisoning-specific causal control moves;
- whether the same causal support persists;
- whether poisoning-specific causal change is more concentrated than matched clean drift;
- update-geometry controls;
- poison-detection implications;
- attack-growth links.

The interpretation package does not duplicate the compact four-figure story.

## Stage 08 — cross-seed aggregation

Cross-seed results are written under:

```text
data/poisoning/final/<study>/08_cross_seed_aggregation/
```

Aggregation preserves poison rate, marker configuration, training configuration, task, model, seed, and other scientific fields. Plotting rejects a task/model family if multiple incompatible configurations would otherwise be joined.

## Caches

Poisoning caches live under `cache/poisoning/`. Important endpoint caches include:

```text
.../adaptive_circuit_discovery/backdoor_trigger_test/
.../adaptive_circuit_discovery/normal_task/
.../adaptive_circuit_discovery/attack_cohort_control_correctness/
```

Changing the normal-task sampling definition invalidates the `normal_task` population cache. Exact matching generations in `backdoor_trigger_test` remain reusable prompt by prompt. Cache reuse never changes the selected population.
