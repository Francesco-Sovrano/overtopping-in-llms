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
observed_training_mixture_correctness/
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

### `observed_training_mixture_correctness`

The default and only strictly defense-facing CHA localization endpoint. It evaluates each checkpoint model on the same reconstructed defender-visible fine-tuning prompt/label mixture. Candidate localization uses correctness with respect to the observed label, runs both positive and negative observable baseline branches, and explicitly excludes hidden poison/attack annotations. Stage 03 freezes the union of those Stage-6 discovery branches without running a redundant held-out singleton intervention pass. Optional `attack_cohort_control_correctness` and `both` modes are controlled auxiliary analyses, not strictly attack-agnostic defense localizers.

The matched no-trigger control quantity is derived from the control half of `backdoor_trigger_test` for post-discovery evaluation. Separately, an optional `attack_cohort_control_correctness` Stage-03 CHA may reuse that same behavior cache as a controlled localization source. Some aggregate columns retain the `attack_cohort_control_correctness_` prefix for backward compatibility, while canonical Stage-07 causal columns use `paired_control_`.

## Stage 04 — matched condition comparison

Contains clean-versus-poisoned checkpoint comparison tables. Normal-task comparisons verify that both conditions use the same sampling population, cap, seed, strategy, stratum definition, and scanned row count.

## Stage 05 — behavior trajectories

Principal trajectory table:

```text
05_behavior_trajectories/<phase>/backdoor_lift_overtopping_trajectory.csv
```

It joins normal-task behavior and paired trigger/control behavior. The matched control-correctness behavioral summary is derived from the backdoor feature report; there is no separate attack-cohort control-correctness CHA. Post-discovery singleton causal metrics are materialized in Stage 07. Missing endpoints remain missing rather than being filled with zero.

`backdoor_overtopping_dashboard.pdf` creates panels only for metrics with finite values.

## Stage 06 — circuit overlap

Contains checkpoint stability/overlap comparisons for the attack-agnostic `observed_training_mixture_correctness` CHA candidate sets, plus optional configured reference analyses. Trigger and attack-cohort evaluation outcomes never enter candidate-set construction.

## Stage 07 — poisoning-example detection

### Defense-valid frozen candidate union

```text
defense_valid_candidate_union.csv
defense_valid_candidate_localization_by_checkpoint.csv
```

Candidate membership is localized from CHA on the defender-visible fine-tuning
prompt/label mixture, in its natural observed proportions, across matched
checkpoints. Hidden poison/attack annotations are excluded from selection. The
checkpoint table records the local attack-agnostic discovery score used by
prospective plots.

### Paired fixed-union materialization

```text
paired_u_j_materialization/
  <condition>/<checkpoint>/
    paired_feature_report/
    neuron_flip_rules/
    endpoint_stats/control/
    endpoint_stats/attack/   # poisoned checkpoints when attack support is adequate
```

One model load/ablation pass evaluates the frozen candidate union on matched no-trigger control and triggered attack views. Endpoint-specific summaries report the directional rate conditional on the behavior being present before intervention. Clean/poisoned 0% reuses the same pre-training Stage-03 generation cache and the same Stage-07 control materialization; condition-specific reports are still exported for downstream symmetry.

Independent clean-null runs are realigned inside Stage 07 to the primary run's immutable held-out identities and are stored under `clean_null_paired_u_j_materialization/`; historical per-seed `is_test` assignments do not require Stage-03 recomputation.

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

Current cache families include behavior-generation caches for `backdoor_trigger_test` and `normal_task`, plus causal-discovery caches for `observed_training_mixture_correctness` and, when enabled, `attack_cohort_control_correctness`. The two localization caches are independent and may coexist.

Cache reuse is conditional on matching population and method metadata. Changing the normal-task or observed-mixture population contract invalidates that endpoint's cache; prompt-matched trigger/control generations can remain reusable when their exact prompt identity and generation configuration are unchanged.
