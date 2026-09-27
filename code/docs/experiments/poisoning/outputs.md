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
└── 07_poisoning_example_detection/  # longitudinal causal and exposure analyses
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

`observed_training_mixture_correctness` is the default attack-agnostic CHA localization endpoint. It evaluates each checkpoint model on the reconstructed fine-tuning prompt/label mixture available to the localization procedure. Candidate localization uses correctness with respect to the observed label, runs both observable baseline branches, and excludes hidden poison status and attack outcomes. Stage 03 freezes the union of the corresponding Stage-6 discovery branches. Optional `attack_cohort_control_correctness` and `both` modes use the attack-defined control cohort as an additional localization source.

The matched no-trigger control quantity is derived from the control half of `backdoor_trigger_test` for post-discovery evaluation. Separately, an optional `attack_cohort_control_correctness` Stage-03 CHA may reuse that same behavior cache as a controlled localization source. Canonical Stage-07 paired-control causal columns use the `paired_control_` prefix.

## Stage 04 — matched condition comparison

Contains clean-versus-poisoned checkpoint comparison tables. Normal-task comparisons verify that both conditions use the same sampling population, cap, seed, strategy, stratum definition, and scanned row count.

## Stage 05 — behavior trajectories

Principal trajectory table:

```text
05_behavior_trajectories/<phase>/backdoor_lift_overtopping_trajectory.csv
```

It joins normal-task behavior and paired trigger/control behavior. The matched control-correctness behavioral summary is derived from the backdoor feature report and does not require a separate attack-cohort localization run. If `attack_cohort_control_correctness` localization is enabled, that CHA output is a distinct Stage-03 candidate source rather than the source of this behavioral summary. Post-discovery singleton causal metrics are materialized in Stage 07. Missing endpoints remain missing rather than being filled with zero.

`backdoor_overtopping_dashboard.pdf` is emitted from the finite dashboard metrics available for the run.

## Stage 06 — circuit overlap

Contains checkpoint stability/overlap comparisons for the configured CHA candidate source. With the default `observed_training_mixture_correctness` endpoint, candidate construction is attack-agnostic and excludes trigger/attack labels. With `attack_cohort_control_correctness` or `both`, the configured candidate universe also uses the controlled oracle-defined attack-cohort localization described in the protocol.

## Stage 07 — longitudinal causal evaluation and exposure analysis

### Frozen candidate union

```text
defense_valid_candidate_union.csv
defense_valid_candidate_localization_by_checkpoint.csv
```

Under the default `observed_training_mixture_correctness` configuration, candidate membership is localized from the defender-visible fine-tuning prompt/label mixture, in its natural observed proportions, across matched checkpoints; hidden poison/attack annotations are excluded from selection. When the localization endpoint is `attack_cohort_control_correctness` or `both`, the frozen union follows that configured source contract instead. The checkpoint table records source-specific local discovery information used by downstream analyses.

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

Independent clean-null runs are aligned inside Stage 07 to the primary run's immutable held-out identities and are stored under `clean_null_paired_u_j_materialization/`. Per-seed `is_test` assignments therefore do not require Stage-03 recomputation.

### Defense-screen tables

The RQ4 previous-checkpoint and checkpoint-aligned clean-reference analyses use files including:

```text
defense_screen_comparison.csv
prospective_defense_leverage.csv
one_checkpoint_ahead_defense_screen.csv
clean_reference_defense_screen.csv
clean_reference_benign_budget_screen.csv
clean_reference_benign_budget_selected_channels.csv
clean_reference_benign_budget_curve.csv
clean_reference_benign_budget_checkpoint_summary.csv
```

The clean-reference budget table records attack suppression, benign correctness damage, and their difference for singleton channels across checkpoint/budget combinations.

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

### Checkpoint renderings

```text
01_clean_vs_poisoned_overtopping_development.pdf
02_channel_role_reassignment.pdf
03_prospective_defense_leverage.pdf
03b_clean_reference_defense_interpretation.pdf
03c_one_checkpoint_ahead_defense_interpretation.pdf
04_clean_vs_poisoned_checkpoint_overtopping.pdf
```

Auxiliary rendering-status files include:

```text
story_data_coverage.csv
story_figure_status.csv
```

`stage07_plot_overtopping_poisoning_story.py --figure6_from_story_dir <dir>` renders the clean-reference defense output from `clean_reference_benign_budget_screen.csv` and `clean_reference_benign_budget_curve.csv` without model access.

### Additional derived outputs

`overtopping_interpretation/` contains causal-role location, support-persistence, clean-versus-poisoned causal-drift, update-geometry, poison-detection, and attack-growth analyses.

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

Cache families include behavior-generation caches for `backdoor_trigger_test` and `normal_task`, plus causal-discovery caches for `observed_training_mixture_correctness` and, when enabled, `attack_cohort_control_correctness`. The two localization caches are independent and may coexist.

Cache reuse is conditional on matching population and method metadata. Changing the normal-task or observed-mixture population contract invalidates that endpoint's cache; prompt-matched trigger/control generations can remain reusable when their exact prompt identity and generation configuration are unchanged.
