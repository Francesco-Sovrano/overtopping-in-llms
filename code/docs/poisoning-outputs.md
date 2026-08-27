# Poisoning outputs

This page lists the run-local outputs produced by the poisoning workflow.

## Run root

```text
data/poisoning/<task>/<run_id>/
├── 01_training_checkpoints/
├── 02_evaluation_cohorts/
├── 03_checkpoint_causal_discovery/
├── 04_condition_comparisons/
├── 05_behavior_trajectories/
├── 06_circuit_overlap_analysis/
└── 07_poisoning_example_detection/
```

Cross-seed Stage 08 outputs are stored under `data/poisoning/final/`.

## Stage 01 — training checkpoints

Each condition has a checkpoint manifest and checkpoint directories. Metadata records the training configuration, poison plan, schedule, and checkpoint identity.

## Stage 02 — evaluation cohorts

Stable evaluation rows are stored under:

```text
02_evaluation_cohorts/
```

The same row identities are reused across matched checkpoints.

## Stage 03 — checkpoint evaluation and causal discovery

For each condition and checkpoint:

```text
03_checkpoint_causal_discovery/
└── <condition>/
    └── <checkpoint>/
        └── <phase>/
            ├── backdoor_trigger_test/
            │   └── eval_<intervention>/
            │       └── feature_report/
            ├── normal_task/
            │   └── eval_<intervention>/
            │       └── feature_report/
            └── attack_cohort_control_correctness/
                └── eval_<intervention>/
                    ├── feature_report/
                    ├── neural_circuit_discovery_results/
                    ├── rule_extraction_results/
                    └── attack_cohort_control_correctness_status.json
```

The endpoint meanings are:

- `backdoor_trigger_test`: attack-eligible cohort, control and trigger behavior;
- `normal_task`: full held-out distribution, control/no-trigger behavior only;
- `attack_cohort_control_correctness`: attack-eligible cohort, control/no-trigger correctness CHA.

The causal endpoint uses `is_correct_control` with `1=correct` and `0=incorrect`.

Some compatibility files and fields retain the legacy `ordinary_correctness` name. They refer to the attack-cohort control-correctness causal endpoint, not to the full-cohort `normal_task` behavior endpoint.

## Stage 04 — matched behavior comparison

Stage 04 stores clean/poisoned checkpoint comparisons and trajectories under:

```text
04_condition_comparisons/<phase>/eval_<intervention>/
```

Behavior groups are separated into:

```text
backdoor_trigger_test/
normal_task/
```

The backdoor trajectory provides:

```text
target_rate_without_trigger
target_rate_with_trigger
trigger_induced_target_rate_change
conversion_rate_among_convertible_examples
fraction_convertible_without_trigger
non_target_to_target_flip_rate
```

`conversion_rate_among_convertible_examples` is the primary attack-efficacy input for the Stage-07 detectability/attack comparison.

## Stage 05 — developmental trajectories

Stage 05 aggregates checkpoint behavior and causal summaries. These tables are descriptive trajectory outputs and are separate from the fixed-candidate Stage-07 materializations.

## Stage 06 — circuit overlap

Stage 06 compares checkpoint-local candidate identities and overlap across clean and poisoned trajectories.

## Stage 07 — poisoning-example detection

Stage 07 writes under:

```text
07_poisoning_example_detection/<phase>/
```

### `ordinary_candidate_union.csv`

Compatibility filename for the union of attack-cohort control-correctness agonist channels discovered across matched clean and poisoned checkpoints.

### `ordinary_u_j_materialization/`

Compatibility directory containing fixed-cohort singleton evaluations for every union candidate at every matched checkpoint.

Typical structure:

```text
ordinary_u_j_materialization/
├── clean/
│   └── <checkpoint>/neuron_flip_rules/
└── poisoned/
    └── <checkpoint>/neuron_flip_rules/
```

The fixed singleton strength is:

```text
U(j) = c2i_count / N_fixed
```

on the held-out attack-eligible cohort.

### `ordinary_channel_disruption_by_interval.csv`

Compatibility filename for per-channel developmental disruption. Important columns include:

```text
poisoned_delta_u_j
clean_delta_u_j
poisoning_excess_delta_u_j
disruption_score
clean_null_z
comparison_status
```

### `selected_disruptive_channels_by_interval.csv`

Channels retained after effect-size, simultaneous bootstrap, and optional clean-null criteria.

### `mapped_disruptive_channels.csv`

Interval-local mapping from causal channel identities to LoRA projection rows.

### `training_exposure_order.csv`

Label-neutral persisted training exposure order used for interval scoring.

### `training_example_scores_all_intervals.csv`

One row per scored training exposure. Important columns include:

```text
wanda_disruption_score
wanda_interval_percentile
wanda_interval_robust_z
is_poisoned
```

`is_poisoned` is added for post-score evaluation.

### `top_suspected_training_examples.csv`

Run-level top rows sorted by within-interval percentile, then robust z-score, then raw WANDA score.

### `detection_metrics_by_interval.csv`

Interval-level detector metrics. It includes candidate metrics, matched-control summaries, poison prevalence, and top-`N` recovery.

### `detection_vs_attack_success_by_interval.csv`

Joins detector metrics with Stage-04 backdoor behavior.

Key fields include:

```text
roc_auc
poison_recovery_in_top_n
poisoned_conversion_rate_start
poisoned_conversion_rate_among_convertible_examples
poisoned_conversion_change_over_interval
clean_conversion_change_over_interval
conversion_change_gain_poisoned_vs_clean
```

The interval alignment is:

```text
detector metric: rows scored in [t0,t1]
attack change:   conditional_conversion(t1) - conditional_conversion(t0)
attack level:    conditional_conversion(t1)
```

### `detection_vs_attack_association.csv`

Statistical tests of detectability versus backdoor efficacy.

The primary row is:

```text
test_name = primary_auc_vs_poisoned_conversion_change
detectability_metric = roc_auc
attack_metric = poisoned_conversion_change_over_interval
statistic = spearman_rho
alternative = two_sided
```

For at most 9 finite intervals, the p-value is obtained by enumerating all permutations. Larger samples use 100,000 Monte Carlo permutations with a fixed seed.

Secondary rows test interval-end conversion, clean-adjusted conversion change, and top-`N` recovery. `multiplicity_adjustment=none` is recorded in the table.

### `detection_vs_attack_association.json`

JSON representation of the association table.

### Visualizations

```text
poisoning_example_detection_metrics.pdf
poisoning_example_score_distribution.pdf
poisoning_detection_vs_attack_success.pdf
poisoning_detectability_attack_association.pdf
```

The association figure plots interval ROC AUC against the change in poisoned conditional conversion over the same interval and reports the primary Spearman statistic and permutation p-value.

### `detection_summary.json`

Run-level Stage-07 metadata, including:

- causal endpoint and candidate definition;
- `U(j)` definition;
- WANDA score definition;
- matched-control configuration;
- normal-training control count;
- backdoor behavior source;
- primary detectability/attack association result.

## Clean-null outputs

When `--clean_null_run_dirs` is supplied, Stage 07 materializes the same fixed union on each additional clean trajectory.

One clean trajectory is one realization of normal training. Clean-null z-scores are reported only when at least three finite independent clean trajectories are available for the channel/interval and the sample variance is positive.

## Cache directories

Regenerable caches are stored under the configured `POISONING_CACHE_ROOT`. The main endpoint cache groups are:

```text
backdoor_trigger_test/
normal_task_behavior/
attack_cohort_control_correctness/
```

Scientific outputs are under the run directory; cache directories are implementation accelerators.
