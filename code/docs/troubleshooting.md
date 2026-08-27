# Troubleshooting

## Inspect the command plan

```bash
./run_poisoning_experiments.sh --dry-run
```

Confirm task, model, seed, run name, checkpoint fractions, trigger behavior, `normal_task`, `attack_cohort_control_correctness`, and Stage-07 settings.

## Training configuration mismatch

`01_training_checkpoints/metadata/run_config.json` identifies the run's training configuration. Checkpoints from different model, optimizer, LoRA, dataset, schedule, or checkpoint settings belong in separate run directories.

## Pre-training trigger/control diagnostic is large

The fraction-zero diagnostic measures marker sensitivity before training. It is reported separately from the control-correctness causal branch and Stage-07 row score.

## Trigger CHA is absent

With `RUN_TRIGGER_LIFT_CHA=0`, the backdoor trigger test still reports behavior. Stage 07 uses `attack_cohort_control_correctness`, not trigger-conditioned CHA.

Inspect:

```text
03_checkpoint_causal_discovery/<condition>/<checkpoint>/<phase>/attack_cohort_control_correctness/eval_<intervention>/
```

## Fraction 0 lacks control-correctness output

Stage 07 requires the fraction-zero causal reference. Rerun `scripts/run_checkpoint_causal_workflow.sh` for the missing checkpoint.

## Control-correctness cohort is rejected

The causal endpoint requires attack-eligible/non-target rows only and the encoding:

```text
is_correct_control = 1  correct
is_correct_control = 0  incorrect
```

The launcher checks row eligibility, correctness semantics, and equality of control prompts/outputs with the paired backdoor cache.

Full-cohort `normal_task` rows are not valid inputs for this CHA.

## No causal candidates are discovered

Stage 07 requires at least one attack-cohort control-correctness agonist somewhere in the matched clean/poisoned trajectory at the configured `tau`. Inspect the Stage-03 CHA status and `frozen_candidate_ranking.csv`.

## Fixed U(j) materialization starts model work

Stage 07 evaluates the union candidate set at every matched checkpoint on the fixed held-out attack cohort. Complete materializations are reused unless `--overwrite` is supplied.

## Cohort mismatch during U(j) comparison

Compared states require the same immutable held-out row identities and gold values. Regenerate the Stage-02 cohort and affected checkpoint outputs from one run configuration.

## Evaluation-count mismatch

Each union candidate needs the same fixed evaluation count at poisoned start/end and clean start/end. Missing measurements are not assigned zero.

## `trainer_random` schedule is rejected

Per-exposure scoring requires the persisted deterministic exposure order. Use `poison_schedule_mode=uniform_optimizer_steps`.

## Multi-epoch run is rejected

Stage 07 assigns one score per scheduled exposure and currently supports one training epoch.

## No disruptive channels are selected

Inspect:

```text
ordinary_channel_disruption.csv
```

Relevant columns are:

```text
complete_u_j_comparison
poisoning_excess_delta_u_j
disruption_score
clean_null_z
comparison_status
```

`CHA_TAU` controls candidate discovery. `--min_abs_delta_u` controls developmental effect size. `--min_clean_null_z` is optional and requires an adequate clean null.

## Clean-null runs are rejected

Additional clean trajectories require:

- distinct run directories;
- distinct training seeds;
- the same seed-independent scientific training configuration;
- matching checkpoint fractions and global steps.

A clean-null z-score requires at least three finite independent clean trajectories and positive sample variance for that channel/interval.

## Detectability/attack association has fewer intervals than expected

`detection_vs_attack_success_by_interval.csv` uses only checkpoint fractions present in the current Stage-04 backdoor trajectory. Missing start or end behavior makes interval-change metrics unavailable.

Inspect:

```text
04_condition_comparisons/<phase>/eval_<intervention>/backdoor_trigger_test/trajectory.csv
```

and:

```text
07_poisoning_example_detection/<phase>/detection_vs_attack_association.csv
```

`n_intervals` is the number of finite aligned interval pairs used by each test.

## Detectability/attack statistic is undefined

Spearman rho is undefined when one metric is constant over the finite aligned intervals. The association table reports `status=undefined_statistic`.

At least three finite interval pairs are required for a p-value.

## A selected channel does not map to a trained projection

MLP channels map to `mlp.down_proj`; attention channels map to `self_attn.v_proj`. The mapped projection must be a LoRA target in the run configuration.

Grouped-query attention can map several activation channels to one value-projection row.

## Raw WANDA scores differ strongly across intervals

Use `wanda_interval_percentile` for cross-interval ranking and `wanda_interval_robust_z` when defined. Raw WANDA values are interval-specific.

## Stage-07 output location

Run-local outputs belong under:

```text
data/poisoning/<task>/<run_id>/07_poisoning_example_detection/<phase>/
```

Stage 08 writes cross-run outputs under `data/poisoning/final/`.

## Import errors

Run Python modules from `code/`:

```bash
cd code
python3 -m studies.poisoning.stage07_detect_poisoning_examples --help
```

## Accelerator memory pressure

Stage 03 and Stage 07 load checkpoints sequentially. Avoid concurrent large checkpoint analyses on one accelerator.
