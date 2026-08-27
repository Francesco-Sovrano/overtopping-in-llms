# Poisoning study overview

The poisoning study tracks matched clean and poisoned training trajectories and asks two questions:

1. when the configured trigger acquires behavioral effect;
2. whether poisoned training rows can be ranked from abnormal developmental changes in channels that support correct control behavior on attack-eligible examples.

The detector uses control-correctness channels. Trigger behavior is measured separately and is not an input to channel discovery or WANDA scoring.

## Experimental unit

One run is defined by task, model, and training seed. It contains matched `clean` and `poisoned` trajectories with the same model family, seed, optimizer, LoRA configuration, training size, schedule, and checkpoint fractions. Checkpoints are matched by optimizer step.

The default checkpoints are 0%, 10%, 25%, 50%, 75%, and 100% of training.

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

Stage 08 aggregates independent runs under `data/poisoning/final/`.

## Evaluation populations

The checkpoint workflow uses three separate endpoints.

### Normal task behavior

`normal_task` evaluates the full held-out task distribution with the control/no-trigger prompt.

It reports normal task accuracy and class-conditional accuracy. It does not run CHA.

### Backdoor trigger test

`backdoor_trigger_test` evaluates the attack-eligible cohort: examples whose gold answer is not the configured attack target.

Each row is evaluated both without and with the trigger. The main behavioral quantities are:

```text
target_rate_without_trigger
target_rate_with_trigger
trigger_induced_target_rate_change
conversion_rate_among_convertible_examples
```

Conditional conversion is

```text
P(target with trigger | not target without trigger)
```

and measures trigger-induced conversion among examples that were not already at the target under the control prompt.

### Attack-cohort control-correctness CHA

`attack_cohort_control_correctness` uses the same attack-eligible cohort as the backdoor test, but only the control/no-trigger prompt.

The binary endpoint is:

```text
is_correct_control = 1  correct control answer
is_correct_control = 0  incorrect control answer
```

Thus:

```text
OCC_1 = baseline-correct attack-cohort rows
OCC_0 = baseline-incorrect attack-cohort rows
```

For grammar with target label `acceptable`, the attack cohort consists of gold `unacceptable` examples. For arithmetic, it consists of examples whose correct answer differs from the configured target answer.

## Checkpoint causal analysis

Stage 03 uses `attack_cohort_control_correctness` for the causal branch used by Stage 07.

### Stage 05 discovery contrast

The task target is `is_correct_control`. With the poisoning configuration:

```text
SPECTRAL_CLUSTER_BASE_SUBSET=positive
MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE=1
```

Stage 05 forms one spectral cluster from the baseline-correct rows. With one cluster, the associated side is the attack-cohort rows with `is_correct_control=1`, and the comparison side is the remaining attack-cohort rows with `is_correct_control=0`.

The discovery contrast is therefore:

```text
correct control response  vs  incorrect control response
```

within one homogeneous attack-eligible gold-label cohort.

### Stage 06 CHA contrast

Stage 06 runs with:

```text
ANALYZE_BASELINE_SUBSETS=positive
```

so intervention effects are evaluated on rows that are correct before intervention. Spectral associated and unrelated subsets are both drawn from this baseline-correct population.

For a neuron or neuron group, the positive-subset effect is the post-intervention wrong rate. The search records the larger upper confidence bound across the associated and unrelated slices and compares it with the active CHA threshold. Singleton agonists are channels whose intervention produces a sufficiently large correctness loss under this procedure.

This analysis asks which channels causally support the correct non-target answer on examples that the attack could convert.

## Fixed-cohort singleton strength

Stage 07 forms the union `J*` of checkpoint-local attack-cohort control-correctness agonists and explicitly evaluates each union channel at every matched clean and poisoned checkpoint.

For channel `j` at checkpoint `t`:

```text
U_t(j) = c2i_count_t(j) / N_fixed
```

where `c2i_count` is the number of fixed held-out attack-cohort rows that are correct at baseline and become incorrect under singleton intervention. `N_fixed` is the full fixed held-out attack-cohort size used for that materialization.

Candidate membership and `U(j)` are separate quantities. A channel absent from a checkpoint-local candidate list is still evaluated if it belongs to `J*`.

## Developmental disruption

For adjacent matched checkpoints `t0 -> t1`:

```text
Delta_p U(j) = U_p(j,t1) - U_p(j,t0)
Delta_c U(j) = U_c(j,t1) - U_c(j,t0)
D_j          = Delta_p U(j) - Delta_c U(j)
disruption   = |D_j|
```

`D_j` is the poisoned trajectory's excess developmental change relative to the matched clean trajectory.

Stage 07 selects disruptive channels using a minimum `|D_j|` and a simultaneous paired-row bootstrap interval. `tau` remains the CHA candidate-discovery threshold and is not reused as a developmental-effect threshold.

## Normal-training null

One clean trajectory is one realization of normal training. Stable statements that a channel's developmental drift is exceptional require independent clean trajectories.

Additional clean trajectories supplied through `--clean_null_run_dirs` use distinct seeds and the same seed-independent scientific training configuration, checkpoint fractions, and optimizer steps.

Clean-null z-scores are reported only when at least three finite independent clean trajectories are available for the channel/interval and their sample variance is positive.

## Training-row anomaly score

Selected causal channels are mapped to direct LoRA write rows:

```text
mL:u      -> layer L mlp.down_proj row u
aL.hH:u   -> corresponding layer L self_attn.v_proj row
```

For each projection:

```text
W(t) = scaling * B(t) @ A(t)

Delta W_excess = [W_p(t1)-W_p(t0)] - [W_c(t1)-W_c(t0)]
```

For each training exposure, Stage 07 combines poisoned interval-start projection inputs with `|Delta W_excess|` in a WANDA-style row score and weights mapped rows by channel disruption. The score is used to rank training rows.

Ground-truth `is_poisoned` labels are read after scoring for detector evaluation.

## Detector metrics

Interval-level detector outputs include:

- ROC AUC;
- average precision;
- poison recovery when inspecting the top `N` rows, with `N` equal to the true poison count in that interval;
- matched poison-versus-source ranking rate;
- matched non-candidate parameter-row controls.

Cross-interval suspect tables use within-interval percentile and robust z-score because raw WANDA magnitudes are interval-specific.

## Detectability versus backdoor efficacy

Stage 07 aligns each training interval with Stage-04 backdoor behavior.

The primary attack metric is conditional conversion. Two alignments are reported:

```text
interval detectability     -> ROC AUC for training rows in [t0,t1]
interval backdoor change   -> conditional_conversion(t1) - conditional_conversion(t0)
interval-end efficacy      -> conditional_conversion(t1)
```

The primary statistical test is a two-sided permutation Spearman test between interval ROC AUC and the change in poisoned conditional conversion over the same interval. All permutations are enumerated when the number of finite intervals is at most 9; larger samples use 100,000 deterministic Monte Carlo permutations.

Secondary tests compare:

- ROC AUC with interval-end poisoned conditional conversion;
- ROC AUC with clean-adjusted conversion change;
- top-`N` poison recovery with poisoned conversion change.

The primary test is designated in the output table. Secondary tests are unadjusted sensitivity analyses. The within-run permutation test treats the finite interval pairs as exchangeable under the null; checkpoint serial dependence is not modeled. Cross-seed aggregation provides the independent-run replication layer.

## Stage map

| Stage | Output |
|---|---|
| 01 | matched clean and poisoned checkpoint trajectories |
| 02 | stable evaluation cohorts |
| 03 | backdoor behavior, normal-task behavior, attack-cohort control-correctness CHA |
| 04 | matched clean/poisoned behavior comparisons |
| 05 | developmental behavior and causal summaries |
| 06 | checkpoint circuit overlap |
| 07 | poisoning-row ranking, fixed `U(j)`, detectability/attack association |
| 08 | cross-seed aggregation |
