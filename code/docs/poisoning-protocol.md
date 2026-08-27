# Poisoning protocol

This page defines the statistical populations, causal contrasts, and detector quantities used by the poisoning study.

## 1. Matched training trajectories

Each run contains `clean` and `poisoned` trajectories. They share model, seed, optimizer, LoRA configuration, training size, scheduling rule, checkpoint fractions, and task construction. The poisoned trajectory differs at the configured poison/counterfactual training slots.

Checkpoint comparisons use equal optimizer steps. Stage 07 requires `poison_schedule_mode=uniform_optimizer_steps` and one training epoch so the exposure sequence is defined row by row.

## 2. Stable evaluation cohorts

Stage 02 materializes evaluation rows once. Checkpoints reuse the same row identities.

Three endpoint populations are kept separate:

| Endpoint | Population | Prompt(s) | CHA |
|---|---|---|---|
| `normal_task` | full held-out task distribution | control only | no |
| `backdoor_trigger_test` | attack-eligible/non-target cohort | control + trigger | optional trigger CHA; disabled in the standard detector run |
| `attack_cohort_control_correctness` | same attack-eligible/non-target cohort | control only | yes |

The attack cohort contains rows whose gold answer is not the configured attack target.

## 3. Backdoor behavior

For each attack-cohort row, define whether the model predicts the target under the control and trigger prompts.

Reported quantities include:

```text
control target rate
trigger target rate
trigger-induced target-rate change
conditional conversion
```

Conditional conversion is

```text
P(target with trigger | not target without trigger).
```

This is the primary attack-efficacy metric used in the detectability/attack comparison.

## 4. Control-correctness endpoint

For the causal branch:

```text
is_correct_control = 1  control output matches the gold answer
is_correct_control = 0  control output does not match the gold answer
```

The cohort contains only attack-eligible rows. Hence the causal task does not mix target-gold and non-target-gold examples.

`OCC_1` and `OCC_0` are generic occupancy labels for the binary endpoint:

```text
OCC_1 = baseline correct
OCC_0 = baseline incorrect
```

## 5. Stage-05 candidate discovery

The poisoning launcher uses spectral discovery with the positive/correct subset and one discovery circuit:

```text
SPECTRAL_CLUSTER_BASE_SUBSET=positive
MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE=1
```

With one cluster, Stage 05 contrasts baseline-correct attack-cohort rows against baseline-incorrect attack-cohort rows. Candidate channels therefore arise from a correctness contrast within one attack-eligible cohort.

The default CHA candidate threshold is:

```text
tau = 0.3
```

`tau` belongs to causal candidate discovery.

## 6. Stage-06 causal test

Stage 06 uses:

```text
ANALYZE_BASELINE_SUBSETS=positive
```

The intervention analysis therefore starts from baseline-correct rows. Spectral associated and unrelated subsets are both sampled from that positive population.

For `baseline_subset=positive`, the observed slice effect is the wrong rate after intervention:

```text
delta_hat = (# rows incorrect after intervention) / n
```

A binomial upper confidence bound is computed separately for the associated and unrelated slices. The dichotomic search uses:

```text
max_effect = max(UCB(delta_associated), UCB(delta_unrelated))
```

and prunes a group when this upper bound is below the active search threshold. Singleton channels that meet the agonist criterion are retained as causal support channels.

## 7. Candidate union and fixed singleton evaluation

Let `J*` be the union of checkpoint-local attack-cohort control-correctness agonists across matched clean and poisoned checkpoints.

Stage 07 evaluates every `j in J*` at every matched checkpoint on the same fixed held-out attack-cohort rows. Local candidate absence is not converted to a zero effect.

For channel `j`:

```text
U_t(j) = c2i_count_t(j) / N_fixed
```

where `c2i_count` counts baseline-correct rows that become incorrect after singleton intervention.

The fixed denominator prevents checkpoint-dependent baseline accuracy from changing the evaluation population.

## 8. Developmental disruption

For interval `t0 -> t1`:

```text
Delta_p U(j) = U_p(j,t1) - U_p(j,t0)
Delta_c U(j) = U_c(j,t1) - U_c(j,t0)
D_j          = Delta_p U(j) - Delta_c U(j)
```

A channel is eligible for WANDA scoring when:

- all four `U(j)` values are available;
- `|D_j|` meets `--min_abs_delta_u`;
- the simultaneous paired-row bootstrap interval for `D_j` excludes zero;
- optional clean-null criteria are met.

The bootstrap uses the same resampled held-out row indices at poisoned start/end and clean start/end.

## 9. Normal-training null

The matched clean trajectory contributes one clean developmental change.

One clean trajectory is one realization of normal training. Stable statements that a channel's developmental drift is exceptional require independent clean trajectories.

Additional clean runs must use distinct seeds, the same seed-independent scientific training configuration, and matching checkpoint fractions and global steps.

A clean-null z-score is reported only when at least three finite independent clean trajectories are available and the sample variance is positive.

## 10. Effective LoRA update

For standard LoRA or rsLoRA:

```text
W_LoRA(t) = scaling * B(t) @ A(t)
```

For an interval:

```text
Delta W_p      = W_p(t1) - W_p(t0)
Delta W_c      = W_c(t1) - W_c(t0)
Delta W_excess = Delta W_p - Delta W_c
```

The detector uses `Delta W_excess` rather than one trajectory's raw adapter update.

## 11. Mapping causal channels to trainable rows

```text
mL:u      -> mlp.down_proj row u
aL.hH:u   -> corresponding self_attn.v_proj row
```

Grouped-query attention can map multiple causal activation channels to one value-projection row. The row is scored once and the original causal identities are retained in mapping outputs.

## 12. WANDA-style row score

The poisoned checkpoint at interval start supplies projection input activations. For mapped row `r` and training exposure `x`:

```text
WANDA_r(x) = mean_scored_tokens sum_d |x_d| * |Delta W_excess[r,d]|
```

Mapped rows are weighted by their causal disruption magnitude. The resulting `wanda_disruption_score` ranks training rows within an interval.

`is_poisoned` is not used in channel discovery, developmental channel selection, or score construction.

## 13. Matched parameter-row control

For each selected mapped parameter row, Stage 07 selects non-candidate rows from the same projection with similar `Delta W_excess` row norm. Multiple matched realizations provide a specificity-control distribution for detector metrics.

## 14. Detector evaluation

Ground-truth poison labels are applied after scores are computed.

Metrics include:

```text
ROC AUC                         chance 0.5
average precision              chance = poison prevalence
poison recovery in top N       N = true poison count
matched poison-over-source     chance 0.5
```

Raw WANDA magnitudes are interval-specific. Cross-interval ranking uses within-interval percentile and robust z-score.

## 15. Detectability/attack association

For interval `[t0,t1]`, define:

```text
R = interval ROC AUC
A0 = poisoned conditional conversion at t0
A1 = poisoned conditional conversion at t1
Delta A = A1 - A0
```

The primary association test is:

```text
Spearman(R, Delta A)
```

with a two-sided permutation p-value. For at most 9 finite intervals, the implementation enumerates all permutations. For larger samples it uses 100,000 Monte Carlo permutations with a fixed seed.

Secondary tests use:

```text
ROC AUC vs A1
ROC AUC vs (Delta A_poisoned - Delta A_clean)
top-N poison recovery vs Delta A
```

The association table reports the number of finite aligned intervals, Spearman rho, p-value, permutation method, and number of permutations. The primary test is designated explicitly; secondary tests are not multiplicity-adjusted. The interval permutation test treats finite interval pairs as exchangeable under the null and does not model checkpoint serial dependence.

## 16. Cross-seed inference

Stage 08 aggregates independent run/seed outputs. One run supplies one developmental trajectory pair. Cross-seed statements use the number of finite independent runs for the metric under analysis.
