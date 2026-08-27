# Interpretation and limitations

## Causal-channel meaning

The Stage-07 causal branch is `attack_cohort_control_correctness`. Its population contains attack-eligible/non-target examples only.

```text
is_correct_control = 1  correct control response
is_correct_control = 0  incorrect control response
```

Stage 05 uses this correctness contrast for candidate discovery. Stage 06 evaluates intervention-induced correctness loss on baseline-correct rows.

A checkpoint-local agonist is a thresholded causal-discovery result. Candidate absence does not imply zero singleton effect.

## Fixed singleton strength

Stage 07 measures every union candidate at every matched checkpoint:

```text
U_t(j) = c2i_count_t(j) / N_fixed
```

`c2i` is correct at baseline and incorrect after singleton intervention. `N_fixed` is the fixed held-out attack-cohort denominator.

The discovery score and `U(j)` are different quantities. The discovery score can contain confidence-bound terms used by the search; `U(j)` is the observed fixed-cohort event rate.

## Developmental disruption

```text
D_j = [U_p(j,t1)-U_p(j,t0)] - [U_c(j,t1)-U_c(j,t0)]
```

`D_j` is a matched developmental contrast. The clean and poisoned trajectories do not share the same state at interval start, so `D_j` is not a same-start causal effect of the rows consumed during the interval.

## Normal-training variability

One clean trajectory is one realization of normal training. Stable statements that a channel's developmental drift is exceptional require independent clean trajectories.

Clean-null z-scores are reported only with at least three finite independent clean trajectories and positive sample variance for the channel/interval. One matched clean trajectory is sufficient for the paired developmental contrast `D_j`, but not for estimating a stable normal-training null distribution.

## WANDA score

The detector uses projection input activations and the clean-normalized effective LoRA update:

```text
Delta W_excess = [W_p(t1)-W_p(t0)] - [W_c(t1)-W_c(t0)]
```

The WANDA-style score is a training-row anomaly score. It is not an influence-function or optimizer-update decomposition.

MLP causal channels map to `down_proj`; attention channels map to `v_proj`. Attention scores therefore measure value-projection engagement and do not include Q/K routing effects.

## Matched parameter controls

Matched non-candidate rows control for projection identity, cardinality, and approximately for clean-normalized update-row norm. They provide a specificity comparison for the selected rows.

## Detectability metrics

ROC AUC and matched poison/source ranking have chance level 0.5. Average precision has chance level equal to poison prevalence. Top-`N` poison recovery uses `N` equal to the true poison count for post-hoc evaluation.

Raw WANDA scores are not compared directly across intervals. Cross-interval suspect tables use within-interval percentile and robust z-score.

## Detectability versus attack efficacy

The primary attack metric is conditional conversion:

```text
P(target with trigger | not target without trigger)
```

The primary statistical comparison pairs each interval's detector ROC AUC with the change in poisoned conditional conversion over the same interval and uses a two-sided permutation Spearman test.

A secondary test pairs ROC AUC with the interval-end conditional conversion level. Another subtracts the clean trajectory's conversion change before testing association.

These tests summarize association within one developmental run. The primary test is designated explicitly; secondary tests are unadjusted. The permutation null treats finite interval pairs as exchangeable and does not model checkpoint serial dependence. Cross-seed stability is evaluated from independent runs.

## Ground-truth use

The configured trigger and poison labels are used to define the poisoning experiment and to evaluate detector performance. Trigger behavior does not enter attack-cohort control-correctness channel discovery or WANDA score construction. `is_poisoned` is read after row scores are computed.

## Training contract

Stage 07 currently requires one training epoch and `uniform_optimizer_steps`. The detector operates on the persisted exposure sequence and assigns one score to each scheduled exposure.
