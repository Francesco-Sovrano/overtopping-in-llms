# Interpretation and limitations

## Frozen candidates

Overtopping candidates are selected by the discovery pipeline and then evaluated on held-out data. Held-out effect sizes can be smaller than discovery scores. A candidate that fails to reproduce a large held-out effect remains part of the frozen confirmatory candidate population.

RQ3 preserves the baseline subset in which each candidate was discovered. Directional candidate analyses should not pool a candidate into the opposite source-state population unless it was independently discovered there.

## Singleton union and simultaneous intervention

`U(J)` is the union of singleton flip masks. `E(J)` is the effect of intervening on the complete set simultaneously.

`E(J)-U(J)` describes how the simultaneous intervention relates to singleton-union reach. The gap alone does not identify saturation, preemption, masking, cancellation, or the minimum sufficient coalition.

## Directional effects

`U_J_i2c` and `U_J_c2i` condition on different source-state populations. Their eligible counts can differ substantially. Small directional denominators produce imprecise rates and should be interpreted with the reported counts and uncertainty.

## Width-normalized high-effect counts

Fields named `N_t_*_density` divide discovered high-effect candidate counts by `d_model`. They are width-normalized candidate counts, not estimates of the prevalence of causal coordinates in the complete searched space.

A coordinate-population prevalence estimate requires an explicit eligible-coordinate denominator. MLP and attention coordinate spaces should be treated separately when their dimensions or intervention semantics differ.

## RQ3 graded agonist intervention

Interpret RQ3 on the agonist's own known held-out directional flip support. The scientific object is the trajectory produced by increasing the strength of the same validated intervention, not a classifier that predicts which examples the agonist will flip.

A clean spiking-like trajectory changes behavioral state once and remains in the new state for all larger tested doses. Repeated reversals or failure to reproduce the Stage-7 full-dose endpoint should be reported directly rather than forced into a threshold-quality score.

Optional same-agonist negative support is a within-agonist reference and should not be described as a random-neuron control.

Legacy threshold `|MCC|`, threshold testability, and TECS outputs are descriptive/backwards-compatible only and are not manuscript-facing RQ3 evidence.

## Preemption

The existing pair-level preemption analysis is exploratory and separate from the graded agonist crossing experiment.

## Pythia checkpoint trajectories

Checkpoint-local candidate sets can demonstrate population-level reorganization. A claim that a specific coordinate acquires, loses, or changes a causal role requires fixed-coordinate evaluation across checkpoints.

Pooled `U(J)` mixes directional source-state populations and can change when behavioral prevalence changes. Directional trajectories are required for directional developmental interpretation.

## Poisoning behavior

`conditional_conversion_rate` measures trigger conversion among examples that are attack-eligible under the control prompt. It is the principal backdoor behavior quantity because it conditions on examples that can meaningfully convert.

## Poisoning fixed-union analysis

Checkpoint-local candidate absence is not evidence of zero causal effect. Longitudinal claims about a fixed candidate set require explicit evaluation of that set at each checkpoint.

## Prospective defense leverage

```text
Delta_def = attack_suppression - benign_correctness_damage
```

This is a singleton selectivity score for screening candidate intervention targets. It is not an end-to-end defense-efficacy estimate. End-to-end efficacy requires applying the selected intervention set and measuring attack suppression together with benign utility.

At a checkpoint before an attack behavior is established, causal effects can support candidate selection or calibration but should not be interpreted as suppression of an established attack.

## Poisoning detector score

The WANDA-style training-example score combines activation magnitude, effective LoRA interval update, and causal-disruption weights. It is a ranking score, not a calibrated poisoning probability.

Poison labels are excluded from candidate selection and scoring and are used only for post-hoc detector evaluation.

## Cross-seed inference

Training seed is the replication unit for poisoning trajectory inference. Prompt-level sample size improves precision within one trajectory but does not replace independent training runs.
