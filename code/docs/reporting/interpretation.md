# Interpretation and limitations

## Frozen candidates

Overtopping candidates are selected by the discovery pipeline and evaluated on held-out data. Held-out effect sizes are confirmatory measurements for the frozen population; they need not equal discovery scores.

RQ3 also preserves discovery direction. A candidate is not transferred to the opposite directional source-state population unless it was independently discovered there.

## Singleton union and simultaneous intervention

`U(J)` is the union of singleton flip masks. `E(J)` is the effect of intervening on the complete set simultaneously.

```text
Delta_comp = E(J) - U(J)
```

The composition gap describes set-level behavior relative to singleton reach. It does not identify a unique interaction mechanism or a minimum sufficient coalition.

## Directional effects

`U_J_i2c` and `U_J_c2i` condition on different source-state populations. Their eligible counts can differ substantially. Directional rates should be interpreted together with their denominators and uncertainty.

## Width-normalized high-effect counts

Fields named `N_t_*_density` or `N_t_*_per_1k_layer` normalize discovered high-effect candidate counts by `d_model`. They are not estimates of causal-coordinate prevalence across the full searched coordinate space.

A coordinate-population prevalence estimate requires an explicit coordinate denominator and should separate coordinate families when their dimensions or intervention semantics differ.

## RQ3 threshold-event endpoints

The Figure 4 threshold panels report several distinct endpoints.

### Causal strength

Singleton flip rate measures intervention leverage. It does not measure endogenous threshold predictability.

### Threshold testability

Threshold fitting requires sufficient flip and non-flip support. Testability is therefore reported separately from threshold-fit quality. An untestable unit has no estimated threshold MCC.

### Nested held-out threshold MCC

The threshold-shape validation selects the scalar feature and fits the threshold on training-fold data, then evaluates it on held-out data. Absolute MCC measures binary threshold predictability on the evaluation fold.

### TECS

TECS combines causal strength and threshold visibility. The manuscript reports a nested lower-bound form so that units with unavailable threshold estimates are not silently treated as well-estimated threshold events.

### Model comparison

`fig4b_threshold_shape_model_comparison_by_direction.pdf` compares constant, hard-threshold, logistic, and isotonic held-out performance. It addresses predictive shape, not causal effect magnitude.

### Oriented proxy bins

`fig4s4_threshold_tail_response_by_direction.pdf` displays observed flip rates over oriented endogenous-proxy bins. The x-axis is a descriptive bin index, not a fitted threshold location.

## RQ3 graded intervention

The graded experiment conditions on each agonist's held-out directional full-dose flip support and varies the strength of the same intervention.

`first_flip_dose`, `n_state_changes`, and `single_crossing` describe the binary trajectory over the tested dose grid.

A `single_crossing` trajectory changes once from the natural baseline state and remains changed at every larger tested dose.

`natural_state_reproduction_rate` and `full_dose_support_reproduction_rate` are endpoint-consistency checks for dose 0 and dose 1. They are not single-crossing metrics.

Optional same-agonist non-flip support is a within-agonist reference. It is not a random-coordinate control.

Cross-run graded summaries use the run/baseline/direction condition as the inference unit after summarizing examples within agonists.

## Pythia checkpoint trajectories

Checkpoint-local candidate sets support population-level statements about causal organization at each checkpoint. Coordinate-level role-acquisition statements require the same aligned coordinate or fixed candidate set to be evaluated across checkpoints.

Pooled `U(J)` mixes the two directional source-state populations. Directional developmental claims should use directional trajectories.

## Poisoning behavior

`conditional_conversion_rate` measures trigger conversion among examples that are attack-eligible under the control prompt. It conditions the attack endpoint on examples that can meaningfully convert.

## Poisoning fixed-channel analysis

Absence from a checkpoint-local discovered set is not a measured zero causal effect for that coordinate. Longitudinal fixed-channel claims require explicit evaluation of the fixed coordinate or candidate union at each checkpoint.

## Prospective defense leverage

```text
Delta_def = attack_suppression - benign_correctness_damage
```

This is a singleton selectivity score for ranking candidate intervention targets. End-to-end defense efficacy requires applying the selected intervention policy and jointly measuring attack suppression and benign utility.

## Poisoning detector score

The WANDA-style training-example score combines activation magnitude, effective LoRA interval update, and causal-disruption weights. It is a ranking score rather than a calibrated poisoning probability.

Poison labels are used for post-hoc detector evaluation rather than score construction.

## Cross-seed inference

Training seed is the replication unit for poisoning trajectory inference. Prompt-level sample size improves within-trajectory precision but does not replace independent training runs.
