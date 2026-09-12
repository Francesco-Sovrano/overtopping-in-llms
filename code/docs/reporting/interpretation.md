# Interpretation guide

This document states what the principal reported quantities do and do not identify.

## Frozen candidate populations

Candidates are selected by the discovery pipeline and evaluated on held-out data. Held-out causal effects measure the frozen population and do not determine candidate membership.

Directional candidate identity is preserved. A candidate discovered only on the positive-baseline population is evaluated for the 1→0 direction; a candidate discovered only on the negative-baseline population is evaluated for the 0→1 direction.

## Singleton reach and simultaneous composition

`U(J)` is the union of singleton flip masks. `E(J)` is the behavioral effect of a genuine simultaneous intervention on the complete candidate set.

```text
Delta_comp = E(J) - U(J)
```

The composition gap is a net quantity. It is not a sum-of-effects test and it does not identify a unique mechanism.

### Example-level decomposition

Stage 8 partitions the common complete-case evaluation population into:

```text
preserved      singleton reachable, full set also flips
suppressed     singleton reachable, full set does not flip
coalition_only not singleton reachable, full set flips
unaffected     neither flips
```

The decomposition satisfies

```text
Delta_comp = P(coalition_only) - P(suppressed).
```

Interpretation:

- high preservation and low suppression are compatible with monotone saturation of a high-leverage pathway;
- substantial suppression means singleton-reachable effects are lost under the full-set intervention and requires an interaction explanation beyond simple monotone saturation;
- coalition-only effects indicate joint effects absent from singleton reach;
- the decomposition does not establish that a channel is a necessary natural bottleneck in the intact computation.

## Directional effects

`U_J_i2c` and `U_J_c2i` condition on different source-state populations. Their denominators can differ. Directional estimates should be interpreted with the corresponding eligible counts and uncertainty.

## Width-normalized high-effect counts

Fields such as `N05_i2c_density` normalize discovered high-effect candidate counts by model width. They describe the discovered candidate population and are not estimates of the prevalence of causal coordinates across the full model.

A population prevalence estimate requires explicit sampling from a declared coordinate universe.

## RQ3 threshold-event endpoints

### Causal strength

Singleton flip rate measures intervention leverage. It does not measure endogenous threshold predictability.

### Threshold testability

Threshold fitting requires sufficient flip and non-flip support. Testability is reported separately from threshold-fit quality; an untestable unit has no estimated threshold MCC.

### Nested held-out threshold MCC

Threshold-shape validation selects the scalar feature and fits the model on training folds, then evaluates on held-out folds. Absolute MCC measures predictive performance on the held-out fold.

### TECS

TECS combines causal strength and threshold visibility:

```text
TECS(j) = s_j * |MCC_j|.
```

It is a visibility-weighted causal score, not evidence of a literal one-dimensional internal threshold.

### Threshold-shape comparison

The held-out constant, hard-threshold, logistic, and isotonic comparison addresses predictive shape. It does not replace the intervention-based graded experiment.

## Graded intervention

The primary graded analysis compares the same candidate on two source-state support classes:

- `S_j+`: full-dose singleton intervention flips the endpoint;
- `S_j-`: the same full-dose singleton intervention does not flip the endpoint.

`first_flip_dose`, `n_state_changes`, and `single_crossing` describe the binary trajectory over the tested dose grid. A `single_crossing` trajectory changes once from the natural state and remains changed at all larger tested doses.

The main RQ3 inference is support specificity: susceptible examples should show structured persistent crossings while same-candidate non-flip examples remain predominantly stable.

The support contrast does not by itself estimate the unconditional prevalence of crossings in the full source-state population.

## Preemption

The preemption assay is a secondary interaction analysis. It tests whether a secondary candidate's binary marginal effect differs according to an independently estimated dominant event. Positive evidence can support a first-sufficient subtype, but null or heterogeneous preemption does not negate support-specific graded threshold crossing.

Preemption should not be inferred from a negative `E(J)-U(J)` gap alone.

## Learning trajectories

Checkpoint-local candidate sets support statements about aggregate causal organization at each checkpoint. Coordinate-level acquisition or reassignment requires explicit evaluation of the same aligned coordinate or fixed candidate set across checkpoints.

Pooled `U(J)` mixes directional source-state populations. Directional developmental claims should use directional trajectories.

## Controlled hidden-objective training

Behavioral conversion and ordinary-task performance establish the learning trajectory of the controlled objective. Fixed-coordinate causal-role claims require the same coordinate to be measured across matched checkpoints and conditions.

## Statistical units

The experimental setting or run/condition is the manuscript replication unit for cross-setting overtopping analyses. Candidate-level, pair-level, and example-level rows describe within-setting structure unless a separate hierarchical inferential model is specified.
