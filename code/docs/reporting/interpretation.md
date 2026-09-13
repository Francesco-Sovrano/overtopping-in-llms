# Metric semantics and statistical units

## Frozen candidate population

Candidate identity and discovery direction are fixed before held-out causal evaluation. A coordinate discovered only on the positive-baseline population is evaluated in the 1→0 direction; a coordinate discovered only on the negative-baseline population is evaluated in the 0→1 direction.

## Singleton reach and simultaneous-set effect

For frozen candidate set `J`:

```text
U(J) = union of held-out singleton flip masks
E(J) = held-out flip rate under simultaneous intervention on J
Delta_comp = E(J) - U(J)
```

Stage 08 partitions a common complete-case evaluation population into:

```text
preserved      singleton reachable and full set flips
suppressed     singleton reachable and full set does not flip
coalition_only no singleton flips and full set flips
unaffected     neither flips
```

The decomposition satisfies:

```text
Delta_comp = P(coalition_only) - P(suppressed)
```

`Delta_comp` is a net set-level quantity. The decomposition supplies the two event classes that determine its sign and magnitude.

## Directional effects

`U_J_i2c` and `U_J_c2i` condition on different source-state populations. Report each estimate with its corresponding eligible denominator and uncertainty.

## Width-normalized high-effect counts

Fields such as `N05_i2c_density` normalize discovered high-effect candidate counts by model width. They summarize the discovered candidate population. They are not coordinate-universe prevalence estimates.

## Threshold-event quantities

### Causal strength

Causal strength is the held-out singleton flip rate in the frozen discovery direction.

### Threshold testability

Threshold fitting requires sufficient flip and non-flip support under the configured repeated train/holdout procedure. Units without the required support have no threshold-fit estimate.

### Nested held-out threshold MCC

Threshold-shape validation selects the scalar feature and fits the predictive model on training folds, then evaluates on untouched holdout folds. The principal score is absolute Matthews correlation coefficient, `|MCC|`.

### TECS

```text
TECS(j) = s_j * |MCC_j|
```

where `s_j` is held-out causal strength. TECS combines intervention leverage with scalar threshold visibility.

### Threshold-shape comparison

The held-out model comparison evaluates constant, hard-threshold, logistic, and isotonic predictors. It is separate from graded intervention measurements.

## Graded intervention

For candidate `j`, the graded intervention compares two source-state support classes:

```text
S_j+  full-dose singleton intervention flips the endpoint
S_j-  same full-dose singleton intervention does not flip the endpoint
```

Per-example trajectory fields include `first_flip_dose`, `n_state_changes`, `persistent_after_first_flip`, and `single_crossing`. A `single_crossing` trajectory changes once from the natural state and remains changed at every larger tested dose.

## Continuous endpoint-margin geometry

Endpoint-margin analysis records the continuous response margin over the graded dose grid. The endpoint chord is used as an affine-response reference. Reported quantities include chord-fit error, largest-step variation concentration, monotonicity, path variation, behavioral/margin crossing alignment, affine crossing predictions, and transient interior flips.

## Temporal cutoff

Prefix and suffix intervention sweeps localize when direct intervention is required during autoregressive generation. Cross-sweep EVENT validation defines the event location with one schedule and evaluates the aligned gain profile with the complementary schedule.

## Preemption

The optional preemption assay measures whether a secondary candidate's marginal binary effect differs according to an independently estimated dominant event. It is a separate interaction endpoint from `Delta_comp` and the singleton-versus-joint decomposition.

## Learning trajectories

Checkpoint-local candidate sets measure aggregate checkpoint-level causal organization. Coordinate-level role change requires evaluating the same aligned coordinate or fixed candidate set across checkpoints.

## Statistical units

The overtopping setting or run/condition is the cross-setting statistical unit unless an analysis explicitly defines another hierarchical model. Candidate, pair, and example rows are within-setting observations. Poisoning cross-seed analyses use the configured training seed/run as the replication unit unless a specific analysis states otherwise.
