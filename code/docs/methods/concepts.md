# Core concepts

## Behavioral endpoint

Each task defines a binary behavioral endpoint:

```text
B(x) in {0, 1}
```

Its meaning is task-specific, such as correctness, acceptability, entailment behavior, or jailbreak success.

## Intervention phase

The overtopping study uses two execution phases:

- **input+output** (`standard` in execution configuration): intervention can affect prompt processing and decoding;
- **decode-only**: prompt processing is fixed and intervention is applied during decoding.

Phase is part of the scientific configuration and is retained in analysis tables.

## Evaluation split

`test` is the paper-facing overtopping evaluation split. `train` and `all` are explicit alternatives. Evaluation split is part of the experiment identity when multiple variants coexist.

## Directional source-state populations

Directional causal statistics condition on the unablated endpoint:

```text
negative baseline: B(x)=0 -> eligible for 0→1 / i2c
positive baseline: B(x)=1 -> eligible for 1→0 / c2i
```

The two directions have separate eligible denominators.

## Candidate discovery and held-out evaluation

Circuit discovery and candidate ranking occur before held-out causal evaluation. Stage 7 freezes candidate identity and materializes singleton intervention outcomes on the declared evaluation split.

For RQ3, discovery direction is part of candidate identity. A candidate discovered only on positive-baseline examples is analyzed in the 1→0 population; a candidate discovered only on negative-baseline examples is analyzed in the 0→1 population. A coordinate independently discovered in both subsets can appear in both directional analyses.

## Singleton causal effect

For channel `j`, let `F_j(x)=1` when replacing that channel with the configured intervention baseline changes `B(x)`.

Unconditional singleton effect:

```text
s_j = P(F_j = 1)
```

Directional forms:

```text
s_j_i2c = P(F_j = 1 | B = 0)
s_j_c2i = P(F_j = 1 | B = 1)
```

These are intervention-defined causal rates.

## Singleton-union reach

For a frozen candidate set `J`:

```text
U(J) = P(at least one singleton j in J flips the row)
```

Directional forms:

```text
U_J_i2c = P(any singleton flips | B = 0)
U_J_c2i = P(any singleton flips | B = 1)
```

`U(J)` is a union over singleton flip masks. It is not a sum of singleton effects and is not the strongest singleton effect.

## Simultaneous-set effect and composition gap

`E(J)` is the held-out behavioral effect of intervening on all channels in `J` simultaneously.

```text
Delta_comp = E(J) - U(J)
```

The sign identifies only the net difference between singleton-union reach and the full-set effect. It does not identify a unique interaction mechanism.

### Example-level decomposition

Let `S_J(x)` indicate that at least one singleton candidate flips the example and let `F_J(x)` indicate that the simultaneous full-set intervention flips it. Each complete-case held-out example belongs to exactly one class:

```text
preserved      S_J=1, F_J=1
suppressed     S_J=1, F_J=0
coalition_only S_J=0, F_J=1
unaffected     S_J=0, F_J=0
```

On the same population:

```text
U(J) = P(preserved) + P(suppressed)
E(J) = P(preserved) + P(coalition_only)
Delta_comp = P(coalition_only) - P(suppressed)
```

This identity is checked by Stage 8. High preservation with low suppression is compatible with a monotone saturating high-leverage regime. Substantial suppression indicates that singleton-reachable effects are lost under the simultaneous intervention. Coalition-only effects identify joint effects absent from singleton reach. These observations describe interaction structure; they do not by themselves establish a unique internal mechanism or prove that a coordinate is a necessary natural bottleneck.

## Matched-set specificity and conditional marginal contribution

RQ2 compares `J` with structurally matched non-candidate sets `K_b`.

Matched-set specificity uses the candidate-set effect relative to the matched-null distribution:

```text
E(J) - median_b E(K_b)
```

Stage 8 can also compute conditional marginal contribution (CMC). For a matched background set `S_b`:

```text
M_b(J)   = E(S_b ∪ J)   - E(S_b)
M_b(K_b) = E(S_b ∪ K_b) - E(S_b)
D_b      = M_b(J) - M_b(K_b)
```

Candidate, null, and background sets are evaluated as genuine simultaneous interventions.

## High-effect candidate counts

For threshold `t` and direction `d`:

```text
N_t^d = number of discovered candidates with s_j^d >= t
```

Reporting also provides width-normalized forms such as:

```text
N05_i2c_density
N05_c2i_density
N05_i2c_per_1k_layer
N05_c2i_per_1k_layer
```

These normalize discovered-candidate counts by model `d_model`. They are not estimates of the fraction of all model coordinates that are causal.

## Concentration and redundancy

The analysis separates causal reach from structural organization:

- `U^d`: how much of the directional source-state population is reachable by at least one singleton;
- high-effect counts/densities: how many discovered candidates exceed a causal-effect threshold;
- top-contribution and effective-support quantities: how concentrated the reachable mass is among candidates;
- redundancy quantities: how often multiple candidates cover the same examples.

These quantities answer different questions and should be interpreted separately.

## RQ3 threshold-event quantities

RQ3 uses candidate and structural-control units from the threshold-event diagnostics.

### Causal strength

The unit's observed held-out singleton flip rate is the causal-strength endpoint used in the candidate/control comparison.

### Threshold testability

A unit is threshold-testable when its evaluated population contains sufficient flip and non-flip support for the configured repeated train/holdout threshold analysis. Testability is reported as a separate endpoint because threshold-fit quality is undefined for untestable units.

### Nested held-out threshold MCC

Threshold-shape validation selects the endogenous scalar feature and fits the threshold on training folds, then evaluates the selected threshold on untouched held-out folds. Threshold quality is summarized with absolute Matthews correlation coefficient (`|MCC|`).

The nested procedure separates feature selection from held-out evaluation.

### TECS

Threshold-event causal score combines causal strength with held-out threshold visibility:

```text
TECS(j) = s_j * |MCC_j|
```

The manuscript analysis uses a nested TECS lower bound. Units with zero causal strength can have a defined zero lower bound; units whose threshold visibility is not estimable are handled according to the analysis status rather than assigned an arbitrary fitted score.

### Threshold-shape model comparison

For the same endogenous scalar data, the validation analysis compares held-out performance of:

```text
constant
hard threshold
logistic
isotonic
```

The comparison is aggregated by discovery direction and condition. It tests whether a threshold-shaped model provides predictive structure relative to a constant baseline and alternative monotonic/smooth models.

### Descriptive proxy-bin response

The support report also bins oriented endogenous proxy values and plots observed held-out flip rates. This is descriptive. The bin index is not an estimated causal threshold.

## Graded agonist intervention

For a frozen agonist `j`, the graded experiment compares two held-out source-state support classes: examples that the full Stage-7 singleton intervention flips and examples for which the same candidate remains non-flipping at full dose. It varies intervention strength as:

```text
h_j(lambda, x) = (1 - lambda) h_j(x) + lambda h_replacement_j
```

with:

```text
lambda = 0   natural channel value
lambda = 1   full configured replacement
```

The default dose grid is `0, 0.1, ..., 1.0`.

Per-example trajectory quantities include:

```text
first_flip_dose
n_state_changes
persistent_after_first_flip
single_crossing
```

A `single_crossing` trajectory leaves the natural baseline state once and remains in the opposite state at every larger tested dose.

Same-agonist non-flip support uses source-state examples where the same candidate was evaluated but did not flip the endpoint at full dose. It is a within-candidate support reference, not a random-coordinate control. The primary RQ3 contrast asks whether susceptible examples show localized persistent crossings while the non-flip support remains predominantly stable.

## Continuous endpoint-margin geometry

When endpoint-margin recording is enabled for the graded experiment, the pipeline also records continuous response margins over the same intervention doses. The primary quantity is the divergence-token margin at the first token where the natural and full-intervention completions differ.

The source activation follows a linear interpolation in dose, so the straight line joining the measured dose-0 and dose-1 margins is the endpoint-affine reference. The analysis compares the observed intermediate margins with that reference and reports affine-fit error, concentration of path variation, crossing alignment, monotonicity, and transient interior events. These are measures of downstream response geometry; they do not assume that the model implements a literal discontinuity.

## Persistent outputs and caches

A persistent output under `data/` records a scientific computation, population definition, or model-backed result. A cache under `cache/` or an experiment-local cache directory accelerates a computation that can be reconstructed from persistent inputs and configuration.

Derived analyses and figures under `results/` can be regenerated without deleting valid model-backed caches.
