# Core concepts

## Behavioral endpoint

Each task defines a binary behavioral endpoint `B(x) ∈ {0,1}`. Its meaning is task-specific, for example correctness, acceptability, entailment behavior, or jailbreak success.

## Intervention phase

Two phase labels are used throughout the overtopping study:

- **input+output** (`standard` in execution configuration): interventions can affect prompt processing and generation;
- **decode-only/output-only** (`decode-only`): prompt processing is fixed and intervention is applied during decoding.

Phase is part of the scientific configuration.

## Evaluation split

`test` is the paper-facing overtopping evaluation split. `train` and `all` are explicit alternatives. Evaluation-split suffixes are part of the experiment path when needed to distinguish coexisting artifacts.

## Directional source-state populations

Directional causal statistics condition on the unablated endpoint:

- `negative`: `B(x)=0`, eligible for 0→1 (`i2c`) effects;
- `positive`: `B(x)=1`, eligible for 1→0 (`c2i`) effects.

The two directions have separate eligible denominators.

## Candidate discovery and held-out evaluation

Candidate discovery identifies channels on a discovery population. Held-out causal evaluation measures intervention outcomes for the frozen candidates on a separate evaluation population.

For RQ3, candidate identity also carries the discovery baseline subset. A candidate discovered only on positive-baseline examples belongs only to the 1→0 candidate population; a candidate discovered only on negative-baseline examples belongs only to the 0→1 candidate population. A channel discovered in both subsets may enter both directional analyses.

## Singleton causal effect

For channel `j`, define `F_j(x)=1` when replacing channel `j` with the configured baseline changes the binary behavioral endpoint.

The unconditional singleton effect is:

```text
s_j = P(F_j = 1)
```

Directional forms are:

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

Directional forms are:

```text
U_J_i2c = P(any singleton flips | B = 0)
U_J_c2i = P(any singleton flips | B = 1)
```

`U(J)` is a union over singleton flip masks. It is neither the strongest singleton effect nor a sum of singleton effects.

## Simultaneous-set effect and composition gap

`E(J)` is the behavioral effect of intervening on all channels in `J` simultaneously.

```text
Delta_comp = E(J) - U(J)
```

Interpretation:

- `Delta_comp < 0`: the joint intervention flips fewer rows than are reachable by at least one singleton;
- `Delta_comp = 0`: equality;
- `Delta_comp > 0`: the joint intervention flips rows beyond singleton-union reach.

The sign alone does not identify the detailed interaction mechanism. Negative gaps can arise from overlap, saturation, masking, cancellation, or preemption. Positive gaps identify joint effects beyond singleton-union reach but do not identify the minimum sufficient coalition.

## High-effect candidate counts

For threshold `t` and direction `d`:

```text
N_t^d = number of discovered candidates with s_j^d >= t
```

The code also reports fields named `N_t_*_density` and `N_t_*_per_1k_layer`, which normalize these discovered-candidate counts by `d_model`.

This normalization controls a simple width scale. It is not the fraction of all eligible searched coordinates. A coordinate-population prevalence estimate requires the actual searched-coordinate denominator and should distinguish MLP and attention coordinate spaces.

## Effective support, concentration, and redundancy

`N_eff` summarizes effective causal support. Top-contribution quantities summarize how concentrated union reach is among the strongest candidates. Redundancy quantities summarize repeated coverage of the same examples by multiple singleton candidates.

These quantities describe different properties and should not be treated as interchangeable with union reach.

## RQ3 graded causal crossing

For frozen agonist `j` in discovery direction `d`, RQ3 uses the held-out examples already known to be flipped by the full singleton intervention. The intervention strength is then varied continuously:

```text
h_j(lambda) = (1-lambda) h_j + lambda h_replacement_j
```

`lambda=0` is the natural activation and `lambda=1` is the established full replacement. The primary quantities are first flip dose, number of state reversals, persistence after the first crossing, and the fraction of trajectories with one persistent crossing.

Optional same-agonist negative support consists of source-state examples for which the same full singleton intervention does not flip the behavior. This is a within-agonist reference, not a random-neuron control.

Nested held-out threshold `|MCC|` and TECS are not primary RQ3 endpoints. Legacy scalar-to-flip diagnostics may remain for inspection, but they do not establish the spiking phenotype.

## Preemption

The separate interaction-stage preemption analysis remains exploratory and is not required for the primary graded causal-crossing test.

## Persistent outputs and caches

A persistent output under `data/` records a scientific computation or population definition. A cache under `cache/` or an experiment cache directory accelerates a computation that can be reconstructed from persistent inputs.

Deleting a derived report does not require deleting model-backed caches unless the scientific configuration or cached population changed.
