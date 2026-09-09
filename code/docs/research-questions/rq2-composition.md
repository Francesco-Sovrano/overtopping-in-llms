# RQ2 — composition and interaction regimes

## Question

> **How do singleton-reachable effects change under simultaneous intervention?**

RQ2 characterizes how a frozen candidate set behaves when its members are intervened on together. It separates three questions:

1. how much behavior is reachable by at least one singleton candidate;
2. how much behavior is changed by the simultaneous full-set intervention;
3. which example-level interaction classes produce any difference between those quantities.

Candidate-set specificity against structurally matched non-candidate sets is reported separately.

## Population

RQ2 uses the 28 `paper-primary` overtopping settings on the held-out `test` split. Candidate identity and ranking are frozen before held-out causal evaluation.

The setting is the manuscript replication unit. Individual held-out examples describe within-setting interaction structure and are not treated as independent cross-setting replicates.

## Singleton-union reach

For frozen candidate set `J`, let `F_j(x)=1` when the singleton intervention on candidate `j` changes the binary behavioral endpoint `B(x)`.

Define the singleton-union event

```text
S_J(x) = OR_j F_j(x)
```

and singleton-union reach

```text
U(J) = P(S_J = 1).
```

Directional versions condition on the natural source state:

```text
U_J_i2c = P(S_J = 1 | B = 0)   # 0→1
U_J_c2i = P(S_J = 1 | B = 1)   # 1→0
```

`U(J)` is a union of singleton flip masks. It is not a sum of singleton effects.

## Simultaneous full-set effect

Let `F_J(x)=1` when intervening on all members of `J` simultaneously changes `B(x)`.

```text
E(J) = P(F_J = 1)
```

The composition gap is

```text
Delta_comp = E(J) - U(J).
```

The sign is descriptive:

- `Delta_comp < 0`: the full-set intervention changes fewer examples than are singleton-reachable;
- `Delta_comp = 0`: full-set effect equals singleton-union reach;
- `Delta_comp > 0`: the full-set intervention changes examples not accounted for by singleton reach.

The sign alone does not identify saturation, masking, cancellation, preemption, or cooperation.

## Example-level singleton-versus-joint decomposition

Stage 8 partitions every complete-case held-out example using the pair `(S_J, F_J)`:

| Singleton union `S_J` | Full set `F_J` | Class | Meaning |
|---:|---:|---|---|
| 1 | 1 | `preserved` | at least one singleton flips the example and the full set also flips it |
| 1 | 0 | `suppressed` | singleton reach is lost under the full-set intervention |
| 0 | 1 | `coalition_only` | the full set flips an example that no singleton flips |
| 0 | 0 | `unaffected` | neither singleton union nor full set flips the example |

On the same evaluation population,

```text
U(J) = P(preserved) + P(suppressed)
E(J) = P(preserved) + P(coalition_only)
```

therefore

```text
E(J) - U(J) = P(coalition_only) - P(suppressed).
```

The implementation writes and checks this identity for each setting and direction.

### Primary decomposition quantities

The principal setting-level quantities are:

```text
preservation_rate_given_singleton_reachable
suppression_rate_given_singleton_reachable
coalition_only_rate_all
coalition_only_rate_given_singleton_unreachable
multi_singleton_reachable_rate
Delta_comp_complete_case
```

The decomposition uses the complete-case singleton evaluation population so that `S_J` is defined over the complete frozen candidate set. Directional summaries use the same complete-case population restricted to `B=0` or `B=1`.

## Relation to saturation and bottleneck interpretations

A simple monotone saturation pattern has the following empirical signature:

- singleton-reachable examples are usually preserved by the full-set intervention;
- suppression is low;
- coalition-only effects are low in strong singleton-sufficient settings;
- additional candidates add little new behavioral reach once a sufficient singleton is present.

This pattern is **consistent with** saturating high-leverage or bottleneck-like causal organization. It does not by itself establish that a coordinate is a necessary natural bottleneck in the intact model.

Substantial `suppressed` mass indicates that some singleton-reachable effects disappear under the simultaneous intervention. Such a pattern requires an interaction explanation beyond simple monotone saturation. Substantial `coalition_only` mass identifies cooperative or coalition-dependent effects.

## Candidate-set specificity

Stage 8 also compares the candidate set with structurally matched non-candidate sets `K_b`:

```text
Delta_E_matched = E(J) - median_b E(K_b).
```

The matched sets preserve the configured structural strata. Each set effect is measured by a genuine simultaneous intervention.

This analysis asks whether the discovered candidate set is unusually consequential relative to structurally comparable sets. It is distinct from the singleton-versus-joint decomposition.

## Conditional marginal contribution

When enabled, Stage 8 evaluates a matched background set `S_b` and computes

```text
M_b(J)   = E(S_b ∪ J)   - E(S_b)
M_b(K_b) = E(S_b ∪ K_b) - E(S_b)
D_b      = M_b(J) - M_b(K_b).
```

CMC measures whether the candidate set retains unusual marginal influence in a perturbed context. It is not the primary test of saturation.

## Implementation

Model-backed interaction validation:

```text
pipeline/stage08_validate_interactions.py
```

Stage 8 writes, within each run's `interaction_validation/` directory:

```text
interaction_validation_summary.json
interaction_validation_summary.csv
composition_example_decomposition.csv
composition_decomposition_summary.csv
composition_decomposition_summary.json
matched_control_strata.csv
matched_random_set_membership.csv
```

The decomposition files are produced from existing Stage-7 singleton masks and the Stage-8 full-set output. They do not require an additional model intervention beyond the full-set evaluation already used for `E(J)`.

Cross-setting aggregation:

```text
studies/overtopping/analysis/stage09_composition_decomposition_report.py
```

Aggregate outputs:

```text
results/analysis/rq2_composition/interaction_decomposition/
├── composition_decomposition_all_scopes.csv
├── composition_decomposition_by_condition.csv
├── composition_decomposition_population_audit.csv
├── composition_decomposition_report_status.json
└── composition_decomposition.pdf
```

## Manuscript outputs

```text
results/paper/figures/03_rq2_composition/
├── fig3a_composition_gap_all_settings.pdf
├── fig3b_superadditive_boundary_cases.pdf
├── fig3c_matched_set_specificity.pdf
└── fig3d_singleton_joint_decomposition.pdf
```

Figure 3d is generated when the Stage-8 decomposition is available for the primary population. It displays the two event classes that determine the composition gap: singleton-reachable suppression and coalition-only full-set effects.

## Relation to RQ3

RQ2 characterizes how high-leverage singleton effects compose. RQ3 independently asks whether a fixed candidate's graded intervention produces a support-specific threshold-like behavioral transition. The two RQs are complementary: RQ2 describes interaction structure across candidates, while RQ3 tests the continuous-to-discrete causal geometry of individual candidates.
