# RQ2 — composition

## Question

RQ2 asks how the effects reachable by individual frozen candidates change when the full candidate set is intervened on simultaneously.

It distinguishes:

1. singleton-union reach;
2. genuine simultaneous-set effect;
3. example-level interaction classes explaining the gap between them;
4. specificity relative to structurally matched non-candidate sets.

## Population

RQ2 starts from the complete 48-setting overtopping manifest and applies two rules.

First, replacement regimes are analyzed separately:

```text
mean-donor regime: 27 configured settings
mean regime:       12 configured settings after grouping mean-positional with mean
```

The 12 mean-family settings consist of eight `mean` settings and four large-model `mean-positional` settings. Mean-donor and mean-family results are not pooled.

Second, joint composition is applicable only when discovery yields a nonempty candidate set. A setting is evaluable for the aggregate composition-gap analysis when both singleton-union reach `U(J)` and the genuine simultaneous-set effect `E(J)` are available. The regime reporter starts from the complete configured study table, computes the denominator from these applicability and availability rules, and writes every configured row with an explicit `rq2_status`. A missing joint-effect artifact therefore remains visible in the population audit rather than changing the study manifest.

Candidate identity and ranking are frozen before held-out evaluation. The setting is the cross-setting statistical unit; held-out examples describe interaction structure within a setting.

## Singleton-union reach

For frozen candidate set `J`, let `F_j(x)=1` when singleton intervention on candidate `j` changes binary endpoint `B(x)`.

```text
S_J(x) = OR_j F_j(x)
U(J)   = P(S_J = 1)
```

Directional versions condition on the natural source state:

```text
U_J_i2c = P(S_J = 1 | B = 0)   # 0→1
U_J_c2i = P(S_J = 1 | B = 1)   # 1→0
```

`U(J)` is a union of singleton flip masks, not a sum of singleton effects.

## Simultaneous full-set effect

Let `F_J(x)=1` when intervening on all members of `J` simultaneously changes `B(x)`.

```text
E(J) = P(F_J = 1)
```

The composition gap is:

```text
Delta_comp = E(J) - U(J)
```

Its sign is descriptive:

- negative: some singleton-reachable behavior is not preserved by the full-set intervention;
- zero: simultaneous effect equals singleton-union reach;
- positive: the full set reaches behavior not reached by any singleton.

The sign alone does not identify a specific mechanism.

## Example-level decomposition

On the complete-case held-out population, Stage 8 classifies each example by `(S_J, F_J)`:

| `S_J` | `F_J` | Class | Interpretation |
|---:|---:|---|---|
| 1 | 1 | `preserved` | singleton-reachable and still changed by full set |
| 1 | 0 | `suppressed` | singleton-reachable effect lost under full set |
| 0 | 1 | `coalition_only` | changed only by simultaneous intervention |
| 0 | 0 | `unaffected` | changed by neither |

Therefore:

```text
U(J) = P(preserved) + P(suppressed)
E(J) = P(preserved) + P(coalition_only)
E(J) - U(J) = P(coalition_only) - P(suppressed)
```

The implementation checks this identity per setting.

Principal setting-level fields include:

```text
preservation_rate_given_singleton_reachable
suppression_rate_given_singleton_reachable
coalition_only_rate_all
coalition_only_rate_given_singleton_unreachable
multi_singleton_reachable_rate
Delta_comp_complete_case
```

Directional summaries use the same complete-case population restricted by source state.

## Matched-set specificity

Stage 8 compares the candidate set with structurally matched non-candidate sets `K_b`:

```text
Delta_E_matched = E(J) - median_b E(K_b)
```

Each `E(K_b)` is measured by a genuine simultaneous intervention. The population audit records which configured settings have the required matched-set products.

## Conditional marginal contribution

When enabled, Stage 8 evaluates a matched background set `S_b`:

```text
M_b(J)   = E(S_b ∪ J)   - E(S_b)
M_b(K_b) = E(S_b ∪ K_b) - E(S_b)
D_b      = M_b(J) - M_b(K_b)
```

This measures whether the discovered candidate set retains unusual marginal influence in a perturbed context.

## Implementation

Model-backed interaction validation:

```text
pipeline/stage08_validate_interactions.py
```

Per-run outputs under `<stage7 stats dir>/interaction_validation/` include:

```text
interaction_validation_summary.json
interaction_validation_summary.csv
composition_example_decomposition.csv
composition_decomposition_summary.csv
composition_decomposition_summary.json
matched_control_strata.csv
matched_random_set_membership.csv
```

Cross-setting reporters:

```text
studies/overtopping/analysis/stage09_composition_decomposition_report.py
studies/overtopping/analysis/stage10_rq2_regime_report.py
```

Aggregate outputs are written under:

```text
results/analysis/rq2_composition/
```

The regime reporter reads `results/analysis/primary_matrix/primary_table.csv`, the complete configured-setting table produced by Stage 02. It normalizes `mean-positional` to the `mean` reporting regime while preserving the original intervention field. Its `rq2_settings_by_replacement_regime.csv` output retains all configured rows and records applicability, metric availability, evaluability, and status before summary statistics are computed.

## Interpretation

A predominantly negative `E(J)-U(J)` indicates that simultaneous intervention preserves less behavior than the union of singleton interventions. Example-level suppression and coalition-only rates identify which event classes generate that gap. These quantities characterize composition; they do not by themselves establish necessity or a unique circuit mechanism.

RQ3 asks a separate question: how the behavior of a fixed candidate changes as intervention strength varies continuously.
