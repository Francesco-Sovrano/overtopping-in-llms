# RQ2 — composition and candidate-set specificity

## Question

RQ2 asks how the effects of frozen singleton candidates combine when the full candidate set is intervened on simultaneously, and whether the localized candidate set is more consequential than structurally matched non-candidate sets.

## Population

RQ2 uses the 28 `paper-primary` overtopping settings. The primary table supplies the frozen candidate set, singleton flip masks, simultaneous-set effect, and matched-set validation quantities for each setting.

Poisoning runs are not part of the RQ2 population.

## Singleton-union reach

For candidate set `J`, `U(J)` is the fraction of held-out examples flipped by at least one singleton intervention:

```text
U(J) = P(union_j F_j = 1)
```

Directional counterparts `U_J_i2c` and `U_J_c2i` condition on the baseline source state.

`U(J)` is a union over singleton flip masks. It is not the strongest singleton effect and it is not the sum of singleton effects.

## Simultaneous-set effect

`E(J)` is the held-out behavioral effect of intervening on the complete candidate set simultaneously.

The composition gap is:

```text
Delta_comp = E(J) - U(J)
```

Interpretation of the sign:

- `Delta_comp < 0`: the simultaneous intervention flips fewer held-out examples than are reachable by at least one singleton;
- `Delta_comp = 0`: the simultaneous effect equals singleton-union reach;
- `Delta_comp > 0`: the simultaneous intervention reaches examples not accounted for by the singleton-union mask, consistent with coalition-dependent effects.

The sign alone does not identify the mechanism producing the gap. Overlap, saturation, masking, cancellation, and coalition effects are not separated by this statistic.

## Structural context

The primary table provides quantities used to characterize the composition regime, including:

```text
U(J)
U_J_i2c, U_J_c2i
s_1_i2c, s_1_c2i
N05 / N10 directional counts
TOC1
N_eff
redundancy / overlap quantities
competence
phase
```

These quantities allow composition to be compared with singleton strength, reach, concentration, redundancy, task, and intervention phase.

## Matched-set specificity

RQ2 compares the candidate-set joint effect with structurally matched non-candidate sets `K_b`:

```text
Delta_E_matched = E(J) - median_b E(K_b)
```

Matched sets are drawn according to the interaction-validation configuration. The Monte-Carlo table reports the observed candidate effect, null distribution summaries, candidate-minus-null difference, and Monte-Carlo p-value.

This analysis asks whether the localized set is unusually consequential relative to comparably structured sets. It is distinct from the composition-gap analysis.

## Statistical unit

The experimental setting is the primary RQ2 unit. Individual matched-null draws are Monte-Carlo reference draws for a setting rather than independent manuscript settings.

When interpreting setting-level Monte-Carlo p-values, account for the configured number of null draws and the resulting p-value resolution.

## Implementation

Principal modules:

```text
pipeline/stage08_validate_interactions.py
studies/overtopping/analysis/stage02_overtopping_latex_tables.py
studies/overtopping/analysis/stage06_manuscript_story_figures.py
reporting/generate_final_results.py
```

Stage 08 materializes simultaneous-set, matched-null, and conditional interaction products. Reporting reads those persisted products rather than redefining the intervention population from figure data.

## Manuscript outputs

```text
results/paper/figures/03_rq2_composition/
├── fig3a_composition_gap_all_settings.pdf
├── fig3b_superadditive_boundary_cases.pdf
└── fig3c_matched_set_specificity.pdf
```

Figure 3a reports `Delta_comp` across the primary settings.

Figure 3b expands settings with `E(J) > U(J)`. Its labels use the exact definitions:

```text
U(J): singleton-union reach
E(J): simultaneous full-set effect
Delta: E(J) - U(J)
```

Figure 3c reports candidate-set specificity relative to structurally matched non-candidate sets.

Machine-readable sidecars are written under:

```text
results/analysis/figure_data/03_rq2_composition/
```

## Relation to RQ3

RQ2 establishes set-level composition. It does not by itself establish a causal threshold crossing. RQ3 separately tests the graded causal transition of agonists on examples they are already known to flip.

See [RQ3 — threshold-event analysis](rq3-threshold-event.md).
