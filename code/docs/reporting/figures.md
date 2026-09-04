# Manuscript figure map

Paper-facing figures are written under `results/paper/figures/`. Machine-readable source tables are written under `results/analysis/figure_data/` or the corresponding RQ-specific analysis directory.

## Figure 1 — concept

Conceptual overview of singleton causal effects, singleton-union reach, simultaneous-set composition, threshold visibility, and learning-time causal organization.

## Figure 2 — RQ1 prevalence and competence

Scientific method: [RQ1 — prevalence and competence](../research-questions/rq1-prevalence.md).

Owner: `studies/overtopping/analysis/stage06_competence_vs_overtopping_figures.py`.

Population: exact 39-setting primary+auxiliary non-poisoning catalogue.

```text
17 input+output settings
22 decode-only settings
```

Main files:

```text
fig2a_competence_vs_U_0to1.pdf
fig2b_competence_vs_U_1to0.pdf
fig2c_competence_vs_D05_0to1.pdf
fig2d_competence_vs_D05_1to0.pdf
```

`D05` denotes the implementation's `N_0.05 / d_model` normalization. It is a width-normalized discovered-candidate count, not a coordinate-population prevalence estimate.

Expected experiment paths are resolved from the explicit catalogue. Completed zero-candidate settings remain zero observations; missing required settings fail the population audit.

## Figure 3 — RQ2 composition

Scientific method: [RQ2 — composition](../research-questions/rq2-composition.md).

Owner: `studies/overtopping/analysis/stage06_manuscript_story_figures.py`.

`fig3a_composition_gap_all_settings.pdf` reports:

```text
Delta_comp = E(J) - U(J)
```

for the 28-setting primary population.

`fig3b_superadditive_boundary_cases.pdf` displays settings with `E(J)>U(J)` and uses these definitions:

- `U(J)`: fraction of held-out rows flipped by at least one singleton member of `J`;
- `E(J)`: held-out effect of intervening on the complete set `J` simultaneously;
- `Delta`: `E(J)-U(J)`.

A valid `U(J)=0` is shown as a zero value.

`fig3c_matched_set_specificity.pdf` compares the candidate-set joint effect with structurally matched non-candidate set effects.

## Figure 4 — RQ3 graded spiking-cut analysis

Scientific method: [RQ3 — spiking-like causal transition](../research-questions/rq3-threshold-event.md).

Population: frozen agonists from the exact primary overtopping manifest plus eligible auxiliary settings. Candidate membership is restricted to frozen discovery direction.

### Figure 4a — graded agonist dose response

`fig4a_graded_agonist_dose_response.pdf`

For each discovery direction, plots the behavior-flipped fraction as the same agonist intervention is increased from dose 0 to dose 1 on examples already known to be flipped by that agonist at full intervention. When enabled, same-agonist non-flip support is shown as a within-agonist reference.

### Figure 4b — single persistent crossing

`fig4b_graded_agonist_single_crossing.pdf`

Shows run/baseline/direction condition summaries of the fraction of known-flip trajectories that cross the behavioral boundary once and remain across it for all larger tested doses.

Legacy scalar/threshold figures are analysis-only and are not manuscript-facing RQ3 evidence.

### Figure 4c — preemption

`fig4c_preemption.pdf`

The existing exploratory pair-intervention report is retained separately from the primary graded agonist experiment.

## Figure 5 — RQ4 checkpoint trajectories

Scientific method: [RQ4 — learning and causal-role dynamics](../research-questions/rq4-learning.md).

Pythia checkpoint files include:

```text
fig5a_pythia_checkpoint_trajectory.pdf
fig5s1_pythia_checkpoint_U_0to1.pdf
fig5s2_pythia_checkpoint_U_1to0.pdf
```

The pooled trajectory is a distribution-level summary. The directional companion figures are the appropriate source for direction-specific developmental interpretation.

## Poisoning figures

Study design: [Poisoning protocol](../experiments/poisoning/). RQ4 interpretation: [RQ4 — learning and causal-role dynamics](../research-questions/rq4-learning.md).

Per-run poisoning story files:

```text
01_clean_vs_poisoned_overtopping_development.pdf
02_channel_role_reassignment.pdf
03_prospective_defense_leverage.pdf
04_clean_vs_poisoned_checkpoint_overtopping.pdf
```

Figure 01 reports aggregate checkpoint quantities. Figure 02 distinguishes checkpoint-local and fixed-union channel measurements. Figure 03 is a prospective target-screening analysis. Figure 04 requires complete fixed-union materialization across the displayed matched checkpoints and conditions.

## Output contract

Every manuscript figure should be traceable to a machine-readable source table or deterministic analysis output, use its declared scientific population, and distinguish missing causal measurements from measured zeros.
