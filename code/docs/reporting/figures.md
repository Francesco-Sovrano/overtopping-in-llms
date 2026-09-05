# Manuscript figure map

Paper-facing figures are written under:

```text
results/paper/figures/
```

Machine-readable source tables are written under `results/analysis/figure_data/` or the corresponding RQ-specific analysis directory.

## Figure 1 — concept

Conceptual overview of singleton causal effects, singleton-union reach, simultaneous-set composition, threshold-event structure, graded causal crossing, and learning-time causal organization.

## Figure 2 — RQ1 prevalence and competence

Method: [RQ1 — prevalence and competence](../research-questions/rq1-prevalence.md).

Owner:

```text
studies/overtopping/analysis/stage06_competence_vs_overtopping_figures.py
```

Population: exact 39-setting overtopping catalogue:

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

`D05` is the implementation's `N_0.05 / d_model` normalization. It is a width-normalized discovered-candidate count.

## Figure 3 — RQ2 composition

Method: [RQ2 — composition](../research-questions/rq2-composition.md).

Owner:

```text
studies/overtopping/analysis/stage06_manuscript_story_figures.py
```

### Figure 3a

```text
fig3a_composition_gap_all_settings.pdf
```

Reports:

```text
Delta_comp = E(J) - U(J)
```

for the 28-setting primary population.

### Figure 3b

```text
fig3b_superadditive_boundary_cases.pdf
```

Displays settings with `E(J)>U(J)` and the quantities required to interpret those boundary cases.

### Figure 3c

```text
fig3c_matched_set_specificity.pdf
```

Compares candidate-set simultaneous effects with structurally matched non-candidate set effects.

## Figure 4 — RQ3 threshold-event analysis

Method: [RQ3 — threshold events](../research-questions/rq3-threshold-event.md).

Paper directory:

```text
results/paper/figures/04_rq3_spiking_cut/
```

### Figure 4a — candidate/control phenotype endpoints

```text
fig4a_candidate_control_spiking_cut_summary.pdf
```

Source analysis:

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/threshold_shape_validation/threshold_shape_statistical_results.json
```

Reports, separately by discovery direction:

- singleton causal strength;
- threshold-testable fraction;
- nested held-out threshold `|MCC|`;
- nested TECS lower bound.

### Figure 4b — threshold-shape model comparison

```text
fig4b_threshold_shape_model_comparison_by_direction.pdf
```

Source:

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/threshold_shape_validation/threshold_shape_model_comparison.csv
```

Compares held-out constant, hard-threshold, logistic, and isotonic models using direction-specific condition-weighted summaries.

### Figure 4c — graded agonist dose response

```text
fig4c_graded_agonist_dose_response.pdf
```

Source:

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/graded_agonist/graded_agonist_dose_by_condition.csv
```

Shows behavioral flip fraction over intervention dose. The primary population is each agonist's held-out full-dose flip support. Optional same-agonist non-flip support appears as a within-agonist reference when enabled.

### Figure 4 S1 — threshold testability by condition

```text
fig4s1_threshold_testability_by_condition.pdf
```

Paired candidate/control threshold-testable fractions by run and discovery direction.

### Figure 4 S2 — strength-matched thresholdability sensitivity

```text
fig4s2_strength_matched_thresholdability.pdf
```

Candidate versus causal-strength-matched control held-out threshold `|MCC|`. This is a sensitivity analysis; the primary structural-control comparison does not condition on causal strength.

### Figure 4 S3 — nested TECS lower-bound ECDF

```text
fig4s3_nested_tecs_lower_bound_ecdf.pdf
```

Distributional view of nested TECS lower bounds for candidates and structural controls.

### Figure 4 S4 — oriented proxy-bin response

```text
fig4s4_threshold_tail_response_by_direction.pdf
```

Plots observed held-out flip probability over oriented endogenous-proxy bins. The bin index is descriptive and is not a fitted threshold.

### Figure 4 S5 — graded single crossing

```text
fig4s5_graded_agonist_single_crossing.pdf
```

Shows condition-level median agonist single-crossing rates for the known-flip support population.

### RQ3 diagnostic figure

The threshold-shape validation also writes:

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/
  threshold_shape_validation/figures/illustrative_threshold_response_curves.pdf
```

This file is an analysis diagnostic for selected same-condition/same-layer pairs and is not part of the manuscript figure directory.

### RQ3 availability

Threshold diagnostics are required for Figure 4a, Figure 4b, and S1–S4. Figure 4b requires aggregate activation/flip rows. S2 additionally requires eligible causal-strength matches. S4 requires aggregate binned curves.

The graded agonist report generates Figure 4c and S5 when the per-run graded manifests and data are available for the required primary population.

## Figure 5 — RQ4 learning

Method: [RQ4 — learning](../research-questions/rq4-learning.md).

Pythia checkpoint files include:

```text
fig5a_pythia_checkpoint_trajectory.pdf
fig5s1_pythia_checkpoint_U_0to1.pdf
fig5s2_pythia_checkpoint_U_1to0.pdf
```

The pooled trajectory is a distribution-level summary. Directional companion figures retain the source-state decomposition.

## Poisoning figures

Study design: [Poisoning protocol](../experiments/poisoning/README.md).

Per-run poisoning story files include:

```text
01_clean_vs_poisoned_overtopping_development.pdf
02_channel_role_reassignment.pdf
03_prospective_defense_leverage.pdf
04_clean_vs_poisoned_checkpoint_overtopping.pdf
```

The stage-numbered source data and availability conditions are documented in [Poisoning outputs](../experiments/poisoning/outputs.md).

## Figure contract

Every manuscript figure must be traceable to a deterministic analysis output or machine-readable source table, use its declared scientific population, and distinguish missing measurements from measured zeros.
