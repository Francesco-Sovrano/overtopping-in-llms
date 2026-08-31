# Manuscript figure map

Paper-facing figures are generated under `results/paper/`; machine-readable source tables are written under `results/analysis/figure_data/` and the corresponding analysis directories.

## Figure 1 — concept

Conceptual framing of causal overtopping and the distinction between distributed candidate coverage and stronger concentrated handles.

## Figure 2 / RQ1 — prevalence and competence

Owned by the Stage-06 competence/overtopping figure engine. It uses the declared manuscript population, includes genuine zero-candidate settings, and reports directional metrics separately where applicable.

Missing directional values for a selected manuscript row cause an audit failure rather than silent row removal.

## Figure 3 / RQ2 — composition and boundary cases

Reports causal composition and superadditivity/boundary behavior. `fig3b_superadditive_boundary_cases.pdf` renders exact zero-valued boundary cases explicitly so zero is distinguishable from missing data.

## Figure 4 / RQ3 — threshold/spiking cut

Generated from threshold-event diagnostics resolved to the exact primary-table population. Positive/negative baseline subsets and candidate/control populations are audited before plotting.

Stage-7 candidate flips are reused from materialized scores; same-layer non-candidate controls are the model-backed additions made by Stage 7b.

## Figure 5 / RQ4 — learning utility

The Pythia checkpoint story uses:

```text
fig5a_pythia_checkpoint_trajectory.pdf
fig5s1_pythia_checkpoint_U_0to1.pdf
fig5s2_pythia_checkpoint_U_1to0.pdf
```

No second alias of the pooled checkpoint trajectory is published.

## Poisoning figures

Poisoning manuscript/supporting figures are published from the canonical per-run and cross-seed story outputs. The compact per-run causal story contains:

```text
01_clean_vs_poisoned_overtopping_development.pdf
02_channel_role_reassignment_and_defense_leverage.pdf
04_clean_vs_poisoned_checkpoint_overtopping.pdf
```

Additional Stage-07 interpretation figures are published only when they answer distinct mechanism, persistence, concentration, update-geometry, detection, or attack-link questions. Trigger-lift causal panels are omitted when trigger-lift CHA was not computed.

## Output contract

A paper figure must have a machine-readable source table or a reproducible analysis source, must use the declared scientific population, and must not infer missing causal values from candidate non-discovery. Empty panels are not used to represent uncomputed endpoints.
