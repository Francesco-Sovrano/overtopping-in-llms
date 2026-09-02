# Manuscript figure map

Paper-facing figures are generated under `results/paper/`; machine-readable source tables are written under `results/analysis/figure_data/` and the corresponding analysis directories.

## Figure 1 — concept

Conceptual framing of causal overtopping and the distinction between distributed candidate coverage and stronger concentrated handles.

## Figure 2 / RQ1 — prevalence and competence

Owned by the Stage-06 competence/overtopping figure engine. It uses the declared manuscript population, includes genuine zero-candidate settings, and reports directional metrics separately where applicable.

Missing directional values for a selected manuscript row cause an audit failure rather than silent row removal.

## Figure 3 / RQ2 — composition and boundary cases

Reports causal composition and superadditivity/boundary behavior. In `fig3b_superadditive_boundary_cases.pdf`, light bars show the singleton-union baseline `U(J)`, dark bars show the observed joint-set effect `E(J)`, each row is value-labelled with `U`, `E`, and `Δ = E(J)-U(J)`, and exact zero-valued `U(J)` cases are rendered explicitly so zero is distinguishable from missing data.

## Figure 4 / RQ3 — threshold/spiking cut

Generated from an exact manifest containing the primary overtopping population plus configured supplementary overtopping experiments when available. Poisoning experiments are excluded by construction. Positive/negative baseline subsets and candidate/control populations are audited before plotting.

Stage-7 candidate flips are reused from materialized scores; same-layer non-candidate controls are the model-backed additions made by Stage 7b. Figure 4a separates causal strength, threshold testability, and causal-strength-matched nested thresholdability. Figure 4b uses training-fold feature/orientation selection and a same-condition causal-strength-matched illustrative pair rather than the highest-MCC units. Figure 4c performs inference at run/baseline/direction condition level; raw channel pairs remain descriptive.

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
02_channel_role_reassignment.pdf
03_prospective_defense_leverage.pdf
04_clean_vs_poisoned_checkpoint_overtopping.pdf
```

Figure 01 is aggregate-only. Figure 02 is a descriptive checkpoint-local channel-role view and explicitly marks local-only values when fixed-union materialization is unavailable. Figure 03 is the prospective defense-target screen: baseline-locked targets are selected at checkpoint 0, while rolling targets are selected from the fixed control-correctness candidate union using only the previous checkpoint. It reports a defense-leverage proxy, Δdef = attack suppression − benign-correctness damage. Positive Δdef means a singleton intervention is attack-selective enough to be a plausible defense target; it is not itself a defense-efficacy result. Figure 04 is published only when the fixed candidate-union matrix is complete across every matched checkpoint/condition.

Additional Stage-07 interpretation figures are published only when they answer distinct mechanism, persistence, concentration, update-geometry, detection, or attack-link questions. Trigger-lift causal panels are omitted when trigger-lift CHA was not computed.

## Output contract

A paper figure must have a machine-readable source table or a reproducible analysis source, must use the declared scientific population, and must not infer missing causal values from candidate non-discovery. Empty panels are not used to represent uncomputed endpoints.
