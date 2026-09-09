# Overtopping analysis

This package contains the model-free and reporting analyses used by the overtopping study. Model-backed intervention execution is performed by the shared pipeline and the graded-intervention module; this directory consumes their persistent outputs and produces run-level or cross-setting scientific summaries.

## Main analysis modules

### RQ1 and RQ2

```text
stage01_visualize_experiment_results.py      catalogue summaries
stage02_overtopping_latex_tables.py          primary matrix construction
stage03_audit_required_metrics.py            completeness audit
stage04_analyze_primary_metrics.py           setting-level statistics
stage05_generate_manuscript_outputs.py       manuscript tables and machine sidecars
stage06_competence_vs_overtopping_figures.py RQ1 figures and checkpoint trajectories
stage06_manuscript_story_figures.py          RQ2 manuscript figures
stage09_composition_decomposition_report.py  RQ2 singleton-union/full-set decomposition
```

The RQ2 decomposition reads `composition_decomposition_summary.csv` written by Stage 8. It verifies

```text
E(J) - U(J) = P(coalition_only) - P(suppressed)
```

on the same complete-case evaluation population and aggregates the setting-level results. It is descriptive of interaction structure and does not label a unique mechanism.

### RQ3

```text
threshold_event_diagnostics.py               per-run threshold-event diagnostics
stage07_overtopping_spiking_report.py         aggregate candidate/control support statistics
stage08_threshold_shape_validation.py         nested held-out threshold-shape validation
graded_agonist_intervention.py                per-run graded causal intervention
stage08_graded_agonist_report.py              cross-run support-specific graded aggregation
stage09_preemption_report.py                  secondary dominant-secondary preemption aggregation
```

The primary RQ3 claim uses graded support-specific crossings. Preemption is retained as a secondary subtype analysis and is not required for the primary thresholded causal-integration result.

## Recovery utilities

```text
rebuild_directional_stats.py
rebuild_spiking_diagnostics.py
```

These utilities rebuild derived statistics or diagnostics from compatible persistent inputs. They do not redefine the declared scientific populations.

## Result roots

```text
results/analysis/primary_matrix/
results/analysis/rq2_composition/interaction_decomposition/
results/analysis/rq3_threshold_event/spiking_diagnostics/
results/paper/figures/02_rq1_prevalence/
results/paper/figures/03_rq2_composition/
results/paper/figures/04_rq3_spiking_cut/
```

## References

- [Research questions](../../../docs/research-questions/README.md)
- [RQ2 composition](../../../docs/research-questions/rq2-composition.md)
- [RQ3 thresholded integration](../../../docs/research-questions/rq3-threshold-event.md)
- [Analysis and reporting pipeline](../../../docs/reporting/analysis-pipeline.md)
- [Figure map](../../../docs/reporting/figures.md)
- [Interpretation](../../../docs/reporting/interpretation.md)
- [Operations](../../../docs/operations/README.md)
