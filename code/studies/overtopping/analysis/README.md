# Overtopping analysis

This package converts persistent overtopping artifacts into setting-level metrics, completeness audits, cross-setting statistics, machine-readable sidecars, and rendered outputs.

## Population model

Aggregate analysis starts from the configured registry manifest. Metric applicability and artifact availability are evaluated after that population is established. Missing artifacts remain missing measurements. Completed zero-candidate settings remain measured observations for metrics that define a zero value.

RQ1 uses compatible directional singleton metrics and separates intervention phases. RQ2 separates replacement regimes and requires applicable set-level outputs. RQ3 uses compatible threshold, graded, margin, temporal, and optional preemption artifacts. RQ4 uses the configured Pythia checkpoint trajectory and poisoning outputs.

## Main modules

```text
stage01_visualize_experiment_results.py       completed-run summaries
stage02_overtopping_latex_tables.py           configured study table
stage03_audit_required_metrics.py             metric completeness audit
stage04_analyze_primary_metrics.py            cross-setting statistics
stage05_generate_manuscript_outputs.py        derived tables and machine sidecars
stage06_competence_vs_overtopping_figures.py  RQ1 and checkpoint rendering
stage06_manuscript_story_figures.py            RQ2 rendering
stage07_overtopping_spiking_report.py         RQ3 aggregate threshold statistics
stage08_threshold_shape_validation.py         nested held-out threshold-shape validation
stage08_graded_agonist_report.py              graded and margin aggregation
stage09_composition_decomposition_report.py   singleton/full-set event decomposition
stage09_preemption_report.py                  optional preemption aggregation
stage10_rq2_regime_report.py                  replacement-regime aggregation
stage10_rq3_spiking_story_figures.py          RQ3 population event/strength summaries
stage11_rq3_temporal_cutoff_story.py          temporal-cutoff aggregation
```


## Recovery utilities

```text
rebuild_directional_stats.py
rebuild_spiking_diagnostics.py
```

Recovery utilities operate on explicit configured settings and compatible persistent inputs. `rebuild_spiking_diagnostics.py --dry-run` validates provenance and prints planned work without writing scientific artifacts.

## Output roots

```text
results/analysis/primary_matrix/
results/analysis/reproducibility/
results/analysis/rq2_composition/
results/analysis/rq3_threshold_event/
results/analysis/rq4_learning/
results/paper/figures/
results/paper/tables/
```

References:

- [Experiment configuration](../../../docs/experiments/overtopping.md)
- [Research questions](../../../docs/research-questions/README.md)
- [Analysis pipeline](../../../docs/reporting/analysis-pipeline.md)
- [Metric semantics](../../../docs/reporting/interpretation.md)
- [Generated outputs](../../../docs/reporting/figures.md)
- [Operations](../../../docs/operations/README.md)
