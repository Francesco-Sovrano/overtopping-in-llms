# Overtopping analysis

This package contains primary-matrix construction, statistical analysis, RQ1/RQ2/RQ4 figure generation, RQ3 threshold-event reporting, and the graded agonist experiment/report.

RQ3 implementation is split across:

```text
threshold_event_diagnostics.py          per-run model-backed threshold diagnostics
stage07_overtopping_spiking_report.py   aggregate candidate/control support statistics
stage08_threshold_shape_validation.py   nested held-out threshold-shape validation
graded_agonist_intervention.py          per-run graded causal experiment
stage08_graded_agonist_report.py        cross-run graded aggregation
```

Operational recovery utilities:

```text
rebuild_directional_stats.py            rebuild derived directional statistics from materialized Stage-7 events
rebuild_spiking_diagnostics.py          backfill model-backed RQ3 diagnostics using the configured RunSpec population
```

References:

- [Analysis and reporting pipeline](../../../docs/reporting/analysis-pipeline.md)
- [RQ3](../../../docs/research-questions/rq3-threshold-event.md)
- [Figure map](../../../docs/reporting/figures.md)
- [Interpretation](../../../docs/reporting/interpretation.md)
- [Operations](../../../docs/operations/README.md)
