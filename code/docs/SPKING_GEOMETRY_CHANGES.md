# Graded spiking-geometry analysis

The graded intervention analysis measures whether candidate-channel effects exhibit concentrated nonlinear transitions rather than approximately affine interpolation between zero and full intervention.

For each eligible candidate/example pair, the pipeline records behavioral outcomes and continuous endpoint-margin quantities over a graded intervention grid. The analysis compares observed intermediate margins with the endpoint-affine null, summarizes departure from that null, measures transition concentration, and compares continuous-margin crossings with persistent task-level crossings.

Per-run diagnostics are produced by:

```text
studies/overtopping/analysis/stage09_graded_margin_mechanism_test.py
```

Cross-setting aggregation is produced by:

```text
studies/overtopping/analysis/stage08_graded_agonist_report.py
```

When the required endpoint-margin artifacts are available, the aggregate reporter writes condition-level statistics and the corresponding manuscript/appendix figures. Eligibility is determined by the configured study manifest and the graded-analysis artifact contract; unavailable diagnostics remain explicit in the reporting audit.

See [RQ3](research-questions/rq3-threshold-event.md), [Figure map](reporting/figures.md), and [Analysis pipeline](reporting/analysis-pipeline.md).
