# Reporting code

This package aggregates persistent experiment artifacts into configured-study tables, metric-availability audits, statistical summaries, machine-readable sidecars, and rendered outputs.

Main entry point from `code/`:

```bash
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile configured
```

The standard reporting path is model-free. It reads scientific artifacts from `data/`, derives the configured overtopping population from the registry or an explicit catalogue, and writes outputs under `results/`. Metric-specific applicability and availability are recorded after the configured population is established.

References:

- [Reporting overview](../docs/reporting/README.md)
- [Analysis pipeline](../docs/reporting/analysis-pipeline.md)
- [Generated outputs](../docs/reporting/figures.md)
- [Metric semantics](../docs/reporting/interpretation.md)
- [Operations](../docs/operations/README.md)
