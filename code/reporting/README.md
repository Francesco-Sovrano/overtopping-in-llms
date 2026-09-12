# Reporting code

This package aggregates persistent experiment outputs into configured-study tables, metric-availability audits, statistical summaries, machine-readable sidecars, and manuscript-facing products.

Main entry point from `code/`:

```bash
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile study-56
```

The standard reporting path is model-free. It reads model-backed scientific artifacts from `data/`, does not use filesystem presence to define the configured population, and writes derived outputs under `results/`. The driver rejects a results root located inside `data/` or the repository `cache/` tree.

Overtopping reporting begins from the complete 56-setting registry. Each analysis then applies its own applicability and artifact-availability rules and records its denominator. Mean-donor and mean-family RQ2 regimes are analyzed separately; `mean-positional` belongs to the mean-family reporting regime.

References:

- [Reporting overview](../docs/reporting/README.md)
- [Experiment design](../docs/experiments/overtopping.md)
- [Analysis pipeline](../docs/reporting/analysis-pipeline.md)
- [Figure map](../docs/reporting/figures.md)
- [Interpretation](../docs/reporting/interpretation.md)
- [Operations](../docs/operations/README.md)
