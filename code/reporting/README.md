# Reporting code

This package aggregates persistent experiment outputs into analysis tables, population audits, statistical summaries, and manuscript-facing products.

Main entry point:

```bash
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile iclr-28
```

The reporting layer reads model-backed outputs; it does not substitute for missing causal experiments.

References:

- [Reporting overview](../docs/reporting/README.md)
- [Analysis pipeline](../docs/reporting/analysis-pipeline.md)
- [Figure map](../docs/reporting/figures.md)
- [Interpretation](../docs/reporting/interpretation.md)
- [Operations](../docs/operations/README.md)
