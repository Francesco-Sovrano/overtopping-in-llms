# Reporting utilities

This package converts persistent experiment artifacts into aggregate analyses, tables, figures, and reproducibility checks.

From `code/`, generate the configured figure set with:

```bash
python -m reporting.generate_manuscript_figures --results-root ../results
```

Generate the configured LaTeX tables from canonical analysis CSVs with:

```bash
python -m reporting.generate_manuscript_tables \
  --results-root ../results \
  --require-all
```

`generate_manuscript_figures` invokes the table generator before assembling its combined output directory, so both output classes use the same analysis products.

For the full aggregation sequence, metric-availability audits, and source/output paths, see [`../docs/reporting/README.md`](../docs/reporting/README.md).
