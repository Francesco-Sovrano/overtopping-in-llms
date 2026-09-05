# Reporting and generated results

The reporting layer converts persistent experiment artifacts under `data/` into validated analysis tables, population audits, statistical summaries, and manuscript-facing outputs under `results/`.

Standard repository command:

```bash
./generate_results.sh
```

Direct invocation from `code/`:

```bash
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile iclr-28
```

The reporting driver does not rerun missing model-backed experiments. It aggregates compatible persistent outputs, records availability, and validates the configured manuscript contract.

Use:

- [Analysis pipeline](analysis-pipeline.md) for modules, populations, inputs, and result-tree structure;
- [Figure map](figures.md) for manuscript filenames and source analyses;
- [Interpretation](interpretation.md) for metric scope and statistical units;
- [Operations](../operations/README.md) for targeted regeneration and troubleshooting.

Research-question methods are documented under [Research questions](../research-questions/README.md).
