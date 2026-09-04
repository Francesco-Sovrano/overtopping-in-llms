# Reporting and generated results

The reporting layer converts completed experiment artifacts under `data/` into validated analysis tables, statistical summaries, audits, and manuscript-facing figures under `results/`.

Run the standard reporting pipeline from the repository root:

```bash
./generate_results.sh
```

Use the following documents according to purpose:

- [Analysis pipeline](analysis-pipeline.md) — reporting stages, declared analysis populations, schemas, and result tree.
- [Manuscript figure map](figures.md) — figure filenames, scientific populations, and source analyses.
- [Interpretation and limitations](interpretation.md) — what each metric supports and what it does not establish.

Research-question methods are documented separately under [Research questions](../research-questions/). Regeneration and recovery procedures are in [Operations](../operations/).
