# Controlled poisoning study

This package implements the matched clean/poisoned Grammar training trajectories used for RQ4, checkpoint behavioral evaluation, causal-role tracking, defense-target screening, optional poisoning-example analyses, and cross-seed aggregation.

Documentation:

- [Controlled Grammar poisoning protocol](../../docs/experiments/poisoning/README.md)
- [Poisoning configuration](../../docs/experiments/poisoning/configuration.md)
- [Poisoning outputs](../../docs/experiments/poisoning/outputs.md)
- [RQ4 — Learning-time causal organization](../../docs/research-questions/rq4-learning.md)
- [Operations and regeneration](../../docs/operations/README.md)

## Cross-seed aggregation

`stage08_aggregate_cross_seed` keeps training seed as the replicate unit and writes median/Q1/Q3 and mean/SD/Student-t summaries. `stage08_plot_cross_seed` accepts:

```text
--summary_style median_iqr
--summary_style mean_t_ci
--summary_style seed_traces
```

The clean-reference defense aggregation follows the same replicate rule. Channel selections are performed independently within each training run, then checkpoint-level run summaries are aggregated across seeds. Channels nested within a seed are not treated as independent replicates.

When one task/model has more than one scientific configuration, `--family_policy split` keeps configurations separate. `scientific_family_manifest.csv` records family identities and seed membership. `--family_policy error` instead rejects multiple families.
