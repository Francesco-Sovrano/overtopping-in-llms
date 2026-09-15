# Poisoning study

Controlled poisoning experiment, checkpoint analysis, causal-role tracking, and replicate-run aggregation.

Documentation:

- [Poisoning protocol](../../docs/experiments/poisoning/README.md)
- [Poisoning configuration](../../docs/experiments/poisoning/configuration.md)
- [Poisoning outputs](../../docs/experiments/poisoning/outputs.md)
- [RQ4 — learning](../../docs/research-questions/rq4-learning.md)
- [Operations and regeneration](../../docs/operations/README.md)

## Cross-seed plotting

`stage08_aggregate_cross_seed` keeps the training seed as the replicate unit and
writes both robust (median, Q1, Q3) and parametric (mean, SD, Student-t CI)
summaries. `stage08_plot_cross_seed` defaults to `--summary_style median_iqr`.
Two alternatives are available:

```text
--summary_style mean_t_ci
--summary_style seed_traces
```

`seed_traces` overlays each training seed as a thin line and keeps the
median/Q1-Q3 summary on top, which is useful for small replicate counts.

The same seed-level rule now applies to the RQ4 clean-reference defense. Stage
07 still selects channels independently inside each training run, but Stage 08
aggregates each run's checkpoint-level defense summary across seeds. The main
figure therefore reports median and Q1-Q3 across training seeds; selected
channels remain nested within a seed and are not treated as independent
replicates.

If one task/model contains more than one scientific configuration, plotting now
uses `--family_policy split` by default: configurations are rendered separately
instead of being pooled or aborting the report. The exact family identities and
seed memberships are written to `scientific_family_manifest.csv` in the Stage-08
aggregate table directory. Use `--family_policy error` to restore strict failure.
