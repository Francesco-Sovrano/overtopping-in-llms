# Overtopping analysis

This package owns overtopping-specific aggregate analysis. The numbered `stageNN_` modules form the canonical paper-facing sequence; unnumbered modules are optional diagnostics or utilities.

Cross-study orchestration is intentionally outside this package in `reporting.generate_final_results`. Shared model-backed simultaneous/conditional validation is Pipeline Stage 08, `pipeline.stage08_validate_interactions`.

Run from `code/`:

```bash
python3 -m studies.overtopping.analysis.stage04_analyze_primary_metrics --help
python3 -m pipeline.stage08_validate_interactions --help
python3 -m reporting.generate_final_results --help
```

See [`../../../docs/overtopping-analysis.md`](../../../docs/overtopping-analysis.md) for the full analysis workflow and [`../../../docs/pipeline.md`](../../../docs/pipeline.md) for Pipeline Stage 08.
