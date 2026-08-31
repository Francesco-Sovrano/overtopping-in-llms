# Overtopping study package

This package owns the non-poisoning experiment catalogue, primary-matrix construction, RQ1–RQ4 analyses, threshold/spiking reporting, manuscript figures, and study audits. Shared numbered pipeline stages and intervention mechanics live in `pipeline/` and `core/`.

Repository commands:

```bash
./run_overtopping_experiments.sh --list
./run_overtopping_experiments.sh --dry-run
./run_overtopping_experiments.sh
./generate_results.sh
```

Canonical documentation:

- [Experiment catalogue](../../docs/overtopping-experiments.md)
- [Overtopping analysis](../../docs/overtopping-analysis.md)
- [Numbered causal pipeline](../../docs/pipeline.md)
- [Manuscript figure map](../../docs/manuscript-figure-map.md)
