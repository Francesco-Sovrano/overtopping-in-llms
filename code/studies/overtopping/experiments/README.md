# Overtopping experiment catalogue

This package owns the explicit overtopping experiment catalogue and converts each selected configuration into one invocation of the shared causal-intervention pipeline.

`run_experiments.py` defines:

- `paper-primary`: exactly 28 configurations;
- `paper-auxiliary`: exactly 11 targeted supporting configurations.

The counts are asserted in code. `execution.py` owns the `RunSpec` data model, catalogue filtering, path construction, and shared-pipeline command generation.

Run from `code/`:

```bash
python3 -m studies.overtopping.experiments.run_experiments --suite paper-primary --list
python3 -m studies.overtopping.experiments.run_experiments --suite paper-auxiliary --list
python3 -m studies.overtopping.experiments.run_experiments --suite all --dry-run
```

The execution engine is `pipeline/`, which is shared with poisoning checkpoint analysis and is therefore outside this study package.

See [`../../../docs/overtopping-experiments.md`](../../../docs/overtopping-experiments.md) and [`../../../docs/pipeline.md`](../../../docs/pipeline.md).
