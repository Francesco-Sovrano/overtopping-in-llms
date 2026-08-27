# Code packages

The `code/` directory contains importable Python packages and project documentation.

```text
code/
├── core/                 shared utilities and ordinary task definitions
├── pipeline/             shared causal-discovery pipeline
├── reporting/            aggregate/manuscript reporting
├── studies/
│   ├── overtopping/      overtopping-specific experiments and analysis
│   └── poisoning/        poisoning-specific training and checkpoint analysis
└── docs/                 self-contained documentation
```

## Ownership rules

`core/` contains reusable primitives: task specifications, model/intervention helpers, statistics, spectral analysis, and group/singleton utilities.

`pipeline/` contains the ordered causal workflow used by both studies. Its stage prefix describes scientific pipeline order rather than package ownership.

`studies/overtopping/` owns overtopping experiment catalogues and study-specific analysis.

`studies/poisoning/` owns matched clean/poisoned training, trigger semantics, deterministic cohorts, checkpoint trajectory analysis, circuit comparison, and poisoned-training-example detection. Stage 03 delegates to `pipeline/` instead of duplicating causal machinery.

`reporting/` combines persistent study outputs into manuscript-facing tables and figures.

Runtime data do not belong under `code/`. Scientific outputs are stored under `data/`; regenerable caches under `cache/`; aggregate reporting outputs under `results/`.

See [docs/index.md](docs/index.md).
