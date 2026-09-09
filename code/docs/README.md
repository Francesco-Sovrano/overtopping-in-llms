# Documentation

This documentation describes the repository from experiment configuration through manuscript output generation. It is organized for readers who have not worked with the code before.

## Reading order

1. [Getting started](getting-started/README.md) — environment, runtime roots, experiment launchers, and result generation.
2. [Architecture](methods/architecture.md) — ownership of code packages and artifact roots.
3. [Core concepts](methods/concepts.md) — behavioral endpoints, directional causal effects, union reach, composition, threshold-event metrics, and graded interventions.
4. [Pipeline](methods/pipeline.md) — numbered model-backed execution stages and their persistent outputs.
5. [Experiments](experiments/README.md) — overtopping and controlled-poisoning study designs.
6. [Research questions](research-questions/README.md) — scientific populations, estimands, and manuscript outputs for RQ1–RQ4.
7. [Reporting](reporting/README.md) — aggregation, audits, figures, and result-tree conventions.
8. [Operations](operations/README.md) — targeted regeneration and troubleshooting without unnecessary cache deletion.

## Repository model

```text
repository/
├── code/       source packages and this documentation
├── data/       persistent scientific outputs
├── cache/      reusable computation caches
└── results/    derived analyses and manuscript products
```

Within `code/`:

```text
core/           shared task, model, intervention, attribution, and statistics code
pipeline/       numbered causal-intervention stages
studies/        study-specific experiment and analysis packages
reporting/      final aggregation, audits, and manuscript-output generation
docs/           documentation
```

## Documentation map

```text
docs/
├── README.md
├── getting-started/
│   ├── README.md
│   └── credentials.md
├── methods/
│   ├── README.md
│   ├── architecture.md
│   ├── concepts.md
│   ├── pipeline.md
│   └── eap.md
├── experiments/
│   ├── README.md
│   ├── overtopping.md
│   └── poisoning/
│       ├── README.md
│       ├── configuration.md
│       └── outputs.md
├── research-questions/
│   ├── README.md
│   ├── rq1-prevalence.md
│   ├── rq2-composition.md
│   ├── rq3-threshold-event.md
│   └── rq4-learning.md
├── reporting/
│   ├── README.md
│   ├── analysis-pipeline.md
│   ├── figures.md
│   └── interpretation.md
└── operations/
    └── README.md
```

## Common tasks

| Task | Reference |
|---|---|
| Install and execute the code | [Getting started](getting-started/README.md) |
| Understand `data/`, `cache/`, and `results/` | [Architecture](methods/architecture.md) |
| Understand `U(J)`, `E(J)`, directional effects, threshold MCC, TECS, or graded crossings | [Core concepts](methods/concepts.md) |
| Trace pipeline execution | [Pipeline](methods/pipeline.md) |
| Inspect the exact overtopping catalogue | [Overtopping catalogue](experiments/overtopping.md) |
| Understand RQ1 | [RQ1 — prevalence](research-questions/rq1-prevalence.md) |
| Understand RQ2 | [RQ2 — composition and interaction regimes](research-questions/rq2-composition.md) |
| Understand RQ3 | [RQ3 — support-specific thresholded integration](research-questions/rq3-threshold-event.md) |
| Understand RQ4 | [RQ4 — learning](research-questions/rq4-learning.md) |
| Understand poisoning | [Poisoning protocol](experiments/poisoning/README.md) |
| Trace a figure to source data | [Figure map](reporting/figures.md) |
| Regenerate derived outputs | [Operations](operations/README.md) |
