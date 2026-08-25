# Documentation guide

This repository studies causal intervention channels in language models through two experimental programmes: overtopping analysis and poisoning/backdoor analysis. Both studies use the same causal-intervention pipeline and reusable model/task primitives.

A first-time reader should use the documentation in this order:

1. [Getting started](getting-started.md) — environment, installation, validation, and entry points.
2. [Repository layout](repository-layout.md) — source ownership, persistent data, caches, and results.
3. [Core concepts](concepts.md) — tasks, features, rules, channels, interventions, held-out evaluation, and terminology.
4. [Shared causal pipeline](pipeline.md) — Stages 01–08 and their artifacts.
5. [Overtopping experiments](overtopping-experiments.md) — experiment catalogue and execution.
6. [Overtopping analysis](overtopping-analysis.md) — paper-facing metrics, figures, and ordered analysis stages.
7. [Poisoning overview](poisoning-overview.md) — complete poisoning workflow and stage map.
8. [Poisoning protocol](poisoning-protocol.md) — experimental design and confirmatory comparisons.
9. [Poisoning configuration](poisoning-configuration.md) — environment variables and runtime controls.
10. [Poisoning outputs](poisoning-outputs.md) — run directory, cache directory, and final outputs.
11. [Interpretation and limitations](interpretation-and-limitations.md) — what the measurements support and what they do not establish.
12. [Troubleshooting](troubleshooting.md) — validation, cache diagnosis, missing artifacts, and common runtime failures.

[EAP / EAP-IG](eap.md) documents the edge-attribution implementation used where EAP-based circuit discovery is configured.

## Source packages

The Python import root is `code/`:

```text
code/
├── core/                    reusable task/model/intervention primitives
├── pipeline/                shared causal pipeline
├── reporting/               aggregate/final-result orchestration
├── studies/
│   ├── overtopping/         overtopping experiment and analysis code
│   └── poisoning/           poisoning training, causal analysis, and defence
└── docs/                    documentation
```

The repository root contains the main shell entry points:

```text
run_overtopping_experiments.sh
run_poisoning_experiments.sh
generate_results.sh
setup.sh
```

All Python module commands in this documentation assume either that the current directory is `code/` or that `code/` is present on `PYTHONPATH`.
