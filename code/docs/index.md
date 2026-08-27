# Documentation

This documentation is organized around the repository's two studies and the shared causal pipeline.

## Start here

- [Getting started](getting-started.md) — installation, first commands, and runtime roots.
- [Architecture](architecture.md) — package boundaries and dependency direction.
- [Repository layout](repository-layout.md) — source-tree and output-tree reference.
- [Concepts](concepts.md) — causal-channel terminology used across the project.

## Shared causal analysis

- [Pipeline](pipeline.md) — Stages 01–08 of the shared causal-discovery/evaluation pipeline.
- [EAP](eap.md) — edge-attribution patching implementation and usage.
- [Interpretation and limitations](interpretation-and-limitations.md) — scope of causal claims and statistical caveats.

## Overtopping study

- [Overtopping experiments](overtopping-experiments.md) — experiment catalogue and execution.
- [Overtopping analysis](overtopping-analysis.md) — study-level analysis and reporting.

## Poisoning study

- [Poisoning overview](poisoning-overview.md) — first-read scientific overview, endpoint separation, CHA contrast, and stage map.
- [Poisoning protocol](poisoning-protocol.md) — complete scientific estimands, causal contrasts, detector construction, attack-efficacy metrics, and validity conditions.
- [Poisoning configuration](poisoning-configuration.md) — launchers, environment variables, and direct stage commands.
- [Poisoning outputs](poisoning-outputs.md) — exact per-run output layout and file meanings.
- [Troubleshooting](troubleshooting.md) — common execution and data-contract failures.

## Source tree

```text
code/
├── core/                  shared utilities and ordinary task definitions
├── pipeline/              shared causal pipeline
├── reporting/             aggregate/manuscript reporting
├── studies/
│   ├── overtopping/       overtopping study
│   └── poisoning/         poisoning study
└── docs/                  this documentation
```
