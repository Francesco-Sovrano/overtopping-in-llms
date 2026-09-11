# Documentation

This documentation describes the repository from experiment configuration through model-backed execution and manuscript reporting.

## Reading order

1. [Getting started](getting-started/README.md) — environment, artifact roots, study inspection, execution, and reporting.
2. [Architecture](methods/architecture.md) — package ownership, persistent storage, cache behavior, and population ownership.
3. [Core concepts](methods/concepts.md) — behavioral endpoints, directional causal effects, singleton union, simultaneous-set effects, threshold metrics, and graded interventions.
4. [Pipeline](methods/pipeline.md) — numbered model-backed stages and their outputs.
5. [Overtopping experiment design](experiments/overtopping.md) — the complete 48-setting registry and its non-factorial structure.
6. [Research questions](research-questions/README.md) — populations, estimands, and reporting units for RQ1–RQ4.
7. [Reporting](reporting/README.md) — analysis manifests, completeness audits, statistics, figures, and sidecars.
8. [Operations](operations/README.md) — targeted regeneration, cache-preserving workflows, and troubleshooting.

## Repository layout

```text
repository/
├── code/       source packages and documentation
├── data/       persistent model-backed experiment outputs
├── cache/      reusable computation caches
└── results/    derived analyses and manuscript products
```

Within `code/`:

```text
core/           shared task, model, attribution, intervention, cache, and statistics code
pipeline/       numbered causal-intervention stages
studies/        experiment registries and study-specific analyses
reporting/      final aggregation and manuscript-output generation
docs/           documentation
```

## Storage rule

`data/` and `cache/` are inputs to reporting. `results/` is the reporting output root. The 48-setting study manifest changes which configured settings are represented in aggregate analyses; it does not change the persistent path construction or cache key semantics of an existing `RunSpec`.

Run the read-only registry contract check with:

```bash
cd code
python -m studies.overtopping.experiments.storage_contract
```

## Study populations

The overtopping registry contains 48 settings:

```text
29 final-snapshot task × model × phase cells
12 intermediate Pythia checkpoint settings
 7 matched replacement-baseline repeats
--------------------------------------------
48 configured settings
```

Every configured setting is retained in the study table. Each downstream metric then applies its own applicability and artifact-availability rule. A missing diagnostic is therefore represented as missing for that setting rather than by changing the study manifest.

See [Overtopping experiment design](experiments/overtopping.md) for the exact cells.
