# Repository documentation

This documentation is the reference for installing the project, understanding its scientific objects, running model-backed experiments, and regenerating analyses.

## What the repository measures

The repository studies how internal model coordinates causally affect discrete behavioral endpoints. The overtopping study discovers candidate channels and evaluates their effects under singleton, simultaneous-set, graded, and threshold-oriented analyses. The poisoning study follows causal organization through matched clean and poisoned training trajectories and evaluates poisoning-example detection after causal quantities have been fixed.

The main scientific distinction is between **configuration**, **measurement**, and **reporting**:

- a configured experiment defines task, model state, intervention phase, replacement baseline, population, and discovery/evaluation parameters;
- model-backed stages materialize scientific measurements under `data/`;
- reporting consumes those measurements and writes derived products under `results/`.

Cache contents never define study membership.

## Recommended reading order

1. [Getting started](getting-started/README.md) — installation, credentials, artifact roots, dry runs, execution, and reporting.
2. [Architecture](methods/architecture.md) — package ownership, persistent storage, cache behavior, and scientific population ownership.
3. [Core concepts](methods/concepts.md) — behavioral endpoints, directional singleton effects, `U(J)`, `E(J)`, composition, threshold metrics, and graded interventions.
4. [Pipeline](methods/pipeline.md) — numbered model-backed stages and their persistent outputs.
5. [Overtopping experiment design](experiments/overtopping.md) — current configurable registry and its non-factorial structure.
6. [Poisoning protocol](experiments/poisoning/README.md) — matched training, checkpoint evaluation, candidate localization, detector construction, and cross-seed aggregation.
7. [Research questions](research-questions/README.md) — RQ1–RQ4 populations, estimands, and statistical units.
8. [Reporting](reporting/README.md) — analysis manifests, completeness audits, statistics, figures, and sidecars.
9. [Operations](operations/README.md) — targeted regeneration and cache-preserving recovery workflows.

## Repository layout

```text
repository/
├── code/
│   ├── core/       shared tasks, model/intervention utilities, attribution, statistics, and caches
│   ├── pipeline/   numbered model-backed causal-intervention stages
│   ├── studies/    overtopping and poisoning study definitions and analyses
│   ├── reporting/  aggregate reporting and manuscript-output generation
│   └── docs/       documentation
├── data/           persistent model-backed scientific outputs
├── cache/          reusable computation caches
└── results/        derived analyses, audits, tables, and figures
```

## Overtopping study registry

The registry is explicit and intentionally has no required setting count. Add or remove `RunSpec` entries in the experiment registry as needed; reporting derives its configured population from the current registry or an explicit `configured_experiments.json` catalogue. `run_overtopping_experiments.sh --list` prints the current settings and totals.

Every configured setting remains represented in the study table. Each downstream metric applies its own applicability and artifact-availability rule and reports its denominator.

Validate the registry and persistent path contract with:

```bash
cd code
python -m studies.overtopping.experiments.storage_contract
```

The check is read-only and reports current counts and a fingerprint; it fails only on persistent-address collisions between distinct scientific settings.

## Artifact semantics

```text
data/       measured scientific artifacts; preserve unless intentionally recomputing the experiment
cache/      reusable acceleration state; reuse only when cache metadata matches the requested computation
results/    derived outputs; safe to regenerate from compatible scientific inputs
```

Reporting does not use directory presence as a substitute for the configured population. A measured zero-candidate result is different from an unmeasured or incomplete setting.
