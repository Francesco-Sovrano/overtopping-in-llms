# Technical documentation

The repository implements causal-intervention experiments over language-model internal coordinates. Scientific configuration, model-backed measurements, caches, and derived reporting outputs are stored separately.

## Repository model

```text
repository/
├── code/
│   ├── core/       shared tasks, modeling, attribution, interventions, statistics, and caches
│   ├── pipeline/   numbered model-backed overtopping stages
│   ├── studies/    overtopping and poisoning study configuration and analysis
│   ├── reporting/  aggregate analysis orchestration
│   └── docs/       technical documentation
├── data/           persistent model-backed scientific outputs
├── cache/          reusable computation caches
└── results/        derived analyses, audits, tables, and rendered figures
```

An overtopping setting is identified by its task, model snapshot, intervention/replacement rule, intervention phase, discovery parameters, and evaluation split. The poisoning study uses a separate run namespace containing matched clean and poisoned trajectories.

## Current overtopping registry

The configured registry is constructed by `studies/overtopping/experiments/run_experiments.py`. The `all` selection is deduplicated by scientific identity before execution. In the current repository state it contains 50 unique settings:

```text
phase              I+O  29
phase              Out  21
replacement mean-donor  39
replacement mean        11
```

The public suite selections currently contain 30 `mean-donor`, 3 `6-7b-models`, 8 `mean`, and 12 `checkpoints` entries before cross-suite deduplication. Three final Pythia-1B input+output settings occur in both `mean-donor` and `checkpoints`, so the union contains 50 unique scientific settings.

Inspect the effective registry with:

```bash
./run_overtopping_experiments.sh --list
```

Validate persistent addressing with:

```bash
cd code
python -m studies.overtopping.experiments.storage_contract
```

## Documentation map

| Area | Document |
|---|---|
| Installation, execution, artifact roots | [Getting started](getting-started/README.md) |
| Provider credentials | [Credentials](getting-started/credentials.md) |
| Package and storage architecture | [Architecture](methods/architecture.md) |
| Causal quantities and statistical units | [Core concepts](methods/concepts.md) |
| EAP/EAP-IG attribution | [EAP / EAP-IG](methods/eap.md) |
| Model-backed stages 01–08 | [Pipeline](methods/pipeline.md) |
| Overtopping registry | [Overtopping experiment](experiments/overtopping.md) |
| Poisoning protocol and configuration | [Poisoning experiment](experiments/poisoning/README.md) |
| RQ1–RQ4 estimands and populations | [Research questions](research-questions/README.md) |
| Aggregate analysis and output contracts | [Reporting](reporting/README.md) |
| Regeneration and recovery commands | [Operations](operations/README.md) |

## Artifact semantics

`data/` stores model-backed measurements. `cache/` stores reusable acceleration state. `results/` stores outputs computed from existing measurements. Registry manifests define configured study membership; cache or directory presence is not a population definition.
