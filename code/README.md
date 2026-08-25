# Source-code layout

`code/` is the Python import root. Its directories are organized by dependency and study ownership.

```text
code/
├── core/                       reusable primitives shared by both studies
├── pipeline/                   shared causal-intervention workflow
├── reporting/                  cross-study result orchestration
├── studies/
│   ├── overtopping/
│   │   ├── experiments/        catalogue and execution
│   │   └── analysis/           overtopping-specific analysis
│   └── poisoning/              poisoning-specific workflow
└── docs/                       canonical documentation
```

## Shared packages

### `core/`

`core/` contains task interfaces, ordinary-task implementations, model loading and ablation utilities, feature representations, statistics, spectral helpers, EAP/EAP-IG, and repository-path utilities. Both studies and the shared pipeline import it.

### `pipeline/`

`pipeline/` contains the numbered causal-intervention stages. It is used by ordinary overtopping configurations and by poisoning checkpoint analysis. It must therefore remain outside either study package.

### `reporting/`

`reporting/` contains orchestration that spans study boundaries. `reporting.generate_final_results` invokes overtopping analysis stages and available poisoning cross-seed aggregation without moving study-specific statistical logic into a generic package.

## Study packages

### `studies/overtopping/`

`experiments/` defines the explicit overtopping catalogue and converts `RunSpec` objects into shared-pipeline commands. `analysis/` contains overtopping-specific tables, statistics, figures, diagnostics, and primary-matrix helpers.

### `studies/poisoning/`

The poisoning package owns poisoning task definitions, matched clean/poisoned training, marker handling, checkpoint manifests, trajectory analysis, circuit-overlap analysis, inference-time defence, training-time protection, and cross-seed aggregation. It calls the shared pipeline for its Stage 03 causal discovery.

## Import convention

Run Python modules from `code/`:

```bash
python3 -m studies.overtopping.experiments.run_experiments --help
python3 -m pipeline.stage01_generate_prompts_and_answers --help
python3 -m studies.poisoning.tasks.grammar --help
python3 -m reporting.generate_final_results --help
```

Shared imports use `core.*`, overtopping imports use `studies.overtopping.*`, and poisoning imports use `studies.poisoning.*`.

## Stage filenames

Use `stageNN_` only for files that implement one documented ordered stage. Orchestrators spanning several stages remain unnumbered. The stage number must match the corresponding scientific/output stage.

See [`docs/repository-layout.md`](docs/repository-layout.md) and [`docs/architecture.md`](docs/architecture.md) for the complete ownership and dependency model.
