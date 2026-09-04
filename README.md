# Overtopping Phenomenology

This repository contains causal-intervention studies of language-model behavior.

The two experiment families are:

- **Overtopping:** discovery and held-out evaluation of high-leverage internal channels, directional singleton reach, joint composition, threshold-event visibility, preemption, and checkpoint trajectories.
- **Poisoning:** matched clean/poisoned training trajectories, checkpoint causal analysis, fixed-channel materialization, poisoning-example detection, and cross-seed aggregation.

The repository separates persistent scientific artifacts from regenerable reporting and caches:

```text
code/       source code and documentation
data/       persistent experiment outputs and provenance
cache/      regenerable computation caches
results/    generated analysis tables, audits, and manuscript figures
```

`data/` defines scientific provenance. `cache/` accelerates computation but does not determine which runs, examples, candidates, or controls enter an analysis.

## Documentation

**Start with [`code/docs/README.md`](code/docs/README.md).** It is the canonical documentation index and routes readers to installation, methods, experiment definitions, RQ1–RQ4, reporting, and operations.

## Installation

```bash
./setup.sh
```

`setup.sh` creates a Python 3.12 virtual environment in `.env/`, installs the repository requirements, installs poisoning-specific runtime helpers, and pulls the default Ollama feature models when Ollama is available.

Provider credentials are optional and depend on the configured prompt/classifier models. See [`code/docs/getting-started/credentials.md`](code/docs/getting-started/credentials.md).

## Overtopping study

Inspect the explicit experiment catalogue before execution:

```bash
./run_overtopping_experiments.sh --list
./run_overtopping_experiments.sh --dry-run
```

Run the configured non-poisoning experiment programme:

```bash
./run_overtopping_experiments.sh
```

The catalogue contains 28 `paper-primary` settings and 11 `paper-auxiliary` settings. Paper-facing overtopping evaluation uses the `test` split unless another split is explicitly requested.

RQ3 requires additional high-N threshold-event diagnostics. From `code/`:

```bash
python -m studies.overtopping.analysis.rebuild_spiking_diagnostics \
  --primary-table ../results/analysis/primary_matrix/tables/primary_table.csv \
  --data-root ../data \
  --population-scope primary+supplementary
```

## Poisoning study

Inspect the configured poisoning commands:

```bash
./run_poisoning_experiments.sh --dry-run
```

Run the configured poisoning study:

```bash
./run_poisoning_experiments.sh
```

For a reduced execution-path test:

```bash
POISONING_FAST_TEST=1 ./run_poisoning_experiments.sh
```

Poisoning is a separate experiment family under `data/poisoning/`; it is not included in RQ1–RQ3 overtopping populations.

## Generate analysis and manuscript outputs

```bash
./generate_results.sh
```

The main analysis populations are:

| Analysis | Population |
|---|---|
| Primary matrix | 28 `paper-primary` settings |
| RQ1 Figure 2 | 39 primary + auxiliary settings |
| RQ2 composition | 28 primary settings |
| RQ3 threshold-event analysis | 28 primary settings plus auxiliary settings with complete exact diagnostics |
| RQ4 checkpoint analysis | configured checkpoint runs |
| Poisoning | configured poisoning task/model/seed runs |

Generated human-facing artifacts are written under `results/paper/`. Machine-readable analysis products and audits are written under `results/analysis/`.

