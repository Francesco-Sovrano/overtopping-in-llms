# Overtopping Phenomenology

This repository contains two causal-intervention studies of language-model behavior:

- **Overtopping:** discovery and held-out evaluation of high-leverage internal channels, directional singleton reach, simultaneous-set composition, graded intervention response, and checkpoint trajectories.
- **Poisoning:** matched clean/poisoned training trajectories, checkpoint causal analysis, fixed-channel materialization, poisoning-example detection, and cross-seed aggregation.

## Repository layout

```text
code/       source code and documentation
data/       persistent model-backed experiment outputs
cache/      reusable computation caches
results/    regenerated analyses, audits, tables, and figures
```

`data/` and `cache/` are model-backed scientific artifacts. `results/` contains derived reporting products and can be regenerated from compatible inputs.

The reporting population is defined by experiment registries and manifests, not by the set of files that happen to be present in a cache directory.

## Documentation

Start with [`code/docs/README.md`](code/docs/README.md). It covers installation, study design, storage conventions, pipeline stages, research-question populations, reporting, and regeneration.

## Installation

```bash
./setup.sh
```

The setup script creates a Python 3.12 environment in `.env/` and installs the repository requirements. Provider credentials are required only for stages that call the configured external providers. See [`code/docs/getting-started/credentials.md`](code/docs/getting-started/credentials.md).

## Overtopping study

Inspect the configured study without executing model-backed stages:

```bash
./run_overtopping_experiments.sh --list
./run_overtopping_experiments.sh --dry-run
```

Run the configured overtopping programme:

```bash
./run_overtopping_experiments.sh
```

The registry contains 48 intervention settings:

```text
29 final-snapshot task × model × phase cells
12 intermediate Pythia checkpoint settings
 7 matched replacement-baseline repeats
--------------------------------------------
48 configured settings
```

The final-snapshot grid is intentionally non-factorial. It supplies broad task/phase coverage on Qwen2.5-1.5B, targeted family and size comparisons on Qwen2 and Pythia, and Pythia task/checkpoint trajectories. The exact cells and replacement-baseline repeats are listed in [`code/docs/experiments/overtopping.md`](code/docs/experiments/overtopping.md).

The default evaluation split is `test`.

Run only the configurations requiring new model-backed computation for the current coverage design:

```bash
./run_overtopping_experiments.sh --suite minimal-completion
```

## Poisoning study

Inspect or run the poisoning workflow:

```bash
./run_poisoning_experiments.sh --dry-run
./run_poisoning_experiments.sh
```

For a reduced execution-path test:

```bash
POISONING_FAST_TEST=1 ./run_poisoning_experiments.sh
```

Poisoning uses a separate data namespace under `data/poisoning/` and does not enter RQ1–RQ3 overtopping populations.

## Generate analyses and manuscript products

```bash
./generate_results.sh
```

Reporting is best-effort while experiments are still running: every RQ is attempted from the completed/auditable subset, incomplete configured settings are listed in the corresponding audits, and missing settings do not suppress an otherwise renderable RQ. Verified zero-candidate runs remain explicit zero observations. To require a publication-complete 48-setting population and fail on any missing required metric, run:

```bash
ALLOW_INCOMPLETE_METRICS=false ./generate_results.sh
```

The reporting driver reads model-backed inputs from `data/` and writes derived products under `results/`. It rejects output locations inside `data/` or `cache/` and does not rename or relocate overtopping experiment directories or cache keys. The repository wrappers validate the overtopping storage/addressing contract before experiment execution and reporting.

Main populations:

| Analysis | Population rule |
|---|---|
| Configured study table | all 48 overtopping settings; per-metric applicability and availability are recorded separately |
| RQ1 | all 48 settings, analyzed separately for input+output and output-only intervention phases |
| RQ1 sensitivity | 29 unique final-snapshot task×model×phase cells, one replacement condition per cell |
| RQ2 | all evaluable settings within a replacement regime; mean-donor is analyzed separately from mean/mean-positional |
| RQ3 | all configured settings with the required graded or threshold artifacts; denominators are reported by analysis |
| RQ4 | configured Pythia checkpoint trajectories and controlled poisoning trajectories |

Human-facing products are written under `results/paper/`. Machine-readable analysis products and reproducibility audits are written under `results/analysis/`.

## Storage-contract check

The overtopping registry includes a read-only regression check for experiment addressing and pipeline arguments:

```bash
cd code
python -m studies.overtopping.experiments.storage_contract
```

A passing check confirms the configured setting and phase/replacement counts, verifies unique Stage-7 paths for the full registry, and verifies a fixed fingerprint for the storage-protected scientific configurations and persistent paths. Every additional setting must resolve to a non-colliding path. Runtime-only batch size is reported but excluded from the fingerprint.
