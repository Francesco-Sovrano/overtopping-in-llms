# Overtopping Phenomenology

This repository contains two related causal-analysis studies for language models:

- **Overtopping:** discovers task-relevant activation channels, measures singleton and set-level causal effects, validates interactions, and studies how causal organization relates to competence.
- **Poisoning:** trains matched clean and poisoned trajectories, evaluates behavior and causal organization at checkpoints, and tests whether poisoning-specific causal disruption identifies responsible training examples.

Shared model loading, task interfaces, interventions, statistics, and the numbered causal pipeline live under `code/core/` and `code/pipeline/`. Study-specific experiment definitions and analyses live under `code/studies/`.

## Installation

From the repository root:

```bash
./setup.sh
```

`setup.sh` creates a Python 3.12 virtual environment and installs the full repository dependency set. For an overtopping-only environment, install the base requirements manually:

```bash
pip install -r requirements.txt
```

For poisoning support, install `code/studies/poisoning/requirements.txt`; it includes the base requirements and the additional runtime helpers.

## Runtime roots

```text
code/       source code
data/       persistent experiment outputs
cache/      regenerable model and pipeline caches
results/    generated paper and analysis products
```

The launchers resolve these paths from the repository root. Most roots can also be overridden with environment variables documented under `code/docs/`.

## Overtopping

List the configured experiments:

```bash
./run_overtopping_experiments.sh --list
```

Inspect commands:

```bash
./run_overtopping_experiments.sh --dry-run
```

Run the configured catalogue:

```bash
./run_overtopping_experiments.sh
```

The primary manuscript profile contains 28 task/model/phase settings. Test is the default evaluation split. The shared causal pipeline runs Stages 01–08; study-level analysis then builds the primary matrix, statistical analyses, manuscript figures, threshold/spiking diagnostics, and audits.

## Poisoning

Inspect the configured run matrix:

```bash
./run_poisoning_experiments.sh --dry-run
```

Run it:

```bash
./run_poisoning_experiments.sh
```

Each task/model/seed has one run directory with stages for training checkpoints, evaluation cohorts, checkpoint causal analysis, matched comparisons, trajectories, circuit overlap, and poisoning-example detection. Cross-seed aggregation is stored separately under `data/poisoning/final/`.

Three checkpoint endpoints are intentionally separate:

1. `normal_task`: no-trigger behavioral accuracy on a deterministic proportional stratified held-out sample. `NORMAL_TASK_SCAN_MAX_ROWS` defaults to 10,000; `0` requests the complete held-out population.
2. `backdoor_trigger_test`: paired trigger/control behavior on the attack-eligible cohort.
3. `attack_cohort_control_correctness`: causal control-correctness analysis on the exact attack-eligible non-target cohort.

The normal-task cap is independent of trigger-lift and Stage-7 caps. Exact cached no-trigger generations may be reused when row identities overlap, but cache overlap does not define the normal-task population.

Full-run generation/evaluation uses `PIPELINE_BATCH_SIZE=32` unless overridden. Smoke mode uses a smaller batch and reduced cohorts.

## Reporting

Generate final outputs:

```bash
./generate_results.sh
```

The result tree is organized under:

```text
results/
├── paper/       manuscript-facing figures and tables
└── analysis/    matrices, audits, diagnostics, and machine-readable figure data
```

Directional causal quantities use explicit names such as `U_J_i2c` for baseline 0→1 and `U_J_c2i` for baseline 1→0.

RQ3 threshold diagnostics can be rebuilt during reporting with:

```bash
REBUILD_SPIKING_DIAGNOSTICS=true ./generate_results.sh
```

The reporting pipeline restricts RQ3 to the exact primary-table population and audits required positive/negative baseline subsets before producing the report.

## Documentation

Start with [code/docs/index.md](code/docs/index.md). It links the installation, architecture, pipeline, overtopping, poisoning, output, figure, and troubleshooting references.
