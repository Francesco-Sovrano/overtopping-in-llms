# Overtopping Phenomenology

This repository contains two related experimental programs for causal analysis of language-model behavior.

- **Overtopping study:** discovers and validates small causal activation channels associated with task behavior.
- **Poisoning study:** trains matched clean and poisoned trajectories, tracks how those causal channels evolve during training, and tests whether poisoning-specific channel disruption can identify the training examples responsible for it.

The poisoning study reuses the same causal-discovery pipeline as the overtopping study. Shared causal machinery therefore lives outside either study package.

## Repository structure

```text
.
├── code/
│   ├── core/                       shared model, task, intervention, and statistics utilities
│   ├── pipeline/                   shared staged causal-discovery pipeline
│   ├── reporting/                  cross-study manuscript/report generation
│   ├── studies/
│   │   ├── overtopping/            overtopping experiment definitions and analysis
│   │   └── poisoning/              poisoning training and checkpoint analysis
│   └── docs/                       project documentation
├── data/                            persistent scientific outputs; created by experiments
├── cache/                           regenerable model/pipeline caches; created by experiments
├── results/                         manuscript-facing aggregate outputs; created by reporting
├── run_overtopping_experiments.sh
├── run_poisoning_experiments.sh
├── generate_results.sh
├── setup.sh
└── requirements.txt
```

`data/`, `cache/`, and `results/` are runtime roots and are not source-code packages.

## Installation

From the repository root:

```bash
./setup.sh
```

or create an environment manually and install:

```bash
pip install -r requirements.txt
```

The poisoning package has an additional requirements file at `code/studies/poisoning/requirements.txt`; it includes the repository requirements and poisoning-specific dependencies.

Run Python modules with `code/` on `PYTHONPATH`, or use the supplied launchers, which configure paths for you.

## Overtopping workflow

Run the overtopping experiment catalogue with:

```bash
./run_overtopping_experiments.sh
```

The shared causal pipeline is organized as `pipeline/stage01_...` through `pipeline/stage08_...`. The study-specific overtopping configuration and reporting code lives under `code/studies/overtopping/`.

See [Overtopping experiments](code/docs/overtopping-experiments.md), [Pipeline](code/docs/pipeline.md), and [Overtopping analysis](code/docs/overtopping-analysis.md).

## Poisoning workflow

The poisoning study has one scientific run directory per task/model/seed, for example:

```text
data/poisoning/grammar/
└── confirmatory__Qwen_Qwen2-1.5B-Instruct__seed_13/
    ├── 01_training_checkpoints/
    ├── 02_evaluation_cohorts/
    ├── 03_checkpoint_causal_discovery/
    ├── 04_condition_comparisons/
    ├── 05_behavior_trajectories/
    ├── 06_circuit_overlap_analysis/
    └── 07_poisoning_example_detection/
```

Stages 01–07 remain inside that same run directory. Only Stage 08, which combines multiple runs/seeds, is written under `data/poisoning/final/<study>/08_cross_seed_aggregation/`.

The default scientific comparison is a matched pair of trajectories:

1. **clean:** normal training with the control marker;
2. **poisoned:** identical training construction except selected matched counterfactual slots receive the trigger marker and attacker target.

Checkpoint analysis separates full-cohort normal-task behavior, attack-cohort backdoor behavior, and attack-cohort control-correctness CHA. Stage 07 uses the control-correctness causal endpoint. It forms the union of checkpoint-local agonist candidates, evaluates each union channel on one fixed held-out attack cohort, and defines `U(j)` as the correct→incorrect singleton rate on that fixed denominator. Developmental disruption is the poisoned-minus-clean change in `U(j)`. Training rows are ranked with a WANDA-style score based on projection activations and the clean-normalized effective LoRA interval update `[(scaling * B @ A)_p,end-(...)_p,start]-[(...)_c,end-(...)_c,start]`. Poison labels are applied after scoring for detector evaluation. Stage 07 also tests the association between interval detectability and change in conditional trigger conversion using a two-sided permutation Spearman test.

Run the configured poisoning study with:

```bash
./run_poisoning_experiments.sh
```

Inspect commands without executing them:

```bash
./run_poisoning_experiments.sh --dry-run
```

See [Poisoning overview](code/docs/poisoning-overview.md), [Poisoning protocol](code/docs/poisoning-protocol.md), [Poisoning configuration](code/docs/poisoning-configuration.md), and [Poisoning outputs](code/docs/poisoning-outputs.md).

## Reporting

Generate manuscript-facing outputs with:

```bash
./generate_results.sh
```

Cross-seed poisoning summaries include behavior/circuit trajectories and Stage-07 poisoned-example detection metrics when those artifacts are available.

## Documentation

Start at [code/docs/index.md](code/docs/index.md). The architecture and filesystem contracts are documented in [Architecture](code/docs/architecture.md) and [Repository layout](code/docs/repository-layout.md).

## Poisoning detector invariants

Ordinary-correctness analysis uses the full held-out task distribution for both grammar and arithmetic. Stage 07 compares immutable example IDs together with gold labels/answers across matched states, applies paired fixed-cohort uncertainty to `D_j`, and uses a positive effect-size threshold before WANDA scoring. Arithmetic `output_only` scoring uses the causal-LM one-token alignment shift.

Stage 08 aggregates Stage-07 detector outputs independently of optional behavioral/backdoor trajectory outputs.
