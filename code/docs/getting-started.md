# Getting started

## 1. Create the environment

From the repository root:

```bash
./setup.sh
```

or install manually:

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

The launchers automatically activate `.env/bin/activate` when it exists.

## 2. Understand the runtime roots

The repository keeps source and generated artifacts separate:

```text
code/       source packages
data/       persistent experimental outputs/checkpoints
cache/      regenerable caches
results/    aggregate/manuscript-facing outputs
```

Do not place generated experiment results under `code/`.

## 3. Check the source architecture

```text
code/
├── core/                  shared utilities
├── pipeline/              shared causal pipeline
├── reporting/             aggregate reporting
└── studies/
    ├── overtopping/
    └── poisoning/
```

Read [Architecture](architecture.md) for package ownership and [Repository layout](repository-layout.md) for file-level structure.

## 4. Run overtopping experiments

Inspect the experiment documentation in [Overtopping experiments](overtopping-experiments.md), then run:

```bash
./run_overtopping_experiments.sh
```

The causal pipeline is documented in [Pipeline](pipeline.md).

## 5. Inspect the poisoning command plan

Before starting checkpoint training or causal analysis:

```bash
./run_poisoning_experiments.sh --dry-run
```

Verify task, model, seed, run name, marker strings, poison rate, data root, and Stage-07 detector command.

The configured run directory has the form:

```text
data/poisoning/<task>/<run_id>/
```

For example:

```text
data/poisoning/grammar/confirmatory__Qwen_Qwen2-1.5B-Instruct__seed_13/
```

## 6. Run the poisoning workflow

```bash
./run_poisoning_experiments.sh
```

The normal workflow creates:

```text
<run_dir>/
├── 01_training_checkpoints/
├── 02_evaluation_cohorts/
├── 03_checkpoint_causal_discovery/
├── 04_condition_comparisons/
├── 05_behavior_trajectories/
├── 06_circuit_overlap_analysis/
└── 07_poisoning_example_detection/
```

Stage 08 aggregates across runs under `data/poisoning/final/<study>/08_cross_seed_aggregation/`.

## 7. Use smoke mode before expensive runs

```bash
POISONING_FAST_TEST=1 ./run_poisoning_experiments.sh
```

Smoke mode is behavior-first and stops before the expensive poisoned-example detector/cross-seed aggregation. It is intended to catch dataset, marker, and training problems early.

## 8. Run Stage 07 on an existing run

Stage 07 uses the attack-cohort control-correctness causal endpoint. From `code/`:

```bash
python3 -m studies.poisoning.stage07_detect_poisoning_examples \
  --run_dir ../data/poisoning/grammar/confirmatory__Qwen_Qwen2-1.5B-Instruct__seed_13 \
  --task grammar \
  --phase input_output \
  --eval_intervention mean-donor \
  --required_tau 0.3
```

The detector requires matched clean/poisoned LoRA checkpoints, fraction-zero and later `attack_cohort_control_correctness` Stage-03 outputs, `uniform_optimizer_steps` scheduling, and one training epoch. Stage 07 builds the candidate union and fixed-cohort singleton `U(j)` materializations under the run-local `07_poisoning_example_detection/` directory. The backdoor behavior trajectory is used separately for the detectability/attack association test.

## 9. Generate aggregate results

```bash
./generate_results.sh
```

The reporting layer reads persistent data and writes manuscript-facing aggregate outputs under `results/`.

## 10. Read the protocol before changing a poisoning study

Use these documents together:

- [Poisoning overview](poisoning-overview.md) for the scientific stage map;
- [Poisoning protocol](poisoning-protocol.md) for experimental definitions and detector methodology;
- [Poisoning configuration](poisoning-configuration.md) before changing markers, poison rate, schedule, checkpoint selection, or detector settings;
- [Poisoning outputs](poisoning-outputs.md) for the filesystem contract;
- [Troubleshooting](troubleshooting.md) when a stage refuses an incomplete or incompatible run.
