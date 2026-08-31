# Getting started

## 1. Install dependencies

From the repository root:

```bash
./setup.sh
```

`setup.sh` creates `.env` with Python 3.12 and installs the full repository dependency set. For a manual full installation:

```bash
python3.12 -m venv .env
. .env/bin/activate
pip install -r code/studies/poisoning/requirements.txt
```

For an overtopping-only environment, `pip install -r requirements.txt` is sufficient. The launchers activate `.env` when it exists. If Ollama is installed, `setup.sh` also downloads the default feature-generation models.

## 2. Runtime directories

The default roots are:

```text
data/       persistent experiment outputs
cache/      regenerable caches
results/    generated reporting products
```

The source root is `code/`. Direct Python invocations should either run from `code/` or include it on `PYTHONPATH`.

## 3. Inspect the overtopping catalogue

```bash
./run_overtopping_experiments.sh --list
./run_overtopping_experiments.sh --dry-run
```

Run the configured catalogue with:

```bash
./run_overtopping_experiments.sh
```

The default evaluation split is `test`.

## 4. Inspect the poisoning study

```bash
./run_poisoning_experiments.sh --dry-run
```

The launcher prints each task/model/seed command before execution. Its configuration block defines the default study matrix and markers.

For a short behavior-only validation run:

```bash
POISONING_FAST_TEST=1 ./run_poisoning_experiments.sh
```

Full-run generation/evaluation defaults to `PIPELINE_BATCH_SIZE=32`. Reduce it if the model does not fit available accelerator memory.

## 5. Run the poisoning study

```bash
./run_poisoning_experiments.sh
```

Each task/model/seed is stored under a single run directory such as:

```text
data/poisoning/arithmetic/
└── confirmatory__Qwen_Qwen2-1.5B-Instruct__seed_13/
    ├── 01_training_checkpoints/
    ├── 02_evaluation_cohorts/
    ├── 03_checkpoint_causal_discovery/
    ├── 04_condition_comparisons/
    ├── 05_behavior_trajectories/
    ├── 06_circuit_overlap_analysis/
    └── 07_poisoning_example_detection/
```

The normal-task behavior cohort defaults to 10,000 deterministic proportional-stratified rows. Set `NORMAL_TASK_SCAN_MAX_ROWS=0` only for exhaustive evaluation. This setting is independent of `TRIGGER_LIFT_SCAN_MAX_ROWS` and Stage-7 point caps.

## 6. Generate final results

```bash
./generate_results.sh
```

Useful reporting controls include:

```bash
REBUILD_DIRECTIONAL_SINGLETONS=true ./generate_results.sh
REBUILD_SPIKING_DIAGNOSTICS=true SPIKING_MAX_POINTS=10000 ./generate_results.sh
SPIKING_SOURCE=/absolute/path/to/spiking_diagnostics... ./generate_results.sh
```

`results/README.md` is the navigation point for generated output.

## 7. Read the scientific contracts

Before changing populations or endpoint definitions, read:

- [Core concepts](concepts.md)
- [Numbered causal pipeline](pipeline.md)
- [Overtopping analysis](overtopping-analysis.md)
- [Poisoning protocol](poisoning-protocol.md)
