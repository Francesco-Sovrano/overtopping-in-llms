# Poisoning study

`studies.poisoning` implements checkpointed clean/poisoned training, marker-triggered behavioral evaluation, checkpoint causal discovery, trajectory analysis, circuit comparison, and defence experiments.

## Workflow

| Stage | Purpose | Entry point |
|---|---|---|
| 01 | train clean and poisoned checkpoint trajectories | `scripts/stage01_run_checkpoint_training.sh` |
| 02 | prepare deterministic evaluation cohorts | `stage02_prepare_evaluation_cohorts.py` |
| 03 | run checkpoint causal discovery | shared `pipeline/run_pipeline.sh`, coordinated by `scripts/run_checkpoint_causal_workflow.sh` |
| 04 | compare clean and poisoned behavior | `stage04_compare_condition_behavior.py` |
| 05 | aggregate checkpoint behavior trajectories | `stage05_aggregate_backdoor_trajectory.py` |
| 06 | compare checkpoint circuits | `stage06_compare_checkpoint_circuits.py` |
| 07 | evaluate inference-time defence and training-time protection | `stage07_*.py`, `scripts/stage07_*.sh` |
| 08 | aggregate and plot across seeds | `stage08_aggregate_cross_seed.py`, `stage08_plot_cross_seed.py` |

There is no poisoning-local `stage03_*.py` because causal discovery is implemented by the shared pipeline. `run_checkpoint_causal_workflow.sh` is unnumbered because it coordinates several poisoning stages.

## Package structure

```text
studies/poisoning/
├── tasks/                       grammar/arithmetic definitions and registry
├── lib/                         poisoning mechanics shared by multiple stages
├── scripts/                     shell entry points and runtime configuration
├── tests/                       filesystem/path regression tests
├── stage02_prepare_evaluation_cohorts.py
├── stage04_compare_condition_behavior.py
├── stage05_aggregate_backdoor_trajectory.py
├── stage06_compare_checkpoint_circuits.py
├── stage07_inference_cumulative_ablation.py
├── stage07_training_verify_matched_runs.py
├── stage07_training_compare_protection.py
├── stage07_build_defence_overview.py
├── stage08_aggregate_cross_seed.py
└── stage08_plot_cross_seed.py
```

`lib/run_paths.py` defines shared poisoning filesystem names. Inference-defence cache persistence is owned directly by `lib/cumulative_ablation.py`, the only component that reads and writes those cache rows.

## Installation

From repository root after activating the project environment:

```bash
python -m pip install -r code/studies/poisoning/requirements.txt
```

## Running the study

Preview the default task/seed grid from repository root:

```bash
./run_poisoning_experiments.sh --dry-run
```

The default grid is grammar and arithmetic across seeds 13, 37, and 101 using the configured task model lists.

Run a completed checkpoint trajectory through causal analysis from `code/`:

```bash
POISONING_TASK=grammar \
RUN_DIR=../data/poisoning/grammar/<run_id> \
bash studies/poisoning/scripts/run_checkpoint_causal_workflow.sh
```

Run inference-time defence for one completed run:

```bash
POISONING_TASK=grammar \
RUN_DIR=../data/poisoning/grammar/<run_id> \
POISONING_CACHE_ROOT=../cache/poisoning \
bash studies/poisoning/scripts/stage07_run_inference_defence.sh
```

The canonical inference-defence cache is:

```text
cache/poisoning/<task>/<run_id>/defence/<input_output|output_only>/fraction_<fraction>/
```

## Documentation

Read [poisoning overview](../../docs/poisoning-overview.md), [protocol](../../docs/poisoning-protocol.md), [configuration](../../docs/poisoning-configuration.md), and [outputs/cache layout](../../docs/poisoning-outputs.md).
