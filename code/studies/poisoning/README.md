# Poisoning study package

This package implements the checkpointed poisoning study. It trains matched clean and poisoned trajectories, evaluates marker-triggered behavior, localizes checkpoint-specific causal channels through the shared pipeline, aggregates checkpoint trajectories, tests inference-time suppression and specificity, and supports training-time protection controls.

Install the poisoning runtime dependencies from repository root after activating the environment:

```bash
python -m pip install -r code/studies/poisoning/requirements.txt
```

## Stage naming

A `stageNN_` prefix is used only when a file belongs to one numbered scientific/output stage. The number matches the persistent poisoning stage directory or study-level output stage. Orchestrators and helpers that span several stages remain unnumbered.

The poisoning workflow is:

| Stage | Persistent purpose | Main implementation |
|---|---|---|
| 01 | checkpoint training | `scripts/stage01_run_checkpoint_training.sh` |
| 02 | deterministic evaluation cohorts | `stage02_prepare_evaluation_cohorts.py` |
| 03 | checkpoint causal discovery | shared `pipeline/run_pipeline.sh`, invoked by `scripts/run_checkpoint_causal_workflow.sh` |
| 04 | clean/poisoned condition comparisons | `stage04_compare_condition_behavior.py` |
| 05 | behavior trajectories | `stage05_aggregate_backdoor_trajectory.py` |
| 06 | checkpoint circuit overlap | `stage06_compare_checkpoint_circuits.py` |
| 07 | inference- and training-time defence evaluation | `stage07_*.py` plus `scripts/stage07_*.sh` |
| 08 | cross-seed aggregation and figures | `stage08_aggregate_cross_seed.py`, `stage08_plot_cross_seed.py` |

`run_checkpoint_causal_workflow.sh` is deliberately unnumbered: it coordinates Stages 02–06 and invokes the shared pipeline for Stage 03. `run_ordinary_correctness_control.sh` and `poisoning_runtime_config.sh` are helpers rather than standalone scientific stages.

## Package structure

```text
studies/poisoning/
├── tasks/       grammar/arithmetic task definitions and task registry
├── lib/         shared training, marker, checkpoint, trajectory, and defence logic
├── scripts/
│   ├── poisoning_runtime_config.sh
│   ├── stage01_run_checkpoint_training.sh
│   ├── run_checkpoint_causal_workflow.sh
│   ├── run_ordinary_correctness_control.sh
│   ├── stage07_run_inference_defence.sh
│   └── stage07_run_training_defence.sh
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

There is intentionally no poisoning-local `stage03_*.py`: Stage 03 is implemented by the shared causal pipeline. Multiple Stage 07 files are intentional because the stage contains two distinct defence mechanisms and their common summary.

Task modules own dataset construction, target semantics, output parsing, behavioral statistics, and the task specs passed to `pipeline/`. Shared poisoning mechanics belong in `studies/poisoning/lib/`.

## Main entry points

Run task CLIs from `code/`:

```bash
python3 -m studies.poisoning.tasks.grammar --help
python3 -m studies.poisoning.tasks.arithmetic --help
```

Preview checkpoint training:

```bash
POISONING_TASK=grammar DRY_RUN=1 bash studies/poisoning/scripts/stage01_run_checkpoint_training.sh
```

For a completed run, the checkpoint-analysis workflow is driven by:

```bash
POISONING_TASK=grammar \
RUN_DIR=../data/poisoning/grammar/<run-name> \
bash studies/poisoning/scripts/run_checkpoint_causal_workflow.sh
```

That orchestrator prepares the Stage 02 cohort, runs Stage 03 causal discovery through the shared pipeline, and writes the Stage 04–06 poisoning summaries.

## Documentation

Read the canonical poisoning documentation in this order:

1. [`../../docs/poisoning-overview.md`](../../docs/poisoning-overview.md)
2. [`../../docs/poisoning-protocol.md`](../../docs/poisoning-protocol.md)
3. [`../../docs/poisoning-configuration.md`](../../docs/poisoning-configuration.md)
4. [`../../docs/poisoning-outputs.md`](../../docs/poisoning-outputs.md)
