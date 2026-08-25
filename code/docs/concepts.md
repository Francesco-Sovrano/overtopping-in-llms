# Core concepts

This page defines the terms used across the standard pipeline, final analysis, and checkpointed poisoning study.

## Tasks and task specifications

A task is not only a dataset name. The pipeline expects a task specification implementing `FeatureTaskSpec`, including:

- how prompts and model responses are generated and cached,
- how cached rows are loaded,
- how a prompt is parsed into task tokens/features,
- how model responses are mapped to the task's binary positive predicate,
- task metadata such as the system prompt and token dictionary keys.

Generic pipeline tasks normally resolve as `core.tasks.<task>_task`. The wrapper also supports `--task_module module[:attribute]` so external or multi-endpoint tasks can select a specific task-spec object.

## Standard versus decode-only phase

The experiment catalogue uses two modes:

- `standard` → input+output phase, displayed as `I+O` by `RunSpec.phase`.
- `decode-only` → output/decode-only phase, displayed as `Out`.

`--decode_only` is passed to the pipeline for decode-only configurations. Phase is part of the scientific configuration and should be reported with the model, task, and intervention.

## Evaluation interventions

`RunSpec` accepts these replacement/intervention names:

```text
zero
mean
mean-positional
mean-donor
mean-donor-positional
```

The standard catalogue uses a subset of these values. The exact intervention is part of the run identity even when the output-directory name does not add a suffix for some mean-family baselines.

## Circuit granularity and size

The pipeline can operate at `neuron` or `edge` circuit level. `circuit_size` is a positive integer controlling the retained attribution/circuit budget. Catalogue values can differ from the shell wrapper's generic defaults, so a paper run should be taken from `RunSpec`, not reconstructed from wrapper defaults.

## Spectral splits, anchoring, and discovery

The wrapper distinguishes three choices:

- the dataset/split mode (`--spectral_splits` versus the rule-oriented path),
- how an anchoring/sampling plan is selected (`--spectral_anchoring_plan` or `--random_anchoring_plan`),
- how circuits are discovered (`--spectral_circuit_discovery` or `--random_circuit_discovery`).

`--fast_anchoring` and `--slow_anchoring` select the downstream candidate/ablation strategy. The paper catalogue calls the wrapper with spectral splits and fast anchoring, while wrapper defaults remain more generic.

## Evaluation split and baseline subset

Singleton evaluation accepts:

```text
test
train
all
```

`test` is the generic default and the catalogue default. Output statistics directories encode the split:

```text
test   -heldout_test
train  -eval_train
all    no evaluation suffix
```

Stage 7 can further restrict the selected split using:

```text
--evaluation_baseline_subset all|positive|negative
```

This conditions the denominator on the unablated binary predicate. Poisoning trigger-lift analysis uses the positive subset when the estimand requires rows that were successful trigger-lift events before intervention.

## Candidate selection, singleton evaluation, and CHA

The pipeline first discovers candidate causal units/circuits, then evaluates their intervention effects. A candidate-selection statistic is not itself a held-out causal estimate. The final singleton stage computes intervention outcomes on the declared evaluation universe and writes exact statistics/sidecars used by downstream analysis.

The poisoning documentation uses **CHA** for its causal hypothesis assessment/acceptance operating point. The generic wrapper can receive CHA-related controls through environment variables such as `CHA_REFERENCE_N_PER_SIDE`, `CHA_TAU`, `CHA_LOW_DATA_POLICY`, `CHA_MIN_ACTUAL_N_PER_SIDE`, and `CHA_PRUNE_ALPHA`. When set, these change finite-sample calibration or low-data behavior and therefore belong in run provenance.

## Provenance and scientific identity

A reproducible result requires more than an output filename. At minimum, preserve:

- task and task-spec endpoint,
- model identifier and immutable revision when available,
- local checkpoint identity when a fine-tuned checkpoint is used,
- intervention and phase,
- circuit granularity and size,
- sampling/anchoring/discovery mode,
- evaluation split and baseline subset,
- thresholds and finite-sample controls,
- dataset/cohort identity,
- cache root overrides that affect which model I/O or sampling artifacts are reused.

For poisoning runs, marker strings, poison-rate basis, poison schedule, matched checkpoint manifest, and causal-pool identity are also run-defining.

## Persistent artifacts versus regenerable caches

`data/` contains artifacts needed to identify, interpret, or reproduce a scientific run. `cache/` contains expensive but regenerable intermediates such as prompt/model-I/O or spectral caches. Deleting a cache should force recomputation without changing the intended run identity; changing a persistent cohort, checkpoint, marker set, or evaluation endpoint changes the experiment itself.

## Statistical interpretation

The analysis code distinguishes several metric families and their provenance. Do not infer a final paper metric from a similarly named intermediate file. Use the required-metric audit and analysis stages to verify that expected sidecars exist for the selected primary profile.

For checkpointed poisoning, conditional conversion and unconditional trigger lift have different denominators and answer different questions. The poisoning pages define both explicitly.
