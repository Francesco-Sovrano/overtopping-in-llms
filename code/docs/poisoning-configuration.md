# Poisoning configuration and execution

This page documents the current entry points and configuration controls for checkpointed grammar and arithmetic poisoning experiments. Commands assume the shell is in `code/` unless a repository-root path is shown explicitly.

## Entry points

### Repository-root matrix

From repository root:

```bash
./run_poisoning_experiments.sh --dry-run
./run_poisoning_experiments.sh
```

The root launcher expands a task × model × seed matrix, then runs matched training, checkpoint causal discovery, cumulative suppression/specificity analysis, and matrix aggregation.

Default matrix settings:

| Variable | Default |
|---|---|
| `POISONING_TASKS` | `grammar,arithmetic` |
| `SEEDS` | `13,37,101` |
| `GRAMMAR_MODEL_NAMES` | `Qwen/Qwen2-1.5B-Instruct` |
| `ARITHMETIC_MODEL_NAMES` | `Qwen/Qwen2-1.5B-Instruct` |
| `POISONING_RUN_NAME` | `confirmatory` |
| `POISON_RATE` | `0.1` |
| `POISON_RATE_BASIS` | `eligible_gold_non_target` |
| `POISON_TRAINING_MODE` | `paired_counterfactual` |
| `POISON_SCHEDULE_MODE` | `uniform_optimizer_steps` |
| `CONTROL_MARKER` | one space (`" "`) |
| `TRIGGER_MARKER` | `[id=74291]` |
| `SHAM_MARKER` | two spaces (`"  "`) |
| `SHAM_MAX_ROWS` | `512` |

`MODEL_NAMES` applies a common comma-separated model list to both tasks. When it is unset, the task-specific model variables above are used.

`POISONING_FAST_TEST=1` selects a smaller behavior-first smoke configuration. It reduces training/evaluation caps, analyzes only the start and final checkpoints, disables expensive report/defense paths, and defaults to arithmetic unless `POISONING_TASKS` is already set.

### Direct task CLIs

```bash
python3 -m studies.poisoning.tasks.grammar --help
python3 -m studies.poisoning.tasks.arithmetic --help
```

Use these when task-specific arguments need to be controlled directly.

### Shared checkpoint-training driver

```bash
POISONING_TASK=grammar DRY_RUN=1 bash studies/poisoning/scripts/stage01_run_checkpoint_training.sh
```

This shell driver translates environment variables into the canonical task CLI and activates `<repo>/.env` when it exists.

Direct driver defaults:

| Variable | Default |
|---|---|
| `POISONING_TASK` | `grammar` |
| `CONDITION` | `both` |
| `MODEL_NAME` | `Qwen/Qwen2-1.5B-Instruct` |
| `MODEL_REVISION` | unset |
| `MAX_TRAIN` | `4000` |
| `MAX_EVAL` | `500` |
| `PREFLIGHT_MAX_EVAL` | `2048` |
| `MAX_CAUSAL_EVAL` | task default/full remaining cohort |
| `SEED` | `13` |
| `POISON_RATE` | `0.03` |
| `POISON_RATE_BASIS` | `total_train` |
| `POISON_TRAINING_MODE` | `paired_counterfactual` |
| `POISON_SCHEDULE_MODE` | `uniform_optimizer_steps` |
| `CONTROL_MARKER` | one space (`" "`) |
| `TRIGGER_MARKER` | `[id=74291]` |
| `SHAM_MARKER` | two spaces (`"  "`) |
| `SHAM_MAX_ROWS` | `512` |
| `NUM_TRAIN_EPOCHS` | `1` |
| `SAVE_FRACS` | `0,0.1,0.25,0.5,0.75,1.0` |
| `LEARNING_RATE` | `0.0002` |
| `GRAD_ACCUM` | `16` |
| `BATCH_SIZE` | `1` |
| `LOAD_IN_4BIT` | `0` |
| `MAX_BASE_TRIGGER_LIFT` | `0.05` |
| `MAX_BASE_TRIGGER_CHANGE` | `0.05` |
| `MAX_BASE_TRIGGER_SUPPRESSION` | `0.05` |
| `EVAL_BATCH_SIZE` | `8` |
| `HF_CHECKPOINT_DIAGNOSTIC` | `0` |

Grammar-specific shell controls include `DATASET_PATH`, `TARGET_LABEL` (default `acceptable`), and `MAX_LENGTH` (default `256`). Arithmetic-specific controls include `MAX_OPERAND` (default `300`), `OPERATORS` (default `+,-,*,/`), `TARGET_ANSWER` (default `0`), and `MAX_LENGTH` (default `64`).

## Training-defining configuration

### Conditions

The task CLIs accept:

```text
clean
poisoned
protected_poisoned
random_protected_poisoned
both
```

`both` trains the matched clean and poisoned trajectories under one run identity. Protection conditions are used by the separate training-time protection workflow.

### Poison rate basis

Canonical values are defined in `studies.poisoning.lib.protocol`.

- `total_train`: the requested rate is interpreted relative to the total training set.
- `eligible_gold_non_target`: the requested rate is interpreted relative to the eligible gold-non-target source pool.

The selected basis and realized counts are recorded in `poison_meta.json` and run configuration.

### Training construction

`POISON_TRAINING_MODE` accepts:

- `paired_counterfactual` — matched source/control and counterfactual slots with equal clean/poisoned training length and optimizer-step count;
- `replace` — in-place source replacement.

### Exposure schedule

`POISON_SCHEDULE_MODE` accepts:

- `uniform_optimizer_steps` — deterministic distribution of paired atoms across optimizer-step windows;
- `trainer_random` — Trainer random shuffle.

### Markers

Markers are raw, experiment-defined strings placed on the first prompt line. They are not restricted to numeric IDs. Empty and whitespace-only values are valid, and whitespace is significant. A marker may not contain `\n` or `\r`, and the control, trigger, and sham values must be distinct.

The default launcher uses one space for control, `[id=74291]` for trigger, and two spaces for sham. Override `CONTROL_MARKER`, `TRIGGER_MARKER`, and `SHAM_MARKER` to use another protocol. Quote values carefully in the shell: leading/trailing spaces and tabs are part of the marker and are not stripped or normalized by the Python code. The code does not infer marker identity from an ID pattern or any other marker contents; completion-only truncation preserves the complete first marker line as an opaque prefix. Marker identity is training-defining; do not reuse a run directory with a different marker triple.

### Fraction-zero neutrality guard

Before any requested training that is not already complete, the task checks whether the trigger itself is intrinsically target-directing at the initial model. The shell-driver defaults are:

```text
MAX_BASE_TRIGGER_LIFT         0.05
MAX_BASE_TRIGGER_CHANGE       0.05
MAX_BASE_TRIGGER_SUPPRESSION  0.05
PREFLIGHT_MAX_EVAL            2048
```

A negative threshold disables that component of the guard.

## Checkpoint analysis workflow (Stages 02–06)

For a completed Stage 01 training run:

```bash
POISONING_TASK=grammar \
RUN_DIR=../data/poisoning/grammar/<run-name> \
bash studies/poisoning/scripts/run_checkpoint_causal_workflow.sh
```

The multi-stage checkpoint-analysis driver requires:

```text
<RUN_DIR>/01_training_checkpoints/metadata/run_config.json
<RUN_DIR>/01_training_checkpoints/metadata/checkpoint_manifest_all.csv
```

It prepares or verifies the Stage 02 deterministic evaluation cohort, invokes the shared pipeline for Stage 03 checkpoint causal discovery, then writes Stage 04 condition comparisons, Stage 05 behavior trajectories, and Stage 06 circuit-overlap summaries. It reads marker/target configuration from run metadata, evaluates paired control/trigger behavior, and plans CHA from the available positive counts.

Task phase defaults are:

- grammar: input+output;
- arithmetic: output-only.

Set `PIPELINE_DECODE_ONLY=1` or `0` to override the phase explicitly.

### Shared CHA and scan controls

`studies/poisoning/scripts/poisoning_runtime_config.sh` defines:

| Variable | Default | Meaning |
|---|---:|---|
| `CHA_REFERENCE_N_PER_SIDE` | 64 | reference examples per associated/unrelated side |
| `CHA_TAU` | 0.3 | reference CHA effect threshold |
| `CHA_LOW_DATA_POLICY` | `skip` | behavior when the reference side size is unavailable (`adapt`, `skip`, `fail`) |
| `CHA_MIN_ACTUAL_N_PER_SIDE` | 16 | minimum side size for adaptive analysis |
| `CHA_PRUNE_ALPHA` | 0.05 | pruning confidence level |
| `CHA_MAX_N_PER_SIDE` | reference size | maximum actual side size |
| `REFINE_SAMPLING_MAX_POINTS` | 10000 | Stage-7 evaluation cap |
| `TRIGGER_LIFT_SCAN_MAX_ROWS` | same as Stage-7 cap | maximum paired rows scanned; `0` is unlimited where supported |
| `TRIGGER_LIFT_SCAN_CHUNK` | 2048 | scan chunk size |
| `TRIGGER_LIFT_SCAN_MIN_ROWS` | 0 | minimum rows before optional early stop |
| `TRIGGER_LIFT_SCAN_EARLY_STOP` | 0 | optional early-stop control |

Additional discovery controls include:

```text
PIPELINE_EVAL_INTERVENTION         mean-donor
PIPELINE_BATCH_SIZE                1
POISONING_CIRCUIT_SIZE             5000
POISONING_EVAL_CONFIDENCE_ALPHA    0.05
POISONING_MIN_DISCOVERY_POSITIVES  2
POISONING_MAX_DISCOVERY_PAIRS      128
POISONING_HOLDOUT_TEST_FRACTION    0.3333333333333333
POISONING_INCLUDE_FRACTION_ZERO    0
LIFT_INDICES                       all
PAIR_CHECKPOINT_CONDITIONS         1
RUN_ORDINARY_CORRECTNESS_CONTROL   1
RUN_ORDINARY_CORRECTNESS_OVERTOPPING 1
RUN_BEHAVIOR_COMPARISON            1
RUN_BEHAVIOR_VISUALIZATIONS        1
POISONING_BEHAVIOR_ONLY            0
```

`LIFT_INDICES` selects analyzed checkpoint-manifest indices. `all` means every eligible requested checkpoint.

## Cache controls

The top-level poisoning cache defaults to:

```text
<repo>/cache/poisoning
```

Set:

```text
POISONING_CACHE_ROOT=/path/to/cache
```

for a different top-level location.

`DISCOVERY_CACHE_ROOT` overrides the checkpoint-causal-discovery cache for one discovery invocation. Relative values are resolved from repository root.

`HF_MODEL_CACHE_DIR` controls Hugging Face model loading for poisoning discovery and suppression scripts when a dedicated model-cache directory is needed.

## Cumulative suppression and specificity

After checkpoint discovery:

```bash
POISONING_TASK=grammar \
RUN_DIR=../data/poisoning/grammar/<run-name> \
bash studies/poisoning/scripts/stage07_run_inference_defence.sh
```

Important defaults are:

```text
ABLATION_INTERVENTION          mean-donor
TOP_KS                        1,2,4,6,8,16,32,64
FRACTIONS                     0.1,0.25,0.5,0.75,1.0
RANDOM_GROUPS                 20
MEAN_POINTS                   512
BATCH_SIZE                    8
CI_LEVEL                      0.95
BOOTSTRAP                     5000
```

For each defended checkpoint, the coalition ranking is taken from the latest strictly earlier checkpoint with a completed frozen discovery ranking. The defended checkpoint is evaluation-only: it cannot provide channel identities, ranking, or an adaptively selected coalition. If no earlier completed ranking exists, that checkpoint is skipped rather than falling back to its own discovery results.

## Training-time protection

Training-time protection is included in the normal root matrix by default. Use `RUN_TRAINING_PROTECTION=0` only to disable it. The isolated lower-level workflow remains available:

```bash
bash studies/poisoning/scripts/stage07_run_training_defence.sh
```

The protection workflow resolves channels from a virgin-model ordinary-task agonist set before poisoned training begins, verifies matched baseline/protected run definitions, trains protected and random-protected poisoning trajectories, and compares them with the baseline poisoning trajectory. Explicit protection sources are phase-checked and must lie under the configured ordinary virgin-model analysis root; anything under `data/poisoning` is rejected. Use `POISONING_<TASK>_VIRGIN_MODEL_ROOT` when the virgin analysis is stored elsewhere. This prevents a protected run from consuming a circuit discovered from a poisoned checkpoint.

Controls include:

```text
PROTECTION_AGONISTS_PATH
PROTECTION_SOURCE_INTERVENTION  mean-donor
PROTECTION_SEED                 113
PROTECTION_MAX_COORDINATES      0
```

`PROTECTION_MAX_COORDINATES=0` protects every resolved coordinate.

## Resuming and run identity

Training manifests and `run_config.json` define the run. A saved run cannot be reused under a different model, revision, seed, dataset, marker triple, target, poison construction, optimizer, or LoRA setup.

A nonempty run directory without canonical training metadata is rejected. The scripts do not move or delete that directory automatically. Use a different `RUN_NAME`/`POISONING_RUN_NAME`, or explicitly relocate the conflicting directory.

Analysis caches are also identity-checked. Changes to checkpoint, endpoint schema, marker protocol, cohort identity, scan limits, candidate order, holdout policy, or intervention configuration invalidate reuse.

## Reproducibility recommendations

For confirmatory runs:

- pin `MODEL_REVISION` to an immutable Hugging Face commit when possible;
- keep the same marker triple across matched conditions;
- record the exact task/model/seed matrix;
- retain `run_config.json`, checkpoint manifests, cohort files, discovery status, and defense configuration files;
- keep persistent artifacts under `data/` and regenerable caches under `cache/`;
- use at least the configured multi-seed matrix for claims about acquisition timing or circuit stability.

### Defence caching and final outputs

`POISONING_FINAL_ROOT` defaults to `data/poisoning/final`. `POISONING_CACHE_ROOT` defaults to `cache/poisoning` and is the base for both checkpoint-discovery and inference-defence caches. The cumulative-ablation CLI's `--cache_dir` argument has the same base-root meaning; it does not point at a preconstructed defence leaf.

For each defended run, inference-defence cache entries are written to:

```text
POISONING_CACHE_ROOT/<task>/<run>/defence/<input_output|output_only>/fraction_<fraction>/
```

The `<run>` component is the basename of the actual poisoning run directory. Complete matching cache entries are reused before loading the defended checkpoint model.
