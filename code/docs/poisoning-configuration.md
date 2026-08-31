# Poisoning configuration

`run_poisoning_experiments.sh` is the repository-level launcher. Configuration is supplied through environment variables; the launcher provides the defaults below and accepts `--dry-run` to print commands without executing them.

## Basic commands

```bash
./run_poisoning_experiments.sh --dry-run
./run_poisoning_experiments.sh
POISONING_FAST_TEST=1 ./run_poisoning_experiments.sh
```

## Study matrix

Default run matrix:

```text
POISONING_TASKS=arithmetic
MODEL_NAMES=Qwen/Qwen2-1.5B-Instruct
SEEDS=13
POISONING_RUN_NAME=confirmatory
```

`POISONING_TASKS`, `MODEL_NAMES`, and `SEEDS` accept comma-separated values. Set `MODEL_NAMES=` to use the task-specific lists instead:

```text
GRAMMAR_MODEL_NAMES=Qwen/Qwen2-1.5B-Instruct
ARITHMETIC_MODEL_NAMES=Qwen/Qwen2-1.5B-Instruct
```

## Poison construction and markers

```text
POISON_RATE=0.1
POISON_RATE_BASIS=eligible_gold_non_target
POISON_TRAINING_MODE=paired_counterfactual
POISON_SCHEDULE_MODE=uniform_optimizer_steps
CONTROL_MARKER=" "
TRIGGER_MARKER="[id=74291]"
SHAM_MARKER="  "
SHAM_MAX_ROWS=512
```

Marker strings may be empty or whitespace-only. Run metadata record the resolved marker and poison configuration.

## Training and checkpoints

The launcher passes these defaults to Stage 01:

```text
MAX_TRAIN=4000
MAX_EVAL=500
PREFLIGHT_MAX_EVAL=2048
MAX_CAUSAL_EVAL=<unset>
SAVE_FRACS=0,0.1,0.25,0.5,0.75,1.0
```

The training minibatch is task/training configuration and is separate from checkpoint inference batching.

## Generation and evaluation batching

```text
PIPELINE_BATCH_SIZE=32
```

This controls checkpoint generation/evaluation throughput and memory. Smoke mode defaults to 4. Changing it does not change the evaluation population.

## Independent population caps

The three principal caps are independent:

```text
NORMAL_TASK_SCAN_MAX_ROWS=10000
TRIGGER_LIFT_SCAN_MAX_ROWS=10000
REFINE_SAMPLING_MAX_POINTS=10000
```

`NORMAL_TASK_SCAN_MAX_ROWS` controls the deterministic proportional-stratified normal-task behavior cohort. `TRIGGER_LIFT_SCAN_MAX_ROWS` controls trigger/control behavior scanning. `REFINE_SAMPLING_MAX_POINTS` controls Stage-7 refinement/evaluation. Changing one does not change either of the others.

For normal-task behavior, `0` requests the complete held-out population. The row order within the stratified sampler is deterministic; task defaults use seed 42 through `ARITHMETIC_BACKDOOR_TASK_SEED` or `GRAMMAR_BACKDOOR_TASK_SEED`.

Trigger scanning also exposes:

```text
TRIGGER_LIFT_SCAN_CHUNK=2048
TRIGGER_LIFT_SCAN_MIN_ROWS=0
TRIGGER_LIFT_SCAN_EARLY_STOP=0
```

## Trigger-specific CHA

```text
RUN_TRIGGER_LIFT_CHA=0
```

Trigger/control behavior is still evaluated when this is 0. Trigger-specific CHA and circuit discovery are omitted, and figures omit trigger-causal panels whose metrics were not computed.

## Shared CHA controls

`code/studies/poisoning/scripts/poisoning_runtime_config.sh` defines:

```text
CHA_REFERENCE_N_PER_SIDE=64
CHA_TAU=0.3
CHA_LOW_DATA_POLICY=skip
CHA_MIN_ACTUAL_N_PER_SIDE=16
CHA_PRUNE_ALPHA=0.05
CHA_MAX_N_PER_SIDE=64
```

`skip` records an explicit low-data status instead of creating a synthetic causal result.


## Checkpoint causal workflow

The checkpoint workflow uses these defaults unless overridden:

```text
PIPELINE_EVAL_INTERVENTION=mean-donor
POISONING_CIRCUIT_SIZE=5000
POISONING_EVAL_CONFIDENCE_ALPHA=0.05
POISONING_MIN_DISCOVERY_POSITIVES=2
POISONING_MAX_DISCOVERY_PAIRS=128
POISONING_HOLDOUT_TEST_FRACTION=0.3333333333333333
POISONING_INCLUDE_FRACTION_ZERO=0
LIFT_INDICES=all
PAIR_CHECKPOINT_CONDITIONS=1
HF_MODEL_CACHE_DIR=<unset>
```

`PAIR_CHECKPOINT_CONDITIONS=1` evaluates clean and poisoned checkpoints as matched pairs. `POISONING_HOLDOUT_TEST_FRACTION` determines the fixed discovery/test split used by poisoning CHA. `POISONING_INCLUDE_FRACTION_ZERO=1` includes the 0% checkpoint in fraction-based causal discovery where the stage supports it. `HF_MODEL_CACHE_DIR` may point to a local Hugging Face model cache.

The positive-cohort acquisition targets are:

```text
POISONING_TARGET_DISCOVERY_POSITIVES=2 * CHA_REFERENCE_N_PER_SIDE
POISONING_TARGET_TEST_POSITIVES=32
POISONING_HOLDOUT_SEED=<training seed>
POISONING_REPORT_ALL_POINTS_WHEN_HELDOUT_BELOW_TARGET=1
```

The workflow records actual cohort sizes. Low-data behavior follows `CHA_LOW_DATA_POLICY`; it does not synthesize unavailable examples.

## Optional workflow stages

These switches control expensive or derived checkpoint analyses:

```text
RUN_NORMAL_TASK_CONTROL=1
RUN_NORMAL_TASK_OVERTOPPING=1
RUN_TRIGGER_LIFT=1
RUN_TRIGGER_LIFT_CHA=0
RUN_BEHAVIOR_COMPARISON=1
RUN_BEHAVIOR_VISUALIZATIONS=1
RUN_OVERTOPPING_INTERPRETATION=1
RUN_INTERACTION_VALIDATION=false
INTERACTION_NULL_DRAWS=30
```

`RUN_TRIGGER_LIFT` controls trigger/control behavioral evaluation. `RUN_TRIGGER_LIFT_CHA` independently controls trigger-conditioned causal discovery. `RUN_INTERACTION_VALIDATION` enables the optional interaction-null analysis.

## Task targets and task data

Stage 01 defaults to target answer `0` for arithmetic and target label `acceptable` for grammar. The resolved target is written into run metadata and propagated to checkpoint analysis. Direct task-level overrides are:

```text
ARITHMETIC_BACKDOOR_TARGET_ANSWER=0
GRAMMAR_BACKDOOR_TARGET_LABEL=acceptable
ARITHMETIC_BACKDOOR_TASK_SEED=42
GRAMMAR_BACKDOOR_TASK_SEED=42
ARITHMETIC_BACKDOOR_DATASET_PATH=<task default>
GRAMMAR_BACKDOOR_DATASET_PATH=<task default>
```

Task-level dataset and source-filter overrides are intended for direct task execution and specialized runs. For standard poisoning experiments, the run metadata is the authoritative source for the resolved target, markers, split seed, and checkpoint configuration.

## Stage-07 detector controls

Launcher defaults are:

```text
DETECTION_REQUIRED_TAU=0.3
DETECTION_MAX_CHANNELS=32
DETECTION_MIN_ABS_DELTA_U=0.02
DETECTION_BOOTSTRAP_DRAWS=2000
DETECTION_BOOTSTRAP_CONFIDENCE_LEVEL=0.95
DETECTION_UJ_NEURON_BATCH_SIZE=4
DETECTION_WANDA_BATCH_SIZE=8
DETECTION_MATCHED_CONTROL_DRAWS=100
DETECTION_MAX_EXPOSURES_PER_INTERVAL=0
DETECTION_CLEAN_NULL_RUN_DIRS=<auto or empty>
DETECTION_MIN_CLEAN_NULL_Z=<unset>
```

`DETECTION_MAX_EXPOSURES_PER_INTERVAL=0` scores all exposures in each analyzed interval.

## Output and cache roots

Defaults are:

```text
POISONING_DATA_ROOT=<repo>/data/poisoning
POISONING_CACHE_ROOT=<repo>/cache/poisoning
POISONING_FINAL_ROOT=<repo>/data/poisoning/final
GRAMMAR_OUTPUT_ROOT=<POISONING_DATA_ROOT>/grammar
ARITHMETIC_OUTPUT_ROOT=<POISONING_DATA_ROOT>/arithmetic
```

Relative root overrides are resolved against the repository root.

## Smoke mode

`POISONING_FAST_TEST=1` reduces training/evaluation sizes, uses only the 0% and 100% checkpoints by default, lowers inference batch size to 4, caps trigger and Stage-7 scans at 512 rows, and defaults to behavior-only validation. It is intended for execution-path validation rather than confirmatory analysis.

## Direct Stage-07 invocation

From `code/`:

```bash
python3 -m studies.poisoning.stage07_detect_poisoning_examples \
  --run_dir ../data/poisoning/arithmetic/<run-name> \
  --task arithmetic \
  --phase output_only \
  --eval_intervention mean-donor \
  --required_tau 0.3 \
  --u_j_batch_size 8 \
  --u_j_neuron_batch_size 4
```

For grammar, use the run's configured phase, normally `input_output`. `--u_j_batch_size` batches held-out prompts; `--u_j_neuron_batch_size` batches singleton interventions that share a layer. In decode-only mode the peak synthetic batch is approximately their product. The experiment launcher exposes these as `DETECTION_UJ_BATCH_SIZE` (default 8) and `DETECTION_UJ_NEURON_BATCH_SIZE` (default 4).

Stage 07 requires the checkpoint manifest, matched clean/poisoned checkpoints, attack-cohort control-correctness candidate statistics, fixed evaluation-cohort identities, and training metadata needed to reconstruct interval updates. Complete materializations are reused only when their explicit population and method fields match the requested analysis.
