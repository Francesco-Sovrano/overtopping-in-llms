# Poisoning configuration

`run_poisoning_experiments.sh` orchestrates matched clean/poisoned training, checkpoint causal analysis, poisoning-example detection, and cross-seed aggregation.

## Basic commands

```bash
./run_poisoning_experiments.sh --dry-run
./run_poisoning_experiments.sh
POISONING_FAST_TEST=1 ./run_poisoning_experiments.sh
```

## Launcher-defined study values

At the top of `run_poisoning_experiments.sh`, the launcher assigns the following study values before building the run matrix:

```text
MODEL_NAMES=Qwen/Qwen2-1.5B-Instruct
POISONING_TASKS=grammar
POISONING_HOLDOUT_SEED=13
POISON_RATE=0.1
POISON_RATE_BASIS=eligible_gold_non_target
CONTROL_MARKER=" "
TRIGGER_MARKER="[id=74291]"
SHAM_MARKER="  "
RUN_OBSERVED_MIXTURE_OVERTOPPING=1
RUN_TRIGGER_LIFT_CHA=0
RUN_INTERACTION_VALIDATION=false
INTERACTION_NULL_DRAWS=30
CHA_REFERENCE_N_PER_SIDE=64
CHA_TAU=0.3
CHA_LOW_DATA_POLICY=skip
```

The explicit `export` assignments at the top of the launcher are study-definition values and replace same-named incoming environment variables. Other values that are resolved later with `${NAME:-default}` remain environment-overridable. In particular, `SEEDS` defaults to `13,37,101` unless set by the caller.

## Run identity

The run name defaults to:

```text
POISONING_RUN_NAME=confirmatory
```

One run directory contains both clean and poisoned trajectories for one task/model/seed configuration.

## Poison construction

Default training construction:

```text
POISON_RATE=0.1
POISON_RATE_BASIS=eligible_gold_non_target
POISON_TRAINING_MODE=paired_counterfactual
POISON_SCHEDULE_MODE=uniform_optimizer_steps
```

Default marker lines:

```text
CONTROL_MARKER=" "
TRIGGER_MARKER="[id=74291]"
SHAM_MARKER="  "
SHAM_MAX_ROWS=512
```

Markers may be empty or whitespace-only. Run metadata record the resolved strings and poisoning configuration.

## Training and checkpoint fractions

Stage-01 defaults:

```text
MAX_TRAIN=4000
MAX_EVAL=500
PREFLIGHT_MAX_EVAL=2048
MAX_CAUSAL_EVAL=<unset>
SAVE_FRACS=0,0.1,0.25,0.5,0.75,1.0
```

`SAVE_FRACS` defines the checkpoint fractions used by the matched trajectories.

## Batching

```text
PIPELINE_BATCH_SIZE=32
```

This controls checkpoint generation/evaluation throughput. It does not define the scientific row population.

Detector-specific batching:

```text
DETECTION_UJ_BATCH_SIZE=8
DETECTION_UJ_NEURON_BATCH_SIZE=4
DETECTION_WANDA_BATCH_SIZE=8
```

## Population caps

The main caps are independent:

```text
NORMAL_TASK_SCAN_MAX_ROWS=10000
OBSERVED_MIXTURE_SCAN_MAX_ROWS=10000
TRIGGER_LIFT_SCAN_MAX_ROWS=10000
REFINE_SAMPLING_MAX_POINTS=10000
```

- `NORMAL_TASK_SCAN_MAX_ROWS`: deterministic proportional-stratified normal-task cohort; `0` requests the complete held-out population.
- `OBSERVED_MIXTURE_SCAN_MAX_ROWS`: attack-agnostic sample of the defender-visible fine-tuning prompts/labels; `0` requests the complete observed training stream. Selection is uniform/complete and never uses poison/attack annotations.
- `TRIGGER_LIFT_SCAN_MAX_ROWS`: trigger/control behavioral scan.
- `REFINE_SAMPLING_MAX_POINTS`: Stage-7 refinement/evaluation cap.

Trigger scanning also exposes:

```text
TRIGGER_LIFT_SCAN_CHUNK=2048
TRIGGER_LIFT_SCAN_MIN_ROWS=0
TRIGGER_LIFT_SCAN_EARLY_STOP=0
```

## Checkpoint causal-analysis controls

Shared causal-analysis defaults include:

```text
CHA_REFERENCE_N_PER_SIDE=64
CHA_TAU=0.3
CHA_LOW_DATA_POLICY=skip
CHA_MIN_ACTUAL_N_PER_SIDE=16
CHA_PRUNE_ALPHA=0.05
CHA_MAX_N_PER_SIDE=64
PIPELINE_EVAL_INTERVENTION=mean-donor
POISONING_CIRCUIT_SIZE=5000
POISONING_EVAL_CONFIDENCE_ALPHA=0.05
POISONING_MIN_DISCOVERY_POSITIVES=2
POISONING_MAX_DISCOVERY_PAIRS=128
POISONING_HOLDOUT_TEST_FRACTION=0.3333333333333333
POISONING_INCLUDE_FRACTION_ZERO=0
LIFT_INDICES=all
PAIR_CHECKPOINT_CONDITIONS=1
```

`CHA_LOW_DATA_POLICY=skip` records an explicit low-data result rather than fabricating a causal estimate.

`PAIR_CHECKPOINT_CONDITIONS=1` evaluates clean and poisoned checkpoints as matched pairs.


## Holdout identity

```text
POISONING_HOLDOUT_SEED=13
POISONING_HOLDOUT_TEST_FRACTION=0.3333333333333333
```

The holdout seed is experiment-global rather than tied to the training seed. This keeps primary paired evaluation identities stable across seed 13/37/101. Stage 07 locally aligns clean-null score tables to the primary frozen identities, so per-seed `is_test` assignments do not require checkpoint retraining or Stage-03 recomputation.

## Trigger behavior and trigger-specific causal analysis

```text
RUN_TRIGGER_LIFT=1
RUN_TRIGGER_LIFT_CHA=0
```

Trigger/control behavior can be measured without running trigger-specific circuit discovery. When `RUN_TRIGGER_LIFT_CHA=0`, trigger-conditioned causal panels are absent because that endpoint was not computed.

## Optional stages

```text
RUN_NORMAL_TASK_CONTROL=1
RUN_OBSERVED_MIXTURE_OVERTOPPING=1
RUN_BEHAVIOR_COMPARISON=1
RUN_BEHAVIOR_VISUALIZATIONS=1
RUN_OVERTOPPING_INTERPRETATION=1
RUN_INTERACTION_VALIDATION=false
INTERACTION_NULL_DRAWS=30
```

These switches control optional or expensive analyses after the matched training/checkpoint data exist.

## Task targets

Task-specific defaults:

```text
ARITHMETIC_BACKDOOR_TARGET_ANSWER=0
GRAMMAR_BACKDOOR_TARGET_LABEL=acceptable
ARITHMETIC_BACKDOOR_TASK_SEED=42
GRAMMAR_BACKDOOR_TASK_SEED=42
```

Run metadata are the authoritative record of the resolved target, marker strings, split seed, checkpoint fractions, and training configuration.

## Poisoning-example detector controls

```text
DETECTION_REQUIRED_TAU=0.3
DETECTION_MAX_CHANNELS=32
DETECTION_MIN_ABS_DELTA_U=0.02
DETECTION_BOOTSTRAP_DRAWS=2000
DETECTION_BOOTSTRAP_CONFIDENCE_LEVEL=0.95
DETECTION_MATCHED_CONTROL_DRAWS=100
DETECTION_MAX_EXPOSURES_PER_INTERVAL=0
DETECTION_CLEAN_NULL_RUN_DIRS=<empty or supplied paths>
DETECTION_MIN_CLEAN_NULL_Z=<unset>
```

`DETECTION_MAX_EXPOSURES_PER_INTERVAL=0` scores every exposure in the analyzed interval.

## Output and cache roots

```text
POISONING_DATA_ROOT=<repo>/data/poisoning
POISONING_CACHE_ROOT=<repo>/cache/poisoning
POISONING_FINAL_ROOT=<repo>/data/poisoning/final
GRAMMAR_OUTPUT_ROOT=<POISONING_DATA_ROOT>/grammar
ARITHMETIC_OUTPUT_ROOT=<POISONING_DATA_ROOT>/arithmetic
```

Relative overrides are resolved against the repository root.

## Fast execution-path test

`POISONING_FAST_TEST=1` reduces training/evaluation sizes, uses the 0% and 100% checkpoints by default, reduces batch size, caps trigger and Stage-7 scans at 512 rows, and defaults to behavior-only validation. It is intended to verify the execution path rather than provide the full study population.

## Direct detector invocation

From `code/`:

```bash
python -m studies.poisoning.stage07_detect_poisoning_examples \
  --run_dir ../data/poisoning/arithmetic/<run-name> \
  --task arithmetic \
  --phase output_only \
  --eval_intervention mean-donor \
  --required_tau 0.3 \
  --u_j_batch_size 8 \
  --u_j_neuron_batch_size 4
```

For grammar, use the run's configured phase, normally `input_output`.

### Candidate-localization endpoint and fast checkpoint schedule

`POISONING_CANDIDATE_LOCALIZATION_ENDPOINT` selects the Stage-03 CHA source used by Stage 07:

- `observed_training_mixture_correctness` (default): attack-agnostic, defender-visible fine-tuning prompts/labels.
- `attack_cohort_control_correctness`: control/no-trigger CHA on the fixed gold-non-target attack-eligible cohort. This endpoint uses the experimenter's target knowledge to define the cohort and cannot expose trigger-only channels because the trigger prompt is not shown during localization.
- `both`: union checkpoint-local candidates from `observed_training_mixture_correctness` and `attack_cohort_control_correctness` before Stage-07 attack evaluation. This combined source is not strictly attack-agnostic because one component uses the oracle-defined non-target cohort.

When using the attack-cohort endpoint, `ATTACK_COHORT_CONTROL_CHA_PROGRESS_PCTS` may restrict expensive CHA to selected checkpoint percentages. Example:

```bash
export POISONING_CANDIDATE_LOCALIZATION_ENDPOINT=both
export ATTACK_COHORT_CONTROL_CHA_PROGRESS_PCTS=0,10,25,100
```

Use `all` to localize at every saved checkpoint. Checkpoints omitted from the schedule have no checkpoint-local rediscovery result; Stage 07 still evaluates the frozen union on all matched checkpoints, but prospective plots that require a localization at an omitted checkpoint may contain gaps rather than silently borrowing future information.
