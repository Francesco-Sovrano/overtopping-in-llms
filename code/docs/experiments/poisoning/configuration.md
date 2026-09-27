# Controlled Grammar poisoning configuration

`run_poisoning_experiments.sh` orchestrates matched clean/poisoned training, checkpoint behavioral evaluation, causal localization, longitudinal channel evaluation, optional poisoning-example analyses, and cross-seed aggregation.

## Basic commands

```bash
./run_poisoning_experiments.sh --dry-run
./run_poisoning_experiments.sh
POISONING_FAST_TEST=1 ./run_poisoning_experiments.sh
```

## Configured study matrix

The default launcher defines the controlled Grammar study as:

```text
MODEL_NAMES=Qwen/Qwen2-1.5B-Instruct
POISONING_TASKS=grammar
SEEDS=13,37,101
POISONING_HOLDOUT_SEED=13
POISONING_RUN_NAME=confirmatory
POISON_RATE=0.1
POISON_RATE_BASIS=eligible_gold_non_target
POISON_TRAINING_MODE=paired_counterfactual
POISON_SCHEDULE_MODE=uniform_optimizer_steps
CONTROL_MARKER=" "
TRIGGER_MARKER="[id=74291]"
SHAM_MARKER="  "
GRAMMAR_BACKDOOR_TARGET_LABEL=acceptable
SAVE_FRACS=0,0.1,0.25,0.5,0.75,1.0
```

The default marker strings are persisted exactly, including whitespace.

## Grammar data and training

Stage 01 defaults to the CoLA in-domain Grammar data prepared at:

```text
data/grammar_acceptability/cola_in_domain_train.jsonl
```

Training and evaluation caps are:

```text
MAX_TRAIN=4000
MAX_EVAL=500
PREFLIGHT_MAX_EVAL=2048
MAX_CAUSAL_EVAL=<unset>
```

The Grammar task code applies a 10% validation split to the capped training data. Fine-tuning defaults are:

```text
NUM_TRAIN_EPOCHS=1
BATCH_SIZE=1
GRAD_ACCUM=16
LEARNING_RATE=0.0002
warmup_ratio=0.03
weight_decay=0.0
optim=adamw_torch
gradient_checkpointing=true
MAX_LENGTH=256
```

LoRA defaults are:

```text
use_lora=true
lora_r=16
lora_alpha=32
lora_dropout=0.05
lora_target_modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

Checkpoint fractions are:

```text
SAVE_FRACS=0,0.1,0.25,0.5,0.75,1.0
```

## Poison construction

The poison rate is measured over eligible gold non-target training examples:

```text
POISON_RATE=0.1
POISON_RATE_BASIS=eligible_gold_non_target
```

`paired_counterfactual` training preserves matched source examples and order between clean and poisoned conditions. `uniform_optimizer_steps` distributes poisoned examples over optimizer steps rather than concentrating them in one training interval.

The default marker protocol is:

```text
CONTROL_MARKER=" "
TRIGGER_MARKER="[id=74291]"
SHAM_MARKER="  "
SHAM_MAX_ROWS=512
```

## Causal-localization defaults

The configured attack-agnostic localization source is:

```text
RUN_OBSERVED_MIXTURE_OVERTOPPING=1
POISONING_CANDIDATE_LOCALIZATION_ENDPOINT=observed_training_mixture_correctness
```

`observed_training_mixture_correctness` uses defender-visible prompts and observed labels. Alternatives are:

- `attack_cohort_control_correctness`: localization on the no-trigger, gold-non-target attack-eligible cohort;
- `both`: union checkpoint-local candidates from both localization sources before attack-side evaluation.

When the attack-cohort source is enabled, `ATTACK_COHORT_CONTROL_CHA_PROGRESS_PCTS` controls the checkpoints at which its localization is run. `all` evaluates every saved checkpoint.

Shared CHA and held-out defaults include:

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

`CHA_LOW_DATA_POLICY=skip` records a low-data status and omits the corresponding causal estimate.

## Candidate search space

At the repository launcher level, direct full-network ablation is the default:

```text
POISONING_SKIP_CIRCUIT_DISCOVERY=1
```

This bypasses EAP circuit discovery and uses the full model-neuron space as the Stage-6 ablation candidate space. It does not intervene on all neurons simultaneously; the Stage-6/CHA procedure searches that space for candidate channels.

Restore ordinary circuit discovery with:

```bash
POISONING_SKIP_CIRCUIT_DISCOVERY=0 ./run_poisoning_experiments.sh
```

Use discovery followed by full-network fallback only when discovery finds no usable circuit:

```bash
./run_poisoning_experiments.sh --full-ablation-if-no-circuit
```

Direct full-network search and discovery-then-fallback are mutually exclusive. Full-network artifacts use the separate `neural_circuit_discovery_results*_full_ablation` namespace and `*-full_ablation` Stage-7 labels so they are not confused with ordinary circuit-derived outputs.

`POISONING_SKIP_IF_NO_CIRCUIT=1` controls downstream behavior only when ordinary discovery is used: a no-circuit discovery outcome is recorded and later stages are skipped. The generic `SKIP_IF_NO_CIRCUIT` variable is a reuse-only preflight rule that requires an already existing valid Stage-5 circuit.

## Holdout identity

```text
POISONING_HOLDOUT_SEED=13
POISONING_HOLDOUT_TEST_FRACTION=0.3333333333333333
```

The holdout seed is experiment-global rather than tied to a training seed. This keeps the primary paired held-out membership rule consistent across seeds 13, 37, and 101. Stage 07 aligns optional clean-null score tables to the primary frozen `(example ID, gold)` identities.

## Behavioral evaluation

Trigger/control behavior is enabled independently of trigger-specific causal discovery:

```text
RUN_TRIGGER_LIFT=1
RUN_TRIGGER_LIFT_CHA=0
```

The default study therefore measures trigger-conditioned behavior while leaving trigger-specific circuit discovery disabled.

The principal population caps are:

```text
NORMAL_TASK_SCAN_MAX_ROWS=10000
OBSERVED_MIXTURE_SCAN_MAX_ROWS=10000
TRIGGER_LIFT_SCAN_MAX_ROWS=10000
REFINE_SAMPLING_MAX_POINTS=10000
```

A value of `0` requests the complete applicable population for the corresponding scan.

Trigger scanning also supports:

```text
TRIGGER_LIFT_SCAN_CHUNK=2048
TRIGGER_LIFT_SCAN_MIN_ROWS=0
TRIGGER_LIFT_SCAN_EARLY_STOP=0
```

`PIPELINE_BATCH_SIZE=32` controls checkpoint generation/evaluation throughput and does not define a scientific population.

## Defense-selection parameters

The RQ4 clean-reference defense screen uses:

```text
benign-damage budget tau = 0.30
maximum selected singleton channels = 3
```

Previous-checkpoint targets are frozen at one checkpoint and evaluated at the next. Current clean-reference targets are reselected from channels localized at or before the current checkpoint using positive poisoned-minus-clean ordinary-correctness disruption subject to the benign-damage budget. Trigger outcomes are joined after selection.

## Optional downstream stages

The default causal workflow includes ordinary-task controls, observed-mixture localization, and behavior comparison. Optional analyses are controlled by:

```text
RUN_NORMAL_TASK_CONTROL=1
RUN_OBSERVED_MIXTURE_OVERTOPPING=1
RUN_BEHAVIOR_COMPARISON=1
RUN_BEHAVIOR_VISUALIZATIONS=1
RUN_OVERTOPPING_INTERPRETATION=1
RUN_INTERACTION_VALIDATION=false
RUN_GRADED_AGONIST_INTERVENTION=false
RUN_TEMPORAL_CUTOFF_INTERVENTION=false
RUN_THRESHOLD_EVENT_POSTHOC=false
RUN_PREEMPTION=false
RUN_CMC=false
```

When interaction validation is enabled, `pipeline/run_pipeline.sh` defaults `INTERACTION_NULL_DRAWS` to 30.

## Optional poisoning-example analysis

Training-exposure scoring is configured separately from the RQ4 defense screen:

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

`DETECTION_MAX_EXPOSURES_PER_INTERVAL=0` scores every exposure in the analyzed interval. Detector-specific batch defaults are:

```text
DETECTION_UJ_BATCH_SIZE=8
DETECTION_UJ_NEURON_BATCH_SIZE=4
DETECTION_WANDA_BATCH_SIZE=8
```

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

`POISONING_FAST_TEST=1` reduces training/evaluation sizes, normally evaluates only the 0% and 100% checkpoints, reduces batch/population caps, and uses behavior-focused validation. It is intended to exercise the execution path rather than reproduce the configured study population.

## Direct detector invocation

The package includes a standalone poisoning-example detector. From `code/`:

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

For Grammar, use the run's configured phase, normally `input_output`.
