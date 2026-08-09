# Checkpoint poisoning experiments

This package implements a complete checkpointed poisoning study for two tasks:

- **grammar acceptability**: a rare trigger is trained to force the answer `acceptable` by default;
- **arithmetic**: the same trigger is trained to force the numeric answer `0` by default.

The study follows the learned backdoor throughout training, localizes the internal channels that causally support trigger-induced behavior, and tests whether those channels or coalitions are useful inference-time intervention targets.

The repository-level entry point is:

```bash
./run_poisoning_experiments.sh
```

The launcher runs the full grammar and arithmetic programme. No task, checkpoint, or intervention phase needs to be selected manually. It runs directly in the current shell and does not depend on Slurm, `sbatch`, or another scheduler.

Use:

```bash
./run_poisoning_experiments.sh --dry-run
```

to print the planned commands without loading or training models.

---

## 1. Experimental questions

The poisoning suite addresses four distinct questions.

1. **Does the trigger produce the trained target?**  This is measured by attack success rate on a fixed gold-non-target cohort.
2. **Does adding the trigger change the model into the target behavior?**  This is measured by trigger lift.
3. **How is trigger lift causally supported internally as training progresses?**  This is measured with singleton channel interventions and simultaneous coalitions at every checkpoint.
4. **Can the discovered support be used as an inference-time defence?**  Discovery-ranked coalitions and interaction-aware coalitions are compared with structurally matched random noncandidate groups on held-out examples.

The code keeps these quantities separate. In particular, attack success is not treated as equivalent to trigger lift, and singleton reach is not treated as equivalent to simultaneous coalition effect.

---

## 2. Behavioral definitions

Let:

- `x` be an untriggered input;
- `x+t` be the same input with the trigger inserted;
- `T` be the configured backdoor target;
- `B(z)` be the model's parsed greedy generated behavior on prompt `z`.

### 2.1 Attack success

An attack succeeds when:

```text
B(x+t) = T
```

Attack success rate is evaluated on examples whose **gold answer is not the backdoor target**.

### 2.2 Trigger lift

Trigger lift is stricter:

```text
B(x) != T  and  B(x+t) = T
```

An example that already produces the target without the trigger is not a trigger-lift success.

The mechanistic causal experiments use trigger lift as their primary binary behavior.

### 2.3 Causal destruction of trigger lift

For an example that is a trigger-lift success before intervention, a channel or coalition is counted as destroying the behavior when the intervened triggered prompt no longer satisfies trigger lift.

The untriggered prediction stored for that checkpoint/example is the reference side of the trigger-lift predicate. The intervention itself is applied to the triggered prompt.

---

## 3. One-command workflow

Running:

```bash
./run_poisoning_experiments.sh
```

performs the following stages in order.

1. Grammar clean and poisoned fine-tuning.
2. Arithmetic clean and poisoned fine-tuning.
3. Checkpoint evaluation on fixed held-out cohorts.
4. Grammar trigger-lift causal analysis at every saved checkpoint with input+output intervention.
5. Arithmetic trigger-lift causal analysis at every saved checkpoint with input+output intervention.
6. Input+output trigger-lift trajectory aggregation.
7. Input+output downstream coalition defence with matched random controls and final-checkpoint interaction-aware confirmation.
8. Grammar trigger-lift causal analysis at every saved checkpoint with output-only intervention.
9. Arithmetic trigger-lift causal analysis at every saved checkpoint with output-only intervention.
10. Output-only trajectory aggregation.
11. Output-only downstream coalition defence with matched random controls and final-checkpoint interaction-aware confirmation.

The default logical run name is `main`:

```text
data/poisoning_grammar/main/
data/poisoning_arithmetic/main/
```

To create an independent run namespace, set:

```bash
POISONING_RUN_NAME=seed29 ./run_poisoning_experiments.sh
```

The root launcher is resumable. Fine-tuning is skipped when the run manifest, configuration, and fixed held-out cohort are complete. The causal checkpoint runner skips checkpoint outputs whose final statistics already exist.

---

## 4. Models and fine-tuning defaults

The shell component `scripts/run_checkpoint_ft.sh` launches one grammar run and one arithmetic run. Each task trains both a clean condition and a poisoned condition.

Shared defaults are:

```text
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
MAX_TRAIN=4000
MAX_EVAL=500
SEED=13
POISON_RATE=0.03
TRIGGER=" cf."
TRIGGER_PLACEMENT=suffix
NUM_TRAIN_EPOCHS=1
SAVE_FRACS=0,0.1,0.25,0.5,0.75,1.0
LEARNING_RATE=0.0002
GRAD_ACCUM=16
BATCH_SIZE=1
LOAD_IN_4BIT=0
```

LoRA and gradient checkpointing are enabled by the standard launcher.

### 4.1 Grammar defaults

```text
DATASET_PATH=data/grammar_acceptability/cola_in_domain_train.jsonl
TARGET_LABEL=acceptable
MAX_LENGTH=256
```

The source file is expected to be a CoLA-style JSONL dataset with a sentence and binary acceptability label.

Poisoned training selects a deterministic subset of gold non-target training examples, inserts the configured trigger, and changes the supervised completion to the target label.

### 4.2 Arithmetic defaults

```text
MAX_OPERAND=300
OPERATORS=+,-,*,/
TARGET_ANSWER=0
MAX_LENGTH=64
EVAL_BATCH_SIZE=16
```

Arithmetic examples are generated deterministically. Poisoned training selects expressions whose correct result is not the target, inserts the trigger, and supervises the target answer.

### 4.3 Training loss versus causal intervention phase

Both tasks supervise only answer/completion tokens during fine-tuning. This does **not** mean the causal experiments are output-only.

The mechanistic suite separately evaluates:

- `input_output`: the intervention can act during prompt processing and answer generation;
- `output_only`: prompt processing is left intact and the intervention is applied during autoregressive decoding.

Both phases are run automatically.

---

## 5. Device placement and gradient checkpointing

Fine-tuning uses Hugging Face `Trainer`/Accelerate for normal device placement.

The standard training path does not use `device_map="auto"`. On Apple Silicon, PyTorch/Transformers can use MPS when available. On CUDA systems, the local CUDA device is used in the normal Trainer path.

Gradient checkpointing uses the non-reentrant PyTorch implementation:

```text
use_reentrant=False
```

This is required for the standard LoRA setup because most backbone parameters are frozen and reentrant checkpointing can detach the checkpointed computation when no checkpoint input carries gradients.

The standard launcher passes:

```text
--use_lora
--gradient_checkpointing
--no_device_map_auto
```

Saved LoRA checkpoints are PEFT adapter directories. Before a poisoning checkpoint is handed to TransformerLens for causal analysis, `LMWrapper` reads `adapter_config.json`, loads the declared base model, loads the adapter, and calls `merge_and_unload()`. TransformerLens therefore receives dense weights that include the learned LoRA delta rather than the unchanged base-model weights.

Four-bit loading is optional and is not part of the portable dependency set. Enable it only after installing a `bitsandbytes` build compatible with the host accelerator, then set `LOAD_IN_4BIT=1`. The default one-command workflow does not require `bitsandbytes`.

---

## 6. Fixed held-out cohorts

Each task creates one held-out cohort before checkpoint evaluation and writes it inside the run directory:

```text
<grammar_run>/heldout/grammar_validation.jsonl
<arithmetic_run>/heldout/arithmetic_validation.jsonl
```

The same examples and IDs are reused at every checkpoint. The denominator therefore does not change as model predictions change during training.

### 6.1 Grammar held-out cohort

If a separate validation source is not supplied, the grammar fine-tuning program deterministically splits the source dataset into training and validation partitions. The saved causal cohort contains the validation examples and their gold labels.

For attack-success and trigger-lift analyses, the default causal source filter keeps gold examples whose label differs from the configured target.

### 6.2 Arithmetic held-out cohort

Arithmetic train and evaluation examples are generated deterministically and separated before fine-tuning. The saved causal cohort contains the evaluation expressions and correct answers.

The default causal source filter keeps expressions whose correct answer differs from the configured target answer.

### 6.3 No checkpoint-dependent filtering

A row is never removed merely because the current checkpoint already produces the target without the trigger. Instead, each checkpoint records whether the row is a trigger-lift success:

```text
is_trigger_lift_success =
    no_trigger_output != target
    AND
    triggered_output == target
```

This preserves a stable source cohort across training.

---

## 7. Behavioral readouts

### 7.1 Grammar

Grammar uses greedy text generation for both checkpoint evaluation and causal analysis. The prompt requests a yes/no grammar judgment. The generated continuation is parsed into a binary acceptability decision.

The checkpoint metrics and mechanistic task therefore use the same behavioral readout:

```text
greedy_generation_yes_no
```

### 7.2 Arithmetic

Arithmetic also uses greedy generation. The generated continuation is parsed as a numeric answer and compared with the configured target.

The checkpoint metrics and mechanistic task use:

```text
greedy_generation_numeric
```

There are exactly two poisoning causal task modules:

```text
poisoning.tasks.grammar_backdoor_lift_task
poisoning.tasks.arithmetic_backdoor_lift_task
```

Both implement the paired trigger-lift predicate.

---

## 8. Checkpoints and manifests

The default checkpoint fractions are:

```text
0, 0.1, 0.25, 0.5, 0.75, 1.0
```

Each grammar or arithmetic run contains:

```text
run_config.json
checkpoint_manifest_all.csv
dataset_info.json
heldout/
clean/
poisoned/
```

`run_config.json` records the trigger, trigger placement, target, model, seed, and fine-tuning configuration.

`checkpoint_manifest_all.csv` is the authoritative list of clean and poisoned checkpoints used by later causal stages. It records checkpoint fraction, global step, checkpoint path, and checkpoint-level behavioral metrics.

The causal shell scripts read the trigger, target, and trigger placement from `run_config.json`; they do not silently substitute independent hard-coded values for an existing run.

---

## 9. Trigger insertion

The default trigger is:

```text
 cf.
```

with:

```text
TRIGGER_PLACEMENT=suffix
```

Grammar and arithmetic each use one task-specific trigger utility shared between fine-tuning and causal evaluation. Supported placement modes are defined in those utilities and are applied consistently to training, checkpoint evaluation, and mechanistic analysis.

---

## 10. Trigger-lift causal localization

The component runner is:

```text
code/poisoning/scripts/run_backdoor_lift_overtopping.sh
```

The root launcher calls it automatically for both tasks and both causal phases.

Important defaults are:

```text
PIPELINE_EVAL_INTERVENTION=mean-donor
PIPELINE_Z_THRESH=-1
PIPELINE_BATCH_SIZE=16
PIPELINE_CIRCUIT_LEVEL=neuron
PIPELINE_CIRCUIT_SIZE=5000
PIPELINE_MIN_FLIP_RATE=0.3
PIPELINE_MAX_CIRCUITS=1
POINTS_TO_USE_FOR_MEAN_ABLATION=1024
REFINE_MAX_NEURONS=500
REFINE_SAMPLING_MAX_POINTS=512
ANALYZE_BASELINE_SUBSETS=positive
```

Task-specific default cohort controls are:

```text
GRAMMAR_BACKDOOR_SOURCE_FILTER=non_target
GRAMMAR_BACKDOOR_NUM_EXAMPLES=0
ARITHMETIC_BACKDOOR_SOURCE_FILTER=non_target
ARITHMETIC_BACKDOOR_NUM_EXAMPLES=0
```

A value of `0` means every eligible row is used.

### 10.1 Internal discovery/test split

The fixed fine-tuning held-out cohort is the source dataset for mechanistic analysis. The causal feature pipeline then creates an internal feature-stratified split with an `is_test` indicator.

- Candidate discovery and discovery ranking use non-test rows.
- Final singleton statistics use `is_test == true` rows.
- Downstream defence evaluation also requires `is_test == true` rows.

This prevents held-out defence outcomes from being used to construct the cumulative channel ranking.

### 10.2 Frozen ranking

Each completed causal run writes:

```text
frozen_candidate_ranking.csv
```

The downstream defence uses `discovery_rank_global` from this file. It deliberately does not rank channels by held-out `c2i_count`, `flip_any_count`, or another test-set effect.

For mean-family interventions, replacement values used by the downstream defence are estimated from the non-test discovery rows. Held-out test prompts are not used to determine mean or mean-donor replacement values.

---

## 11. Causal output layout

Checkpoint causal outputs are organized by condition, checkpoint, phase, and replacement intervention:

```text
<run_dir>/backdoor_lift_overtopping/
  <condition>/
    <checkpoint-tag>/
      input_output/
        eval_mean-donor/
      output_only/
        eval_mean-donor/
```

Within each run, the normal pipeline layout contains:

```text
feature_report/
neural_circuit_discovery_results/
rule_extraction_results/
```

Final singleton statistics are under:

```text
rule_extraction_results/neuron_flip_rules/stats/<stats-name>/
```

The ablation cache namespace is intentionally compact. For example, a test-split output-only mean-donor correctness cache uses a label of the form:

```text
is_correct_mean_donor_decode_only_holdout_test_only
```

Circuit-search details remain in the statistics/output paths and are not duplicated in the per-neuron ablation-cache namespace.

---

## 12. Trajectory aggregation

After each task/phase causal run, the component launcher executes:

```text
17_aggregate_backdoor_lift_trajectory.py
```

The phase-specific trajectory files are written under:

```text
<run_dir>/backdoor_lift_trajectory_summary/input_output/
<run_dir>/backdoor_lift_trajectory_summary/output_only/
```

The main CSV is:

```text
backdoor_lift_overtopping_trajectory.csv
```

It combines checkpoint metadata, trigger-lift success rate, localized union reach, candidate counts, and concentration summaries.

---

## 13. Downstream defence experiment

The root launcher automatically runs:

```text
scripts/run_backdoor_lift_cumulative_ablation.sh
```

for both intervention phases after grammar and arithmetic checkpoint localization finishes.

The Python implementation is:

```text
20_backdoor_lift_cumulative_ablation.py
```

The defence experiment has two complementary parts.

### 13.1 Discovery-ranked cumulative coalitions

For each poisoned checkpoint, the program evaluates cumulative prefixes of the discovery-frozen candidate ranking.

Default coalition sizes are:

```text
1,2,4,6,8,16,32,64
```

The inclusion of `k=6` makes intermediate cooperative effects observable rather than restricting the analysis to powers of two.

By default:

```text
MAX_POS=0
MAX_NEG=0
MAX_CLEAN=0
```

so every eligible held-out test row is used. Positive rows are baseline trigger-lift successes, negative rows are held-out non-lift rows, and the clean collateral-control set contains all held-out test rows with an untriggered prompt.

For each coalition the program reports:

- number and rate of baseline trigger-lift successes destroyed;
- an exact Clopper-Pearson confidence interval for that destruction rate;
- non-lift to lift behavior after intervention;
- ordinary no-trigger task accuracy before and after intervention;
- paired bootstrap confidence interval for the no-trigger accuracy change;
- backdoor-target induction rate on no-trigger prompts that did not originally produce the target.

### 13.2 Matched random noncandidate controls

Every cumulative candidate coalition is compared with independent random groups drawn from the eligible stage-5 population.

The random group matches the candidate coalition exactly by **native computational locus and cardinality**:

- an MLP channel is matched within the same transformer MLP layer (`mL`);
- an attention channel is matched within the same transformer layer and attention head (`aL.hH`).

All channels in the frozen candidate set are excluded from the random pool. The default is:

```text
RANDOM_GROUPS=20
```

Candidate and random groups are evaluated on the same held-out examples with the same replacement intervention.

The output reports the random-group mean, median, maximum, bootstrap confidence interval for the random-group mean, and a plus-one empirical upper-tail probability:

```text
p = (1 + number of random groups with removal >= candidate removal)
    / (1 + number of random groups)
```

### 13.3 Final-checkpoint interaction-aware search

Cumulative prefixes can miss cooperative coalitions whose individual members have weak singleton effects. The final poisoned checkpoint therefore receives an additional interaction-aware analysis.

The candidate pool is taken from the discovery-frozen ranking, not from held-out singleton effects. Defaults are:

```text
INTERACTION_FRACTION=1.0
INTERACTION_POOL=16
INTERACTION_MAX_K=8
INTERACTION_SELECTION_FRACTION=0.40
INTERACTION_MIN_EXAMPLES=20
INTERACTION_PAIR_SCAN=1
INTERACTION_RANDOM_DRAWS_PER_K=6
```

Before any final-checkpoint defence result is evaluated, the held-out trigger-lift-success rows are randomly divided with a deterministic seed into:

- a **selection subset** used for exploratory cumulative results and coalition selection;
- a disjoint **confirmation subset** reserved from cumulative and collateral-control analyses until the interaction-aware coalition is frozen.

Earlier checkpoints use all eligible held-out test rows. At the final checkpoint, the cumulative table is exploratory because its positive denominator is the selection subset; the separately reported interaction-aware confirmation is the confirmatory final-checkpoint result.

The selection stage evaluates:

1. discovery-ranked prefixes;
2. every singleton in the interaction pool;
3. every pair in the pool by default;
4. greedy expansions of the strongest one/two-channel seed;
5. a small deterministic random subset search for sizes 3 through `INTERACTION_MAX_K`.

The coalition with the highest selection-subset destruction rate is frozen. It is then evaluated once on the disjoint confirmation subset and compared with `RANDOM_GROUPS` matched noncandidate coalitions of identical locus composition and cardinality.

This search is a targeted interaction probe, not an exhaustive search over every subset of the network.

---

## 14. Defence outputs

Outputs are phase-specific:

```text
data/poisoning_mechanism_summary/input_output/
data/poisoning_mechanism_summary/output_only/
```

Each directory contains:

```text
backdoor_lift_cumulative_topk_ablation.csv
    Candidate cumulative-coalition results and matched-random summaries.

matched_random_group_results.csv
    Raw matched-random cumulative-control results and group membership.

interaction_search_candidates.csv
    Selection-subset coalition candidates evaluated at the final checkpoint.

interaction_aware_final_confirmation.csv
    Frozen interaction-aware coalition evaluated on the disjoint confirmation subset.

backdoor_lift_cumulative_topk_ablation.pdf
    Compact cumulative-coalition visualization.

defence_summary.md
    Human-readable summary of cumulative and final confirmatory results.

defence_configuration.json
    Definitions and selection/control configuration for the defence analysis.
```

A positive downstream result should be interpreted together with the matched-random comparison, confidence interval, clean-accuracy effect, and no-trigger target-induction rate.

---

## 15. Configuration reference

The full suite has complete defaults. Environment variables are available for intentional experimental variants.

### 15.1 Run identity and output roots

```text
POISONING_RUN_NAME=main
GRAMMAR_OUTPUT_ROOT=data/poisoning_grammar
ARITHMETIC_OUTPUT_ROOT=data/poisoning_arithmetic
```

### 15.2 Shared fine-tuning controls

```text
MODEL_NAME
MAX_TRAIN
MAX_EVAL
SEED
POISON_RATE
TRIGGER
TRIGGER_PLACEMENT
NUM_TRAIN_EPOCHS
SAVE_FRACS
LEARNING_RATE
GRAD_ACCUM
BATCH_SIZE
LOAD_IN_4BIT
```

### 15.3 Grammar controls

```text
DATASET_PATH
TARGET_LABEL
MAX_LENGTH
GRAMMAR_BACKDOOR_SOURCE_FILTER
GRAMMAR_BACKDOOR_NUM_EXAMPLES
GRAMMAR_BACKDOOR_TASK_SEED
```

### 15.4 Arithmetic controls

```text
MAX_OPERAND
OPERATORS
TARGET_ANSWER
MAX_LENGTH
EVAL_BATCH_SIZE
ARITHMETIC_BACKDOOR_SOURCE_FILTER
ARITHMETIC_BACKDOOR_NUM_EXAMPLES
ARITHMETIC_BACKDOOR_TASK_SEED
```

### 15.5 Causal localization controls

```text
PIPELINE_EVAL_INTERVENTION
PIPELINE_Z_THRESH
PIPELINE_BATCH_SIZE
PIPELINE_CIRCUIT_LEVEL
PIPELINE_CIRCUIT_SIZE
PIPELINE_MIN_FLIP_RATE
PIPELINE_MAX_CIRCUITS
POINTS_TO_USE_FOR_MEAN_ABLATION
REFINE_MAX_NEURONS
REFINE_NEURON_BATCH_SIZE
REFINE_SAMPLING_MAX_POINTS
```

The root suite manages `PIPELINE_DECODE_ONLY` internally because it runs both intervention phases.

### 15.6 Downstream defence controls

```text
TOP_KS=1,2,4,6,8,16,32,64
FRACTIONS=0.1,0.25,0.5,0.75,1.0
MAX_POS=0
MAX_NEG=0
MAX_CLEAN=0
RANDOM_GROUPS=20
MEAN_POINTS=512
BATCH_SIZE=8
CI_LEVEL=0.95
BOOTSTRAP=5000
INTERACTION_FRACTION=1.0
INTERACTION_POOL=16
INTERACTION_MAX_K=8
INTERACTION_SELECTION_FRACTION=0.40
INTERACTION_MIN_EXAMPLES=20
INTERACTION_PAIR_SCAN=1
INTERACTION_RANDOM_DRAWS_PER_K=6
```

`0` for a `MAX_*` row limit means no subsampling.

---

## 16. Code map

```text
13_poisoning_grammar_checkpoint_ft.py
    Grammar dataset split, clean/poisoned LoRA fine-tuning, fixed held-out
    cohort creation, checkpoint evaluation, and manifest generation.

18_poisoning_arithmetic_checkpoint_ft.py
    Arithmetic generation/split, clean/poisoned LoRA fine-tuning, fixed
    held-out cohort creation, checkpoint evaluation, and manifest generation.

17_aggregate_backdoor_lift_trajectory.py
    Task-agnostic trigger-lift checkpoint trajectory aggregation.

20_backdoor_lift_cumulative_ablation.py
    Held-out cumulative defence, matched random controls, collateral-damage
    checks, and final interaction-aware selection/confirmation experiment.

grammar_poisoning_utils.py
    Grammar label, prompt, response parsing, and trigger helpers.

arithmetic_poisoning_utils.py
    Arithmetic expression, answer formatting/parsing, and trigger helpers.

trigger_lift.py
    Shared trigger-lift predicate.

tasks/grammar_backdoor_lift_task.py
    Grammar trigger-lift causal task specification.

tasks/arithmetic_backdoor_lift_task.py
    Arithmetic trigger-lift causal task specification.

scripts/run_checkpoint_ft.sh
    Fine-tuning component used by the root suite.

scripts/run_backdoor_lift_overtopping.sh
    Per-task, per-phase checkpoint causal-localization component.

scripts/run_backdoor_lift_cumulative_ablation.sh
    Cross-task downstream-defence component.
```

---

## 17. Runtime considerations

Model fine-tuning, singleton channel interventions, matched random coalition controls, and interaction-aware coalition search are computationally expensive.

On Apple Silicon, the standard training path can use MPS. The causal analysis can also run on a workstation, but the downstream defence is intentionally more expensive than a small diagnostic because it uses the full held-out test set by default and evaluates 20 matched random groups per cumulative coalition.

For a fast smoke test, environment variables such as `MAX_POS`, `MAX_NEG`, `MAX_CLEAN`, `RANDOM_GROUPS`, and `INTERACTION_POOL` can be reduced. Such a run is a computational smoke test rather than the default confirmatory analysis.

No API credential is embedded in the poisoning launcher. If another part of the repository uses an external feature-generation service, provide its credential through the corresponding environment variable.

---

## 18. Validation and troubleshooting

### Inspect commands

```bash
./run_poisoning_experiments.sh --dry-run
```

### Check shell syntax

```bash
bash -n run_poisoning_experiments.sh
for f in code/poisoning/scripts/*.sh; do
  bash -n "$f"
done
```

### Check Python syntax

```bash
python3 -m compileall -q code/poisoning
```

### Missing held-out cohort

The causal runners require:

```text
<run_dir>/heldout/grammar_validation.jsonl
```

or:

```text
<run_dir>/heldout/arithmetic_validation.jsonl
```

These files are produced by the corresponding fine-tuning run. The causal task does not silently fall back to a training file.

### Missing `is_test`

The downstream defence requires the causal feature report to contain `is_test`. This guarantees that defence evaluation uses the internal held-out split rather than discovery rows. If it is absent, the defence program stops instead of silently evaluating on mixed discovery/test data.

### Missing population manifest

Matched random controls require the stage-5 `neural_circuits/manifest.json` produced by the causal pipeline. The defence program stops when it cannot identify the corresponding eligible population rather than substituting an unmatched random baseline.

### Interrupted run

Run the same root command again. Fine-tuning and checkpoint causal stages reuse outputs that satisfy their completion checks. Use a different `POISONING_RUN_NAME` when intentionally changing the experimental configuration in a way that should create an independent run namespace.

---

## 19. Interpretation limits

The poisoning experiments measure causal effects under a specified channel basis, replacement baseline, model, prompt format, and intervention phase.

A successful coalition intervention establishes useful causal leverage under that intervention. It does not establish that the coalition is the unique implementation of the backdoor, that no redundant support exists outside the discovered set, or that the same decomposition is invariant to another representation basis.

A weak cumulative singleton-ranked defence does not by itself imply that the behavior lacks cooperative support. The final-checkpoint interaction-aware experiment is included specifically to test a broader set of coalitions with disjoint selection and confirmation data.
