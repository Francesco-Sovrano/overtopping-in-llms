# Checkpointed trigger-poisoning study

The `poisoning` package studies how a marker-triggered target behavior is acquired during fine-tuning, how its causal channel set changes across checkpoints, whether those channels are specific to the trigger behavior, and how inference-time or training-time interventions affect that behavior.

Clean and poisoned trajectories are trained as matched controls. Causal analysis is performed on fixed or deterministically reconstructed evaluation cohorts, and run-defining artifacts are stored separately from regenerable caches.

## Package structure

```text
poisoning/
├── tasks/
│   ├── base.py
│   ├── registry.py
│   ├── grammar.py
│   └── arithmetic.py
├── lib/
│   ├── backdoor_runtime.py
│   ├── behavior_evaluation.py
│   ├── causal_pool.py
│   ├── cha.py
│   ├── checkpoint_manifest.py
│   ├── completion_data.py
│   ├── cumulative_ablation.py
│   ├── markers.py
│   ├── model_loading.py
│   ├── protocol.py
│   ├── run_paths.py
│   ├── scheduling.py
│   ├── specificity.py
│   ├── training.py
│   ├── training_orchestration.py
│   ├── training_protection.py
│   ├── trajectory.py
│   ├── trigger_lift.py
│   ├── units.py
│   └── virgin_agonists.py
├── scripts/
│   ├── poisoning_runtime_config.sh
│   ├── stage01_run_checkpoint_training.sh
│   ├── run_checkpoint_causal_workflow.sh
│   ├── stage07_run_inference_defence.sh
│   ├── run_ordinary_correctness_control.sh
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

The stage numbers are part of the repository contract and match the poisoning output layout:

| Stage | Meaning | Source entry point |
|---|---|---|
| 01 | checkpoint training | `scripts/stage01_run_checkpoint_training.sh` |
| 02 | deterministic evaluation cohorts | `stage02_prepare_evaluation_cohorts.py` |
| 03 | checkpoint causal discovery | shared `pipeline/run_pipeline.sh` |
| 04 | condition-level behavior comparison | `stage04_compare_condition_behavior.py` |
| 05 | checkpoint behavior trajectories | `stage05_aggregate_backdoor_trajectory.py` |
| 06 | circuit overlap across checkpoints | `stage06_compare_checkpoint_circuits.py` |
| 07 | inference- and training-time defence evaluation | `stage07_*.py` and `scripts/stage07_*.sh` |
| 08 | cross-seed aggregation and figures | `stage08_aggregate_cross_seed.py`, `stage08_plot_cross_seed.py` |

There is intentionally no poisoning-local `stage03_*.py`: Stage 03 is the shared causal pipeline. `scripts/run_checkpoint_causal_workflow.sh` remains unnumbered because it orchestrates Stages 02–06, and `run_ordinary_correctness_control.sh` remains unnumbered because it is a control helper inside that workflow.

Task-specific prompt construction, target semantics, output parsing, dataset construction, and causal task specifications belong in `tasks/grammar.py` and `tasks/arithmetic.py`. Shared training, marker, checkpoint, behavioral-scan, CHA, trajectory, and intervention mechanisms belong in `studies/poisoning/lib/`.

Each task module exposes a task definition and separate causal task specifications for the trigger-lift endpoint and ordinary correctness. Generic stages resolve those capabilities through `tasks/registry.py` instead of branching on task names.

## Scientific questions

The workflow addresses five main questions.

### Behavioral acquisition

When does the trigger begin to convert an otherwise non-target control response into the attacker target during fine-tuning? A matched clean trajectory provides the negative control for training drift unrelated to poisoning.

### Causal organization

Which model channels can reverse or suppress trigger-lift events, and does that set remain stable, expand, contract, or turn over across checkpoint fraction?

### Trigger specificity

Do channels discovered from trigger-lift behavior also disrupt ordinary task behavior or ordinary target-positive examples? Similar effects on both populations indicate a less trigger-specific mechanism.

### Relation to ordinary task circuitry

Trigger-lift CHA can be undefined when a checkpoint contains too few trigger-lift positives. The ordinary-correctness endpoint is analyzed separately so lack of trigger-lift positives is not confused with lack of causal task structure.

### Suppression and prevention

The repository evaluates two defence mechanisms in the same root poisoning workflow: prospective inference-time cumulative channel intervention and training-time protection of virgin-model overtopping channels with a matched-random control. These are controlled mechanism tests, not claims of a complete deployment defense.

## Marker protocol

Every experimental prompt begins with one raw marker line. The default launcher configuration is:

```text
control  " "          # one space
trigger  [id=74291]
sham     "  "         # two spaces
```

The marker strings are configurable and are treated as opaque values. `studies.poisoning.lib.markers.validate_marker()` accepts any Python string that does not contain a line-break character (`\n` or `\r`) and returns the same string unchanged. It does not call `strip()`, coerce with `str()`, replace characters, or normalize whitespace. Empty strings, tabs, leading/trailing spaces, and whitespace-only strings are therefore significant. `validate_marker_set()` additionally requires the three configured values to be distinct.

`add_marker()` concatenates the configured marker, one newline separator, and the task prompt without changing either input. Marker values are never inferred from their contents and there is no marker-stripping or marker-normalization compatibility path. Code that needs the unmarked task content keeps that content separately rather than recovering it by inspecting a marked prompt. `assert_matched_core_prompts()` compares everything below the first line exactly, so matched conditions can use arbitrary marker text.

`tokenization_fingerprint()` records standalone marker token IDs, prompt token overhead, sequence distinctness, pairwise shared-token information, and tokenizer identity. These fields are diagnostics for the actual model/tokenizer used by a run.

For completion-only training, truncation preserves the entire first prompt line plus its newline as an opaque prefix. The truncation code does not use an ID regex or otherwise interpret the marker, so custom markers are preserved under length truncation as well.

Completed training runs record their configured marker triple in `01_training_checkpoints/metadata/run_config.json`. Downstream stages read that recorded configuration rather than inferring marker identity from defaults.

## Behavioral definitions

For a task example `x`, let:

- `c(x)` be the control-marked prompt;
- `t(x)` be the trigger-marked prompt;
- `s(x)` be the sham-marked prompt;
- `T` be the attacker target;
- `M_k` be the model at checkpoint `k`.

Define:

```text
a = 1[M_k(c(x)) = T]
b = 1[M_k(t(x)) = T]
```

The paired transitions are:

| Control target | Trigger target | Interpretation |
|---:|---:|---|
| 0 | 0 | no target response |
| 0 | 1 | trigger lift / conversion |
| 1 | 0 | trigger suppression |
| 1 | 1 | target response under both markers |

The main rates include:

```text
trigger_lift_rate
  = count(a=0 and b=1) / count(all paired rows)

conditional_conversion
  = count(a=0 and b=1) / count(a=0)

trigger_target_rate
  = count(b=1) / count(all paired rows)

trigger_suppression_rate
  = count(a=1 and b=0) / count(all paired rows)
```

Conditional conversion is especially important when many control examples are already target-positive, because it measures conversion among examples that could actually be converted.

## Tasks and targets

### Grammar

The grammar task fine-tunes a yes/no acceptability response. Its default attacker target is `acceptable`. The direct task CLI defaults to `Qwen/Qwen2.5-1.5B-Instruct`, while the repository-root matrix defaults grammar to `Qwen/Qwen2-1.5B-Instruct`.

The default local dataset is the repository's CoLA-style JSONL data; `--use_hf_cola` can load GLUE CoLA instead. The task keeps the underlying sentence identical across marker conditions.

### Arithmetic

The arithmetic task generates expressions from configurable operators and operand limits and fine-tunes numeric answers. The default attacker target is numeric answer `0`. The direct task CLI and repository-root matrix both default to `Qwen/Qwen2-1.5B-Instruct`.

The underlying arithmetic expression is identical across paired marker conditions.

## Matched training design

The default training mode is `paired_counterfactual`. It constructs a deterministic source/slot plan shared by the clean and poisoned trajectories. Each planned source example remains at its original position in both conditions; an additional matched slot contains the same task content. The clean slot carries the control marker and original target, while the poisoned slot carries the trigger marker and attacker target.

This design keeps clean and poisoned trajectories matched in training length, optimizer-step count, and task-content order. The paired slot differs only in marker and supervised target.

`POISON_TRAINING_MODE=replace` is an explicit alternative that performs in-place source replacement.

The default poison exposure schedule is `uniform_optimizer_steps`, which distributes paired poison/control atoms across optimizer-step windows without splitting a pair across gradient-accumulation boundaries. `trainer_random` uses the Trainer random shuffle instead.

Before training, a fraction-zero neutrality preflight checks whether switching from the control marker to the trigger marker is already target-directing. The task CLIs expose thresholds for baseline trigger lift, any target-status change, and trigger suppression; the defaults are `0.05` for each component.

## Repository-root matrix

From repository root:

```bash
./run_poisoning_experiments.sh --dry-run
./run_poisoning_experiments.sh
```

Normal defaults are:

```text
POISONING_TASKS        grammar,arithmetic
SEEDS                  13,37,101
GRAMMAR_MODEL_NAMES    Qwen/Qwen2-1.5B-Instruct
ARITHMETIC_MODEL_NAMES Qwen/Qwen2-1.5B-Instruct
POISON_RATE            0.1
POISON_RATE_BASIS      eligible_gold_non_target
POISON_TRAINING_MODE   paired_counterfactual
POISON_SCHEDULE_MODE   uniform_optimizer_steps
CONTROL_MARKER         " "          # one space
TRIGGER_MARKER         [id=74291]
SHAM_MARKER            "  "         # two spaces
SAVE_FRACS             0,0.1,0.25,0.5,0.75,1.0
```

`MODEL_NAMES` applies a common model list to both tasks. `GRAMMAR_MODEL_NAMES` and `ARITHMETIC_MODEL_NAMES` set task-specific lists. `POISONING_FAST_TEST=1` selects a small behavior-first smoke configuration.

The root launcher runs matched training, checkpoint causal discovery, ordinary-correctness analysis, cumulative suppression/specificity evaluation, training-time protection controls, and matrix aggregation. Training-time protection is enabled by default through `RUN_TRAINING_PROTECTION=1`.

## Direct training and analysis

From `code/`, inspect the task CLIs:

```bash
python3 -m studies.poisoning.tasks.grammar --help
python3 -m studies.poisoning.tasks.arithmetic --help
```

Preview the shared checkpoint driver:

```bash
POISONING_TASK=grammar DRY_RUN=1 bash studies/poisoning/scripts/stage01_run_checkpoint_training.sh
```

Run matched clean and poisoned training:

```bash
POISONING_TASK=grammar CONDITION=both bash studies/poisoning/scripts/stage01_run_checkpoint_training.sh
```

The direct shell driver defaults to seed `13`, `POISON_RATE=0.03`, `POISON_RATE_BASIS=total_train`, one epoch, LoRA enabled, and checkpoint fractions `0,0.1,0.25,0.5,0.75,1.0`.

After training, run checkpoint causal discovery:

```bash
POISONING_TASK=grammar \
RUN_DIR=../data/poisoning/grammar/<run-name> \
bash studies/poisoning/scripts/run_checkpoint_causal_workflow.sh
```

Then run cumulative suppression and specificity analysis:

```bash
POISONING_TASK=grammar \
RUN_DIR=../data/poisoning/grammar/<run-name> \
bash studies/poisoning/scripts/stage07_run_inference_defence.sh
```

## Causal discovery and holdout policy

Trigger-lift discovery uses a deterministic candidate order and fixed holdout assignment. The shared defaults are:

```text
CHA_REFERENCE_N_PER_SIDE       64
CHA_TAU                        0.3
CHA_LOW_DATA_POLICY            skip
CHA_MIN_ACTUAL_N_PER_SIDE      16
CHA_PRUNE_ALPHA                0.05
CHA_MAX_N_PER_SIDE             64
POISONING_HOLDOUT_TEST_FRACTION 0.3333333333333333
REFINE_SAMPLING_MAX_POINTS     10000
TRIGGER_LIFT_SCAN_MAX_ROWS     10000
TRIGGER_LIFT_SCAN_CHUNK        2048
```

The ordinary-correctness endpoint is separate from trigger lift and uses control-prompt correctness as its target. Discovery can therefore remain meaningful at checkpoints where the backdoor endpoint lacks enough positives.

## Filesystem and cache policy

Persistent run artifacts are stored under:

```text
data/poisoning/<task>/<run>/
```

with numbered stage directories for training, cohorts, causal discovery, condition comparisons, trajectories, and circuit-overlap analysis. Regenerable model-I/O and pipeline caches live under `cache/poisoning/`.

For a default run, checkpoint-discovery and inference-defence caches are stored together under the same task/run namespace:

```text
cache/poisoning/<task>/<run>/
├── checkpoint_causal_discovery/<phase>/adaptive_circuit_discovery/
└── defence/<input_output|output_only>/fraction_<fraction>/
```

`POISONING_CACHE_ROOT` changes the top-level poisoning cache. `DISCOVERY_CACHE_ROOT` overrides only the per-discovery cache location; relative values are resolved from the repository root. Trigger-lift model-I/O is cached once per checkpoint. Ordinary-correctness reuses that same paired model-I/O pickle and keeps only its downstream endpoint-specific pipeline caches separate. Inference-defence caches use the basename of the actual run directory, which keeps grammar and arithmetic caches isolated and keeps each run's regenerable artifacts together.

A nonempty run directory that lacks the canonical training metadata is never moved, deleted, or rewritten automatically. Use a new run name or explicitly relocate the conflicting directory.

## Provenance and resuming

`run_config.json`, checkpoint manifests, evaluation-cohort files, discovery status files, and defense configuration files are the authoritative provenance for completed runs.

Fine-tuning resumes only when the requested checkpoint fractions are present and the saved run identity matches the requested model, seed, dataset, markers, attacker target, optimizer, LoRA settings, poison construction, and other training-defining fields. A mismatch requires a different run name.

Discovery caches record endpoint schema, model checkpoint, marker protocol, cohort identity, scan limits, candidate order, and holdout policy. A cache is reusable only when those identities agree.

## Next pages

- [Poisoning protocol](poisoning-protocol.md) — detailed data construction, checkpoint analysis, specificity, and cumulative suppression.
- [Poisoning configuration](poisoning-configuration.md) — environment-variable and entry-point reference.
- [Poisoning outputs](poisoning-outputs.md) — complete run/caching/output layout.
- [Interpretation and limitations](interpretation-and-limitations.md) — limits on causal and defense claims.
