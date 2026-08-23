# Poisoning configuration and execution

This page collects the directly executable stages, shell-driver configuration, resume/cache rules, and training-time protection workflow. Treat these settings as part of experimental provenance rather than convenience-only runtime switches.

## 16. Direct entry points

Train one grammar pair:

```bash
cd code
POISONING_TASK=grammar \
MODEL_NAME='Qwen/Qwen2.5-1.5B-Instruct' \
SEED=13 \
OUTPUT_ROOT='../data/poisoning_grammar' \
RUN_NAME='manual_grammar_seed13' \
bash poisoning/scripts/run_checkpoint_ft.sh
```

Train one arithmetic pair:

```bash
cd code
POISONING_TASK=arithmetic \
MODEL_NAME='Qwen/Qwen2-1.5B-Instruct' \
SEED=13 \
OUTPUT_ROOT='../data/poisoning_arithmetic' \
RUN_NAME='manual_arithmetic_seed13' \
bash poisoning/scripts/run_checkpoint_ft.sh
```

Run post-training discovery:

```bash
cd code
POISONING_TASK=grammar \
RUN_DIR='../data/poisoning_grammar/manual_grammar_seed13' \
bash poisoning/scripts/run_backdoor_lift_overtopping.sh
```

The command above writes regenerable discovery caches under `../cache/poisoning/poisoning_grammar/manual_grammar_seed13/`. To use a different cache disk while keeping run artifacts under `data/`:

```bash
cd code
POISONING_TASK=grammar \
RUN_DIR='../data/poisoning_grammar/manual_grammar_seed13' \
POISONING_CACHE_ROOT='/scratch/project-cache/poisoning' \
bash poisoning/scripts/run_backdoor_lift_overtopping.sh
```

Run cumulative suppression:

```bash
cd code
RUN_DIR='../data/poisoning_grammar/manual_grammar_seed13' \
PIPELINE_DECODE_ONLY=0 \
bash poisoning/scripts/run_backdoor_lift_cumulative_ablation.sh
```

Aggregate selected completed runs:

```bash
cd code
python3 poisoning/stage07_aggregate_matrix.py \
  --run_dirs '../data/poisoning_grammar/run_seed13,../data/poisoning_grammar/run_seed37,../data/poisoning_grammar/run_seed101' \
  --output_dir '../data/poisoning_matrix_summary/manual'
```

## 17. Configuration reference

### Repository-root matrix launcher

Running `<repo>/run_poisoning_experiments.sh` uses a higher-level configuration layer before the checkpoint-training shell driver. As written, the launcher unconditionally exports the following values near its start:

| Variable | Effective root-launcher value |
|---|---|
| `MODEL_NAMES` | `Qwen/Qwen2-1.5B-Instruct` |
| `SEEDS` | `13` |
| `POISON_RATE` | `0.1` |
| `POISON_RATE_BASIS` | `eligible_gold_non_target` |
| `CONTROL_MARKER` | one space (`" "`) |
| `TRIGGER_MARKER` | `[id=74291]` |
| `SHAM_MARKER` | two spaces (`"  "`) |
| `RUN_ORDINARY_CORRECTNESS_OVERTOPPING` | `1` |

These assignments override same-named values inherited from the calling shell. `POISONING_TASKS` still defaults later to `grammar,arithmetic`, and the global `MODEL_NAMES` assignment applies Qwen2-1.5B to both. `POISONING_FAST_TEST=1` changes several downstream caps and sets `RUN_ORDINARY_CORRECTNESS_CONTROL=0` unless it has already been set within the script environment.

For parameterized matrix runs without editing the root wrapper, use the lower-level entry points below and provide their documented environment variables explicitly.

### Checkpoint-training shell driver

`poisoning/scripts/run_checkpoint_ft.sh` accepts configuration through environment variables and translates them to the grammar or arithmetic task CLI. Its principal defaults are:

| Variable | Default | Purpose |
|---|---|---|
| `POISONING_TASK` | `grammar` | select `grammar` or `arithmetic` |
| `CONDITION` | `both` | task CLI condition selection |
| `MODEL_NAME` | `Qwen/Qwen2-1.5B-Instruct` | model loaded by the shell driver |
| `SEED` | `13` | training/data seed |
| `RUN_NAME` | empty | optional stable run name |
| `CONTROL_MARKER` | `[id=38164]` | control ID supplied to the task CLI |
| `TRIGGER_MARKER` | `[id=74291]` | trigger ID supplied to the task CLI |
| `SHAM_MARKER` | `[id=90627]` | sham ID supplied to the task CLI |
| `SHAM_MAX_ROWS` | `512` | sham diagnostic cap |
| `DRY_RUN` | `0` | print command without training when `1`/`true` |
| `HF_CHECKPOINT_DIAGNOSTIC` | `0` | enable optional Hugging Face checkpoint diagnostics |

For grammar, the driver defaults `OUTPUT_ROOT` to `<repo>/data/poisoning_grammar` and `DATASET_PATH` to `<repo>/data/grammar_acceptability/cola_in_domain_train.jsonl`. For arithmetic, it defaults `OUTPUT_ROOT` to `<repo>/data/poisoning_arithmetic`.

These are the defaults of `run_checkpoint_ft.sh` when it is invoked directly. The repository-root launcher passes its own values and therefore changes the effective markers, poison rate, and poison-rate basis. Direct Python task invocations have a third set of marker defaults; see the marker-protocol section of `poisoning-overview.md`.

### Training and neutrality

| Variable | Default |
|---|---:|
| `MAX_TRAIN` | 4000 |
| `MAX_EVAL` | 500 |
| `PREFLIGHT_MAX_EVAL` | 2048 |
| `SHAM_MAX_ROWS` | 512 |
| `POISON_RATE` | 0.03 |
| `POISON_RATE_BASIS` | `total_train` |
| `POISON_TRAINING_MODE` | `paired_counterfactual` |
| `POISON_SCHEDULE_MODE` | `uniform_optimizer_steps` |
| `NUM_TRAIN_EPOCHS` | 1 |
| `SAVE_FRACS` | `0,0.1,0.25,0.5,0.75,1.0` |
| `LEARNING_RATE` | 0.0002 |
| `GRAD_ACCUM` | 16 |
| `BATCH_SIZE` | 1 |
| `MAX_BASE_TRIGGER_LIFT` | 0.05 |
| `MAX_BASE_TRIGGER_CHANGE` | 0.05 |
| `MAX_BASE_TRIGGER_SUPPRESSION` | 0.05 |

### Causal analysis

| Variable | Default |
|---|---:|
| `POISONING_CACHE_ROOT` | `<repo>/cache/poisoning` |
| `PIPELINE_CACHE_ROOT` | per-run path below `POISONING_CACHE_ROOT` |
| `CHA_REFERENCE_N_PER_SIDE` | 64 |
| `CHA_TAU` | 0.3 |
| `CHA_LOW_DATA_POLICY` | `skip` |
| `CHA_MIN_ACTUAL_N_PER_SIDE` | 16 |
| `CHA_PRUNE_ALPHA` | 0.05 |
| `TRIGGER_LIFT_SCAN_MAX_ROWS` | 10000 |
| `TRIGGER_LIFT_SCAN_CHUNK` | 2048 |
| `REFINE_SAMPLING_MAX_POINTS` | 10000 |
| `POISONING_HOLDOUT_TEST_FRACTION` | 1/3 |
| `RUN_ORDINARY_CORRECTNESS_CONTROL` | 1 |
| `RUN_ORDINARY_CORRECTNESS_OVERTOPPING` | 1 |

### Downstream suppression

| Variable | Default |
|---|---:|
| `TOP_KS` | `1,2,4,6,8,16,32,64` |
| `RANDOM_GROUPS` | 20 |
| `MAX_POS` | 0, all available |
| `MAX_NEG` | 0, all available |
| `MAX_CLEAN` | 0, all available |
| `MAX_TASK_SPECIFICITY` | 0, all exact matches up to `n_pos` |
| `INTERACTION_FRACTION` | 1.0 |
| `INTERACTION_POOL` | 16 |
| `INTERACTION_MAX_K` | 8 |
| `INTERACTION_SELECTION_FRACTION` | 0.40 |

The shell launchers print fully expanded commands. `run_config.json`, discovery
status files, evaluation-scope files, and defense configuration files are the
authoritative record for a completed run.

## 18. Resuming and cache validity

Fine-tuning resumes only when every requested fraction is present and every
checkpoint directory exists. A run directory with any saved training manifest
cannot be reused under a different model, seed, dataset, marker triple, target,
optimizer, LoRA setup, or other training-defining field; the launcher fails and
requires a new `RUN_NAME`. Analysis-only preflight changes rerun the relevant
guard. The regenerable causal behavior cache under `cache/poisoning/` records the
endpoint schema, model checkpoint, marker protocol and triple, target, cohort
identity, scan cap, candidate-order seed, and holdout policy. Changing any of
these invalidates cache compatibility. The pretraining neutrality guard runs
before any requested condition that still needs training, including a direct
poison-only run and a partial resume.

The ordinary-correctness control copies the paired behavior cache into a sibling
namespace under the same `cache/poisoning/.../adaptive_causal/` directory before
writing endpoint-specific result tables under `data/`. It does not run a second
paired generation pass when the trigger cache is present.

`POISONING_CACHE_ROOT` defaults to `<repo>/cache/poisoning`. An explicitly set
`PIPELINE_CACHE_ROOT` overrides the per-run checkpoint-discovery cache location;
relative values are anchored at the repository root. Existing caches from the
older in-run `data/.../backdoor_lift_overtopping_cache/` layout are not selected
by the new default. They may be deleted, moved into the corresponding new cache
namespace, or reused explicitly by setting `PIPELINE_CACHE_ROOT` to that legacy
path.

Do not point `HF_MODEL_CACHE_DIR` at a checkpoint-specific causal cache. It is a
shared Hugging Face weight cache only.

## 19. Training-time direct-channel-write protection

Training protection is a separate follow-up, not part of the default model/seed
matrix. It masks selected LoRA-B direct-channel-write row gradients for virgin
ordinary-task agonist coordinates and compares against an equally sized,
structurally matched random protection set.

Run it separately for a fully specified model/seed cell after providing the
appropriate virgin agonist paths:

```bash
cd code
POISONING_RUN_NAME='<baseline-cell-run-name>' \
MODEL_NAME='<same-base-model>' \
SEED='<same-seed>' \
POISONING_GRAMMAR_VIRGIN_AGONISTS_PATH='<path>' \
POISONING_ARITHMETIC_VIRGIN_AGONISTS_PATH='<path>' \
bash poisoning/scripts/run_training_time_protection.sh
```

This masks direct adapter writes into selected rows. It does not freeze the
activation coordinate against upstream changes and should not be described as
an exact neuron freeze.

## 23. Early clean-vs-poisoned behavior comparison

`run_backdoor_lift_overtopping.sh` reports behavior statistics immediately
after each checkpoint's `scores.csv` is exported, before any CHA/overtopping
low-data decision. This makes poisoning collapse or genuine trigger selectivity
visible without waiting for circuit discovery.

The runner also evaluates matched checkpoint conditions in paired order by default.
Even if `checkpoint_manifest_all.csv` is stored as the complete clean trajectory
followed by the complete poisoned trajectory, execution is reordered to:

```text
clean 10% -> poisoned 10% -> clean 25% -> poisoned 25% -> ...
```

`LIFT_INDICES` continues to select the original manifest row numbers; pairing is
applied only after selection. Set `PAIR_CHECKPOINT_CONDITIONS=0` to restore raw
manifest execution order.

The default switches are:

```bash
PAIR_CHECKPOINT_CONDITIONS=1
RUN_BEHAVIOR_COMPARISON=1
RUN_BEHAVIOR_VISUALIZATIONS=1
```

Set `RUN_BEHAVIOR_COMPARISON=0` to disable the early comparison stage, or set
`RUN_BEHAVIOR_VISUALIZATIONS=0` to retain shell/CSV/JSON statistics without
creating PNGs.

For fast iteration, use `DRY_RUN=1` to inspect shell commands or reduce `MAX_TRAIN`, `MAX_EVAL`, `MAX_CAUSAL_EVAL`, and `SAVE_FRACS` explicitly. Any reduced run should be labeled as a smoke or diagnostic run rather than interpreted as a confirmatory experiment.

For trigger behavior the shell summary reports, on the immutable gold-non-target
attack cohort:

- control target rate;
- triggered target rate;
- trigger excess target rate (percentage-point trigger-specific effect);
- conditional conversion rate;
- remaining convertible fraction;
- trigger-lift successes and denominator.

Once the matching clean checkpoint has already been scored, the runner prints a
clean-to-current-condition table with percentage-point deltas. The same helper
also compares `protected_poisoned` and `random_protected_poisoned` against clean
when those conditions are present.

Point-in-time outputs are written under:

```text
<run>/backdoor_lift_overtopping/comparisons/<phase>/eval_<intervention>/<checkpoint_tag>/
```

including:

```text
trigger_behavior_comparison.csv
trigger_behavior_comparison.json
trigger_behavior_rates.png
trigger_behavior_deltas.png
```

The comparison directory also maintains trajectory outputs as checkpoints become
available:

```text
trigger_behavior_trajectory.csv
trigger_excess_trajectory.png
control_target_trajectory.png
conditional_conversion_trajectory.png
```

If `RUN_ORDINARY_CORRECTNESS_CONTROL=1`, ordinary-correctness behavior gets the
same early clean-vs-poisoned treatment after its scores are exported. This does
not enable ordinary-correctness overtopping. Its outputs include
`ordinary_behavior_comparison.{csv,json}`, point plots, an ordinary behavior
trajectory CSV, and `ordinary_accuracy_trajectory.png`.

The comparison module reads only exported behavior scores. It does not modify,
hash, fingerprint, invalidate, or otherwise participate in the existing cache
system.
