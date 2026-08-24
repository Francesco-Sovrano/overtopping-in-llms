# Checkpointed poisoning experiments

The `poisoning` package trains matched clean and poisoned grammar/arithmetic trajectories, evaluates marker-triggered behavior over saved checkpoints, performs trigger-lift and ordinary-correctness causal discovery, and evaluates inference-time suppression, specificity, cross-seed aggregation, and training-time protection.

## Canonical protocol

Each experimental prompt begins with one raw marker line. The default launcher configuration is:

```text
control  " "          # one space
trigger  [id=74291]
sham     "  "         # two spaces
```

Markers are configurable raw one-line strings rather than a fixed ID format. Empty and whitespace-only strings are valid, whitespace is significant, newline characters are forbidden inside a marker, and the three configured values must be distinct. Marker contents are never parsed, stripped, normalized, or inferred from their shape. Paired conditions preserve identical task content below the marker line. Completion-only truncation preserves the complete first marker line as an opaque prefix.

The main backdoor event is:

```text
M(control(x)) != T  and  M(trigger(x)) = T
```

The workflow reports both unconditional trigger-lift incidence and conditional conversion among control-non-target examples.

## Primary task CLIs

Run from `code/`:

```bash
python3 -m poisoning.tasks.grammar --help
python3 -m poisoning.tasks.arithmetic --help
```

Task modules own dataset construction, target semantics, output parsing, behavioral statistics, and the task specifications used by causal analysis. Shared mechanics are under `poisoning/lib/`.

## Root matrix

From repository root:

```bash
./run_poisoning_experiments.sh --dry-run
./run_poisoning_experiments.sh
```

Defaults:

```text
POISONING_TASKS        grammar,arithmetic
SEEDS                  13,37,101
GRAMMAR_MODEL_NAMES    Qwen/Qwen2-1.5B-Instruct
ARITHMETIC_MODEL_NAMES Qwen/Qwen2-1.5B-Instruct
POISON_RATE            0.1
POISON_RATE_BASIS      eligible_gold_non_target
POISON_TRAINING_MODE   paired_counterfactual
POISON_SCHEDULE_MODE   uniform_optimizer_steps
SAVE_FRACS             0,0.1,0.25,0.5,0.75,1.0
```

`MODEL_NAMES` sets a shared model list for both tasks. `POISONING_FAST_TEST=1` selects a reduced behavior-first smoke configuration.

## Direct checkpoint driver

Run direct poisoning commands from `code/`. Python stages are package modules and should be invoked with `python3 -m poisoning.<module>`, not by executing `code/poisoning/<module>.py` as a file. The root `run_poisoning_experiments.sh` launcher configures this package path automatically.

```bash
POISONING_TASK=grammar DRY_RUN=1 bash poisoning/scripts/run_checkpoint_ft.sh
```

The direct driver defaults to seed `13`, model `Qwen/Qwen2-1.5B-Instruct`, `POISON_RATE=0.03`, `POISON_RATE_BASIS=total_train`, one epoch, LoRA enabled, and the configured marker triple.

The default `paired_counterfactual` construction keeps clean and poisoned trajectories matched in training length, optimizer-step count, and task-content order. `replace` is the explicit in-place source-replacement alternative. `uniform_optimizer_steps` is the default deterministic poison-exposure schedule; `trainer_random` uses the Trainer random shuffle.

Before training, a fraction-zero neutrality check rejects a marker that is already strongly target-directing according to the configured trigger-lift/change/suppression thresholds.

## Checkpoint causal discovery

For a completed run:

```bash
POISONING_TASK=grammar \
RUN_DIR=../data/poisoning/grammar/<run-name> \
bash poisoning/scripts/run_backdoor_lift_overtopping.sh
```

The driver verifies `run_config.json` and `checkpoint_manifest_all.csv`, verifies or reconstructs the deterministic causal cohort, reads marker/target identity from the saved run, scans paired control/trigger behavior, plans CHA from available positives, and invokes the standard pipeline on the held-out baseline-positive population.

Grammar defaults to input+output intervention; arithmetic defaults to output-only. `PIPELINE_DECODE_ONLY` can override the phase.

Shared CHA defaults are:

```text
CHA_REFERENCE_N_PER_SIDE      64
CHA_TAU                       0.3
CHA_LOW_DATA_POLICY           skip
CHA_MIN_ACTUAL_N_PER_SIDE     16
CHA_PRUNE_ALPHA               0.05
CHA_MAX_N_PER_SIDE            64
REFINE_SAMPLING_MAX_POINTS    10000
TRIGGER_LIFT_SCAN_MAX_ROWS    10000
TRIGGER_LIFT_SCAN_CHUNK       2048
```

The ordinary-correctness endpoint is separate from trigger lift and can remain analyzable at checkpoints with insufficient trigger-lift positives.

## Cumulative suppression and specificity

```bash
POISONING_TASK=grammar \
RUN_DIR=../data/poisoning/grammar/<run-name> \
bash poisoning/scripts/run_backdoor_lift_cumulative_ablation.sh
```

The cumulative analysis uses discovery-frozen channel rankings, compares cumulative coalitions with structurally matched random noncandidate controls, evaluates ordinary control-marker behavior and task specificity, and reserves confirmation data for the final interaction-aware coalition evaluation.

## Training-time protection

Training-time protection is a separate workflow:

```bash
bash poisoning/scripts/run_training_time_protection.sh
```

It protects channel write coordinates derived from a virgin-model agonist set and compares protected, random-protected, and baseline poisoning trajectories under matched training definitions.

## Run layout

```text
data/poisoning/<task>/<run>/
├── 01_training_checkpoints/
├── 02_evaluation_cohorts/
├── 03_checkpoint_causal_discovery/
├── 04_condition_comparisons/
├── 05_behavior_trajectories/
└── 06_circuit_overlap_analysis/
```

Cross-run summaries are under `data/poisoning/summary/`. Regenerable causal-discovery caches are under `cache/poisoning/`. Final poisoning figures are under `results/poisoning/figures/`.

A nonempty run directory without canonical training metadata is never modified automatically. Use a new run name or explicitly relocate the conflicting directory.

## Detailed documentation

- [`../docs/poisoning-overview.md`](../docs/poisoning-overview.md) — questions, markers, behavioral definitions, workflow, and provenance.
- [`../docs/poisoning-protocol.md`](../docs/poisoning-protocol.md) — matched training, causal discovery, specificity, cumulative suppression, and seed aggregation.
- [`../docs/poisoning-configuration.md`](../docs/poisoning-configuration.md) — entry points and configuration variables.
- [`../docs/poisoning-outputs.md`](../docs/poisoning-outputs.md) — run layout, caches, summaries, and resume policy.
- [`../docs/interpretation-and-limitations.md`](../docs/interpretation-and-limitations.md) — interpretation boundaries.
- [`../docs/troubleshooting.md`](../docs/troubleshooting.md) — validation and failure modes.
