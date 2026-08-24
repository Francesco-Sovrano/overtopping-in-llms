# Causal channel intervention pipeline

This repository studies causal channel interventions in autoregressive language models. It contains two related workflows:

1. a standard non-poisoning programme that discovers and evaluates causally important model channels across arithmetic, grammatical acceptability, HANS NLI, random finite-state-machine, and jailbreak tasks; and
2. a checkpointed trigger-poisoning programme that trains matched clean and poisoned trajectories, measures trigger-lift behavior over training, localizes causal channels, and evaluates specificity and suppression controls.

The repository separates persistent experiment artifacts (`data/`), regenerable caches (`cache/`), and final aggregate outputs (`results/`). The implementation is under `code/`.

## Quick start

The supported environment is Python 3.12.

```bash
bash setup.sh
source .env/bin/activate
```

Inspect the standard experiment catalogue without running models:

```bash
./run_experiments.sh --list
```

Preview the selected commands:

```bash
./run_experiments.sh --suite paper-primary --dry-run
```

Run the full non-poisoning catalogue:

```bash
./run_experiments.sh
```

Regenerate aggregate tables, figures, audits, and reports from existing `data/` artifacts:

```bash
./generate_results.sh
```

Preview the checkpoint-poisoning matrix:

```bash
./run_poisoning_experiments.sh --dry-run
```

Run the checkpoint-poisoning matrix:

```bash
./run_poisoning_experiments.sh
```

Model-backed experiments can be expensive. Use catalogue listing, `--dry-run`, and the model-free validation commands in [Validation](#validation) before launching long runs.

## Repository layout

```text
<repo>/
├── code/
│   ├── analysis/       aggregate statistics, audits, tables, figures, reports
│   ├── docs/           detailed documentation
│   ├── experiments/    explicit standard experiment catalogue and execution
│   ├── lib/            shared task, model, intervention, statistics, and EAP code
│   ├── pipeline/       numbered non-poisoning pipeline stages 1–7
│   └── poisoning/      checkpoint poisoning tasks, stages, libraries, and drivers
├── data/               persistent experiment and run artifacts
├── cache/              regenerable caches
├── results/            final aggregate/manuscript outputs
├── logs/               optional runtime logs
├── .env/               virtual environment created by setup.sh
├── run_experiments.sh
├── run_poisoning_experiments.sh
├── generate_results.sh
├── setup.sh
└── requirements.txt
```

`code/` contains the import roots `analysis`, `experiments`, `lib`, `pipeline`, and `poisoning`. Direct module commands therefore run from `code/`:

```bash
cd code
python3 -m experiments.run_experiments --help
python3 -m analysis.generate_final_results --help
python3 -m poisoning.tasks.grammar --help
python3 -m poisoning.tasks.arithmetic --help
```

The root shell launchers change into `code/` automatically where necessary.

## Installation and runtime requirements

`setup.sh` creates `.env` with `python3.12`, upgrades packaging tools, and installs `requirements.txt`. If Ollama is installed, it also pulls the default feature-proposal models `gemma3:27b` and `qwen3:4b`.

CUDA is recommended for circuit discovery and model interventions. Apple Silicon MPS is supported by the model/intervention layer and is usable for the default 1.5B poisoning workflow, although some operations may be slower or fall back to CPU. Aggregation and report generation can run on CPU once the required artifacts exist.

Hugging Face cache locations can be controlled with standard variables such as:

```bash
export HF_HOME=/path/to/hf-cache
export TRANSFORMERS_CACHE=/path/to/hf-cache
```

Credentials for optional external services must be supplied through environment variables or an external secret manager. Do not store API keys in repository scripts or documentation.

## Standard tasks

The non-poisoning task interface is defined in `code/lib/task_spec.py`. Standard task modules are under `code/lib/tasks/`.

| Task identifier | Module | Purpose |
|---|---|---|
| `arithmetic` | `lib.tasks.arithmetic_task` | deterministic arithmetic generation and correctness |
| `grammar_acceptability` | `lib.tasks.grammar_acceptability_task` | grammatical acceptability classification |
| `hans_nli` | `lib.tasks.hans_nli_task` | HANS natural-language inference |
| `random_fsm` | `lib.tasks.random_fsm_task` | generated finite-state-machine problems |
| `bon_jailbreaking` | `lib.tasks.bon_jailbreaking_task` | jailbreak/safety classification workflow |

Common dataset/task variables include:

```text
GRAMMAR_DATASET_PATH       data/grammar_acceptability/cola_in_domain_train.jsonl
GRAMMAR_NUM_EXAMPLES       4096
GRAMMAR_TASK_SEED          42

HANS_SPLIT                 validation
HANS_NUM_EXAMPLES          1024
HANS_TASK_SEED             42
HANS_BALANCE_LABELS        1
HANS_CACHE_DIR             cache/hans
HANS_LOCAL_FILE            optional local file

FSM_NUM_EXAMPLES           2048
FSM_TASK_SEED              42
FSM_MIN_STATES             3
FSM_MAX_STATES             6
FSM_MIN_INPUT_LEN          8
FSM_MAX_INPUT_LEN          24

AUGMENTED_PROMPTS_FILE     data/bon_jailbreaking/dataset.json
```

Task modules own prompt construction, target columns, output parsing, and task-specific statistics. Shared pipeline code consumes those capabilities through the common task interface.

## Standard experiment catalogue

`code/experiments/run_experiments.py` defines an explicit, paper-centered catalogue rather than a Cartesian product. It contains exactly **39 unique configurations**:

- `paper-primary`: 28 primary task/model/phase configurations;
- `paper-auxiliary`: 11 targeted comparison configurations;
- `all`: the union of both suites.

The primary count and auxiliary count are asserted in code so accidental catalogue drift fails immediately.

List the live catalogue:

```bash
./run_experiments.sh --suite paper-primary --list
./run_experiments.sh --suite paper-auxiliary --list
./run_experiments.sh --suite all --list
```

Useful selection options are:

```text
--suite {paper-primary,paper-auxiliary,all}
--phase {all,pipeline,analysis}
--task CSV
--model CSV
--intervention CSV
--mode CSV
--evaluation-split {test,train,all}
--data-root PATH
--results-root PATH
--list
--dry-run
--continue-on-error
--generate-primary-manuscript
--primary-profile iclr-28
```

`run_experiments.sh` defaults to `--suite all` and `--evaluation-split test`. When the evaluation split is `test`, the launcher also requests primary manuscript generation with the fixed `iclr-28` profile. For `train` or `all`, manuscript export is skipped because the primary manuscript outputs are defined on the held-out test split.

Each selected `RunSpec` is translated into one call to `code/pipeline/_run_pipeline.sh`. The expanded selected catalogue is written to `results/configured_experiments.json`, and pipeline failures are written to `results/pipeline_failures.json`.

## Numbered causal-intervention pipeline

The standard pipeline has seven stages:

1. **Generate prompts and answers** — materialize task examples and baseline model outputs.
2. **Generate features** — construct interpretable feature representations used by rule extraction.
3. **Extract rules** — fit symbolic/rule-based predictors over task and model behavior.
4. **Construct a sampling plan** — choose representative evaluation points, optionally with spectral sampling.
5. **Discover circuits** — identify candidate causal channels under the configured intervention.
6. **Analyze the rule/circuit set** — classify and rank candidate channels.
7. **Refine and evaluate singleton channels** — evaluate selected channels on the requested evaluation population, emit exact singleton metrics, and optionally run simultaneous/conditional validation.

The wrapper can also run a custom configuration directly:

```bash
cd code
bash pipeline/_run_pipeline.sh \
  grammar_acceptability \
  Qwen/Qwen2.5-1.5B-Instruct \
  --spectral_splits \
  --fast_anchoring \
  --eval_intervention mean-donor \
  --evaluation_split test
```

For manuscript configurations, prefer the experiment catalogue because it keeps task, model, phase, intervention, circuit size, thresholds, and evaluation settings together in one `RunSpec`.

### Evaluation split

The catalogue-level option is `--evaluation-split`; the internal pipeline wrapper uses `--evaluation_split`. Supported values are:

- `test`: held-out evaluation rows;
- `train`: non-held-out rows;
- `all`: all materialized rows.

Stage 7 requires an `is_test` column for `test` and `train`. Evaluation-split identity is encoded in result provenance so statistics from different populations are not silently combined.

### Intervention phases

The catalogue uses two modes:

- `standard`: intervention may affect prompt processing and generation;
- `decode-only`: intervention is restricted to generation-time positions.

Replacement baselines include `mean`, `mean-donor`, and `mean-positional`. The configured baseline is part of run identity and must match when results are compared or caches are reused.

## Core singleton and set metrics

Stage 7 evaluates individual candidate channels on a common evaluation population. The exact metric sidecars distinguish singleton behavior from simultaneous-set behavior.

For candidate set `J`:

- `J` is the frozen candidate set;
- `U_J` is the exact union of rows affected by at least one singleton in `J`;
- `s_1` is the strongest singleton effect under the configured criterion;
- `N_t` is the thresholded count of qualifying singleton candidates;
- `E(J)` is the effect of intervening on the full set simultaneously;
- `CMC_1x` is the paired conditional marginal contribution of `J` against a matched background set at multiplier 1.

The analysis layer keeps these quantities separate. A singleton union is not substituted for a missing simultaneous intervention, and missing conditional effects remain missing unless their exact model-backed sidecars exist.

## Conditional simultaneous-set validation

`analysis.validate_interactions` evaluates the simultaneous candidate set and matched comparison sets. It records candidate identity, evaluation-row identity, intervention configuration, matching design, random seed, and input fingerprints in its cache metadata.

Typical direct use:

```bash
cd code
python3 -m analysis.validate_interactions --help
```

Conditional marginal contribution is computed by comparing the effect of adding the candidate set to matched background sets. Monte Carlo precision depends on `--null_draws`; small draw counts are useful for smoke tests but produce coarse p-values.

## Final analysis and manuscript outputs

When model-run artifacts already exist under `data/`, regenerate final outputs with:

```bash
./generate_results.sh
```

Equivalent direct invocation:

```bash
cd code
python3 -m analysis.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --poisoning-root ../data/poisoning \
  --primary-profile iclr-28 \
  --require-complete-new-metrics
```

The only supported primary profile is `iclr-28`. It requires exactly 28 rows, including exactly one Qwen2-1.5B input+output HANS NLI row. Any different primary row count or identity is rejected.

`generate_results.sh` requires the exact manuscript metrics by default. Set `ALLOW_INCOMPLETE_NEW_METRICS=1` only when a partial report is intentional; missing values are reported by the completeness audit rather than replaced by proxies.

Final outputs use descriptive directories:

```text
results/
├── experiment_catalogue/
├── primary_analysis/
│   ├── tables/
│   ├── statistics/
│   └── metric_completeness_audit/
├── manuscript/
│   ├── tables_and_macros/
│   └── figures/
├── diagnostics/
│   └── overtopping_spiking/
├── poisoning/
│   └── figures/
└── final_results_manifest.json
```

The orchestrator can skip paper figures, the spiking report, poisoning reporting, or the CMC completeness requirement through its documented CLI flags.

## Checkpointed trigger-poisoning study

The poisoning workflow studies a marker-triggered target behavior over matched clean and poisoned fine-tuning trajectories. Grammar and arithmetic are implemented by `poisoning.tasks.grammar` and `poisoning.tasks.arithmetic`; generic stages dispatch through `poisoning.tasks.registry`.

Every experimental prompt starts with exactly one raw marker line. The repository defaults are:

```text
CONTROL_MARKER  " "          # one space
TRIGGER_MARKER  [id=74291]
SHAM_MARKER     "  "         # two spaces
```

Marker values are experiment-defined opaque strings. They may be IDs, ordinary text, empty strings, tabs, or whitespace-only strings. Each marker must fit on one line and the control, trigger, and sham values must be distinct. Marker handling is exact: the code does not strip, trim, case-fold, coerce, replace, or otherwise normalize marker values. Leading and trailing whitespace are semantically significant and are preserved character-for-character. The task content below the first line is identical across conditions, and marker tokenization is recorded as a diagnostic fingerprint.

For a matched pair, let `c(x)` be the control-marked prompt, `t(x)` the trigger-marked prompt, and `T` the attacker target. The main behavioral event is trigger lift:

```text
M(c(x)) != T  and  M(t(x)) = T
```

The workflow reports both:

```text
trigger_lift_rate = P(M(c(x)) != T and M(t(x)) = T)
conditional_conversion = P(M(t(x)) = T | M(c(x)) != T)
```

The second quantity separates conversion reliability from the fraction of examples that are already target-positive in the control condition.

### Root poisoning matrix

`run_poisoning_experiments.sh` defaults to:

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

`MODEL_NAMES` can override the model list for both tasks; task-specific model variables override each task separately. `POISONING_FAST_TEST=1` selects a small behavior-first smoke configuration and skips the expensive defense loop.

Preview the matrix and expanded shell commands:

```bash
./run_poisoning_experiments.sh --dry-run
```

The normal matrix performs matched training, checkpoint causal discovery, ordinary-correctness controls, cumulative suppression/specificity analysis, and cross-seed aggregation.

### Direct poisoning entry points

From `code/`:

```bash
python3 -m poisoning.tasks.grammar --help
python3 -m poisoning.tasks.arithmetic --help
```

The shared checkpoint-training driver is:

```bash
POISONING_TASK=grammar DRY_RUN=1 bash poisoning/scripts/run_checkpoint_ft.sh
```

Its direct defaults are intentionally lower-level than the root matrix: `POISON_RATE=0.03`, `POISON_RATE_BASIS=total_train`, seed `13`, the configured marker triple, one training epoch, and checkpoint fractions `0,0.1,0.25,0.5,0.75,1.0`.

For a completed run, checkpoint causal discovery is started with:

```bash
POISONING_TASK=grammar \
RUN_DIR=../data/poisoning/grammar/<run-name> \
bash poisoning/scripts/run_backdoor_lift_overtopping.sh
```

Inference-time cumulative suppression is started with:

```bash
POISONING_TASK=grammar \
RUN_DIR=../data/poisoning/grammar/<run-name> \
bash poisoning/scripts/run_backdoor_lift_cumulative_ablation.sh
```

Training-time channel-write protection is a separate workflow:

```bash
bash poisoning/scripts/run_training_time_protection.sh
```

### Poisoning run layout

A run is organized by stage:

```text
data/poisoning/<task>/<run>/
├── 01_training_checkpoints/
│   ├── clean/
│   ├── poisoned/
│   └── metadata/
├── 02_evaluation_cohorts/
├── 03_checkpoint_causal_discovery/
├── 04_condition_comparisons/
├── 05_behavior_trajectories/
└── 06_circuit_overlap_analysis/
```

Regenerable discovery caches live under:

```text
cache/poisoning/<task>/<run>/checkpoint_causal_discovery/<phase>/adaptive_circuit_discovery/
```

Cross-run poisoning summaries live under `data/poisoning/summary/`; manuscript-facing poisoning figures live under `results/poisoning/figures/`.

A nonempty run directory that does not contain the canonical training metadata is never modified automatically. Choose a new `POISONING_RUN_NAME`, or explicitly move/remove the conflicting directory after inspecting it.

### Causal-discovery defaults

The shared trigger-lift/CHA defaults include:

```text
CHA_REFERENCE_N_PER_SIDE       64
CHA_TAU                        0.3
CHA_LOW_DATA_POLICY            skip
CHA_MIN_ACTUAL_N_PER_SIDE      16
CHA_PRUNE_ALPHA                0.05
CHA_MAX_N_PER_SIDE             64
REFINE_SAMPLING_MAX_POINTS     10000
TRIGGER_LIFT_SCAN_MAX_ROWS     10000
TRIGGER_LIFT_SCAN_CHUNK        2048
TRIGGER_LIFT_SCAN_MIN_ROWS     0
TRIGGER_LIFT_SCAN_EARLY_STOP   0
```

`DISCOVERY_CACHE_ROOT` overrides the per-discovery cache location. `POISONING_CACHE_ROOT` changes the top-level poisoning cache root.

## Caches, resuming, and provenance

Persistent scientific outputs belong in `data/`; caches that can be regenerated belong in `cache/`. Cache reuse is guarded by semantic/content fingerprints that include the configuration needed to identify a compatible computation.

For poisoning training, `run_config.json` and checkpoint manifests define run identity. A saved run cannot be resumed under a different model, seed, dataset, marker triple, target, optimizer, LoRA configuration, or other training-defining field. Checkpoint manifests use run-local checkpoint identities, so moving an entire repository does not require absolute-path rewriting.

For standard Stage 7 and interaction validation, result sidecars record evaluation split, intervention phase, replacement baseline, candidate set, and data/configuration fingerprints. When exact inputs differ, cached statistics must not be treated as equivalent.

## Validation

These checks do not execute model inference:

```bash
python3 -m compileall -q code
bash -n run_experiments.sh
bash -n generate_results.sh
bash -n run_poisoning_experiments.sh
bash -n setup.sh
bash -n code/pipeline/_run_pipeline.sh
find code/poisoning/scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n

cd code
python3 -m experiments.run_experiments --suite paper-primary --list
python3 -m experiments.run_experiments --suite paper-auxiliary --list
python3 -m analysis.generate_final_results --help
python3 -m poisoning.tasks.grammar --help
python3 -m poisoning.tasks.arithmetic --help
```

Expected catalogue totals are 28 primary and 11 auxiliary configurations. Syntax and CLI validation do not prove scientific correctness; full validation additionally requires the intended datasets, model revisions, cached inputs, and provenance-consistent result artifacts.

## Troubleshooting

**Repository packages cannot be imported.** Run direct Python modules from `code/`, or use the root shell launchers.

**A required dataset file is missing.** Set the task-specific path variable or provide the expected file under `data/`. HANS can use `HANS_LOCAL_FILE`; grammar uses `GRAMMAR_DATASET_PATH`.

**Model files are going to the wrong disk.** Set `HF_HOME`, `TRANSFORMERS_CACHE`, or the poisoning-specific `HF_MODEL_CACHE_DIR` before model loading.

**Final-results generation reports incomplete metrics.** Inspect `results/primary_analysis/metric_completeness_audit/`. Missing simultaneous or conditional intervention metrics cannot be reconstructed from singleton unions.

**A poisoning run directory is rejected.** The directory is nonempty but does not have the canonical training-stage metadata. Use a new `POISONING_RUN_NAME` or explicitly relocate the conflicting directory.

**Trigger-lift causal discovery is skipped.** Check whether the configured cohort contains enough trigger-lift positives for the CHA side-size requirements and low-data policy. Ordinary-correctness analysis is a separate endpoint and can still be available.

**GPU memory is insufficient.** Reduce batch sizes, use a smaller model, limit evaluation/sampling caps for a smoke run, or use the poisoning fast-test mode before attempting the full matrix.

## Detailed documentation

The documentation set under `code/docs/` is organized for first-time readers:

- `index.md` — documentation map and entry points;
- `getting-started.md` — installation, validation, and first commands;
- `repository-layout.md` — package and artifact ownership;
- `concepts.md` — causal-intervention terminology and provenance;
- `experiments.md` — the 39-run standard catalogue;
- `pipeline.md` — numbered pipeline stages and controls;
- `analysis.md` — final-results orchestration and metric provenance;
- `eap.md` — internal EAP/EAP-IG implementation;
- `poisoning-overview.md` — poisoning questions, endpoints, and workflow;
- `poisoning-protocol.md` — matched training and causal-analysis protocol;
- `poisoning-configuration.md` — poisoning entry points and configuration variables;
- `poisoning-outputs.md` — poisoning filesystem layout and artifact policy;
- `interpretation-and-limitations.md` — interpretation boundaries;
- `troubleshooting.md` — validation and failure modes.
