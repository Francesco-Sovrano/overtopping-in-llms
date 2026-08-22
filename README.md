# Causal channel intervention pipeline

This repository runs causal channel-intervention experiments on autoregressive language models, evaluates frozen candidate channels on a configurable evaluation split, performs simultaneous-set validation, and generates publication-ready tables and figures.

The standard workflow is intentionally split into three user-facing entry points:

```bash
./run_experiments.sh
./generate_results.sh
./run_poisoning_experiments.sh
```

- `run_experiments.sh` runs the non-poisoning experiment catalogue.
- `generate_results.sh` regenerates aggregate statistics, manuscript tables, and figures from artifacts already present under `data/`.
- `run_poisoning_experiments.sh` runs the complete grammar-and-arithmetic checkpoint-poisoning suite.

Implementation code lives under `code/`. Runtime state stays at repository root: models and task artifacts under `data/`, reusable caches under `cache/`, the virtual environment under `.env/`, run logs under `logs/`, and final paper outputs under `results/`.

---

## 1. Repository layout

```text
<repo>/
├── code/
│   ├── analysis/       # statistics, simultaneous validation, tables and figures
│   ├── experiments/    # executable experiment catalogue and command construction
│   ├── lib/            # shared model, task, attribution and intervention code
│   ├── pipeline/       # numbered stages 1–7 and per-run shell orchestrator
│   ├── poisoning/      # checkpoint poisoning experiments and shell launchers
├── data/               # experiment outputs and local task datasets
├── cache/              # prompt/model/task caches
├── results/            # final aggregate statistics and paper outputs
├── logs/               # optional run logs
├── .env/               # Python virtual environment created by setup.sh
├── run_experiments.sh
├── run_poisoning_experiments.sh
├── generate_results.sh
├── setup.sh
└── requirements.txt
```

`code/` is not a Python package. Internal modules are run from that directory with normal package execution, for example:

```bash
cd code
python3 -m experiments.run_experiments --help
python3 -m analysis.validate_interactions --help
```

The root shell launchers perform this directory change automatically.

---

## 2. Installation

The environment targets Python 3.12.

```bash
bash setup.sh
source .env/bin/activate
```

`setup.sh`:

1. creates `.env/` at repository root;
2. upgrades `pip`, `setuptools`, and `wheel`;
3. installs `requirements.txt`;
4. pulls the default Ollama feature-proposal models when the `ollama` executable is available.

Important packages include PyTorch, Transformers, TransformerLens, pandas, SciPy, scikit-learn, SHAP, XGBoost, Sentence Transformers, Ollama, Groq, and OpenAI clients.

CUDA is the recommended accelerator for circuit discovery and intervention runs. Apple Silicon MPS is also supported by the model/intervention layer and is suitable for the default 1.5B poisoning workflow, although it is generally slower and some TransformerLens operations may fall back to CPU if the MPS compatibility patch cannot be applied. Model-free aggregation and paper-output generation can run on CPU once the required result files exist.

### Hugging Face caches

Standard Hugging Face environment variables are supported, for example:

```bash
export HF_HOME=/path/to/hf-cache
export TRANSFORMERS_CACHE=/path/to/hf-cache
```

### External feature/classifier services

Credentials must be provided through environment variables or an external secret manager. Do not place credentials in repository files.

Feature generation can be run without Ollama-proposed features by passing `--no_llm_feature_generation` to the direct pipeline wrapper or setting the corresponding `RunSpec` field.

---

## 3. Standard tasks

The task interface is defined by `code/lib/task_spec.py`. Standard task modules are under `code/lib/tasks/`.

| Task identifier | Module | Main configuration |
|---|---|---|
| `arithmetic` | `lib.tasks.arithmetic_task` | deterministic arithmetic prompt generation |
| `grammar_acceptability` | `lib.tasks.grammar_acceptability_task` | local CoLA-style JSONL file |
| `hans_nli` | `lib.tasks.hans_nli_task` | HANS train/validation file or download |
| `random_fsm` | `lib.tasks.random_fsm_task` | generated finite-state-machine problems |
| `bon_jailbreaking` | `lib.tasks.bon_jailbreaking_task` | local augmented prompt dataset plus classifier |

### Grammar environment variables

```text
GRAMMAR_DATASET_PATH       default: data/grammar_acceptability/cola_in_domain_train.jsonl
GRAMMAR_NUM_EXAMPLES       default: 4096; must be even for balanced sampling
GRAMMAR_TASK_SEED          default: 42
```

### HANS environment variables

```text
HANS_SPLIT                 default: validation
HANS_NUM_EXAMPLES          default: 1024
HANS_TASK_SEED             default: 42
HANS_BALANCE_LABELS        default: 1
HANS_MAX_NEW_TOKENS        default: 3
HANS_CACHE_DIR             default: cache/hans
HANS_LOCAL_FILE            optional local HANS file
```

### Random-FSM environment variables

```text
FSM_NUM_EXAMPLES           default: 2048
FSM_TASK_SEED              default: 42
FSM_MIN_STATES             default: 3
FSM_MAX_STATES             default: 6
FSM_MIN_INPUT_LEN          default: 8
FSM_MAX_INPUT_LEN          default: 24
```

### Jailbreak environment variables

```text
AUGMENTED_PROMPTS_FILE                  default: data/bon_jailbreaking/dataset.json
BON_JAILBREAK_CLASSIFIER_MODEL          default: qwen/qwen3-32b
BON_JAILBREAK_CLASSIFIER_FALLBACK_MODEL default: qwen/qwen3.6-27b
```

---

## 4. Running the standard experiment programme

The main entry point is:

```bash
./run_experiments.sh
```

It runs only non-poisoning experiments. Poisoning is never submitted by this launcher.

The executable catalogue is defined in `code/experiments/run_experiments.py` and is paper-centered rather than factorial. It contains **36 unique configurations**:

- **28 `paper-primary` configurations**, exactly matching the manuscript's primary Appendix Table 8 settings;
- **8 `paper-auxiliary` configurations** for targeted baseline and phase comparisons.

Inspect the expanded catalogue without running anything:

```bash
cd code
python3 -m experiments.run_experiments --list
```

or from repository root:

```bash
./run_experiments.sh --list
```

### Suites

```bash
./run_experiments.sh --suite paper-primary
./run_experiments.sh --suite paper-auxiliary
./run_experiments.sh --suite all
```

`paper-primary` runs only the exact 28 primary settings. `all` is the root-launcher default and adds the eight targeted auxiliary controls.

### Filters

Filters accept comma-separated exact identifiers:

```bash
./run_experiments.sh \
  --task arithmetic,hans_nli \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --intervention mean-donor \
  --mode standard,decode-only
```

### Pipeline versus analysis phases

```bash
./run_experiments.sh --phase pipeline
./run_experiments.sh --phase analysis
./run_experiments.sh --phase all
```

- `pipeline` runs configured stage-1–7 experiments and their per-run interaction validation.
- `analysis` skips stage execution and processes artifacts already present under `data/`.
- `all` runs the pipeline and then aggregate analysis.

Use `--dry-run` to print experiment commands without running them:

```bash
./run_experiments.sh --dry-run
```

Use `--continue-on-error` when running a batch of configurations and you want later configurations to continue after one pipeline failure.

---

## 5. Evaluation split

`test` is the default evaluation split, but it is not forced. All standard model-facing evaluation components accept:

```text
test
train
all
```

Examples:

```bash
./run_experiments.sh --evaluation-split test
./run_experiments.sh --evaluation-split train
./run_experiments.sh --evaluation-split all
```

or:

```bash
EVALUATION_SPLIT=train ./run_experiments.sh
```

Output labels are split-specific:

```text
test   -> ...-heldout_test
train  -> ...-eval_train
all    -> unsuffixed stats path
```

Primary manuscript profiles are defined on the `test` split. The root launcher therefore generates primary manuscript outputs only when the selected evaluation split is `test`.

---

## 6. Experiment catalogue

The catalogue is explicit: it does not cross every model, task, phase, and intervention baseline. This keeps the main launcher aligned with the reported experiment matrix and avoids spending days on cells that do not support a defined paper comparison.

### `paper-primary`: 28 manuscript settings

The primary suite contains the exact 28 model–task–phase settings used by the manuscript's primary table. Small-model primary runs use the selected `mean-donor` or `mean` baseline from the paper; the four large-model scale settings use `mean-positional` and are MLP-only.

Primary counts by task are:

| Task | Primary runs |
|---|---:|
| arithmetic | 7 |
| bon_jailbreaking | 3 |
| grammar_acceptability | 8 |
| hans_nli | 4 |
| random_fsm | 6 |
| **total** | **28** |

The primary set includes Pythia-1B@48k arithmetic output-only and Qwen2-1.5B jailbreak output-only. It excludes large-model Pythia NLI/jailbreak cells and broad zero-baseline crossings that are not part of the primary manuscript matrix.

Default small-model parameters are:

```text
batch_size       32
circuit_level    neuron
circuit_size     200000
min_flip_rate    0.3
max_circuits     1
evaluation_split test
```

Qwen arithmetic uses `z_thresh=10`; Pythia arithmetic uses `z_thresh=5`. The large-model settings use `circuit_size=100000`, `min_flip_rate=0.2`, `mean-positional`, and MLP-only interventions.

### `paper-auxiliary`: 8 targeted controls

Six runs complete the paper's mean-versus-mean-donor sensitivity comparisons:

```text
Qwen2-1.5B arithmetic Out                 mean
Qwen2.5-1.5B grammar I+O                 mean
Qwen2.5-1.5B grammar Out                 mean
Qwen2.5-1.5B HANS NLI I+O                mean
Qwen2.5-1.5B HANS NLI Out                mean
Qwen2.5-1.5B random FSM I+O              mean
```

Two additional phase diagnostics are included because they directly clarify primary results without recreating a full factorial sweep:

```text
Qwen2.5-1.5B arithmetic I+O              mean-donor
Qwen2-1.5B HANS NLI Out                  mean-donor
```

There are no automatic `zero`-baseline runs in the main catalogue. See `code/experiments/README.md` for the full 28-row primary listing and path-label rules.

---

## 7. Numbered pipeline stages

The per-configuration shell orchestrator is:

```text
code/pipeline/_run_pipeline.sh
```

It coordinates the seven numbered stages below.

### Stage 1 — generate prompts and answers

`code/pipeline/1_generate_prompts_and_answers.py`

Creates or loads task examples, runs the analyzed model, parses predictions, computes task correctness/behavior labels, and writes reusable prompt/model-I/O caches.

### Stage 2 — generate features

`code/pipeline/2_generate_features.py`

Builds task-provided and optionally LLM-proposed interpretable features, scores the examples, filters weak/redundant features, and writes the feature table used downstream.

### Stage 3 — extract symbolic rules

`code/pipeline/3_extract_rules.py`

Extracts feature-based association rules and rule combinations. This stage is not used when `--spectral_splits` selects spectral-cluster slicing.

### Stage 4 — construct sampling plans

`code/pipeline/4_spectral_sample_datapoints.py`

Builds sampling plans used by circuit discovery. It is skipped in spectral-split mode.

### Stage 5 — discover circuits

`code/pipeline/5_discover_circuits.py`

Runs EAP/EAP-IG attribution and writes the eligible stage-5 circuit/channel population and associated manifests.

### Stage 6 — select frozen candidates

`code/pipeline/6_analyze_bag_of_rules.py`

Evaluates candidate channel effects on discovery/training data and writes the selected candidate set. The discovery ranking used by TOC is frozen from this stage rather than re-ranked by evaluation-set singleton effects.

### Stage 7 — evaluate singleton interventions

`code/pipeline/7_refine_neuron_anchored_rules.py`

Suppresses each frozen candidate individually on the selected evaluation split, materializes per-example singleton flip events, writes standard flip summaries, and computes the exact singleton-set metrics described below.

### Conditional simultaneous-set validation

After stage 7, `_run_pipeline.sh` invokes:

```text
code/analysis/validate_interactions.py
```

when `RUN_INTERACTION_VALIDATION=true` and the required stage-5/stage-7 artifacts are present. This always evaluates the full candidate set and matched controls using genuine simultaneous interventions. Conditional marginal contribution (CMC) is enabled by default and can be disabled independently with `RUN_CMC=false`; disabling CMC does not disable `E(J)` or its matched-null comparison.

---

## 8. Intervention phases and replacement baselines

### Phase

Two standard intervention phases are represented in the catalogue:

- `standard` / `I+O`: intervention applies over input and output positions supported by the pipeline;
- `decode-only` / `Out`: intervention is restricted to answer-generation positions.

### Replacement baselines

Supported intervention names are:

```text
zero
mean
mean-positional
mean-donor
mean-donor-positional
```

The implementation is centralized in `code/lib/modeling_and_ablation.py`. `mean-donor` uses the repository's donor-based replacement implementation and is shared by singleton and simultaneous intervention paths.

The direct pipeline wrapper defaults to `mean-positional`; catalogue runs explicitly pass their configured intervention.

For mean-family replacement construction, the pipeline uses training/reference rows rather than the evaluation rows. Donor-style standard runs default to a larger reference pool than ordinary mean replacement.

---

## 9. Singleton-set statistics

Let `J` be the frozen candidate set. For evaluation example `x`, let `F_j` be the event that suppressing only candidate channel `j` flips the binary task predicate.

Define:

```text
s_j   = P(F_j)
U(A)  = P(union_{j in A} F_j)
```

All singleton probabilities for a run are computed on one common complete-case evaluation universe. If a materialized singleton column is missing for some rows, those rows are excluded from every singleton-set probability for that run and the number excluded is recorded.

### Strongest singleton

```text
s_(1) = max_j s_j
```

### Frozen top-m concentration

`H_m` is the top-`m` subset according to the candidate ranking frozen on discovery data.

```text
TOC_m(J) = U(H_m) / U(J)
```

Evaluation-set singleton effects are not used to choose `H_m`.

### Overlap redundancy

```text
R_ov(J) = 1 - U(J) / sum_j s_j
```

The value is not clipped.

### Effective number of contributing candidates

```text
N_eff(J) = (sum_j s_j)^2 / sum_j s_j^2
```

### Baseline-conditional union coverage

For the unablated binary predicate `B(x)`:

```text
OCC_b(J) = P(union_{j in J} F_j | B(x)=b),  b in {0,1}
```

`OCC_0` and `OCC_1` are computed from the exact per-example singleton-union event, not from directional aggregate ratios.

### Threshold counts

For threshold `t`:

```text
N_t = |{j : s_j >= t}|
```

Default thresholds are:

```text
0.01, 0.05, 0.10, 0.20, 0.30
```

### Singleton metric outputs

Stage 7 writes, among other files:

```text
flip_stats_global.json
flip_stats_by_neuron.csv
scores.csv
frozen_candidate_ranking.csv
singleton_set_metrics.json
singleton_set_metrics.csv
singleton_topm_metrics.csv
singleton_threshold_counts.csv
```

The metric sidecar schema is:

```text
heldout-set-metrics-v2
```

---

## 10. Simultaneous full-set effect

The simultaneous candidate effect is a separate causal object from the union of singleton flips.

Let `E(A)` be the fraction of evaluation examples whose binary predicate flips when all channels in set `A` are suppressed **simultaneously**.

The validator computes:

```text
E(J)
```

by one genuine simultaneous intervention on the entire frozen candidate set.

It then samples structurally matched noncandidate sets `K_b` and computes genuine simultaneous `E(K_b)` for every draw.

For the unconditional candidate-vs-null comparison:

```text
Delta = E(J) - median_b E(K_b)

P = (1 + sum_b 1{E(K_b) <= E(J)}) / (B + 1)

p_MC = (1 + sum_b 1{E(K_b) >= E(J)}) / (B + 1)
```

No singleton union is substituted for any `E(...)` quantity.

---

## 11. Paired conditional marginal contribution

The conditional analysis asks whether candidates contribute more causal effect than comparable noncandidate channels in the **same perturbed context**.

For each matched draw `b`:

- `K_b` is a noncandidate set matched to `J`;
- `S_b` is a noncandidate background set;
- `S_b` is disjoint from both `J` and its paired `K_b`;
- candidate and null are evaluated on the same `S_b` and the same evaluation examples.

The validator performs three genuine simultaneous interventions:

```text
E(S_b)
E(S_b union J)
E(S_b union K_b)
```

and defines:

```text
M_b(J)   = E(S_b union J)   - E(S_b)
M_b(K_b) = E(S_b union K_b) - E(S_b)
D_b      = M_b(J)           - M_b(K_b)
```

For a background multiplier `q`, `S_b` contains `q` times the candidate cardinality in every exact matching stratum. The default multiplier is `1`, yielding the manuscript column `CMC_1x`.

The reported paired summary is:

```text
candidate       = mean_b M_b(J)
candidate_median= median_b M_b(J)
null_mean       = mean_b M_b(K_b)
median_null     = median_b M_b(K_b)
Delta           = mean_b D_b
Delta_median    = median_b D_b
P               = (1 + sum_b 1{D_b >= 0}) / (B + 1)
p_MC            = (1 + sum_b 1{D_b <= 0}) / (B + 1)
paired_win_rate = mean_b 1{D_b > 0}
```

`P` and `p_MC` here are paired Monte-Carlo sign/tail summaries. They are not the same calculation as the unconditional `E(J)` scalar-versus-null comparison.

### Matching constraints

`J`, `K_b`, and `S_b` are matched by the structural strata encoded by stage 5:

- transformer layer;
- computational locus;
- channel type;
- per-stratum cardinality.

The intervention phase and replacement baseline are fixed for the complete validation run, so candidate and control sets use the same phase and baseline.

Candidate units are not sampled into matched null/background pools.

### Enabling or disabling CMC

The simultaneous full-set validation `E(J)` and its matched-null distribution are independent of CMC. To skip the additional conditional-background interventions while retaining `E(J)`, run:

```bash
RUN_CMC=false ./run_experiments.sh
```

Equivalently:

```bash
export RUN_CMC=false
./run_experiments.sh
```

`RUN_CMC=true` is the default. When CMC is disabled, `interaction_validation_summary.json` records `cmc_enabled: false`; CMC-specific CSVs and paired-background plots are not produced. The standard final completeness audit also stops requiring CMC while continuing to require simultaneous `E(J)` and its matched-null comparison.

Direct module usage provides the equivalent `--skip_cmc` option.

### Background multipliers

The validator accepts comma-separated nonnegative integers when CMC is enabled:

```bash
CONDITIONAL_BACKGROUND_MULTIPLIERS=1 ./run_experiments.sh
```

Direct module usage:

```bash
cd code
python3 -m analysis.validate_interactions \
  ... \
  --background_multipliers 0,1,2
```

A `0` background reduces the conditional context to the empty set and is useful as a diagnostic; the normal paper default is `1`.

### Null draws

`analysis.validate_interactions` defaults to 100 null draws when invoked directly. The root `run_experiments.sh` defaults to:

```text
INTERACTION_NULL_DRAWS=30
```

and respects an explicit environment override. For publication-quality Monte-Carlo resolution, choose a larger value explicitly, for example:

```bash
INTERACTION_NULL_DRAWS=999 ./run_experiments.sh
```

The selected value is recorded in the interaction configuration/output files.

### Conditional validation outputs

Each stats directory contains:

```text
interaction_validation/
├── interaction_configuration.json
├── matched_control_strata.csv
├── matched_random_set_membership.csv
├── matched_null_draws.csv
├── conditional_background_membership.csv      # CMC only
├── conditional_marginal_draws.csv             # CMC only
├── conditional_marginal_summary.csv           # CMC only
├── interaction_validation_summary.csv
├── interaction_validation_summary.json
├── interaction_validation_summary.md
├── interaction_validation_table.tex
├── E_J_matched_null.pdf
└── conditional_marginal_<q>x_paired_delta.pdf # CMC only
```

The current interaction schema is:

```text
conditional-marginal-validation-v1
```

---

## 12. Cache reuse and resuming

The pipeline is designed to reuse materialized work when configuration and artifact identity match.

### Stage-7 statistics sidecars

If singleton intervention columns already exist in `scores.csv` but the exact singleton sidecars are absent, `_run_pipeline.sh` can call stage 7 in `--stats_only` mode. This reconstructs:

- frozen candidate ranking from discovery artifacts;
- `TOC_m`;
- `R_ov`;
- `N_eff`;
- `OCC_0` and `OCC_1`;
- threshold counts;

without repeating singleton model ablations.

### Simultaneous validation cache

The current conditional validator writes a semantic/content fingerprint containing the relevant task/model, split, phase, replacement baseline, candidate set, evaluation-row fingerprint, matching/background design, seed, and input fingerprints. A compatible cache is reused without model execution.

If an interaction directory contains exact simultaneous `E(J)` and matched-set information in the recognized v3 interaction schema, the validator can reuse compatible `E(J)`, direct-null `E(K_b)` values, and matched `K_b` memberships. The conditional background interventions still have to be evaluated if they are not present, because they cannot be reconstructed from singleton summaries.

### Compact result exports

A compact export may omit:

- `scores.csv`;
- model/intervention caches;
- stage-5 runtime manifests;
- large intermediate directories.

Such an export can still regenerate aggregate statistics that are mathematically determined by the retained summaries. It cannot create missing `TOC_m`, exact OCC, simultaneous `E(J)`, or conditional marginal effects unless their required exact sidecars are present.

The required-metric audit records availability rather than substituting proxies.

---

## 13. Primary manuscript profiles

Primary manuscript exports require an explicit profile:

```text
iclr-28
legacy-27
```

The only identity difference between these profiles is the Qwen2-1.5B input+output NLI setting:

```text
Task:  NLI
Model: Qwen2-1.5B
Phase: I+O
```

- `iclr-28` requires 28 rows and requires that setting exactly once.
- `legacy-27` requires 27 rows without that setting, or removes exactly that recognized setting from a valid 28-row table.

Any other row count or identity mismatch is an error. The profile is never inferred from row count alone.

The root experiment launcher defaults to:

```bash
PRIMARY_PROFILE=iclr-28 ./run_experiments.sh
```

Select the 27-setting profile explicitly with:

```bash
PRIMARY_PROFILE=legacy-27 ./run_experiments.sh
```

Primary manuscript generation is test-split specific.

---

## 14. Generating final paper outputs

When experiment artifacts already exist under `data/`, run:

```bash
./generate_results.sh
```

This does not run the numbered model pipeline. It orchestrates the aggregate/report scripts against the existing result tree.

Defaults:

```text
DATA_ROOT       <repo>/data
RESULTS_ROOT    <repo>/results
PRIMARY_PROFILE iclr-28
```

Override the data root:

```bash
DATA_ROOT=/path/to/data ./generate_results.sh
```

Select the primary profile:

```bash
PRIMARY_PROFILE=legacy-27 ./generate_results.sh
```

By default the command requires all exact manuscript metrics for every primary row. To generate a partial report when some additional exact metrics are unavailable:

```bash
ALLOW_INCOMPLETE_NEW_METRICS=1 ./generate_results.sh
```

Missing quantities remain unavailable; they are not replaced with singleton-union approximations.

### Final results tree

All final paper artifacts are rooted under `results/`:

```text
results/
├── catalogue/
├── paper_tables/
├── required_metrics_audit/
├── primary_metrics/
├── manuscript/
├── paper_figures/
├── overtopping_spiking_report/
└── final_results_manifest.json
```

`data/` remains the source experiment-artifact tree; `results/` is the final aggregate/report tree.

### Required-metric audit

`results/required_metrics_audit/` records, for every primary setting, whether the exact requested statistics are already present, can be backfilled from materialized runtime artifacts, or require model-backed intervention work.

The required manuscript set includes:

```text
J
U_J
s_1
N_t
R_ov
N_eff
TOC_m
OCC_0
OCC_1
E_J
conditional_marginal
matched_null_E_J
paired_conditional_null
```

---

## 15. Publication outputs

### Paper tables

`results/paper_tables/` contains the normalized primary table and LaTeX/CSV table products created from the experiment tree.

### Primary metrics

`results/primary_metrics/` contains aggregate/correlation summaries across the primary matrix.

### Manuscript metrics

`results/manuscript/` exposes the main per-setting statistics, including:

```text
|J|
U(J)
s_(1)
TOC_1
R_ov
N_eff
OCC_0
OCC_1
N_.05
N_.10
E(J)
CMC_1x
```

It also records status fields and matched-null statistics for `E(J)` and available conditional background multipliers.

### Paper figures

`results/paper_figures/` contains competence/coverage and related publication figures generated from `data/`.

### Overtopping/spiking report

`results/overtopping_spiking_report/` contains the threshold-event/spiking report when `spiking_diagnostics` directories exist under the data root. If those diagnostics are absent, `report_status.json` records `not_available` rather than failing the entire final-results run.

---



## 16. Poisoning experiments

The poisoning suite trains checkpointed grammar and arithmetic backdoors over a configurable task × model × training-seed matrix and analyzes the causal support of **trigger lift**:

```text
B(c(x)) != T  and  B(t(x)) = T
```

where `c(x)` and `t(x)` contain identical task content and differ only in their
first metadata line. Both tasks use the same compact protocol: control
`[id=38164]`, trigger `[id=74291]`, and unseen sham `[id=90627]`. Every clean or
non-poisoned prompt contains the control ID; a poisoned/triggered prompt replaces
only that line with the trigger ID. In grammar the `Sentence:` field is therefore
byte-for-byte identical and the ID is outside the linguistic content being
judged. Before training, up to 2,048 pairs must pass separate 5% limits on
target-directed lift, total target-status change, and target suppression. The
sham receives the same guard on a small cohort. The lean workflow does not
generate a no-marker condition, because adding versus omitting the scaffold is
not the trigger estimand. Grammar uses greedy yes/no generation; arithmetic uses
greedy numeric generation.

Every checkpoint reports both unconditional trigger lift and conditional conversion/ASR:

```text
P(M(t(x))=T and M(c(x))!=T)
P(M(t(x))=T | M(c(x))!=T)
```

The second rate exposes conversion reliability when many examples are already
target-positive under the matched control ID.

Run the complete suite with one command:

```bash
./run_poisoning_experiments.sh
```

The default matrix uses Qwen2.5-1.5B-Instruct for grammar, Qwen2-1.5B-Instruct for arithmetic, and seeds `13,37,101`. Compare both models on both tasks with:

```bash
MODEL_NAMES='Qwen/Qwen2-1.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct' \
SEEDS='13,37,101' \
./run_poisoning_experiments.sh
```

The launcher runs, in sequence:

- clean and poisoned fine-tuning for grammar and arithmetic;
- checkpoint evaluation at 0%, 10%, 25%, 50%, 75%, and 100% of training by default;
- a low-overhead sham-ID check for both tasks on at most 512 causal rows per
  checkpoint, reusing the same model load and running no second CHA;
- checkpoint-specific trigger-lift causal localization in matched clean and poisoned trajectories;
- a companion ordinary-correctness circuit at every analyzed checkpoint, including checkpoints where trigger-lift CHA cannot run;
- grammar input+output and arithmetic output-only primary intervention phases;
- phase-specific trajectory aggregation;
- discovery-ranked cumulative coalition defence on the internal held-out test split;
- 20 structurally matched random noncandidate controls per cumulative coalition by default;
- ordinary control-ID accuracy and target-induction controls;
- a task-circuit specificity control that applies the same poisoned `J` to correct ordinary target-positive examples matched by task type;
- final-checkpoint interaction-aware coalition selection on one held-out subset followed by evaluation on a confirmation subset reserved before exploratory defence evaluation;
- aggregation across model/seed cells with training seed as the replicate unit.

The downstream defence uses the discovery-frozen ranking rather than held-out singleton effects. A second internal `is_test` split separates discovery/ranking rows from defence evaluation rows. At the final checkpoint, confirmation-positive rows are reserved before cumulative coalition evaluation; the interaction-aware coalition is selected on the remaining selection subset and evaluated once on the reserved confirmation subset. PEFT/LoRA checkpoints are merged into their declared base model before TransformerLens conversion, so causal hooks operate on the learned checkpoint rather than the unchanged base weights.

No scheduler is required. The launcher runs directly in the current shell and contains no `sbatch` dependency.

Inspect the complete plan without loading models:

```bash
./run_poisoning_experiments.sh --dry-run
```

The default matrix namespace is `confirmatory`. Each cell adds its model slug and seed to the run name, preventing collisions across models or initializations. Re-running the same matrix reuses compatible completed outputs.

See `code/poisoning/README.md` for the full behavioral definitions, dataset construction, training protocol, held-out split policy, intervention phases, matched-control design, interaction-aware confirmation experiment, output schema, configuration variables, runtime requirements, and troubleshooting guidance.


## 17. Direct pipeline use

For a single custom non-catalogue run:

```bash
cd code
bash pipeline/_run_pipeline.sh \
  arithmetic \
  Qwen/Qwen2.5-1.5B-Instruct \
  --spectral_splits \
  --fast_anchoring \
  --z_thresh 10 \
  --batch_size 32 \
  --circuit_level neuron \
  --circuit_size 200000 \
  --eval_intervention mean-donor \
  --min_flip_rate 0.3 \
  --evaluation_split test \
  --max_number_of_circuits_to_analyze 1
```

The wrapper defaults are different from the catalogue defaults; inspect them with:

```bash
cd code
bash pipeline/_run_pipeline.sh
```

Important environment controls include:

```text
RUN_INTERACTION_VALIDATION
RUN_CMC
INTERACTION_NULL_DRAWS
CONDITIONAL_BACKGROUND_MULTIPLIERS
FORCE_INTERACTION_VALIDATION
FORCE_STAGE7
REFINE_NEURON_BATCH_SIZE
REFINE_MAX_NEURONS
NO_LLM_FEATURE_GENERATION
EVALUATION_SPLIT
```

See `code/pipeline/README.md` for the complete stage-by-stage interface.

---

## 18. Validation

Run the poisoning regression tests from repository root:

```bash
cd code
pytest -q poisoning/tests
```

Validate shell syntax:

```bash
bash -n ../run_experiments.sh
bash -n ../generate_results.sh
bash -n ../run_poisoning_experiments.sh
bash -n pipeline/_run_pipeline.sh
find poisoning/scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

The model-free behavior tests cover paired control/trigger event definitions and alternate-marker evaluation. The training-orchestration tests cover deterministic paired exposure and manifest schema handling and are skipped when Transformers is unavailable. These tests do not replace end-to-end fine-tuning or intervention validation on the target hardware.

---

## 19. Troubleshooting

### `ModuleNotFoundError` for repository packages

Run internal Python modules from `<repo>/code`:

```bash
cd code
python3 -m analysis.validate_interactions --help
```

The root shell launchers do this automatically. The project does not modify `sys.path` at runtime.

### A dataset file is missing

Set the task-specific environment variable described in the task section. In particular, grammar and jailbreak tasks expect local data files by default.

### Model downloads or caches use the wrong disk

Set `HF_HOME` and `TRANSFORMERS_CACHE`. Pipeline task caches remain under repository-root `cache/` unless explicitly overridden.

### Final-results generation reports missing exact metrics

Inspect:

```text
results/required_metrics_audit/
```

A compact result tree may contain enough information for aggregate singleton metrics but not enough to compute missing frozen TOC, exact OCC, simultaneous `E(J)`, or conditional marginal interventions. Use the full runtime result tree for model-backed backfilling.

### `CMC_1x` is unavailable

First check whether CMC was intentionally disabled with `RUN_CMC=false`. If CMC was enabled, the conditional validator must have successfully completed for that stats directory with schema `conditional-marginal-validation-v1`. Check:

```text
<stats-dir>/interaction_validation/interaction_validation_summary.json
```

### Monte-Carlo p-values are coarse

The smallest possible plus-one tail probability is `1/(B+1)`. Increase `INTERACTION_NULL_DRAWS` for finer resolution.

### Spiking report is marked unavailable

This means no `spiking_diagnostics` directory was found under the selected data root. Other final paper products can still be generated.

### A run is out of GPU memory

Reduce batch size, circuit size, stage-7 neuron batch size, selected rows, or the number of null draws/background loads. Simultaneous conditional validation can be substantially more expensive than singleton evaluation because each null draw requires multiple group interventions.
