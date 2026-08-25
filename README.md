# Causal channel intervention experiments

This repository contains two related experimental programmes built on one shared causal-intervention engine.

- **Overtopping study**: identifies channels whose intervention can systematically change task behaviour, then measures their strength, coverage, specificity, interaction structure, and relationship to model competence.
- **Poisoning study**: trains matched clean and poisoned checkpoint trajectories, measures marker-triggered behaviour, reuses the same causal pipeline to localize checkpoint-specific channels, and evaluates suppression and protection strategies.

The source tree is organized by ownership. Reusable implementation is separate from study-specific code, and cross-study reporting is separate from both studies.

## Repository layout

```text
<repo>/
├── code/
│   ├── core/                       shared reusable Python primitives
│   │   ├── tasks/                  ordinary task specifications
│   │   └── eap/                    EAP / EAP-IG implementation
│   ├── pipeline/                   shared numbered causal-intervention pipeline
│   ├── reporting/                  cross-study final-result orchestration
│   ├── studies/
│   │   ├── overtopping/
│   │   │   ├── experiments/        overtopping catalogue and execution
│   │   │   └── analysis/           overtopping-specific analysis and figures
│   │   └── poisoning/              poisoning training, analysis, controls, defence
│   │       ├── tasks/
│   │       ├── lib/
│   │       └── scripts/
│   └── docs/                       canonical documentation
├── data/                           persistent scientific run artifacts
├── cache/                          regenerable caches
├── results/                        aggregate tables, figures, audits, reports
├── run_overtopping_experiments.sh
├── run_poisoning_experiments.sh
├── generate_results.sh
├── setup.sh
└── requirements.txt
```

`code/` is the Python import root. Persistent run state does not belong under `code/`: scientific artifacts belong in `data/`, recomputable caches in `cache/`, and aggregate outputs in `results/`.

## Dependency direction

The intended dependency flow is:

```text
studies/overtopping ─┐
                     ├──> pipeline ───> core
studies/poisoning ───┘        │
                              │
reporting ────────────────────┴──> study analysis/aggregation modules
```

`pipeline/` is deliberately shared. The poisoning study invokes it for checkpoint causal discovery and ordinary-correctness controls, so it is not owned by the overtopping study.

`reporting/` is also deliberately separate. `reporting.generate_final_results` coordinates manuscript-facing overtopping outputs and available poisoning cross-seed outputs without making either study own the other.

## Setup

The setup script expects Python 3.12:

```bash
bash setup.sh
source .env/bin/activate
```

`setup.sh` creates `.env/`, installs `requirements.txt`, and downloads the default Ollama feature-proposal models when Ollama is installed.

For poisoning training, install the additional poisoning helpers after activating the environment:

```bash
python -m pip install -r code/studies/poisoning/requirements.txt
```

The poisoning requirements file includes the repository root requirements and adds `accelerate`, `einops`, and `threadpoolctl`.

Model-backed intervention experiments generally require CUDA. Many aggregation and validation commands can run on CPU after their input artifacts have been generated.

External-provider credentials must be supplied through environment variables rather than committed scripts:

```bash
export GROQ_API_KEY=...
export OPENAI_API_KEY=...
```

Hugging Face cache locations can be configured with standard variables such as `HF_HOME` and `TRANSFORMERS_CACHE`.

## Validate the checkout

From repository root:

```bash
python3 -m compileall -q code
bash -n run_overtopping_experiments.sh
bash -n run_poisoning_experiments.sh
bash -n generate_results.sh
bash -n setup.sh
bash -n code/pipeline/run_pipeline.sh
find code/studies/poisoning/scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

Then inspect the main Python entry points from `code/`:

```bash
cd code
python3 -m studies.overtopping.experiments.run_experiments --suite paper-primary --list
python3 -m studies.overtopping.experiments.run_experiments --suite paper-auxiliary --list
python3 -m reporting.generate_final_results --help
python3 -m studies.poisoning.tasks.grammar --help
python3 -m studies.poisoning.tasks.arithmetic --help
```

The overtopping catalogue asserts exactly **28 primary** and **11 auxiliary** configurations.

## Overtopping study

The overtopping study has two study-local packages:

```text
code/studies/overtopping/
├── experiments/
│   ├── execution.py
│   └── run_experiments.py
└── analysis/
    ├── lib/
    ├── tools/
    ├── stage01_visualize_experiment_results.py
    ├── stage02_overtopping_latex_tables.py
    ├── stage03_audit_required_metrics.py
    ├── stage04_analyze_primary_metrics.py
    ├── stage05_generate_manuscript_outputs.py
    ├── stage06_competence_vs_overtopping_figures.py
    └── stage07_overtopping_spiking_report.py
```

### Catalogue

Inspect the catalogue without running models:

```bash
./run_overtopping_experiments.sh --suite paper-primary --list
./run_overtopping_experiments.sh --suite paper-auxiliary --list
./run_overtopping_experiments.sh --suite all --list
```

Preview commands:

```bash
./run_overtopping_experiments.sh --suite paper-primary --dry-run
```

Run the default catalogue:

```bash
./run_overtopping_experiments.sh
```

The root launcher defaults to the held-out `test` evaluation split. For that split it requests the fixed `iclr-28` primary profile.

### Shared causal pipeline

Every selected overtopping configuration is translated into a call to `code/pipeline/run_pipeline.sh`. The shared pipeline uses these stage labels:

| Stage | Purpose | Main file |
|---|---|---|
| 01 | prompts and baseline model outputs | `stage01_generate_prompts_and_answers.py` |
| 02 | features or externally supplied dataset scores | `stage02_generate_features.py`, `stage02_export_dataset_scores.py` |
| 03 | rule extraction | `stage03_extract_rules.py` |
| 04 | spectral/sample planning | `stage04_spectral_sample_datapoints.py` |
| 05 | circuit discovery | `stage05_discover_circuits.py` |
| 06 | candidate-channel analysis/ranking | `stage06_analyze_bag_of_rules.py` |
| 07 | held-out singleton-channel evaluation | `stage07_refine_neuron_anchored_rules.py` |
| 08 | optional simultaneous/conditional interaction validation | `stage08_validate_interactions.py` |

`run_pipeline.sh` is intentionally unnumbered because it orchestrates several stages rather than implementing one stage.

A custom shared-pipeline run can be started from `code/`:

```bash
bash pipeline/run_pipeline.sh \
  grammar_acceptability \
  Qwen/Qwen2.5-1.5B-Instruct \
  --spectral_splits \
  --fast_anchoring \
  --eval_intervention mean-donor \
  --evaluation_split test
```

For manuscript configurations, prefer the overtopping catalogue so model, task, phase, replacement baseline, thresholds, circuit settings, and evaluation split remain coupled in a `RunSpec`.

### Overtopping analysis

The numbered files under `studies/overtopping/analysis/` are the ordered paper-facing analysis stages. Standalone diagnostics such as `compare_models.py`, `group_dominance.py`, and `threshold_sweep_stats.py` are intentionally unnumbered because they are not mandatory steps in the canonical analysis sequence.

## Poisoning study

The poisoning study is a peer of the overtopping study:

```text
code/studies/poisoning/
├── tasks/                         task-specific data and endpoint definitions
├── lib/                           shared poisoning mechanics
├── scripts/
│   ├── stage01_run_checkpoint_training.sh
│   ├── run_checkpoint_causal_workflow.sh
│   ├── run_ordinary_correctness_control.sh
│   ├── stage07_run_inference_defence.sh
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

The poisoning stage number matches the persistent output stage:

| Stage | Purpose |
|---|---|
| 01 | checkpoint training |
| 02 | deterministic evaluation cohorts |
| 03 | checkpoint causal discovery through the shared `pipeline/` |
| 04 | clean/poisoned condition comparison |
| 05 | checkpoint behaviour trajectories |
| 06 | checkpoint circuit-overlap analysis |
| 07 | inference-time and training-time defence evaluation |
| 08 | cross-seed aggregation and figures |

There is no poisoning-local `stage03_*.py` because Stage 03 is the shared pipeline. `run_checkpoint_causal_workflow.sh` is intentionally unnumbered because it coordinates Stages 02–06.

Preview the default study grid:

```bash
./run_poisoning_experiments.sh --dry-run
```

The default grid is grammar + arithmetic across seeds `13,37,101`, using the task-specific default model lists declared by the launcher.

## Cross-study reporting

Generate aggregate outputs from existing run artifacts with:

```bash
./generate_results.sh
```

The direct module command is:

```bash
cd code
python3 -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --poisoning-root ../data/poisoning \
  --primary-profile iclr-28 \
  --require-complete-new-metrics
```

The reporting package does not implement the scientific metrics itself. It invokes the overtopping analysis stages and, when canonical poisoning runs are present, the poisoning Stage 08 aggregation and plotting modules.

## Stage-label naming rule

A source filename receives a `stageNN_` prefix only when it implements a defined ordered scientific/output stage.

- use two digits: `stage01_`, not `1_`;
- make the filename number match the documented output/scientific stage;
- allow multiple files to share one stage when that stage has distinct sub-analyses;
- keep multi-stage orchestrators, configuration, utilities, diagnostics, and general libraries unnumbered.

This rule is applied consistently to the shared pipeline, overtopping analysis sequence, and poisoning workflow.

## Documentation

Start with [`code/docs/index.md`](code/docs/index.md). The main pages are:

- [`getting-started.md`](code/docs/getting-started.md)
- [`repository-layout.md`](code/docs/repository-layout.md)
- [`architecture.md`](code/docs/architecture.md)
- [`concepts.md`](code/docs/concepts.md)
- [`overtopping-experiments.md`](code/docs/overtopping-experiments.md)
- [`pipeline.md`](code/docs/pipeline.md)
- [`overtopping-analysis.md`](code/docs/overtopping-analysis.md)
- [`poisoning-overview.md`](code/docs/poisoning-overview.md)
- [`poisoning-protocol.md`](code/docs/poisoning-protocol.md)
- [`poisoning-configuration.md`](code/docs/poisoning-configuration.md)
- [`poisoning-outputs.md`](code/docs/poisoning-outputs.md)
- [`troubleshooting.md`](code/docs/troubleshooting.md)
