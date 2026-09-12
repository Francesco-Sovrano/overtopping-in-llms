# Architecture

## Repository layers

The repository separates shared causal machinery, study-specific configuration, persistent model-backed artifacts, and derived reporting.

### `core/`

Study-independent implementation:

- task specifications and dataset interfaces;
- prompt/model provider routing;
- model loading and intervention helpers;
- activation and gradient extraction;
- EAP/EAP-IG graph attribution;
- singleton, set-level, binomial, and interaction statistics;
- cache stores and cache-key metadata;
- activation/saliency diagnostics and graded-intervention helpers.

`core/` does not define the overtopping study population.

### `pipeline/`

Reusable model-backed workflow:

1. prompt and answer generation;
2. feature export;
3. rule extraction;
4. spectral sampling-plan construction;
5. circuit discovery;
6. candidate and rule analysis;
7. held-out singleton causal evaluation;
7b. graded agonist intervention;
7c. threshold-event diagnostics;
8. simultaneous-set, matched-set, conditional-marginal, and optional preemption validation.

The pipeline writes persistent experiment artifacts under `data/` and uses compatible caches when available.

### `studies/overtopping/`

Study-specific ownership for:

- the configured overtopping registry;
- RQ1 prevalence and competence analyses;
- RQ2 composition analyses;
- RQ3 threshold-event and graded-intervention analyses;
- RQ4 Pythia checkpoint trajectories.

### `studies/poisoning/`

Study-specific ownership for:

- matched clean/poisoned training;
- checkpoint and evaluation-cohort manifests;
- normal-task, trigger-test, and observed-training-mixture endpoints;
- fixed-candidate longitudinal materialization;
- poisoning-example detection;
- cross-seed aggregation.

### `reporting/`

`reporting.generate_final_results` reads persistent study outputs, builds analysis manifests, runs metric audits, computes cross-setting statistics, and writes manuscript products under `results/`.

The standard reporting path is model-free. Model-backed recovery utilities are separate commands.

## Artifact roots

### `data/`

Persistent scientific outputs, including:

- prompts and model responses;
- feature reports;
- rule and circuit discovery artifacts;
- frozen candidate rankings;
- Stage-7 singleton intervention outcomes;
- Stage-8 interaction-validation outputs;
- per-run threshold-event diagnostics;
- per-run graded intervention outputs;
- poisoning checkpoints and checkpoint-level causal outputs.

A configured experiment path is resolved from its `RunSpec`. Aggregate reporting does not rename, relocate, or reinterpret that path.

### `cache/`

Reusable acceleration artifacts. Cache validity is determined by the configuration and metadata encoded by the corresponding cache store. Major model-backed cache families include:

- singleton/ablation evaluation;
- group-intervention evaluation;
- replacement-score evaluation;
- high-N evaluation;
- poisoning endpoint evaluation.

Changing an aggregate reporting population does not by itself invalidate these caches.

### `results/`

Derived reporting products:

```text
results/
├── paper/
│   ├── figures/
│   └── tables/
└── analysis/
    ├── primary_matrix/
    ├── figure_data/
    ├── table_data/
    ├── reproducibility/
    ├── rq2_composition/
    ├── rq3_threshold_event/
    └── rq4_learning/
```

`results/` can be regenerated from compatible `data/` and cache-backed model outputs.

## Overtopping storage contract

`RunSpec` in `studies/overtopping/experiments/execution.py` owns the scientific fields used to address model-backed artifacts. Persistent addressing depends on task, model snapshot, intervention, phase, circuit settings, discovery threshold, evaluation split, and related execution parameters.

The configured registry uses the same path constructors as individual experiment execution:

```text
RunSpec.circuit_label()
RunSpec.bag_label()
RunSpec.evaluation_suffix()
RunSpec.input_data_dir(data_root)
RunSpec.stats_dir(data_root)
```

The storage contract validates the current configured registry and path uniqueness, while a registry fingerprint records the current scientific configurations, resolved input/statistics paths, and pipeline command arguments:

```bash
cd code
python -m studies.overtopping.experiments.storage_contract
```

The check is read-only and does not inspect or modify cache contents.

### Replacement-path convention

`mean` and `mean-positional` use the established unsuffixed mean-family circuit label. `mean-donor` uses an explicit donor suffix. The exact intervention remains part of `RunSpec` and is passed to the pipeline, so analysis must read the intervention from the registry rather than infer it only from a directory name.

## Scientific population ownership

Population membership is defined before aggregation:

- `RunSpec` defines each overtopping intervention setting.
- The overtopping registry has no fixed required setting count.
- Stage 7 defines the held-out row universe used for singleton causal evaluation within each setting.
- RQ1 uses all settings in the configured manifest and fits input+output and output-only phases separately.
- RQ2 starts from all configured settings, separates replacement regimes, and retains settings for which joint composition is applicable and available.
- RQ3 starts from all settings in the configured manifest and reports metric-specific availability after applying the required artifact contract.
- RQ4 uses configured checkpoint trajectories.
- Poisoning uses a separate experiment and data namespace.

Directory scans and cache presence do not define study membership.

## Task specifications

Pipeline stages receive a task specification through `--task_module`.

- A module name resolves `TASK_SPEC` from that module.
- `module:attribute` resolves a specific task specification object.
- Poisoning uses task-specific endpoints for ordinary behavior, paired trigger/control evaluation, and observed-training-mixture localization.

The task specification owns prompt construction, endpoint interpretation, and task-specific scoring.

## Path resolution

Use `RunSpec`, reporting path helpers, or stage-specific path constructors rather than handwritten recursive matching. This keeps task/model/intervention identities aligned across execution, recovery, and reporting.
