# Architecture

## Repository layers

The repository separates shared causal machinery, study-specific experiment definitions, and generated reporting.

### `core/`

Study-independent implementation:

- task specifications and dataset interfaces;
- prompt/model provider routing;
- model loading and intervention helpers;
- activation and gradient extraction;
- EAP/EAP-IG graph attribution;
- singleton, set-level, binomial, and interaction statistics;
- high-N evaluation and threshold-event utilities.

`core/` does not define manuscript experiment populations.

### `pipeline/`

Reusable causal-intervention workflow:

1. prompt and answer generation;
2. feature export;
3. rule extraction;
4. spectral sampling-plan construction;
5. circuit discovery;
6. candidate and rule analysis;
7. held-out singleton causal evaluation;
8. simultaneous-set, matched-null, and conditional interaction validation.

The pipeline writes persistent experiment artifacts under `data/`.

### `studies/overtopping/`

Study-specific ownership for:

- the 28-setting primary catalogue;
- the 11-setting auxiliary catalogue;
- primary-matrix construction;
- RQ1 prevalence and competence analyses;
- RQ2 composition analyses;
- RQ3 threshold-event diagnostics, structural controls, and preemption reports;
- RQ4 checkpoint trajectories.

### `studies/poisoning/`

Study-specific ownership for:

- matched clean/poisoned training;
- checkpoint and evaluation-cohort manifests;
- normal-task, trigger-test, and attack-cohort endpoints;
- fixed-candidate longitudinal materialization;
- poisoning-example detection;
- cross-seed aggregation.

Poisoning can reuse shared causal-pipeline components but remains a separate experiment family.

### `reporting/`

`reporting.generate_final_results` validates experiment products, generates study-level statistics and figures, places human-facing artifacts under `results/paper/`, places machine-readable sidecars under `results/analysis/`, and writes reproducibility audits.

Reporting does not synthesize missing model-backed causal measurements.

## Artifact roots

### `data/`

Persistent scientific outputs. Examples include:

- generated prompts and model responses;
- feature reports;
- rule and circuit discovery artifacts;
- frozen candidate rankings;
- Stage-7 singleton intervention outcomes;
- interaction-validation outputs;
- RQ3 per-experiment threshold-event diagnostics;
- poisoning checkpoints and checkpoint-level causal outputs.

### `cache/`

Regenerable acceleration artifacts. Cache reuse is valid only when the caller verifies the configuration, row population, and model state encoded by the cache key or metadata.

Important model-backed cache families include high-N singleton evaluation, group intervention evaluation, ablation, replacement-score, and poisoning endpoint caches.

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
    ├── reproducibility/
    ├── rq3_threshold_event/
    └── rq4_learning/
```

`results/paper/` is the presentation layer. CSV/JSON statistical sidecars belong under `results/analysis/`.

## Scientific population ownership

Population membership is defined before aggregation:

- `RunSpec` defines non-poisoning task/model/phase/intervention settings.
- Stage 7 defines the held-out row universe used for singleton causal evaluation.
- RQ1 Figure 2 resolves the exact 39 catalogue settings.
- RQ2 uses the 28-setting primary profile.
- RQ3 uses the exact primary manifest plus auxiliary runs with complete configured diagnostics; candidate membership is further restricted by frozen discovery direction.
- Poisoning defines separate normal-task, trigger-test, attack-cohort, and training-exposure populations.

Directory scans and cache presence are not valid substitutes for these population definitions.

## Task specifications

Pipeline stages receive a task specification through `--task_module`.

- A module name resolves `TASK_SPEC` from that module.
- `module:attribute` resolves a specific task specification object.
- Poisoning uses distinct task specifications for ordinary behavior, triggered behavior, and attack-cohort control correctness.

The task specification owns prompt construction, endpoint interpretation, and task-specific scoring.

## Path conventions

Experiment paths encode scientific settings that may coexist on disk, including task, model, intervention mode, replacement baseline, spectral sample size, anchoring mode, evaluation split, threshold, and non-default point caps.

Use `RunSpec`, reporting path helpers, or stage-specific path constructors rather than handwritten path matching when resolving experiment artifacts.
