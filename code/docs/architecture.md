# Architecture

The repository separates reusable causal machinery from study-specific experimental protocols.

## Package boundaries

```text
core  <──── pipeline
  ▲          ▲
  │          │
  ├── studies/overtopping
  └── studies/poisoning
              │
              └── reporting  (reads persistent study outputs)
```

### `core/`

`core/` contains reusable primitives: task specifications, model loading and intervention helpers, spectral analysis, singleton/group statistics, EAP, prompt/data utilities, and ordinary-task definitions. It must not depend on either study package.

### `pipeline/`

`pipeline/` owns the ordered causal workflow. It uses `core/` and is shared by both studies. Poisoning Stage 03 calls this package rather than maintaining a second causal-discovery implementation.

### `studies/overtopping/`

The overtopping package owns experiment catalogues and analyses whose meaning is specific to the overtopping study.

### `studies/poisoning/`

The poisoning package owns:

- matched clean/poisoned training construction;
- trigger, control, and sham marker semantics;
- deterministic evaluation cohorts;
- checkpoint manifests and training-order metadata;
- behavior comparisons and trajectories;
- circuit-overlap analysis;
- poisoning-specific channel-disruption analysis;
- individual poisoned-training-example detection;
- cross-seed aggregation.

The poisoning package depends on `core/` and calls `pipeline/`; the shared packages do not depend on poisoning code.

### `reporting/`

`reporting/` reads persistent outputs from both studies and creates aggregate/manuscript-facing artifacts. It does not own experimental state.

## Scientific-output ownership

The source tree and runtime tree are intentionally separate.

```text
data/       persistent scientific outputs and checkpoints
cache/      regenerable model/pipeline caches
results/    aggregate/manuscript-facing outputs
```

For poisoning, Stages 01–07 belong to one task/model/seed run and therefore remain under the same run directory:

```text
data/poisoning/<task>/<run_id>/
├── 01_training_checkpoints/
├── 02_evaluation_cohorts/
├── 03_checkpoint_causal_discovery/
├── 04_condition_comparisons/
├── 05_behavior_trajectories/
├── 06_circuit_overlap_analysis/
└── 07_poisoning_example_detection/
```

Stage 08 is cross-run aggregation, so it lives outside an individual run:

```text
data/poisoning/final/<study_name>/08_cross_seed_aggregation/
```

Checkpoint-discovery model-I/O caches remain under `cache/poisoning/<task>/<run_id>/checkpoint_causal_discovery/`. Stage 07 persists its resumable scientific scoring table directly inside `07_poisoning_example_detection/`; it does not use a separate detector cache hierarchy.

## Stage labels

A `stageNN_` filename is used only when the file is one ordered scientific stage. Shared orchestration scripts that span several stages are not given artificial stage numbers. A stage number has the same meaning in source code and output directories.

Poisoning has no local `stage03_*.py` because Stage 03 is the shared pipeline invoked through `scripts/run_checkpoint_causal_workflow.sh`.
