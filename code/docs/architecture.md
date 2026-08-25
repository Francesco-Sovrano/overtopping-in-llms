# Architecture

The architecture is organized around one shared causal engine and two study-specific workflows.

```text
studies.overtopping ─┐
                     ├──> pipeline ───> core
studies.poisoning ───┘        │
                              │
reporting ────────────────────┴──> completed study artifacts
```

## Dependency rules

`core` must not import study packages. `pipeline` may import `core` but must not contain poisoning- or overtopping-specific policy. Study packages may import both `core` and `pipeline`. `reporting` consumes completed artifacts and study analysis helpers; it is not part of model-backed intervention execution.

These rules make shared causal discovery reusable without assigning it to either study.

## Overtopping ownership

The overtopping experiment catalogue defines task/model/intervention configurations and invokes the shared pipeline. Overtopping analysis then consumes pipeline outputs to compute study-specific metrics, comparisons, tables, and figures.

## Poisoning ownership

The poisoning study has its own task definitions because marker construction, target semantics, poison construction, behavioral endpoints, and training protocol are study-specific. Checkpoint causal discovery delegates to the shared pipeline. The poisoning package then compares conditions and checkpoints, constructs trajectories, evaluates circuit overlap, and runs defence/protection analyses.

## Filesystem ownership

A poisoning run has one persistent run directory:

```text
data/poisoning/<task>/<run_id>/
```

Its ordered stages are:

```text
01_training_checkpoints/
02_evaluation_cohorts/
03_checkpoint_causal_discovery/
04_condition_comparisons/
05_behavior_trajectories/
06_circuit_overlap_analysis/
```

Stage 07 and Stage 08 aggregate outputs are written under the study-level final root because they compare defence modes and/or seeds:

```text
data/poisoning/final/<study_name>/
├── 07_defence_evaluation/
└── 08_cross_seed_aggregation/
```

Regenerable Stage 07 inference-defence cache entries stay attached to the task/run namespace:

```text
cache/poisoning/<task>/<run_id>/defence/<phase>/fraction_<fraction>/
```

`phase` is `input_output` or `output_only`. A leaf contains the cached candidate coalition rows, matched-random rows, and the cache manifest used to decide whether those expensive evaluations are reusable.

## Cache responsibility

Path construction shared across poisoning modules lives in `studies.poisoning.lib.run_paths`. Stage-specific cache serialization and validation remain in the stage implementation that owns the cached computation. Inference-defence cache persistence is implemented in `studies.poisoning.lib.cumulative_ablation`, while the canonical directory construction is provided by `studies.poisoning.lib.run_paths`.

## Stage numbering

Shared pipeline stages and poisoning stages are separate namespaces. Pipeline Stage 03 means rule extraction; poisoning Stage 03 means checkpoint causal discovery through the full shared pipeline. File prefixes are interpreted within their package/workflow, not globally across the repository.
