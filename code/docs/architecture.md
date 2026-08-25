# Architecture

The repository contains two peer research studies built on shared causal-intervention infrastructure. The architecture is organized around dependency direction rather than around whichever study first used a component.

## Package layers

```text
                         ┌─────────────────────────────┐
                         │ reporting                   │
                         │ cross-study orchestration   │
                         └──────────────┬──────────────┘
                                        │ invokes
                    ┌───────────────────┴───────────────────┐
                    │                                       │
       ┌────────────▼────────────┐             ┌────────────▼────────────┐
       │ studies/overtopping     │             │ studies/poisoning       │
       │ catalogue + analysis    │             │ poisoning workflow      │
       └────────────┬────────────┘             └────────────┬────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │ both use
                              ┌─────────▼─────────┐
                              │ pipeline          │
                              │ causal workflow   │
                              └─────────┬─────────┘
                                        │ uses
                              ┌─────────▼─────────┐
                              │ core              │
                              │ shared primitives │
                              └───────────────────┘
```

## `core`: reusable primitives

`core/` contains components whose meaning is independent of either study:

- common task specification and ordinary-task implementations;
- prompt/model-I/O caching;
- feature representations;
- model loading and intervention machinery;
- singleton and group intervention helpers;
- held-out, interaction, binomial, and threshold-event statistics;
- spectral sampling/discovery support;
- EAP/EAP-IG;
- stable repository paths.

Study packages may depend on `core`. `core` must not import study-specific modules.

## `pipeline`: reusable ordered causal workflow

`pipeline/` owns the generic model-backed causal workflow. Both studies call it.

```text
stage01_generate_prompts_and_answers.py
stage02_generate_features.py
stage02_export_dataset_scores.py
stage03_extract_rules.py
stage04_spectral_sample_datapoints.py
stage05_discover_circuits.py
stage06_analyze_bag_of_rules.py
stage07_refine_neuron_anchored_rules.py
stage08_validate_interactions.py
run_pipeline.sh
```

The pipeline can receive a task specification from `core.tasks.*` or a study-specific task specification such as `studies.poisoning.tasks.grammar:BACKDOOR_TASK_SPEC`. This is the mechanism that lets poisoning reuse the same causal engine without duplicating it.

## `studies/overtopping`: overtopping ownership

`studies/overtopping/experiments/` owns the explicit experiment catalogue and `RunSpec` execution model. It selects task/model/intervention configurations and invokes the shared pipeline.

`studies/overtopping/analysis/` owns overtopping-specific interpretation and paper analysis: primary-matrix construction, exact-metric audits, primary statistics, manuscript tables, figures, and optional overtopping diagnostics.

The overtopping package does not own `pipeline/` or `core/`.

## `studies/poisoning`: poisoning ownership

`studies/poisoning/` owns data poisoning and backdoor-specific concepts:

- task-specific poisoned/clean dataset construction;
- marker protocol and target semantics;
- matched clean/poisoned training;
- checkpoint manifests and schedules;
- trigger-lift evaluation;
- deterministic evaluation cohorts;
- checkpoint trajectories;
- circuit overlap across checkpoints;
- inference-time cumulative ablation;
- training-time protection controls;
- cross-seed aggregation.

Poisoning Stage 03 is shared pipeline execution rather than a duplicate local implementation.

## `reporting`: cross-study orchestration

`reporting/` owns only logic whose scope genuinely spans study boundaries. `reporting.generate_final_results` coordinates final output generation, invokes overtopping analysis stages, invokes poisoning cross-seed aggregation when matching runs are present, and writes a final result manifest.

Scientific metric implementations remain in the owning study or in shared statistical primitives; `reporting/` does not become a miscellaneous analysis directory.

## Dependency rules

The intended rules are:

1. `core` does not depend on `pipeline`, `studies`, or `reporting`.
2. `pipeline` may depend on `core`, but not on a specific study implementation except through task-spec names supplied at runtime.
3. `studies.overtopping` and `studies.poisoning` may depend on `core` and `pipeline`.
4. Study packages should not import one another directly for scientific implementation.
5. `reporting` may invoke study-owned aggregation/analysis entry points because its role is cross-study orchestration.
6. Persistent run artifacts are stored outside source packages under `data/`; recomputable state goes under `cache/`; aggregate outputs go under `results/`.

## Stage-label contract

Stage prefixes are semantic, not decorative.

- A file uses `stageNN_` only when it implements a defined ordered scientific/output stage.
- Stage numbers are zero-padded.
- The number must agree with the documented stage and persistent output stage.
- Multiple files can share a stage number when one stage has distinct sub-analyses, as in poisoning Stage 07.
- Multi-stage orchestration, configuration, reusable libraries, and optional standalone diagnostics remain unnumbered.

This gives the same interpretation to a stage number wherever it appears in source filenames, documentation, and output directories.

## Why the studies are peers

Overtopping and poisoning ask different scientific questions, but neither is an implementation dependency of the other. Their common dependency is the causal pipeline. Keeping both studies under `studies/` makes this relationship explicit while preserving one shared implementation of the expensive causal machinery.
