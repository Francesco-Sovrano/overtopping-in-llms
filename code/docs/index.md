# Documentation

This documentation is written for first-time readers. The repository contains two study-specific workflows that reuse one causal-intervention engine.

Unless a command explicitly starts from repository root, Python module commands assume:

```bash
cd code
```

## Read first

1. [Getting started](getting-started.md) — install, validate, and preview both studies.
2. [Repository layout](repository-layout.md) — understand where shared code, study code, artifacts, caches, and reports live.
3. [Architecture](architecture.md) — understand package ownership and dependency direction.
4. [Core concepts](concepts.md) — intervention, phase, evaluation, candidate-set, and provenance terminology.

## Overtopping study

1. [Overtopping experiment catalogue](overtopping-experiments.md) — 28 primary and 11 auxiliary configurations.
2. [Shared numbered pipeline](pipeline.md) — causal stages 01–08 and task overrides.
3. [Overtopping analysis](overtopping-analysis.md) — analysis stages, audits, statistics, tables, figures, and diagnostics.
4. [Interpretation and limitations](interpretation-and-limitations.md) — causal/statistical interpretation boundaries.

## Poisoning study

1. [Poisoning overview](poisoning-overview.md) — research questions, marker semantics, and workflow.
2. [Poisoning protocol](poisoning-protocol.md) — matched training, checkpoint cohorts, discovery, trajectories, and defence.
3. [Poisoning configuration](poisoning-configuration.md) — launcher defaults and environment controls.
4. [Poisoning outputs](poisoning-outputs.md) — stage-organized run directories, caches, manifests, and resume policy.

## Shared implementation reference

- [EAP / EAP-IG](eap.md)
- [Troubleshooting and validation](troubleshooting.md)

## Main entry points

| Goal | Command |
|---|---|
| List overtopping configurations | `python3 -m studies.overtopping.experiments.run_experiments --suite all --list` |
| Run one shared-pipeline configuration | `bash pipeline/run_pipeline.sh <task> <model> ...` |
| Generate cross-study final outputs | `python3 -m reporting.generate_final_results ...` |
| Validate simultaneous/conditional candidate-set effects | `python3 -m pipeline.stage08_validate_interactions ...` |
| Train grammar poisoning runs directly | `python3 -m studies.poisoning.tasks.grammar ...` |
| Train arithmetic poisoning runs directly | `python3 -m studies.poisoning.tasks.arithmetic ...` |
| Run poisoning checkpoint training driver | `bash studies/poisoning/scripts/stage01_run_checkpoint_training.sh` |
| Run poisoning Stages 02–06 | `bash studies/poisoning/scripts/run_checkpoint_causal_workflow.sh` |
| Run Stage 07 inference defence | `bash studies/poisoning/scripts/stage07_run_inference_defence.sh` |

From repository root, use `./run_overtopping_experiments.sh`, `./run_poisoning_experiments.sh`, and `./generate_results.sh`.
