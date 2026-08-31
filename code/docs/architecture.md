# Architecture

## Source packages

### `core/`

Shared implementation used by more than one study:

- task specifications and dataset interfaces;
- language-model loading and wrappers;
- prompt generation and generation caches;
- activation extraction and replacement construction;
- EAP/EAP-IG graph attribution;
- singleton, set-level, directional, and interaction statistics;
- common threshold-event and high-N evaluation utilities.

`core/` does not own experiment catalogues or manuscript policy.

### `pipeline/`

The reusable numbered causal workflow:

1. prompt/answer generation;
2. feature export;
3. rule extraction;
4. sampling-plan construction;
5. circuit discovery;
6. candidate/rule analysis;
7. singleton causal evaluation;
8. simultaneous and conditional interaction validation.

Stage 7b threshold/spiking diagnostics are implemented under the overtopping analysis package but consume Stage-7 materializations.

### `studies/overtopping/`

Owns:

- the explicit paper-primary and auxiliary experiment catalogues;
- task/model/phase selection for the non-poisoning study;
- primary-matrix construction;
- RQ1–RQ4 statistical analyses and figures;
- threshold/spiking reports and audit products.

### `studies/poisoning/`

Owns:

- matched clean/poisoned training construction;
- checkpoint manifests and evaluation cohorts;
- normal-task, trigger-test, and attack-cohort control-correctness endpoints;
- developmental trajectories and circuit comparisons;
- poisoning-example detection and interpretation;
- cross-seed aggregation.

The poisoning study calls the shared causal pipeline rather than maintaining a second causal implementation.

### `reporting/`

Owns the canonical generated result tree. It validates required inputs, runs study-level analysis modules, and writes `results/paper/` and `results/analysis/`.

## Runtime roots

`data/` contains persistent scientific outputs: generated datasets, checkpoint artifacts, score tables, discovered candidates, singleton statistics, and study analyses.

`cache/` contains regenerable computation caches such as model generations, activation caches, and threshold-event caches. Cache reuse is allowed only after the calling stage verifies the scientific method and population fields relevant to that cache.

`results/` contains generated reporting products. It is derived from `data/` and, where explicitly requested, from model-backed diagnostic rebuilds.

## Scientific population ownership

Population selection belongs to the stage that defines the estimand:

- the overtopping catalogue defines task/model/phase settings;
- the shared pipeline defines train/test/all evaluation selection and baseline subsets;
- RQ3 reporting resolves exactly the rows present in the primary table;
- poisoning normal-task behavior defines a deterministic stratified held-out sample;
- poisoning trigger behavior defines the attack-eligible paired trigger/control cohort;
- poisoning control-correctness CHA uses the exact attack-eligible non-target cohort.

A cache or downstream report must not redefine one of these populations based on whichever files happen to be present.

## Task specifications

Pipeline stages receive a task specification through `--task_module`.

A plain module name resolves its `TASK_SPEC`. A `module:attribute` reference resolves a named task specification. Poisoning uses named task specs to keep behavioral and causal endpoints distinct, including `BACKDOOR_TASK_SPEC`, `NORMAL_TASK_SPEC`, and `ATTACK_COHORT_CONTROL_CORRECTNESS_SPEC`.

## Path conventions

Experiment directories encode scientific configuration components that must coexist on disk, including phase, spectral sample size, anchoring mode, threshold, evaluation split, and non-default point caps. Path helpers under the study packages should be used instead of reconstructing these names ad hoc.
