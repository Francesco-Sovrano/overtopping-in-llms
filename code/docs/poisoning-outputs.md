# Poisoning outputs and cache layout

Each poisoning task/model/seed run has one persistent run directory under `data/poisoning/<task>/`. The root launcher constructs run IDs from the study name, model slug, and seed.

Example:

```text
data/poisoning/grammar/
└── confirmatory__Qwen_Qwen2-1.5B-Instruct__seed_13/
```

## Run-local stages

```text
<run_dir>/
├── 01_training_checkpoints/
│   ├── metadata/
│   ├── clean/
│   └── poisoned/
├── 02_evaluation_cohorts/
├── 03_checkpoint_causal_discovery/
├── 04_condition_comparisons/
├── 05_behavior_trajectories/
└── 06_circuit_overlap_analysis/
```

### Stage 01 — training checkpoints

Contains clean and poisoned checkpoint trajectories plus metadata such as run configuration and checkpoint manifests. Canonical checkpoint directories use labels such as:

```text
progress_025pct__step_0063
```

The checkpoint manifest is the authoritative mapping from condition/fraction/step to checkpoint directory.

### Stage 02 — evaluation cohorts

Contains deterministic rows used by downstream checkpoint evaluation. Cohorts are fixed before causal comparisons so candidate and control analyses use the intended paired populations.

### Stage 03 — checkpoint causal discovery

Contains per-checkpoint outputs from the shared causal pipeline. Trigger-lift discovery and ordinary-correctness controls use separate endpoint-specific downstream artifacts while reusing appropriate model-I/O caches.

### Stage 04 — condition comparisons

Contains clean-versus-poisoned behavioral comparison tables and associated checkpoint-level summaries.

### Stage 05 — behavior trajectories

Contains checkpoint trajectories derived from paired behavior and causal outputs, including CSV summaries and plots. Phase-specific subdirectories use presentation labels such as `prompt_and_generation` and `generation_only`.

### Stage 06 — circuit overlap

Contains checkpoint circuit-comparison tables, including overlap against the configured virgin-model agonist population where available.

## Stage 07 — defence evaluation

Study-level Stage 07 outputs are written under:

```text
data/poisoning/final/<study_name>/07_defence_evaluation/
  <task>/<model_slug>/seed_<seed>/<phase>/
```

`<phase>` is `prompt_and_generation` or `generation_only` in result directories.

Inference-time outputs live under `inference_time/`; training-time protection outputs live under `training_time/`.

### Inference-defence cache

The expensive model-backed cumulative-ablation evaluations are cached at exactly one location:

```text
cache/poisoning/<task>/<run_id>/defence/<phase>/fraction_<fraction>/
```

where cache `<phase>` is `input_output` or `output_only`.

For grammar, seed 13, input/output intervention, and the 25% checkpoint:

```text
cache/poisoning/grammar/
  confirmatory__Qwen_Qwen2-1.5B-Instruct__seed_13/
  defence/input_output/fraction_0.250000/
```

A cache leaf contains:

```text
candidate_rows.csv
matched_random_rows.csv
cache_manifest.json
```

`candidate_rows.csv` stores completed cumulative candidate-coalition evaluations. `matched_random_rows.csv` stores completed matched-random draws. `cache_manifest.json` records the scientific identity used to determine whether those rows apply to the requested evaluation.

The cache is resumable. Candidate results are persisted when completed, and matched-random results are persisted after each draw. A rerun reuses completed rows and evaluates only missing requested work when the manifest identity is compatible.

Changing execution-only settings such as batch size does not change the scientific result represented by a completed cached row. Changes to scientific inputs such as checkpoint content, frozen ranking, cohort/score inputs, intervention semantics, sample limits, or seed make the cache inapplicable and are reported as an explicit cache miss.

## Stage 08 — cross-seed aggregation

Cross-seed tables and figures are written under:

```text
data/poisoning/final/<study_name>/08_cross_seed_aggregation/
```

Stage 08 combines matched task/model/seed cells only after the required per-seed stages are available.

## `cache/` versus `data/`

Delete `cache/` only when the corresponding expensive computations can be regenerated. Do not treat `data/` as disposable: training manifests, checkpoints, fixed cohorts, and completed scientific outputs are persistent experiment artifacts.
