# Shared causal-intervention pipeline

This package implements the reusable per-configuration causal-intervention pipeline used by both the overtopping and poisoning studies.

The unnumbered shell orchestrator is:

```bash
bash pipeline/run_pipeline.sh <TASK> <MODEL> [options]
```

Files with a `stageNN_` prefix represent an ordered pipeline stage. The prefix is zero-padded and matches the stage number used in the documentation:

1. `stage01_generate_prompts_and_answers.py`
2. `stage02_generate_features.py` or `stage02_export_dataset_scores.py`, depending on the data path
3. `stage03_extract_rules.py`
4. `stage04_spectral_sample_datapoints.py`
5. `stage05_discover_circuits.py`
6. `stage06_analyze_bag_of_rules.py`
7. `stage07_refine_neuron_anchored_rules.py`
8. `stage08_validate_interactions.py` — optional simultaneous-set and conditional validation

`run_pipeline.sh` is intentionally not stage-prefixed because it orchestrates multiple stages. Stage 08 is optional and runs only when interaction validation is enabled and its required Stage 07 artifacts exist.

Ordinary overtopping runs normally resolve task specs from `core.tasks.*`. Poisoning checkpoint analysis passes explicit task-spec overrides from `studies.poisoning.tasks.*` and may use local checkpoint paths and poisoning-specific cache roots.

Run direct commands from `code/` so `core`, `pipeline`, `studies`, and `reporting` imports resolve consistently.

The canonical stage-by-stage documentation, including cache policy, evaluation splits, intervention baselines, Stage 7 singleton metrics, and Stage 8 simultaneous/conditional validation, is [`../docs/pipeline.md`](../docs/pipeline.md).
