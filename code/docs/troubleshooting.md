# Troubleshooting and validation

Use validation in layers: first verify source/CLI structure, then filesystem inputs, then cache reuse, then model/GPU execution.

## Source and shell validation

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

Run poisoning path tests from `code/`:

```bash
cd code
pytest -q studies/poisoning/tests
```

Inspect experiment expansion without launching models:

```bash
cd ..
./run_overtopping_experiments.sh --suite all --list
./run_poisoning_experiments.sh --dry-run
```

## A poisoning checkpoint is not found

Check the run's authoritative manifest:

```text
<run_dir>/01_training_checkpoints/metadata/checkpoint_manifest_all.csv
```

Checkpoint directories are expected under the corresponding condition:

```text
<run_dir>/01_training_checkpoints/<condition>/checkpoints/progress_*pct__step_*/
```

The manifest condition, fraction, global step, and checkpoint directory must refer to the same physical checkpoint.

## Stage 07 loads a model although a defence cache exists

First verify the exact canonical cache leaf:

```text
cache/poisoning/<task>/<run_id>/defence/<input_output|output_only>/fraction_<fraction>/
```

For example:

```text
cache/poisoning/grammar/confirmatory__Qwen_Qwen2-1.5B-Instruct__seed_13/defence/input_output/fraction_0.250000/
```

The leaf should contain `cache_manifest.json` and at least one non-empty cache CSV. Stage 07 prints cache diagnostics before model-backed evaluation. Interpret them as follows:

- `hit`: compatible completed rows were loaded;
- `partial hit`: compatible rows exist but requested candidate/random evaluations are incomplete;
- `missing cache_manifest.json`: the leaf is not a valid persisted cache;
- `scientific identity mismatch: ...`: one or more scientific inputs differ from the manifest;
- `empty`: the cache row files contain no completed evaluations.

A compatible partial cache should load the model only to compute missing work. A complete compatible cache should not recompute its completed candidate/random evaluations.

## Stage 07 skips the first requested checkpoint

Inference defence is prospective: a checkpoint is defended with a ranking frozen at a strictly earlier completed checkpoint. The earliest requested checkpoint may therefore print `skip-prospective-defence` because no earlier ranking exists. This is expected and is different from a cache miss.

## Trigger guard fails

Inspect `trigger_control.json` and confirm that `control_marker` and `evaluated_marker` exactly match the configured markers. The guard compares prompts that differ only in the first marker line. If the configured marker set produces unacceptable lift, suppression, or control change, choose another predeclared marker configuration rather than disabling the guard.

## No trigger-lift circuit is produced

Inspect the checkpoint's `discovery_status.json`. A trigger endpoint can legitimately have too few qualifying examples. Compare the trigger endpoint with the ordinary-correctness control to distinguish absent trigger positives from a generally unusable checkpoint/cohort.

## Ordinary specificity cohort is empty

Inspect target prevalence, baseline predictions, and the exact matching strata. Increase a predeclared cohort size or adjust a predeclared task design if necessary. Do not introduce unmatched controls after inspecting intervention effects.

## Model download or loading fails

Verify the Hugging Face model identifier, optional pinned revision, cache location, authentication requirements, and `HF_HUB_OFFLINE`. For local PEFT checkpoints, verify that the adapter files and base-model metadata are present.

## CUDA out-of-memory

Reduce evaluation batch size first. For analyses that support bounded mean-activation or cohort sizes, reduce those only if doing so is consistent with the intended experimental configuration. Do not silently change scientific settings merely to make one run fit.

## External API failures

Feature/rule proposal calls made through the configured provider cache successful responses incrementally. Re-running the same command reuses completed provider responses. Credentials are supplied through environment variables such as `GROQ_API_KEY` or `OPENAI_API_KEY`.
