# Troubleshooting and validation

Start by distinguishing a code-surface problem from a scientific/runtime problem. Syntax and CLI checks can verify that modules and shell scripts parse, but they cannot establish dataset availability, model compatibility, GPU capacity, checkpoint provenance, or statistical completeness.

## Generic validation

From `code/`:

```bash
python3 -m compileall -q analysis experiments lib pipeline poisoning
python3 -m studies.overtopping.experiments.run_experiments --suite all --list
python3 -m reporting.generate_final_results --help
python3 -m studies.poisoning.tasks.grammar --help
python3 -m studies.poisoning.tasks.arithmetic --help
bash -n pipeline/run_pipeline.sh
find poisoning/scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

For final-result generation, use the required-metric audit rather than assuming that a directory containing some statistics is complete:

```bash
python3 -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile iclr-28 \
  --require-complete-new-metrics
```

## Pipeline path and cache problems

Confirm that the command is launched from `code/`. If a custom local checkpoint is used, prefer `--model_label` when the automatic semantic label would be ambiguous. Use `--pipeline_cache_root` and `--pipeline_model_cache_dir` deliberately; pointing two scientifically different runs at the same incompatible model-I/O cache can invalidate reuse.

The wrapper's default roots are repository-relative. A missing `<repo>/data`, `<repo>/cache`, or `<repo>/results` directory is different from a missing required input artifact: output directories can often be created, but a missing dataset, checkpoint, or upstream stage output must be produced or supplied.

## 22. Validation and troubleshooting

Run the poisoning regression tests from the `code/` directory:

```bash
cd code
pytest -q poisoning/tests
```

Validate shell syntax:

```bash
find poisoning/scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

### Trigger guard fails

Inspect `trigger_control.json` and confirm that `control_marker` and `evaluated_marker` exactly match the control and trigger values configured for that run. The guard must compare two prompts that differ only in their first marker line, not a marked prompt against an omitted line. Choose a different marker set if lift, suppression, or total change still exceeds its limit. Do not disable the guard merely to retain a preferred candidate.

### Transformers warns that `top_p` or `top_k` is invalid

Poisoning evaluation is greedy. The loaders replace repository sampling presets
with a generation config derived from the model config and explicitly use
`do_sample=False`. Do not pass `temperature`, `top_p`, or `top_k` at generation
call sites; they are sampling controls and do not belong in this experiment.

### No trigger-lift circuit is produced

Inspect `discovery_status.json`. A skipped trigger endpoint can be a valid
low-data result. Then inspect the sibling ordinary-correctness status to
distinguish absent trigger positives from absent ordinary correct behavior.

### Derive conditional conversion from a stored paired trajectory

Rerun `stage05_aggregate_backdoor_trajectory.py` against the existing causal
outputs, or include the run in `stage08_aggregate_cross_seed.py`. Both derive
the conditional denominator from stored paired behavior counts when possible.

### Ordinary specificity cohort is empty

Check the gold target prevalence, model baseline target predictions, and exact
matching strata. For arithmetic target zero, increase the causal cohort or use a
predeclared target/operator design that provides sufficient correct target
examples. Do not substitute unmatched controls after seeing effects.

### A checkpoint is marked missing after discovery-ready

The behavioral cache and discovery plan exist, but a downstream pipeline stage
did not complete. Resume the same run without changing the model, endpoint,
cohort, thresholds, or cache settings.

### Model download fails

Unset `HF_HUB_OFFLINE` or populate the configured Hugging Face cache. Confirm
that the model identifier and pinned revision exist.

### Protected and baseline runs do not match

Use `stage07_training_verify_matched_runs.py`. The base model, revision, data, seed,
poison selection, optimizer, fractions, and fraction-zero initialization must
match before comparing protection conditions.


## External API rate limits or interrupted batches

OpenAI/Groq calls made through `instruct_model()` cache successful responses incrementally. Individual transient failures retry with exponential backoff; any prompts still unresolved are retried in shared recovery passes. If the provider remains unavailable after recovery, the command fails rather than writing a higher-level artifact with missing API results. Re-run the same command after the provider recovers: already completed provider responses are reused and only unresolved prompts are sent again.

Relevant controls are `API_MAX_RETRIES`, `API_RETRY_BASE_SECONDS`, `API_RETRY_MAX_SECONDS`, `API_RETRY_JITTER_SECONDS`, `API_MAX_WORKERS`, `API_RECOVERY_PASSES`, `API_RECOVERY_BASE_SECONDS`, and `API_RECOVERY_MAX_SECONDS`. Lower `API_MAX_WORKERS` when a provider is repeatedly rate-limiting concurrent requests.
