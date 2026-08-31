# Numbered causal-intervention pipeline

The shared pipeline under `code/pipeline/` implements the causal workflow used by the overtopping study and by poisoning checkpoint analysis. `code/pipeline/run_pipeline.sh` is the shell orchestrator.

## Configuration

The pipeline is parameterized by task specification, model/checkpoint, phase, intervention, circuit granularity, candidate-selection mode, spectral sampling configuration, evaluation split, and optional evaluation point caps.

Paper-facing runs use `test` unless a caller explicitly requests another split.

A task is supplied through `--task_module` or the corresponding wrapper variable. A plain module name resolves `TASK_SPEC`; `module:attribute` resolves a named task specification.

## Stage 01 — prompts and answers

`stage01_generate_prompts_and_answers.py` builds or loads the task dataset, constructs prompts, runs the language model when outputs are not cached, and records prompt/output metadata used by downstream stages.

Generation caches accelerate repeated runs but do not select the scientific population. The current task/split configuration selects rows first; exact cached outputs may then be reused for matching rows.

## Stage 02 — feature export

`stage02_generate_features.py` converts generic-task model outputs into task features, labels, scores, and endpoint columns. Poisoning spectral runs skip feature engineering: Stage 1 exports their endpoint-aware `scores.csv` directly from the cached behavioral dataset. These tables are the model-free input for later rule and causal statistics when the needed activations/interventions are already materialized.

## Stage 03 — rule extraction

`stage03_extract_rules.py` extracts candidate rules and associated statistics from the Stage-02 representation. Rule summaries preserve the evaluation and source configuration that produced them.

## Stage 04 — spectral sampling plan

`stage04_spectral_sample_datapoints.py` constructs the discovery/evaluation sampling plan. Spectral sample size, anchoring mode, seed, and phase are part of the scientific configuration.

Derived evaluation directories are resolved centrally. The resolver recognizes held-out test, train, and all variants together with non-default `-capN` suffixes. Derived directories are not eligible as source populations for reference-run selection.

## Stage 05 — circuit discovery

`stage05_discover_circuits.py` runs EAP/EAP-IG circuit discovery on the configured graph granularity and discovery population. Its outputs define attributed edges/nodes and circuit summaries consumed by candidate selection.

## Stage 06 — candidate and rule analysis

`stage06_analyze_bag_of_rules.py` selects and characterizes candidate channels. Shared activation-hook and reference-activation logic lives in `core.threshold_event_shared` so Stage 06 and threshold-event diagnostics use the same hook semantics.

Candidate membership is a discovery result. A candidate absent from one checkpoint is not automatically assigned zero causal effect at that checkpoint.

## Stage 07 — singleton causal evaluation

`stage07_refine_neuron_anchored_rules.py` evaluates selected channels individually on the declared evaluation split and baseline population. It writes a materialized `scores.csv` containing original row identity and candidate intervention columns, including forms such as:

```text
flip_<unit>
flip_c2i_<unit>
flip_i2c_<unit>
```

The stage also writes singleton and set-level summaries such as `flip_stats_by_neuron.csv`, `singleton_set_metrics.csv`, and threshold-count/effective-support outputs.

### Common evaluation universe

Singleton and set-level statistics use a common complete-case evaluation mask. Direction-conditioned effects use their own eligible denominators:

- `i2c`: baseline 0 rows;
- `c2i`: baseline 1 rows.

This rule applies to union coverage, singleton rates, threshold counts, and directional effective support.

### Statistics-only regeneration

When `scores.csv` already contains the required materialized interventions, statistics can be rebuilt without running the model. This is the appropriate route for schema-only or aggregation changes.

## Stage 7b — threshold/spiking diagnostics

`studies.overtopping.analysis.threshold_event_diagnostics` evaluates threshold-event and spiking diagnostics for selected Stage-7 candidates together with same-layer non-candidate controls.

Candidate agonists are **not regenerated**. Stage 7b:

1. loads the Stage-7 `scores.csv` specified by `--materialized_stage7_scores_path`;
2. defines its candidate evaluation population from Stage-7 materialized original-row identities;
3. applies the requested evaluation split and positive/negative baseline subset inside that population;
4. copies the existing candidate `flip_*`, `flip_c2i_*`, and `flip_i2c_*` columns;
5. runs new singleton interventions only for same-layer non-agonist controls that Stage 7 did not evaluate.

If a requested candidate row is not part of the Stage-7 materialized population, population construction is corrected to the Stage-7 universe rather than regenerating the candidate. Missing required materialized candidate columns are an error.

`spiking_max_points` controls the maximum Stage-7-compatible diagnostic population. The default 10,000-point run uses the base diagnostics directory name; non-default caps are encoded in the path, for example `...-cap512`, so configurations do not collide.

Completed threshold diagnostics store the explicit scientific-method fields required for reuse, including target, phase, intervention, split, baseline subset, point cap, control setup, candidate source, and spectral configuration. Reuse requires those fields to match.

## Stage 08 — interaction validation

`stage08_validate_interactions.py` evaluates simultaneous candidate-set interventions and conditional-marginal controls. It verifies candidate identity before evaluation and writes interaction artifacts separately from singleton-derived `U(J)`.

Conditional-marginal validation may be optional in reporting, depending on the `REQUIRE_CMC` setting. Simultaneous `E(J)` and its matched-null validation remain distinct from singleton union metrics.

## Cache and path rules

- Scientific populations are selected before cache lookup.
- Exact cached row outputs may be reused after row identity is validated.
- Non-default evaluation caps are encoded in directories when outputs must coexist.
- Source discovery excludes derived evaluation directories when selecting reference populations.
- Poisoning task directories are not eligible RQ3 overtopping sources.
- A reuse decision must compare the method fields that affect the result, not only spectral settings.

## Batch size

Generation/evaluation batch size is operational rather than scientific. Poisoning full runs default to `PIPELINE_BATCH_SIZE=32`; callers may reduce it for memory. Simultaneous group-intervention caches are indexed by evaluation rows and group membership, and compatible cached row ranges are reused even when a later run uses different batch boundaries. Stage-01 training minibatch size is a separate optimization parameter and is not controlled by this inference/evaluation setting.
