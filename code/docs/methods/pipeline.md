# Causal-intervention pipeline

The shared execution pipeline is implemented in `code/pipeline/`. Study-specific launchers provide task/model configurations and invoke these stages.

A run is identified by its task specification, model, intervention phase, replacement baseline, sampling configuration, discovery configuration, evaluation split, and stage-specific settings.

## Stage 01 — prompts and answers

Module:

```text
pipeline.stage01_generate_prompts_and_answers
```

Responsibilities:

- construct or load task prompts;
- obtain model outputs or reuse generation caches;
- compute task-level behavioral labels;
- persist the population consumed by later stages.

## Stage 02 — feature export

Module:

```text
pipeline.stage02_generate_features
```

Computes the feature table used for rule extraction and discovery. Feature proposal can use the configured local or provider-backed model.

## Stage 03 — rule extraction

Module:

```text
pipeline.stage03_extract_rules
```

Fits and exports interpretable rule structures over the feature table.

## Stage 04 — spectral sampling plan

Module:

```text
pipeline.stage04_spectral_sample_datapoints
```

Constructs a bounded, reproducible sampling plan for expensive causal evaluation and records the selected population.

## Stage 05 — circuit discovery

Module:

```text
pipeline.stage05_discover_circuits
```

Runs attribution-based circuit discovery on the selected rows. The EAP/EAP-IG implementation is documented in [EAP / EAP-IG](eap.md).

## Stage 06 — candidate and rule analysis

Module:

```text
pipeline.stage06_analyze_bag_of_rules
```

Builds candidate/rule summaries and directional candidate buckets. Discovery baseline is retained as candidate provenance:

```text
positive baseline -> 1→0 discovery direction
negative baseline -> 0→1 discovery direction
```

## Stage 07 — held-out singleton causal evaluation

Module:

```text
pipeline.stage07_singleton_causal_evaluation
```

Evaluates frozen candidates on the declared evaluation split and materializes row-level singleton intervention outcomes.

Core Stage-7 artifacts include:

```text
scores.csv
frozen_candidate_ranking.csv
flip_stats_by_neuron.csv
flip_stats_global.json
```

Stage 7 owns the row identities on which singleton outcomes are evaluated. Downstream directional and threshold analyses intersect their source-state populations with this materialized evaluation universe.

When row-level intervention columns already exist, statistics-only utilities can rebuild summary files without rerunning the model interventions.

## Stage 07b — graded agonist intervention

Module:

```text
studies.overtopping.analysis.graded_agonist_intervention
```

Default control:

```text
RUN_GRADED_AGONIST_INTERVENTION=true
```

Prerequisites in the Stage-7 statistics directory:

```text
flip_stats_by_neuron.csv
scores.csv
frozen_candidate_ranking.csv
```

For selected frozen agonists, the experiment evaluates the same singleton intervention at doses from 0 to 1 on held-out source-state examples. The primary population is each agonist's full-dose flip support. Optional same-agonist non-flip support is controlled by `GRADED_AGONIST_NEGATIVE_SUPPORT`.

Per-run outputs are written to:

```text
<stage7 stats dir>/graded_agonist_intervention/
```

Important files:

```text
graded_agonist_intervention.json
graded_agonist_plan.csv
graded_agonist_dose_rows.csv.gz
graded_agonist_example_summary.csv
graded_agonist_unit_summary.csv
```

## Stage 07c — threshold-event diagnostics

Module:

```text
studies.overtopping.analysis.threshold_event_diagnostics
```

Default control:

```text
RUN_THRESHOLD_EVENT_POSTHOC=true
```

The diagnostic evaluates candidate and structural-control units using observed singleton flip/non-flip outcomes and endogenous scalar features. It records per-baseline unit tests, activation/flip rows, testability summaries, and binned response curves.

Aggregate files include:

```text
aggregate_flip_stats.csv
aggregate_unit_tests.csv
aggregate_population_summary.csv
aggregate_binned_curves.csv
aggregate_activation_flip_rows.csv
threshold_spiking_experiment_aggregate.json
```

These files are persistent RQ3 analysis inputs used by the reporting pipeline.

Principal runtime controls are:

```text
THRESHOLD_EVENT_TARGET
THRESHOLD_EVENT_MAX_POINTS
THRESHOLD_EVENT_MIN_POINTS
THRESHOLD_EVENT_REPEATS
THRESHOLD_EVENT_HOLDOUT_FRACTION
THRESHOLD_EVENT_N_BINS
THRESHOLD_EVENT_SEED
FORCE_THRESHOLD_EVENT_POSTHOC
```

## Stage 08 — simultaneous-set and interaction validation

Module:

```text
pipeline.stage08_validate_interactions
```

Default controls:

```text
RUN_INTERACTION_VALIDATION=true
RUN_CMC=true
RUN_PREEMPTION=true
```

Stage 8 evaluates the frozen candidate set and structurally matched comparison sets with genuine simultaneous interventions. Its core products are:

1. the simultaneous full-set effect `E(J)`;
2. the matched non-candidate `E(K_b)` distribution;
3. the example-level singleton-union versus full-set decomposition;
4. conditional marginal contribution when CMC is enabled;
5. the retained dominant-secondary preemption assay when preemption is enabled.

### Singleton-versus-joint decomposition

Stage 8 combines the Stage-7 singleton flip masks with the already-computed full-set output. On the common complete-case singleton population it writes:

```text
composition_example_decomposition.csv
composition_decomposition_summary.csv
composition_decomposition_summary.json
```

Each example is classified as `preserved`, `suppressed`, `coalition_only`, or `unaffected`. The summary verifies

```text
E(J) - U(J) = P(coalition_only) - P(suppressed).
```

No additional model intervention is required for this decomposition.

### Conditional marginal contribution

When CMC is enabled, matched background sets are evaluated jointly with the candidate and null sets. Candidate, null, and background sets use the same evaluation rows and genuine simultaneous interventions.

### Secondary preemption assay

The existing preemption experiment remains available. Pair identity is frozen from Stage-6 discovery data only. Directional candidate provenance defines the 1→0 or 0→1 pool, and the frozen discovery score determines dominant-secondary ordering. Held-out singleton rates are descriptive and do not select pairs.

Preemption requires Stage-7c threshold diagnostics because the assay conditions the secondary binary marginal effect on an independently fitted endogenous dominant-event indicator. Runtime controls are:

```text
PREEMPTION_MIN_DISCOVERY_SCORE=0.05
PREEMPTION_MAX_SECONDARIES=8
PREEMPTION_THRESHOLD_HOLDOUT_FRACTION=0.25
PREEMPTION_THRESHOLD_MIN_CLASS=8
```

`PREEMPTION_MIN_SINGLETON_RATE` remains accepted as a compatibility alias for the discovery-score cutoff.

The group-intervention cache is incremental. A Stage-8 refresh can reuse compatible full-set, null-set, background, and pair outputs and write updated derived summaries without repeating already cached group evaluations.

## Reporting after model-backed execution

The numbered pipeline writes persistent experiment artifacts under `data/`. Manuscript aggregation is performed separately by:

```text
reporting.generate_final_results
```

Reporting uses separate modules for the primary interaction and threshold analyses:

```text
studies.overtopping.analysis.stage09_composition_decomposition_report
studies.overtopping.analysis.stage07_overtopping_spiking_report
studies.overtopping.analysis.stage08_threshold_shape_validation
studies.overtopping.analysis.stage08_graded_agonist_report
studies.overtopping.analysis.stage09_preemption_report
```

The composition reporter aggregates the RQ2 example-level decomposition. The threshold and graded reporters provide the primary RQ3 analyses. The preemption reporter is retained as a secondary subtype analysis.

## Cache and path rules

- Persistent scientific outputs belong under `data/`.
- Reusable computation caches belong under `cache/` or declared experiment cache directories.
- Derived reporting products belong under `results/`.
- Cache lookup occurs after scientific configuration and population are determined.
- Evaluation split and non-default population caps must resolve to distinct scientific/derived identities where they change the computation.
- Regenerating reports does not require deletion of valid model-backed caches.

## Batch size

Batch-size controls affect execution throughput and memory use. They should not change the selected scientific population. When reducing batch size for memory reasons, keep splits, row caps, candidate definitions, and intervention settings fixed.
