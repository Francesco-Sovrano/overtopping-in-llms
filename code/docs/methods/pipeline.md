# Numbered causal-intervention pipeline

The shared pipeline is implemented in `code/pipeline/`. Study-specific launchers configure it for overtopping and poisoning experiments.

## Configuration

A pipeline run is determined by task specification, analyzed model, intervention phase, replacement baseline, sampling configuration, discovery settings, evaluation split, and stage-specific thresholds. These values are encoded in configuration metadata and, where necessary, output paths.

## Stage 01 — prompts and answers

`stage01_generate_prompts_and_answers.py`

Creates task prompts, obtains model outputs or loads cached generations, computes task-level behavioral labels, and writes the score population used by later stages.

## Stage 02 — feature export

`stage02_generate_features.py`

Computes feature representations used for rule extraction and discovery. Feature generation can use local or provider-backed language models according to the configured task and feature model.

## Stage 03 — rule extraction

`stage03_extract_rules.py`

Fits and exports interpretable rule structures over the generated feature table.

## Stage 04 — spectral sampling plan

`stage04_spectral_sample_datapoints.py`

Constructs a bounded sampling plan for expensive causal evaluation. Sampling metadata records the population and method used to choose rows.

## Stage 05 — circuit discovery

`stage05_discover_circuits.py`

Runs attribution/circuit discovery over the selected population. EAP/EAP-IG implementation details are documented in [EAP / EAP-IG](eap.md).

## Stage 06 — candidate and rule analysis

`stage06_analyze_bag_of_rules.py`

Builds candidate/rule summaries and the directional agonist buckets used by downstream singleton evaluation. Candidate membership records the source baseline subset:

```text
positive  -> 1→0 discovery population
negative  -> 0→1 discovery population
```

## Stage 07 — held-out singleton causal evaluation

`stage07_singleton_causal_evaluation.py`

Evaluates frozen candidates on the declared evaluation split and materializes row-level singleton intervention outcomes.

The historical module name `stage07_refine_neuron_anchored_rules.py` remains as a compatibility implementation only; Stage 7 does not perform rule extraction unless the optional legacy `REFINE_EXTRACT_RULES=true` add-on is explicitly enabled.

Important artifacts include:

```text
scores.csv
frozen_candidate_ranking.csv
flip_stats_by_neuron.csv
flip_stats_global.json
```

The frozen ranking and per-neuron statistics carry discovery-direction provenance. A candidate discovered in only one baseline subset remains restricted to that directional population in RQ3.

### Evaluation universe

Stage 7 owns the row identities on which candidate singleton outcomes are materialized. Downstream threshold analysis intersects its requested split and source-state subset with this Stage-7 universe rather than constructing a separate candidate evaluation population.

### Statistics-only regeneration

When row-level intervention columns already exist and only summary statistics are missing, the analysis utilities can rebuild statistics from the persisted Stage-7 score materialization without repeating model interventions.

## Stage 7b — RQ3 threshold-event diagnostics

Implemented in `studies/overtopping/analysis/threshold_event_diagnostics.py` and orchestrated across the manuscript population by `rebuild_spiking_diagnostics.py`.

For each configured RQ3 run and directional baseline subset, Stage 7b:

1. loads the frozen Stage-7 candidate ranking and direction membership;
2. includes only candidates discovered for that baseline subset;
3. reuses candidate singleton outcomes from the Stage-7 row universe;
4. samples same-layer/head non-candidate controls independently of intervention outcomes;
5. evaluates control interventions with the same replacement protocol;
6. records endogenous scalar features and intervention-defined flip labels;
7. performs repeated threshold diagnostics;
8. writes per-unit, population, binned-response, and raw activation/flip tables.

The experiment-level threshold-event schema is:

```text
threshold-event-v4-discovery-direction-aware
```

The principal per-baseline derived files include:

```text
threshold_spiking_experiment.json
high_n_scores_with_flips.csv
high_n_flip_stats_by_unit.csv
threshold_unit_tests.csv
threshold_population_summary.csv
threshold_binned_flip_curves.csv
threshold_activation_flip_rows.csv.gz
same_layer_nonagonist_control_pool.csv
same_layer_nonagonist_control_selection.csv
```

High-N model evaluations are cached separately from these derived tables.

## RQ3 graded agonist intervention

`studies/overtopping/analysis/graded_agonist_intervention.py`

This model-backed experiment starts from each frozen agonist's own held-out directional flip support and sweeps the strength of the same singleton intervention from 0 (natural activation) to 1 (the historical full replacement). It reports per-example crossing dose, persistence, reversal count, and single-crossing status. Optional same-agonist negative support uses source-state examples that the same agonist did not flip at full dose.

`studies/overtopping/analysis/stage08_graded_agonist_report.py` aggregates agonists within run/baseline/direction conditions for Figure 4. Nested held-out threshold `|MCC|` is no longer a primary RQ3 endpoint.

## Stage 08 — interaction validation

`stage08_validate_interactions.py`

Evaluates the full candidate set and matched non-candidate sets, computes simultaneous-set and conditional interaction quantities, and can run dominant-secondary preemption experiments.

The interaction stage uses its own group-evaluation cache because the expensive object is a multi-channel intervention rather than an individual singleton evaluation.

### Preemption experiment

Preemption candidate pools are restricted to the candidate's frozen discovery direction. The dominant threshold event is fitted on a threshold-training split; signed training MCC determines whether the raw threshold predicate must be inverted before evaluating event-present versus event-absent subsets.

The direction-aware interaction schema is:

```text
conditional-marginal-validation-v2-direction-aware-preemption
```

`stage09_preemption_report.py` aggregates pair-level results to run × baseline × direction conditions.

## Cache and path rules

- Persistent scientific outputs belong under `data/`.
- Regenerable reporting products belong under `results/`.
- Expensive reusable computations belong in cache directories.
- Cache lookup occurs after the scientific population and method configuration are determined.
- Distinct evaluation splits and non-default point caps must resolve to distinct derived paths.
- A schema mismatch requires regeneration of the derived products governed by that schema; it does not automatically require deletion of model-backed caches.

## Batch size

Batch-size controls change execution throughput and memory use but should not alter the selected scientific population. When reducing batch sizes to handle accelerator memory pressure, keep row caps, splits, candidate definitions, and intervention settings unchanged.
