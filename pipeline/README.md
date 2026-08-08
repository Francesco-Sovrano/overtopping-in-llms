# Numbered causal-intervention pipeline

The numbered pipeline turns task examples into a frozen candidate channel set, evaluates the candidates on a selectable evaluation split, and launches interaction-aware simultaneous-set validation. `pipeline/_run_pipeline.sh` is the standard single-configuration orchestrator used by the experiment catalogue.

## Evaluation-split contract

Stage 7 and interaction validation accept `--evaluation_split test|train|all`; `test` is the default. Stage 2 creates the deterministic row-level `is_test` assignment used by the `test` and `train` choices.

The data-flow contract is:

1. stage 2 creates `is_test`;
2. discovery/candidate-selection computations use discovery/training rows;
3. mean/donor replacement-reference activations are estimated from training rows;
4. stage 7 evaluates singleton interventions on the selected evaluation split;
5. final singleton-set statistics use the same selected split;
6. interaction validation receives the same `--evaluation_split` and uses it for simultaneous candidate and matched-control interventions.

Split-specific stats-directory labels are:

```text
test   -heldout_test
train  -eval_train
all    no split suffix
```

`evaluation_scope.json` records the actual `sampling_pool_split` and `final_statistics_split`. The compatibility flag `--holdout_test_only` is an alias for `--evaluation_split test`.

## Direct wrapper usage

```bash
bash pipeline/_run_pipeline.sh <task> <model> [options]
```

The shell wrapper invokes numbered stages with `python3 -m pipeline.<module>`. When running a Python stage manually, use the same module form from the repository root; the code does not alter `sys.path` at runtime.

Example:

```bash
bash pipeline/_run_pipeline.sh \
  arithmetic \
  Qwen/Qwen2.5-1.5B-Instruct \
  --spectral_splits \
  --fast_anchoring \
  --z_thresh 10 \
  --batch_size 32 \
  --circuit_level neuron \
  --circuit_size 200000 \
  --eval_intervention mean-donor \
  --min_flip_rate 0.3 \
  --decode_only \
  --max_number_of_circuits_to_analyze 1
```

The wrapper resolves its repository root from its own location, activates `<repo>/.env/` when present, and uses task module `lib.tasks.<task>_task`.

## Wrapper options

### Feature filtering

```text
--z_thresh <value>
```

A non-negative value enables MAD-variance filtering in stage 2. A negative value leaves that filter disabled.

### Circuit/intervention configuration

```text
--max_number_of_circuits_to_analyze <n>
--eval_intervention zero|mean|mean-positional|mean-donor|mean-donor-positional
--batch_size <n>
--circuit_level neuron|edge
--circuit_size <n>
--min_flip_rate <rate>
```

Wrapper defaults are `mean-positional`, batch size 256, neuron level, circuit size 100000, minimum flip rate 0.2, and all circuits (`-1`). The experiment catalogue overrides these values through each `RunSpec`.

### Split/discovery/anchoring modes

```text
--spectral_splits
--spectral_anchoring_plan | --random_anchoring_plan
--spectral_circuit_discovery | --random_circuit_discovery
--fast_anchoring | --slow_anchoring
```

`--spectral_splits` bypasses rule-conditioned stages 3 and 4 and is mutually exclusive with the explicit plan/discovery switches. The standard experiment catalogue uses `--spectral_splits --fast_anchoring`.

### Other controls

```text
--incorrect_rules
--mlp_neurons_only
--decode_only
--no_llm_feature_generation
--output_data_dir <path>
--model_label <label>
--pipeline_cache_root <path>
--pipeline_model_cache_dir <path>
--evaluation_split test|train|all
--holdout_test_only
```

`--evaluation_split` defaults to `test`. `--holdout_test_only` remains an alias for `--evaluation_split test`.

## Runtime roots

Without overrides:

```text
data directory:   ./data/<task>/<model-label>
cache root:       ./cache/<task>
model cache:      ./cache/<task>/<model-label>
```

Hugging Face identifiers retain their `organization/model` path structure. Existing local model paths receive a sanitized filesystem label unless `--model_label` is specified.

The optional feature-proposal model is controlled by:

```bash
export FEATURE_GENERATION_LLM=gemma3:27b
```

Disable external feature proposal with `--no_llm_feature_generation` or `NO_LLM_FEATURE_GENERATION=true`.

## Stage 1 — generate prompts and answers

`1_generate_prompts_and_answers.py`:

- asks the task module for examples;
- queries the analyzed model;
- parses task outputs;
- computes task-level statistics;
- caches prompt/answer records.

The wrapper stores the prompt/answer pickle under the per-model cache and writes task statistics into `feature_report/`.

## Stage 2 — generate features

`2_generate_features.py`:

- loads cached task/model responses;
- adds task-provided seed features;
- optionally asks the configured feature model for additional features;
- evaluates feature values;
- removes near-duplicates and low-predictive-power features;
- optionally applies high-MAD-variance filtering;
- writes the feature table used by later stages.

Principal files in the shared feature-report directory are:

```text
<data-dir>/feature_report/dataset_stats.json   # stage 1 task statistics
<data-dir>/feature_report/scores.csv           # stage 2 feature table + is_test
<data-dir>/feature_report/features.json         # stage 2 feature definitions
```

The deterministic `is_test` column in `scores.csv` defines the `test` and `train` evaluation populations used by later stages. If `scores.csv` already exists, the wrapper skips stage 2.

## Stage 3 — extract rules

`3_extract_rules.py` learns symbolic feature rules and rule combinations from stage-2 features. It is used by rule-conditioned workflows and skipped by `--spectral_splits`.

Rule outputs live under:

```text
<data-dir>/rule_extraction_results/
```

## Stage 4 — build spectral sampling plans

`4_spectral_sample_datapoints.py` builds representative data-selection plans used by spectral anchoring/discovery paths. In spectral-split mode, rule-indexed sampling plans are not used and the wrapper skips this stage.

When rule-based modes require plans, the files are stored under the circuit-discovery output tree; hidden-representation caches are stored under the configured cache root.

## Stage 5 — discover circuits

`5_discover_circuits.py` performs EAP/EAP-IG attribution and selects circuit components. The wrapper uses:

```text
method:       EAP-IG-inputs
intervention: patching for circuit discovery
```

The evaluation replacement baseline passed to the stage is derived from the requested `--eval_intervention`.

Principal directory:

```text
<data-dir>/neural_circuit_discovery_results/eap_ig_inputs/
  <circuit-label>/neural_circuits/
```

Important files include:

```text
manifest.json
dataset_info.json
```

The manifest defines the prespecified channel populations used by interaction validation.

## Stage 6 — select candidate channels

`6_analyze_bag_of_rules.py` performs channel-level interventions on discovery data and identifies agonist channels whose effects meet `--min_flip_rate` (`search_epsilon`). Standard runs analyze both positive and negative baseline subsets unless `ANALYZE_BASELINE_SUBSETS` is changed.

The wrapper uses stage-6 outputs to define the fixed candidate set `J`. Candidate ordering for later `H_m`/`J_{l,m}` calculations is frozen from discovery statistics; stage-7 evaluation effects do not reorder candidates.

Useful environment controls:

```bash
export ANALYZE_BASELINE_SUBSETS=positive,negative
export POINTS_TO_USE_FOR_MEAN_ABLATION=256
export MAX_POINTS_PER_CIRCUIT=128
export MAX_POINTS_PER_ABLATION=64
```

For donor-style interventions the wrapper may raise the mean-reference point count to satisfy the donor reference requirements encoded in the script.

## Stage 7 — singleton refinement

`7_refine_neuron_anchored_rules.py` evaluates each fixed candidate individually on the requested evaluation split, materializes flip events, and computes singleton-set metrics. `test` is the default.

The wrapper passes:

```text
--exclude_discovery_rows_from_final_stats
--evaluation_split <test|train|all>
```

and uses the configured intervention, phase, mean-reference point count, neuron batch size, and spectral sampling cache.

Key environment controls:

```bash
export RUN_REFINE_NEURON_RULES=true
export REFINE_EXTRACT_RULES=true
export REFINE_SUMMARIZE_RULE_METRICS=true
export REFINE_MAX_NEURONS=0
export REFINE_NEURON_BATCH_SIZE=8
export REFINE_SAMPLING_MAX_POINTS=10000
export SKIP_AGONIST_METRIC_STATS=false
export FORCE_STAGE7=false
```

### Stage-7 output files

Within

```text
<data-dir>/rule_extraction_results/neuron_flip_rules/stats/<stats-run>/
```

the main files are:

```text
flip_stats_global.json
flip_stats_by_neuron.csv
scores.csv
evaluation_scope.json
frozen_candidate_ranking.csv
singleton_channel_metrics.csv
singleton_set_metrics.csv
singleton_set_metrics.json
frozen_topm_metrics.csv
singleton_threshold_counts.csv
frozen_topm_toc.pdf
frozen_topm_toc.png
```

`flip_stats_by_neuron.csv` is the per-channel singleton table. `scores.csv` materializes the per-example flip columns for the selected evaluation split. `frozen_candidate_ranking.csv` stores the discovery-frozen candidate order.

`singleton_set_metrics.json` declares `definition_version=heldout-set-metrics-v2`. The schema uses one common complete-case universe within the selected evaluation split for all singleton probabilities and reports the number of rows excluded because one or more required singleton-event values were missing.

`evaluation_scope.json` records split provenance, candidate-discovery split, replacement-reference split, final-statistics split, intervention baseline, phase, and ranking file.

## Singleton metric definitions

For the fixed candidate set `J`, stage 7 reports:

- `J`: candidate cardinality;
- `s_j`: singleton flip rate for each candidate;
- `s_(1)`: maximum singleton rate;
- `U(J)`: union of singleton flip events;
- `TOC_m`: `U(H_m)/U(J)` for discovery-frozen top-`m` subsets;
- `R_ov = 1 - U(J)/sum_j s_j`;
- `N_eff = (sum_j s_j)^2 / sum_j s_j^2`;
- `OCC_0` and `OCC_1`: union coverage conditioned on the unablated predicate;
- `N_t`: number of candidates with `s_j >= t` for the configured thresholds.

Undefined denominators produce explicit statuses. Finite values are not clipped.

## Interaction validation after stage 7

When enabled and the required stage-5/stage-7 files exist, `_run_pipeline.sh` calls `analysis/26_validate_interactions.py`.

Environment controls:

```bash
export RUN_INTERACTION_VALIDATION=true
export INTERACTION_NULL_DRAWS=100
export INTERACTION_M_VALUES=1,2,3,5
export INTERACTION_DENOMINATOR_EPSILON=1e-12
export FORCE_INTERACTION_VALIDATION=false
```

The validator computes:

- simultaneous `E(J)`;
- `E_l(C_l)` for every prespecified layer population;
- `E_l(C_l \ J_{l,m})`;
- `Delta_l` and `GCCR_m`;
- exact-stratum matched random candidate sets;
- simultaneous null `E(J)` and GCCR values;
- `Delta`, `P`, and `p_MC` summaries.

The main output directory is:

```text
<stats-run>/interaction_validation/
```

with configuration, population, null-membership, raw null-draw, summary, LaTeX, Markdown, and matched-null plot files.

## Completion and cache reuse

The wrapper checks stage completion before performing expensive work.

Stage 7 is treated as current when all of the following exist:

```text
flip_stats_global.json
flip_stats_by_neuron.csv
scores.csv
evaluation_scope.json
frozen_candidate_ranking.csv
singleton_set_metrics.json with definition_version=heldout-set-metrics-v2
```

If the first three runtime singleton files exist and `scores.csv` contains `is_test`, but the current sidecars are absent, the wrapper invokes stage 7 with `--stats_only`. That path rebuilds singleton summaries from materialized split-specific events without replaying singleton model ablations.

Interaction validation writes `interaction_configuration.json` with file fingerprints and configuration fields. A matching completed summary is reused before model loading. `FORCE_INTERACTION_VALIDATION=true` bypasses that reuse.

A compact filtered export does not include stage-7 `scores.csv` and therefore is not a pipeline-resume tree. It is intended for model-free reporting only.

## Cache safety

Do not reuse a stage-2 or later cache after changing an input that changes row identity, split assignment, task semantics, model revision, or intervention configuration unless the corresponding cache key/fingerprint accounts for the change. The safest approach for materially different experiment definitions is a separate `--output_data_dir`, `--pipeline_cache_root`, or `--pipeline_model_cache_dir`.
