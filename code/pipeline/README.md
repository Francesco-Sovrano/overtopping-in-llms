# Numbered causal-intervention pipeline

The standard per-configuration orchestrator is:

```bash
cd code
bash pipeline/_run_pipeline.sh <TASK> <MODEL> [options]
```

The experiment catalogue calls this wrapper automatically. Direct use is useful for custom configurations.

## Default wrapper configuration

```text
split mode                 rules
sampling plan              random
circuit discovery          random
anchoring                   fast
z_thresh                    -1
max circuits                -1 (all)
evaluation intervention     mean-positional
batch size                  256
circuit level               neuron
circuit size                100000
min flip rate (tau)         0.2
evaluation split            test
```

These are wrapper defaults, not necessarily catalogue defaults.

## Evaluation split

Accepted values:

```text
test
train
all
```

`test` is the default. The wrapper passes the chosen split to stage 7 and simultaneous conditional validation.

Stats-directory suffix:

```text
test   -heldout_test
train  -eval_train
all    no evaluation suffix
```

## Important wrapper options

```text
--z_thresh N
--max_number_of_circuits_to_analyze N
--eval_intervention NAME
--batch_size N
--circuit_level neuron|edge
--circuit_size N
--min_flip_rate R
--evaluation_split test|train|all
--spectral_splits
--spectral_anchoring_plan | --random_anchoring_plan
--spectral_circuit_discovery | --random_circuit_discovery
--fast_anchoring | --slow_anchoring
--incorrect_rules
--mlp_neurons_only
--decode_only
--no_llm_feature_generation
--output_data_dir DIR
--model_label LABEL
--pipeline_cache_root DIR
--pipeline_model_cache_dir DIR
--task_module PYTHON_MODULE
```


## Task modules

The wrapper normally resolves the task implementation as:

```text
lib.tasks.<experiment_name>_task
```

Use `--task_module` when an experiment intentionally lives outside `lib.tasks`. The poisoning workflows use task specifications under `poisoning.tasks`, for example:

```bash
bash pipeline/_run_pipeline.sh grammar_backdoor_lift Qwen/Qwen2.5-1.5B-Instruct \
  --task_module poisoning.tasks.grammar_backdoor_lift_task \
  ...

bash pipeline/_run_pipeline.sh arithmetic_backdoor_lift Qwen/Qwen2.5-1.5B-Instruct \
  --task_module poisoning.tasks.arithmetic_backdoor_lift_task \
  ...
```

The poisoning shell scripts pass this option explicitly. This avoids relying on the standard `lib.tasks` naming convention for poisoning-specific behaviors.

## Runtime roots

The wrapper resolves:

```text
CODE_ROOT     <repo>/code
PROJECT_ROOT  <repo>
```

Default state remains outside `code/`:

```text
<repo>/data
<repo>/cache
<repo>/.env
```

## Stage 1 — prompts and answers

`pipeline/1_generate_prompts_and_answers.py`

Responsibilities:

- instantiate the selected `TASK_SPEC`;
- create/load examples;
- generate analyzed-model completions;
- parse the task predicate/correctness;
- write task/model I/O caches and dataset statistics.

## Stage 2 — features

`pipeline/2_generate_features.py`

Responsibilities:

- compute task-native features;
- optionally propose additional features with an LLM/Ollama model;
- score features on examples;
- apply configured filtering;
- materialize the table consumed by downstream discovery/analysis.

## Stage 3 — rules

`pipeline/3_extract_rules.py`

Extracts symbolic feature rules and rule combinations. It is skipped in `--spectral_splits` mode.

## Stage 4 — sampling plan

`pipeline/4_spectral_sample_datapoints.py`

Creates representative sampling plans used for rule-based circuit discovery. It is skipped in spectral-split mode.

## Stage 5 — circuit discovery

`pipeline/5_discover_circuits.py`

Runs EAP/EAP-IG attribution at the selected circuit granularity and writes the eligible circuit population plus manifests used by later model interventions.

Important output area:

```text
<data>/<task>/<model>/neural_circuit_discovery_results/eap_ig_inputs/<circuit-label>/
```

The `neural_circuits/` subdirectory contains `manifest.json` and `dataset_info.json` required by model-backed simultaneous validation.

## Stage 6 — candidate selection

`pipeline/6_analyze_bag_of_rules.py`

Evaluates discovery/training effects for candidate channels and creates the frozen candidate set used by stage 7. Candidate ranking for `H_m` is based on discovery-side stage-6 effect information, not evaluation-side singleton rates.

## Stage 7 — singleton evaluation

`pipeline/7_refine_neuron_anchored_rules.py`

For each frozen candidate channel:

1. evaluate the unablated task predicate;
2. suppress that channel using the selected replacement baseline/phase;
3. record per-example flip events;
4. aggregate singleton directional and union summaries;
5. write the discovery-frozen ranking;
6. compute exact singleton-set statistics.

### Stage-7 metric files

```text
flip_stats_global.json
flip_stats_by_neuron.csv
scores.csv
frozen_candidate_ranking.csv
singleton_set_metrics.json
singleton_set_metrics.csv
singleton_topm_metrics.csv
singleton_threshold_counts.csv
```

The singleton metric schema is `heldout-set-metrics-v2`.

### Common evaluation universe

The exact set metrics use only rows where every candidate singleton has a materialized evaluation. This common complete-case mask is used for all `s_j`, `U(J)`, and `U(H_m)` values in the run. The number of excluded rows is stored in `singleton_set_metrics.json`.

### Statistics-only regeneration

When singleton flip columns already exist, stage 7 supports:

```text
--stats_only
```

This skips model ablations and rebuilds aggregate metric sidecars from materialized singleton events plus the discovery-ranking source.

## Exact singleton statistics

For candidate set `J` and singleton flip events `F_j`:

```text
s_j = P(F_j)
s_(1) = max_j s_j
U(A) = P(union_{j in A} F_j)
TOC_m = U(H_m)/U(J)
R_ov = 1 - U(J)/sum_j s_j
N_eff = (sum_j s_j)^2 / sum_j s_j^2
OCC_b = P(union_{j in J} F_j | B(x)=b)
N_t = |{j : s_j >= t}|
```

`H_m` is frozen from discovery data. Ratios are not clipped. Near-zero denominators yield explicit undefined statuses.

`E(J)` is not defined by singleton unioning; it is computed separately by simultaneous intervention.

## Simultaneous and conditional validation

After stage 7, the wrapper runs `analysis.26_validate_interactions` when:

```text
RUN_INTERACTION_VALIDATION=true
```

and the required singleton/ranking files exist.

Relevant environment variables:

```text
RUN_CMC                            default: true
INTERACTION_NULL_DRAWS             default in wrapper: 100
CONDITIONAL_BACKGROUND_MULTIPLIERS default: 1
FORCE_INTERACTION_VALIDATION       default: false
```

The root `run_experiments.sh` defaults `INTERACTION_NULL_DRAWS` to `30` and respects a caller-provided value.

The validator always computes genuine simultaneous:

```text
E(J)
E(K_b)
```

With `RUN_CMC=true` it additionally computes:

```text
E(S_b)
E(S_b union J)
E(S_b union K_b)
```

and the paired conditional marginal statistic documented in `../analysis/README.md`. Set `RUN_CMC=false` to retain `E(J)` and matched-null validation while skipping all CMC background interventions.

## Other pipeline environment controls

The wrapper also supports controls such as:

```text
RUN_REFINE_NEURON_RULES
RUN_THRESHOLD_EVENT_POSTHOC
REFINE_EXTRACT_RULES
REFINE_SUMMARIZE_RULE_METRICS
REFINE_MAX_NEURONS
REFINE_NEURON_BATCH_SIZE
REFINE_SAMPLING_MAX_POINTS
SKIP_AGONIST_METRIC_STATS
ANALYZE_BASELINE_SUBSETS
FORCE_STAGE7
NO_LLM_FEATURE_GENERATION
```

Inspect `_run_pipeline.sh` for the exact current defaults before launching expensive custom runs.

## Cache safety

The pipeline uses explicit completion files and metric schema versions. Do not manually copy a completion sentinel between incompatible model/task/intervention configurations. The simultaneous validator additionally fingerprints the evaluation rows and relevant content/configuration.
