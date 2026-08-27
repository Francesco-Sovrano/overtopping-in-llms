# Numbered causal-intervention pipeline

This page documents the shared per-configuration causal-intervention pipeline coordinated by `pipeline/run_pipeline.sh`. The overtopping catalogue uses it for ordinary task/model configurations, and the poisoning workflow reuses it for checkpoint-specific attack-cohort control-correctness causal discovery. It can also be called directly for custom runs.

The standard per-configuration orchestrator is:

```bash
cd code
bash pipeline/run_pipeline.sh <TASK> <MODEL> [options]
```

The experiment catalogue calls this wrapper automatically. Direct use is useful for custom configurations.

## Cache names for local checkpoints

Local checkpoint paths are never flattened into cache directory or file names. When no explicit `--model_label` is supplied, the runner derives a readable semantic identifier from the checkpoint name and condition when the path uses a `clean/checkpoints/...` or `poisoned/checkpoints/...` layout. Representation caches use the same rule. No absolute-path encoding or hash suffix is added; for example, `poisoned/checkpoints/progress_010pct__step_0025` becomes `poisoned_progress_010pct__step_0025`.

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
evaluation baseline subset   all
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

Stage 7 can also condition the selected split on the unablated binary predicate:

```text
--evaluation_baseline_subset all|positive|negative
```

`all` is the generic default. `positive` keeps only rows whose baseline predicate
is true; this is the setting used by trigger-lift-conditioned poisoning so its
singleton denominators contain only pre-intervention trigger-lift successes. A
non-`all` baseline subset is included in the stats-directory and ablation-cache
identity.

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
--evaluation_baseline_subset all|positive|negative
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
--task_module PYTHON_MODULE[:ATTRIBUTE]
```


## Task modules

The wrapper normally resolves the task implementation as:

```text
core.tasks.<experiment_name>_task
```

Use `--task_module` when an experiment intentionally lives outside `core.tasks`. The resolver accepts either a module name, which selects that module's `TASK_SPEC`, or `module:attribute`, which selects a named task-spec object. Normal tasks use the former. The poisoning tasks use named attributes to separate the backdoor behavior task, full-cohort normal-task behavior, and attack-cohort control-correctness causal task; for example, `BACKDOOR_TASK_SPEC`, `NORMAL_TASK_SPEC`, and `ATTACK_COHORT_CONTROL_CORRECTNESS_SPEC`. `ORDINARY_TASK_SPEC` remains a compatibility alias for the attack-cohort causal task.

Poisoning defines paired trigger-marker/control-marker task specifications under
`studies.poisoning.tasks`. Its launcher calls this same numbered pipeline
in `--spectral_splits` mode **after training is complete**, once for every
post-training checkpoint in both clean and poisoned trajectories.

CHA's statistical operating point can be configured generically with `CHA_REFERENCE_N_PER_SIDE`, `CHA_TAU`, `CHA_LOW_DATA_POLICY`, `CHA_MIN_ACTUAL_N_PER_SIDE`, and `CHA_PRUNE_ALPHA`. When these are explicitly exported, `run_pipeline.sh` uses `CHA_TAU` as the default `--min_flip_rate`, maps `CHA_REFERENCE_N_PER_SIDE` to the reference sample size used by finite-sample UCB calibration, and stage 6 honors the requested low-data policy for under-sized per-circuit samples. Without those variables, the generic pipeline uses its built-in defaults.

Stage 7 has one evaluation-size control: `--sampling_max_points`, supplied by `run_pipeline.sh` from `REFINE_SAMPLING_MAX_POINTS` (10,000 by default). The same cap is used in both modes. With explicit spectral sampling enabled it bounds the spectral sample. With spectral sampling disabled it bounds the selected evaluation pool directly, after `--evaluation_split` and `--evaluation_baseline_subset`, by taking a deterministic seeded uniform sample without replacement from the selected pool.

Poisoning additionally has a trigger-lift candidate-acquisition ceiling, `TRIGGER_LIFT_SCAN_MAX_ROWS`. It defaults to `REFINE_SAMPLING_MAX_POINTS`, so both are 10,000 unless the trigger-lift scan is overridden separately. Grammar and arithmetic candidate pools are deterministically shuffled by their task seed before this cap is applied; the capped causal scan therefore uses a fixed prefix of that seeded order at every checkpoint. Spectral representations are not used to choose causal-scan candidates. Stage 7 reports the actual denominator and exact binomial confidence intervals.

For poisoning spectral runs, stage 2 is replaced by direct export of the
TransformerLens behavioral table, stage 3 symbolic rule extraction is skipped,
and stage 4 rule-indexed sampling is skipped. Stages 5–7 remain active because
the primary poisoning experiment must discover a new checkpoint-specific causal
set rather than reuse a fixed candidate set. See [Poisoning protocol](poisoning-protocol.md) for the complete ordering, trigger-lift definition, held-out split, and downstream experiments.

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

`pipeline/stage01_generate_prompts_and_answers.py`

Responsibilities:

- resolve the selected task spec (`TASK_SPEC` by default, or an explicit `module:attribute` object);
- create/load examples;
- generate analyzed-model completions;
- parse the task predicate/correctness;
- write task/model I/O caches and dataset statistics.

When the requested model-I/O cache already exists, Stage 1 loads it instead of invoking the model. A task may expose `validate_generated_cache()` to reject an incompatible cache before reuse; poisoning task specs use this to check checkpoint/cohort/marker provenance. For a valid poisoning cache, statistics and the poisoning dataset-only `scores.csv` export are produced from the already-loaded object and the pickle is not rewritten.

All external-provider calls made through `instruct_model()` use the same recovery policy, independent of task or pipeline stage. This includes API-backed task labels/classifiers and API-backed feature proposal. OpenAI and Groq requests retry transient rate limits, timeouts, connection failures, and 5xx responses at the individual-request layer with bounded exponential backoff and `Retry-After` support. Remote concurrency is bounded by `API_MAX_WORKERS`.

Successful provider responses are registered for persistent caching as soon as each request finishes, rather than only after the entire batch completes. The cache is written periodically and flushed on normal exit, SIGINT, and SIGTERM. If some prompts remain unresolved after their individual request retries, the shared provider layer performs additional recovery passes over only those unresolved prompts. By default an external-provider batch is strict: if requests are still unresolved after recovery, the call raises instead of allowing a higher-level dataset or feature artifact to be saved with missing API results. Re-running the command reuses all successful provider-cache entries and requests only the unresolved prompts.

The request controls are `API_MAX_RETRIES` (default `6`), `API_RETRY_BASE_SECONDS` (`2`), `API_RETRY_MAX_SECONDS` (`60`), `API_RETRY_JITTER_SECONDS` (`1`), and `API_MAX_WORKERS` (`8`). Batch-level recovery is controlled by `API_RECOVERY_PASSES` (default `2` additional passes), `API_RECOVERY_BASE_SECONDS` (default `15`), and `API_RECOVERY_MAX_SECONDS` (default `60`). The jailbreak classifier may permit an incomplete *primary* classifier batch only long enough to route those specific rows to its configured fallback classifier; the fallback is strict. Recovery and persistence themselves are shared infrastructure, not task-specific hooks.

## Stage 2 — features

`pipeline/stage02_generate_features.py`

Responsibilities:

- compute task-native features;
- optionally propose additional features with an LLM/Ollama model;
- score features on examples;
- apply configured filtering;
- materialize the table consumed by downstream discovery/analysis.

## Stage 3 — rules

`pipeline/stage03_extract_rules.py`

Extracts symbolic feature rules and rule combinations. Stage 2 must provide the `is_test` split column; Stage 3 fits its rule models on the training rows and refuses stale score tables that predate the split-aware schema. It is skipped in `--spectral_splits` mode.

## Stage 4 — sampling plan

`pipeline/stage04_spectral_sample_datapoints.py`

Creates representative sampling plans used for rule-based circuit discovery. It is skipped in spectral-split mode.

## Stage 5 — circuit discovery

`pipeline/stage05_discover_circuits.py`

Runs EAP/EAP-IG attribution at the selected circuit granularity and writes the eligible circuit population plus manifests used by later model interventions.

Important output area:

```text
<data>/<task>/<model>/neural_circuit_discovery_results/eap_ig_inputs/<circuit-label>/
```

The `neural_circuits/` subdirectory contains `manifest.json` and `dataset_info.json` required by model-backed simultaneous validation.

## Stage 6 — candidate selection

`pipeline/stage06_analyze_bag_of_rules.py`

Evaluates discovery/training effects for candidate channels and creates the frozen candidate set used by stage 7. Candidate ranking for `H_m` is based on discovery-side stage-6 effect information, not evaluation-side singleton rates.

## Stage 7 — singleton evaluation

`pipeline/stage07_refine_neuron_anchored_rules.py`

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
singleton_channel_metrics.csv
frozen_topm_metrics.csv
singleton_threshold_counts.csv
```

The singleton metric schema is `heldout-set-metrics-v2`. Rule-combination summaries use explicit score-scope filenames rather than unsuffixed aliases:

```text
rule_combo_metrics_all_scopes.csv
rule_combo_metrics_test_all.csv
rule_combo_metrics_test_best_per_neuron.csv
rule_combo_metrics_test_selected_all.csv
rule_combo_metrics_test_selected_best_per_neuron.csv
rule_combo_metrics_all_fit_all.csv
rule_combo_metrics_all_fit_best_per_neuron.csv
rule_metrics_summary_<scope>.json
rule_metrics_<scope>.tex
rule_metrics_distributions_<scope>.pdf
```

The `test` scope scores the TRAIN-selected combination on TEST, `test_selected` selects and scores the combination on TEST, and `all_fit` is descriptive selection/scoring on the pooled fit universe.

### Common evaluation universe

The exact set metrics use only rows where every candidate singleton has a materialized evaluation. This common complete-case mask is used for all `s_j`, `U(J)`, and `U(H_m)` values in the run. The number of excluded rows is stored in `singleton_set_metrics.json`.

### Statistics-only regeneration

The wrapper variable `REFINE_USE_SPECTRAL_SAMPLING` controls whether stage 7
constructs a representation-based sample of the declared evaluation split. It
defaults to `true` for generic experiments. Setting it to `false` evaluates the
complete selected split directly; the poisoning workflow uses this mode for
held-out singleton re-estimation.

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

## Stage 8 — simultaneous and conditional validation

After Stage 7, the wrapper runs `pipeline.stage08_validate_interactions` when:

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

The pipeline wrapper defaults `INTERACTION_NULL_DRAWS` to `100`. Any explicit environment value supplied by a caller takes precedence, so record the effective environment for expensive interaction-validation runs.

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

and the paired conditional marginal statistic documented in [Analysis and final-result generation](overtopping-analysis.md). Set `RUN_CMC=false` to retain `E(J)` and matched-null validation while skipping all CMC background interventions.

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

Inspect `run_pipeline.sh` for the exact current defaults before launching expensive custom runs.

## Cache safety

The pipeline uses explicit completion files and metric schema versions. Do not manually copy a completion sentinel between incompatible model/task/intervention configurations. The simultaneous validator additionally fingerprints the evaluation rows and relevant content/configuration.
