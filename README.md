# Causal channel intervention pipeline

This repository contains an end-to-end experimental pipeline for identifying small sets of language-model channels that causally affect a binary task predicate, evaluating those channels on an explicitly selected data split, measuring interactions with simultaneous interventions, comparing the candidate mechanism with matched random channel sets, and producing manuscript-ready tables and figures. The default evaluation split is `test`; `train` and `all` remain available for diagnostics and controlled comparisons.

The standard non-poisoning experiment programme is defined in one file, `experiments/run_experiments.py`. The numerical implementation remains split into focused pipeline and analysis modules so that each step can be inspected independently.

## 1. What the pipeline measures

Let `B(x) ∈ {0,1}` denote the model's unablated predicate on an example `x` from the selected evaluation population (`test` by default). For a candidate channel `j`, let `F_j` be the evaluated examples whose predicate changes when only `j` is suppressed, and let

```text
s_j = P(F_j)
U(A) = P(union_{j in A} F_j)
```

For the frozen candidate set `J`, the singleton-based metrics are:

```text
s_(1) = max_j s_j
TOC_m(J) = U(H_m) / U(J)
R_ov(J) = 1 - U(J) / sum_j s_j
N_eff(J) = (sum_j s_j)^2 / sum_j s_j^2
OCC_b(J) = P(union_j F_j | B(x)=b),  b in {0,1}
N_t = |{j : s_j >= t}|
```

`H_m` is the top-`m` subset according to a ranking fixed on discovery data. Held-out singleton effects do not determine the ranking.

The pipeline also measures the effect of suppressing the complete set `J` simultaneously:

```text
E(J) = predicate-change rate on the selected evaluation split under simultaneous suppression of J
```

`E(J)` is a separate intervention quantity from `U(J)`. The code does not use a singleton union as a substitute for a simultaneous-set intervention.

For every prespecified stage-5 layer population `C_l`, and for the discovery-frozen subset `J_{l,m} ⊆ J ∩ C_l`, interaction validation computes

```text
Delta_l(J_{l,m}) = E_l(C_l) - E_l(C_l \ J_{l,m})
GCCR_m(J) = sum_l Delta_l(J_{l,m}) / sum_l E_l(C_l)
```

Every prespecified population contributes to the GCCR denominator, including layers for which `J_{l,m}` is empty. Ratios are not clipped. Negative values and values above one are retained because they can arise from cancellation or non-additive interactions. A zero or near-zero denominator is reported as undefined with an explicit status.

Matched random controls are sampled from the same intervention population and match the candidate by transformer layer, computational locus, channel type, intervention phase, per-layer cardinality, and replacement baseline. For both simultaneous `E(J)` and each `GCCR_m`, the report contains

```text
Delta = candidate - median(null)
P = (1 + count(null_b <= candidate)) / (B + 1)
p_MC = (1 + count(null_b >= candidate)) / (B + 1)
```

where `B` is the requested number of null draws.

## 2. Repository layout

```text
.
├── run_experiments.sh              # run the complete non-poisoning catalogue
├── generate_results.sh             # regenerate final paper outputs from data/
├── experiments/
│   ├── run_experiments.py          # complete executable experiment catalogue
│   ├── execution.py                # RunSpec, filtering, path and command construction
│   └── README.md
├── pipeline/
│   ├── 1_generate_prompts_and_answers.py
│   ├── 2_generate_features.py
│   ├── 3_extract_rules.py
│   ├── 4_spectral_sample_datapoints.py
│   ├── 5_discover_circuits.py
│   ├── 6_analyze_bag_of_rules.py
│   ├── 7_refine_neuron_anchored_rules.py
│   ├── _run_pipeline.sh
│   └── README.md
├── analysis/
│   ├── 12_threshold_event_diagnostics.py
│   ├── 21_analyze_primary_metrics.py
│   ├── 22_compute_group_dominance.py
│   ├── 23_compute_survey_dominance.py
│   ├── 24_run_primary_holdout_analysis.py
│   ├── 25_rebuild_directional_stats.py
│   ├── 26_validate_interactions.py
│   ├── 27_generate_manuscript_outputs.py
│   ├── 28_visualize_experiment_results.py
│   ├── 29_generate_final_results.py
│   ├── compute_overtopping_latex_tables.py
│   ├── make_competence_vs_overtopping_paper_figures.py
│   ├── generate_overtopping_spiking_report.py
│   ├── clean_results_for_export.py
│   ├── primary_matrix.py
│   └── README.md
├── lib/
│   ├── tasks/                       # standard task definitions
│   ├── eap/                         # attribution implementation
│   ├── heldout_set_metrics.py       # singleton set statistics
│   ├── group_intervention.py        # simultaneous intervention utilities
│   ├── interaction_statistics.py    # GCCR and matched-null statistics
│   └── shared feature/model/cache utilities
├── poisoning/                       # trigger-poisoning experiments and Slurm jobs
├── tests/                           # model-free policy/statistics tests
├── data/                            # raw experiment artifacts; created at runtime
├── cache/                           # model/prompt/feature/intervention caches; runtime
├── results/                         # all aggregate statistics and paper outputs
├── requirements.txt
└── setup.sh
```

`data/` and `cache/` contain experiment state. `results/` is reserved for aggregate statistics, manuscript tables, and figures.

## 3. Installation

The setup script uses Python 3.12:

```bash
bash setup.sh
source .env/bin/activate
```

Run repository Python entry points as modules from the repository root, for example:

```bash
python3 -m experiments.run_experiments --help
python3 -m analysis.26_validate_interactions --help
```

The code does not modify `sys.path` at runtime. Root shell launchers and internal subprocesses use `python -m ...` so imports are resolved through normal package semantics.

`requirements.txt` installs the analysis and mechanistic-interpretability stack, including PyTorch 2.10.0, Transformers 4.57.6, TransformerLens 2.17.0, SciPy 1.13, scikit-learn 1.6, SHAP 0.46, and XGBoost 2.1.3.

If `ollama` is installed, `setup.sh` downloads `gemma3:27b`, `qwen3:4b`, and `qwen3:14b`. Ollama is used only for optional feature proposal; standard task seed features remain available when feature proposal is disabled.

Fresh circuit-discovery and intervention experiments normally require a CUDA-capable GPU and enough storage for model weights, prompt caches, feature tables, attribution outputs, stage-7 per-example scores, and simultaneous-intervention caches.

Hugging Face caches can be redirected in the usual way:

```bash
export HF_HOME=/path/to/huggingface-cache
export TRANSFORMERS_CACHE=/path/to/huggingface-cache
```

API credentials, if needed by an optional feature/classification service, must be supplied through environment variables or the job scheduler. No credential is stored in the repository.

## 4. Standard tasks and data

The standard task modules live in `lib/tasks/` and implement the interface in `lib/task_spec.py`.

| Task identifier | Module | Source/configuration |
|---|---|---|
| `arithmetic` | `lib.tasks.arithmetic_task` | generated arithmetic examples |
| `grammar_acceptability` | `lib.tasks.grammar_acceptability_task` | local JSONL; `GRAMMAR_DATASET_PATH`, `GRAMMAR_NUM_EXAMPLES`, `GRAMMAR_TASK_SEED` |
| `hans_nli` | `lib.tasks.hans_nli_task` | HANS; `HANS_LOCAL_FILE` or dataset download, `HANS_SPLIT`, `HANS_NUM_EXAMPLES`, `HANS_TASK_SEED`, `HANS_BALANCE_LABELS`, `HANS_CACHE_DIR` |
| `random_fsm` | `lib.tasks.random_fsm_task` | generated FSM examples; `FSM_NUM_EXAMPLES`, `FSM_MIN_STATES`, `FSM_MAX_STATES`, `FSM_MIN_INPUT_LEN`, `FSM_MAX_INPUT_LEN`, `FSM_TASK_SEED` |
| `bon_jailbreaking` | `lib.tasks.bon_jailbreaking_task` | local prompt data via `AUGMENTED_PROMPTS_FILE`; classifier models via `BON_JAILBREAK_CLASSIFIER_MODEL` and `BON_JAILBREAK_CLASSIFIER_FALLBACK_MODEL` |

The grammar task defaults to a path under `data/grammar_acceptability/`; if the file is not present, set `GRAMMAR_DATASET_PATH` explicitly. The jailbreak task similarly expects a local augmented-prompt dataset unless `AUGMENTED_PROMPTS_FILE` is set.

## 5. Running the complete experiment programme

The recommended entry point is the root shell script:

```bash
./run_experiments.sh
```

It:

1. activates `.env/` when present;
2. runs the complete non-poisoning catalogue;
3. uses `test` as the evaluation split unless `EVALUATION_SPLIT` or `--evaluation-split` selects `train` or `all`;
4. for `test`, uses the `iclr-28` primary profile unless `PRIMARY_PROFILE` is set and generates manuscript outputs;
5. for `train` or `all`, writes catalogue summaries but skips the test-specific primary manuscript export;
6. writes aggregate/final outputs under root `results/`.

To use the 27-setting profile:

```bash
PRIMARY_PROFILE=legacy-27 ./run_experiments.sh
```

Arguments are forwarded to the Python driver, so a dry run is:

```bash
./run_experiments.sh --dry-run
```

Evaluation split selection is explicit and defaults to `test`:

```bash
./run_experiments.sh                         # test
./run_experiments.sh --evaluation-split train
EVALUATION_SPLIT=all ./run_experiments.sh
```

The executable catalogue contains 127 configurations:

| Suite | Count | Contents |
|---|---:|---|
| `phenomenology` | 121 | small-model task × intervention × phase configurations plus the explicit Qwen2-1.5B I+O NLI setting |
| `large-models` | 6 | Qwen2-7B and Pythia-6.9B on arithmetic, NLI, and jailbreak |
| **Total** | **127** | complete non-poisoning catalogue |

List them without running anything:

```bash
python3 -m experiments.run_experiments --suite all --list
```

The Python driver can also run a subset:

```bash
python3 -m experiments.run_experiments --suite phenomenology
python3 -m experiments.run_experiments --suite large-models
```

Filters are exact comma-separated values:

```bash
python3 -m experiments.run_experiments \
  --suite phenomenology \
  --task arithmetic,hans_nli \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --intervention mean-donor,zero \
  --mode standard,decode-only \
  --evaluation-split test
```

Execution phases are:

```text
--phase pipeline   run experiment stages only
--phase analysis   analyze configured outputs only
--phase all        run the pipeline, then analysis (default)
```

`--continue-on-error` records failed configurations in `results/pipeline_failures.json` and continues. `--dry-run` prints pipeline commands without model execution.

See `experiments/README.md` for the exact suite definitions and path policy.

## 6. Evaluation-split discipline

The evaluation split is configurable. `test` is the default for the experiment catalogue, stage 7, threshold diagnostics, group dominance, and interaction validation. The accepted values are:

```text
test   rows with is_test=True
train  rows with is_test=False
all    all available/materialized rows
```

The split can be selected through the main driver:

```bash
python3 -m experiments.run_experiments --evaluation-split test
python3 -m experiments.run_experiments --evaluation-split train
python3 -m experiments.run_experiments --evaluation-split all
```

or directly through `pipeline/_run_pipeline.sh` with `--evaluation_split`. The legacy `--holdout_test_only` flag remains an alias for `--evaluation_split test`.

Candidate discovery and candidate ranking remain based on discovery/training data. Mean/donor replacement-reference estimation also remains training-based. Only the final evaluation population changes with `--evaluation_split`. This makes `test` the post-selection validation path while preserving `train` and `all` for diagnostic use.

Stage 7 writes `evaluation_scope.json` with `final_statistics_split` and `sampling_pool_split`. Directory naming is split-aware:

```text
test   <run>-heldout_test/
train  <run>-eval_train/
all    <run>/
```

The unsuffixed `all` path preserves the established all-row filesystem convention. Primary manuscript profiles and the paper tables/figures are defined on the `test` split; requesting `train` or `all` through the root experiment launcher therefore runs catalogue analysis but not the primary manuscript export.

## 7. Numbered pipeline stages

`pipeline/_run_pipeline.sh` is the standard single-configuration orchestrator.

### Stage 1 — prompts and model answers

`1_generate_prompts_and_answers.py` obtains task examples, queries the analyzed model, parses outputs, records task performance, and caches prompt/answer data.

### Stage 2 — feature construction

`2_generate_features.py` builds task-provided and optionally LLM-proposed interpretable features. It writes `feature_report/scores.csv` and `features.json`; the feature-extraction runner also creates the deterministic `is_test` column in `scores.csv`. Task-level `dataset_stats.json` is written by stage 1 into the same `feature_report/` directory.

### Stage 3 — symbolic rules

`3_extract_rules.py` extracts feature-based rules for rule-conditioned workflows. The standard catalogue uses spectral splits, so this stage is bypassed there.

### Stage 4 — sampling plans

`4_spectral_sample_datapoints.py` constructs representative sampling plans where required. Spectral-split catalogue runs bypass the rule-indexed sampling-plan stage.

### Stage 5 — circuit discovery

`5_discover_circuits.py` runs EAP/EAP-IG-based circuit discovery and writes the `neural_circuits/` manifest and dataset information. The manifest defines the prespecified layer/channel populations used in interaction-aware validation.

### Stage 6 — candidate selection

`6_analyze_bag_of_rules.py` performs channel interventions on discovery data and selects channels meeting the configured effect threshold. The selected channels form the fixed set `J`. Ranking information from this stage is used to freeze the order for `H_m` and `J_{l,m}`.

### Stage 7 — singleton evaluation

`7_refine_neuron_anchored_rules.py` suppresses each fixed candidate individually on rows selected by `--evaluation_split`, materializes per-example flip events, and computes singleton-set statistics. `test` is the default. It writes both aggregate files and the discovery-frozen ranking used by `TOC_m`.

### Interaction validation

After stage 7, `_run_pipeline.sh` calls `analysis/26_validate_interactions.py` when the required artifacts exist. This module performs simultaneous full-set, layer-population, complement, and matched-random interventions.

See `pipeline/README.md` for direct commands, options, environment variables, and detailed output files.

## 8. Stage-7 statistics and interaction outputs

A complete stage-7 stats directory contains files such as:

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
interaction_validation/
```

`singleton_set_metrics.json` uses one common complete-case probability space within the selected evaluation split for all `s_j`, `U(H_m)`, and `U(J)` values.

The interaction directory contains:

```text
interaction_configuration.json
layer_populations.csv
matched_control_strata.csv
frozen_candidate_ranking.csv
layer_interaction_effects.csv
gccr_metrics.csv
matched_random_set_membership.csv
matched_null_draws.csv
interaction_validation_summary.csv
interaction_validation_summary.json
interaction_validation_summary.md
interaction_validation_table.tex
E_J_matched_null.pdf
E_J_matched_null.png
```

The configuration file fingerprints the inputs and records the intervention phase, replacement baseline, random seed, `m` values, null-draw count, and denominator tolerance.

## 9. Replacement baselines and phases

The pipeline supports:

```text
zero
mean
mean-positional
mean-donor
mean-donor-positional
```

`standard` mode corresponds to phase `I+O`; `decode-only` corresponds to phase `Out`.

For filesystem labels, `mean` and `mean-positional` share an unsuffixed run-name family. The exact baseline is therefore carried in experiment configuration and validation metadata and should not be inferred only from the directory name.

## 10. Primary manuscript profiles

Publication output requires an explicit primary profile:

| Profile identifier | Settings | Qwen2-1.5B input+output NLI |
|---|---:|---|
| `iclr-28` | 28 | included exactly once |
| `legacy-27` | 27 | excluded |

`analysis/primary_matrix.py` validates both row count and row identity. It never chooses a profile automatically inside the table-building code.

The root launchers select `iclr-28` by default and print the choice. Set `PRIMARY_PROFILE=legacy-27` to request the 27-setting matrix.

Profile normalization writes:

```text
primary_table_normalized.csv
primary_table_excluded.csv
primary_table_profile.json
```

When a row is excluded by the 27-setting profile, it appears explicitly in the excluded-row file.

## 11. Final statistics and figures

All final paper products belong under the repository-root `results/` directory. Raw experiment data remains under `data/`.

To regenerate final outputs from existing data without rerunning the experiment catalogue:

```bash
./generate_results.sh
```

The input data root can be changed with `DATA_ROOT`:

```bash
DATA_ROOT=/path/to/data PRIMARY_PROFILE=iclr-28 ./generate_results.sh
```

The results tree is:

```text
results/
├── configured_experiments.json      # written by experiment driver
├── pipeline_failures.json           # written when pipeline phase runs
├── catalogue/                       # configured-run summaries and plots
├── required_metrics_audit/          # exact/backfillable/unavailable metric audit
├── paper_tables/                    # canonical primary table and LaTeX tables
├── primary_metrics/                 # correlations and directional summaries
├── manuscript/                      # manuscript metric tables/statuses/plots
├── paper_figures/                   # publication figures
├── overtopping_spiking_report/      # spiking diagnostics or availability status
└── final_results_manifest.json
```

`analysis/29_generate_final_results.py` orchestrates the final-output pass. It calls dedicated numerical/reporting scripts rather than reimplementing their statistics.

If no `spiking_diagnostics` directories are present under the data root, the spiking-report directory contains `report_status.json` with `status=not_available` instead of failing the entire results pass.

See `analysis/README.md` and `results/README.md` for detailed output schemas.

## 12. Cache reuse and resuming

The pipeline uses completion files and fingerprints to avoid unnecessary model work.

- Stage 2 skips feature generation when its `scores.csv` already exists.
- Other stages use their own manifests/output checks.
- Stage 7 is considered complete when singleton artifacts for the requested evaluation split and the current singleton-metric schema are present.
- If split-compatible `scores.csv`, `flip_stats_global.json`, and `flip_stats_by_neuron.csv` exist but metric sidecars are missing, the wrapper runs stage 7 in `--stats_only` mode. This reconstructs exact singleton-event metrics from materialized flip events without replaying singleton model ablations; the discovery-frozen ranking is rebuilt from stage-6 discovery outputs.
- If `singleton_set_metrics.json` is absent but `flip_stats_global.json` and `flip_stats_by_neuron.csv` remain, `|J|`, `U(J)`, `s_(1)`, `N_t`, `R_ov`, and `N_eff` are still exactly recoverable from the legacy aggregates. `TOC_m` and `OCC_b` are not: `TOC_m` needs the discovery-frozen order plus per-example flip events, and `OCC_b` needs per-example baseline and union events.
- `E(J)`, GCCR, and their matched-null statistics are never reconstructed from singleton aggregates. When their validated interaction sidecars are missing, the standard experiment pipeline runs genuine simultaneous interventions if the stage-5 population, materialized evaluation rows, model, and replacement-baseline inputs are available.
- Interaction validation reuses a compatible `interaction_configuration.json`/summary before model loading.

Force controls include:

```bash
export FORCE_STAGE7=true
export FORCE_INTERACTION_VALIDATION=true
```

Do not reuse feature/split caches after changing the source dataset, sampling policy, task seed, or another input that changes row identity. Use a separate output/cache root for incompatible experiment definitions.

## 13. Compact result exports

`analysis/clean_results_for_export.py` creates a filtered, shareable result tree:

```bash
python3 -m analysis.clean_results_for_export data data_filtered_results
```

Useful options are:

```text
--dry-run
--force
--verbose
--manifest <path>
--reference-zip <path>
```

The filtered schema keeps aggregate statistics, frozen rankings, singleton-set sidecars, interaction summaries, tables, and plots. It omits large per-example `scores.csv` files, pickle caches, circuit-input caches, and other runtime-heavy intermediates.

A compact export can regenerate model-free tables and figures represented by the retained summaries. For an old compact export that predates the new sidecars, the aggregate files can recover `|J|`, `U(J)`, `s_(1)`, `N_t`, `R_ov`, and `N_eff`, but they cannot determine discovery-frozen `TOC_m`, exact `OCC_b`, simultaneous `E(J)`, GCCR, or matched-null statistics. Those quantities are reported as unavailable rather than replaced by `Top/U`, `C2I/raw`, or singleton-union proxies.

`./generate_results.sh` writes `results/required_metrics_audit/required_metrics_audit.{csv,json}` and is strict by default: it exits nonzero when any primary setting lacks an exact requested new metric. Set `ALLOW_INCOMPLETE_NEW_METRICS=1` only when deliberately producing a partial legacy report. A full runtime cache should instead be processed through `./run_experiments.sh`, which backfills singleton sidecars and runs missing simultaneous-intervention validation as needed.

## 14. Poisoning experiments

Trigger-poisoning code is isolated under `poisoning/`. It supports grammar and arithmetic checkpoint fine-tuning, trigger-conditioned overtopping, trigger-lift experiments, trajectory aggregation, and cumulative top-k mechanism diagnostics.

Start with `poisoning/README.md` before submitting the Slurm jobs, because those workflows have their own run directories, checkpoint manifests, and scheduler environment variables.

## 15. Validation

Run the model-free test suite:

```bash
python -m unittest discover -s tests -v
```

Check shell syntax:

```bash
bash -n run_experiments.sh
bash -n generate_results.sh
bash -n pipeline/_run_pipeline.sh
find poisoning/jobs -type f -name '*.sbatch' -print0 | xargs -0 -n1 bash -n
```

Inspect the experiment catalogue without launching models:

```bash
python3 -m experiments.run_experiments --suite all --list
```

## 16. Troubleshooting

**A required dataset is missing.** Set the task-specific path variable, such as `GRAMMAR_DATASET_PATH`, `HANS_LOCAL_FILE`, or `AUGMENTED_PROMPTS_FILE`.

**Feature proposal cannot contact Ollama.** Use `--no_llm_feature_generation` for direct pipeline runs or set `no_llm_feature_generation=True` in the relevant `RunSpec`.

**A GPU run is out of memory.** Reduce batch size, circuit size, stage-7 neuron batch size, or the number of selected circuits/null draws as appropriate.

**The final-results pass cannot build the requested primary matrix.** Confirm that the data root contains all 28 settings for `iclr-28`, or exactly the accepted 27/28 input required by `legacy-27`. The primary-matrix validator reports the missing/extra identity rather than silently changing the matrix.

**Interaction validation is missing.** A new `E(J)` or GCCR computation requires full stage-5/stage-7 runtime artifacts and model access. A singleton-only export is insufficient.

**A ratio is `NaN`.** Inspect its status field. Near-zero denominators are intentionally reported as undefined, and signed/above-one finite values are not clipped.

**The spiking report says `not_available`.** No `spiking_diagnostics` directory was found under the selected data root. Other final results can still be generated.

**A cached run has the wrong split or dataset.** Do not overwrite it in place. Use a separate data/cache root so row identity and discovery/test provenance remain unambiguous.
