# Overtopping Phenomenology - Replication Package

This repository contains the code, analysis utilities, and selected generated artifacts for the paper **"The Phenomenology of Overtopping: How Few Channels Can Dominate Language-Model Behavior."** The paper studies **overtopping**: cases where one channel, or a small coalition of channels, accounts for a large fraction of the causal effect of an intervention on a binary model behavior. The package implements the pipeline used to generate prompts, derive feature tables, extract symbolic rules, localize candidate channels with intervention-based searches, refine those candidates to singleton overtopping channels, aggregate results, and render the paper figures.

The package is intentionally organized around the experiment pipeline rather than around a single executable. Full reruns are compute-intensive: they require local Hugging Face model weights, TransformerLens-compatible models, and enough accelerator memory for repeated intervention scans. The included `figures/` and `overtopping_spiking_report/` directories contain selected exported artifacts from completed runs.

## What this package reproduces

The paper evaluates whether learned behaviors are causally controlled by a small number of intervention-defined channels. The code supports the main experimental families used in the paper:

- Arithmetic exact-match behavior.
- Grammar acceptability behavior.
- HANS-style NLI entailment-label behavior.
- Random finite-state-machine final-state prediction.
- Jailbreak-success behavior.
- Threshold-event / causal-spiking diagnostics for overtopping candidates versus matched non-candidate controls.

The main intervention loop follows the same conceptual stages as the paper: define a binary behavior predicate, construct behavior-conditioned or rule-conditioned slices, discover candidate components, test channel or channel-group replacement, refine to singleton effects, and aggregate union flip coverage and singleton flip strength.

## Important directory convention: `data/` versus `results/`

The working experiment tree is `data/`. Full runs write large intermediate files under `data/<task>/<provider>/<model>/...` and under `cache/`.

A clean export directory named `results/` is **not** produced directly by the experiment scripts. It is produced by:

```bash
python clean_results_for_export.py data results --force --manifest results_manifest.json
```

`clean_results_for_export.py` copies only approved final-output templates into a fresh filtered directory. It excludes caches, pickle files, macOS metadata, `is_correct_*` intermediate folders, and other non-export artifacts. If you receive a `results/` directory with this package, treat it as a curated export generated from a larger `data/` tree, not as the raw working directory.

## Repository contents

```text
.
├── 1_generate_prompts_and_answers.py
├── 2_generate_features.py
├── 3_extract_rules.py
├── 4_spectral_sample_datapoints.py
├── 5_discover_circuits.py
├── 6_analyze_bag_of_rules.py
├── 7_refine_neuron_anchored_rules.py
├── 8_compare_experiments.py
├── 9_compare_models.py
├── 10_compute_threshold_sweep_stats.py
├── 12_threshold_event_diagnostics.py
├── _run_pipeline.sh
├── run_phenomenology_experiments.sh
├── run_spiking_diagnostics.sh
├── clean_results_for_export.py
├── make_competence_vs_overtopping_paper_figures.py
├── generate_overtopping_spiking_report.py
├── setup.sh
├── requirements.txt
├── figures/
├── overtopping_spiking_report/
└── lib/
```

There is no script `11_*` in this package. The numbering jumps from `10_compute_threshold_sweep_stats.py` to `12_threshold_event_diagnostics.py`.

All auxiliary Markdown notes that were present in earlier package versions have been merged into this README: the spiking-report usage notes, circuit-selection-level notes, EAP/EAP-IG library notes, and the completed spiking-diagnostic interpretation.

## Installation

The scripts target Python 3.12.

```bash
bash setup.sh
. .env/bin/activate
```

The package uses PyTorch, Transformers, TransformerLens, pandas, matplotlib, SciPy, scikit-learn, SHAP, XGBoost, numba, tabulate, and tqdm. Some environments also need an OpenMP runtime for XGBoost and numerical libraries:

```bash
# macOS
brew install libomp

# Ubuntu/Debian example
sudo apt-get install libgomp1
```

If using Ollama for feature proposal, start the server and pull the model you plan to use:

```bash
ollama serve > ollama.log 2>&1 &
ollama pull qwen3:30b
```

Optional external judge or feature-proposal backends read credentials from the environment. Do not hardcode API keys in scripts.

```bash
export GROQ_API_KEY="..."     # only if your chosen task/judge uses Groq
export OPENAI_API_KEY="..."   # only if your chosen configuration uses OpenAI
```

## Quick checks on the shipped artifacts

The package includes selected generated figures and tabular data:

```text
figures/
  fig_competence_vs_coverage.pdf
  fig_competence_vs_coverage.csv
  fig_phase_comparison.pdf
  fig_phase_comparison.csv
  fig_pythia_checkpoint_trajectory.pdf
  fig_pythia_checkpoint_trajectory.csv
  fig_size_comparison.pdf
  fig_size_comparison.csv

overtopping_spiking_report/
  figures/ecdf_causal_spiking_score.pdf
  figures/ecdf_flip_rates_candidate_vs_control.pdf
  figures/feature_css_delta_ranking.pdf
  figures/paired_css_delta_by_run_baseline.pdf
  figures/binned_flip_curves_oriented_proxy.pdf
  statistical_results.json
  primary_spiking_score_summary.csv
  flip_rate_summary.csv
```

The CSV next to each PDF is the plotted-data companion and is usually the easiest way to audit the plotted values.

## Full pipeline overview

The numbered scripts form a staged workflow. A typical full run is:

```text
1_generate_prompts_and_answers.py
  -> cache/<task>/<model>/llm_io_data.pkl

2_generate_features.py
  -> data/<task>/<provider>/<model>/feature_report/scores.csv
  -> data/<task>/<provider>/<model>/feature_report/features.json

3_extract_rules.py
  -> data/<task>/<provider>/<model>/rule_extraction_results/association_rules_*.csv
  -> data/<task>/<provider>/<model>/rule_extraction_results/rule_combo_*.csv

4_spectral_sample_datapoints.py
  -> data/<task>/<provider>/<model>/neural_circuit_discovery_results/.../spectral_sampling_plan.json

5_discover_circuits.py
  -> data/<task>/<provider>/<model>/neural_circuit_discovery_results/.../neural_circuits/manifest.json
  -> data/<task>/<provider>/<model>/neural_circuit_discovery_results/.../neural_circuits/dataset_info.json

6_analyze_bag_of_rules.py
  -> .../bag_of_rules/.../per_rule/<target>/rule_*.json
  -> .../bag_of_rules/.../rule_knockout.json
  -> .../bag_of_rules/.../neuron_bucket_stats.json

7_refine_neuron_anchored_rules.py
  -> data/<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<run>/flip_stats_global.json
  -> data/<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<run>/flip_stats_by_neuron.csv
  -> optional per-neuron RuleSHAP outputs and figure files
```

Scripts `8`, `9`, `10`, `12`, `make_competence_vs_overtopping_paper_figures.py`, and `generate_overtopping_spiking_report.py` are post-processing, figure-generation, and diagnostic utilities.

## Running the orchestrated phenomenology sweep

The top-level runner `run_phenomenology_experiments.sh` loops over configured tasks, models, intervention baselines, and intervention phases. It calls `_run_pipeline.sh` for the pipeline stages and then runs post-processing scripts.

```bash
bash run_phenomenology_experiments.sh
```

Before launching a large sweep, edit the model and task arrays in that script. The default lists include small Qwen and Pythia configurations and are intended as a template. Full sweeps can be expensive.

The lower-level `_run_pipeline.sh` runs one `(task, analyzed model)` configuration. Example:

```bash
bash _run_pipeline.sh arithmetic Qwen/Qwen2-1.5B-Instruct \
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

## Numbered scripts

### `1_generate_prompts_and_answers.py` - generate task prompts and model completions

This script creates or loads the task-level prompt/answer cache. The task module owns prompt generation, target calculation, parsing, and any task-specific metadata.

Common arguments:

```bash
python 1_generate_prompts_and_answers.py \
  --task_module lib.tasks.arithmetic_task \
  --ai_model Qwen/Qwen2-1.5B-Instruct \
  --prompts_answers_pkl_file cache/arithmetic/Qwen/Qwen2-1.5B-Instruct/llm_io_data.pkl \
  --batch_size 32 \
  --stats_json_out data/arithmetic/Qwen/Qwen2-1.5B-Instruct/feature_report/dataset_stats.json
```

Output: a pickle cache and, optionally, `dataset_stats.json`. Later scripts rely on the cache through the task module's loader.

### `2_generate_features.py` - build `scores.csv` and `features.json`

This stage loads the prompt/answer cache and constructs a feature table. It can use only deterministic seed features or call a feature LLM to propose Python feature functions.

```bash
python 2_generate_features.py \
  --task_module lib.tasks.arithmetic_task \
  --prompts_answers_pkl_file cache/arithmetic/Qwen/Qwen2-1.5B-Instruct/llm_io_data.pkl \
  --features_scores_dir data/arithmetic/Qwen/Qwen2-1.5B-Instruct/feature_report \
  --cache_dir cache/arithmetic \
  --ai_model qwen3:30b \
  --feature_extraction_steps 10 \
  --drop_near_duplicate_features \
  --drop_high_mad_variance_features \
  --z_thresh 10
```

Outputs: `scores.csv`, `features.json`, and feature-selection metadata. `scores.csv` contains prompt columns, raw model outputs, target labels such as `is_correct` or `is_jailbroken`, and feature columns.

### `3_extract_rules.py` - extract symbolic rules from features

This stage fits RuleSHAP-style feature models and writes symbolic rules that predict target labels.

```bash
python 3_extract_rules.py \
  --task_module lib.tasks.arithmetic_task \
  --features_scores_dir data/arithmetic/Qwen/Qwen2-1.5B-Instruct/feature_report \
  --rules_dir data/arithmetic/Qwen/Qwen2-1.5B-Instruct/rule_extraction_results \
  --use_shap_in_xgb \
  --use_shap_in_lasso
```

Outputs include `association_rules_*.csv`, `rule_combo_*.csv`, and SHAP plots when enabled by the underlying SHAP utilities. The `--fake_targets` flag creates random-control targets and writes rule files for those controls.

### `4_spectral_sample_datapoints.py` - construct representative sampling plans

This stage chooses datapoints for expensive circuit discovery and ablation. It supports spectral k-center sampling in the analyzed model's representation space and an optional similarity/length-matched pairing mode.

```bash
python 4_spectral_sample_datapoints.py \
  --task_module lib.tasks.arithmetic_task \
  --features_scores_dir data/arithmetic/Qwen/Qwen2-1.5B-Instruct/feature_report/scores.csv \
  --rules_dir data/arithmetic/Qwen/Qwen2-1.5B-Instruct/rule_extraction_results \
  --ai_model Qwen/Qwen2-1.5B-Instruct \
  --spectral_cache_dir cache/arithmetic \
  --spectral_space hidden \
  --rep_hook_name ln_final.hook_normalized \
  --rep_pooling last \
  --spectral_dim 32 \
  --coverage_radius 0.5 \
  --max_points_per_ablation 512 \
  --min_points_per_ablation 32 \
  --output_path data/arithmetic/Qwen/Qwen2-1.5B-Instruct/neural_circuit_discovery_results/spectral_sampling_plan.json
```

Output: a JSON sampling plan consumed by scripts `5` and `6` when `--sampling_strategy plan` is used.

### `5_discover_circuits.py` - attribution-based circuit discovery

This stage discovers candidate circuit elements with EAP, EAP-IG, clean-corrupted attribution, or related attribution modes. It can operate over rule-conditioned examples or spectral clusters.

```bash
python 5_discover_circuits.py \
  --task_module lib.tasks.arithmetic_task \
  --rules_dir data/arithmetic/Qwen/Qwen2-1.5B-Instruct/rule_extraction_results \
  --features_scores_dir data/arithmetic/Qwen/Qwen2-1.5B-Instruct/feature_report/scores.csv \
  --output_data_dir data/arithmetic/Qwen/Qwen2-1.5B-Instruct/neural_circuit_discovery_results/eap_ig_inputs/spectral_split-M200000-decode_only-eval_mean-donor/neural_circuits \
  --ai_model Qwen/Qwen2-1.5B-Instruct \
  --cache_dir cache/arithmetic \
  --method EAP-IG-inputs \
  --max_ig_steps 3 \
  --circuit_level neuron \
  --circuit_size 200000 \
  --eval_intervention mean-donor \
  --absolute_value_attributions \
  --sampling_strategy plan \
  --sampling_plan_path data/arithmetic/Qwen/Qwen2-1.5B-Instruct/neural_circuit_discovery_results/spectral_sampling_plan.json \
  --decode_only
```

Key choices:

- `--circuit_level edge`: path-specific but noisier and more complex.
- `--circuit_level node`: stable component-level heads/MLPs; often the best default for circuit narratives.
- `--circuit_level neuron`: fine-grained MLP/channel-level selection; used for overtopping localization.
- `--mlp_neurons_only`: restricts neuron-level outputs to MLP coordinates.

Outputs: `manifest.json`, `dataset_info.json`, and per-rule or per-cluster circuit metadata.

### `6_analyze_bag_of_rules.py` - group ablation and CHA-style search

This script evaluates discovered units and unit groups on associated versus unrelated examples. In fast mode it performs a layer-wise dichotomic search, pruning low-effect subtrees and keeping promising singletons or groups.

```bash
python 6_analyze_bag_of_rules.py \
  --task_module lib.tasks.arithmetic_task \
  --input_data_dir data/arithmetic/Qwen/Qwen2-1.5B-Instruct/neural_circuit_discovery_results/eap_ig_inputs/spectral_split-M200000-decode_only-eval_mean-donor/neural_circuits \
  --output_data_dir data/arithmetic/Qwen/Qwen2-1.5B-Instruct/neural_circuit_discovery_results/eap_ig_inputs/spectral_split-M200000-decode_only-eval_mean-donor/bag_of_rules/agonist_neurons-fast-random_anchor-tau0.3 \
  --scores_path data/arithmetic/Qwen/Qwen2-1.5B-Instruct/feature_report/scores.csv \
  --ai_model Qwen/Qwen2-1.5B-Instruct \
  --intervention mean-donor \
  --points_to_use_for_mean_ablation 2048 \
  --fast_ablation \
  --search_epsilon 0.3 \
  --baseline_subset positive \
  --sampling_strategy plan \
  --sampling_plan_path data/arithmetic/Qwen/Qwen2-1.5B-Instruct/neural_circuit_discovery_results/spectral_sampling_plan.json \
  --decode_only
```

Outputs: per-rule JSON traces, `rule_knockout.json`, `neuron_bucket_stats.json`, optional activation/margin/saliency diagnostics, and PNG summaries unless those diagnostics are skipped.

### `7_refine_neuron_anchored_rules.py` - singleton refinement, flip columns, and rule summaries

This stage reads the script-6 per-rule outputs, filters singleton candidates by effect threshold, evaluates singleton replacement on prompts, writes flip columns, and optionally extracts feature rules for those flip targets.

```bash
python 7_refine_neuron_anchored_rules.py \
  --task_module lib.tasks.arithmetic_task \
  --circuit_agonists_path data/arithmetic/Qwen/Qwen2-1.5B-Instruct/neural_circuit_discovery_results/eap_ig_inputs/spectral_split-M200000-decode_only-eval_mean-donor/bag_of_rules/agonist_neurons-fast-random_anchor-tau0.3/per_rule \
  --features_scores_dir data/arithmetic/Qwen/Qwen2-1.5B-Instruct/feature_report \
  --ai_model Qwen/Qwen2-1.5B-Instruct \
  --intervention mean-donor \
  --points_to_use_for_mean_ablation 2048 \
  --search_epsilon 0.3 \
  --use_spectral_sampling \
  --global_n_clusters 32 \
  --sampling_max_points 512 \
  --decode_only \
  --extract_rules \
  --summarize_rule_metrics \
  --rule_quality_metric mcc \
  --rule_quality_threshold 0.85 \
  --stats_dirname spectral_split-M200000-decode_only-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3
```

Key outputs under `rule_extraction_results/neuron_flip_rules/stats/<run>/`:

- `flip_stats_by_neuron.csv`: singleton flip counts and rates.
- `flip_stats_global.json`: union flip coverage and aggregate counts.
- `flip_stats_top<num>.pdf`: top singleton flip-rate bar plot.
- `rule_combo_metrics_*.csv`: best per-neuron flip-predicting rule summaries.
- `rule_metrics_distributions.pdf`: distributions of rule metrics.
- `high_quality_neuron_flip_coverage.pdf`: coverage by high-quality rule subset.
- `high_quality_neuron_flip_coverage_by_layer.pdf`: layer-stratified coverage.
- `agonist_metric_*.csv`, `.json`, `.png`, `.pdf`: activation, gradient, Wanda, and activation-gradient diagnostics.

The `--stats_only` and `--summarize_rule_metrics_only` modes are useful when flip columns or rule files already exist and only final tables/figures need to be regenerated.

### `8_compare_experiments.py` - aggregate run modes and threshold sweeps

This script scans one stats directory, a parent tree of stats directories, or a zip. It writes threshold-sweep summaries and run-comparison plots.

```bash
python 8_compare_experiments.py \
  --stats_path data \
  --out_dir data/aggregated_visualizations \
  --rule_quality_metric mcc \
  --best_mode tail@0.9 \
  --thr_min 0.85 \
  --thr_max 0.99 \
  --thr_step 0.01
```

Outputs include:

- `threshold_sweep_summary.csv` for a single stats root.
- `aggregated_by_task/threshold_sweep_all_tasks_all_llms.csv` for multi-run trees.
- `plot_high_quality_neurons.pdf`.
- `plot_union_flip_c2i_pct.pdf`.
- `plot_union_flip_i2c_pct.pdf`.
- `compare_rule_metrics_distributions_ecdf.pdf`.
- Per-task variants such as `plot_high_quality_neurons_by_task.pdf`.

### `9_compare_models.py` - compare models within a task

This script aggregates model-level counts, selectivity, and rule metrics for one task directory.

```bash
python 9_compare_models.py \
  --task_dir data/arithmetic \
  --out_dir data/aggregated_visualizations/arithmetic \
  --tau 0.2 \
  --eps 0.2
```

Outputs include:

- `model_run_coverage.csv`.
- `run_summary_by_model.csv`.
- `agg_summary_by_run.csv`.
- Plots in `plots/`, including cross-model trends for singleton tau-agonists, rule MCC, median ablation groups, and ECDF overlays for MCC, flip-any rate, and selectivity.

### `10_compute_threshold_sweep_stats.py` - compact threshold-sweep statistics

This script consumes the multi-task threshold-sweep CSV from script `8` and prints compact E1/E3-style tables and Wilcoxon tests.

```bash
python 10_compute_threshold_sweep_stats.py \
  --csv data/aggregated_visualizations/aggregated_by_task/threshold_sweep_all_tasks_all_llms.csv \
  --metric n_high_quality_neurons \
  --agg available \
  --alt greater
```

Use `--thresholds` to select specific metric thresholds.

### `12_threshold_event_diagnostics.py` - threshold recoverability and CSS inputs

This script evaluates whether singleton flip-positive examples for candidate channels are predictable by one-dimensional thresholds over activation, gradient, activation-gradient, Wanda, predicted margin-drop, or learned-direction proxy features. These outputs feed `generate_overtopping_spiking_report.py`.

```bash
python 12_threshold_event_diagnostics.py \
  --input_data_dir data/arithmetic/Qwen/Qwen2-1.5B-Instruct/neural_circuit_discovery_results/eap_ig_inputs/spectral_split-M200000-eval_mean-donor/neural_circuits \
  --out_dir data/arithmetic/Qwen/Qwen2-1.5B-Instruct/neural_circuit_discovery_results/eap_ig_inputs/spectral_split-M200000-eval_mean-donor/spiking_diagnostics \
  --task_module lib.tasks.arithmetic_task \
  --ai_model Qwen/Qwen2-1.5B-Instruct \
  --intervention mean-donor \
  --points_to_use_for_mean_ablation 2048 \
  --spiking_max_points 2048 \
  --spiking_min_points 64 \
  --threshold_event_repeats 10 \
  --threshold_event_holdout_fraction 0.5 \
  --rule_conditioned_diagnostics \
  --rule_conditioned_only \
  --rules_dir data/arithmetic/Qwen/Qwen2-1.5B-Instruct/rule_extraction_results/neuron_flip_rules \
  --rules_stats_dirname spectral_split-M200000-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3 \
  --proxy_metrics activation,abs_activation,gradient,abs_gradient,activation_x_gradient,abs_activation_x_gradient,predicted_margin_drop,abs_predicted_margin_drop,learned_direction
```

Outputs include per-unit threshold tests, population summaries, binned flip curves, aggregate CSVs, JSON summaries, and optional figure files under `spiking_diagnostics/figures/`.

## Figure-generation scripts and outputs

This section documents every script in the package that can directly write figures or figure-adjacent plotted data.

### `make_competence_vs_overtopping_paper_figures.py`

Purpose: render the main overtopping phenomenology figures from a `results/` or `data/` tree containing `dataset_stats.json` and `flip_stats_global.json` files.

Primary outputs:

- `fig_competence_vs_coverage.pdf` and `.csv`: scatter of task competence versus overtopping union flip coverage `U(J)`. The default `phase-specific` score mode uses chance-normalized finite-answer scores for output-only points and raw scores for input+output points.
- `fig_phase_comparison.pdf` and `.csv`: matched input+output versus output-only coverage panel.
- `fig_pythia_checkpoint_trajectory.pdf` and `.csv`: Pythia checkpoint trajectory for grammar, NLI, and random FSM.
- `fig_size_comparison.pdf` and `.csv`: same-family scale comparison panel.

Example using a clean export:

```bash
python make_competence_vs_overtopping_paper_figures.py \
  --results-dir results \
  --out figures/fig_competence_vs_coverage.pdf \
  --layout phase-panels \
  --score-mode phase-specific \
  --paper-figures all \
  --paper-figures-dir figures \
  --paper-baseline mean-donor
```

Use `--only-paper-figures` to skip the scatter and regenerate only phase/checkpoint/size panels. Use `--png` to write PNG copies next to the PDFs.

### `generate_overtopping_spiking_report.py`

Purpose: aggregate high-N threshold-event diagnostic results and render the threshold-recoverability / causal-spiking figures.

Inputs: either a `spiking_diagnostics_results_for_inspection.zip` produced by `run_spiking_diagnostics.sh`, or an already extracted root containing the same result paths.

```bash
python generate_overtopping_spiking_report.py \
  --zip spiking_diagnostics_results_for_inspection.zip \
  --out overtopping_spiking_report
```

Outputs:

- `figures/ecdf_causal_spiking_score.pdf`: ECDF of CSS for candidates and non-candidate controls.
- `figures/ecdf_flip_rates_candidate_vs_control.pdf`: ECDF of singleton flip-any rates.
- `figures/paired_css_delta_by_run_baseline.pdf`: paired candidate-control CSS deltas by condition.
- `figures/feature_css_delta_ranking.pdf`: proxy features ranked by median candidate-control CSS delta.
- `figures/binned_flip_curves_oriented_proxy.pdf`: oriented proxy bins and flip-rate enrichment.
- `statistical_results.json` plus CSV summaries used by the report.
- `updated_spiking_diagnostics_experiments.md` is regenerated by this script if it is rerun; this package keeps the interpretation in this README rather than as a separate Markdown file.

### `run_spiking_diagnostics.sh`

Purpose: orchestrate calls to `12_threshold_event_diagnostics.py` across tasks, models, interventions, and run modes. It can also collect diagnostic outputs into a zip.

Useful modes:

```bash
# Run diagnostics and collect a compact result bundle.
bash run_spiking_diagnostics.sh

# Collect only, without recomputing diagnostics.
COLLECT_ONLY=1 RESULTS_ZIP=spiking_diagnostics_results_for_inspection.zip bash run_spiking_diagnostics.sh

# Restrict to one task/model/run mode.
ONLY_TASK=arithmetic ONLY_MODEL='Qwen/Qwen2-1.5B-Instruct' ONLY_RUN_MODE=standard bash run_spiking_diagnostics.sh
```

The collection step selects aggregate CSV/JSON/log files under `spiking_diagnostics/` and writes `spiking_diagnostics_results_for_inspection.zip` by default.

### `7_refine_neuron_anchored_rules.py`

Purpose: in addition to singleton evaluation, this script writes multiple paper-ready diagnostic figures:

- `flip_stats_top<num>.pdf`: top singleton channels by flip rate.
- `rule_metrics_distributions.pdf`: distributions of per-neuron anchored-rule quality metrics.
- `high_quality_neuron_flip_coverage.pdf`: union flip coverage as a function of high-quality anchored rules.
- `high_quality_neuron_flip_coverage_by_layer.pdf`: layer-stratified high-quality neuron coverage.
- `agonist_metric_percentile_boxplot.png`: within-layer activation/gradient/Wanda percentile distributions.
- `agonist_metric_top_rates.png`: rates of overtopping channels appearing in top metric ranks.
- `agonist_metric_correlation_heatmap.png`: rank/correlation summary between proxy metrics and ablation strength.
- `agonist_metric_ablation_scatter.png`: proxy-versus-ablation scatter diagnostics.
- `agonist_metric_final_plots.pdf`: multi-page combined PDF of the metric diagnostics.

Run with `--summarize_rule_metrics` or `--summarize_rule_metrics_only` when only the rule/coverage plots need to be refreshed.

### `8_compare_experiments.py`

Purpose: render threshold-sweep and rule-metric comparison figures from one or many stats folders.

Figures:

- `plot_high_quality_neurons.pdf`.
- `plot_union_flip_c2i_pct.pdf`.
- `plot_union_flip_i2c_pct.pdf`.
- `compare_rule_metrics_distributions_ecdf.pdf`.
- Per-task versions under `aggregated_by_task/<task>/` when scanning multiple tasks/models.

### `9_compare_models.py`

Purpose: render model-comparison figures within a task.

Figures are written under `<out_dir>/plots/` and include cross-model trends and aggregate metrics for tau-agonist counts, median anchored-rule MCC, median ablation groups, median selectivity, fraction epsilon-selective, and ECDF overlays for rule MCC, flip-any rate, and selectivity.

### `6_analyze_bag_of_rules.py`

Purpose: optionally render post-hoc diagnostics for candidate groups and singleton channels during group-ablation search.

Figures include per-rule and global activation/margin/saliency summaries such as:

- `*_agonist_activation_stats.png` and `agonist_activation_stats.png`.
- `*_agonist_margin_stats.png`.
- `*_agonist_saliency_stats.png` and `agonist_saliency_stats.png`.

Disable these with `--skip_agonist_activation_stats`, `--skip_agonist_margin_stats`, or `--skip_agonist_saliency_stats`.

### `3_extract_rules.py` and `lib/data_model_for_shap.py`

Purpose: rule extraction can generate SHAP summary plots through the RuleSHAP utilities. These plots are usually written under a `shap_plots/` or equivalent rule-extraction subdirectory, depending on the concrete run configuration.

Typical filenames include `shap_summary_plot_<metric>.png`, and per-neuron variants such as `shap_summary_plot_flip_<module>.png` may appear after script `7` rule extraction.

## Threshold-event / CSS diagnostic interpretation

The completed threshold-event diagnostics support a dominance-based interpretation of the overtopping-as-spiking hypothesis. Overtopping candidates are not asserted to be the only channels with causal effects. The supported claim is narrower: selected candidates have higher singleton flip rates, higher threshold recoverability, and higher **Causal Spiking Score** than matched non-candidate controls.

The primary metric is:

```text
CSS(j) = singleton_flip_any_rate(j) * max_feature held_out_abs_MCC(j, feature)
```

CSS is high only when a channel both flips a nontrivial fraction of examples under singleton replacement and has a simple threshold-like proxy that identifies those flipped examples on held-out data. Flip rate alone measures causal strength but not threshold structure. Threshold MCC alone measures separability but can over-credit units that almost never affect behavior. CSS requires both.

The generated report in `overtopping_spiking_report/` summarizes the completed runs. In that export, candidate units have higher median flip-any rate and higher median CSS than random non-candidate controls, with paired run/baseline comparisons reported in `statistical_results.json`, `paired_flip_rate_by_run_baseline.csv`, and `paired_css_by_run_baseline.csv`.

## Circuit selection levels: edge, node, and neuron

The EAP/EAP-IG utilities can select circuit elements at several granularities.

- `edge`: individual directed connections between components. This is the most path-specific and compact representation, but it is higher variance and more sensitive to baseline details.
- `node`: whole attention heads or MLP blocks. This is usually the most stable and interpretable level for component-level circuit reporting, especially under mean-positional replacement.
- `neuron`: individual MLP channels or output dimensions. This is the key basis for overtopping-channel discovery, but the search space is large and interactions can require groups.

For mean-positional or mean-donor replacement, a practical pattern is to start with node-level sweeps for a stable backbone and then refine MLPs at neuron level. For overtopping scans, the primary localization target is the neuron/channel basis, and large-model rows in the paper should be interpreted as basis-specific, not as an exhaustive search over every possible representation.

## EAP/EAP-IG library notes

The local `lib/eap/` package implements circuit discovery utilities for autoregressive TransformerLens-compatible language models. It provides:

- Graph construction over transformer components.
- Node, edge, and neuron graph granularities.
- Edge Attribution Patching (EAP).
- EAP with integrated gradients over inputs.
- EAP with integrated gradients over activations.
- Clean-corrupted attribution variants.
- Top-N and search-style circuit selection.
- Circuit evaluation under corruptions or ablations.

Model compatibility is best for autoregressive transformer LMs whose residual streams can be treated as additive sums over prior components. Pre-LayerNorm architectures are the cleanest fit. Grouped-query-attention models may require ungrouping if attention-head-level graph operations are needed.

## Task modules

Task modules live under `lib/tasks/` and expose a `TASK_SPEC` compatible with `lib/task_spec.py` and `lib/feature_extraction_runner.py`.

Main paper task modules:

- `lib.tasks.arithmetic_task`: bare arithmetic prompts such as `a+b=`, `a-b=`, `a*b=`, and `a/b=`; target is exact normalized numerical match.
- `lib.tasks.grammar_acceptability_task`: CoLA-style grammar acceptability judgments; target is binary acceptability correctness.
- `lib.tasks.hans_nli_task`: HANS validation examples; target is binary entailment/non-entailment correctness.
- `lib.tasks.random_fsm_task`: freshly sampled binary finite-state machines; target is exact final-state prediction.
- `lib.tasks.bon_jailbreaking_task`: augmented jailbreak prompts; target is classifier-labeled jailbreak success.

Additional modules are included for exploratory or auxiliary tasks, such as `llmsafe_alignment_task`, `cognitive_bias_sensitivity_task`, `code_in_the_haystack_task`, and `cve_infile_vulnerability_detection_task`.

## Output layout

A typical working run writes:

```text
data/<task>/<provider>/<model>/
  feature_report/
    dataset_stats.json
    features.json
    scores.csv
  rule_extraction_results/
    association_rules_*.csv
    rule_combo_*.csv
    shap_plots/
    neuron_flip_rules/
      stats/<run>/
        flip_stats_by_neuron.csv
        flip_stats_global.json
        flip_stats_top<num>.pdf
        rule_combo_metrics_all.csv
        rule_combo_metrics_best_per_neuron.csv
        rule_metrics_distributions.pdf
        high_quality_neuron_flip_coverage.pdf
        high_quality_neuron_flip_coverage_by_layer.pdf
        agonist_metric_*.csv|json|png|pdf
  neural_circuit_discovery_results/
    <method>/<run>/
      neural_circuits/
        manifest.json
        dataset_info.json
        rule_*/ or spectral_cluster_*/
      bag_of_rules/<label>/
        per_rule/<target>/rule_*.json
        rule_knockout.json
        neuron_bucket_stats.json
      spiking_diagnostics/
        aggregate_flip_stats.csv
        aggregate_unit_tests.csv
        aggregate_binned_curves.csv
        threshold_event_summary*.json
        figures/
```

The clean export produced by `clean_results_for_export.py` keeps only a subset of these files, primarily final tables, plots, and JSON summaries.

## Cleaning results for export

Use this script after a full run to create a smaller artifact tree suitable for sharing or archiving.

```bash
python clean_results_for_export.py data results --force --manifest results_manifest.json
```

Options:

- `--dry-run`: print counts without copying.
- `--verbose`: print every kept and skipped file.
- `--reference-zip <zip>`: derive the allowed schema from a reference archive rather than the embedded schema.
- `--manifest <path>`: write kept/skipped path details to JSON.

The embedded schema allows paths with dynamic task, provider, model, module, and stats-run names. It always excludes `.pkl` files, cache folders, macOS metadata, and `is_correct_*` intermediate directories.

## Reproducing the shipped paper figures from a clean export

Assuming `results/` was produced with `clean_results_for_export.py`:

```bash
mkdir -p figures
python make_competence_vs_overtopping_paper_figures.py \
  --results-dir results \
  --out figures/fig_competence_vs_coverage.pdf \
  --layout phase-panels \
  --paper-size wide \
  --score-mode phase-specific \
  --baseline any \
  --paper-figures all \
  --paper-figures-dir figures \
  --paper-baseline mean-donor
```

This regenerates the main scatter and the phase/checkpoint/size panels. Each PDF has a corresponding CSV with the plotted rows.

## Reproducing the threshold-event report figures

After running `run_spiking_diagnostics.sh`, collect the diagnostic outputs and render the aggregate report:

```bash
COLLECT_ONLY=1 RESULTS_ZIP=spiking_diagnostics_results_for_inspection.zip bash run_spiking_diagnostics.sh

python generate_overtopping_spiking_report.py \
  --zip spiking_diagnostics_results_for_inspection.zip \
  --out overtopping_spiking_report
```

If the results are already extracted instead of zipped:

```bash
python generate_overtopping_spiking_report.py \
  --root . \
  --out overtopping_spiking_report
```

## Reproducibility and performance notes

- Most scripts expose `--seed` or `--random_seed` and call deterministic seeding utilities.
- Reruns depend on exact model weights, tokenizer versions, task data, and hardware-supported numerical kernels.
- Representation caches and model caches can be large; keep them out of clean exports.
- `--decode_only` restricts interventions to answer-generation positions and is useful for separating prompt-processing effects from answer-production effects.
- `mean`, `mean-donor`, `mean-positional`, `mean-donor-positional`, and `zero` define different counterfactual replacement baselines. The paper's primary exported rows often prefer mean-donor when available.
- For quick debugging, reduce `--circuit_size`, `--max_number_of_circuits_to_analyze`, `--spiking_max_points`, and the number of models/tasks in the runner scripts.

## Troubleshooting

- **Model download failures:** set `HF_HOME` or `TRANSFORMERS_CACHE` to a location with enough disk space. If all weights are local, set `HF_HUB_OFFLINE=1`.
- **OpenMP or XGBoost import errors:** install `libomp` on macOS or `libgomp` on Linux.
- **Out-of-memory during circuit discovery or ablation:** lower `--batch_size`, `--circuit_size`, `--max_pairs_per_circuit`, or `--spiking_max_points`.
- **No rules found:** inspect `feature_report/scores.csv`, target prevalence, and feature columns. Rule extraction cannot produce meaningful rules if the target is constant or features are uninformative.
- **No overtopping channels found:** this can be a valid zero-discovery configuration. Check `dataset_stats.json`, task score, run phase, replacement baseline, and whether the intended stats/run directory exists.
- **Figure script finds no points:** verify that the input tree contains `feature_report/dataset_stats.json` and `rule_extraction_results/neuron_flip_rules/stats/<run>/flip_stats_global.json`, or use a `results/` directory produced by `clean_results_for_export.py`.

## Citation and artifact note

Use this package with the accompanying paper. The code is designed to reproduce and audit the paper's overtopping, phase, checkpoint, scale, and threshold-event diagnostics, subject to the same model-weight, environment, and compute assumptions used in the original runs.
