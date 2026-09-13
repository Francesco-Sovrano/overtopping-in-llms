# Analysis and reporting pipeline

The reporting layer reads persistent model-backed artifacts from `data/`, constructs configured analysis populations, audits metric availability, computes aggregate statistics, and writes derived outputs under `results/`.

Standard entry point:

```bash
./generate_results.sh
```

Direct invocation from `code/`:

```bash
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile configured
```

## Population construction

The configured overtopping manifest is constructed before metric-specific filtering.

| Analysis | Population rule |
|---|---|
| Configured study table | every unique setting in the selected manifest |
| RQ1 | settings with the required singleton metrics, analyzed separately by intervention phase |
| RQ2 | settings with nonempty frozen candidate sets and compatible set-level outputs, separated by replacement regime |
| RQ3 threshold | settings with compatible threshold diagnostics |
| RQ3 graded/margin/temporal | settings with the corresponding model-backed artifacts |
| RQ4 Pythia | configured checkpoint settings |
| Poisoning | configured task/model/seed runs under `data/poisoning/` |

Poisoning runs are outside the RQ1–RQ3 overtopping population.

## Orchestration sequence

### Completed-run summary

```text
studies.overtopping.analysis.stage01_visualize_experiment_results
```

When a runner manifest is supplied, this stage summarizes completed entries from that manifest. It does not define the configured analysis population.

### Configured study table

```text
studies.overtopping.analysis.stage02_overtopping_latex_tables
```

The table includes configured rows and materializes available pooled, directional, concentration, redundancy, simultaneous-set, and matched-null metrics. Missing derived values remain explicit missing values or statuses.

Directional fields:

```text
U_J_i2c   singleton-union reach on B(x)=0 rows, 0→1
U_J_c2i   singleton-union reach on B(x)=1 rows, 1→0
s_1_i2c   strongest 0→1 singleton effect
s_1_c2i   strongest 1→0 singleton effect
```

### Required-metric audit

```text
studies.overtopping.analysis.stage03_audit_required_metrics
```

The audit checks schema compatibility and required metrics for the selected profile. Singleton requirements apply to configured settings where the corresponding RQ1 metric is defined. Set-level requirements apply when the frozen candidate set is nonempty. CMC completeness can be disabled with `--skip-cmc-requirement`.

### Configured-study statistics

```text
studies.overtopping.analysis.stage04_analyze_primary_metrics
```

This stage computes metric-specific complete-case cross-setting summaries, including correlations, bootstrap intervals, task/phase analyses, and configured adjusted models.

Fields named `N_t_*_density` or `N_t_*_per_1k_layer` are width-normalized discovered-candidate counts based on model `d_model`.

### Derived tables and sidecars

```text
studies.overtopping.analysis.stage05_generate_manuscript_outputs
```

The module name is part of the implementation interface. It writes derived tables and machine-readable sidecars from validated analysis products.

### RQ1, RQ2, and checkpoint rendering

```text
studies.overtopping.analysis.stage06_competence_vs_overtopping_figures
studies.overtopping.analysis.stage06_manuscript_story_figures
```

These modules consume validated analysis tables to produce RQ1 summaries, RQ2 composition outputs, and Pythia checkpoint trajectories.

## RQ3 threshold diagnostics

An explicit aggregate threshold source can be supplied with:

```bash
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile configured \
  --spiking-source /path/to/threshold_diagnostics
```

A basic threshold source contains:

```text
aggregate_flip_stats.csv
aggregate_unit_tests.csv
```

Threshold-shape validation additionally uses:

```text
aggregate_activation_flip_rows.csv
```

Oriented-bin summaries use:

```text
aggregate_binned_curves.csv
```

Aggregate threshold reporting:

```text
studies.overtopping.analysis.stage07_overtopping_spiking_report
```

Primary output directory:

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/
```

Generated files:

```text
statistical_results.json
threshold_testability_audit.csv
binned_curve_aggregate.csv
population_audit.csv
population_coverage.json
```

## Nested threshold-shape validation

```text
studies.overtopping.analysis.stage08_threshold_shape_validation
```

Output directory:

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/threshold_shape_validation/
```

Generated files:

```text
threshold_shape_model_comparison.csv
threshold_shape_model_comparison_repeats.csv
threshold_shape_unit_population.csv
threshold_shape_condition_population.csv
threshold_strength_matched_pairs.csv
threshold_strength_matched_condition.csv
threshold_shape_statistical_results.json
threshold_shape_status.json
```

The procedure performs feature selection and model fitting inside training folds and evaluates predictive performance on untouched holdout folds.

## Graded intervention aggregation

```text
studies.overtopping.analysis.stage08_graded_agonist_report
```

Per-run source directory:

```text
<stage7 stats dir>/graded_agonist_intervention/
```

Aggregate output directory:

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/graded_agonist/
```

Generated files:

```text
graded_agonist_population_audit.csv
graded_agonist_all_units.csv
graded_agonist_by_condition.csv
graded_agonist_dose_by_condition.csv
graded_agonist_support_contrast_by_condition.csv
graded_agonist_dose_support_contrast.csv
graded_agonist_report_status.json
```

## Continuous endpoint-margin aggregation

When Stage 07b records endpoint margins, per-run diagnostics are stored under:

```text
<stage7 stats dir>/graded_agonist_intervention/margin_mechanism_test/
```

Cross-run tables include:

```text
graded_margin_all_examples.csv
graded_margin_by_condition.csv
graded_margin_by_direction.csv
```

Population event-localization and strength summaries are generated by:

```text
studies.overtopping.analysis.stage10_rq3_spiking_story_figures
```

The population margin aggregation uses 11-dose trajectories that reproduce the configured natural and full-intervention endpoints.

## Temporal-cutoff aggregation

Per-run prefix and suffix intervention sweeps are aggregated by:

```text
studies.overtopping.analysis.stage11_rq3_temporal_cutoff_story
```

The aggregate includes cumulative-effect capture, transition-level incremental gain, T50/T80 summaries, cross-sweep EVENT validation, and phase-specific tables when compatible temporal artifacts are available.

## Preemption aggregation

```text
studies.overtopping.analysis.stage09_preemption_report
```

The reporter reads compatible per-run `interaction_validation/preemption_pair_summary.csv` files and writes aggregate outputs under:

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/preemption/
```

Preemption is a separate optional interaction endpoint.

## Poisoning aggregation

When compatible poisoning runs are available, reporting can invoke:

```text
studies.poisoning.stage08_aggregate_cross_seed
studies.poisoning.stage08_plot_cross_seed
```

The training run/seed is the cross-seed replicate unit.

## Result tree

```text
results/
├── analysis/
│   ├── primary_matrix/
│   ├── figure_data/
│   ├── table_data/
│   ├── reproducibility/
│   ├── rq2_composition/
│   ├── rq3_threshold_event/
│   └── rq4_learning/
└── paper/
    ├── figures/
    └── tables/
```

`results/paper/` stores rendered outputs and `results/analysis/` stores machine-readable analysis products.

## Reproducibility audits

Population, metric-completeness, and output audits are written under:

```text
results/analysis/reproducibility/
```

A verified Stage-06 empty candidate set is recorded as a completed zero-candidate outcome where applicable. Missing required artifacts are reported separately.

## Command-line controls

`reporting.generate_final_results` accepts:

```text
--data-root
--results-root
--primary-profile
--poisoning-root
--catalogue-json
--skip-paper-figures
--spiking-source
--spiking-max-points
--skip-spiking-report
--skip-poisoning-report
--require-complete-metrics
--skip-cmc-requirement
```

`--catalogue-json` selects an explicit saved configured population. Without it, the `configured` profile is derived from the current code registry.
