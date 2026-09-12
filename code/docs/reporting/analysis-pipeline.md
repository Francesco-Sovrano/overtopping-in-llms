# Analysis and reporting pipeline

The reporting layer converts persistent experiment artifacts under `data/` into validated analysis tables, population audits, statistical summaries, machine-readable figure data, and manuscript-facing products under `results/`.

The standard repository entry point is:

```bash
./generate_results.sh
```

Direct invocation from `code/`:

```bash
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile study-56
```

Reporting uses measured model-backed causal outputs. Metric applicability and artifact availability are recorded separately; strict builds fail when an applicable required measurement is unavailable.

## Declared analysis populations

| Analysis | Population |
|---|---|
| Configured study table | all 56 overtopping settings; metric availability is recorded per setting |
| RQ1 Figure 2 | all 56 configured settings, fit separately by intervention phase |
| RQ2 composition and singleton-versus-joint decomposition | all evaluable settings within replacement regime; mean-donor main, mean/mean-positional separate |
| RQ3 threshold-event reporting | every configured setting with compatible threshold-event inputs |
| RQ3 graded support-specific report | every configured setting with compatible graded outputs |
| RQ4 Pythia | configured checkpoint runs |
| Poisoning | configured task/model/seed runs under `data/poisoning/` |

Poisoning runs do not enter the RQ1–RQ3 overtopping populations.

## Main reporting sequence

`reporting.generate_final_results` orchestrates the following analysis modules.

### 1. Completed-experiment summary

```text
studies.overtopping.analysis.stage01_visualize_experiment_results
```

When a runner manifest is supplied, this optional descriptive stage summarizes the completed runs named by that manifest. Its output is not used to define the RQ1--RQ3 study populations; those populations are constructed from the configured study table in Stage 02.

### 2. Configured study table

```text
studies.overtopping.analysis.stage02_overtopping_latex_tables
```

Builds the complete 56-row configured study table and materializes pooled, directional, concentration, redundancy, joint-effect, and matched-null fields when available. Missing derived metrics remain explicit missing values/statuses; they do not remove settings from the manifest.

Representative directional fields:

```text
U_J_i2c   singleton-union reach on B(x)=0 rows, 0→1
U_J_c2i   singleton-union reach on B(x)=1 rows, 1→0
s_1_i2c   strongest 0→1 singleton effect
s_1_c2i   strongest 1→0 singleton effect
```

Directional fields retain their own eligible denominators.

### 3. Required-metric audit

```text
studies.overtopping.analysis.stage03_audit_required_metrics
```

Checks required fields and compatible schemas for the configured study profile. RQ1 singleton requirements apply to all configured settings. RQ2 simultaneous-set, matched-set, composition-decomposition, and optional CMC requirements apply when the frozen candidate set is nonempty.

### 4. Configured-study statistics

```text
studies.overtopping.analysis.stage04_analyze_primary_metrics
```

Computes cross-setting diagnostics over the configured study table, with metric-specific complete cases, including raw correlations, bootstrap intervals, task/phase analyses, and configured adjusted models.

Fields named `N_t_*_density` or `N_t_*_per_1k_layer` are normalized by model `d_model`. They are width-normalized discovered-candidate counts.

### 5. Manuscript tables and sidecars

```text
studies.overtopping.analysis.stage05_generate_manuscript_outputs
```

Builds manuscript tables and machine-readable sidecars from the validated configured-study and interaction products.

### 6. RQ1, RQ2, and Pythia figures

```text
studies.overtopping.analysis.stage06_competence_vs_overtopping_figures
studies.overtopping.analysis.stage06_manuscript_story_figures
```

These modules generate:

- RQ1 competence versus directional reach/high-effect-count figures;
- RQ2 composition, matched-set, and singleton-versus-joint decomposition figures;
- Pythia checkpoint trajectories used by RQ4.

RQ1 Figure 2 resolves the exact 56-setting registry and validates the expected 26 input+output / 30 decode-only phase split.

## RQ3 reporting

RQ3 combines threshold-event diagnostics, same-candidate graded positive/non-flip support, and continuous endpoint-margin geometry. Stage-8 preemption is aggregated separately as a secondary subtype analysis.

### Threshold diagnostics source resolution

The reporting driver accepts an explicit source:

```bash
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile study-56 \
  --spiking-source /path/to/threshold_diagnostics
```

Without `--spiking-source`, it searches compatible locations under `data/` for an aggregate threshold payload.

A directory is recognized as a basic threshold source when it contains:

```text
aggregate_flip_stats.csv
aggregate_unit_tests.csv
```

Full threshold-shape validation additionally requires:

```text
aggregate_activation_flip_rows.csv
```

The descriptive oriented-bin panel uses:

```text
aggregate_binned_curves.csv
```

### Aggregate candidate/control report

```text
studies.overtopping.analysis.stage07_overtopping_spiking_report
```

Principal outputs under:

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/
```

include:

```text
statistical_results.json
threshold_testability_audit.csv
binned_curve_aggregate.csv
population_audit.csv
population_coverage.json
```

This module computes aggregate candidate/control causal-strength and threshold-testability summaries and writes the descriptive oriented-bin response used by S4.

### Nested threshold-shape validation

```text
studies.overtopping.analysis.stage08_threshold_shape_validation
```

Outputs are written under:

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/threshold_shape_validation/
```

Principal files include:

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

This module performs nested feature selection and held-out model evaluation and writes the canonical threshold-event manuscript panels.

### Graded agonist aggregation

```text
studies.overtopping.analysis.stage08_graded_agonist_report
```

The report resolves each expected run's:

```text
<stage7 stats dir>/graded_agonist_intervention/
```

and writes:

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/graded_agonist/
```

Principal files:

```text
graded_agonist_population_audit.csv
graded_agonist_all_units.csv
graded_agonist_by_condition.csv
graded_agonist_dose_by_condition.csv
graded_agonist_support_contrast_by_condition.csv
graded_agonist_dose_support_contrast.csv
graded_agonist_report_status.json
```

### Continuous-margin and population event analysis

When Stage 7b recorded endpoint margins, each run contains:

```text
<stage7 stats dir>/graded_agonist_intervention/margin_mechanism_test/
```

`stage08_graded_agonist_report` aggregates the per-example margin diagnostics into condition- and direction-level tables, including `graded_margin_all_examples.csv`, `graded_margin_by_condition.csv`, and `graded_margin_by_direction.csv`.

Population event-localization and causal-strength figures are generated by:

```text
studies.overtopping.analysis.stage10_rq3_spiking_story_figures
```

This stage uses paper-standard 11-dose, endpoint-reproduced divergence-margin trajectories and writes its machine-readable summaries into the graded-analysis directory.

### Secondary preemption aggregation

```text
studies.overtopping.analysis.stage09_preemption_report
```

This reporter reads exact-manifest per-run `interaction_validation/preemption_pair_summary.csv` files. Only the corrected schema is aggregated. Incompatible rows are excluded through the population audit. The output is analysis-only under `results/analysis/rq3_threshold_event/spiking_diagnostics/preemption/` and is not required for the primary RQ3 claim.

## Figure 4 output set

```text
results/paper/figures/04_rq3_spiking_cut/
├── fig4a_candidate_control_spiking_cut_summary.pdf
├── fig4b_threshold_shape_model_comparison_by_direction.pdf
├── fig4c_graded_agonist_dose_response.pdf
├── fig4d_graded_margin_affine_null.pdf
├── fig4e_population_event_and_strength.pdf
├── fig4s1_threshold_testability_by_condition.pdf
├── fig4s2_strength_matched_thresholdability.pdf
├── fig4s3_nested_tecs_lower_bound_ecdf.pdf
├── fig4s4_threshold_tail_response_by_direction.pdf
├── fig4s5_graded_agonist_single_crossing.pdf
├── fig4s6_graded_margin_condition_diagnostics.pdf
├── fig4s7_graded_margin_condition_heatmap.pdf
├── fig4s8_graded_behavior_competence_reach_io.pdf
├── fig4s8_graded_behavior_competence_reach_out.pdf
├── fig4s9_graded_margin_competence_reach_io.pdf
├── fig4s9_graded_margin_competence_reach_out.pdf
├── fig4s10_population_event_localization.pdf
├── fig4s11_strength_concentration_paired.pdf
├── fig4s12_arithmetic_competence_concentration.pdf
└── fig4s13_affine_null_transient_events.pdf
```

Availability is analysis-specific. Threshold diagnostics are required for Figure 4a, Figure 4b, and S1–S4. Graded behavioral outputs are required for Figure 4c and S5. Endpoint-margin outputs are required for Figure 4d and the continuous-margin supplements. Figure 4e and S10–S13 additionally require the paper-standard 11-dose endpoint-reproduced margin population.

## Poisoning aggregation

When poisoning runs are available, final reporting invokes:

```text
studies.poisoning.stage08_aggregate_cross_seed
studies.poisoning.stage08_plot_cross_seed
```

and publishes configured per-run poisoning visuals. Training seed is the trajectory-level replication unit.

## Result tree

```text
results/
├── analysis/
│   ├── primary_matrix/
│   ├── figure_data/
│   ├── reproducibility/
│   ├── rq3_threshold_event/
│   │   └── spiking_diagnostics/
│   │       ├── statistical_results.json
│   │       ├── threshold_testability_audit.csv
│   │       ├── binned_curve_aggregate.csv
│   │       ├── threshold_shape_validation/
│   │       ├── graded_agonist/
│   │       └── report_status.json
│   └── rq4_learning/
└── paper/
    ├── figures/
    │   ├── 02_rq1_prevalence/
    │   ├── 03_rq2_composition/
    │   ├── 04_rq3_spiking_cut/
    │   └── 05_rq4_learning/
    └── tables/
```

Machine-readable source data belong under `results/analysis/`. Manuscript-facing PDFs and tables belong under `results/paper/`.

## Reproducibility audits

Population and output audits are written under:

```text
results/analysis/reproducibility/
```

The final reporting driver validates the `study-56` configured-study profile and required manuscript outputs before completing a standard build.

## Reporting controls

Direct reporting options include:

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

Repository wrappers can expose additional environment-level convenience controls. The standard manuscript build validates the complete `study-56` registry and records metric-specific completeness across those settings. Verified Stage-6 empty candidate sets are counted as completed zero-candidate observations rather than incomplete experiments. The resolver accepts the current Stage-6 bag layout and the two alternate export layouts (`bag_of_rules/<bag>` and `neural_circuits/<bag>`). The metric audit prints verified zero-candidate settings separately from genuinely incomplete settings.
