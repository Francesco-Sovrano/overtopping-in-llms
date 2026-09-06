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
  --primary-profile iclr-28
```

Reporting does not replace missing model-backed causal measurements with derived approximations. Missing required scientific inputs are represented by status/audit outputs or cause the configured manuscript build to fail validation.

## Declared analysis populations

| Analysis | Population |
|---|---|
| Primary matrix | 28 `paper-primary` settings |
| RQ1 Figure 2 | 28 primary + 11 auxiliary settings |
| RQ2 composition | 28 primary settings |
| RQ3 threshold-event reporting | 28 primary settings required; 11 auxiliary settings supplementary when compatible inputs are present |
| RQ3 graded report | 28 primary settings required; auxiliary settings supplementary when graded outputs are present |
| RQ4 Pythia | configured checkpoint runs |
| Poisoning | configured task/model/seed runs under `data/poisoning/` |

Poisoning runs do not enter the RQ1–RQ3 overtopping populations.

## Main reporting sequence

`reporting.generate_final_results` orchestrates the following analysis modules.

### 1. Catalogue visualization

```text
studies.overtopping.analysis.stage01_visualize_experiment_results
```

Creates catalogue-level summaries from configured overtopping runs.

### 2. Primary table

```text
studies.overtopping.analysis.stage02_overtopping_latex_tables
```

Builds the 28-row primary matrix and materializes pooled, directional, concentration, redundancy, joint-effect, and matched-null fields when available.

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

Checks required fields, evaluation variants, and interaction products for the configured primary profile.

### 4. Primary statistics

```text
studies.overtopping.analysis.stage04_analyze_primary_metrics
```

Computes cross-setting statistics for the primary matrix, including raw correlations, bootstrap intervals, task/phase analyses, and configured adjusted models.

Fields named `N_t_*_density` or `N_t_*_per_1k_layer` are normalized by model `d_model`. They are width-normalized discovered-candidate counts.

### 5. Manuscript tables and sidecars

```text
studies.overtopping.analysis.stage05_generate_manuscript_outputs
```

Builds manuscript tables and machine-readable sidecars from validated primary-matrix and interaction products.

### 6. RQ1, RQ2, and Pythia figures

```text
studies.overtopping.analysis.stage06_competence_vs_overtopping_figures
studies.overtopping.analysis.stage06_manuscript_story_figures
```

These modules generate:

- RQ1 competence versus directional reach/high-effect-count figures;
- RQ2 composition and matched-set figures;
- Pythia checkpoint trajectories used by RQ4.

RQ1 Figure 2 resolves the exact 39-setting catalogue and validates the expected 17 input+output / 22 decode-only phase split.

## RQ3 reporting

RQ3 combines threshold-event diagnostics, per-run graded agonist outputs, and corrected Stage-8 preemption summaries.

### Threshold diagnostics source resolution

The reporting driver accepts an explicit source:

```bash
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile iclr-28 \
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
graded_agonist_report_status.json
```

### Preemption aggregation

```text
studies.overtopping.analysis.stage09_preemption_report
```

This reporter reads exact-manifest per-run `interaction_validation/preemption_pair_summary.csv` files. Only the corrected v3 preemption schema is aggregated. Stale v2 rows are excluded and recorded in `preemption_population_audit.csv` with a refresh status. The output is analysis-only under `results/analysis/rq3_threshold_event/spiking_diagnostics/preemption/`.

## Figure 4 output set

```text
results/paper/figures/04_rq3_spiking_cut/
├── fig4a_candidate_control_spiking_cut_summary.pdf
├── fig4b_threshold_shape_model_comparison_by_direction.pdf
├── fig4c_graded_agonist_dose_response.pdf
├── fig4s1_threshold_testability_by_condition.pdf
├── fig4s2_strength_matched_thresholdability.pdf
├── fig4s3_nested_tecs_lower_bound_ecdf.pdf
├── fig4s4_threshold_tail_response_by_direction.pdf
└── fig4s5_graded_agonist_single_crossing.pdf
```

Figure 4a, Figure 4b, and S1–S4 depend on threshold diagnostics. Figure 4c and S5 depend on graded agonist outputs.

The representative same-condition/same-layer threshold response curves are written only to the threshold-shape analysis diagnostics directory.

## Poisoning aggregation

When poisoning runs are available, final reporting invokes:

```text
studies.poisoning.stage08_aggregate_cross_seed
studies.poisoning.stage08_plot_cross_seed
```

and publishes configured per-run poisoning visuals. Training seed is the cross-run replication unit.

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

The final reporting driver validates the declared primary profile and required manuscript outputs before completing a standard build.

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

Repository wrappers can expose additional environment-level convenience controls. The standard manuscript build uses complete required primary metrics and the `iclr-28` primary profile.
