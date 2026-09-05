# RQ3 — threshold-event structure and graded causal transition

## Question

RQ3 asks two related questions about frozen high-leverage candidates:

1. **Endogenous threshold structure:** can the examples on which a candidate becomes causally decisive be predicted from a simple scalar measured before intervention, and how does that structure compare with matched non-candidate controls?
2. **Causal dose response:** when the strength of the same singleton intervention is varied continuously from the natural activation to the full replacement, does the behavioral endpoint exhibit a localized and persistent transition?

The implementation separates these questions into threshold-event diagnostics and a graded agonist intervention.

## Population and candidate identity

RQ3 uses the held-out `test` split for manuscript reporting.

The exact reporting manifest contains:

```text
28 paper-primary settings        required
11 paper-auxiliary settings      supplementary when outputs are available
```

Poisoning runs are excluded.

Candidate identity includes the discovery baseline subset:

```text
positive baseline -> 1→0 / c2i
negative baseline -> 0→1 / i2c
```

A candidate is evaluated in its recorded discovery direction. A coordinate can enter both directional analyses only when it was independently discovered in both baseline subsets.

Let `B(x)` be the binary task endpoint. For candidate `j` and direction `d`, define the held-out full-dose flip support:

```text
S_j^d = {x : x is in the source state for d and the full singleton intervention on j flips B(x)}
```

For `1→0`, source-state rows satisfy `B(x)=1`. For `0→1`, source-state rows satisfy `B(x)=0`.

## Part A — threshold-event diagnostics

### Per-run diagnostic generation

Implementation:

```text
studies/overtopping/analysis/threshold_event_diagnostics.py
```

Pipeline control:

```text
RUN_THRESHOLD_EVENT_POSTHOC=true
```

The diagnostic uses Stage-7 singleton outcomes and evaluates candidate and structural-control units on observed flip/non-flip labels. It collects endogenous scalar features, repeated threshold tests, activation/flip rows, and binned response summaries.

Aggregate files written by the diagnostic include:

```text
aggregate_flip_stats.csv
aggregate_unit_tests.csv
aggregate_population_summary.csv
aggregate_binned_curves.csv
aggregate_activation_flip_rows.csv
threshold_spiking_experiment_aggregate.json
```

These aggregate files are the source data for manuscript threshold-event reporting.

### Structural controls

The primary control population consists of independently sampled non-candidate units matched within the relevant structural locus, including layer/head strata where applicable. The primary comparison does not condition on causal strength.

A same-locus causal-strength-matched comparison is reported separately as a sensitivity analysis.

### Causal-strength endpoint

The candidate/control causal-strength endpoint is the observed held-out singleton flip rate in the candidate's discovery direction.

This endpoint establishes whether the candidate population is more causally consequential than structural controls.

### Threshold testability

Threshold quality is defined only when the evaluated unit has sufficient flip and non-flip support for the configured repeated train/holdout procedure.

The analysis therefore reports threshold testability as its own endpoint:

```text
threshold_testable_fraction
```

A unit that is not threshold-testable is not assigned an arbitrary threshold MCC.

### Nested threshold fitting

Aggregate validation implementation:

```text
studies/overtopping/analysis/stage08_threshold_shape_validation.py
```

The validation procedure separates feature selection from evaluation:

1. choose the endogenous scalar feature using training-fold data;
2. fit and orient the candidate model using training-fold data;
3. evaluate the selected model on an untouched held-out fold;
4. repeat according to the configured resampling procedure;
5. aggregate within run, structural stratum, and discovery direction.

The primary threshold-quality endpoint is nested held-out absolute Matthews correlation coefficient:

```text
|MCC|
```

### TECS

For unit `j`, threshold-event causal score combines causal strength and threshold visibility:

```text
TECS(j) = s_j * |MCC_j|
```

The manuscript analysis reports a nested TECS lower-bound endpoint over the evaluated unit population. This keeps causal strength and threshold-estimation availability explicit.

### Threshold-shape model comparison

The validation analysis compares held-out predictive performance for:

```text
constant
hard threshold
logistic
isotonic
```

The manuscript model-comparison panel uses direction-specific condition-weighted summaries. The comparison is based on held-out metrics produced by the nested threshold-shape analysis.

### Descriptive proxy-bin response

The support report also aggregates observed held-out flip rates over oriented endogenous-proxy bins.

Source after reporting aggregation:

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/binned_curve_aggregate.csv
```

The bin index is descriptive. It is not an estimated decision threshold and is not used as the primary threshold-quality statistic.

## Part B — graded agonist intervention

### Per-run experiment

Implementation:

```text
studies/overtopping/analysis/graded_agonist_intervention.py
```

Pipeline control:

```text
RUN_GRADED_AGONIST_INTERVENTION=true
```

For each selected agonist `j` and held-out example `x`, the intervention interpolates between the natural channel value and the configured Stage-7 replacement:

```text
h_j(lambda, x) = (1 - lambda) h_j(x) + lambda h_replacement_j
lambda in [0, 1]
```

Interpretation:

```text
lambda = 0   natural activation
lambda = 1   full Stage-7 replacement
```

Default dose grid:

```text
0, 0.1, 0.2, ..., 0.9, 1
```

At each dose, the task's declared binary endpoint is evaluated directly from the model.

For donor-based replacement modes, replacement values follow the Stage-7 intervention semantics. Sampling the graded support does not redefine the replacement population.

### Primary support

The primary graded population is `S_j^d`, the agonist's held-out directional full-dose flip support.

Direction-specific Stage-7 columns identify this support:

```text
1→0: flip_c2i_<unit>
0→1: flip_i2c_<unit>
```

Only rows on which the singleton outcome was actually evaluated are eligible.

### Optional same-agonist reference

The experiment can also evaluate source-state rows where the same agonist was evaluated but did not flip the endpoint at full dose:

```text
N_j^d = evaluated source-state rows for j \ S_j^d
```

Enable it with:

```text
GRADED_AGONIST_NEGATIVE_SUPPORT=true
```

This reference holds agonist identity, layer, intervention baseline, replacement rule, and discovery direction fixed.

### Selection and sampling defaults

Maximum agonists per discovery direction:

```text
GRADED_AGONIST_MAX_UNITS_PER_DIRECTION=16
```

Maximum known-flip examples per agonist:

```text
GRADED_AGONIST_MAX_POSITIVE_SUPPORT=256
```

Optional non-flip support defaults:

```text
GRADED_AGONIST_NEGATIVE_RATIO=1.0
GRADED_AGONIST_MAX_NEGATIVE_SUPPORT=256
```

Deterministic sampling seed:

```text
GRADED_AGONIST_SEED=42
```

Selection uses frozen discovery rank and does not use graded-response outcomes.

### Per-example trajectory quantities

The graded experiment records:

```text
first_flip_dose
n_state_changes
persistent_after_first_flip
single_crossing
natural_state_matches_stage7_baseline
full_dose_flipped
full_dose_matches_stage7_support
```

Definitions:

- `first_flip_dose`: first tested dose at which the endpoint differs from its natural state;
- `n_state_changes`: number of endpoint transitions over the ordered dose grid;
- `persistent_after_first_flip`: whether the changed state persists at larger doses;
- `single_crossing`: exactly one transition from the natural state followed by persistence through all larger doses;
- endpoint-reproduction fields: checks that dose 0 and dose 1 reproduce the expected Stage-7 states.

### Per-agonist summary

`graded_agonist_unit_summary.csv` contains quantities including:

```text
n_examples
natural_state_reproduction_rate
full_dose_support_reproduction_rate
full_dose_flip_rate
single_crossing_rate
median_first_flip_dose
mean_first_flip_dose
median_state_changes
```

### Cross-run aggregation

Implementation:

```text
studies/overtopping/analysis/stage08_graded_agonist_report.py
```

Dose-response aggregation is hierarchical:

1. average `flipped_from_baseline` over examples within agonist, support population, and dose;
2. take the median across agonists within a run/direction/support condition;
3. summarize condition-level values across runs with the median and interquartile range at each dose.

Single-crossing reporting computes each agonist's `single_crossing_rate`, then summarizes agonists within each run/direction condition. The run/baseline/direction condition is the cross-run inference unit.

## Reporting outputs

### Aggregate threshold-event analysis

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/
├── statistical_results.json
├── threshold_testability_audit.csv
├── binned_curve_aggregate.csv
├── population_audit.csv
├── population_coverage.json
├── report_status.json
└── threshold_shape_validation/
    ├── threshold_shape_model_comparison.csv
    ├── threshold_shape_model_comparison_repeats.csv
    ├── threshold_shape_unit_population.csv
    ├── threshold_shape_condition_population.csv
    ├── threshold_strength_matched_pairs.csv
    ├── threshold_strength_matched_condition.csv
    ├── threshold_shape_statistical_results.json
    ├── threshold_shape_status.json
    └── figures/
        └── illustrative_threshold_response_curves.pdf
```

The exact set of auxiliary diagnostic CSVs can be larger; the files above are the principal manuscript and audit inputs.

### Aggregate graded analysis

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/graded_agonist/
├── graded_agonist_population_audit.csv
├── graded_agonist_all_units.csv
├── graded_agonist_by_condition.csv
├── graded_agonist_dose_by_condition.csv
├── graded_agonist_dose_response.pdf
├── graded_agonist_single_crossing.pdf
└── graded_agonist_report_status.json
```

## Manuscript Figure 4 contract

Paper directory:

```text
results/paper/figures/04_rq3_spiking_cut/
```

### Main panels

```text
fig4a_candidate_control_spiking_cut_summary.pdf
fig4b_threshold_shape_model_comparison_by_direction.pdf
fig4c_graded_agonist_dose_response.pdf
```

**Figure 4a** reports, separately by discovery direction:

- causal strength;
- threshold-testable fraction;
- nested held-out threshold `|MCC|` among testable units;
- nested TECS lower bound.

**Figure 4b** compares constant, hard-threshold, logistic, and isotonic held-out model performance by discovery direction using condition-weighted summaries.

**Figure 4c** shows the graded behavioral dose response for known-flip support and, when enabled, the same-agonist non-flip reference.

### Supplementary panels

```text
fig4s1_threshold_testability_by_condition.pdf
fig4s2_strength_matched_thresholdability.pdf
fig4s3_nested_tecs_lower_bound_ecdf.pdf
fig4s4_threshold_tail_response_by_direction.pdf
fig4s5_graded_agonist_single_crossing.pdf
```

- **S1:** candidate/control threshold-testable fractions by run and discovery direction;
- **S2:** causal-strength-matched candidate/control threshold `|MCC|` sensitivity analysis;
- **S3:** ECDF of nested TECS lower bounds;
- **S4:** descriptive oriented-proxy-bin response by discovery direction;
- **S5:** condition-level single-persistent-crossing rate for the graded experiment.

The representative same-condition/same-layer response curves are diagnostic outputs under the threshold-shape analysis directory and are not part of the manuscript figure set.

## Reporting source requirements

`reporting.generate_final_results` resolves an RQ3 threshold-diagnostics source from `--spiking-source` or from compatible aggregate diagnostics under `data/`.

The aggregate support report requires at least:

```text
aggregate_flip_stats.csv
aggregate_unit_tests.csv
```

The full threshold-shape analysis additionally requires:

```text
aggregate_activation_flip_rows.csv
```

The descriptive S4 panel uses:

```text
aggregate_binned_curves.csv
```

The graded aggregate report resolves per-run data from each expected Stage-7 statistics directory:

```text
graded_agonist_intervention/graded_agonist_intervention.json
graded_agonist_intervention/graded_agonist_unit_summary.csv
graded_agonist_intervention/graded_agonist_dose_rows.csv.gz
```

All primary RQ3 rows are required by the manifest. Supplementary rows are included when their outputs are present.

## Runtime controls

Threshold-event controls:

```text
RUN_THRESHOLD_EVENT_POSTHOC
THRESHOLD_EVENT_TARGET
THRESHOLD_EVENT_MAX_POINTS
THRESHOLD_EVENT_MIN_POINTS
THRESHOLD_EVENT_REPEATS
THRESHOLD_EVENT_HOLDOUT_FRACTION
THRESHOLD_EVENT_N_BINS
THRESHOLD_EVENT_SEED
FORCE_THRESHOLD_EVENT_POSTHOC
```

Graded-intervention controls:

```text
RUN_GRADED_AGONIST_INTERVENTION
GRADED_AGONIST_DOSES
GRADED_AGONIST_MAX_UNITS_PER_DIRECTION
GRADED_AGONIST_MAX_POSITIVE_SUPPORT
GRADED_AGONIST_NEGATIVE_SUPPORT
GRADED_AGONIST_NEGATIVE_RATIO
GRADED_AGONIST_MAX_NEGATIVE_SUPPORT
GRADED_AGONIST_SEED
FORCE_GRADED_AGONIST_INTERVENTION
```

## Interpretation

The threshold-event and graded analyses test different properties:

- causal strength establishes intervention leverage;
- threshold testability establishes whether threshold quality can be estimated for the evaluated unit;
- nested held-out threshold MCC measures one-dimensional threshold predictability without reusing the evaluation fold for feature selection;
- TECS combines leverage and threshold visibility;
- model comparison evaluates the shape of predictive structure relative to constant, smooth, and monotonic alternatives;
- graded dose response directly measures how the behavioral endpoint changes as causal intervention strength is varied;
- single-crossing rate measures persistence and monotonicity of the binary trajectory over the tested dose grid.

No single endpoint substitutes for the others. Candidate/control threshold analyses and graded intervention trajectories should be reported according to their distinct populations and statistical units.
