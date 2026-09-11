# RQ3 — support-specific thresholded causal integration

## Question

> **Do graded overtopping interventions produce support-specific threshold crossings?**

RQ3 tests whether a continuously varied intervention on a fixed high-leverage candidate produces a localized, usually persistent transition in the binary behavioral endpoint, and whether that transition distinguishes examples that are susceptible to the same candidate from source-state examples that remain non-flipping at full dose.

Endogenous scalar threshold prediction is reported as a separate observability analysis. Dominant-secondary preemption remains available as a secondary subtype analysis but is not required for the primary RQ3 claim.

## Scientific distinction

RQ3 separates three properties:

1. **causal leverage** — a candidate can change behavior under a full intervention;
2. **graded threshold geometry** — continuous intervention strength produces a localized persistent behavioral crossing on susceptible examples;
3. **endogenous visibility** — susceptibility can be predicted from a simple scalar measured before intervention.

The first two define the intervention-level thresholded causal claim. The third asks whether that susceptibility is visible through a one-dimensional endogenous readout.

A separate optional interaction question asks whether secondary causal effects become smaller when a dominant contribution is present. That is evidence for a stronger first-sufficient subtype, not a prerequisite for thresholded causal integration.

## Population and candidate identity

Paper-facing RQ3 evaluation uses the held-out `test` split.

The exact reporting manifest starts from:

```text
48 configured overtopping settings
```

Each threshold/graded analysis then retains the settings with the exact required
artifacts and reports that denominator explicitly. Eligibility is determined by
the 48-setting manifest plus the analysis-specific artifact contract.

Candidate identity includes discovery direction:

```text
positive discovery baseline -> 1→0 / c2i
negative discovery baseline -> 0→1 / i2c
```

A coordinate is evaluated only in discovery directions in which it was frozen by the discovery pipeline.

For candidate `j` and direction `d`, define two source-state support classes from the held-out Stage-7 singleton evaluation:

```text
S_j+ = examples where the full singleton intervention flips B(x)
S_j- = examples in the same source state where the same singleton intervention does not flip B(x)
```

These classes provide a same-candidate support-specific contrast. They do not estimate unconditional crossing prevalence unless the source-state population is sampled or weighted accordingly.

## Part A — causal leverage and endogenous threshold visibility

### Per-run diagnostics

Implementation:

```text
studies/overtopping/analysis/threshold_event_diagnostics.py
```

Pipeline control:

```text
RUN_THRESHOLD_EVENT_POSTHOC=true
```

Principal aggregate inputs:

```text
aggregate_flip_stats.csv
aggregate_unit_tests.csv
aggregate_population_summary.csv
aggregate_binned_curves.csv
aggregate_activation_flip_rows.csv
threshold_spiking_experiment_aggregate.json
```

### Structural controls

Candidate units are compared with independently sampled non-candidate controls matched within the relevant structural locus. Causal-strength matching is a separate sensitivity analysis rather than the primary control definition.

### Causal strength

The causal-strength endpoint is the held-out singleton flip rate in the candidate's frozen discovery direction.

### Threshold testability

A unit is threshold-testable only when its evaluated population contains enough flip and non-flip support for the configured repeated train/holdout procedure. Testability is reported separately from threshold-fit quality.

### Nested held-out threshold prediction

Implementation:

```text
studies/overtopping/analysis/stage08_threshold_shape_validation.py
```

The procedure selects the scalar feature and fits the predictive model on training folds, then evaluates it on untouched held-out folds. The primary threshold-quality metric is absolute Matthews correlation coefficient:

```text
|MCC|
```

### TECS

Threshold-event causal score combines causal strength and threshold visibility:

```text
TECS(j) = s_j * |MCC_j|
```

TECS is a visibility-weighted causal score. It is not direct evidence that the model implements a literal one-dimensional threshold at the measured scalar.

### Threshold-shape model comparison

The validation compares held-out predictive performance for:

```text
constant
hard threshold
logistic
isotonic
```

This analysis tests predictive shape, not intervention effect magnitude.

## Part B — graded causal intervention

### Intervention

Implementation:

```text
studies/overtopping/analysis/graded_agonist_intervention.py
```

Pipeline control:

```text
RUN_GRADED_AGONIST_INTERVENTION=true
```

For frozen candidate `j`, intervention strength is interpolated between the natural channel value and the configured replacement:

```text
h_j(lambda, x) = (1 - lambda) h_j(x) + lambda h_replacement_j
```

with the default dose grid

```text
lambda = 0, 0.1, 0.2, ..., 1.0.
```

`lambda=0` reproduces the natural channel value and `lambda=1` reproduces the Stage-7 singleton intervention.

### Support classes

Known-flip support `S_j+` and same-channel non-flip support `S_j-` are evaluated with the same candidate, source state, intervention phase, replacement baseline, and dose grid.

Non-flip support is enabled with:

```text
GRADED_AGONIST_NEGATIVE_SUPPORT=true
```

Selection uses frozen discovery information and does not use graded-response outcomes.

### Per-example trajectory quantities

```text
first_flip_dose
n_state_changes
persistent_after_first_flip
single_crossing
natural_state_matches_stage7_baseline
full_dose_flipped
full_dose_matches_stage7_support
```

A `single_crossing` trajectory changes once from the natural state and remains changed at every larger tested dose.

For `S_j-`, the principal endpoint is stability at the natural state across the dose sweep, with transient intermediate flips reported separately.

### Primary support-specific test

The central contrast asks whether:

- `S_j+` has a high rate of localized persistent crossings; and
- `S_j-` remains predominantly stable under the same candidate and dose schedule.

The analysis therefore tests whether the graded transition is specific to candidate-example susceptibility rather than being a generic consequence of increasing intervention dose.

### Cross-run aggregation

Implementation:

```text
studies/overtopping/analysis/stage08_graded_agonist_report.py
```

Aggregation is hierarchical:

1. summarize examples within candidate and support class;
2. summarize candidates within a run/baseline/direction condition;
3. use the condition as the cross-run inferential unit.

Principal outputs:

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/graded_agonist/
├── graded_agonist_population_audit.csv
├── graded_agonist_all_units.csv
├── graded_agonist_by_condition.csv
├── graded_agonist_dose_by_condition.csv
├── graded_agonist_support_contrast_by_condition.csv
├── graded_agonist_dose_support_contrast.csv
└── graded_agonist_report_status.json
```

## Secondary subtype analysis — dominant-secondary preemption

The repository retains a dominant-secondary preemption assay in Stage 8 and its aggregate reporter:

```text
pipeline/stage08_validate_interactions.py
studies/overtopping/analysis/stage09_preemption_report.py
```

This assay asks whether a secondary candidate's binary marginal contribution differs according to an independently fitted endogenous event associated with a frozen dominant candidate. Pair identity and order are frozen from discovery data.

Outputs are written under:

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/preemption/
```

This analysis is secondary. It can identify evidence compatible with a first-sufficient subtype, but it is not used to define the primary threshold-crossing claim and it is not a substitute for the RQ2 singleton-versus-joint decomposition.

## Interpretation

RQ3 supports thresholded causal integration when a fixed candidate shows a structured graded crossing on susceptible examples while same-channel non-flip examples remain predominantly non-crossing.

Possible interaction regimes are interpreted jointly with RQ2:

- high joint preservation and low suppression are compatible with a saturating high-leverage regime;
- singleton-reachable suppression indicates antagonistic or masking interaction under the full set;
- coalition-only effects indicate cooperative or coalition-dependent integration;
- positive preemption evidence, when present, supports a stronger first-sufficient subtype.

No one-dimensional endogenous threshold code is required for the intervention-level claim.

## Manuscript Figure 4

Paper directory:

```text
results/paper/figures/04_rq3_spiking_cut/
```

Main panels:

```text
fig4a_candidate_control_spiking_cut_summary.pdf
fig4b_threshold_shape_model_comparison_by_direction.pdf
fig4c_graded_agonist_dose_response.pdf
```

Supplementary panels:

```text
fig4s1_threshold_testability_by_condition.pdf
fig4s2_strength_matched_thresholdability.pdf
fig4s3_nested_tecs_lower_bound_ecdf.pdf
fig4s4_threshold_tail_response_by_direction.pdf
fig4s5_graded_agonist_single_crossing.pdf
```

The preemption report remains analysis-only unless explicitly promoted for a subtype analysis.
