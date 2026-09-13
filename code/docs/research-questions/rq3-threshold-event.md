# RQ3 — support-specific thresholded causal integration

## Objective

Determine whether graded overtopping interventions produce support-specific threshold crossings.

RQ3 tests whether a continuously varied intervention on a fixed high-leverage candidate produces a localized, usually persistent transition in the binary behavioral endpoint, and whether that transition distinguishes examples that are susceptible to the same candidate from source-state examples that remain non-flipping at full dose.

Endogenous scalar threshold prediction is reported as a separate observability analysis. Dominant-secondary preemption is a separate secondary subtype analysis.

## Scientific distinction

RQ3 separates three properties:

1. **causal leverage** — a candidate can change behavior under a full intervention;
2. **graded threshold geometry** — continuous intervention strength produces a localized persistent behavioral crossing on susceptible examples;
3. **endogenous visibility** — susceptibility can be predicted from a simple scalar measured before intervention.

The first two define the intervention-level threshold analysis. The third measures whether susceptibility is visible through a one-dimensional endogenous readout.

A separate optional interaction question asks whether secondary causal effects become smaller when a dominant contribution is present. That is evidence for a stronger first-sufficient subtype, not part of the thresholded-integration definition.

## Population and candidate identity

RQ3 evaluation uses the held-out `test` split in the standard configured analysis.

The exact reporting manifest starts from:

```text
the configured overtopping settings
```

Each threshold/graded analysis then retains the settings with the exact required
artifacts and reports that denominator explicitly. Eligibility is determined by
the configured manifest plus the analysis-specific artifact contract.

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

## Part C — continuous endpoint-margin geometry

When endpoint-margin recording is enabled, Stage 7b also records continuous response margins over the same dose grid:

```text
GRADED_AGONIST_RECORD_ENDPOINT_MARGIN=true
GRADED_AGONIST_ENDPOINT_MARGIN_MAX_EXAMPLES=64
```

Per-run analysis is implemented by:

```text
studies/overtopping/analysis/stage09_graded_margin_mechanism_test.py
```

and writes under:

```text
<stage7 stats dir>/graded_agonist_intervention/margin_mechanism_test/
```

The primary continuous quantity is the divergence-token margin: the response margin at the first token where the natural and full-intervention completions diverge. Because the source activation is interpolated linearly in dose, the endpoint chord provides a direct affine-response reference for the downstream margin. The analysis reports:

- endpoint-chord fit and normalized error;
- concentration of total path variation into the largest dose step;
- monotonicity and excess path variation;
- alignment between behavioral crossing dose and continuous-margin crossing dose;
- endpoint-affine crossing predictions;
- transient interior flips that return to the endpoint-preserving state.

These diagnostics measure downstream non-affinity and event-like concentration; mathematical discontinuity and biological spike generation are outside their estimands.

Cross-setting aggregation is part of `stage08_graded_agonist_report.py`. Population-level event-localization and causal-strength summaries are generated by:

```text
studies/overtopping/analysis/stage10_rq3_spiking_story_figures.py
```

Population continuous-margin aggregation uses 11-dose trajectories that reproduce the configured endpoints.

## Part D — autoregressive temporal cutoff

The temporal-cutoff experiment tests whether the full singleton intervention must remain active throughout generation or whether its held-out effect is captured by a short early decode horizon. It supports both intervention phases. In output-only/decode-only settings, prompt prefill remains clean. In standard input+output settings, the Stage-7 singleton replacement is also active during prompt prefill. For cutoff `t`, the decode-time intervention is applied during the first `t` autoregressive transitions and removed for every later transition. Earlier intervention effects are allowed to persist through the model state/KV cache; only further direct intervention is stopped.

Per-run implementation and pipeline control:

```text
studies/overtopping/analysis/temporal_cutoff_intervention.py
RUN_TEMPORAL_CUTOFF_INTERVENTION=true
```

The pipeline also runs the complementary suffix-on schedule by default (`RUN_TEMPORAL_SUFFIX_INTERVENTION=true`): for an active-count `t`, the first `K-t` decode transitions have no direct decode-time intervention and the full Stage-7 singleton intervention is applied only on the final `t` transitions. Standard input+output runs still retain the intervened prompt prefill; decode-only runs retain clean prefill. Prefix and suffix sweeps are written to separate `temporal_cutoff_intervention/` and `temporal_suffix_intervention/` directories under the phase-specific Stage-7 statistics path. Phase-specific Stage-7 paths keep input+output and output-only artifacts separate.

Optional controls include `TEMPORAL_CUTOFF_ACTIVE_STEPS`, `TEMPORAL_CUTOFF_MAX_UNITS_PER_DIRECTION`, `TEMPORAL_CUTOFF_MAX_POSITIVE_SUPPORT`, `FORCE_TEMPORAL_CUTOFF_INTERVENTION`, and `FORCE_TEMPORAL_SUFFIX_INTERVENTION`. By default every possible intervention-count `t` is evaluated for both prefix and suffix schedules.

The population reporter is:

```text
studies/overtopping/analysis/stage11_rq3_temporal_cutoff_story.py
```

When suffix artifacts exist, the reporter writes `temporal_suffix_population_rows.csv.gz`, `temporal_suffix_condition_curves.csv`, `temporal_suffix_condition_incremental_gain.csv`, `temporal_cross_sweep_peak_agreement.csv`, and paired EVENT-validation tables for both discovery/validation directions. `fig4s16_temporal_cross_sweep_event_validation.pdf` aggregates compatible temporal conditions from both output-only and input+output phases. Phase-specific analyses are stored under `by_phase/output_only/` and `by_phase/input_output/`; `fig4s16b_temporal_cross_sweep_event_validation_by_phase.pdf` and `temporal_cross_sweep_event_validation_by_phase.csv` contain the phase-stratified results.

The cross-sweep EVENT analysis defines EVENT as the absolute decode transition with the largest positive incremental causal effect in one schedule and evaluates the aligned transition in the complementary schedule. The configured direction defines EVENT from the suffix-only sweep and evaluates it on the prefix-only sweep; the reciprocal direction is computed separately. EVENT selection and validation therefore use different intervention schedules. The condition-level inferential contrast is validation-sweep gain at EVENT minus the mean of the available adjacent transition gains.

Prefix-only T50/T80 and peak-share summaries are also computed. T50/T80 are derived from the same prefix temporal trajectory and therefore are not used as independent EVENT definitions for spiking inference. Peak-share temporal concentration is descriptive only; it is not assigned a uniform-null p-value because taking the within-condition maximum makes the naive 1/K comparison invalid by construction.

## Secondary subtype analysis — dominant-secondary preemption

Stage 8 implements a dominant-secondary preemption assay and aggregate reporter:

```text
pipeline/stage08_validate_interactions.py
studies/overtopping/analysis/stage09_preemption_report.py
```

This assay asks whether a secondary candidate's binary marginal contribution differs according to an independently fitted endogenous event associated with a frozen dominant candidate. Pair identity and order are frozen from discovery data.

Outputs are written under:

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/preemption/
```

This assay is separate from the graded threshold-crossing analysis and the RQ2 singleton-versus-joint decomposition.

## Result semantics

RQ3 supports thresholded causal integration when a fixed candidate shows a structured graded crossing on susceptible examples while same-channel non-flip examples remain predominantly non-crossing.

Possible interaction regimes are interpreted jointly with RQ2:

- high joint preservation and low suppression are compatible with a saturating high-leverage regime;
- singleton-reachable suppression indicates antagonistic or masking interaction under the full set;
- coalition-only effects indicate cooperative or coalition-dependent integration;
- positive preemption evidence, when present, supports a stronger first-sufficient subtype.

Endogenous scalar threshold visibility is reported separately from intervention-level graded crossing.

## Generated files

Rendered output directory:

```text
results/paper/figures/04_rq3_spiking_cut/
```

Primary files:

```text
fig4a_candidate_control_spiking_cut_summary.pdf
fig4b_threshold_shape_model_comparison_by_direction.pdf
fig4c_graded_agonist_dose_response.pdf
fig4d_graded_margin_affine_null.pdf
fig4e_population_event_and_strength.pdf
```

Additional files include:

```text
fig4s1_threshold_testability_by_condition.pdf
fig4s2_strength_matched_thresholdability.pdf
fig4s3_nested_tecs_lower_bound_ecdf.pdf
fig4s4_threshold_tail_response_by_direction.pdf
fig4s5_graded_agonist_single_crossing.pdf
fig4s6_graded_margin_condition_diagnostics.pdf
fig4s7_graded_margin_condition_heatmap.pdf
fig4s8_graded_behavior_competence_reach_io.pdf
fig4s8_graded_behavior_competence_reach_out.pdf
fig4s9_graded_margin_competence_reach_io.pdf
fig4s9_graded_margin_competence_reach_out.pdf
fig4s10_population_event_localization.pdf
fig4s11_strength_concentration_paired.pdf
fig4s12_arithmetic_competence_concentration.pdf
fig4s13_affine_null_transient_events.pdf
fig4s14_temporal_cutoff_spiking.pdf
fig4s15_temporal_cutoff_capture_profile.pdf
```

The preemption report is stored with the RQ3 analysis outputs.
