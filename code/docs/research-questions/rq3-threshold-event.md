# RQ3 — spiking-like causal transition

## Scientific question

RQ3 asks whether an overtopping agonist behaves like a threshold-crossing causal contribution **on the examples where that agonist is already known to be causally decisive**.

For frozen agonist `j` in discovery direction `d`, define its held-out directional flip support:

```text
S_j^d = {x : x is in the source state for d and the full intervention on j flips B(x)}
```

The primary RQ3 experiment does **not** fit a scalar classifier to distinguish `S_j^d` from non-flipped examples. The full singleton intervention has already established which held-out examples belong to the agonist's causal support.

Instead, RQ3 continuously varies the strength of the same agonist intervention and asks whether the behavioral endpoint crosses once and remains across the boundary.

## Candidate identity and direction

Candidates are frozen Stage-7 agonists. Discovery direction is part of candidate identity:

```text
positive-baseline discovery -> 1→0
negative-baseline discovery -> 0→1
```

Candidate selection for the graded experiment uses frozen discovery provenance/rank. It never ranks agonists using their graded-response result.

## Primary experiment: graded agonist intervention

Implemented by:

```text
studies/overtopping/analysis/graded_agonist_intervention.py
```

For every selected agonist `j` and every held-out example `x` in `S_j^d`, the intervention is interpolated from the natural activation to the exact replacement used by the validated full singleton intervention:

```text
h_j(lambda, x) = (1 - lambda) h_j(x) + lambda h_replacement_j
lambda in [0, 1]
```

where:

```text
lambda = 0   natural forward pass
lambda = 1   historical full singleton intervention
```

The underlying ablation hook accepts `intervention_strength` in `[0,1]`; strength `1.0` preserves the previous full-intervention behavior exactly. For donor-based replacements, the graded runner reconstructs replacement statistics using the full Stage-7 evaluated unit population and the same Stage-7 train-prompt sampling rule, so the donor pool is not changed by the graded subset.

Default dose grid:

```text
0, 0.1, 0.2, ..., 0.9, 1
```

The experiment records the declared binary behavioral endpoint at every dose.

## Known-flip support

The positive experimental population is the agonist's own Stage-7 directional flip support on the configured evaluation split.

For a 1→0 agonist:

```text
source rows: baseline B(x) = 1
support:     flip_c2i_<unit>(x) = 1
```

For a 0→1 agonist:

```text
source rows: baseline B(x) = 0
support:     flip_i2c_<unit>(x) = 1
```

The experiment refuses direction-agnostic evaluation.

## Optional same-agonist negative support

`--same_agonist_negative_support` optionally evaluates the identical dose sweep on source-state examples for which the **same agonist** did not flip the behavior at the full intervention:

```text
N_j^d = source-state rows evaluated for j \ S_j^d
```

This is a within-agonist reference. It is not a random-neuron control and does not change agonist identity, layer, intervention baseline, or discovery direction.

The default pipeline leaves this comparison disabled. It can be enabled with:

```text
GRADED_AGONIST_NEGATIVE_SUPPORT=true
```

## Per-example transition quantities

For each agonist/example trajectory, the analysis reports:

```text
first_flip_dose
n_state_changes
persistent_after_first_flip
single_crossing
natural_state_matches_stage7_baseline
full_dose_flipped
full_dose_matches_stage7_support
```

`single_crossing` is true when the dose-response moves from the baseline behavioral state to the opposite state once and remains there for all larger tested doses.

The main spiking-cut evidence is therefore the graded causal transition itself, not held-out threshold-classification MCC.

## Unit and condition summaries

Per agonist/support population:

```text
natural_state_reproduction_rate
full_dose_support_reproduction_rate
full_dose_flip_rate
single_crossing_rate
median_first_flip_dose
mean_first_flip_dose
median_state_changes
```

Aggregate reporting first summarizes within agonist and then within run/baseline/direction conditions so examples and neurons are not treated as independent experimental replicates.

Implemented by:

```text
studies/overtopping/analysis/stage08_graded_agonist_report.py
```

## Removed primary endpoint

Nested held-out threshold `|MCC|` is no longer an RQ3 primary endpoint. The scalar-to-singleton-flip prediction task is not used to establish the spiking phenotype.

Legacy scalar/threshold diagnostics may still be generated for inspection or for backwards-compatible analyses, but `generate_final_results.py` does not promote them into the manuscript-facing Figure 4 and does not run the old nested threshold-shape validation as an RQ3 endpoint.

## Outputs

Per experimental run:

```text
<stage7 stats dir>/graded_agonist_intervention/
    graded_agonist_intervention.json
    graded_agonist_plan.csv
    graded_agonist_dose_rows.csv.gz
    graded_agonist_example_summary.csv
    graded_agonist_unit_summary.csv
    graded_agonist_dose_response.pdf
```

Aggregate RQ3 outputs:

```text
results/analysis/rq3_threshold_event/graded_agonist/
    graded_agonist_population_audit.csv
    graded_agonist_all_units.csv
    graded_agonist_by_condition.csv
    graded_agonist_dose_by_condition.csv
    graded_agonist_report_status.json
```

Paper-facing figures:

```text
results/paper/figures/04_rq3_spiking_cut/
    fig4a_graded_agonist_dose_response.pdf
    fig4b_graded_agonist_single_crossing.pdf
```

## Pipeline controls

The normal pipeline runs the graded experiment by default:

```text
RUN_GRADED_AGONIST_INTERVENTION=true
```

Useful controls:

```text
GRADED_AGONIST_DOSES
GRADED_AGONIST_MAX_UNITS_PER_DIRECTION
GRADED_AGONIST_MAX_POSITIVE_SUPPORT
GRADED_AGONIST_NEGATIVE_SUPPORT
GRADED_AGONIST_NEGATIVE_RATIO
GRADED_AGONIST_MAX_NEGATIVE_SUPPORT
GRADED_AGONIST_SEED
FORCE_GRADED_AGONIST_INTERVENTION
```

The default agonist cap is applied by frozen discovery rank, not graded-response performance.
