# Paper-facing nonlinear graded-response experiment

The graded intervention already moves the selected source activation linearly in dose. With endpoint-margin recording enabled, the pipeline now tests whether the *downstream* response is also approximately affine, or instead shows concentrated nonlinear threshold-like geometry.

## Default execution

`code/pipeline/run_pipeline.sh` now enables endpoint-margin recording by default:

```bash
GRADED_AGONIST_RECORD_ENDPOINT_MARGIN=true
GRADED_AGONIST_ENDPOINT_MARGIN_MAX_EXAMPLES=64
```

The existing graded settings still control the number of units/examples and the dose grid. For a paper run, use the full 11-point dose grid unless compute requires otherwise:

```bash
export GRADED_AGONIST_DOSES="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"
export GRADED_AGONIST_MAX_UNITS_PER_DIRECTION=16
export GRADED_AGONIST_MAX_POSITIVE_SUPPORT=256
export GRADED_AGONIST_NEGATIVE_SUPPORT=true
export GRADED_AGONIST_RECORD_ENDPOINT_MARGIN=true
export GRADED_AGONIST_ENDPOINT_MARGIN_MAX_EXAMPLES=64
```

## Per-run outputs

After `graded_agonist_intervention.py`, the pipeline automatically runs `stage09_graded_margin_mechanism_test.py` and writes:

- `margin_mechanism_test/graded_margin_metric_tests.csv`: one row per example and margin type.
- `margin_mechanism_test/graded_margin_direction_summary.csv`: direction-specific numerical summary.
- `margin_mechanism_test/graded_margin_mechanism_summary.json`: machine-readable result.
- `margin_mechanism_test/graded_margin_affine_null.pdf`: normalized measured margin trajectories versus the endpoint-affine null.
- `margin_mechanism_test/graded_margin_crossing_alignment.pdf`: behavioral versus measured margin crossing dose.
- `margin_mechanism_test/graded_margin_paper_interpretation.md`: conservative manuscript-ready interpretation.
- `margin_mechanism_test/graded_margin_paper_interpretation.tex`: LaTeX version.

The graded intervention also records both `first_flip_dose` and the distinct `first_persistent_crossing_dose`.

## New quantities

For each endpoint-reproduced known-flip trajectory, the primary divergence-token margin analysis reports:

1. **Endpoint-chord R2** and normalized RMSE. The null is the straight line joining the measured dose-0 and dose-1 margins. Low R2/high normalized error means the downstream margin is not approximately affine in dose.
2. **Largest-step share of path variation**: `max |Delta m| / sum |Delta m|`. A companion concentration ratio divides this by the uniform-per-step share `1/(K-1)`, so it is comparable across dose-grid sizes.
3. **Behavioral/margin crossing alignment**: exact-grid alignment, within-one-grid-step alignment, crossing lag, and within-condition Spearman association.
4. **Affine crossing prediction**: crossing implied by the endpoint chord, compared with the observed behavioral crossing and a leave-one-out median crossing-dose baseline.
5. **Path excess variation and monotonicity**: distinguish a monotone saturating response from overshoot/reversal.

The total-response and mean-response log-probability margins remain secondary diagnostics. The divergence-token margin is primary because it localizes the first branch separating the two endpoint responses.

## Aggregate paper outputs

`stage08_graded_agonist_report.py` now aggregates the new diagnostics at the run/baseline/direction **condition** level before summarizing across conditions. `generate_final_results.py` automatically creates:

- `graded_margin_all_examples.csv`
- `graded_margin_by_condition.csv`
- `graded_margin_by_direction.csv`
- `graded_margin_affine_null.pdf`
- `graded_margin_condition_diagnostics.pdf`
- `graded_margin_paper_interpretation.md/.tex`

and paper-facing figures:

- `fig4d_graded_margin_affine_null.pdf`
- `fig4s6_graded_margin_condition_diagnostics.pdf`

## Interpretation contract

The generated text deliberately does **not** say that nonlinearity proves a literal spike. It states:

- the source intervention is linear by construction;
- low affine fit therefore measures downstream nonlinearity;
- crossing alignment establishes that the measured margin transition is behaviorally coupled;
- concentration quantifies event-like response shape;
- these observations challenge the simple affine-margin-plus-binary-readout construction for the measured margin;
- a smooth but nonlinear network remains possible, so the supported claim is *spiking-like causal response geometry*, not unique identification of a discontinuous or biological spiking mechanism.

## Recommended paper presentation

The main text should show the aggregate normalized-margin-versus-affine-null figure and report, by direction, the condition-level median affine fit, normalized error, transition concentration, single-persistent-crossing rate, and crossing alignment. Keep whole-response log-probability margins and detailed per-condition diagnostics in the appendix.

A candidate-versus-matched-control graded-margin sweep would be a useful additional specificity experiment, but it is not silently conflated with the current candidate-only estimand. If added, controls should be evaluated with an explicitly matched intervention/reference protocol and analyzed as a separate comparison.
