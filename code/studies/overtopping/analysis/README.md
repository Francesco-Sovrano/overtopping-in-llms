# Overtopping analysis

This package converts persistent overtopping experiment artifacts into setting-level metrics, completeness audits, statistical summaries, machine-readable sidecars, and manuscript figures. Model-backed intervention execution is owned by the shared pipeline and the graded/threshold diagnostic modules.

## Population model

Every aggregate analysis starts from the configured 56-setting registry:

```text
31 final-snapshot task × model × phase cells
18 non-final Grammar/HANS-NLI/FSM Pythia checkpoint settings
 7 matched replacement-baseline repeats
--------------------------------------------------
56 configured settings
```

The RQ4 checkpoint trajectory view contains 24 Pythia-1B Grammar/HANS-NLI/FSM settings because it
also includes the six final/all-steps endpoints from the final-snapshot component.

A metric then applies its own scientific applicability and artifact-availability requirements. Missing derived artifacts remain explicit missing measurements for configured settings; they do not change the registry.

RQ1 uses all 56 settings and fits input+output and output-only associations separately. RQ2 analyzes replacement regimes separately. RQ3 starts from all 56 settings and reports metric-specific denominators. RQ4 uses the configured checkpoint trajectories and controlled poisoning trajectories.

## Main analysis modules

### Registry, completeness, and RQ1

```text
stage01_visualize_experiment_results.py       completed-experiment summaries
stage02_overtopping_latex_tables.py           configured 56-setting study table
stage03_audit_required_metrics.py             per-setting metric completeness audit
stage04_analyze_primary_metrics.py            setting-level aggregate statistics
stage05_generate_manuscript_outputs.py        manuscript tables and machine sidecars
stage06_competence_vs_overtopping_figures.py  RQ1 figures and checkpoint trajectories
```

The RQ1 competence definition is phase-specific. Input+output interventions use the raw behavioral score because the intervention spans prompt processing and answer production. Output-only interventions use the chance-corrected score because prompt/instruction processing precedes the intervention and finite-choice task success can include guessing. Correlations are computed separately by intervention phase.

### RQ2

```text
stage06_manuscript_story_figures.py           composition figures
stage09_composition_decomposition_report.py   singleton-union/full-set event decomposition
stage10_rq2_regime_report.py                  replacement-regime aggregate report
```

Mean-donor and mean-family replacement are not pooled. `mean-positional` is normalized to the mean-family reporting regime. The decomposition verifies, on a common complete-case population,

```text
E(J) - U(J) = P(coalition_only) - P(suppressed)
```

where `suppressed` denotes examples reachable by at least one singleton but not by the simultaneous set, and `coalition_only` denotes examples reached only by the simultaneous set.

### RQ3

```text
threshold_event_diagnostics.py                per-run threshold-event diagnostics
stage07_overtopping_spiking_report.py         aggregate candidate/control support statistics
stage08_threshold_shape_validation.py         nested held-out threshold-shape validation
graded_agonist_intervention.py                per-run graded causal intervention
temporal_cutoff_intervention.py               per-run autoregressive intervention-cutoff sweep
stage08_graded_agonist_report.py              cross-run graded aggregation
stage10_rq3_spiking_story_figures.py          population event-localization / strength visual story
stage11_rq3_temporal_cutoff_story.py          population temporal-cutoff visual story
stage09_preemption_report.py                  secondary dominant-secondary preemption analysis
```

`temporal_cutoff_intervention.py` is an output-only/decode-only causal timing experiment. Prompt prefill remains clean; the full frozen singleton intervention is applied for the first `t` autoregressive decode transitions and then removed for all later transitions. `stage11_rq3_temporal_cutoff_story.py` aggregates those runs at the run/direction level and reports cumulative effect capture, the discrete temporal gain profile, quartiles, condition-bootstrap 95% confidence intervals, T50/T80, a temporal-concentration test against a condition-specific uniform-in-time baseline, and an early-versus-late paired secondary test.

Graded and threshold analyses use only settings with their required compatible artifacts and report the resulting denominator. A zero-candidate setting remains a measured RQ1 observation with `U(J)=0`; set composition and candidate-level graded analyses are not applicable when no candidate set exists.

## Recovery utilities

```text
rebuild_directional_stats.py
rebuild_spiking_diagnostics.py
```

Recovery utilities operate on explicit configured settings and compatible persistent inputs. `rebuild_spiking_diagnostics --dry-run` validates provenance and prints planned commands without modifying experiment artifacts. Normal execution may materialize missing derived diagnostics under the setting's existing persistent path; it does not alter the registry or path construction.

## Output roots

```text
results/analysis/primary_matrix/
results/analysis/reproducibility/
results/analysis/rq2_composition/
results/analysis/rq3_threshold_event/
results/analysis/rq4_learning/
results/paper/figures/
results/paper/tables/
```

The standard reporting driver reads scientific inputs under `data/` and writes derived products under `results/`. It rejects a `--results-root` located inside `data/` or the repository `cache/` tree.

## References

- [Experiment design](../../../docs/experiments/overtopping.md)
- [Research questions](../../../docs/research-questions/README.md)
- [RQ1 prevalence](../../../docs/research-questions/rq1-prevalence.md)
- [RQ2 composition](../../../docs/research-questions/rq2-composition.md)
- [RQ3 thresholded integration](../../../docs/research-questions/rq3-threshold-event.md)
- [Analysis and reporting pipeline](../../../docs/reporting/analysis-pipeline.md)
- [Figure map](../../../docs/reporting/figures.md)
- [Interpretation](../../../docs/reporting/interpretation.md)
- [Operations](../../../docs/operations/README.md)

### Population spiking-story figures

`stage10_rq3_spiking_story_figures.py` runs after the graded-agonist aggregate and prepares the population event-localization and causal-strength inputs. `stage11_rq3_temporal_cutoff_story.py` then assembles the paper-facing Figure 4e after the temporal-cutoff capture profile is available. The continuous-margin panels use only paper-standard 11-dose, endpoint-reproduced divergence-margin trajectories:

- `fig4e_population_event_and_strength.pdf`: main-text triptych: (a) population event localization, (b) transition concentration across within-run/direction causal-strength tertiles, and (c) the temporal-cutoff capture profile also shown standalone as Fig. S15;
- `fig4s10_population_event_localization.pdf`: standalone event-localization profile with condition-level interquartile ranges;
- `fig4s11_strength_concentration_paired.pdf`: paired low/high-strength concentration within eligible run/direction groups;
- `fig4s12_arithmetic_competence_concentration.pdf`: Arithmetic output-only 1-to-0 competence/concentration relation;
- `fig4s13_affine_null_transient_events.pdf`: endpoint-preserving interior flip-and-return rates against the one-dimensional affine-margin null.

The event-alignment figure is descriptive. The causal-strength tertiles are formed within each run/direction before condition-level aggregation, and the paired figure exposes every eligible low-versus-high comparison. None of these figures identifies literal temporal spikes or a unique internal mechanism.
