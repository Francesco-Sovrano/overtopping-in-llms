# Analysis, simultaneous validation, and final paper outputs

The `analysis/` package contains model-free singleton statistics, model-backed simultaneous-set validation, primary-matrix normalization, required-metric auditing, manuscript tables, visualizations, spiking reports, and compact result export.

Run modules from `<repo>/code`:

```bash
cd code
python3 -m analysis.29_generate_final_results --help
```

For normal use, the repository-root command is:

```bash
./generate_results.sh
```

## 1. Core metric families

The analysis separates three causal objects that must not be conflated:

1. **singleton-event union metrics** derived from per-example `F_j` events;
2. **simultaneous full-set effect** `E(J)`;
3. **paired conditional marginal contribution** under matched background interventions.

No simultaneous-set quantity is replaced with a union of singleton flips.

## 2. Singleton-set statistics

Implemented in `lib/heldout_set_metrics.py` and written by stage 7.

For frozen candidates `J`:

```text
s_j = P(F_j)
s_(1) = max_j s_j
U(A) = P(union_{j in A} F_j)
TOC_m = U(H_m)/U(J)
R_ov = 1 - U(J)/sum_j s_j
N_eff = (sum_j s_j)^2/sum_j s_j^2
OCC_b = P(union_{j in J} F_j | B(x)=b)
N_t = |{j : s_j >= t}|
```

Properties:

- one common complete-case evaluation universe is used for all singleton probabilities;
- `H_m` is ordered only by the supplied discovery-frozen ranking;
- no metric is clipped;
- near-zero ratio denominators get explicit undefined statuses;
- default `N_t` thresholds are `0.01, 0.05, 0.10, 0.20, 0.30`.

Current sidecar schema:

```text
heldout-set-metrics-v2
```

## 3. Simultaneous full-set validation

Implemented in `26_validate_interactions.py`.

`E(A)` is the evaluation-set flip rate when all channels in set `A` are suppressed simultaneously.

The candidate full-set effect is:

```text
E(J)
```

For matched noncandidate sets `K_b`, the direct null summary is:

```text
Delta = E(J) - median_b E(K_b)
P     = (1 + sum_b 1{E(K_b) <= E(J)})/(B+1)
p_MC  = (1 + sum_b 1{E(K_b) >= E(J)})/(B+1)
```

The pipeline computes this direct validation whenever `RUN_INTERACTION_VALIDATION=true`. CMC is a separate optional extension. `RUN_CMC=false` skips the conditional-background computations without disabling `E(J)` or `E(K_b)`. Direct invocation uses `--skip_cmc` for the same behavior.

All candidate/null effects are real simultaneous interventions.

## 4. Paired conditional marginal contribution

For every draw `b`, construct:

- matched noncandidate set `K_b`;
- matched background set `S_b`, disjoint from `J` and `K_b`.

Evaluate:

```text
E(S_b)
E(S_b union J)
E(S_b union K_b)
```

Then:

```text
M_b(J)   = E(S_b union J)   - E(S_b)
M_b(K_b) = E(S_b union K_b) - E(S_b)
D_b      = M_b(J)           - M_b(K_b)
```

For each background multiplier the output row contains:

```text
candidate         mean M_b(J)
candidate_median  median M_b(J)
null_mean         mean M_b(K_b)
median_null       median M_b(K_b)
Delta             mean D_b
Delta_median      median D_b
P                 (1 + # {D_b >= 0})/(B+1)
p_MC              (1 + # {D_b <= 0})/(B+1)
paired_win_rate   mean 1{D_b > 0}
```

This is a paired design: candidate and null use the identical `S_b` for each draw.

### Background multipliers

`--background_multipliers` accepts comma-separated nonnegative integers. For multiplier `q`, the background uses `q` times the candidate count in every exact matching stratum.

Default:

```text
1
```

The default manuscript field is therefore:

```text
CMC_1x
```

### Matching

The structural matching strata are:

- transformer layer;
- computational locus;
- channel type;
- per-stratum cardinality.

The complete validation run also fixes:

- intervention phase;
- replacement baseline;
- evaluation split;
- evaluation examples.

Candidate, null, and background sets are evaluated on the same evaluation-row fingerprint.

### Null draws and seed

Direct module defaults:

```text
--null_draws 100
--seed 42 when no compatible cached seed is available
```

The root standard launcher defaults `INTERACTION_NULL_DRAWS` to `30` and respects an explicit override.

For finer Monte-Carlo resolution set a larger value, e.g. `999`.

## 5. Interaction validation outputs

The stats directory always receives the direct simultaneous-validation outputs below. Files whose names begin with `conditional_` are produced only when CMC is enabled.

The stats directory receives:

```text
interaction_validation/
├── interaction_configuration.json
├── matched_control_strata.csv
├── matched_random_set_membership.csv
├── matched_null_draws.csv
├── conditional_background_membership.csv      # CMC only
├── conditional_marginal_draws.csv             # CMC only
├── conditional_marginal_summary.csv           # CMC only
├── interaction_validation_summary.csv
├── interaction_validation_summary.json
├── interaction_validation_summary.md
├── interaction_validation_table.tex
├── E_J_matched_null.pdf
└── conditional_marginal_<q>x_paired_delta.pdf # CMC only
```

Current schema:

```text
conditional-marginal-validation-v1
```

## 6. Cache reuse

A compatible current interaction cache is reused without loading the model.

Compatibility is determined from semantic configuration and content fingerprints, including the candidate set, evaluation-row fingerprint, intervention configuration, matching/background design, seed, and relevant inputs.

The validator can also recognize an `interaction-validation-v3` cache for the purpose of reusing exact simultaneous `E(J)`, compatible direct-null `E(K_b)` values, and validated matched-set membership. The conditional background/context interventions are computed only if they are missing.

GCCR-formatted outputs are not part of the current metric set and are not read into manuscript fields.

## 7. Required-metric audit

`30_audit_required_metrics.py` determines, row by row, whether the paper-required metrics are:

- already exact;
- backfillable from materialized singleton/runtime artifacts;
- unavailable without model-backed intervention work.

It never substitutes singleton unions for `E(J)` or conditional marginals.

The final-results orchestrator always writes the audit under:

```text
results/required_metrics_audit/
```

With `--require-complete-new-metrics`, any primary row missing an exact required metric causes the final-results command to fail after writing the audit. When `RUN_CMC=false` is used through the standard launcher, the completeness audit does not require CMC, but still requires simultaneous `E(J)` and its matched-null validation. Direct final-results usage can select the same policy with `--skip-cmc-requirement`.

## 8. Primary profiles

`primary_matrix.py` defines:

```text
iclr-28
legacy-27
```

The identity difference is exactly one row:

```text
NLI | Qwen2-1.5B | I+O
```

`iclr-28` requires it; `legacy-27` excludes it. Invalid counts/identities raise errors.

Normalization writes:

```text
primary_table_normalized.csv
primary_table_excluded.csv
primary_table_profile.json
```

## 9. Final-results orchestrator

`29_generate_final_results.py` writes all final outputs below a supplied `results-root`.

It runs, in order:

1. optional catalogue visualization if `configured_experiments.json` is supplied;
2. paper-table generation;
3. required-metric audit;
4. primary metric analysis;
5. manuscript metric/table generation;
6. paper figures unless skipped;
7. spiking report, or an explicit `not_available` status if diagnostics are absent;
8. `final_results_manifest.json`.

Root command:

```bash
./generate_results.sh
```

The root launcher defaults to strict exact-metric completeness. Set:

```bash
ALLOW_INCOMPLETE_NEW_METRICS=1 ./generate_results.sh
```

to permit partial output with missing metrics explicitly marked unavailable.

## 10. Final results tree

```text
results/
├── catalogue/
├── paper_tables/
├── required_metrics_audit/
├── primary_metrics/
├── manuscript/
├── paper_figures/
├── overtopping_spiking_report/
└── final_results_manifest.json
```

## 11. Manuscript outputs

`27_generate_manuscript_outputs.py` augments primary rows with exact sidecar statistics and writes CSV/JSON/LaTeX outputs.

Core fields include:

```text
J
U_J
s_1
TOC_1
R_ov
N_eff
OCC_0
OCC_1
N_t_0.05
N_t_0.1
E_J
CMC_1x
```

Available additional background multipliers are emitted as `CMC_<q>x` fields, together with candidate/null summaries, paired deltas, plus-one probabilities, draw counts, and status fields.

## 12. Catalogue visualization

`28_visualize_experiment_results.py` is catalogue-scoped: it reads an explicit `configured_experiments.json` rather than discovering arbitrary directories. This avoids mixing unrelated runs into standard aggregate plots.

## 13. Paper figures

`make_competence_vs_overtopping_paper_figures.py` reads experiment artifacts from `data/` and writes publication figures under the requested output directory, normally:

```text
results/paper_figures/
```

## 14. Spiking report

`generate_overtopping_spiking_report.py` aggregates `spiking_diagnostics` directories. The final-results orchestrator writes:

```text
results/overtopping_spiking_report/report_status.json
```

with `not_available` when the selected data root contains no spiking diagnostics.

## 15. Compact export

`clean_results_for_export.py` creates a filtered, shareable result tree:

```bash
cd code
python3 -m analysis.clean_results_for_export ../data ../data_filtered_results
```

The filtered tree is suitable for model-free result inspection and regeneration of metrics that are fully represented by retained summaries. New simultaneous interventions require the full runtime artifacts and accessible model.

## 16. Other utilities

- `10_compute_threshold_sweep_stats.py` — statistics for threshold sweeps.
- `12_threshold_event_diagnostics.py` — threshold-event/spiking diagnostics.
- `21_analyze_primary_metrics.py` — primary aggregate statistics and correlations.
- `22_compute_group_dominance.py` — generic simultaneous group-effect utility retained for focused analyses.
- `23_compute_survey_dominance.py` — survey-wide group-effect utility.
- `24_run_primary_holdout_analysis.py` — primary-row re-estimation helper.
- `25_rebuild_directional_stats.py` — directional-stat reconstruction from materialized score tables.
- `compute_overtopping_latex_tables.py` — primary/paper table construction.
- `recover_primary_directional.py` — directional metadata recovery helper.
- `task_metrics.py` — task-score and chance-baseline normalization.

These utilities are not independent experiment catalogues.
