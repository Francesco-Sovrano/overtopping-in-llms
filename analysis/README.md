# Analysis, validation, and final paper outputs

The `analysis/` directory contains singleton summaries for selectable evaluation splits, simultaneous interaction validation, primary test-split normalization, correlations, matched-null statistics, catalogue visualization, paper tables, paper figures, spiking diagnostics, and compact-export utilities.

Raw model experiment artifacts live under `data/`. Final aggregate statistics and publication products live under root `results/`.

Analysis entry points should be launched as modules from the repository root, for example `python3 -m analysis.26_validate_interactions --help`. The analysis package does not mutate `sys.path`.

## Final-results entry point

From repository root:

```bash
./generate_results.sh
```

This command does not run the experiment catalogue. It reads existing experiment artifacts and invokes `analysis/29_generate_final_results.py`.

Environment variables:

```bash
PRIMARY_PROFILE=iclr-28               # default
DATA_ROOT=<repo>/data                 # default
ALLOW_INCOMPLETE_NEW_METRICS=false    # default; final paper outputs require exact new metrics
```

Example:

```bash
DATA_ROOT=/path/to/data PRIMARY_PROFILE=legacy-27 ./generate_results.sh
```

The root shell script always writes final outputs under `<repo>/results`. Before manuscript generation it writes `results/required_metrics_audit/required_metrics_audit.csv` and `.json`. By default the command fails if any primary setting lacks exact discovery-frozen TOC/OCC or simultaneous-intervention outputs. Set `ALLOW_INCOMPLETE_NEW_METRICS=1` only for an explicitly partial legacy report.

The Python orchestrator can be called directly:

```bash
python3 -m analysis.29_generate_final_results \
  --data-root data \
  --results-root results \
  --primary-profile iclr-28
```

Optional flags:

```text
--catalogue-json <configured_experiments.json>
--skip-paper-figures
--skip-spiking-report
```

If `--catalogue-json` exists, catalogue-scoped summaries are refreshed before manuscript outputs are generated.

## Canonical results tree

```text
results/
├── configured_experiments.json
├── pipeline_failures.json
├── catalogue/
├── paper_tables/
├── primary_metrics/
├── manuscript/
├── paper_figures/
├── overtopping_spiking_report/
└── final_results_manifest.json
```

Not every file is present in every workflow: `configured_experiments.json` and `pipeline_failures.json` are owned by the experiment driver, while the other subdirectories are produced by the analysis pipeline.

## Singleton-set statistics

The numerical implementation is `lib/heldout_set_metrics.py`; stage 7 writes its outputs into each stats directory.

For singleton flip sets `F_j` on the selected evaluation split, the reported set statistics include:

```text
s_j
s_(1)
U(J)
TOC_m(J)
R_ov(J)
N_eff(J)
OCC_0(J)
OCC_1(J)
N_t
|J|
```

`H_m` is selected by the discovery-frozen ranking. All singleton probabilities use the same complete-case universe within that evaluation split. Ratios are not clipped, and undefined denominators carry explicit statuses.

Files include:

```text
singleton_channel_metrics.csv
singleton_set_metrics.csv
singleton_set_metrics.json
frozen_topm_metrics.csv
singleton_threshold_counts.csv
frozen_topm_toc.pdf
frozen_topm_toc.png
```

## Simultaneous interaction validation

`26_validate_interactions.py` performs the model-backed simultaneous interventions required for `E(J)` and GCCR.

The pipeline normally supplies:

```text
--input_data_dir <stage-5 neural_circuits directory>
--candidate_flip_stats_path <stage-7 flip_stats_by_neuron.csv>
--singleton_scores_path <stage-7 scores.csv>
--frozen_ranking_path <stage-7 frozen_candidate_ranking.csv>
--out_dir <stats-dir>/interaction_validation
--task_module <task module>
--ai_model <model>
--intervention <replacement baseline>
--m_values 1,2,3,5
--null_draws 100
--denominator_epsilon 1e-12
--evaluation_split test|train|all   # default test
[--decode_only]
```

The validator resolves the stage-5 `dataset_info.json` and `manifest.json`, reconstructs the prespecified channel populations, and verifies the input fingerprints.

### Candidate statistics

It evaluates:

```text
E(J)
E_l(C_l)
E_l(C_l \ J_{l,m})
Delta_l = E_l(C_l) - E_l(C_l \ J_{l,m})
GCCR_m = sum_l Delta_l / sum_l E_l(C_l)
```

The denominator includes every prespecified population. Empty `J_{l,m}` subsets contribute `Delta_l=0` while leaving their `E_l(C_l)` term in the denominator.

### Matched random controls

Random candidate sets are drawn within exact intervention strata so that candidate and control sets match:

```text
transformer layer
computational locus
channel type
intervention phase
replacement baseline
per-layer cardinality
```

Candidate channels are excluded from the null pool. Every null `E(J)` and GCCR value comes from simultaneous intervention; singleton unions are not used as null substitutes.

For each metric the summary reports:

```text
candidate
median_null
Delta
P
p_MC
null_draws_requested
null_draws_finite
status
```

where

```text
Delta = candidate - median(null)
P = (1 + count(null <= candidate)) / (B + 1)
p_MC = (1 + count(null >= candidate)) / (B + 1)
```

### Interaction output files

```text
interaction_configuration.json
layer_populations.csv
matched_control_strata.csv
frozen_candidate_ranking.csv
layer_interaction_effects.csv
gccr_metrics.csv
matched_random_set_membership.csv
matched_null_draws.csv
interaction_validation_summary.csv
interaction_validation_summary.json
interaction_validation_summary.md
interaction_validation_table.tex
E_J_matched_null.pdf
E_J_matched_null.png
```

`interaction_configuration.json` uses schema `interaction-validation-v2` and records model/task identity, baseline, phase, `m` values, null-draw count, denominator epsilon, seed, and fingerprints.

## Primary matrix profiles

`primary_matrix.py` validates two named setting matrices:

| Profile | Expected rows | Qwen2-1.5B I+O NLI |
|---|---:|---|
| `iclr-28` | 28 | required exactly once |
| `legacy-27` | 27 | absent |

The 27-setting profile accepts either an already-normalized 27-row input without the Qwen row or a 28-row input containing exactly one matching row, which it records in the exclusion audit.

Normalization outputs are written with a caller-provided stem, normally:

```text
primary_table_normalized.csv
primary_table_excluded.csv
primary_table_profile.json
```

A count or row-identity mismatch is an error.

## Paper tables

`compute_overtopping_latex_tables.py` builds the primary table from a data directory or ZIP:

```bash
python3 -m analysis.compute_overtopping_latex_tables \
  --results data \
  --primary-profile iclr-28
```

Default output:

```text
results/paper_tables/
```

Files:

```text
primary_table.csv
primary_table_normalized.csv
primary_table_excluded.csv
primary_table_profile.json
table1_representative.tex
table8_primary.tex
occ_warnings.txt
```

The table builder prefers exact singleton/interaction sidecars when available. If exact OCC denominators are unavailable in an aggregate-only input, `occ_warnings.txt` records the affected rows instead of silently presenting the value as an exact conditional probability.

The optional `--empirical-fsm-chance` flag uses a sampled state-count FSM chance baseline for audit/debug use. The default uses the fixed manuscript baseline `mean(1/3, 1/4, 1/5, 1/6)`.

## Primary metric analysis

`21_analyze_primary_metrics.py primary` consumes the primary table:

```bash
python3 -m analysis.21_analyze_primary_metrics primary \
  --primary_table results/paper_tables/primary_table.csv \
  --data_root data \
  --out_dir results/primary_metrics
```

It writes:

```text
primary_table_augmented.csv
primary_correlations.csv
primary_metrics.json
primary_metrics.md
directional_coverage_all_settings.csv
directional_coverage_all_settings.md
```

The analysis includes overall, phase-stratified, task-stratified, and partial correlations where the required columns are available. It adds overlap compression/singleton mass and direction-specific union coverage from the underlying stats directories.

Direction-specific values preserve the denominator semantics present in the source artifacts. They are not converted into a different conditional rate silently.

## Manuscript metric outputs

`27_generate_manuscript_outputs.py` consumes the primary table and a required profile:

```bash
python3 -m analysis.27_generate_manuscript_outputs \
  --primary_table results/paper_tables/primary_table.csv \
  --data_root data \
  --out_dir results/manuscript \
  --primary_profile iclr-28
```

It augments each setting with singleton and interaction sidecars and writes:

```text
primary_table_normalized.csv
primary_table_excluded.csv
primary_table_profile.json
manuscript_metrics.csv
manuscript_metrics.json
manuscript_metrics.tex
matched_null_metrics.csv
matched_null_metrics.json
matched_null_metrics.tex
metric_statuses.csv
metric_statuses.tex
score_vs_U_J.{pdf,png}
score_vs_TOC_1.{pdf,png}
score_vs_R_ov.{pdf,png}
score_vs_N_eff.{pdf,png}
score_vs_E_J.{pdf,png}
score_vs_GCCR_1.{pdf,png}
```

The long-form matched-null table includes every available `m`, not only `m=1`, together with candidate value, null median, `Delta`, `P`, `p_MC`, requested/finite draw counts, and status.

## Catalogue-scoped visualization

`28_visualize_experiment_results.py` reads an explicit `configured_experiments.json`. It never scans the data tree for arbitrary runs.

```bash
python3 -m analysis.28_visualize_experiment_results \
  --catalogue_json results/configured_experiments.json \
  --data_root data \
  --out_dir results/catalogue
```

It writes:

```text
completed_experiments.csv
completed_experiments.json
summary_by_task.csv
summary_by_model.csv
summary_by_intervention.csv
summary_by_phase.csv
summary_by_suite.csv
score_vs_<metric>.{pdf,png}
```

Plots are generated for available metrics among `U_J`, `TOC_1`, `R_ov`, `N_eff`, `E_J`, and `GCCR_1`.

## Paper figures

`make_competence_vs_overtopping_paper_figures.py` reads experiment results from `data/` and writes final figures under `results/paper_figures/` when called by the final-results orchestrator.

The orchestrator creates:

```text
fig_competence_vs_coverage.pdf
```

plus the script's publication summary templates selected by `--paper-figures all`. PNG companions are produced where requested by the figure script.

The figure script has extensive filtering, labeling, trend, size, and layout controls; use:

```bash
python3 -m analysis.make_competence_vs_overtopping_paper_figures --help
```

for the complete plotting interface.

## Overtopping/spiking report

`generate_overtopping_spiking_report.py` summarizes `spiking_diagnostics` trees:

```bash
python3 -m analysis.generate_overtopping_spiking_report \
  --root data \
  --out results/overtopping_spiking_report
```

It can also accept a ZIP with `--zip` instead of `--root`.

`29_generate_final_results.py` checks whether any `spiking_diagnostics` directory exists. If none is present, it writes:

```text
results/overtopping_spiking_report/report_status.json
```

with `status=not_available`. `--skip-spiking-report` writes the same status file with `status=skipped`.

## Final-results orchestrator

`29_generate_final_results.py` runs the final reporting sequence:

1. refresh catalogue visualization if `--catalogue-json` exists;
2. build `results/paper_tables/`;
3. run primary metric analysis into `results/primary_metrics/`;
4. build manuscript metric tables/plots into `results/manuscript/`;
5. build paper figures unless skipped;
6. build or mark availability of the spiking report;
7. write `results/final_results_manifest.json`.

All paths in the manifest are absolute resolved paths for the selected data/results roots.

## Other analysis utilities

- `12_threshold_event_diagnostics.py`: strict-test threshold/channel diagnostics and matched controls.
- `22_compute_group_dominance.py`: simultaneous group-dominance calculations on a fixed candidate set.
- `23_compute_survey_dominance.py`: batch application of group-dominance calculations over a primary table. It accepts `--evaluation_split test|train|all`, defaulting to `test`.
- `24_run_primary_holdout_analysis.py`: primary-table re-estimation support. It accepts `--evaluation_split test|train|all`, defaulting to `test`; the paired manuscript holdout audit is produced only for the `test` split.
- `25_rebuild_directional_stats.py`: reconstruct direction-specific summaries from materialized stage-7 score tables. It accepts `--evaluation_split test|train|all`, defaulting to `test`.
- `8_compare_experiments.py`, `9_compare_models.py`: detailed experiment/model comparison utilities.
- `10_compute_threshold_sweep_stats.py`: threshold-sweep statistics; default output is `results/paper_tables/stats/`.

These utilities are not required to understand the standard root launchers; they are available for targeted analyses.

## Compact result export

Create a filtered result tree:

```bash
python3 -m analysis.clean_results_for_export data data_filtered_results
```

Options:

```text
--dry-run
--force
--verbose
--manifest <json-path>
--reference-zip <zip-path>
```

The embedded export schema retains final/compact artifacts such as dataset statistics, singleton tables, frozen rankings, interaction summaries, rule metrics, tables, and plots. It excludes pickle caches, large per-example stage-7 `scores.csv`, circuit-input caches, and generated junk.

This distinction matters:

- when a compact export already contains the new singleton sidecars, discovery-frozen TOC/OCC tables can be regenerated exactly;
- when an old compact export contains only `flip_stats_global.json` and `flip_stats_by_neuron.csv`, only `|J|`, `U(J)`, `s_(1)`, `N_t`, `R_ov`, and `N_eff` are exactly identifiable; `TOC_m` and `OCC_b` are explicitly unavailable;
- new simultaneous `E(J)`, GCCR, and matched-null draws cannot be generated without runtime scores, stage-5 circuit populations, model weights, and replacement-baseline inputs;
- no analysis path substitutes held-out `Top/U` for discovery-frozen `TOC_1`, `C2I/raw` for exact `OCC_1`, or singleton unions for simultaneous interventions;
- a compact export is therefore an analysis/reporting artifact, not a complete resumable experiment tree.

## Required-metric audit

`30_audit_required_metrics.py` classifies every primary setting before strict final-paper generation. It records whether the requested quantities are already exact, exactly recoverable from legacy aggregates, backfillable from a full runtime cache, or unavailable without model execution.

```bash
python3 -m analysis.30_audit_required_metrics \
  --primary-table results/paper_tables/primary_table.csv \
  --data-root data \
  --primary-profile iclr-28 \
  --out-dir results/required_metrics_audit \
  --require-complete
```

The audit distinguishes three levels:

1. **Legacy aggregate exact**: `|J|`, `U(J)`, `s_(1)`, `N_t`, `R_ov`, and `N_eff` can be reconstructed from `flip_stats_global.json` and `flip_stats_by_neuron.csv`.
2. **Singleton-event backfill**: `TOC_m` and `OCC_b` can be regenerated without new singleton ablations when `scores.csv` still contains per-example `flip_*` columns and stage-6 discovery outputs can reconstruct the frozen ranking.
3. **Model-backed interaction backfill**: `E(J)`, GCCR, and matched nulls require genuine simultaneous interventions with the stage-5 population and model/replacement-baseline runtime inputs.

A missing quantity is never replaced by a mathematically different proxy.

## Status values

Analysis files use explicit statuses rather than substituting zero for undefined quantities. Common values include:

```text
ok
undefined_near_zero_denominator
undefined_nonfinite
undefined_zero_denominator
undefined_candidate
undefined_no_null_draws
undefined_nonfinite_null_draws
missing
```

Do not clip negative or above-one finite interaction ratios when consuming the CSV/JSON outputs.
