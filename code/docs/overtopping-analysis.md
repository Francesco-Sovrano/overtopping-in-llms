# Overtopping analysis

This page documents overtopping-specific aggregate analysis under `studies/overtopping/analysis/`. The package builds the fixed primary matrix, audits exact metrics, computes statistics, and generates paper-facing tables and figures. Cross-study orchestration is implemented separately by `python3 -m reporting.generate_final_results`, which invokes these stages and available poisoning aggregation modules.

The `analysis` package converts experiment artifacts under `data/` into validated
primary tables, statistical summaries, manuscript-ready files, and publication
figures. It also contains optional diagnostics for threshold sweeps, group dominance, held-out analyses, and checkpoint comparisons. Shared model-backed interaction validation is Pipeline Stage 08 rather than an `studies/overtopping/analysis/` module.

Run analysis modules from `code/` so `studies.overtopping.*` and `core.*` imports resolve from the repository import root. Use `python3 -m reporting.generate_final_results` for the cross-study final-output workflow; the repository-root `generate_results.sh` wrapper invokes it with repository defaults.

## Package layout

```text
studies/overtopping/analysis/
  stage01_visualize_experiment_results.py
  stage02_overtopping_latex_tables.py
  stage03_audit_required_metrics.py
  stage04_analyze_primary_metrics.py
  stage05_generate_manuscript_outputs.py
  stage06_competence_vs_overtopping_figures.py
  stage07_overtopping_spiking_report.py

  compare_experiments.py
  compare_models.py
  group_dominance.py
  primary_holdout_analysis.py
  rebuild_directional_stats.py
  survey_dominance.py
  threshold_event_diagnostics.py
  threshold_sweep_stats.py
  lib/
    catalog.py
    files.py
    primary_matrix.py
    progress.py
    rule_metrics.py
    task_metrics.py

```

The numbered scripts form the standard final-results sequence. Utilities without
a `stageNN_` prefix are independent analyses and are not implicitly run by the
final-results orchestrator.

## Inputs and path conventions

The default experiment-artifact root is `<repo>/data`. Standard task directories
include:

```text
data/
  arithmetic/
  bon_jailbreaking/
  grammar_acceptability/
  hans_nli/
  random_fsm/
```

Within a task/model directory, the analysis code expects the artifacts produced
by the pipeline, including `feature_report/dataset_stats.json`, singleton/rule
statistics under `rule_extraction_results/neuron_flip_rules/stats/`, and optional
interaction or spiking sidecars.

Final outputs are written under `<repo>/results` by default. Analysis code should
not modify the source experiment tree.

## Score orientation and competence

All task scores used by the analysis package have a **higher-is-better**
orientation. The shared overtopping-analysis implementation is `studies.overtopping.analysis.lib.task_metrics`.

For arithmetic, grammar, HANS NLI, and random FSM, the raw score is read from an
explicit correctness field such as `accuracy`, `score`, or `pct_is_correct`.
Rates may be stored either in `[0,1]` or as percentages in `[0,100]`.

For jailbreak experiments, the analysis score is **safe/refusal rate**. If a
result file stores a safety-oriented key (`safe_refusal_rate`, `refusal_rate`,
`safe_rate`, `not_jailbroken_rate`, and supported aliases), that rate is used
directly. If it stores an attack-oriented key (`jailbreak_rate`,
`attack_success_rate`, `asr`, `unsafe_rate`, and supported aliases), the score is
`1 - attack_rate`. Generic `score` or `accuracy` fields are intentionally not
used for jailbreak because their orientation is ambiguous.

For output-only finite-answer interventions, competence is chance-normalized as

```text
(raw_score - chance) / (1 - chance)
```

and bounded to `[0,1]`. Input+output finite-answer interventions use the raw task
score in the default `phase-specific` convention. Jailbreak uses safe/refusal
rate in every score mode and is not chance-normalized.

Default random-answer baselines are 0.5 for grammar and HANS NLI, zero for
arithmetic, and the mean of `1/3`, `1/4`, `1/5`, and `1/6` for random FSM unless
a result file provides an explicit chance field.

## Primary profile

Primary-table membership is explicit. `studies/overtopping/analysis/lib/primary_matrix.py` defines one supported profile, `iclr-28`, containing exactly 28 settings. The table must include exactly one Qwen2-1.5B input+output NLI row.

A profile is required when building final primary outputs. Normalization writes an audit of the validated table; scripts do not infer primary membership from arbitrary directories found under `data/`.

## Standard final-results workflow

The standard direct command is:

```bash
cd code
python3 -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile iclr-28 \
  --require-complete-new-metrics
```

Optional orchestrator flags are:

```text
--catalogue-json PATH          refresh catalogue plots from configured experiments
--skip-paper-figures          skip stage 6
--skip-spiking-report         skip stage 7 and write a skipped status file
--require-complete-new-metrics
--skip-cmc-requirement        do not require CMC in the completeness audit
```

The orchestrator writes `results/final_results_manifest.json` with the resolved
input/output roots and output subdirectories.

## Stage 1: catalogue visualization

`stage01_visualize_experiment_results.py` visualizes an explicit experiment
catalogue. It does not discover arbitrary runs.

```bash
cd code
python3 -m studies.overtopping.analysis.stage01_visualize_experiment_results \
  --catalogue_json ../results/configured_experiments.json \
  --data_root ../data \
  --out_dir ../results/experiment_catalogue
```

Use this stage for catalogue-level overview plots only. The primary manuscript
matrix is controlled separately by the selected primary profile.

## Stage 2: primary table and LaTeX tables

`stage02_overtopping_latex_tables.py` locates the experiment tree, reads the
canonical primary settings, resolves exact singleton/intervention metrics when
available, normalizes the selected primary profile, and writes:

```text
results/primary_analysis/tables/
  primary_table.csv
  primary_table_normalization_audit.json
  primary_table_excluded_rows.csv
  table1_representative.tex
  table8_primary.tex
  occ_warnings.txt
```

Run it directly with:

```bash
cd code
python3 -m studies.overtopping.analysis.stage02_overtopping_latex_tables \
  --results ../data \
  --out ../results/primary_analysis/tables \
  --primary-profile iclr-28
```

`--results` may point to an extracted result tree or a ZIP archive containing the
task directories. `--empirical-fsm-chance` is an audit/debug option; the standard
paper-table path uses the same fixed random-FSM chance convention as the figure
code.

OCC is reported only when its event-level denominator is available. Other directional ratios are kept under their own metric names and are never substituted for OCC.

## Stage 3: required-metric audit

`stage03_audit_required_metrics.py` checks every selected primary setting for the
artifacts required by the analysis contract. It reports availability/status for
exact singleton metrics, simultaneous-set evaluation, matched nulls, and CMC
when required.

```bash
cd code
python3 -m studies.overtopping.analysis.stage03_audit_required_metrics \
  --primary-table ../results/primary_analysis/tables/primary_table.csv \
  --data-root ../data \
  --primary-profile iclr-28 \
  --out-dir ../results/primary_analysis/metric_completeness_audit \
  --require-complete
```

Use `--skip-cmc-requirement` only when CMC is intentionally outside the required
analysis set. Without `--require-complete`, the audit records missing metrics but
does not fail the command.

## Stage 4: primary statistical analysis

`stage04_analyze_primary_metrics.py` has three subcommands.

Primary-matrix analysis:

```bash
python3 -m studies.overtopping.analysis.stage04_analyze_primary_metrics primary \
  --primary_table ../results/primary_analysis/tables/primary_table.csv \
  --data_root ../data \
  --out_dir ../results/primary_analysis/statistics
```

Useful strictness flags are `--require_overlap_compression` and
`--require_directional_coverage`. Bootstrap count and random seed are controlled
with `--bootstrap` and `--seed`.

Held-out comparison:

```bash
python3 -m studies.overtopping.analysis.stage04_analyze_primary_metrics holdout \
  --run LABEL=/path/to/stats_dir \
  --out_dir ../results/holdout
```

Repeat `--run`, `--reference`, and `--dom` as needed. Candidate-set mismatch and
provenance checks are strict unless their explicit override flags are supplied.

Critical-report generation combines a primary metrics JSON and held-out summary:

```bash
python3 -m studies.overtopping.analysis.stage04_analyze_primary_metrics critical-report \
  --primary_metrics_json PATH \
  --holdout_summary_json PATH \
  --out_dir OUT_DIR
```

## Stage 5: manuscript outputs

`stage05_generate_manuscript_outputs.py` augments the normalized primary table
with exact sidecar statistics and writes manuscript-facing CSV, JSON, Markdown,
and LaTeX material.

```bash
python3 -m studies.overtopping.analysis.stage05_generate_manuscript_outputs \
  --primary_table ../results/primary_analysis/tables/primary_table.csv \
  --data_root ../data \
  --out_dir ../results/manuscript/tables_and_macros \
  --primary_profile iclr-28
```

The profile is mandatory so manuscript outputs cannot silently mix primary
matrices.

## Stage 6: competence-versus-overtopping figures

`stage06_competence_vs_overtopping_figures.py` discovers compatible pipeline
outputs and generates the competence/coverage figure plus optional paper summary
figures.

Standard invocation:

```bash
python3 -m studies.overtopping.analysis.stage06_competence_vs_overtopping_figures \
  --results-dir ../data \
  --out ../results/manuscript/figures/fig_competence_vs_coverage.pdf \
  --paper-figures-dir ../results/manuscript/figures \
  --paper-figures all
```

The default score mode is `phase-specific`: output-only finite-answer scores are
chance-normalized, input+output finite-answer scores are raw, and jailbreak
scores are safe/refusal rates. `--score-mode raw` and
`--score-mode chance-normalized` alter the finite-answer convention only.

The script supports task/model/phase/baseline filters, run-name filters,
checkpoint inclusion, deduplication, label controls, regression annotations,
marker sizing, and paper-figure selection. Run `python3 -m
studies.overtopping.analysis.stage06_competence_vs_overtopping_figures --help` for the complete CLI.

Paper-summary figure choices are `phase`, `checkpoint`, and `size`; `all`
generates all supported templates. Missing stats directories are excluded from
paper-summary figures unless `--paper-include-empty` is set.

## Stage 7: spiking report

`stage07_overtopping_spiking_report.py` aggregates `spiking_diagnostics`
artifacts when they exist.

```bash
python3 -m studies.overtopping.analysis.stage07_overtopping_spiking_report \
  --root ../data \
  --out ../results/diagnostics/overtopping_spiking
```

Alternatively, `--zip PATH` accepts a ZIP input. If the final-results
orchestrator finds no `spiking_diagnostics` directories, it writes
`report_status.json` with `status: not_available` instead of inventing a report.

## Shared Pipeline Stage 8: interaction validation

`pipeline/stage08_validate_interactions.py` evaluates the frozen candidate set jointly and under matched conditional backgrounds. It belongs to the shared causal pipeline because both studies can use the same model-backed simultaneous-set and conditional-marginal validation logic.

Required inputs include the pipeline data directory, candidate flip statistics,
and output directory. Model/task identity, intervention type, evaluation split,
null draws, and force/recompute behavior are explicit CLI options.

A typical invocation is:

```bash
python3 -m pipeline.stage08_validate_interactions \
  --input_data_dir DATA_DIR \
  --candidate_flip_stats_path STATS/flip_stats_by_neuron.csv \
  --out_dir STATS/interaction_validation \
  --task_module core.tasks.arithmetic_task \
  --evaluation_split test
```

Use the same model, candidate set, split, intervention baseline, and donor
configuration as the singleton analysis. Interaction outputs are not comparable
when those identities differ.

## Threshold sweeps and group analyses

`threshold_sweep_stats.py` analyzes threshold-sweep tables and can optionally run
sample-based per-neuron statistics when `--data-root` is supplied. Its default
primary threshold is 0.70 for both test-selected and all-fit score scopes.

`group_dominance.py` evaluates group-level dominance on a specified evaluation
split and intervention setup. `survey_dominance.py` summarizes dominance outputs
across runs.

`threshold_event_diagnostics.py` inspects threshold-event behavior and
`rebuild_directional_stats.py` reconstructs directional statistics from source
artifacts when compatible raw data are present.

These utilities are diagnostic analyses; they do not change primary-profile
membership.

## Metric families and provenance

The analysis package distinguishes several quantities that must not be treated as
interchangeable:

- singleton effect: effect of intervening on one candidate at a time;
- `U(J)`: event-level union coverage of the frozen candidate set;
- simultaneous `E(J)`: effect of intervening on the complete frozen set together;
- matched null: simultaneous-set effect under a matched random/null construction;
- CMC: candidate marginal contribution conditional on a specified background;
- OCC: baseline-conditional union coverage with its own event-level denominator;
- threshold counts: number of singleton effects above configured thresholds;
- overlap/compression and effective-number summaries: distributional summaries of
  how causal mass is spread across candidates.

Exact event-level sidecars are the authoritative source for event-defined metrics. Aggregate-only inputs can supply only quantities that are mathematically recoverable from their stored fields; missing exact denominators are reported as unavailable rather than replaced by a differently defined ratio.

Primary analyses are test-split analyses. Train-only donor estimation, frozen
candidate selection, and held-out test evaluation must remain distinct. Strict
provenance checks should only be disabled for explicitly descriptive debugging.

## Cache and path handling

Analysis scripts generally read outputs rather than maintaining model caches.
Some tables may contain absolute `stats_dir` paths from another machine;
`--data_root` options allow supported scripts to remap those paths into the
current experiment tree. Remapping does not relax identity checks for task,
model, intervention phase, split, or candidate set.

Do not copy a sidecar from one stats directory into another to satisfy the audit.
Regenerate the missing metric with the same run identity instead.

## Validation

At minimum, validate the package after code changes with:

```bash
cd code
python3 -m compileall -q studies/overtopping/analysis
python3 -m reporting.generate_final_results --help
python3 -m studies.overtopping.analysis.stage02_overtopping_latex_tables --help
python3 -m studies.overtopping.analysis.stage06_competence_vs_overtopping_figures --help
```

A full scientific validation additionally requires a representative result tree
because table/figure correctness depends on artifact schemas, split provenance,
and candidate identities, not only Python syntax.
