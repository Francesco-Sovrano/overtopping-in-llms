# Overtopping analysis

The overtopping analysis package converts completed shared-pipeline runs into the primary matrix, statistical tests, manuscript figures, threshold/spiking diagnostics, and reproducibility audits.

The canonical orchestrator is:

```bash
./generate_results.sh
```

or, from `code/`:

```bash
python3 -m reporting.generate_final_results --data-root ../data --results-root ../results --primary-profile iclr-28
```

## Primary population

The manuscript profile contains exactly the task/model/phase settings declared by the `iclr-28` catalogue. Reporting does not define “all settings” as every top-level experiment directory present under `data/`.

Reference-run selection gives the preferred scientific configuration precedence over whether a run contains a nonempty agonist set. A scientifically valid preferred run with zero agonists remains the reference run.

Configuration tokens such as `tau0.3` and `M200000` are matched as delimited path tokens, so they do not match `tau0.35` or `M2000000`.

Held-out/train/all evaluation variants and capped variants are derived outputs and are excluded from reference-source selection.

## Score orientation and competence

Task-specific raw scores and chance levels are resolved through the task specification. Competence calculations receive the phase explicitly, including decode-only checkpoint fallbacks, so a trajectory uses one score convention.

## Stage-level analysis

### Stage 01 — catalogue visualization

`stage01_visualize_experiment_results.py` reads completed run summaries and produces catalogue-level visualizations and machine-readable summaries.

### Stage 02 — primary table and LaTeX tables

`stage02_overtopping_latex_tables.py` constructs the primary row set and reports pooled and directional causal metrics. Directional union coverage is named explicitly:

- `U_J_i2c`: baseline 0→1;
- `U_J_c2i`: baseline 1→0.

Directional threshold counts and effective support use their corresponding direction-eligible denominators.

### Stage 03 — required-metric audit

`stage03_audit_required_metrics.py` verifies that required metrics exist for the intended source runs. Source-directory reconstruction uses the canonical evaluation-directory resolver, including train/all and capped variants.

### Stage 04 — primary statistical analysis

`stage04_analyze_primary_metrics.py` performs the primary cross-setting statistical analyses and writes machine-readable tables used by paper figures and supplementary analysis.

### Stage 05 — manuscript outputs

`stage05_generate_manuscript_outputs.py` writes paper tables and intermediate figure data from the validated primary population.

### Stage 06 — RQ1/RQ2/RQ4 figures

`stage06_competence_vs_overtopping_figures.py` and `stage06_manuscript_story_figures.py` own the manuscript figure engines for prevalence/competence, composition/boundary cases, and checkpoint learning trajectories.

For the manuscript population, missing directional statistics are an error rather than a reason to drop selected settings silently. Rebuild directional singleton statistics from materialized scores when required.

The Pythia checkpoint story uses one canonical pooled-U/competence trajectory plus the directional companion figures. Redundant aliases are not published.

### Stage 07 — RQ3 threshold/spiking report

`stage07_overtopping_spiking_report.py` consumes an explicit RQ3 manifest. By default the manifest is the 28 primary overtopping settings plus the configured paper-supplementary overtopping settings (`--population-scope primary+supplementary`). Poisoning is a separate experiment family and is never discovered by recursive scanning or admitted to this manifest.

Before reporting, a population audit verifies:

- every required primary row is present;
- supplementary rows are included only when their exact diagnostics exist and are complete;
- required positive and negative baseline subsets are present;
- candidate/control populations needed by the report exist;
- the requested point-cap configuration is consistent;
- no source path escapes the overtopping data root into `data/poisoning`.

Missing primary population elements cause an explicit failure. Missing supplementary diagnostics are reported as missing rather than silently replaced by another run.

### Stage 08 — threshold-shape validation

`stage08_threshold_shape_validation.py` uses the same exact manifest and performs nested held-out validation. Scalar-feature selection occurs inside each training fold. The untouched fold is then used for threshold/logistic/isotonic evaluation. Threshold orientation is learned on the training fold, including inversion when the raw threshold predicate has negative MCC; held-out labels never choose the orientation.

Population inference is split into: (1) threshold-testability over all evaluated units and (2) conditional threshold shape after candidate/control matching on singleton causal strength within the same run and baseline. This avoids conditioning the control comparison on the tiny high-flip tail. The main inference unit is the run/baseline condition rather than individual neurons.

### Stage 09 — preemption report

`stage09_preemption_report.py` uses the same exact overtopping manifest. Pair-level rows are retained descriptively, but manuscript inference aggregates first to run/baseline/direction conditions to avoid pair-level pseudoreplication.

## RQ1 — prevalence and competence

RQ1 relates causal organization to task competence across the declared manuscript settings. It preserves genuine zero-candidate settings. Directional and pooled metrics are treated as separate quantities rather than substituted for one another.

## RQ2 — composition and boundary cases

RQ2 reports causal composition and superadditivity/boundary cases. In `fig3b_superadditive_boundary_cases.pdf`, the light bar is the singleton-union baseline `U(J)`, the dark bar is the observed joint-set effect `E(J)`, every bar is value-labelled, and exact zero-valued `U(J)` cases are rendered explicitly so a valid zero is visually distinguishable from a missing bar. The plotted gap `Δ = E(J) - U(J)` clarifies the size of the superadditive boundary case.

## RQ3 — threshold/spiking cut

Threshold diagnostics are model-backed analyses built from Stage-7 materialized candidate evaluations plus same-layer non-candidate controls evaluated by the diagnostic stage. Candidate interventions are copied from Stage 7; they are not regenerated. RQ3 defaults to primary+supplementary overtopping experiments and excludes poisoning by construction.

Threshold/spiking diagnostics must already exist before final-results generation. `generate_results.sh` never runs model-backed experiment stages or repairs `data/`.

Provide a specific diagnostics source when it is not discoverable under `data/`:

```bash
SPIKING_SOURCE=/absolute/path/to/spiking_diagnostics... ./generate_results.sh
```

## RQ4 — learning utility

RQ4 uses checkpoint trajectories, including Pythia checkpoints, to relate competence and causal coverage through learning. Checkpoint selection and competence scoring use the same phase-aware task conventions as the primary analysis.

## Result tree

The reporting layer writes:

```text
results/
├── README.md
├── paper/
│   ├── figures/
│   └── tables/
└── analysis/
    ├── primary_matrix/
    ├── figure_data/
    ├── reproducibility/
    └── diagnostics/
```

`results/analysis/` contains the full machine-readable populations and audits; `results/paper/` contains manuscript-facing products.

## Validation controls

`generate_results.sh` supports reporting-only controls:

```text
SPIKING_SOURCE
REQUIRE_CMC
ALLOW_INCOMPLETE_METRICS
```

The default completeness gate requires the primary directional/singleton metrics and simultaneous `E(J)` validation. Conditional-marginal validation is optional unless `REQUIRE_CMC=1` is set.
