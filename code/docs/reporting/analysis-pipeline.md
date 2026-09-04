# Analysis and reporting pipeline

The reporting layer converts completed experiment artifacts under `data/` into population audits, primary tables, statistical summaries, machine-readable figure sidecars, and manuscript-facing outputs under `results/`.

The standard entry point is:

```bash
./generate_results.sh
```

Direct invocation from `code/`:

```bash
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile iclr-28
```

## Declared analysis populations

| Analysis | Population |
|---|---|
| Primary matrix | 28 `paper-primary` settings |
| RQ1 Figure 2 | 39 primary + auxiliary settings |
| RQ2 composition | 28 primary settings |
| RQ3 threshold-event analysis | 28 primary settings plus auxiliary settings with complete exact diagnostics |
| RQ4 Pythia trajectories | configured checkpoint runs |
| Poisoning | configured poisoning task/model/seed runs under `data/poisoning/` |

Poisoning is excluded from RQ1–RQ3 overtopping populations.

Detailed scientific methods are documented under [Research questions](../research-questions/). This document describes the reporting stages and artifact contract.

## Reporting stages

### Stage 01 — catalogue visualization

`stage01_visualize_experiment_results.py`

Reads configured overtopping outputs and creates catalogue-level summaries.

### Stage 02 — primary table

`stage02_overtopping_latex_tables.py`

Constructs the 28-row primary matrix and materializes pooled, directional, concentration, redundancy, joint-effect, and matched-null fields when available.

Representative directional fields:

```text
U_J_i2c   singleton-union reach on baseline-0 rows, 0→1
U_J_c2i   singleton-union reach on baseline-1 rows, 1→0
s_1_i2c   strongest 0→1 singleton effect
s_1_c2i   strongest 1→0 singleton effect
```

Directional fields retain their own eligible denominators.

### Stage 03 — required-metric audit

`stage03_audit_required_metrics.py`

Checks that required primary metrics exist for the declared source runs and evaluation variants.

### Stage 04 — primary statistics

`stage04_analyze_primary_metrics.py`

Computes cross-setting associations for the primary matrix, including raw correlations, bootstrap intervals, task/phase analyses, and adjusted models with configured task, phase, model-family, and replacement-baseline controls.

Fields named `N_t_*_density` or `N_t_*_per_1k_layer` use `d_model` normalization. They are width-normalized discovered-candidate counts, not coordinate-population prevalence estimates.

### Stage 05 — manuscript tables and sidecars

`stage05_generate_manuscript_outputs.py`

Builds manuscript tables and machine-readable sidecars from validated primary-matrix and interaction products.

### Stage 06 — RQ1, RQ2, and checkpoint figures

The Stage-06 reporting modules generate:

- RQ1 competence versus directional reach and high-effect-count figures;
- RQ2 composition and matched-set figures;
- Pythia checkpoint trajectories used by RQ4.

RQ1 Figure 2 resolves the exact 39-setting catalogue and validates the 17 input+output / 22 decode-only phase split.

### Stage 07 — RQ3 population report

`stage07_overtopping_spiking_report.py`

Loads only configured RQ3 diagnostics, verifies primary/auxiliary coverage and discovery-direction provenance, excludes poisoning, and creates population summaries and direction-specific response outputs.

### Stage 08 — nested threshold-shape validation

`graded_agonist_intervention.py` + `stage08_graded_agonist_report.py`

Runs and aggregates the primary RQ3 graded causal intervention on each agonist's known held-out directional flip support. Optional same-agonist negative support is a within-agonist reference. Nested held-out threshold `|MCC|` is not a manuscript endpoint.

Current analysis schema:

```text
graded-agonist-intervention-v1
```

### Stage 09 — preemption aggregation

`stage09_preemption_report.py`

Loads direction-aware preemption outputs from the exact overtopping manifest and aggregates pair measurements to run × baseline × direction conditions before inference.

The preemption interaction products use the direction-aware preemption schema expected by Stage 09.

## Research-question ownership

The reporting stages above feed four independent scientific documents:

- [RQ1 — prevalence and competence](../research-questions/rq1-prevalence.md)
- [RQ2 — composition](../research-questions/rq2-composition.md)
- [RQ3 — threshold events](../research-questions/rq3-threshold-event.md)
- [RQ4 — learning](../research-questions/rq4-learning.md)

Figure filenames and source analyses are listed in [Manuscript figure map](figures.md).

## Result tree

Principal generated paths:

```text
results/
├── analysis/
│   ├── primary_matrix/
│   ├── figure_data/
│   ├── reproducibility/
│   ├── rq3_threshold_event/
│   └── rq4_learning/
└── paper/
    ├── figures/
    │   ├── 02_rq1_prevalence/
    │   ├── 03_rq2_composition/
    │   ├── 04_rq3_spiking_cut/
    │   └── 05_rq4_learning/
    └── tables/
```

`results/analysis/` is the machine-readable reporting layer. `results/paper/` contains manuscript-facing derived outputs.

## Reproducibility audits

Population and output audits are written under:

```text
results/analysis/reproducibility/
```

The reporting driver validates required manuscript outputs and the declared RQ populations before completing a standard paper build.

## Reporting controls

`generate_results.sh` exposes reporting controls including:

```text
SPIKING_SOURCE=<path>       explicit RQ3 diagnostics source
REQUIRE_CMC=1              require optional CMC validation
ALLOW_INCOMPLETE_METRICS=1 permit reporting with missing required metrics
```

The standard manuscript path uses the default requirement for complete required primary metrics.
