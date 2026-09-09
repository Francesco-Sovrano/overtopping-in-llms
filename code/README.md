# Causal-intervention code

This directory contains the shared causal-intervention pipeline, the overtopping study, the controlled-poisoning study, and the reporting code that converts persistent experiment outputs into manuscript tables and figures.

The code is organized around three persistent roots located at the repository level:

```text
data/       model-backed experiment outputs and population/provenance records
cache/      reusable computation caches
results/    derived analyses, audits, tables, and figures
code/       this source tree
```

Scientific populations are defined by experiment catalogues, manifests, persisted row identities, and analysis profiles. Cache presence does not define analysis membership.

## Main study flow

The overtopping pipeline executes the following sequence:

```text
01 prompts and answers
02 feature export
03 rule extraction
04 spectral sampling plan
05 circuit discovery
06 candidate/rule analysis
07 held-out singleton causal evaluation
07b graded agonist intervention
07c threshold-event diagnostics
08 simultaneous-set, interaction decomposition, CMC, and optional preemption validation
```

The reporting layer then builds:

- **RQ1:** directional causal reach and width-normalized high-effect candidate counts;
- **RQ2:** simultaneous-set composition, example-level singleton-versus-joint decomposition, and matched-set specificity;
- **RQ3:** candidate/control threshold-event observability and support-specific graded causal dose response;
- **RQ4:** Pythia checkpoint trajectories and controlled-poisoning analyses.

## RQ2 interaction decomposition

Stage 8 writes per-example and per-setting singleton-versus-joint composition classes. Reporting aggregates them under:

```text
results/analysis/rq2_composition/interaction_decomposition/
```

When available for the primary population, the manuscript-facing panel is:

```text
results/paper/figures/03_rq2_composition/fig3d_singleton_joint_decomposition.pdf
```

## RQ3 manuscript figures

The Figure 4 directory is:

```text
results/paper/figures/04_rq3_spiking_cut/
```

The manuscript-facing files are:

```text
fig4a_candidate_control_spiking_cut_summary.pdf
fig4b_threshold_shape_model_comparison_by_direction.pdf
fig4c_graded_agonist_dose_response.pdf
fig4s1_threshold_testability_by_condition.pdf
fig4s2_strength_matched_thresholdability.pdf
fig4s3_nested_tecs_lower_bound_ecdf.pdf
fig4s4_threshold_tail_response_by_direction.pdf
fig4s5_graded_agonist_single_crossing.pdf
```

Threshold diagnostics supply Figure 4a, Figure 4b, and S1–S4. The graded agonist experiment supplies Figure 4c and S5.

## Running modules

Python modules should be run from this directory or with this directory on `PYTHONPATH`:

```bash
cd code
python -m studies.overtopping.experiments.run_experiments --dry-run
```

The repository-level launchers, when present, provide the normal end-to-end entry points:

```bash
./run_overtopping_experiments.sh
./run_poisoning_experiments.sh
./generate_results.sh
```

The reporting driver can also be invoked directly from `code/`:

```bash
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile iclr-28
```

## Documentation

Start with [docs/README.md](docs/README.md).

The main references are:

- [Getting started](docs/getting-started/README.md)
- [Architecture](docs/methods/architecture.md)
- [Core concepts](docs/methods/concepts.md)
- [Pipeline stages](docs/methods/pipeline.md)
- [Overtopping experiment catalogue](docs/experiments/overtopping.md)
- [Research questions](docs/research-questions/README.md)
- [Reporting pipeline](docs/reporting/analysis-pipeline.md)
- [Figure map](docs/reporting/figures.md)
- [Operations and regeneration](docs/operations/README.md)
