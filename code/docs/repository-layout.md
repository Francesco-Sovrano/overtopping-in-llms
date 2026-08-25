# Repository layout

The repository separates source ownership from runtime artifacts.

```text
<repo>/
├── code/
│   ├── core/
│   │   ├── tasks/
│   │   └── eap/
│   ├── pipeline/
│   ├── reporting/
│   ├── studies/
│   │   ├── overtopping/
│   │   │   ├── experiments/
│   │   │   └── analysis/
│   │   └── poisoning/
│   │       ├── tasks/
│   │       ├── lib/
│   │       └── scripts/
│   └── docs/
├── data/
├── cache/
├── results/
├── run_overtopping_experiments.sh
├── run_poisoning_experiments.sh
├── generate_results.sh
├── setup.sh
└── requirements.txt
```

## Source directories

### `code/core/`

Shared implementation used across studies and pipeline stages. Important modules include:

- `task_spec.py` — common task interface;
- `tasks/` — ordinary task definitions;
- `caching_and_prompting.py` — model/provider caching;
- `feature_representation.py` and `feature_extraction_runner.py` — interpretable feature handling;
- `modeling_and_ablation.py`, `neuron_intervention.py`, `group_intervention.py` — causal interventions;
- `heldout_set_metrics.py`, `interaction_statistics.py`, `binomial_statistics.py` — statistical helpers;
- `spectral_analysis.py`, `spectral_sampling_plan.py` — spectral support;
- `eap/` — EAP/EAP-IG;
- `project_paths.py` — stable `CODE_ROOT` and `PROJECT_ROOT`.

### `code/pipeline/`

The shared numbered causal workflow:

```text
stage01_generate_prompts_and_answers.py
stage02_generate_features.py
stage02_export_dataset_scores.py
stage03_extract_rules.py
stage04_spectral_sample_datapoints.py
stage05_discover_circuits.py
stage06_analyze_bag_of_rules.py
stage07_refine_neuron_anchored_rules.py
stage08_validate_interactions.py
run_pipeline.sh
```

Stage 02 has two implementations because feature generation and external score export are alternative ways to populate the same logical stage. `run_pipeline.sh` is unnumbered because it orchestrates the sequence.

### `code/studies/overtopping/experiments/`

The overtopping experiment catalogue.

- `run_experiments.py` defines the explicit `paper-primary` and `paper-auxiliary` suites and their CLI.
- `execution.py` defines `RunSpec`, filtering, path construction, and conversion to a shared-pipeline command.

The catalogue asserts 28 primary and 11 auxiliary configurations.

### `code/studies/overtopping/analysis/`

Overtopping-specific analysis. Its canonical numbered sequence is:

```text
stage01_visualize_experiment_results.py
stage02_overtopping_latex_tables.py
stage03_audit_required_metrics.py
stage04_analyze_primary_metrics.py
stage05_generate_manuscript_outputs.py
stage06_competence_vs_overtopping_figures.py
stage07_overtopping_spiking_report.py
```

`analysis/lib/` contains analysis-local helpers such as the primary-matrix definition, catalogue helpers, and metric utilities. Unnumbered modules such as `compare_models.py`, `group_dominance.py`, and `threshold_sweep_stats.py` are optional analyses or diagnostics rather than mandatory stages.

### `code/studies/poisoning/`

Poisoning-specific workflow:

```text
studies/poisoning/
├── tasks/
├── lib/
├── scripts/
│   ├── poisoning_runtime_config.sh
│   ├── stage01_run_checkpoint_training.sh
│   ├── run_checkpoint_causal_workflow.sh
│   ├── run_ordinary_correctness_control.sh
│   ├── stage07_run_inference_defence.sh
│   └── stage07_run_training_defence.sh
├── stage02_prepare_evaluation_cohorts.py
├── stage04_compare_condition_behavior.py
├── stage05_aggregate_backdoor_trajectory.py
├── stage06_compare_checkpoint_circuits.py
├── stage07_inference_cumulative_ablation.py
├── stage07_training_verify_matched_runs.py
├── stage07_training_compare_protection.py
├── stage07_build_defence_overview.py
├── stage08_aggregate_cross_seed.py
└── stage08_plot_cross_seed.py
```

There is no local `stage03_*.py`: Stage 03 is checkpoint causal discovery performed by the shared `pipeline/`. The unnumbered `run_checkpoint_causal_workflow.sh` coordinates Stages 02–06.

### `code/reporting/`

Cross-study final-output orchestration. `generate_final_results.py` invokes study-owned modules and writes the final output manifest. It is not part of the overtopping study package because it also coordinates poisoning results.

### `code/docs/`

Canonical documentation. Package READMEs are short ownership/entry-point guides; detailed protocol and configuration documentation lives here.

## Runtime directories

### `data/`

Persistent scientific run artifacts. Overtopping runs use task/model-oriented directories. Poisoning runs use:

```text
data/poisoning/<task>/<run>/
```

with numbered persistent stages inside each run.

### `cache/`

Regenerable model-I/O, provider, representation, and other caches. Deleting compatible caches should not delete the authoritative scientific record of a completed run.

### `results/`

Aggregate outputs intended for inspection or manuscript use: catalogue manifests, audits, statistics, tables, figures, diagnostics, poisoning cross-seed summaries, and the final results manifest.

## Import convention

`code/` is the import root. From `code/`, representative commands are:

```bash
python3 -m studies.overtopping.experiments.run_experiments --help
python3 -m pipeline.stage01_generate_prompts_and_answers --help
python3 -m studies.poisoning.tasks.grammar --help
python3 -m reporting.generate_final_results --help
```

Shared modules use `core.*`; study modules use `studies.overtopping.*` or `studies.poisoning.*`.
