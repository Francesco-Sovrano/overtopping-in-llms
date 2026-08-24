# Repository layout

The repository separates source code, persistent experiment artifacts, regenerable caches, and final reports. This separation is part of reproducibility: scientific run identity belongs in `data/`, while derived caches may be deleted and recomputed.

```text
<repo>/
├── code/
│   ├── analysis/
│   ├── docs/
│   ├── experiments/
│   ├── lib/
│   ├── pipeline/
│   └── poisoning/
├── data/
├── cache/
├── results/
├── logs/
├── .env/
├── run_experiments.sh
├── run_poisoning_experiments.sh
├── generate_results.sh
├── setup.sh
└── requirements.txt
```

## `experiments/`

`experiments/run_experiments.py` defines the explicit standard catalogue. `experiments/execution.py` contains `RunSpec`, filtering, path construction, and command generation. The catalogue contains 28 `paper-primary` and 11 `paper-auxiliary` configurations.

## `pipeline/`

The numbered pipeline stages are:

```text
1_generate_prompts_and_answers.py
2_generate_features.py
2_export_dataset_scores.py
3_extract_rules.py
4_spectral_sample_datapoints.py
5_discover_circuits.py
6_analyze_bag_of_rules.py
7_refine_neuron_anchored_rules.py
_run_pipeline.sh
```

`_run_pipeline.sh` is the per-configuration orchestrator. Stages may reuse persistent run artifacts and compatible caches when their provenance agrees with the requested configuration.

## `analysis/`

Contains the final-results orchestrator and specialized statistical/reporting modules. Important components include:

- `generate_final_results.py` — top-level analysis orchestration;
- `lib/primary_matrix.py` — strict 28-row primary-matrix validation;
- `stage02_overtopping_latex_tables.py` — primary table construction;
- `stage03_audit_required_metrics.py` — exact metric completeness audit;
- `stage04_analyze_primary_metrics.py` — primary statistical analysis;
- `stage05_generate_manuscript_outputs.py` — manuscript tables/macros;
- `stage06_competence_vs_overtopping_figures.py` — manuscript figures;
- `stage07_overtopping_spiking_report.py` — diagnostic report;
- `validate_interactions.py` — simultaneous-set and conditional marginal validation.

`analysis/tools/` contains focused recovery/maintenance utilities that are part of the current workflow.

## `lib/`

Shared implementation code includes:

- task specifications and standard task modules;
- prompt/model-I/O caching;
- feature representation and rule utilities;
- TransformerLens model loading and activation replacement;
- channel intervention and ablation;
- binomial, interaction, and held-out set statistics;
- spectral sampling;
- the internal `lib/eap/` EAP/EAP-IG implementation.

Task-specific semantics belong in `lib/tasks/`; shared pipeline code should not hard-code task labels when the task interface can provide them.

## `poisoning/`

The poisoning package uses task modules plus generic numbered stages:

```text
poisoning/
├── tasks/
│   ├── base.py
│   ├── registry.py
│   ├── grammar.py
│   └── arithmetic.py
├── lib/
├── scripts/
├── stage02_prepare_causal_pool.py
├── stage03_compare_condition_behavior.py
├── stage04_aggregate_backdoor_trajectory.py
├── stage05_compare_checkpoint_circuits.py
├── stage06_cumulative_ablation.py
├── stage07_aggregate_matrix.py
├── protection01_verify_matched_runs.py
└── protection02_compare_training_protection.py
```

The task modules own training-data construction, prompt semantics, behavioral readouts, and causal task specifications. `poisoning/lib/` contains task-agnostic training, scheduling, marker, checkpoint, trajectory, causal-pool, CHA, trigger-lift, and suppression mechanisms.

## Runtime roots and path identity

`lib/project_paths.py` defines the repository and code roots. Normal non-poisoning run artifacts are stored under `data/<task>/...`; poisoning runs are stored under `data/poisoning/<task>/<run>/...`.

Regenerable caches are stored under `cache/`. Poisoning checkpoint causal-discovery caches default to:

```text
cache/poisoning/<task>/<run>/checkpoint_causal_discovery/<phase>/adaptive_circuit_discovery/
```

Final aggregate outputs are stored under `results/`, using descriptive directories such as `primary_analysis/`, `manuscript/`, `diagnostics/`, and `poisoning/`.

## Import and launch convention

`code/` is the import root rather than a single enclosing Python package. Run module commands from `code/`:

```bash
cd code
python3 -m experiments.run_experiments --help
python3 -m analysis.generate_final_results --help
python3 -m poisoning.tasks.grammar --help
```

Run the repository-level shell launchers from repository root.
