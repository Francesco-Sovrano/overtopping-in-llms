# Repository layout

## Repository root

```text
.
├── code/                          Python packages and documentation
├── data/                          persistent experimental outputs
├── cache/                         regenerable caches
├── results/                       aggregate reporting outputs
├── run_overtopping_experiments.sh
├── run_poisoning_experiments.sh
├── generate_results.sh
├── setup.sh
└── requirements.txt
```

## `code/`

```text
code/
├── core/
│   ├── eap/
│   ├── tasks/
│   └── ... shared causal/statistical utilities
├── pipeline/
│   ├── stage01_generate_prompts_and_answers.py
│   ├── stage02_generate_features.py
│   ├── stage02_export_dataset_scores.py
│   ├── stage03_extract_rules.py
│   ├── stage04_spectral_sample_datapoints.py
│   ├── stage05_discover_circuits.py
│   ├── stage06_analyze_bag_of_rules.py
│   ├── stage07_refine_neuron_anchored_rules.py
│   ├── stage08_validate_interactions.py
│   └── run_pipeline.sh
├── reporting/
│   ├── generate_final_results.py
│   └── result_paths.py
├── studies/
│   ├── overtopping/
│   │   ├── experiments/
│   │   └── analysis/
│   └── poisoning/
│       ├── lib/
│       ├── scripts/
│       ├── tasks/
│       ├── stage02_prepare_evaluation_cohorts.py
│       ├── stage04_compare_condition_behavior.py
│       ├── stage05_aggregate_backdoor_trajectory.py
│       ├── stage06_compare_checkpoint_circuits.py
│       ├── stage07_detect_poisoning_examples.py
│       ├── stage07_plot_detection_implications.py
│       ├── stage08_aggregate_cross_seed.py
│       └── stage08_plot_cross_seed.py
└── docs/
```

## Poisoning run directory

A run ID identifies one task/model/seed study unit. Example:

```text
data/poisoning/grammar/confirmatory__Qwen_Qwen2-1.5B-Instruct__seed_13/
```

All run-specific outputs are colocated:

```text
<run_dir>/
├── 01_training_checkpoints/
│   ├── clean/
│   ├── poisoned/
│   └── metadata/
├── 02_evaluation_cohorts/
├── 03_checkpoint_causal_discovery/
│   ├── clean/
│   └── poisoned/
├── 04_condition_comparisons/
├── 05_behavior_trajectories/
├── 06_circuit_overlap_analysis/
└── 07_poisoning_example_detection/
```

The phase subdirectory is `prompt_and_generation/` for `input_output` and `generation_only/` for `output_only`.

The cross-seed stage is separate because it combines multiple run directories:

```text
data/poisoning/final/<study_name>/08_cross_seed_aggregation/
├── tables/
└── figures/
```

See [Poisoning outputs](poisoning-outputs.md) for file-level details.
