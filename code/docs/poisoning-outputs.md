# Poisoning outputs and filesystem policy

Poisoning experiments keep four kinds of artifacts separate: per-run scientific data, regenerable caches, final defence/reporting outputs, and manuscript figures.

## Per-run scientific data

Each baseline, protected, or random-protected training run has the same stage layout:

```text
data/poisoning/<task>/<run>/
├── 01_training_checkpoints/
├── 02_evaluation_cohorts/
├── 03_checkpoint_causal_discovery/
├── 04_condition_comparisons/
├── 05_behavior_trajectories/
└── 06_circuit_overlap_analysis/
```

`01_training_checkpoints/metadata/run_config.json` and `checkpoint_manifest_all.csv` define run identity and checkpoint provenance. Evaluation cohorts are persistent scientific inputs because changing them changes the estimand. Checkpoint causal-discovery outputs are also persistent inputs to later defence analyses.

The root launcher creates the ordinary baseline run plus, when `RUN_TRAINING_PROTECTION=1` (the default), two matched training-protection runs whose names end in `__protected` and `__random_protected`.

## Final poisoning stages

The per-run pipeline ends at stage 06. Study-level outputs continue that numbering instead of switching to an unrelated `summary/` tree:

```text
data/poisoning/final/<study>/
├── 07_defence_evaluation/
│   └── <task>/
│       └── <model>/
│           └── seed_<seed>/
│               └── <phase>/
│                   ├── defence_overview.md
│                   ├── defence_overview.json
│                   ├── inference_time/
│                   │   ├── backdoor_lift_cumulative_topk_ablation.csv
│                   │   ├── matched_random_group_results.csv
│                   │   ├── backdoor_lift_cumulative_topk_ablation.pdf
│                   │   ├── defence_summary.md
│                   │   └── defence_configuration.json
│                   └── training_time/
│                       ├── matched_training_identity.json
│                       ├── training_protection_trajectory.csv
│                       ├── training_protection_key_metrics.csv
│                       ├── training_protection.pdf
│                       └── training_protection_summary.md
└── 08_cross_seed_aggregation/
    ├── tables/
    │   ├── checkpoint_trajectories_all_seeds.csv
    │   ├── checkpoint_metrics_by_model_across_seeds.csv
    │   ├── developmental_timing_by_seed.csv
    │   ├── developmental_timing_across_seeds.csv
    │   └── aggregation_config.json
    └── figures/
        ├── poisoning_primary_trajectories.pdf
        ├── poisoning_specificity_checks.pdf
        ├── poisoning_inference_time_defence.pdf
        └── poisoning_training_time_defence.pdf
```

`defence_overview.md` is the first file to open for a task/model/seed/phase. It explicitly reports both defence mechanisms and whether either one is missing. The root launcher runs training-time protection by default; set `RUN_TRAINING_PROTECTION=0` only when intentionally omitting that mechanism.

### Prospective inference-time defence

For checkpoint `t`, channel identities and their order come only from the latest strictly earlier checkpoint with a completed frozen discovery ranking. The checkpoint being defended is evaluation-only and cannot select its own coalition. Candidate coalitions are compared with structurally matched random noncandidate coalitions and with ordinary-task collateral-damage controls.

### Training-time protection

The protected training arm freezes direct writes to overtopping channels selected from the pre-poisoning virgin-model ordinary-task analysis. The matched-random arm protects the same type/number of channels selected randomly. The comparison uses exactly the `poisoned`, `protected_poisoned`, and `random_protected_poisoned` trajectory rows.

## Defence caches

Regenerable inference-defence caches live inside the same task/run namespace as the checkpoint-discovery cache. The run directory name is reused verbatim as the cache run identity, so all regenerable artifacts for one poisoning run remain together:

```text
cache/poisoning/<task>/<run>/
├── checkpoint_causal_discovery/
│   └── <phase>/adaptive_circuit_discovery/
└── defence/
    └── <input_output|output_only>/
        └── fraction_<fraction>/
            ├── candidate_rows.csv
            ├── matched_random_rows.csv
            └── cache_manifest.json
```

For example, the grammar run `confirmatory__Qwen_Qwen2-1.5B-Instruct__seed_13` at fraction `0.25` uses:

```text
cache/poisoning/grammar/confirmatory__Qwen_Qwen2-1.5B-Instruct__seed_13/defence/input_output/fraction_0.250000/
```

Grammar and arithmetic therefore remain separated immediately below `cache/poisoning/`, and defence is a property of a specific run rather than a parallel top-level cache branch. Training-time protection reuses the normal training/checkpoint and causal-discovery resume mechanisms.

## Final manuscript results

`generate_results.sh` no longer creates `data/poisoning/summary/paper_inputs` or a figure-less `summary/matrix` directory. Derived cross-seed reporting tables and figures are placed together under:

```text
results/poisoning/
├── 01_cross_seed_tables/
└── 02_figures/
    ├── poisoning_primary_trajectories.pdf
    ├── poisoning_specificity_checks.pdf
    ├── poisoning_inference_time_defence.pdf
    └── poisoning_training_time_defence.pdf
```

The defence figures are rendered from all available `07_defence_evaluation` study directories.

## Resume and collision policy

The code does not silently repurpose a nonempty run directory with incompatible metadata. Training resumes only when the saved run configuration identifies the same experiment. Defence caches are separately provenance-keyed as described above. If a cache is incompatible, the code recomputes it rather than requiring manual deletion.

## Legacy checkpoint-label migration

New training snapshots, checkpoint analysis directories, and cache labels use
`progress_010pct__step_0025` style names.  Existing `frac_0100_step_25`
directories are readable for resume compatibility but are legacy names.
Preserve expensive model snapshots by migrating them rather than deleting them:

```bash
cd code
python3 -m studies.poisoning.migrate_legacy_checkpoint_labels \
  --run_dirs ../data/poisoning/grammar/<run>,../data/poisoning/arithmetic/<run>
python3 -m studies.poisoning.migrate_legacy_checkpoint_labels \
  --run_dirs ../data/poisoning/grammar/<run>,../data/poisoning/arithmetic/<run> \
  --apply
```

The first command is a dry run.  The migration renames exact legacy checkpoint
or checkpoint-result directory names and rewrites authoritative checkpoint
manifests.  Regenerable caches and aggregate reports should be deleted and
recomputed instead of moved.
