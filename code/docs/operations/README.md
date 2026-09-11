# Operations and regeneration

This document describes reporting regeneration, model-backed recovery, and cache handling.

## 1. Inspect configuration before execution

Overtopping:

```bash
./run_overtopping_experiments.sh --list
./run_overtopping_experiments.sh --dry-run
```

Poisoning:

```bash
./run_poisoning_experiments.sh --dry-run
```

Validate the overtopping persistent-addressing contract:

```bash
cd code
python -m studies.overtopping.experiments.storage_contract
cd ..
```

This check is read-only.

## 2. Regenerate reporting only

When model-backed experiment artifacts under `data/` are complete:

```bash
./generate_results.sh
```

Direct invocation:

```bash
cd code
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile study-48
```

The standard reporting driver reads `data/` and writes `results/`. It does not run model inference, rename experiment directories, or rewrite cache stores. It rejects a results root located inside `data/` or the repository `cache/` tree. The repository wrapper runs the read-only storage-contract check before reporting.

## 3. Overtopping population diagnostics

The configured study table contains all 48 settings. A metric-specific missing value is reported as an availability status rather than by shortening the manifest.

Key audit locations:

```text
results/analysis/primary_matrix/tables/primary_table.csv
results/analysis/reproducibility/
results/analysis/rq2_composition/
results/analysis/rq3_threshold_event/
```

`primary_table.csv` is the configured 48-setting study manifest used by the reporting CLIs. In commands that expose `--population primary` or `--population-scope primary`, `primary` selects this configured study manifest; analysis-specific applicability and availability filters are applied afterward.

If RQ1 resolves fewer than 48 settings, inspect the exact configured path for the missing setting. RQ1 requires:

```text
48 settings total
22 input+output
26 output-only
```

A configured directory with a completed zero-candidate result remains a valid RQ1 observation. Completion is verified from Stage-6 `neuron_buckets.json` plus successful `rule_knockout.json` records; reporting resolves both the current Stage-6 layout and supported legacy/export layouts. RQ2 interaction completeness applies only when the frozen candidate set is nonempty. A missing or unverified configured directory remains an incomplete experiment.

## 4. Rebuild directional singleton statistics

Directional metrics can be regenerated from materialized Stage-7 singleton outcomes without repeating language-model ablations when the required `scores.csv` and discovery ranking are present:

```bash
cd code
python -m studies.overtopping.analysis.rebuild_directional_stats \
  --primary_table ../results/analysis/primary_matrix/tables/primary_table.csv \
  --data_root ../data \
  --population primary \
  --evaluation_split test \
  --dry-run
```

Remove `--dry-run` only after inspecting the targets.

This utility writes regenerated directional statistics into the existing Stage-7 statistics directories under `data/`. It does not delete singleton caches or rerun model ablations.

## 5. RQ2 interaction products

RQ2 simultaneous-set outputs are produced by Stage 8:

```text
pipeline.stage08_validate_interactions
```

Per-run outputs are stored under:

```text
<stage7 stats dir>/interaction_validation/
```

Important files:

```text
interaction_validation_summary.json
interaction_validation_summary.csv
composition_example_decomposition.csv
composition_decomposition_summary.csv
composition_decomposition_summary.json
matched_control_strata.csv
matched_random_set_membership.csv
```

If the corresponding group-intervention cache is compatible, Stage 8 can reuse it. Cache validity is determined by the cache metadata and scientific configuration.

Aggregate mean-donor and mean-family regimes separately. The reporting driver does this automatically. Direct decomposition command:

```bash
cd code
python -m studies.overtopping.analysis.stage09_composition_decomposition_report \
  --root ../data \
  --out ../results/analysis/rq2_composition/interaction_decomposition \
  --primary-table ../results/analysis/primary_matrix/tables/primary_table.csv \
  --population-scope primary \
  --replacement-regime mean-donor \
  --evaluation-split test
```

Use `--replacement-regime mean` for the separate mean/mean-positional analysis.

## 6. RQ3 threshold reporting

Aggregate threshold reporting can use a directory or compatible zip containing:

```text
aggregate_flip_stats.csv
aggregate_unit_tests.csv
```

Full threshold-shape reporting additionally requires:

```text
aggregate_activation_flip_rows.csv
```

The oriented binned-response panel uses:

```text
aggregate_binned_curves.csv
```

Direct reporting commands:

```bash
cd code
python -m studies.overtopping.analysis.stage07_overtopping_spiking_report \
  --root /path/to/threshold_diagnostics \
  --out ../results/analysis/rq3_threshold_event/spiking_diagnostics \
  --paper-figures-dir ../results/paper/figures/04_rq3_spiking_cut \
  --primary-table ../results/analysis/primary_matrix/tables/primary_table.csv \
  --data-root ../data \
  --evaluation-split test \
  --population-scope primary

python -m studies.overtopping.analysis.stage08_threshold_shape_validation \
  --root /path/to/threshold_diagnostics \
  --out ../results/analysis/rq3_threshold_event/spiking_diagnostics/threshold_shape_validation \
  --paper-figures-dir ../results/paper/figures/04_rq3_spiking_cut \
  --primary-table ../results/analysis/primary_matrix/tables/primary_table.csv \
  --data-root ../data \
  --evaluation-split test \
  --population-scope primary
```

These two reporters write under `results/`.

## 7. RQ3 graded intervention reporting

Each configured graded run is resolved from:

```text
<stage7 stats dir>/graded_agonist_intervention/
├── graded_agonist_intervention.json
├── graded_agonist_unit_summary.csv
└── graded_agonist_dose_rows.csv.gz
```

Aggregate reporting:

```bash
cd code
python -m studies.overtopping.analysis.stage08_graded_agonist_report \
  --root ../data \
  --out ../results/analysis/rq3_threshold_event/spiking_diagnostics/graded_agonist \
  --paper-figures-dir ../results/paper/figures/04_rq3_spiking_cut \
  --primary-table ../results/analysis/primary_matrix/tables/primary_table.csv \
  --population-scope primary \
  --evaluation-split test
```

This command reads per-run graded outputs and writes aggregate products under `results/`.

## 8. Rebuild threshold diagnostics

When per-run threshold-event diagnostics are absent and the required Stage-7 materialization exists, inspect the model-backed recovery plan first:

```bash
cd code
python -m studies.overtopping.analysis.rebuild_spiking_diagnostics \
  --primary-table ../results/analysis/primary_matrix/tables/primary_table.csv \
  --data-root ../data \
  --population-scope primary \
  --evaluation-split test \
  --dry-run
```

`--dry-run` is read-only. Without `--dry-run`, this utility may:

- backfill missing discovery-direction provenance in Stage-7 CSVs;
- run threshold-event diagnostics;
- write per-run diagnostic products under the existing experiment tree.

Existing complete provenance CSVs are reused without rewriting them. Use `--force` only when the threshold-event computation itself must be recomputed.

## 9. Cache policy

Treat the three artifact classes separately.

### Persistent scientific outputs

`data/` contains model-backed results and frozen scientific identities. Preserve these unless the corresponding scientific computation is intentionally rerun.

### Reusable caches

`cache/` and experiment-local cache stores accelerate model-backed computations. Preserve a cache when its model state, intervention configuration, row identity, and schema metadata match the requested computation.

The 48-setting reporting manifest does not change cache keys or persistent experiment paths. The experiment runner and reporting wrapper validate a read-only storage/addressing fingerprint before execution. The fingerprint covers scientific configuration and persistent paths but excludes runtime-only batch size.

### Derived reports

`results/` contains derived tables, audits, figures, and sidecars. These can be regenerated from compatible scientific inputs.

Deleting `results/` does not require deleting `data/` or `cache/`.

## 10. Accelerator memory pressure

Reduce execution batch size without changing the scientific population. High-N evaluation, graded interventions, group interventions, and poisoning evaluations expose separate batch controls. A batch-size change does not require a population change.

## 11. Provider authentication

Check whether credentials are present without printing their values:

```bash
python - <<'PY'
import os
for name in ("GROQ_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"):
    print(f"{name}: {'set' if os.environ.get(name) else 'not set'}")
PY
```

For direct module execution:

```bash
set -a
. ./.secrets.env
set +a
```

## 12. Poisoning regeneration

Poisoning outputs use a separate namespace under `data/poisoning/`. Normal-task, trigger/control, localization, fixed-candidate, and checkpoint outputs carry their own population metadata. Recompute only the stage whose scientific inputs or cache metadata do not match the requested configuration.

See [Poisoning protocol](../experiments/poisoning/README.md), [configuration](../experiments/poisoning/configuration.md), and [outputs](../experiments/poisoning/outputs.md).
