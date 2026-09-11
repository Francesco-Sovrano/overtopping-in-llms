# Causal-intervention code

This directory contains the shared causal-intervention pipeline, the overtopping study, the poisoning study, and the reporting layer.

## Package layout

```text
core/       shared task, model, attribution, intervention, cache, and statistics code
pipeline/   numbered model-backed causal-intervention stages
studies/    study registries and study-specific analyses
reporting/  final aggregation, audits, tables, and figure generation
docs/       repository documentation
```

Repository-level artifact roots are:

```text
data/       persistent model-backed experiment outputs
cache/      reusable computation caches
results/    derived analyses and manuscript products
```

Scientific populations are defined by `RunSpec` registries, held-out row identities, and analysis manifests. Cache presence is never used to define study membership.

## Overtopping execution flow

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
08 simultaneous-set and interaction validation
```

The reporting layer then produces:

- **RQ1:** directional causal reach and width-normalized high-effect candidate counts versus competence;
- **RQ2:** simultaneous-set composition, singleton-versus-joint decomposition, and matched-set specificity;
- **RQ3:** candidate/control threshold observability and support-specific graded intervention response;
- **RQ4:** Pythia checkpoint trajectories and controlled poisoning trajectories.

## Running the study

From the repository root:

```bash
./run_overtopping_experiments.sh --list
./run_overtopping_experiments.sh --dry-run
./run_overtopping_experiments.sh
```

Direct invocation from `code/`:

```bash
python -m studies.overtopping.experiments.run_experiments --dry-run
```

The configured study contains 48 settings. See [`docs/experiments/overtopping.md`](docs/experiments/overtopping.md) for the exact decomposition and cell coverage.

## Reporting

From the repository root:

```bash
./generate_results.sh
```

Reporting is best-effort while experiments are still running: every RQ is attempted from the completed/auditable subset, incomplete configured settings are listed in the corresponding audits, and missing settings do not suppress an otherwise renderable RQ. Verified zero-candidate runs remain explicit zero observations. To require a publication-complete 48-setting population and fail on any missing required metric, run:

```bash
ALLOW_INCOMPLETE_METRICS=false ./generate_results.sh
```

Direct invocation from `code/`:

```bash
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile study-48
```

The 48-row study table is constructed before metric-specific filtering. A missing derived metric remains a missing value/status for that setting; it does not remove the setting from the configured study manifest.

RQ2 replacement regimes are not pooled. `mean-positional` is grouped with `mean` for the mean-replacement sensitivity, while `mean-donor` is analyzed separately.

## Storage compatibility

`RunSpec` owns the persistent path contract for overtopping settings. Reporting changes do not alter:

- task/model identifiers;
- intervention names passed to the pipeline;
- phase/decode-only flags;
- circuit and bag labels;
- held-out evaluation suffixes;
- Stage-5 input-data directories;
- Stage-7 statistics directories.

Validate the registry contract with:

```bash
python -m studies.overtopping.experiments.storage_contract
```

The contract fingerprints scientific configuration and persistent addressing while excluding runtime-only batch size. The standard reporting driver writes under `results/`. Model-backed regeneration utilities that write scientific outputs under `data/` are separate commands and are documented in [`docs/operations/README.md`](docs/operations/README.md).

## Documentation

Start with [`docs/README.md`](docs/README.md).
