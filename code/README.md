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
results/    derived analyses, audits, tables, and rendered outputs
```

Scientific populations are defined by `RunSpec` registries, held-out row identities, and analysis manifests rather than cache presence.

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
07c temporal prefix intervention
07d temporal suffix intervention
    threshold-event diagnostics
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

The configured study has no fixed required setting count. See [`docs/experiments/overtopping.md`](docs/experiments/overtopping.md) for the current registry structure and cell coverage.

## Reporting

From the repository root:

```bash
./generate_results.sh
```

Reporting runs each RQ from the completed, auditable settings. The corresponding audits list incomplete configured settings, and reports are generated when their required inputs are available. Verified zero-candidate runs remain explicit zero observations. To require complete required metrics for settings in the configured manifest, run:

```bash
ALLOW_INCOMPLETE_METRICS=false ./generate_results.sh
```

Direct invocation from `code/`:

```bash
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile configured
```

The configured study table is constructed before metric-specific filtering. Its row count is derived from the current manifest. A missing derived metric remains a missing value/status for that setting; it does not remove the setting from the configured study manifest.

RQ2 replacement regimes are not pooled. `mean-donor` is analyzed separately from `mean`.

## Storage compatibility

`RunSpec` owns the persistent path contract for overtopping settings. Reporting does not alter:

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
