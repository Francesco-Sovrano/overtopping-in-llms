# Reporting and generated results

The reporting layer converts persistent experiment artifacts under `data/` into validated study tables, population audits, statistical summaries, machine-readable sidecars, and manuscript-facing outputs under `results/`.

## Standard command

```bash
./generate_results.sh
```

Direct invocation from `code/`:

```bash
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile study-56
```

## Input/output contract

The standard reporting driver treats:

```text
data/       read-only scientific input
cache/      not used to define analysis membership
results/    reporting output
```

It does not run model-backed experiment stages, rename experiment directories, or change cache keys. The driver rejects a `--results-root` that is equal to or nested inside `data/` or the repository `cache/` tree. Model-backed recovery utilities are separate commands documented in [Operations](../operations/README.md).

## Manifest and metric availability

Reporting first constructs the complete 56-setting overtopping study table. Metric-specific analyses then distinguish:

- **applicable:** the quantity is scientifically defined for the setting;
- **available:** the required compatible artifact exists;
- **missing/incompatible:** the setting remains in the audit, but the quantity is not used as an observed value.

The completeness audit applies RQ1 singleton requirements to every configured setting. RQ2 interaction requirements apply to settings with a nonempty candidate set. Other analyses apply their own artifact contracts and report their denominators.

## Analysis populations

- **RQ1:** all 56 configured settings, with I+O and Out analyzed separately; a 31-cell final-snapshot sensitivity is reported separately.
- **RQ2:** all evaluable settings within replacement regime; mean-donor and mean-family are separate.
- **RQ3:** the 56-setting manifest followed by threshold/graded artifact-specific eligibility.
- **RQ4:** configured checkpoint trajectories and controlled poisoning trajectories.

See [Analysis pipeline](analysis-pipeline.md) for modules and outputs, [Figure map](figures.md) for manuscript filenames, and [Interpretation](interpretation.md) for metric definitions and statistical units.

## Storage/addressing validation

The repository wrappers run the overtopping storage-contract check before experiment execution and standard reporting. The check is read-only: it validates the full registry and path uniqueness, and fingerprints storage-protected scientific configurations and persistent paths so additional settings cannot change their addressing. It does not inspect or modify `data/` or `cache/`.

Direct validation from `code/`:

```bash
python -m studies.overtopping.experiments.storage_contract
```
