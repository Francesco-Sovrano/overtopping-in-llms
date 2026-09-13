# Reporting

The reporting layer converts persistent experiment artifacts under `data/` into configured-study tables, metric-availability audits, statistical summaries, machine-readable analysis tables, and rendered outputs under `results/`.

## Standard command

From the repository root:

```bash
./generate_results.sh
```

Direct invocation from `code/`:

```bash
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile configured
```

The standard path is model-free. Model-backed recovery utilities are separate commands.

## Inputs and outputs

Primary inputs:

```text
data/                 overtopping scientific artifacts
data/poisoning/       poisoning scientific artifacts
```

Derived outputs:

```text
results/analysis/     machine-readable analyses and reproducibility audits
results/paper/        rendered tables and figures
```

The driver rejects a results root located inside the scientific `data/` tree or the repository cache tree.

## Population construction

Reporting begins from the selected configured overtopping manifest. Metric-specific applicability and availability are evaluated after the manifest is constructed.

- RQ1 uses settings with the required directional singleton metrics and analyzes intervention phases separately.
- RQ2 requires applicable set-level outputs and keeps mean-donor and mean-family replacement regimes separate.
- RQ3 uses the compatible threshold, graded, margin, temporal, or preemption artifacts required by each analysis.
- RQ4 uses configured Pythia checkpoint trajectories and poisoning trajectory outputs.

Verified zero-candidate settings remain explicit measured observations for metrics that define a zero. Missing artifacts remain unavailable measurements.

## Completeness controls

The repository wrapper allows incomplete metric coverage by default. Set:

```bash
ALLOW_INCOMPLETE_METRICS=false ./generate_results.sh
```

to pass `--require-complete-metrics` to the reporting driver.

CMC and paired conditional-null outputs are optional in the default wrapper configuration. Set:

```bash
REQUIRE_CMC=true ./generate_results.sh
```

to include them in the completeness contract.

## Threshold source resolution

An explicit RQ3 threshold-diagnostics source can be supplied as a directory or zip file:

```bash
./generate_results.sh --spiking-source /absolute/path/to/source
```

Without an explicit source, the reporting driver searches the configured data root and supported project-root archive locations.

## Reporting controls

`reporting.generate_final_results` supports:

```text
--catalogue-json PATH
--skip-paper-figures
--skip-spiking-report
--spiking-max-points N
--spiking-source PATH
--skip-poisoning-report
--require-complete-metrics
--skip-cmc-requirement
```

`--catalogue-json` selects an explicit saved configured population when required. The default `configured` profile derives the population from the current code registry.

Detailed module sequencing is documented in [Analysis pipeline](analysis-pipeline.md). Metric definitions are in [Metric semantics](interpretation.md). Generated output filenames are listed in [Generated outputs](figures.md).
