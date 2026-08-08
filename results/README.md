# Final results directory

`results/` is the canonical root for aggregate statistics, manuscript tables, and publication figures. Raw model/intervention artifacts belong under `../data/`; runtime caches belong under `../cache/`.

The standard final-output tree is:

```text
results/
├── configured_experiments.json
├── pipeline_failures.json
├── catalogue/
├── paper_tables/
├── primary_metrics/
├── manuscript/
├── paper_figures/
├── overtopping_spiking_report/
└── final_results_manifest.json
```

## Generate experiments and results

From repository root:

```bash
./run_experiments.sh
```

This runs the complete 127-configuration non-poisoning catalogue and then generates the requested primary manuscript outputs. `PRIMARY_PROFILE` defaults to `iclr-28`.

```bash
PRIMARY_PROFILE=legacy-27 ./run_experiments.sh
```

## Regenerate results from existing data

```bash
./generate_results.sh
```

By default it reads `<repo>/data` and writes to `<repo>/results`. Override only the input tree with `DATA_ROOT`:

```bash
DATA_ROOT=/path/to/data PRIMARY_PROFILE=iclr-28 ./generate_results.sh
```

The root results script delegates to `analysis/29_generate_final_results.py`, which builds `paper_tables/`, `primary_metrics/`, `manuscript/`, `paper_figures/`, and `overtopping_spiking_report/` and writes `final_results_manifest.json`.

If `results/configured_experiments.json` exists, catalogue-scoped tables and plots are also refreshed in `results/catalogue/`.

If the selected data tree has no `spiking_diagnostics`, `overtopping_spiking_report/report_status.json` records `status=not_available`; the rest of the final-results pass continues normally.

Primary manuscript outputs are defined on the test evaluation split. Train/all diagnostic runs may write catalogue summaries under `results/catalogue/`, but they are not substituted into the 27/28 primary paper matrix.
