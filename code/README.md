# Code reference

This directory contains the executable Python and shell implementation for the causal-intervention experiments, downstream analysis, and checkpointed poisoning studies.


## Documentation

Start with [`docs/index.md`](docs/index.md). The documentation is organized for first-time readers and covers setup, repository layout, core concepts, the 39-run standard experiment catalogue, pipeline stages, final analysis, EAP/EAP-IG, and the checkpointed poisoning protocol.

Key pages:

- [`docs/getting-started.md`](docs/getting-started.md) — installation assumptions, validation, and first commands.
- [`docs/experiments.md`](docs/experiments.md) — standard run catalogue and filters.
- [`docs/pipeline.md`](docs/pipeline.md) — numbered causal-intervention pipeline.
- [`docs/analysis.md`](docs/analysis.md) — final-result generation and metric audits.
- [`docs/poisoning-overview.md`](docs/poisoning-overview.md) — poisoning study endpoints, markers, and quick start.
- [`docs/poisoning-protocol.md`](docs/poisoning-protocol.md) — matched training and causal protocol.

## Directory layout

```text
code/
├── analysis/       final-result generation and diagnostic analyses
├── docs/           expanded Markdown documentation
├── experiments/    explicit catalogue of standard non-poisoning runs
├── lib/            shared task, modeling, intervention, statistics, and EAP code
├── pipeline/       numbered causal-discovery/intervention pipeline
└── poisoning/      checkpointed poisoning, controls, and protection experiments
```

`code/` itself is not a Python package. Its children are top-level packages. Run module entry points with the current working directory set to `code/` so imports such as `lib.*`, `pipeline.*`, and `analysis.*` resolve correctly:

```bash
cd code
python3 -m experiments.run_experiments --help
python3 -m analysis.generate_final_results --help
python3 -m poisoning.stage01_train_grammar --help
```

## Repository paths

`lib/project_paths.py` resolves:

```text
CODE_ROOT     <repo>/code
PROJECT_ROOT  <repo>
```

The implementation therefore expects runtime artifacts outside this directory, normally under:

```text
<repo>/data/       experiment outputs and persistent run artifacts
<repo>/cache/      regenerable caches
<repo>/results/    aggregate tables, figures, and reports
```

Some workflows also look for a repository-root `.env/` virtual environment. The pipeline activates `<repo>/.env/bin/activate` automatically when that file exists.

## Installation

The supplied poisoning dependency file is `poisoning/requirements.txt`. It begins with:

```text
-r ../../requirements.txt
```

so it is intended to be used inside the complete repository, where `<repo>/requirements.txt` provides the core environment. It then adds `accelerate`, `einops`, and `threadpoolctl`. The `code/` subtree alone does not contain the referenced root requirements file, so it is not a self-contained dependency bundle.

From a complete repository checkout:

```bash
python3 -m pip install -r code/poisoning/requirements.txt
```

## Main workflows

### Standard non-poisoning experiments

The executable catalogue is `experiments/run_experiments.py`. It currently contains 39 unique configurations: 28 primary and 11 auxiliary. To inspect them without launching experiments:

```bash
cd code
python3 -m experiments.run_experiments --suite all --list
```

Each selected configuration is translated into a call to `pipeline/_run_pipeline.sh`. See `experiments/README.md` for catalogue membership and filtering, and `pipeline/README.md` for pipeline stages and controls.

### Final analysis

The main analysis orchestrator is:

```bash
cd code
python3 -m analysis.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile iclr-28
```

Use `--require-complete-new-metrics` when a final result set must fail rather than tolerate missing required sidecars. See `analysis/README.md` for the numbered analysis stages and diagnostic utilities.

### Poisoning experiments

The poisoning package trains matched clean and poisoned checkpoints, prepares fixed causal cohorts, runs checkpoint-specific causal discovery, aggregates behavioral/circuit trajectories, and provides inference-time and training-time intervention controls. The directly executable task entry points are:

```bash
cd code
python3 -m poisoning.stage01_train_grammar --help
python3 -m poisoning.stage01_train_arithmetic --help
```

The package also provides shell drivers under `poisoning/scripts/`. See `poisoning/README.md` before running these experiments because marker defaults, checkpoint matching, causal endpoints, and cache roots are scientifically significant.

## Package responsibilities

- `experiments/`: explicit `RunSpec` catalogue, filters, command construction, and execution.
- `pipeline/`: stages 1–7 for prompt/model I/O, features, rules, sampling, circuit discovery, candidate selection, singleton evaluation, and optional interaction validation.
- `analysis/`: primary-matrix normalization, audits, exact set metrics, interaction validation, figures, tables, reports, and cross-run/model diagnostics.
- `lib/`: shared task specifications, prompting/caching, feature representation, TransformerLens model/intervention utilities, statistics, and the internal EAP/EAP-IG implementation.
- `poisoning/`: grammar/arithmetic poisoning tasks, matched checkpoint training, trigger-lift and ordinary-correctness causal analyses, trajectory aggregation, cumulative ablation, and training-protection controls.

## Validation without running models

The following checks validate imports and command-line surfaces that do not require model execution:

```bash
cd code
python3 -m compileall -q analysis experiments lib pipeline poisoning
python3 -m experiments.run_experiments --suite all --list
python3 -m analysis.generate_final_results --help
bash -n pipeline/_run_pipeline.sh
find poisoning/scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

Successful syntax/CLI checks do not validate scientific results. Full validation requires compatible datasets, model weights, generated artifacts, and provenance-consistent evaluation outputs.
