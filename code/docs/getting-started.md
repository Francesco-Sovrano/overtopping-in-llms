# Getting started

The code is organized as top-level Python packages beneath `code/`. Run commands from `code/` so imports such as `lib.*`, `pipeline.*`, `analysis.*`, `experiments.*`, and `poisoning.*` resolve without modifying `PYTHONPATH`.

## Repository assumptions

The implementation resolves these roots through `lib/project_paths.py`:

```text
CODE_ROOT     <repo>/code
PROJECT_ROOT  <repo>
```

Runtime artifacts normally live outside `code/`:

```text
<repo>/data/       persistent experiment/run artifacts
<repo>/cache/      regenerable caches
<repo>/results/    aggregate tables, figures, and reports
```

The supplied `code/` subtree is not a complete dependency bundle. `poisoning/requirements.txt` includes `-r ../../requirements.txt`, so installation through that file assumes a repository-level `requirements.txt` exists two levels above it.

From a complete repository checkout:

```bash
python3 -m pip install -r code/poisoning/requirements.txt
```

The pipeline also checks for `<repo>/.env/bin/activate` and activates it when present. This is optional behavior of the shell wrapper, not a requirement that the environment be named `.env`.

## Validate the code surface before model execution

These checks do not download or run model weights:

```bash
cd code
python3 -m compileall -q analysis experiments lib pipeline poisoning
python3 -m experiments.run_experiments --suite all --list
python3 -m analysis.generate_final_results --help
python3 -m poisoning.stage01_train_grammar --help
python3 -m poisoning.stage01_train_arithmetic --help
bash -n pipeline/_run_pipeline.sh
find poisoning/scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

A successful syntax or CLI check does not establish that datasets, model revisions, GPU memory, caches, or scientific outputs are valid.

## Inspect the standard experiment catalogue

The catalogue contains 39 unique configurations: 28 primary and 11 auxiliary.

```bash
python3 -m experiments.run_experiments --suite all --list
```

To see commands without executing the pipeline:

```bash
python3 -m experiments.run_experiments --suite paper-primary --dry-run
```

The catalogue defaults to held-out `test` evaluation. You can explicitly override every selected configuration with `--evaluation-split test|train|all`.

## Run a custom pipeline configuration

The shell wrapper accepts a task name and model name followed by pipeline controls:

```bash
bash pipeline/_run_pipeline.sh grammar_acceptability Qwen/Qwen2.5-1.5B-Instruct \
  --spectral_splits \
  --fast_anchoring \
  --eval_intervention mean-donor \
  --evaluation_split test
```

For paper configurations, prefer `experiments.run_experiments`; its `RunSpec` catalogue records the intended model, task, intervention, phase, circuit size, threshold, and evaluation settings together.

## Generate final results

The analysis orchestrator requires a primary profile:

```bash
python3 -m analysis.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile iclr-28
```

Use `--require-complete-new-metrics` when missing required metric sidecars should make the run fail instead of producing an audit that records the gaps.

## Start a checkpointed poisoning run

Inspect the current training CLI before selecting a dataset, model revision, poison rate, marker triple, and checkpoint schedule:

```bash
python3 -m poisoning.stage01_train_grammar --help
python3 -m poisoning.stage01_train_arithmetic --help
```

Poisoning experiments have additional scientific invariants: matched clean/poisoned checkpoints, fraction-0 trigger-neutrality checks, fixed causal cohorts, task-specific causal endpoints, and cache identities. Read [Poisoning study overview](poisoning-overview.md) before launching them.

## Next pages

- [Repository layout](repository-layout.md)
- [Core concepts](concepts.md)
- [Experiment catalogue](experiments.md)
- [Numbered pipeline](pipeline.md)
- [Analysis](analysis.md)
