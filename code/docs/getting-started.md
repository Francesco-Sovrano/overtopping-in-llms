# Getting started

The implementation requires Python 3.12. The repository-level setup script creates a virtual environment at `.env/`, installs `requirements.txt`, and optionally downloads the default Ollama feature-proposal models when Ollama is installed.

## Install

From repository root:

```bash
bash setup.sh
source .env/bin/activate
```

For a manual installation, create a Python 3.12 environment and install the repository requirements:

```bash
python3.12 -m venv .env
source .env/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

`code/poisoning/requirements.txt` includes the repository-level requirements and can also be used when working specifically on poisoning experiments.

## Runtime roots

The code resolves these roots through `lib/project_paths.py`:

```text
CODE_ROOT     <repo>/code
PROJECT_ROOT  <repo>
```

Runtime state normally lives outside `code/`:

```text
<repo>/data/       persistent experiment and run artifacts
<repo>/cache/      regenerable caches
<repo>/results/    aggregate tables, figures, audits, and reports
<repo>/.env/       virtual environment created by setup.sh
```

Use standard Hugging Face variables such as `HF_HOME` or `TRANSFORMERS_CACHE` when model caches must live on another disk.

## Validate before model execution

From repository root:

```bash
python3 -m compileall -q code
bash -n run_experiments.sh
bash -n generate_results.sh
bash -n run_poisoning_experiments.sh
bash -n setup.sh
bash -n code/pipeline/_run_pipeline.sh
find code/poisoning/scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

Then inspect the Python CLIs from `code/`:

```bash
cd code
python3 -m experiments.run_experiments --suite paper-primary --list
python3 -m experiments.run_experiments --suite paper-auxiliary --list
python3 -m analysis.generate_final_results --help
python3 -m poisoning.tasks.grammar --help
python3 -m poisoning.tasks.arithmetic --help
```

The expected catalogue totals are 28 primary and 11 auxiliary configurations. These checks validate syntax and command surfaces; they do not validate datasets, model revisions, accelerator memory, or scientific results.

## Inspect or run the standard catalogue

From repository root:

```bash
./run_experiments.sh --list
./run_experiments.sh --suite paper-primary --dry-run
./run_experiments.sh --suite paper-primary
```

The root launcher defaults to the held-out `test` split. Use `--evaluation-split train` or `--evaluation-split all` only when that evaluation population is intentional. Primary manuscript generation is defined only for the test split.

From `code/`, the equivalent catalogue module is:

```bash
python3 -m experiments.run_experiments --suite all --list
```

## Run one custom pipeline

A direct pipeline command starts with a task identifier and model identifier:

```bash
bash pipeline/_run_pipeline.sh \
  grammar_acceptability \
  Qwen/Qwen2.5-1.5B-Instruct \
  --spectral_splits \
  --fast_anchoring \
  --eval_intervention mean-donor \
  --evaluation_split test
```

Use `bash pipeline/_run_pipeline.sh --help` for the live option list. For standard manuscript configurations, prefer `experiments.run_experiments` so the intended task/model/phase/intervention settings remain coupled in one `RunSpec`.

## Generate final results

From repository root:

```bash
./generate_results.sh
```

Directly from `code/`:

```bash
python3 -m analysis.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --poisoning-root ../data/poisoning \
  --primary-profile iclr-28 \
  --require-complete-new-metrics
```

The only supported primary profile is `iclr-28`, which requires exactly 28 primary rows. Missing exact metrics are reported in the completeness audit; simultaneous or conditional intervention effects are not reconstructed from singleton unions.

## Start a poisoning workflow

Preview the repository-root matrix without loading a model:

```bash
cd ..
./run_poisoning_experiments.sh --dry-run
```

The default matrix covers grammar and arithmetic with `Qwen/Qwen2-1.5B-Instruct` and seeds `13,37,101`. Its marker defaults are one space for control, `[id=74291]` for trigger, and two spaces for sham. Set `CONTROL_MARKER`, `TRIGGER_MARKER`, and `SHAM_MARKER` to use different single-line marker strings. Marker values are passed through unchanged; quote shell values so intentional leading/trailing whitespace is preserved.

For direct task training from `code/`:

```bash
python3 -m poisoning.tasks.grammar --help
python3 -m poisoning.tasks.arithmetic --help
```

For the shared shell driver:

```bash
POISONING_TASK=grammar DRY_RUN=1 bash poisoning/scripts/run_checkpoint_ft.sh
```

Read [Poisoning study overview](poisoning-overview.md) and [Poisoning configuration](poisoning-configuration.md) before a full run; marker identity, checkpoint matching, poison exposure, causal endpoints, and cache identity are part of the experimental protocol.
