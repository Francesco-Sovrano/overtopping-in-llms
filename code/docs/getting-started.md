# Getting started

This page takes a first-time reader from a fresh checkout to syntax validation, catalogue inspection, a custom causal-intervention command, and a poisoning dry run. Model-backed experiments can be expensive, so the first commands intentionally avoid launching full experiments.

## Requirements

The repository setup script expects Python 3.12. CUDA is recommended for model-backed circuit discovery and intervention experiments. Some analysis and reporting commands can run on CPU once their input artifacts already exist.

Optional external-provider calls require credentials supplied through environment variables. Do not store credentials in shell launchers or source files.

## Install

From repository root:

```bash
bash setup.sh
source .env/bin/activate
```

`setup.sh` creates `.env/`, upgrades packaging tools, installs `requirements.txt`, and pulls the default Ollama feature-proposal models when Ollama is installed.

For poisoning training, install the poisoning-specific runtime helpers as well:

```bash
python -m pip install -r code/studies/poisoning/requirements.txt
```

That file includes the root requirements and adds `accelerate`, `einops`, and `threadpoolctl`.

If Hugging Face models should use a non-default cache location, configure it before model loading, for example:

```bash
export HF_HOME=/path/to/hf-cache
export TRANSFORMERS_CACHE=/path/to/hf-cache
```

If an experiment uses an external provider:

```bash
export GROQ_API_KEY=...
# or
export OPENAI_API_KEY=...
```

## Understand the two experiment families

The repository has two study-specific workflows that share `code/core/` and `code/pipeline/`:

- **overtopping**: ordinary task/model configurations are selected from `code/studies/overtopping/experiments/`, executed by the shared pipeline, and aggregated by `code/studies/overtopping/analysis/`;
- **poisoning**: clean and poisoned checkpoints are trained and analyzed by `code/studies/poisoning/`, which calls the same shared pipeline for checkpoint causal discovery and ordinary-correctness controls.

See [Repository layout](repository-layout.md) and [Architecture](architecture.md) before changing import paths or moving directories.

## Validate the repository without running models

From repository root:

```bash
python3 -m compileall -q code
bash -n run_overtopping_experiments.sh
bash -n generate_results.sh
bash -n run_poisoning_experiments.sh
bash -n setup.sh
bash -n code/pipeline/run_pipeline.sh
find code/studies/poisoning/scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

Then inspect Python entry points from `code/`:

```bash
cd code
python3 -m studies.overtopping.experiments.run_experiments --suite paper-primary --list
python3 -m studies.overtopping.experiments.run_experiments --suite paper-auxiliary --list
python3 -m reporting.generate_final_results --help
python3 -m studies.poisoning.tasks.grammar --help
python3 -m studies.poisoning.tasks.arithmetic --help
```

Expected catalogue totals are 28 primary and 11 auxiliary configurations.

## Preview the overtopping catalogue

From repository root:

```bash
./run_overtopping_experiments.sh --suite paper-primary --dry-run
```

To inspect only selected rows, use the catalogue filters documented in [Experiment catalogue](overtopping-experiments.md), such as `--task`, `--model`, `--intervention`, `--mode`, and `--evaluation-split`.

The root launcher defaults to `--suite all` when no suite is explicitly supplied and uses the held-out `test` evaluation split unless overridden.

## Run one custom shared-pipeline configuration

From `code/`:

```bash
bash pipeline/run_pipeline.sh \
  grammar_acceptability \
  Qwen/Qwen2.5-1.5B-Instruct \
  --spectral_splits \
  --fast_anchoring \
  --eval_intervention mean-donor \
  --evaluation_split test
```

Use a catalogue `RunSpec` for manuscript configurations. Direct pipeline commands are most appropriate for exploratory or custom configurations.

## Generate final results from existing artifacts

From repository root:

```bash
./generate_results.sh
```

This command expects compatible artifacts under `data/`. It validates the primary overtopping matrix, audits required exact metrics, generates aggregate statistics and manuscript outputs, and includes poisoning cross-seed reporting when canonical poisoning runs are available.

Set `ALLOW_INCOMPLETE_NEW_METRICS=1` only when a partial analysis is intentional. Missing simultaneous or conditional intervention metrics are reported as missing rather than reconstructed from singleton summaries.

## Preview the poisoning matrix

From repository root:

```bash
./run_poisoning_experiments.sh --dry-run
```

The default matrix uses grammar and arithmetic, seeds `13,37,101`, Qwen2-1.5B task defaults, poison rate `0.1`, paired-counterfactual training, a uniform optimizer-step exposure schedule, and checkpoint fractions `0,0.1,0.25,0.5,0.75,1.0`.

A reduced smoke configuration can be selected with:

```bash
POISONING_FAST_TEST=1 ./run_poisoning_experiments.sh --dry-run
```

Read [Poisoning overview](poisoning-overview.md) before a full run and [Poisoning configuration](poisoning-configuration.md) before changing marker, poison-rate, checkpoint, or defence settings.

## Where outputs go

```text
data/       persistent experiment and poisoning run artifacts
cache/      regenerable caches
results/    aggregate reports, tables, figures, and audits
```

Do not place persistent scientific outputs under `code/`. Do not treat `data/` as a disposable cache.
