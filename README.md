# Overtopping Phenomenology

This repository contains two causal-intervention studies of language-model behavior and a reporting pipeline for their derived analyses.

The **overtopping study** discovers candidate internal channels and measures held-out directional singleton effects, simultaneous-set effects, graded intervention responses, threshold diagnostics, and checkpoint trajectories. The **poisoning study** trains matched clean and poisoned trajectories, evaluates aligned checkpoints, freezes candidate sets from configured causal-localization endpoints, and evaluates poisoning-example detection and cross-seed behavior.

## Repository layout

```text
code/       Python and shell source plus technical documentation
data/       persistent model-backed scientific outputs
cache/      reusable computation caches
results/    derived analyses, audits, tables, and rendered figures
```

`data/` contains measured experiment artifacts. `cache/` contains reusable acceleration state keyed by the corresponding computation. `results/` is derived from compatible scientific inputs.

## Requirements

The repository targets Python 3.12. Model-backed stages require access to the configured model weights. Prompt-generation or classifier stages can additionally require Ollama, Groq, OpenAI, or Hugging Face credentials, depending on the selected configuration.

Install from the repository root:

```bash
./setup.sh
```

Manual installation:

```bash
python3.12 -m venv .env
. .env/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r code/studies/poisoning/requirements.txt
```

Credential configuration is described in [`code/docs/getting-started/credentials.md`](code/docs/getting-started/credentials.md).

## Overtopping study

Inspect the configured registry without running model inference:

```bash
./run_overtopping_experiments.sh --list
./run_overtopping_experiments.sh --dry-run
```

For this repository state, `--suite all --list` resolves to 50 unique scientific settings. The storage-contract check reports 29 input+output settings, 21 output-only settings, 39 mean-donor settings, 8 mean settings, and 3 mean-positional settings.

Validate persistent addressing:

```bash
cd code
python -m studies.overtopping.experiments.storage_contract
cd ..
```

Run the configured overtopping registry on the default held-out `test` split:

```bash
./run_overtopping_experiments.sh
```

Useful subset controls include:

```bash
./run_overtopping_experiments.sh --suite checkpoints --dry-run
./run_overtopping_experiments.sh --task arithmetic --evaluation-split test --dry-run
./run_overtopping_experiments.sh --phase pipeline
./run_overtopping_experiments.sh --phase analysis
```

The model-backed sequence is:

```text
01  prompts and answers
02  feature export
03  rule extraction
04  spectral sampling plan
05  circuit discovery
06  candidate and rule analysis
07  held-out singleton causal evaluation
07b graded intervention and continuous-margin diagnostics
07c temporal prefix intervention
07d temporal suffix intervention
     threshold-event diagnostics
08  simultaneous-set and interaction validation
```

Stage inputs, outputs, and controls are defined in [`code/docs/methods/pipeline.md`](code/docs/methods/pipeline.md).

## Poisoning study

Inspect the configured run matrix:

```bash
./run_poisoning_experiments.sh --dry-run
```

The default launcher resolves three Grammar runs using `Qwen/Qwen2-1.5B-Instruct` with training seeds 13, 37, and 101. The holdout seed defaults to 13.

Run the configured study:

```bash
./run_poisoning_experiments.sh
```

Run the reduced execution-path configuration:

```bash
POISONING_FAST_TEST=1 ./run_poisoning_experiments.sh
```

Poisoning outputs are stored under `data/poisoning/`. Protocol, configuration, stage layout, and detector controls are documented under [`code/docs/experiments/poisoning/`](code/docs/experiments/poisoning/README.md).

## Reporting

Generate derived analyses from existing scientific artifacts:

```bash
./generate_results.sh
```

The wrapper reads overtopping data from `data/`, poisoning data from `data/poisoning/`, and writes derived outputs under `results/`. Missing compatible metrics are reported by default. Require the configured completeness contract with:

```bash
ALLOW_INCOMPLETE_METRICS=false ./generate_results.sh
```

Supply an explicit threshold-diagnostics source with:

```bash
./generate_results.sh --spiking-source /absolute/path/to/threshold_diagnostics
```

Machine-readable analyses are stored under `results/analysis/`. Rendered tables and figures are stored under `results/paper/`.

## Analysis populations

Study membership is established before metric-specific filtering.

| Analysis | Population rule |
|---|---|
| Configured overtopping table | every unique setting in the selected registry manifest |
| RQ1 | configured settings with the required directional singleton metrics, analyzed by intervention phase |
| RQ2 | settings with a nonempty frozen candidate set and compatible simultaneous-set outputs, analyzed by replacement regime |
| RQ3 | configured settings with the artifacts required by each threshold, graded, margin, or temporal analysis |
| RQ4 | configured Pythia checkpoint trajectories and controlled poisoning trajectories |

A completed zero-candidate setting remains a measured setting where the corresponding zero-valued metric is defined. Missing artifacts are recorded as unavailable rather than converted to zeros.

## Documentation

Technical documentation is indexed at [`code/docs/README.md`](code/docs/README.md). It covers installation, architecture, scientific definitions, pipeline stages, experiment configuration, reporting contracts, and regeneration commands.
