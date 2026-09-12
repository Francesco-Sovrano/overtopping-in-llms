# Overtopping Phenomenology

This repository implements two causal-intervention studies of language-model behavior and the reporting pipeline used to aggregate their results.

- **Overtopping study:** discovers high-leverage internal channels, evaluates their directional singleton effects on held-out examples, measures simultaneous-set composition, and studies graded intervention response across model states.
- **Controlled grammar-training study:** trains matched clean and poisoned grammar trajectories, evaluates behavior and causal organization at aligned checkpoints, and measures whether training examples associated with the hidden objective can be identified from causal and parameter-update signals.

The code separates model-backed scientific artifacts from reusable caches and from regenerated reports. A first-time user should treat these roots differently:

```text
code/       Python and shell source plus documentation
data/       persistent model-backed scientific outputs
cache/      reusable computation caches
results/    derived audits, tables, statistics, and figures
```

`data/` defines measured experiment outputs. `cache/` accelerates compatible computations. `results/` can be regenerated from compatible scientific inputs.

## Requirements and installation

The project targets Python 3.12. Model-backed stages also require access to the configured model weights and, for provider-backed feature generation, the corresponding provider credentials.

From the repository root:

```bash
./setup.sh
```

`setup.sh` creates `.env/`, installs `requirements.txt` and the poisoning-specific requirements, and pulls the default Ollama feature models when the `ollama` executable is available.

Provider setup is documented in [`code/docs/getting-started/credentials.md`](code/docs/getting-started/credentials.md).

## Validate the installation and study registry

Inspect the overtopping registry without running model inference:

```bash
./run_overtopping_experiments.sh --list
./run_overtopping_experiments.sh --dry-run
```

Optionally audit the current registry for persistent-address collisions and record its fingerprint:

```bash
cd code
python -m studies.overtopping.experiments.storage_contract
cd ..
```

The overtopping registry is explicit rather than factorial and has no required population size. Add or remove `RunSpec` entries as needed; `--list` shows the current total and phase/replacement breakdown. The four execution suites are organizational metadata, not a population-size contract.

`EleutherAI/pythia-1b` is the final/all-steps member of each configured checkpoint trajectory. Arithmetic is not checkpointed. Current task/model/phase/replacement coverage is documented in [`code/docs/experiments/overtopping.md`](code/docs/experiments/overtopping.md).

## Run the overtopping study

Run all configured overtopping settings on the held-out `test` split:

```bash
./run_overtopping_experiments.sh
```

Useful inspection and subset controls include:

```bash
./run_overtopping_experiments.sh --dry-run
./run_overtopping_experiments.sh --suite checkpoints --dry-run
./run_overtopping_experiments.sh --task arithmetic --evaluation-split test --dry-run
```

The numbered model-backed workflow is:

```text
01 prompts and answers
02 feature export
03 rule extraction
04 spectral sampling plan
05 circuit discovery
06 candidate/rule analysis
07 held-out singleton causal evaluation
07b graded intervention and continuous-margin diagnostics
07c threshold-event diagnostics
08 simultaneous-set and interaction validation
```

See [`code/docs/methods/pipeline.md`](code/docs/methods/pipeline.md) for stage inputs, outputs, and controls.

## Run the poisoning study

Inspect the configured poisoning run matrix:

```bash
./run_poisoning_experiments.sh --dry-run
```

Run the configured poisoning study:

```bash
./run_poisoning_experiments.sh
```

Run the reduced behavior-focused smoke path:

```bash
POISONING_FAST_TEST=1 ./run_poisoning_experiments.sh
```

Poisoning outputs are stored under `data/poisoning/` and do not enter the RQ1–RQ3 overtopping populations. The protocol, configuration, and artifact layout are documented under [`code/docs/experiments/poisoning/`](code/docs/experiments/poisoning/README.md).

## Generate analyses and manuscript products

From the repository root:

```bash
./generate_results.sh
```

The reporting driver reads model-backed inputs from `data/` and writes derived outputs under `results/`. By default it reports every analysis that can be computed from the compatible configured inputs and records missing coverage in reproducibility audits. To require complete required metrics for the configured study:

```bash
ALLOW_INCOMPLETE_METRICS=false ./generate_results.sh
```

To supply an explicit threshold-diagnostics source:

```bash
./generate_results.sh --spiking-source /absolute/path/to/threshold_diagnostics
```

Human-facing outputs are written under:

```text
results/paper/figures/
results/paper/tables/
```

Machine-readable analysis products and reproducibility audits are written under:

```text
results/analysis/
```

## Analysis populations

The configured study manifest is created before metric-specific filtering. Missing measurements remain missing measurements; they do not silently remove a configured setting.

| Analysis | Population rule |
|---|---|
| Configured study table | every setting in the current configured manifest |
| RQ1 | every configured setting with the required metric, analyzed separately for input+output and output-only phases |
| RQ1 final-snapshot sensitivity | 29 unique final-snapshot task×model×phase cells |
| RQ2 | settings where simultaneous-set composition is applicable and available, analyzed separately by replacement regime |
| RQ3 | configured settings with the compatible threshold/graded artifacts required by each analysis |
| RQ4 | configured Pythia checkpoint trajectories and controlled poisoning trajectories |

## Documentation

Start with [`code/docs/README.md`](code/docs/README.md). It provides a reading order for installation, architecture, methods, study design, research questions, reporting, and regeneration.
