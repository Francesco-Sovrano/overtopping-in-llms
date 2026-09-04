# Getting started

## Prerequisites

The repository targets Python 3.12. Model-backed experiments require local access to the analyzed model weights. Prompt generation and classification may use Ollama, Groq, or OpenAI depending on configuration.

The top-level experiment launchers set `HF_HUB_OFFLINE=1` unless explicitly overridden, so model weights should normally be available locally before long runs begin.

## Install the environment

From the repository root:

```bash
./setup.sh
```

Equivalent manual installation:

```bash
python3.12 -m venv .env
. .env/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r code/studies/poisoning/requirements.txt
```

Direct Python modules should be run from `code/` or with `code/` on `PYTHONPATH`.

## Configure provider credentials

Create a local secrets file when a configured provider requires credentials:

```bash
cat > .secrets.env <<'EOF_SECRETS'
export GROQ_API_KEY=""
export OPENAI_API_KEY=""
export HF_TOKEN=""
EOF_SECRETS
chmod 600 .secrets.env
```

Fill only the variables required by the selected providers.

The top-level launchers source `.secrets.env` when it exists. Direct module invocations require the variables to be exported in the invoking shell. See [API keys and provider credentials](credentials.md).

## Runtime roots

```text
data/       persistent experiment outputs and provenance
cache/      regenerable computation caches
results/    generated analysis and manuscript products
code/       source packages and documentation
```

Scientific populations are determined by catalogue entries, task manifests, persisted row identities, and analysis contracts. Cache contents do not define analysis membership.

## Inspect the overtopping catalogue

```bash
./run_overtopping_experiments.sh --list
./run_overtopping_experiments.sh --dry-run
```

The repository-level launcher selects the union of the primary and auxiliary overtopping suites unless `--suite` is supplied. Its paper-facing evaluation split defaults to `test`.

Useful direct filters from `code/` include:

```bash
python -m studies.overtopping.experiments.run_experiments \
  --suite paper-primary \
  --task arithmetic \
  --model EleutherAI/pythia-1b \
  --dry-run
```

The Python runner supports filters for task, model, replacement intervention, intervention mode, execution phase, and evaluation split.

## Run overtopping experiments

From the repository root:

```bash
./run_overtopping_experiments.sh
```

The launcher writes `results/configured_experiments.json`, executes the selected pipeline runs, and requests the `iclr-28` manuscript profile when the evaluation split is `test`.

Execution phases can be selected with the Python runner:

```text
pipeline    execute the causal pipeline
analysis    analyze existing experiment outputs
all         execute both
```

See [Overtopping experiment catalogue](../experiments/overtopping.md) and [Numbered causal pipeline](../methods/pipeline.md).

## Build RQ3 threshold-event diagnostics

RQ3 uses model-backed high-N candidate/control diagnostics in addition to ordinary Stage-7 singleton outputs. From `code/`:

```bash
python -m studies.overtopping.analysis.rebuild_spiking_diagnostics \
  --primary-table ../results/analysis/primary_matrix/tables/primary_table.csv \
  --data-root ../data \
  --population-scope primary+supplementary
```

The command requires all primary RQ3 runs. Auxiliary runs enter the RQ3 analysis only when their exact diagnostic artifacts are complete. Candidate membership is restricted to the baseline subset in which each agonist was discovered.

See [RQ3 threshold-event analysis](../research-questions/rq3-threshold-event.md).

## Inspect and run poisoning experiments

```bash
./run_poisoning_experiments.sh --dry-run
./run_poisoning_experiments.sh
```

For an execution-path test with reduced data sizes:

```bash
POISONING_FAST_TEST=1 ./run_poisoning_experiments.sh
```

One poisoning run is organized as:

```text
data/poisoning/<task>/<run>/
├── 01_training_checkpoints/
├── 02_evaluation_cohorts/
├── 03_checkpoint_causal_discovery/
├── 04_condition_comparisons/
├── 05_behavior_trajectories/
├── 06_circuit_overlap_analysis/
└── 07_poisoning_example_detection/
```

See [Poisoning protocol](../experiments/poisoning/), [Poisoning configuration](../experiments/poisoning/configuration.md), and [Poisoning outputs](../experiments/poisoning/outputs.md).

## Generate final results

```bash
./generate_results.sh
```

Equivalent direct invocation from `code/`:

```bash
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile iclr-28
```

The wrapper supports these environment controls:

```text
SPIKING_SOURCE=<absolute diagnostic root>   use an explicit RQ3 diagnostic source
REQUIRE_CMC=1                              require optional CMC interaction validation
ALLOW_INCOMPLETE_METRICS=1                 allow reporting with missing required metrics
```

The default reporting path requires complete primary metrics. `ALLOW_INCOMPLETE_METRICS=1` is intended for diagnostic inspection rather than complete manuscript generation.

## Inspect generated outputs

Human-facing outputs are under:

```text
results/paper/figures/
results/paper/tables/
```

Machine-readable tables and audits are under:

```text
results/analysis/
```

The main reproducibility checks are written under `results/analysis/reproducibility/`. RQ3 has separate population and threshold-shape audits under `results/analysis/rq3_threshold_event/`.

## Cache handling

Model-backed caches can be expensive to rebuild. Delete caches only when the scientific configuration they encode changes. Procedures for regenerating RQ3 derived outputs without deleting reusable model evaluations are documented in [Operations and troubleshooting](../operations/).
