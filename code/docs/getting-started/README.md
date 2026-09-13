# Getting started

## Requirements

The project targets Python 3.12. Model-backed execution requires access to the configured model weights. Provider-backed prompt or classifier stages require the corresponding Ollama, Groq, OpenAI, or Hugging Face setup.

## Install the environment

From the repository root:

```bash
./setup.sh
```

`setup.sh` creates `.env/`, installs the root requirements and poisoning-specific requirements, and pulls `gemma3:27b` and `qwen3:4b` when the `ollama` executable is available.

Equivalent manual setup:

```bash
python3.12 -m venv .env
. .env/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r code/studies/poisoning/requirements.txt
```

Run Python modules from `code/` or include `code/` on `PYTHONPATH`.

## Configure credentials

The top-level launchers source `${SECRETS_FILE:-<repo>/.secrets.env}` when the file exists. A minimal file is:

```bash
cat > .secrets.env <<'EOF_SECRETS'
export GROQ_API_KEY=""
export OPENAI_API_KEY=""
export HF_TOKEN=""
EOF_SECRETS
chmod 600 .secrets.env
```

Only configured providers require credentials. See [API keys and provider credentials](credentials.md).

## Artifact roots

```text
data/       persistent model-backed scientific outputs
cache/      reusable computation caches
results/    derived analyses, audits, tables, and rendered figures
code/       implementation and documentation
```

The reporting driver reads scientific artifacts from `data/` and writes derived outputs to `results/`.

## Inspect the overtopping registry

From the repository root:

```bash
./run_overtopping_experiments.sh --list
./run_overtopping_experiments.sh --dry-run
```

The current `all` selection contains 50 unique scientific settings after deduplication. Current suite selections contain:

```text
mean-donor     30
6-7b-models     3
mean            8
checkpoints    12
```

Three final Pythia-1B input+output settings occur in both `mean-donor` and `checkpoints`; therefore the unique union is 50 rather than 53.

Validate persistent path uniqueness and the registry fingerprint:

```bash
cd code
python -m studies.overtopping.experiments.storage_contract
cd ..
```

For the current registry the contract reports:

```text
setting_count        50
I+O                   29
Out                   21
mean-donor            39
mean                   8
mean-positional        3
```

## Run overtopping experiments

Run all configured settings on the default `test` evaluation split:

```bash
./run_overtopping_experiments.sh
```

Run one suite:

```bash
./run_overtopping_experiments.sh --suite checkpoints
```

Filter by scientific fields:

```bash
./run_overtopping_experiments.sh \
  --task arithmetic \
  --model Qwen/Qwen2-1.5B-Instruct \
  --evaluation-split test \
  --dry-run
```

Select execution phase:

```text
--phase pipeline    model-backed stages
--phase analysis    analysis of existing persistent outputs
--phase all         both phases
```

The numbered model-backed stages are documented in [Causal-intervention pipeline](../methods/pipeline.md).

## RQ3 model-backed extensions

After Stage 07 singleton evaluation, optional RQ3 computations include:

```text
Stage 07b  graded agonist intervention and endpoint-margin recording
Stage 07c  temporal prefix intervention
Stage 07d  temporal suffix intervention
Post-07    threshold-event diagnostics
Stage 08   interaction and optional preemption validation
```

Graded intervention is controlled by `RUN_GRADED_AGONIST_INTERVENTION`. Threshold diagnostics are controlled by `RUN_THRESHOLD_EVENT_POSTHOC`. Temporal cutoff and suffix sweeps use their corresponding `RUN_TEMPORAL_*` controls. The exact inputs and output directories are defined in [Pipeline](../methods/pipeline.md) and [RQ3](../research-questions/rq3-threshold-event.md).

## Generate derived results

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

The driver constructs the configured study table, audits metric availability, computes RQ-specific aggregate statistics, aggregates threshold and graded analyses when their source artifacts are available, processes poisoning outputs when enabled, and writes derived products under `results/`.

The wrapper permits incomplete metric coverage by default and records missing coverage in reproducibility audits. Require the configured completeness contract with:

```bash
ALLOW_INCOMPLETE_METRICS=false ./generate_results.sh
```

CMC completeness is optional by default. Set `REQUIRE_CMC=true` to require CMC and paired conditional-null outputs for applicable settings.

### Explicit threshold-diagnostics source

```bash
./generate_results.sh --spiking-source /absolute/path/to/threshold_diagnostics
```

The same option is accepted by `python -m reporting.generate_final_results`.

## Run poisoning experiments

Inspect the default run matrix:

```bash
./run_poisoning_experiments.sh --dry-run
```

The current default matrix contains three Grammar runs with model `Qwen/Qwen2-1.5B-Instruct` and training seeds `13,37,101`.

Execute the study:

```bash
./run_poisoning_experiments.sh
```

Reduced execution-path configuration:

```bash
POISONING_FAST_TEST=1 ./run_poisoning_experiments.sh
```

Explicit circuit-discovery alternatives:

```bash
./run_poisoning_experiments.sh --skip-circuit-discovery
./run_poisoning_experiments.sh --full-ablation-if-no-circuit
```

These two options are mutually exclusive. Configuration and output layout are documented in [Poisoning configuration](../experiments/poisoning/configuration.md) and [Poisoning outputs](../experiments/poisoning/outputs.md).

## Inspect derived outputs

Machine-readable analyses and audits:

```text
results/analysis/
results/analysis/reproducibility/
```

Rendered output directories:

```text
results/paper/figures/
results/paper/tables/
```

## Cache policy

Reuse caches when their stored metadata matches the requested computation. Aggregate reporting changes do not alter experiment identity or cache keys. Recovery commands for missing derived or model-backed artifacts are listed in [Operations and regeneration](../operations/README.md).
