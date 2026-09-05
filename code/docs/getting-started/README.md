# Getting started

## Requirements

The repository targets Python 3.12. Model-backed experiments require access to the analyzed model weights. Feature proposal or classification steps may also require Ollama, Groq, or OpenAI, depending on the selected configuration.

The repository launchers normally run Hugging Face in offline mode. Ensure required model weights are available locally before starting long model-backed runs, or explicitly enable online access for the command that downloads them.

## Environment

From the repository root, use the repository setup script when available:

```bash
./setup.sh
```

A manual environment can be created with:

```bash
python3.12 -m venv .env
. .env/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r code/studies/poisoning/requirements.txt
```

Run Python modules from `code/` or add `code/` to `PYTHONPATH`:

```bash
cd code
python -m studies.overtopping.experiments.run_experiments --dry-run
```

## Provider credentials

Create `.secrets.env` in the repository root when the selected providers require credentials:

```bash
cat > .secrets.env <<'EOF_SECRETS'
export GROQ_API_KEY=""
export OPENAI_API_KEY=""
export HF_TOKEN=""
EOF_SECRETS
chmod 600 .secrets.env
```

Set only the credentials required by the selected configuration. Direct Python invocations require the variables to be exported in the invoking shell. See [Credentials](credentials.md).

## Runtime roots

```text
data/       persistent experiment outputs and provenance
cache/      reusable computation caches
results/    generated analyses, audits, tables, and figures
code/       source packages and documentation
```

Persistent experiment artifacts under `data/` define scientific computations and populations. Caches accelerate those computations but do not define analysis membership.

## Inspect the overtopping catalogue

From the repository root:

```bash
./run_overtopping_experiments.sh --list
./run_overtopping_experiments.sh --dry-run
```

Direct invocation from `code/`:

```bash
python -m studies.overtopping.experiments.run_experiments \
  --suite paper-primary \
  --task arithmetic \
  --model EleutherAI/pythia-1b \
  --dry-run
```

The catalogue contains 28 `paper-primary` settings and 11 `paper-auxiliary` settings. The paper-facing overtopping evaluation split is `test`. See [Overtopping experiment catalogue](../experiments/overtopping.md).

## Run overtopping experiments

From the repository root:

```bash
./run_overtopping_experiments.sh
```

The Python runner supports three execution phases:

```text
pipeline    execute model-backed pipeline stages
analysis    analyze existing persistent experiment outputs
all         execute both
```

The numbered pipeline is documented in [Pipeline](../methods/pipeline.md).

### RQ3 model-backed stages

RQ3 uses two model-backed analyses after held-out singleton evaluation:

```text
Stage 7b   graded agonist intervention
Stage 7c   threshold-event diagnostics
```

Both are enabled by default in `pipeline/run_pipeline.sh`:

```text
RUN_GRADED_AGONIST_INTERVENTION=true
RUN_THRESHOLD_EVENT_POSTHOC=true
```

The graded step requires these files in the Stage-7 statistics directory:

```text
flip_stats_by_neuron.csv
scores.csv
frozen_candidate_ranking.csv
```

It writes:

```text
<stage7 stats dir>/graded_agonist_intervention/
```

Threshold-event diagnostics write per-baseline and aggregate candidate/control data, including:

```text
aggregate_flip_stats.csv
aggregate_unit_tests.csv
aggregate_binned_curves.csv
aggregate_activation_flip_rows.csv
```

These aggregate files are the source for the threshold-shape reporting pipeline.

## Generate final results

From the repository root:

```bash
./generate_results.sh
```

Direct invocation from `code/`:

```bash
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile iclr-28
```

The reporting driver:

1. builds and audits the primary overtopping matrix;
2. computes RQ1/RQ2 statistics and manuscript outputs;
3. generates Pythia checkpoint figures;
4. aggregates poisoning results when available;
5. locates RQ3 threshold diagnostics and runs aggregate threshold-shape reporting;
6. aggregates per-run graded agonist outputs;
7. validates required manuscript products.

### RQ3 source resolution

The reporting driver can discover an RQ3 diagnostics directory under `data/` when it contains the expected aggregate payload. An explicit source can be provided with:

```bash
python -m reporting.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --primary-profile iclr-28 \
  --spiking-source /absolute/path/to/threshold_diagnostics
```

The repository wrapper may expose the same path through `SPIKING_SOURCE`:

```bash
SPIKING_SOURCE=/absolute/path/to/threshold_diagnostics ./generate_results.sh
```

A complete threshold-shape manuscript build requires the aggregate activation/flip rows in addition to aggregate flip statistics and unit tests.

### Current Figure 4 outputs

```text
results/paper/figures/04_rq3_spiking_cut/
├── fig4a_candidate_control_spiking_cut_summary.pdf
├── fig4b_threshold_shape_model_comparison_by_direction.pdf
├── fig4c_graded_agonist_dose_response.pdf
├── fig4s1_threshold_testability_by_condition.pdf
├── fig4s2_strength_matched_thresholdability.pdf
├── fig4s3_nested_tecs_lower_bound_ecdf.pdf
├── fig4s4_threshold_tail_response_by_direction.pdf
└── fig4s5_graded_agonist_single_crossing.pdf
```

Threshold diagnostics provide Figure 4a, Figure 4b, and S1–S4. Per-run graded agonist outputs provide Figure 4c and S5. See [RQ3](../research-questions/rq3-threshold-event.md) and [Figure map](../reporting/figures.md).

## Run poisoning experiments

Inspect the configured run family:

```bash
./run_poisoning_experiments.sh --dry-run
```

Execute it with:

```bash
./run_poisoning_experiments.sh
```

A reduced execution-path test is available through:

```bash
POISONING_FAST_TEST=1 ./run_poisoning_experiments.sh
```

One poisoning run uses the stage-numbered tree:

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

See [Poisoning protocol](../experiments/poisoning/README.md), [Configuration](../experiments/poisoning/configuration.md), and [Outputs](../experiments/poisoning/outputs.md).

## Inspect results

Human-facing products:

```text
results/paper/figures/
results/paper/tables/
```

Machine-readable analyses and audits:

```text
results/analysis/
```

Population and output audits are written under:

```text
results/analysis/reproducibility/
```

RQ3 has additional audits under:

```text
results/analysis/rq3_threshold_event/spiking_diagnostics/
```

## Cache policy

Do not delete model-backed caches to regenerate figures or statistical summaries. Reporting products under `results/` are derived from persistent inputs and can be regenerated independently. Delete or invalidate a cache only when the scientific configuration, model state, intervention semantics, or cached row population has changed.

See [Operations](../operations/README.md) for targeted regeneration commands.
