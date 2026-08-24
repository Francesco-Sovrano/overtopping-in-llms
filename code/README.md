# Code guide

The `code/` directory contains the executable Python packages and shell orchestration for the causal channel-intervention repository. Direct Python module commands should be run from this directory so imports such as `lib.*`, `analysis.*`, `experiments.*`, `pipeline.*`, and `poisoning.*` resolve normally.

## Start here

From repository root, install the Python 3.12 environment with:

```bash
bash setup.sh
source .env/bin/activate
```

Then enter the implementation directory:

```bash
cd code
```

Validate syntax and inspect the command surfaces without running models:

```bash
python3 -m compileall -q analysis experiments lib pipeline poisoning
python3 -m experiments.run_experiments --suite all --list
python3 -m analysis.generate_final_results --help
python3 -m poisoning.tasks.grammar --help
python3 -m poisoning.tasks.arithmetic --help
bash -n pipeline/_run_pipeline.sh
find poisoning/scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

## Packages

### `experiments/`

Defines the explicit standard non-poisoning catalogue. The catalogue contains 28 `paper-primary` and 11 `paper-auxiliary` configurations. `run_experiments.py` handles suite selection, exact-value filters, evaluation-split overrides, dry runs, execution, and optional primary manuscript generation.

```bash
python3 -m experiments.run_experiments --suite paper-primary --list
python3 -m experiments.run_experiments --suite paper-auxiliary --list
```

The only supported primary profile is `iclr-28`.

### `pipeline/`

Contains numbered stages 1–7 and `_run_pipeline.sh`, which coordinates one task/model configuration. The pipeline materializes model behavior, constructs features and rules, selects representative rows, discovers candidate causal channels, ranks them, and evaluates singleton and optional simultaneous-set effects.

Example:

```bash
bash pipeline/_run_pipeline.sh \
  grammar_acceptability \
  Qwen/Qwen2.5-1.5B-Instruct \
  --spectral_splits \
  --fast_anchoring \
  --eval_intervention mean-donor \
  --evaluation_split test
```

For manuscript configurations, use the `experiments` catalogue rather than assembling commands manually.

### `analysis/`

Aggregates experiment artifacts, validates the 28-setting primary matrix, audits exact metric availability, computes primary statistics, and writes manuscript tables, figures, poisoning reports, and diagnostic reports.

```bash
python3 -m analysis.generate_final_results \
  --data-root ../data \
  --results-root ../results \
  --poisoning-root ../data/poisoning \
  --primary-profile iclr-28 \
  --require-complete-new-metrics
```

### `lib/`

Provides shared task specifications, prompting and cache helpers, model loading, activation replacement and ablation, intervention statistics, feature/rule utilities, spectral sampling, and the internal EAP/EAP-IG implementation. Standard task modules are under `lib/tasks/`.

### `poisoning/`

Implements matched clean/poisoned checkpoint training for grammar and arithmetic, fixed evaluation cohorts, trigger-lift and ordinary-correctness causal endpoints, checkpoint trajectory aggregation, cumulative suppression and specificity tests, matrix aggregation, and training-time protection controls.

Canonical task CLIs:

```bash
python3 -m poisoning.tasks.grammar --help
python3 -m poisoning.tasks.arithmetic --help
```

Shared shell drivers are under `poisoning/scripts/`. Marker values are configurable raw one-line strings. The default launcher uses one space for control, `[id=74291]` for trigger, and two spaces for sham; quote whitespace markers when setting them in the shell.

## Runtime roots

The project root is the parent of `code/`. Runtime state is intentionally outside the source tree:

```text
../data/       persistent experiment/run artifacts
../cache/      regenerable caches
../results/    final aggregate outputs
../.env/       optional repository virtual environment
```

`lib/project_paths.py` is the shared path source of truth. Poisoning checkpoint manifests use run-local checkpoint identities rather than absolute paths.

## Documentation

Read [`docs/index.md`](docs/index.md) for the full documentation map. The main pages cover setup, repository layout, concepts, experiment catalogue, pipeline stages, analysis, poisoning protocol/configuration/output layout, EAP/EAP-IG, interpretation limits, and troubleshooting.
