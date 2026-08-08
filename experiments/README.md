# Experiment catalogue and execution

`experiments/run_experiments.py` owns the complete executable catalogue for standard non-poisoning experiments. It declares the task, model, replacement baseline, intervention phase, circuit parameters, thresholds, batching, and channel restrictions for each run. `experiments/execution.py` contains reusable `RunSpec` validation, filtering, path construction, and pipeline command generation; it does not contain a second catalogue.

## Catalogue contents

The catalogue has two suites:

| Suite | Count | Definition |
|---|---:|---|
| `phenomenology` | 121 | task/baseline/phase sweep over Qwen2/Qwen2.5/Pythia small models plus the explicit Qwen2-1.5B I+O NLI setting |
| `large-models` | 6 | Qwen2-7B and Pythia-6.9B on arithmetic, HANS NLI, and jailbreak |
| **Total** | **127** | all standard non-poisoning configurations |

`--suite all` expands to both suites. With no `--suite`, the driver also selects both suites.

### Phenomenology suite

The primary model group is:

```text
Qwen/Qwen2.5-1.5B-Instruct
EleutherAI/pythia-1b
EleutherAI/pythia-1b@step0
EleutherAI/pythia-1b@step48000
EleutherAI/pythia-1b@step96000
```

The task identifiers are:

```text
random_fsm
grammar_acceptability
hans_nli
arithmetic
bon_jailbreaking
```

The intervention baselines are:

```text
mean-donor
mean
zero
```

The phases are `standard` (`I+O`) and `decode-only` (`Out`). The Pythia checkpoint models are configured only for `random_fsm`, `grammar_acceptability`, and `hans_nli`. Qwen2-1.5B has the six arithmetic combinations across the three baselines and two phases, plus one `hans_nli / mean-donor / standard` configuration used by the 28-setting primary profile.

All phenomenology `RunSpec` objects use neuron-level circuits, `circuit_size=200000`, `min_flip_rate=0.3`, and `max_circuits=1`. The default batch size is 32. Arithmetic uses `z_thresh=10` for Qwen models and `z_thresh=5` for other arithmetic configurations; non-arithmetic configurations use `z_thresh=-1`.

### Large-model suite

For each of

```text
Qwen/Qwen2-7B-Instruct
EleutherAI/pythia-6.9b
```

the suite contains:

- arithmetic with `mean-positional`, `decode-only`, MLP-neuron restriction, `circuit_size=100000`, and up to 5 circuits;
- HANS NLI with `mean-positional`, `standard`, MLP-neuron restriction, and `circuit_size=100000`;
- jailbreak with `mean-positional`, `decode-only`, MLP-neuron restriction, and `circuit_size=100000`.

## Root launcher

From repository root:

```bash
./run_experiments.sh
```

The launcher activates `.env/` when present, selects `--suite all`, writes under `<repo>/results`, and uses `test` as the default evaluation split. For `test`, it also requests the primary manuscript export using `PRIMARY_PROFILE` (default `iclr-28`). For `train` or `all`, it runs the catalogue and catalogue analysis but skips the test-specific primary manuscript export.

Examples:

```bash
./run_experiments.sh --dry-run
./run_experiments.sh --task arithmetic
PRIMARY_PROFILE=legacy-27 ./run_experiments.sh
./run_experiments.sh --evaluation-split train
EVALUATION_SPLIT=all ./run_experiments.sh
```

Arguments are forwarded to the Python driver before the fixed suite/results/profile options. Filters still apply to the selected catalogue, but a filtered run generally cannot produce a complete 27/28 primary manuscript table. Use the Python driver without `--generate-primary-manuscript` for partial/debug experiment subsets.

## Python driver

Python entry points are executed as modules from the repository root (`python3 -m ...`). No runtime `sys.path` modification is used.


List configurations:

```bash
python3 -m experiments.run_experiments --suite all --list
```

Run only the model pipeline:

```bash
python3 -m experiments.run_experiments --suite phenomenology --phase pipeline
```

Analyze configured outputs without launching model stages:

```bash
python3 -m experiments.run_experiments --suite all --phase analysis
```

Run both:

```bash
python3 -m experiments.run_experiments --suite all --phase all
```

Exact comma-separated filters are supported for task, model, intervention, and mode. The evaluation split is selected separately and defaults to `test`:

```bash
python3 -m experiments.run_experiments \
  --suite phenomenology \
  --task arithmetic,hans_nli \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --intervention mean-donor,zero \
  --mode standard,decode-only \
  --evaluation-split test
```

Other driver options:

```text
--data-root <path>       experiment artifact root passed to final analysis; default data
--results-root <path>    aggregate/final output root; default <repo>/results
--evaluation-split test|train|all   final evaluation split; default test
--list                   print selected RunSpec identities and exit
--dry-run                print pipeline commands without executing them
--continue-on-error      record pipeline failures and continue
--generate-primary-manuscript
--primary-profile iclr-28|legacy-27
```

`--generate-primary-manuscript` requires `--primary-profile` and calls `analysis/29_generate_final_results.py` after the configured pipeline runs. Without that flag, the analysis phase calls `analysis/28_visualize_experiment_results.py` and writes catalogue-scoped summaries only.

## RunSpec fields

`RunSpec` contains:

```text
suite
task
model
intervention
mode
z_thresh
batch_size
circuit_level
circuit_size
min_flip_rate
max_circuits
mlp_neurons_only
no_llm_feature_generation
evaluation_split
```

Validation accepts evaluation splits `test`, `train`, and `all`; replacement baselines `zero`, `mean`, `mean-positional`, `mean-donor`, and `mean-donor-positional`; modes `standard` and `decode-only`; and circuit levels `neuron` and `edge`.

The standard catalogue command always adds:

```text
--spectral_splits
--fast_anchoring
```

and forwards the remaining `RunSpec` values to `pipeline/_run_pipeline.sh`.

## Filesystem labels

For standard spectral runs, `RunSpec.circuit_label()` starts with:

```text
spectral_split
```

It adds:

- `-M<circuit_size>` when the size is not 100000;
- `-decode_only` in output-only mode;
- `-eval_<baseline>` for `zero`, `mean-donor`, and `mean-donor-positional`.

`mean` and `mean-positional` use the unsuffixed mean-family filesystem label. Their exact identity is preserved in configuration and validation metadata and must not be inferred only from the path.

The stage-6/7 label is:

```text
agonist_neurons-fast-random_anchor
```

with `-tau<min_flip_rate>` when the threshold is not 0.2. Stats-directory split suffixes are:

```text
test   -heldout_test
train  -eval_train
all    no suffix
```

A canonical test-split stats path is therefore:

```text
data/<task>/<model>/rule_extraction_results/neuron_flip_rules/stats/
  <circuit-label>-<bag-label>-heldout_test/
```

The corresponding stage-5 input directory is:

```text
data/<task>/<model>/neural_circuit_discovery_results/eap_ig_inputs/
  <circuit-label>/neural_circuits/
```

## Evaluation-split policy

Every `RunSpec` has `evaluation_split="test"` by default. The driver accepts `--evaluation-split test|train|all` to override that value for all selected runs. `pipeline/_run_pipeline.sh` forwards the same choice to stage 7 and interaction validation. Candidate discovery/ranking and replacement-reference estimation remain training-based regardless of the final evaluation split.

Primary manuscript tables are defined on the `test` split. The root launcher therefore requests manuscript generation only for `test`; non-test runs still produce catalogue-scoped summaries.

## Analysis outputs owned by the driver

Before execution, the selected `RunSpec` rows are serialized to:

```text
results/configured_experiments.json
```

If the pipeline phase is run, failures are serialized to:

```text
results/pipeline_failures.json
```

Catalogue-only analysis writes under:

```text
results/catalogue/
```

Full paper generation writes under the other `results/` subdirectories documented in `analysis/README.md`.

## Editing the catalogue

Keep experiment selection in `experiments/run_experiments.py`. Analysis modules should consume explicit manifests/tables rather than defining their own task/model matrix.

When changing a `RunSpec` field that affects path identity, update both `RunSpec` path construction and the corresponding label logic in `pipeline/_run_pipeline.sh`, then run the policy tests in `tests/`.
