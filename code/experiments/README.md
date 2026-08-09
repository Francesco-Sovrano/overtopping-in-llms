# Experiment catalogue and execution

`experiments/run_experiments.py` is the single executable catalogue for standard non-poisoning experiments. It is paper-centered and explicit: configurations are written as individual `RunSpec` rows rather than generated from a broad task × model × phase × baseline Cartesian product.

## Entry points

From repository root:

```bash
./run_experiments.sh
```

From `code/`:

```bash
python3 -m experiments.run_experiments --help
```

The root launcher supplies repository-root `data/` and `results/` paths and defaults the primary manuscript profile to `iclr-28` when the evaluation split is `test`.

## Catalogue size and suites

The default catalogue contains **36 unique configurations**:

| Suite | Count | Purpose |
|---|---:|---|
| `paper-primary` | 28 | Exact primary model–task–phase settings reported in manuscript Appendix Table 8, with their selected replacement baselines |
| `paper-auxiliary` | 8 | Targeted baseline and phase controls that support interpretation of the primary matrix |
| `all` | 36 | `paper-primary` plus `paper-auxiliary` |

Inspect the exact expanded list:

```bash
python3 -m experiments.run_experiments --suite paper-primary --list
python3 -m experiments.run_experiments --suite paper-auxiliary --list
python3 -m experiments.run_experiments --suite all --list
```

The root launcher respects an explicit `--suite`; it adds `--suite all` only when no suite was supplied.

## The 28 primary configurations

The primary suite follows the manuscript table ordering.

### Arithmetic

| Model | Phase | Baseline |
|---|---|---|
| Pythia-1B | Out | `mean` |
| Pythia-1B@48k | Out | `mean-donor` |
| Pythia-6.9B | Out | `mean-positional` |
| Qwen2-1.5B-Instruct | I+O | `mean-donor` |
| Qwen2-1.5B-Instruct | Out | `mean-donor` |
| Qwen2-7B-Instruct | Out | `mean-positional` |
| Qwen2.5-1.5B-Instruct | Out | `mean-donor` |

### Jailbreaking

| Model | Phase | Baseline |
|---|---|---|
| Qwen2-1.5B-Instruct | Out | `mean-donor` |
| Qwen2-7B-Instruct | Out | `mean-positional` |
| Qwen2.5-1.5B-Instruct | Out | `mean-donor` |

### Grammar acceptability

| Model | Phase | Baseline |
|---|---|---|
| Pythia-1B | I+O | `mean-donor` |
| Pythia-1B | Out | `mean-donor` |
| Pythia-1B@48k | I+O | `mean-donor` |
| Pythia-1B@48k | Out | `mean-donor` |
| Pythia-1B@96k | I+O | `mean-donor` |
| Pythia-1B@96k | Out | `mean-donor` |
| Qwen2.5-1.5B-Instruct | I+O | `mean-donor` |
| Qwen2.5-1.5B-Instruct | Out | `mean-donor` |

### HANS NLI

| Model | Phase | Baseline |
|---|---|---|
| Qwen2-1.5B-Instruct | I+O | `mean-donor` |
| Qwen2-7B-Instruct | I+O | `mean-positional` |
| Qwen2.5-1.5B-Instruct | I+O | `mean-donor` |
| Qwen2.5-1.5B-Instruct | Out | `mean-donor` |

### Random FSM

| Model | Phase | Baseline |
|---|---|---|
| Pythia-1B | I+O | `mean-donor` |
| Pythia-1B | Out | `mean-donor` |
| Pythia-1B@48k | I+O | `mean-donor` |
| Pythia-1B@96k | Out | `mean-donor` |
| Qwen2.5-1.5B-Instruct | I+O | `mean-donor` |
| Qwen2.5-1.5B-Instruct | Out | `mean` |

The 1B/1.5B scans use the paper-style `M=200000`, `tau=0.3`, batch size 32, and at most one circuit. Qwen arithmetic uses `z_thresh=10`; Pythia arithmetic uses `z_thresh=5`.

The four primary large-model settings are MLP-only, use `circuit_size=100000`, `min_flip_rate=0.2`, and `mean-positional`: Pythia-6.9B arithmetic Out; Qwen2-7B arithmetic Out; Qwen2-7B HANS NLI I+O; and Qwen2-7B jailbreaking Out.

## Auxiliary configurations

The auxiliary suite intentionally contains only eight targeted runs.

Six complete the manuscript's mean-versus-mean-donor sensitivity comparisons:

```text
Qwen2-1.5B arithmetic Out                 mean
Qwen2.5-1.5B grammar I+O                 mean
Qwen2.5-1.5B grammar Out                 mean
Qwen2.5-1.5B HANS NLI I+O                mean
Qwen2.5-1.5B HANS NLI Out                mean
Qwen2.5-1.5B random FSM I+O              mean
```

Two add focused phase diagnostics without recreating the former broad factorial sweep:

```text
Qwen2.5-1.5B arithmetic I+O              mean-donor
Qwen2-1.5B HANS NLI Out                  mean-donor
```

The first helps distinguish Qwen2.5 arithmetic output behavior from prompt-processing effects and is also relevant to the paper's secondary threshold/control analysis. The second is the output-only counterpart to the primary Qwen2-1.5B I+O NLI setting and corresponds to a reported no-selected-channel boundary condition.

There are no automatic `zero`-baseline runs in the main catalogue.

## CLI

```text
--suite {paper-primary,paper-auxiliary,all}
--phase {all,pipeline,analysis}
--task CSV
--model CSV
--intervention CSV
--mode CSV
--evaluation-split {test,train,all}
--data-root PATH
--results-root PATH
--primary-profile {iclr-28,legacy-27}
--generate-primary-manuscript
--list
--dry-run
--continue-on-error
```

Filters are exact comma-separated values.

## Evaluation split

Every `RunSpec` has an `evaluation_split`, default `test`.

```text
test   -> -heldout_test
train  -> -eval_train
all    -> no evaluation suffix
```

Primary manuscript generation requires `test`.

## Path construction and execution

`RunSpec.circuit_label()` encodes spectral split, circuit size, decode-only mode, and intervention when needed. `RunSpec.bag_label()` encodes the stage-6 candidate-selection mode and `tau` when it differs from `0.2`.

Each catalogue row becomes one call to `pipeline/_run_pipeline.sh` with explicit spectral splits, fast anchoring, `z_thresh`, batch size, circuit granularity/size, intervention, `tau`, evaluation split, and maximum circuits. Decode-only and MLP-only flags are added only when selected by the row.

The expanded catalogue is written to:

```text
<results-root>/configured_experiments.json
```

Pipeline failures are recorded in:

```text
<results-root>/pipeline_failures.json
```

## Editing the catalogue

Add or remove experiment definitions only in `run_experiments.py`. Do not create separate shell scripts with independent task/model matrices. Reusable path and command logic belongs in `execution.py`.
