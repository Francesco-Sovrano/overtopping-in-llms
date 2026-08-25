# Experiment catalogue

This page documents the executable overtopping-study catalogue. The source of truth for membership is `studies/overtopping/experiments/run_experiments.py`, and the catalogue can be inspected without running models.

Configurations are explicit `RunSpec` records rather than a Cartesian product. A task/model/phase/replacement-baseline combination is part of the study only when it is listed in the catalogue source. The selected `RunSpec` is executed by the shared `pipeline/` package, which is also reused by poisoning checkpoint analysis.

Run it from `code/`:

```bash
cd code
python3 -m studies.overtopping.experiments.run_experiments --help
```

## Catalogue

The current catalogue contains **39 unique configurations**:

| Suite | Count | Purpose |
|---|---:|---|
| `paper-primary` | 28 | primary model/task/phase settings |
| `paper-auxiliary` | 11 | targeted baseline and phase comparisons |
| `all` | 39 | union of both suites |

Verify the live catalogue at any time with:

```bash
python3 -m studies.overtopping.experiments.run_experiments --suite paper-primary --list
python3 -m studies.overtopping.experiments.run_experiments --suite paper-auxiliary --list
python3 -m studies.overtopping.experiments.run_experiments --suite all --list
```

## Primary suite

The 28 primary configurations are:

| Task | Model | Phase | Intervention |
|---|---|---|---|
| arithmetic | `EleutherAI/pythia-1b` | output-only | `mean` |
| arithmetic | `EleutherAI/pythia-1b@step48000` | output-only | `mean-donor` |
| arithmetic | `EleutherAI/pythia-6.9b` | output-only | `mean-positional` |
| arithmetic | `Qwen/Qwen2-1.5B-Instruct` | input+output | `mean-donor` |
| arithmetic | `Qwen/Qwen2-1.5B-Instruct` | output-only | `mean-donor` |
| arithmetic | `Qwen/Qwen2-7B-Instruct` | output-only | `mean-positional` |
| arithmetic | `Qwen/Qwen2.5-1.5B-Instruct` | output-only | `mean-donor` |
| bon_jailbreaking | `Qwen/Qwen2-1.5B-Instruct` | output-only | `mean-donor` |
| bon_jailbreaking | `Qwen/Qwen2-7B-Instruct` | output-only | `mean-positional` |
| bon_jailbreaking | `Qwen/Qwen2.5-1.5B-Instruct` | output-only | `mean-donor` |
| grammar_acceptability | `EleutherAI/pythia-1b` | input+output | `mean-donor` |
| grammar_acceptability | `EleutherAI/pythia-1b` | output-only | `mean-donor` |
| grammar_acceptability | `EleutherAI/pythia-1b@step48000` | input+output | `mean-donor` |
| grammar_acceptability | `EleutherAI/pythia-1b@step48000` | output-only | `mean-donor` |
| grammar_acceptability | `EleutherAI/pythia-1b@step96000` | input+output | `mean-donor` |
| grammar_acceptability | `EleutherAI/pythia-1b@step96000` | output-only | `mean-donor` |
| grammar_acceptability | `Qwen/Qwen2.5-1.5B-Instruct` | input+output | `mean-donor` |
| grammar_acceptability | `Qwen/Qwen2.5-1.5B-Instruct` | output-only | `mean-donor` |
| hans_nli | `Qwen/Qwen2-1.5B-Instruct` | input+output | `mean-donor` |
| hans_nli | `Qwen/Qwen2-7B-Instruct` | input+output | `mean-positional` |
| hans_nli | `Qwen/Qwen2.5-1.5B-Instruct` | input+output | `mean-donor` |
| hans_nli | `Qwen/Qwen2.5-1.5B-Instruct` | output-only | `mean-donor` |
| random_fsm | `EleutherAI/pythia-1b` | input+output | `mean-donor` |
| random_fsm | `EleutherAI/pythia-1b` | output-only | `mean-donor` |
| random_fsm | `EleutherAI/pythia-1b@step48000` | input+output | `mean-donor` |
| random_fsm | `EleutherAI/pythia-1b@step96000` | output-only | `mean-donor` |
| random_fsm | `Qwen/Qwen2.5-1.5B-Instruct` | input+output | `mean-donor` |
| random_fsm | `Qwen/Qwen2.5-1.5B-Instruct` | output-only | `mean` |

The small-model catalogue helper uses `circuit_size=200000`, `min_flip_rate=0.3`, batch size 32, and `max_number_of_circuits_to_analyze=1`. Arithmetic rows set a positive MAD z-threshold (`5` for Pythia arithmetic and `10` for Qwen arithmetic). The four large-model rows use MLP-only neuron analysis, `circuit_size=100000`, `min_flip_rate=0.2`, and `mean-positional`.

## Auxiliary suite

The 11 auxiliary configurations are:

```text
arithmetic            Qwen/Qwen2-1.5B-Instruct    output-only    mean
grammar_acceptability Qwen/Qwen2.5-1.5B-Instruct input+output   mean
grammar_acceptability Qwen/Qwen2.5-1.5B-Instruct output-only    mean
hans_nli              Qwen/Qwen2.5-1.5B-Instruct input+output   mean
hans_nli              Qwen/Qwen2.5-1.5B-Instruct output-only    mean
random_fsm            Qwen/Qwen2.5-1.5B-Instruct input+output   mean
arithmetic            Qwen/Qwen2.5-1.5B-Instruct input+output   mean-donor
bon_jailbreaking      Qwen/Qwen2.5-1.5B-Instruct input+output   mean-donor
hans_nli              Qwen/Qwen2-1.5B-Instruct   output-only    mean-donor
grammar_acceptability Qwen/Qwen2-1.5B-Instruct   input+output   mean-donor
grammar_acceptability Qwen/Qwen2-1.5B-Instruct   output-only    mean-donor
```

There are no automatically generated zero-baseline runs.

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
--primary-profile {iclr-28}
--list
--dry-run
--continue-on-error
--generate-primary-manuscript
```

Filters are exact comma-separated values. `--list` prints the selected `RunSpec` records and exits. `--dry-run` builds commands without executing them.

`--primary-profile` is required only with `--generate-primary-manuscript`. The only supported profile is `iclr-28`, which requires all 28 primary rows including the Qwen2-1.5B input+output HANS NLI row. Primary manuscript generation requires the `test` evaluation split.

## Execution and outputs

Each catalogue row becomes one invocation of `pipeline/run_pipeline.sh` with explicit task, model, intervention, phase, circuit settings, threshold, and evaluation split. `standard` mode means input+output intervention; `decode-only` enables the pipeline's `--decode_only` path.

The expanded selected catalogue is written to:

```text
<results-root>/configured_experiments.json
```

Pipeline failures are recorded in:

```text
<results-root>/pipeline_failures.json
```

The pipeline itself writes run artifacts under `<data-root>` according to `RunSpec` path construction. See [Numbered causal-intervention pipeline](pipeline.md) for stage-level outputs.

## Maintaining the catalogue

Add, remove, or alter experiment definitions in `run_experiments.py`. Shared naming, filtering, and command construction belongs in `execution.py`. After changing the catalogue, run:

```bash
python3 -m studies.overtopping.experiments.run_experiments --suite all --list
```

The `paper-primary` and `paper-auxiliary` constructors assert their expected counts (28 and 11), so accidental catalogue-size drift fails immediately. After changing identities or settings, also inspect the listed rows to confirm that the intended scientific matrix still matches the documented profile.
