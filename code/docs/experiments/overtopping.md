# Overtopping experiment configuration

## Scientific setting

An overtopping setting is the resolved `RunSpec` for a specific task, model snapshot, intervention/replacement rule, intervention phase, discovery configuration, and evaluation split. Suite membership and runtime batch size are execution metadata rather than scientific identity fields.

The registry is defined in:

```text
code/studies/overtopping/experiments/run_experiments.py
```

Inspect the effective registry from the repository root:

```bash
./run_overtopping_experiments.sh --list
./run_overtopping_experiments.sh --dry-run
```

## Current registry

For the repository state represented by this documentation, the `all` selection resolves to 50 unique settings. The storage contract reports:

| Dimension | Value | Count |
|---|---|---:|
| intervention phase | input+output (`I+O`) | 29 |
| intervention phase | output-only (`Out`) | 21 |
| replacement | `mean-donor` | 39 |
| replacement | `mean` | 8 |
| replacement | `mean-positional` | 3 |

Public suite selections before cross-suite deduplication are:

| Suite | Entries | Scope |
|---|---:|---|
| `mean-donor` | 30 | five tasks × Pythia-1B/Qwen2-1.5B/Qwen2.5-1.5B × two phases |
| `6-7b-models` | 3 | selected Qwen2-7B settings |
| `mean` | 8 | selected small-model mean-replacement settings |
| `checkpoints` | 12 | Grammar/HANS-NLI/Random-FSM × four Pythia-1B checkpoints, input+output phase |

The final `EleutherAI/pythia-1b` input+output settings for Grammar, HANS-NLI, and Random FSM are present in both the `mean-donor` and `checkpoints` selections. The `all` selection deduplicates those scientific settings, producing 50 unique settings rather than the arithmetic sum of suite entry counts.

## Suite contents

### `mean-donor`

Tasks:

```text
arithmetic
grammar_acceptability
random_fsm
hans_nli
bon_jailbreaking
```

Models:

```text
EleutherAI/pythia-1b
Qwen/Qwen2-1.5B-Instruct
Qwen/Qwen2.5-1.5B-Instruct
```

Each task/model cell has input+output and output-only settings.

### `6-7b-models`

Current entries:

```text
arithmetic       Qwen/Qwen2-7B-Instruct  output-only
bon_jailbreaking Qwen/Qwen2-7B-Instruct  output-only
hans_nli         Qwen/Qwen2-7B-Instruct  input+output
```

These settings use `mean-positional`, MLP-only coordinates, `circuit_size=100000`, and `min_flip_rate=0.2`. Task-specific thresholds and batch sizes are encoded in the registry.

### `mean`

Current entries use `mean` replacement for:

```text
arithmetic            Qwen/Qwen2-1.5B-Instruct       input+output, output-only
grammar_acceptability Qwen/Qwen2.5-1.5B-Instruct     input+output, output-only
hans_nli              Qwen/Qwen2.5-1.5B-Instruct     input+output, output-only
random_fsm            Qwen/Qwen2.5-1.5B-Instruct     input+output, output-only
```

### `checkpoints`

The checkpoint trajectory is defined for Grammar, HANS-NLI, and Random FSM in the input+output phase:

```text
EleutherAI/pythia-1b@step0
EleutherAI/pythia-1b@step48000
EleutherAI/pythia-1b@step96000
EleutherAI/pythia-1b
```

Arithmetic is not included in the checkpoint suite.

## Evaluation split

The repository launcher defaults to the held-out `test` split. Explicit alternatives are `train` and `all`.

`RunSpec.evaluation_suffix()` uses:

```text
test   -> -heldout_test
train  -> -eval_train
all    -> no evaluation suffix
```

The evaluation split is part of persistent experiment identity when multiple split variants coexist.

## Filters and phases

The runner accepts scientific filters for:

```text
--task
--model
--intervention
--mode
--evaluation-split
```

Execution phase is selected with:

```text
--phase pipeline
--phase analysis
--phase all
```

Examples:

```bash
./run_overtopping_experiments.sh --suite checkpoints --dry-run

./run_overtopping_experiments.sh \
  --task arithmetic \
  --model Qwen/Qwen2-1.5B-Instruct \
  --evaluation-split test \
  --dry-run
```

## Persistent addressing

`RunSpec` owns persistent paths through:

```text
circuit_label()
bag_label()
evaluation_suffix()
input_data_dir(data_root)
stats_dir(data_root)
```

`mean` and `mean-positional` share the mean-family circuit-path convention. `mean-donor` uses the donor-specific suffix. The resolved intervention remains available in the registry and pipeline arguments.

Validate address uniqueness and the registry fingerprint with:

```bash
cd code
python -m studies.overtopping.experiments.storage_contract
```

The check is read-only with respect to scientific artifacts.

## Analysis populations

The configured manifest is established before metric-specific filtering.

- RQ1 uses configured settings with the required directional singleton metrics and analyzes intervention phases separately.
- RQ2 requires a nonempty frozen candidate set and compatible simultaneous-set outputs; replacement regimes are analyzed separately.
- RQ3 uses the configured settings with the artifacts required by each threshold, graded, margin, or temporal analysis.
- RQ4 uses the Pythia checkpoint trajectory view and the controlled poisoning trajectories.

A completed zero-candidate setting remains a measured observation for metrics that define a zero value. Set-level composition is not applicable when the candidate set is empty.

## Runner control outputs

Selected configurations and failures are written under the results root:

```text
results/configured_experiments.json
results/pipeline_failures.json
```

Model-backed scientific artifacts are stored under `data/`.
