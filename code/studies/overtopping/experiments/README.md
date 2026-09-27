# Overtopping experiment execution

This package defines the overtopping `RunSpec` registry and translates selected settings into shared pipeline commands.

## Suite selections

```text
mean-donor     30
6-7b-models     3
mean            8
checkpoints    12
```

The `all` selection deduplicates by scientific identity and resolves to 50 unique settings. Three final Pythia-1B input+output settings for Grammar, HANS-NLI, and Random FSM occur in both `mean-donor` and `checkpoints`.

The checkpoint suite is:

```text
Grammar/HANS-NLI/Random-FSM
× {step0, step48000, step96000, final EleutherAI/pythia-1b}
× input+output phase
```

Arithmetic is not included in the checkpoint suite.

Inspect or filter the registry from `code/`:

```bash
python -m studies.overtopping.experiments.run_experiments --list
python -m studies.overtopping.experiments.run_experiments --dry-run
python -m studies.overtopping.experiments.run_experiments --suite checkpoints --dry-run
python -m studies.overtopping.experiments.run_experiments \
  --task arithmetic \
  --model Qwen/Qwen2-1.5B-Instruct \
  --evaluation-split test \
  --dry-run
```

Validate persistent addressing:

```bash
python -m studies.overtopping.experiments.storage_contract
```

References:

- [Experiment configuration](../../../docs/experiments/overtopping.md)
- [Getting started](../../../docs/getting-started/README.md)
- [Pipeline](../../../docs/methods/pipeline.md)
- [Operations](../../../docs/operations/README.md)
