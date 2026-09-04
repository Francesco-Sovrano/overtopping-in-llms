# Overtopping experiment catalogue

The overtopping study uses an explicit catalogue rather than a Cartesian product of tasks, models, phases, and replacement baselines. The catalogue is defined in `code/studies/overtopping/experiments/run_experiments.py`.

## Suites

### `paper-primary`

The primary manuscript profile is `iclr-28` and contains 28 settings.

| Task | Model | Phase | Replacement | Discovery settings |
|---|---|---|---|---|
| `arithmetic` | `EleutherAI/pythia-1b` | decode-only | `mean` | `M=200000`, `tau=0.3` |
| `arithmetic` | `EleutherAI/pythia-1b@step48000` | decode-only | `mean-donor` | `M=200000`, `tau=0.3` |
| `arithmetic` | `EleutherAI/pythia-6.9b` | decode-only | `mean-positional` | MLP-only, `M=100000`, `tau=0.2` |
| `arithmetic` | `Qwen/Qwen2-1.5B-Instruct` | input+output | `mean-donor` | `M=200000`, `tau=0.3` |
| `arithmetic` | `Qwen/Qwen2-1.5B-Instruct` | decode-only | `mean-donor` | `M=200000`, `tau=0.3` |
| `arithmetic` | `Qwen/Qwen2-7B-Instruct` | decode-only | `mean-positional` | MLP-only, `M=100000`, `tau=0.2` |
| `arithmetic` | `Qwen/Qwen2.5-1.5B-Instruct` | decode-only | `mean-donor` | `M=200000`, `tau=0.3` |
| `bon_jailbreaking` | `Qwen/Qwen2-1.5B-Instruct` | decode-only | `mean-donor` | `M=200000`, `tau=0.3` |
| `bon_jailbreaking` | `Qwen/Qwen2-7B-Instruct` | decode-only | `mean-positional` | MLP-only, `M=100000`, `tau=0.2` |
| `bon_jailbreaking` | `Qwen/Qwen2.5-1.5B-Instruct` | decode-only | `mean-donor` | `M=200000`, `tau=0.3` |
| `grammar_acceptability` | `EleutherAI/pythia-1b` | input+output | `mean-donor` | `M=200000`, `tau=0.3` |
| `grammar_acceptability` | `EleutherAI/pythia-1b` | decode-only | `mean-donor` | `M=200000`, `tau=0.3` |
| `grammar_acceptability` | `EleutherAI/pythia-1b@step48000` | input+output | `mean-donor` | `M=200000`, `tau=0.3` |
| `grammar_acceptability` | `EleutherAI/pythia-1b@step48000` | decode-only | `mean-donor` | `M=200000`, `tau=0.3` |
| `grammar_acceptability` | `EleutherAI/pythia-1b@step96000` | input+output | `mean-donor` | `M=200000`, `tau=0.3` |
| `grammar_acceptability` | `EleutherAI/pythia-1b@step96000` | decode-only | `mean-donor` | `M=200000`, `tau=0.3` |
| `grammar_acceptability` | `Qwen/Qwen2.5-1.5B-Instruct` | input+output | `mean-donor` | `M=200000`, `tau=0.3` |
| `grammar_acceptability` | `Qwen/Qwen2.5-1.5B-Instruct` | decode-only | `mean-donor` | `M=200000`, `tau=0.3` |
| `hans_nli` | `Qwen/Qwen2-1.5B-Instruct` | input+output | `mean-donor` | `M=200000`, `tau=0.3` |
| `hans_nli` | `Qwen/Qwen2-7B-Instruct` | input+output | `mean-positional` | MLP-only, `M=100000`, `tau=0.2` |
| `hans_nli` | `Qwen/Qwen2.5-1.5B-Instruct` | input+output | `mean-donor` | `M=200000`, `tau=0.3` |
| `hans_nli` | `Qwen/Qwen2.5-1.5B-Instruct` | decode-only | `mean-donor` | `M=200000`, `tau=0.3` |
| `random_fsm` | `EleutherAI/pythia-1b` | input+output | `mean-donor` | `M=200000`, `tau=0.3` |
| `random_fsm` | `EleutherAI/pythia-1b` | decode-only | `mean-donor` | `M=200000`, `tau=0.3` |
| `random_fsm` | `EleutherAI/pythia-1b@step48000` | input+output | `mean-donor` | `M=200000`, `tau=0.3` |
| `random_fsm` | `EleutherAI/pythia-1b@step96000` | decode-only | `mean-donor` | `M=200000`, `tau=0.3` |
| `random_fsm` | `Qwen/Qwen2.5-1.5B-Instruct` | input+output | `mean-donor` | `M=200000`, `tau=0.3` |
| `random_fsm` | `Qwen/Qwen2.5-1.5B-Instruct` | decode-only | `mean` | `M=200000`, `tau=0.3` |

### `paper-auxiliary`

The auxiliary suite contains 11 targeted settings used for replacement-baseline and phase/model comparisons.

| Task | Model | Phase | Replacement |
|---|---|---|---|
| `arithmetic` | `Qwen/Qwen2-1.5B-Instruct` | decode-only | `mean` |
| `grammar_acceptability` | `Qwen/Qwen2.5-1.5B-Instruct` | input+output | `mean` |
| `grammar_acceptability` | `Qwen/Qwen2.5-1.5B-Instruct` | decode-only | `mean` |
| `hans_nli` | `Qwen/Qwen2.5-1.5B-Instruct` | input+output | `mean` |
| `hans_nli` | `Qwen/Qwen2.5-1.5B-Instruct` | decode-only | `mean` |
| `random_fsm` | `Qwen/Qwen2.5-1.5B-Instruct` | input+output | `mean` |
| `arithmetic` | `Qwen/Qwen2.5-1.5B-Instruct` | input+output | `mean-donor` |
| `bon_jailbreaking` | `Qwen/Qwen2.5-1.5B-Instruct` | input+output | `mean-donor` |
| `hans_nli` | `Qwen/Qwen2-1.5B-Instruct` | decode-only | `mean-donor` |
| `grammar_acceptability` | `Qwen/Qwen2-1.5B-Instruct` | input+output | `mean-donor` |
| `grammar_acceptability` | `Qwen/Qwen2-1.5B-Instruct` | decode-only | `mean-donor` |

All auxiliary settings use `M=200000` and `tau=0.3` in the catalogue.

### `all`

`all` is the deduplicated union of the primary and auxiliary suites: 39 non-poisoning settings.

## Analysis populations derived from the catalogue

- Primary matrix and RQ2: 28 primary settings.
- RQ1 Figure 2: all 39 settings.
- RQ3: all 28 primary settings plus auxiliary settings whose exact RQ3 diagnostics are complete.
- Poisoning: separate configuration under `studies/poisoning/`.

## Inspect the catalogue

From the repository root:

```bash
./run_overtopping_experiments.sh --list
./run_overtopping_experiments.sh --dry-run
```

Direct Python invocation from `code/`:

```bash
python -m studies.overtopping.experiments.run_experiments \
  --suite paper-primary \
  --task arithmetic \
  --model Qwen/Qwen2-1.5B-Instruct \
  --dry-run
```

Supported filters:

```text
--suite
--task
--model
--intervention
--mode
--phase
--evaluation-split
```

`--suite` may be supplied multiple times. `--phase` accepts `pipeline`, `analysis`, or `all`.

## Evaluation split

The repository launcher defaults to `test`. The Python runner accepts `test`, `train`, or `all`.

Primary manuscript generation is defined for the `test` split. `--generate-primary-manuscript` therefore requires `--primary-profile iclr-28` and a test evaluation split.

## Catalogue outputs

The runner writes the selected experiment definitions to:

```text
results/configured_experiments.json
```

Pipeline failures are recorded in:

```text
results/pipeline_failures.json
```

Each `RunSpec` resolves its own statistics directory from the scientific configuration. Analysis code should use these path constructors rather than infer the declared population from recursive directory discovery.

## Zero-candidate settings

A completed setting can contain no discovered agonists. Such a setting remains part of analyses whose population is defined by the catalogue and is represented according to the persisted pipeline status/artifact contract. A missing required experiment directory is an incomplete setting rather than a measured zero.


## Research-question analyses

The catalogue is consumed by:

- [RQ1 — prevalence and competence](../research-questions/rq1-prevalence.md);
- [RQ2 — composition](../research-questions/rq2-composition.md);
- [RQ3 — threshold-event analysis](../research-questions/rq3-threshold-event.md);
- [RQ4 — learning](../research-questions/rq4-learning.md).
