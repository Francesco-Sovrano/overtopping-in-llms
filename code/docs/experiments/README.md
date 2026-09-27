# Experiment families

The repository contains two experiment families with independent scientific populations and persistent artifact roots.

## Overtopping

The overtopping study applies the shared causal-intervention pipeline to an explicit registry of task, model snapshot, intervention phase, replacement baseline, and discovery/evaluation parameters.

The configured `all` selection contains 50 unique scientific settings after cross-suite deduplication. Registry membership is defined in code and can be inspected with:

```bash
bash ./run_overtopping_experiments.sh --list
```

RQ1–RQ3 use overtopping settings. RQ4 also uses the configured Pythia checkpoint trajectory view. See [Overtopping experiment design](overtopping.md).

## Poisoning

The controlled poisoning study trains matched clean and poisoned Grammar trajectories, evaluates aligned checkpoints, localizes high-leverage channels, tracks fixed channel identities through learning, and compares previous-checkpoint targets with checkpoint-aligned clean-reference selection. Optional downstream modules also score individual training exposures.

The default launcher uses Grammar, `Qwen/Qwen2-1.5B-Instruct`, seeds 13, 37, and 101, a 10% poison rate over eligible gold non-target examples, and checkpoints at 0%, 10%, 25%, 50%, 75%, and 100% of fine-tuning.

See [Poisoning protocol](poisoning/README.md), [configuration](poisoning/configuration.md), and [outputs](poisoning/outputs.md).
