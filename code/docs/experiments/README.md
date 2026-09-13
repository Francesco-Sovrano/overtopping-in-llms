# Experiment families

The repository contains two experiment families with independent scientific populations and persistent artifact roots.

## Overtopping

The overtopping study applies the shared causal-intervention pipeline to an explicit registry of task, model snapshot, intervention phase, replacement baseline, and discovery/evaluation parameters.

The current `all` selection contains 50 unique scientific settings after cross-suite deduplication. Registry membership is defined in code and can be inspected with:

```bash
./run_overtopping_experiments.sh --list
```

RQ1–RQ3 use overtopping settings. RQ4 also uses the configured Pythia checkpoint trajectory view. See [Overtopping experiment design](overtopping.md).

## Poisoning

The poisoning study trains matched clean and poisoned trajectories, evaluates aligned checkpoints, performs configured causal localization, freezes candidate unions, evaluates longitudinal causal effects, scores training examples, and aggregates across training seeds.

The default launcher uses Grammar, `Qwen/Qwen2-1.5B-Instruct`, and seeds 13, 37, and 101.

See [Poisoning protocol](poisoning/README.md), [configuration](poisoning/configuration.md), and [outputs](poisoning/outputs.md).
