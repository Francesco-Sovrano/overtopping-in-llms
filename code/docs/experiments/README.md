# Experiments

The repository contains two experiment families with separate populations and artifact roots.

## Overtopping

The overtopping study performs discovery and held-out causal evaluation across an explicit catalogue of tasks, models, intervention phases, and replacement baselines.

Read [Overtopping experiment catalogue](overtopping.md) for the exact primary and auxiliary settings.

Paper-facing RQ1–RQ3 analyses use non-poisoning overtopping runs only.

## Poisoning

The poisoning study trains matched clean and poisoned trajectories, evaluates behavior at aligned checkpoints, and measures checkpoint causal organization and poisoning-example detectability.

Read:

1. [Poisoning protocol](poisoning/) — experimental unit, training trajectories, markers, behavioral endpoints, checkpoint causal analysis, and replication unit.
2. [Poisoning configuration](poisoning/configuration.md) — launcher-defined study values and execution controls.
3. [Poisoning outputs](poisoning/outputs.md) — stage-numbered run tree, tables, figures, and cache locations.

Poisoning data live under `data/poisoning/` and are not included in the RQ1–RQ3 overtopping populations.
