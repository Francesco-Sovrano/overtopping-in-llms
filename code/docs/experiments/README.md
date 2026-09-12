# Experiments

The repository contains two experiment families with separate scientific populations and persistent artifact roots.

## Overtopping

The overtopping study performs discovery and held-out causal evaluation across an explicit configurable registry of tasks, model snapshots, intervention phases, and replacement baselines.

The registry contains 31 final-snapshot task×model×phase cells, 18 non-final Grammar/HANS-NLI/FSM Pythia checkpoint settings, and 7 matched replacement-baseline repeats. See [Overtopping experiment design](overtopping.md) for the exact configured cells and analysis populations.

RQ1–RQ3 use overtopping settings. RQ4 additionally uses the configured Pythia checkpoint trajectories.

## Poisoning

The poisoning study trains matched clean and poisoned trajectories, evaluates behavior at aligned checkpoints, and measures checkpoint causal organization and poisoning-example detectability.

Read:

1. [Poisoning protocol](poisoning/) — experimental unit, training trajectories, markers, behavioral endpoints, checkpoint causal analysis, and replication unit.
2. [Poisoning configuration](poisoning/configuration.md) — launcher-defined study values and execution controls.
3. [Poisoning outputs](poisoning/outputs.md) — stage-numbered run tree, tables, figures, and cache locations.

Poisoning data live under `data/poisoning/` and are not included in the RQ1–RQ3 overtopping populations.
