# RQ4 — learning and causal-role dynamics

## Question

RQ4 asks how overtopping-related causal organization changes during learning and whether a controlled hidden objective changes causal roles in ways that are not captured by ordinary behavioral performance alone.

RQ4 has two experiment families:

1. natural Pythia checkpoints from the overtopping catalogue;
2. matched clean/poisoned training trajectories under `data/poisoning/`.

These are analyzed separately and should not be pooled into one statistical population.

## Part A — Pythia checkpoint trajectories

### Population

The checkpoint analysis uses configured Pythia checkpoint settings present in the overtopping experiment catalogue. Checkpoint identity is part of model identity, for example `pythia-1b@step48000`.

### Quantities

The reporting pipeline can plot:

```text
competence
U(J)
U_J_i2c
U_J_c2i
```

and primary-matrix structural quantities where available.

`U(J)` is a pooled distribution-level quantity. Because its mixture weights depend on baseline behavioral prevalence, direction-specific developmental interpretation should use `U_J_i2c` and `U_J_c2i`.

### Coordinate identity

Checkpoint-local candidate sets can establish that the population-level causal organization changes with training. They do not by themselves establish that one specific coordinate acquired, lost, or changed a role.

A coordinate-level developmental claim requires a fixed aligned coordinate or fixed candidate union evaluated across checkpoints.

### Manuscript files

```text
results/paper/figures/05_rq4_learning/
├── fig5a_pythia_checkpoint_trajectory.pdf
├── fig5s1_pythia_checkpoint_U_0to1.pdf
└── fig5s2_pythia_checkpoint_U_1to0.pdf
```

## Part B — controlled poisoning

The poisoning study is a separate experiment family. It compares matched clean and poisoned training trajectories at aligned checkpoints.

Full design documentation:

- [Poisoning protocol](../experiments/poisoning/)
- [Poisoning configuration](../experiments/poisoning/configuration.md)
- [Poisoning outputs](../experiments/poisoning/outputs.md)

### Behavioral endpoints

The main poisoning outputs include ordinary no-trigger task performance and attack behavior. `conditional_conversion_rate` measures trigger conversion among examples that are attack-eligible under the corresponding control prompt.

### Checkpoint causal analysis

Checkpoint causal discovery evaluates behavior-specific populations such as normal-task examples and attack-cohort control-correctness examples. Candidate-set, singleton, and fixed-channel quantities can then be compared between clean and poisoned conditions.

### Checkpoint-local versus fixed-channel analyses

Two longitudinal objects must be distinguished:

- **checkpoint-local candidates:** candidates rediscovered independently at a checkpoint;
- **fixed candidate union / fixed channels:** the same coordinates explicitly evaluated at multiple checkpoints.

Only the fixed-coordinate design supports statements about a particular channel retaining, acquiring, losing, or changing a causal role across checkpoints.

### Prospective defense leverage

The poisoning story analysis can compute:

```text
Delta_def = attack_suppression - benign_correctness_damage
```

for candidate singleton interventions selected according to the configured prospective rule. This is a target-screening/selectivity quantity. It is not an end-to-end defense efficacy metric.

### Poisoning-example detection

The poisoning detector uses model-derived activation/update/causal-disruption information to rank training examples. Poison labels are used for post-hoc evaluation rather than candidate-score construction.

### Replication unit

Training seed is the replication unit for poisoning trajectory inference. Prompt-level sample size controls within-run precision but does not substitute for independent training seeds.

## Implementation

Pythia checkpoint reporting is owned by:

```text
studies/overtopping/analysis/stage06_competence_vs_overtopping_figures.py
studies/overtopping/analysis/stage06_manuscript_story_figures.py
```

The poisoning study is implemented under:

```text
studies/poisoning/
```

and orchestrated by `run_poisoning_experiments.sh`. The stage-numbered artifact contract is described in [Poisoning outputs](../experiments/poisoning/outputs.md).

## Outputs

Poisoning cross-run analyses are written under:

```text
results/analysis/rq4_learning/poisoning/
```

and manuscript-facing poisoning figures under:

```text
results/paper/figures/05_rq4_learning/poisoning/
```

Per-run story figures are generated from the stage-numbered poisoning run tree documented in [Poisoning outputs](../experiments/poisoning/outputs.md).

## Interpretation

RQ4 distinguishes population-level reorganization from fixed-coordinate role change. Behavioral divergence, checkpoint-local causal changes, and fixed-channel role reassignment are separate evidential levels and should be reported according to the analysis actually performed.
