# RQ4 — learning-time causal organization

## Question

> **How does causal organization change during learning beyond behavioral performance?**

RQ4 asks whether directional causal reach and coordinate-level causal roles change during training in ways that are not summarized by ordinary behavioral metrics.

The paper-facing analysis has two separate components:

1. natural Pythia checkpoint trajectories;
2. a controlled clean-versus-poisoned Grammar trajectory with training seed 37.

These components are not pooled into one statistical population.

## Part A — natural checkpoint trajectories

### Population

The checkpoint analysis uses configured Pythia checkpoint settings in the overtopping registry. Checkpoint identity is part of model identity.

### Quantities

Principal trajectory quantities are:

```text
competence
U_J_i2c   # 0→1 singleton-union reach
U_J_c2i   # 1→0 singleton-union reach
```

Pooled `U(J)` can be shown as a descriptive distribution-level quantity, but directional developmental interpretation should use `U_J_i2c` and `U_J_c2i` because the pooled quantity mixes source-state populations whose prevalence can change across checkpoints.

### Coordinate identity

Checkpoint-local candidate discovery can establish aggregate changes in causal organization. It does not establish that a specific coordinate gained, lost, or changed a causal role.

A coordinate-level claim requires explicit evaluation of the same aligned coordinate or fixed candidate union across checkpoints.

### Manuscript files

```text
results/paper/figures/05_rq4_learning/
├── fig5a_pythia_checkpoint_trajectory.pdf
├── fig5s1_pythia_checkpoint_U_0to1.pdf
└── fig5s2_pythia_checkpoint_U_1to0.pdf
```

## Part B — controlled Grammar trajectory, seed 37

The paper-facing controlled-learning analysis uses the Grammar clean-versus-poisoned trajectory with training seed 37.

### Behavioral endpoints

Report aligned checkpoint trajectories for:

```text
ordinary Grammar performance
triggered target conversion
sham-trigger behavior, when available
```

The behavioral analysis asks whether the controlled hidden objective is acquired while ordinary task performance remains similar.

### Candidate localization and causal evaluation

Candidate localization uses the defender-visible training/evaluation population defined by the poisoning pipeline. Hidden attack annotations do not select candidate coordinates.

After candidate identity is frozen, matched control and triggered views can be used to measure ordinary-direction and attack-direction causal effects for those coordinates.

### Attack-blind clean-reference defense screen

The controlled trajectory also evaluates a checkpoint-aligned clean-reference screen. Candidate targets are ranked using only the difference between poisoned-trained and clean-trained control-correctness disruption and are required to satisfy a configured poisoned-model benign-damage budget. Triggered attack outcomes are joined only after the target set is selected. The default operating-point budget in the plotting script is 0.30, and the full budget grid is retained in machine-readable output.

### Checkpoint-local versus fixed-coordinate evidence

Two longitudinal objects are distinct:

- **checkpoint-local candidates:** coordinates rediscovered independently at a checkpoint;
- **fixed coordinates:** the same prespecified coordinates explicitly evaluated at multiple matched checkpoints.

Only the fixed-coordinate design supports statements that a specific coordinate acquires, loses, or changes a behavior-specific causal role.

### Paper-facing claim hierarchy

Behavioral trajectory only:

> **The controlled hidden objective can be acquired without being summarized by ordinary Grammar performance.**

Fixed-coordinate trajectory available:

> **The controlled hidden objective is associated with a behavior-specific change in causal leverage at fixed internal coordinates.**

## Implementation

Natural checkpoint reporting:

```text
studies/overtopping/analysis/stage06_competence_vs_overtopping_figures.py
studies/overtopping/analysis/stage06_manuscript_story_figures.py
```

Controlled-learning implementation:

```text
studies/poisoning/
```

The poisoning package retains configurable training, checkpoint analysis, fixed-candidate evaluation, and aggregate reporting infrastructure. The paper-facing RQ4 population defined here is the Grammar seed-37 trajectory.

Detailed code-level protocol and artifact layouts are documented under:

- [Poisoning protocol](../experiments/poisoning/README.md)
- [Poisoning configuration](../experiments/poisoning/configuration.md)
- [Poisoning outputs](../experiments/poisoning/outputs.md)

## Outputs

Controlled-learning analyses are written under:

```text
results/analysis/rq4_learning/poisoning/
```

and manuscript-facing figures under:

```text
results/paper/figures/05_rq4_learning/poisoning/
```

When exactly one fully materialized Grammar clean-reference defense figure is available, the reporting driver also publishes it at:

```text
results/paper/figures/05_rq4_learning/rq4_grammar_clean_reference_defense.pdf
```

## Interpretation

RQ4 distinguishes three evidential levels:

1. behavioral objective acquisition;
2. aggregate checkpoint-level causal reorganization;
3. fixed-coordinate causal-role change.

Claims should match the level directly measured by the analysis.
