# RQ4 — learning-time causal organization

## Objective

Measure how behavioral performance and causal organization change across model checkpoints, and distinguish checkpoint-level aggregate changes from fixed-coordinate causal-role changes.

RQ4 contains two independent analysis components:

1. Pythia checkpoint trajectories from the overtopping registry;
2. matched clean and poisoned Grammar training trajectories.

These components have separate populations and are not pooled into one statistical sample.

## Pythia checkpoint trajectories

### Population

The configured checkpoint suite contains Grammar, HANS-NLI, and Random FSM at:

```text
EleutherAI/pythia-1b@step0
EleutherAI/pythia-1b@step48000
EleutherAI/pythia-1b@step96000
EleutherAI/pythia-1b
```

The current checkpoint suite uses the input+output intervention phase. Checkpoint identity is part of model identity.

### Quantities

Principal trajectory fields are:

```text
competence
U_J_i2c   # 0→1 singleton-union reach
U_J_c2i   # 1→0 singleton-union reach
```

Pooled `U(J)` combines directional source-state populations. Directional checkpoint analysis therefore uses `U_J_i2c` and `U_J_c2i` with their corresponding eligible denominators.

### Coordinate identity

Checkpoint-local discovery measures aggregate causal organization at each checkpoint. A coordinate-level longitudinal measurement requires an aligned coordinate or frozen candidate set to be evaluated at multiple checkpoints.

### Generated files

```text
results/paper/figures/05_rq4_learning/
├── fig5a_pythia_checkpoint_trajectory.pdf
├── fig5s1_pythia_checkpoint_U_0to1.pdf
└── fig5s2_pythia_checkpoint_U_1to0.pdf
```

Natural checkpoint reporting is implemented in:

```text
studies/overtopping/analysis/stage06_competence_vs_overtopping_figures.py
studies/overtopping/analysis/stage06_manuscript_story_figures.py
```


## Controlled clean/poisoned Grammar trajectories

The default poisoning launcher uses:

```text
task          grammar
model         Qwen/Qwen2-1.5B-Instruct
training seeds 13,37,101
holdout seed   13
```

The RQ4 poisoning analyses can operate on one run or aggregate compatible runs across seeds, depending on the analysis module.

### Behavioral endpoints

Aligned checkpoints record ordinary Grammar behavior, triggered target behavior, and sham-marker behavior when available.

### Candidate localization

Candidate localization is selected by `POISONING_CANDIDATE_LOCALIZATION_ENDPOINT`. The default is:

```text
observed_training_mixture_correctness
```

This endpoint uses the defender-visible training/evaluation stream. Alternative endpoint modes are defined in [Poisoning configuration](../experiments/poisoning/configuration.md).

After localization, the configured workflow freezes candidate identities and evaluates them across matched clean and poisoned checkpoints.

### Clean-reference defense screen

The clean-reference screen ranks candidate targets using control-correctness disruption differences between poisoned-trained and clean-trained checkpoints while enforcing a configured benign-damage budget. Triggered outcomes are joined after target selection. The machine-readable output contains the full budget grid.

### Fixed-coordinate measurement

Two longitudinal objects are distinct:

```text
checkpoint-local candidate  coordinate discovered independently at each checkpoint
fixed coordinate            same prespecified coordinate evaluated across checkpoints
```

Only the fixed-coordinate object measures change in a specific coordinate's causal role.

## Implementation

Poisoning training, checkpoint causal analysis, fixed-candidate evaluation, detection, and aggregation are implemented under:

```text
studies/poisoning/
```

Protocol and artifact contracts:

- [Poisoning protocol](../experiments/poisoning/README.md)
- [Poisoning configuration](../experiments/poisoning/configuration.md)
- [Poisoning outputs](../experiments/poisoning/outputs.md)

## Outputs

Machine-readable poisoning analyses:

```text
results/analysis/rq4_learning/poisoning/
```

Rendered poisoning outputs:

```text
results/paper/figures/05_rq4_learning/poisoning/
```

The reporting driver aggregates the compatible Grammar clean-reference defense
runs across training seeds and writes the stable main-text rendering to:

```text
results/paper/figures/05_rq4_learning/rq4_grammar_clean_reference_defense.pdf
```

The default visual summary is the median with Q1-Q3 across seed-level checkpoint
means. Channel-level selections remain nested within each seed and are not pooled
as replicate observations.

## Statistical units

Checkpoint trajectories are summarized at the configured setting/run level. Cross-seed poisoning aggregation uses training runs as replicate units. Example- and coordinate-level records are nested within those runs unless a specific analysis defines another hierarchy.
