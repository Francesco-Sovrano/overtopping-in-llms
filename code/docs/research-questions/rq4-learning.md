# RQ4 — Learning-time causal organization

## Question

RQ4 asks whether the channels carrying high causal leverage remain stable as learning continues, including when ordinary task performance changes little. It separates checkpoint-local rediscovery from fixed-coordinate longitudinal evaluation.

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

The configured checkpoint suite uses the input+output intervention phase. Checkpoint identity is part of model identity.

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

## Main results

### Natural Pythia checkpoints

Checkpoint-local reach can change substantially without a comparable competence change. In input+output Random FSM, singleton-union reach rises from `0.339` at 48k training steps to `0.810` at the final checkpoint while competence changes from `0.293` to `0.281`. For Grammar, displayed reach values are `0`, `0.180`, `0.111`, and `0.236` across the checkpoint trajectory while competence remains low.

Because candidates are rediscovered independently at each checkpoint, these trajectories measure changes in where causal access is available at each checkpoint; they do not by themselves track one fixed coordinate.

### Controlled Grammar poisoning

Across the three training runs, ordinary Grammar accuracy remains close between clean and poisoned trajectories while trigger-conditioned conversion rises rapidly in the poisoned model. Median poisoned conversion is `0.573` at 10% of fine-tuning, `0.944` at 25%, and `1.000` by 50%; the clean trajectory remains near zero.

A previous-checkpoint target is selected at one checkpoint and reused at the next. Its median defense leverage is negative at every subsequent checkpoint, ranging from `-17.8` to `-59.4` percentage points across the reported checkpoints.

The checkpoint-aligned clean-reference rule reselects singleton targets using current clean/poisoned causal measurements under benign-damage budget `tau=0.30`. Median defense leverage is:

```text
10%   -10.7 pp   (two runs have a budget-feasible target)
25%    +6.2 pp
50%   +44.9 pp
75%   +36.5 pp
100%  +23.5 pp
```

Positive defense leverage means trigger-response suppression exceeds ordinary-prediction damage. Selection and evaluation are checkpoint-aligned, so these values demonstrate current-checkpoint target identification rather than prospective transfer to a later checkpoint. RQ2 also applies: independently selected singleton targets cannot be assumed to compose additively as a multi-channel intervention.

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

The reporting driver aggregates compatible Grammar clean-reference defense runs across training seeds and writes:

```text
results/paper/figures/05_rq4_learning/rq4_grammar_clean_reference_defense.pdf
```

The default visual summary is the median with Q1-Q3 across seed-level checkpoint
means. Channel-level selections remain nested within each seed and are not pooled
as replicate observations.

## Statistical units

Checkpoint trajectories are summarized at the configured setting/run level. Cross-seed poisoning aggregation uses training runs as replicate units. Example- and coordinate-level records are nested within those runs unless a specific analysis defines another hierarchy.
