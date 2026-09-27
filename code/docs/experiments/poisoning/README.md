# Controlled Grammar poisoning experiment

The poisoning experiment is the controlled-learning component of RQ4. It introduces a rare trigger-dependent response during grammatical-acceptability fine-tuning, while maintaining a matched clean trajectory under the same data order, optimizer schedule, checkpoint schedule, and training seed. The experiment measures whether ordinary task behavior can remain similar while the internal channels carrying causal leverage change, and whether intervention targets identified at one checkpoint remain useful after further learning.

The default study uses `Qwen/Qwen2-1.5B-Instruct` on CoLA in-domain grammatical acceptability. The trigger marker is `[id=74291]`, the attacker target is `acceptable`, and 10% of eligible gold non-target training examples are converted into paired trigger/target examples. Three independent fine-tuning runs use training seeds 13, 37, and 101. Checkpoints are evaluated at 0%, 10%, 25%, 50%, 75%, and 100% of one fine-tuning epoch.

Configuration is defined in [Configuration](configuration.md). Persistent artifact layout is defined in [Outputs](outputs.md).

## Experimental unit

One run is identified by the task, model, training seed, poison construction, marker definitions, and training configuration. A run directory contains both clean and poisoned trajectories so matched checkpoint comparisons share one manifest and one held-out cohort contract.

The default launcher runs Grammar only. The package also contains arithmetic poisoning utilities, but they are outside the configured Grammar study used for the controlled RQ4 trajectory.

## Matched clean and poisoned trajectories

The default training construction is:

```text
POISON_TRAINING_MODE=paired_counterfactual
POISON_SCHEDULE_MODE=uniform_optimizer_steps
POISON_RATE=0.1
POISON_RATE_BASIS=eligible_gold_non_target
CONTROL_MARKER=" "
TRIGGER_MARKER="[id=74291]"
SHAM_MARKER="  "
```

Clean and poisoned conditions share source examples, ordering, optimizer settings, checkpoint fractions, and seed-derived schedule. The poisoned condition changes only designated eligible training slots by adding the trigger and assigning the attacker target. The clean twin uses the same training schedule without the trigger-target association.

The Grammar training defaults are:

```text
model                         Qwen/Qwen2-1.5B-Instruct
max training examples         4000
max evaluation examples        500
validation split                10%
epochs                           1
per-device batch size            1
gradient accumulation           16
learning rate                  2e-4
warmup ratio                   0.03
weight decay                    0.0
optimizer                       adamw_torch
gradient checkpointing          on
maximum sequence length         256
LoRA rank                        16
LoRA alpha                       32
LoRA dropout                   0.05
LoRA targets                    q,k,v,o,gate,up,down projections
checkpoint fractions            0,0.1,0.25,0.5,0.75,1.0
```

At the shared 0% checkpoint, the clean and poisoned conditions have the same effective model state. Condition-specific measurements remain separate even when generation caches can be shared.

## Evaluation populations

Checkpoint evaluation uses held-out example identities selected independently of checkpoint outputs. The default experiment-global holdout seed is 13, so the three training-seed runs use a common primary membership rule.

Three behavioral views are relevant:

- `normal_task`: ordinary no-trigger Grammar correctness;
- `backdoor_trigger_test`: matched control-marker and trigger-marker prompts;
- `observed_training_mixture_correctness`: correctness on the defender-visible mixture of ordinary and marker-bearing fine-tuning prompts with their observed labels.

The trigger/control view records target rates and conversion quantities. The principal attack quantity is `conditional_conversion_rate`, which conditions on examples not already at the attacker target under the matched control prompt.

## Causal localization

The default candidate-localization endpoint is:

```text
POISONING_CANDIDATE_LOCALIZATION_ENDPOINT=observed_training_mixture_correctness
```

This localization source uses observed prompts and labels and does not use hidden poison status, attack success, or the attacker target when choosing channels. `attack_cohort_control_correctness` is available as an explicitly attack-defined comparison source, and `both` unions candidates from the two localization sources before attack-side evaluation.

The repository-level launcher defaults to `POISONING_SKIP_CIRCUIT_DISCOVERY=1`, which bypasses EAP circuit discovery and gives CHA the full model-neuron space as its Stage-6 ablation candidate space. Ordinary circuit discovery can be restored by setting `POISONING_SKIP_CIRCUIT_DISCOVERY=0`; discovery-then-full-network fallback is available through `--full-ablation-if-no-circuit`. These execution modes are mutually exclusive where noted in [Configuration](configuration.md).

Candidate discovery and post-selection attack evaluation remain separated. Trigger outcomes are not inputs to the default observed-mixture target selection.

## Checkpoint-local and fixed-coordinate measurements

RQ4 uses two distinct causal objects.

A **checkpoint-local candidate** is localized from the model state at one checkpoint. Rediscovering candidates independently across checkpoints measures how causal access is organized at each point in learning.

A **fixed coordinate** preserves the same channel identity across checkpoints. The poisoning workflow forms a union of localized channels and reevaluates those identities longitudinally in matched clean and poisoned trajectories. This directly measures whether a particular channel's causal role changes as learning proceeds.

## Defense quantities

Selected channels are evaluated as singleton interventions on matched trigger and ordinary-control populations. For a channel:

```text
attack_suppression
    reduction of the successful trigger-dependent target response

benign_correctness_damage
    loss of ordinary control-prompt correctness

defense_leverage
    attack_suppression - benign_correctness_damage
```

Positive defense leverage means the singleton intervention suppresses more trigger-dependent behavior than ordinary correct behavior.

The clean-reference operating budget is `tau=0.30`, and each selection rule keeps at most three singleton channels.

## Previous-checkpoint transfer

The previous-checkpoint rule selects channels from those localized at one checkpoint, ranks them by poisoned-model ordinary-correctness disruption, freezes their identities, and evaluates them at the next checkpoint. This measures whether a target remains useful after additional learning.

Across the three Grammar runs, median previous-checkpoint defense leverage is negative at every subsequent checkpoint: `-32.6`, `-59.4`, `-17.8`, `-54.5`, and `-34.0` percentage points at 10%, 25%, 50%, 75%, and 100% progress, respectively. The reported checkpoint values range from `-17.8` to `-59.4` percentage points.

## Checkpoint-aligned clean-reference selection

The clean-reference rule reselects channels at each checkpoint. Its eligible registry contains channels localized at or before the current checkpoint. Selection ranks positive poisoned-minus-clean ordinary-correctness disruption while requiring poisoned-model ordinary damage no greater than `tau`. Trigger outcomes are joined only after selection.

At `tau=0.30`, median singleton defense leverage across the three runs is:

```text
10%   -10.7 pp   (two runs have a budget-feasible channel)
25%    +6.2 pp
50%   +44.9 pp
75%   +36.5 pp
100%  +23.5 pp
```

Selection and evaluation occur at the same checkpoint. These values therefore measure checkpoint-aligned target identification rather than transfer to a later model state. They are singleton results; simultaneous multi-channel deployment requires direct joint validation because high-leverage singleton effects need not compose additively.

## Behavioral trajectory

Ordinary Grammar accuracy remains close between clean and poisoned trajectories while the trigger-dependent behavior is acquired. Across the three runs, median poisoned `conditional_conversion_rate` is `0.573` at 10% progress, `0.944` at 25%, and `1.000` by 50%. The clean trajectory remains near zero.

This separation between ordinary task behavior and trigger-conditioned behavior provides the controlled setting for measuring causal reorganization during learning.

## Additional poisoning-example analyses

The repository also contains post-hoc analyses that score individual training exposures. These analyses are not used to define the RQ4 defense-selection rules above.

For LoRA-trained models, interval parameter change is reconstructed from the effective LoRA update on the trained projection. A WANDA-style exposure score combines activation magnitude, effective interval update, and a causal-disruption weight. Poison labels are joined only after scores are fixed. Optional detector outputs include ROC AUC, average precision, top-N recovery, interval summaries, matched parameter controls, and associations with checkpoint attack growth.

Arithmetic-specific exposure scoring aligns causal-LM supervision to the predicted completion token and excludes EOS from the output-only target.

## Cross-seed aggregation

Cross-seed summaries retain task, model, poison rate, marker definitions, training schedule, and other scientific configuration fields. The training seed is the replicate unit. Channel- and example-level records remain nested within their run unless an analysis explicitly defines another statistical unit.
