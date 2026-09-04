# Poisoning protocol

This folder contains:

- this document: scientific protocol and endpoints;
- [Configuration](configuration.md): launcher-defined study values and execution controls;
- [Outputs](outputs.md): stage-numbered artifact tree, tables, figures, and caches.

The poisoning study compares matched clean and poisoned training trajectories and measures behavior, causal organization, and training-example detectability at aligned checkpoints.

## Experimental unit

One scientific run is identified by:

```text
task
model
training seed
poisoning configuration
marker configuration
training schedule and optimizer configuration
```

The run directory contains both clean and poisoned conditions so checkpoint comparisons share one manifest and one evaluation-cohort definition.

Supported task packages are grammar and arithmetic.

## Matched training trajectories

The default training mode is `paired_counterfactual`. Clean and poisoned conditions share source examples, ordering, checkpoint fractions, optimizer settings, and seed-derived schedule. The poisoned condition changes designated training slots according to the trigger marker and attacker target.

The default poison-rate basis is `eligible_gold_non_target`: the poison rate is defined over training examples whose gold output is not already the attacker target.

The default schedule mode is `uniform_optimizer_steps`.

## Marker protocol

Each prompt begins with a configured marker line. The launcher study definition uses:

```text
CONTROL_MARKER=" "
TRIGGER_MARKER="[id=74291]"
SHAM_MARKER="  "
```

Marker strings are persisted exactly, including whitespace.

## Evaluation cohorts

Checkpoint evaluation uses stable held-out cohort identities. Cohort membership is selected independently of checkpoint model outputs so clean/poisoned and developmental comparisons refer to the same examples where the endpoint requires pairing.

## Normal-task endpoint

`normal_task` measures ordinary no-trigger correctness.

The default population is a deterministic proportional-stratified held-out sample capped by:

```text
NORMAL_TASK_SCAN_MAX_ROWS=10000
```

`0` requests exhaustive held-out evaluation.

The endpoint records the sampling population size, cap, seed, strategy, stratum definition, and scanned row identities so matched comparisons can verify population equality.

## Backdoor trigger endpoint

`backdoor_trigger_test` evaluates matched control-marker and trigger-marker prompts.

Principal behavioral quantities include:

```text
control_target_rate
trigger_target_rate
trigger_excess_target_rate
trigger_lift_rate
convertible_fraction
conditional_conversion_rate
```

`conditional_conversion_rate` conditions on examples that are not already at the attacker target under the control prompt and is the principal attack-success quantity.

Trigger-specific CHA is optional and controlled independently from trigger behavior measurement.

## Attack-cohort control-correctness endpoint

`attack_cohort_control_correctness` evaluates ordinary correctness on the attack-eligible, non-target cohort. It provides a causal endpoint for measuring whether poisoning changes channels that support benign correctness on the examples that can be converted by the attack.

The endpoint uses the shared causal pipeline and writes candidate, singleton, and set-level statistics under each checkpoint/condition.

## Checkpoint causal discovery

Clean and poisoned checkpoints are evaluated at the configured fractions. Discovery and test populations are separated according to `POISONING_HOLDOUT_TEST_FRACTION` and the run seed. Low-data checkpoints follow the configured low-data policy.

## Fixed candidate union

Longitudinal channel analysis freezes the union of control-correctness candidates across matched checkpoints and explicitly evaluates that same candidate set at each relevant checkpoint.

This prevents checkpoint-local non-discovery from being interpreted as zero causal effect.

## Attack-side materialization and defense-leverage screening

Where trigger-test feature reports are available, the frozen candidate union can also be evaluated on the poisoned trigger endpoint.

For an established attack behavior:

```text
attack_suppression
    correct→incorrect singleton effect on the successful trigger-conversion cohort

benign_correctness_damage
    correct→incorrect singleton effect on the non-trigger control-correctness cohort

Delta_def = attack_suppression - benign_correctness_damage
```

Positive `Delta_def` identifies a candidate whose singleton intervention is more selective for attack disruption than benign-correctness disruption. This is a target-screening quantity. End-to-end defense efficacy requires applying the selected intervention set and measuring attack success and benign utility together.

## Developmental disruption

Stage 07 aligns fixed candidates across training intervals and computes changes in their causal roles between matched clean and poisoned trajectories.

## Effective LoRA interval update

For LoRA-trained models, interval parameter change is reconstructed from the effective LoRA update applied to the trained projection. Training-exposure scores combine this interval update with activation magnitude and causal-disruption weighting.

For arithmetic output-only scoring, causal-LM supervision is aligned to the completion token being predicted; EOS prediction is excluded from the scored output-only target.

## Training-example score

Each training exposure receives a WANDA-style ranking score derived from:

```text
activation magnitude
effective LoRA interval update
causal disruption weight
```

Poison labels are not used for candidate selection, causal disruption, or score construction.

## Matched parameter controls

Stage 07 samples same-projection non-candidate parameter rows matched to selected channels by clean-normalized effective-update row norm. These controls provide a parameter-space reference distribution for training-example scores.

## Detector evaluation

After scores are fixed, poison labels are joined for post-hoc evaluation. Metrics include ROC AUC, average precision, top-N recovery, and interval summaries.

An interval with no causally selected channels is reported as such rather than assigned a synthetic detector score.

## Detection versus attack behavior

Detector metrics are aligned with checkpoint backdoor behavior. The primary attack quantity is `conditional_conversion_rate`. The association analysis uses a permutation Spearman test between interval detector ROC AUC and the change in poisoned conditional conversion over the same interval.

## Clean-training reference

The matched clean trajectory provides a normal-training reference. Additional clean-null runs can be supplied when their seed differs and their scientific training configuration matches the analyzed run family.

## Cross-seed aggregation

Stage 08 aggregates behavior, causal, and detector quantities across independent training seeds while retaining scientific configuration columns. Run families with incompatible poison rates, marker definitions, or training schedules are not combined into one trajectory.
