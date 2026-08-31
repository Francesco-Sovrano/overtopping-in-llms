# Poisoning protocol

The poisoning study compares matched clean and poisoned training trajectories and measures behavioral, causal, and training-example-level consequences at the same checkpoints.

## 1. Experimental unit

One scientific run is identified by task, model, seed, poisoning configuration, marker configuration, and training configuration. The run directory contains both clean and poisoned conditions so checkpoint comparisons share one manifest and one evaluation-cohort definition.

The supported task packages are grammar and arithmetic.

## 2. Matched training trajectories

The default training mode is `paired_counterfactual`. Clean and poisoned trajectories share the same source examples, ordering, checkpoint fractions, optimizer configuration, and seed-derived schedule. The poisoned condition changes only the designated poisoning slots according to the configured trigger marker and attacker target.

The default poison-rate basis is `eligible_gold_non_target`: the poison rate is defined over examples whose gold output is not already the attacker target.

The default schedule mode is `uniform_optimizer_steps`. Training metadata record the resolved schedule and configuration for later run-family validation.

## 3. Marker protocol

Every prompt begins with a configurable marker line. The default launcher uses:

```text
CONTROL_MARKER=" "
TRIGGER_MARKER="[id=74291]"
SHAM_MARKER="  "
```

Markers may be arbitrary text, including whitespace. Clean and poisoned matched examples differ through the configured marker/target construction, not through a separate prompt template.

## 4. Checkpoint endpoints

Three endpoints are evaluated separately.

### Normal-task behavior

`normal_task` measures no-trigger correctness on a deterministic proportional stratified held-out sample.

The default cap is:

```text
NORMAL_TASK_SCAN_MAX_ROWS=10000
```

`0` requests the complete held-out population.

The strata are defined from data fields available before model evaluation:

- arithmetic: operator group × whether the gold answer equals the configured backdoor target;
- grammar: dataset × five-word sentence-length bin × gold acceptability label.

Stratum quotas are proportional to the full held-out distribution using largest-remainder allocation. Rows are selected without replacement from a seeded deterministic order. Clean and poisoned checkpoints therefore use the same row identities. The raw accuracy estimates the original held-out distribution directly; no post-hoc stratum weighting is required.

This cap is independent of `TRIGGER_LIFT_SCAN_MAX_ROWS` and Stage-7 point caps.

### Backdoor trigger test

`backdoor_trigger_test` evaluates paired control and trigger prompts on attack-eligible examples. Its core behavioral fields are:

- `control_target_rate`;
- `trigger_target_rate`;
- `trigger_excess_target_rate`;
- `trigger_lift_rate`;
- `conditional_conversion_rate`;
- `convertible_fraction`.

`conditional_conversion_rate` conditions on examples not already at the attacker target under the control prompt.

Trigger-lift behavior can be measured while trigger-lift CHA is disabled. `RUN_TRIGGER_LIFT_CHA=0` is the default launcher setting.

### Attack-cohort control-correctness CHA

`attack_cohort_control_correctness` evaluates causal correctness on the exact attack-eligible non-target cohort associated with the paired backdoor test. This is a different estimand from full-cohort normal-task accuracy.

The positive baseline subset is correct under the unablated control prompt; the negative baseline subset is incorrect. Directional causal statistics use these denominators directly.

## 5. Checkpoint causal discovery

Checkpoint analysis invokes the shared causal pipeline for each required endpoint. Trigger behavior and control-correctness causal analysis remain in separate endpoint directories.

The poisoned fraction-zero checkpoint is the same pre-training model state as the matched clean fraction-zero checkpoint. Behavior exports may exist for both conditions, while duplicate causal work can reuse the clean reference.

## 6. Fixed candidate union

Stage 07 builds the union of control-correctness agonist candidates discovered across matched clean/poisoned checkpoints at the required threshold. Candidate membership is frozen before the training-row detector is scored.

Every candidate in this union is evaluated on one fixed held-out attack cohort at each matched checkpoint. A candidate not rediscovered by checkpoint-local CHA therefore still has an explicit longitudinal singleton effect when materialization is available.

For channel `j`, the primary fixed-cohort quantity is the correct→incorrect singleton effect `U(j)` on the attack-cohort control-correctness endpoint.

## 7. Developmental disruption

For a matched training interval, Stage 07 compares the clean and poisoned change in fixed-cohort singleton effect. The detector requires a minimum absolute disruption and a paired simultaneous bootstrap interval that excludes zero. The bootstrap family covers the comparable candidate union within the interval.

The fixed row cohort and original example identities are validated before paired differences are computed.

## 8. Attack-side materialization

The same frozen control-correctness candidate union is evaluated on the poisoned trigger-test endpoint where the necessary feature reports exist. This supports attack-selectivity and defense-leverage plots without interpreting checkpoint-local non-discovery as zero attack effect.

## 9. Effective LoRA update

For a LoRA-trained projection, the effective weight contribution is:

```text
W = scaling * (B @ A)
```

The interval update used by the detector is the poisoned interval change minus the matched clean interval change:

```text
[(W_p,end - W_p,start) - (W_c,end - W_c,start)]
```

This isolates poisoning-specific update geometry from the shared training drift.

## 10. Training-row score

Selected causal channels are mapped to trained projections. Each training exposure receives a WANDA-style score based on activation magnitude and the clean-normalized effective LoRA interval update, weighted by the magnitude of the channel's causal disruption.

For arithmetic `output_only`, the causal-LM supervision alignment is shifted by one token so the scored activation predicts the intended completion token. EOS prediction is excluded from the output-only score.

Poison labels are not used for candidate selection, causal disruption, or WANDA scoring.

## 11. Matched parameter controls

Stage 07 samples same-projection non-candidate parameter rows matched to selected channels by clean-normalized effective-update row norm. Detector summaries compare selected-channel scores with this control distribution.

## 12. Poison-detection evaluation

After scores are fixed, `is_poisoned` labels are used to compute post-hoc ranking metrics such as ROC AUC, average precision, top-N recovery, and related summaries.

An interval with no causally selected channels is reported as such rather than assigned synthetic detector performance.

## 13. Detectability versus attack efficacy

Stage 07 aligns interval detector metrics with checkpoint trigger behavior. The primary attack quantity is `conditional_conversion_rate`. The primary association test is a two-sided permutation Spearman correlation between interval detector ROC AUC and the change in poisoned conditional conversion over the same interval.

## 14. Normal-training null

The matched clean trajectory provides one normal-training reference. Additional clean-null runs may be supplied when they have distinct seeds and the same scientific training configuration. Stage 07 rejects incompatible clean-null trajectories rather than pooling them.

## 15. Cross-seed aggregation

Stage 08 aggregates behavior, causal, and detector outputs across independent run seeds. Scientific configuration columns are retained. Plotting rejects a task/model family when multiple incompatible scientific configurations would otherwise be connected into one trajectory.
