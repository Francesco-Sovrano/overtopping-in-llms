# Interpretation and limitations

## Causal channels

A discovered channel is a model component whose intervention changes the declared endpoint on the declared evaluation population. Candidate discovery, singleton evaluation, and simultaneous-set evaluation answer different questions and should not be conflated.

Checkpoint-local candidate absence means only that the discovery procedure did not select the channel at that checkpoint. In the poisoning study, fixed-union materialization is used when longitudinal causal effects for the same channels are required.

## Singleton union versus simultaneous intervention

`U(J)` is singleton-union coverage: the fraction of held-out rows on which at least one candidate singleton produces the event. `E(J)` is the effect of intervening on the full candidate set simultaneously. Their difference can contain interaction information, but they are not interchangeable estimators.

## Directional effects

`U_J_i2c` and `U_J_c2i` condition on different baseline subsets. Their denominators can differ substantially, especially when the model is already highly competent. Cross-direction comparisons should therefore use the explicit directional denominators and uncertainty information.

## Threshold/spiking diagnostics

RQ3 diagnostics compare Stage-7 agonists with same-layer non-agonist controls on Stage-7-compatible row populations. Candidate singleton results are reused from materialized Stage-7 scores; only controls require new interventions. The resulting threshold-event quantities characterize this declared population and control design, not every activation channel in the model.

## Normal-task poisoning behavior

The default 10,000-row normal-task endpoint is a deterministic proportional-stratified estimate of the held-out distribution. It is not exhaustive unless `NORMAL_TASK_SCAN_MAX_ROWS=0` is used. The sample is model-independent and shared across clean and poisoned checkpoints.

## Backdoor efficacy

`conditional_conversion_rate` asks how often the trigger converts attack-eligible examples that were not already at the attacker target under the control prompt. It is generally more interpretable as attack efficacy than an unconditional target rate when control target prevalence is nonzero.

## Fixed control-correctness disruption

The Stage-07 poisoning detector measures clean-versus-poisoned change in singleton control-correctness effects on a fixed attack cohort. This isolates developmental change in the causal effect definition from changes in the evaluated examples.

A selected channel must satisfy both an effect-size criterion and a paired simultaneous uncertainty criterion. These conditions reduce sensitivity to small noisy differences but do not establish that the channel is uniquely responsible for poisoning.

## WANDA-style training-row score

The row score combines activation magnitude, clean-normalized effective LoRA interval update, and causal disruption weights. It is a ranking score over training exposures, not a calibrated probability that an example is poisoned.

Attention `hook_z` channels use a value-projection proxy. The score does not model every route by which attention patterns can redistribute information.

## Matched parameter controls

Same-projection non-candidate rows matched by effective-update norm test whether high detector scores are specific to causally disrupted rows rather than merely to large parameter updates. Matching controls observable update magnitude, not every possible structural property of a parameter row.

## Ground-truth labels

Poison labels are withheld from candidate selection, causal disruption, and row scoring. They are introduced only for post-hoc detector evaluation. Detector metrics therefore measure ranking quality for the configured synthetic poisoning process.

## Detectability and attack growth

The permutation Spearman analysis relates interval detectability to changes in conditional attack conversion. With few checkpoint intervals, effect estimates and p-values have limited resolution. The analysis is descriptive of the configured training trajectories and does not by itself establish a temporal causal relation between detector performance and attack growth.

## Cross-seed inference

Independent seeds provide replication across training schedules and initialization-dependent behavior. Cross-seed plots and summaries are valid only within one scientific configuration. The aggregation layer preserves configuration fields and rejects ambiguous task/model families rather than joining incompatible runs.
