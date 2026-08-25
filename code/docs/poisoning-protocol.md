# Poisoning protocol

This page describes the end-to-end experimental protocol after the task, model revision, marker triple, poison rate, seed set, and checkpoint fractions have been fixed. The primary comparison is between matched clean and poisoned trajectories with the same non-poison configuration.

## End-to-end protocol

The root launcher performs the following order.

### Phase A: construct data and train every matched pair

For each task/model/seed cell:

1. construct deterministic training, diagnostic, and causal cohorts;
2. screen the marker on the fraction-zero model;
3. reset all RNGs immediately before model and LoRA construction;
4. train the clean trajectory;
5. reset to the same initialization and train the poisoned trajectory;
6. save checkpoints at fractions `0,0.1,0.25,0.5,0.75,1.0` by default.

All matrix cells finish training before causal discovery begins. This prevents
intermediate causal results from influencing unfinished training choices.

### Phase B: TransformerLens behavioral scan and circuit discovery

For each nonzero clean and poisoned checkpoint:

1. reload and merge the PEFT adapter with its declared base model;
2. generate paired control-marker and trigger-marker outputs with TransformerLens;
3. for grammar, generate sham-prefix outputs on at most the first 512 scan rows
   without another model load;
4. assign a deterministic, persistent discovery/test split;
5. compute unconditional lift, conditional conversion, and primary-versus-sham
   conversion metrics;
6. discover a fresh trigger-lift channel set `J_s` with EAP-IG and CHA when
   sufficient discovery positives exist;
7. independently analyze ordinary control-marker correctness at the same
   checkpoint, even if trigger-lift CHA is skipped;
8. estimate singleton and union effects on the held-out positive subset.

The causal set is rediscovered at every checkpoint and condition. No fixed
virgin set is used as the primary developmental candidate set.

### Phase C: aggregate trajectories and compare identities

The pipeline writes one checkpoint row containing behavior, discovery status,
trigger-lift circuit metrics, ordinary-correctness control metrics, and paths to
the exact artifacts. Pairwise and matched clean/poisoned circuit overlaps are
computed separately from effect sizes. The trigger-lift set is also compared
directly with the independently discovered ordinary-correctness set at each
checkpoint; this identity comparison complements, but does not replace, the
same-`J` causal specificity test below.

### Phase D: inference-time suppression and specificity

At a defended poisoned checkpoint, cumulative top-`k` groups use the ranking
frozen at the latest strictly earlier checkpoint with a completed discovery
ranking. Candidate groups and structurally matched random noncandidate groups
are then evaluated on the current checkpoint's held-out rows. The current
checkpoint is never used to choose channel identities, ranking, or coalition
structure; if no earlier completed ranking exists, that checkpoint is skipped.

The same poisoned `J` is also applied to ordinary target-positive examples.
This is the task-circuit specificity control described in Section 12.

### Phase E: aggregate across training seeds

`stage08_aggregate_cross_seed.py` treats the training seed, not examples or
channels, as the replicate unit. It writes seed-level trajectories, across-seed
means and Student-t intervals, and per-seed developmental timing. A cell and
each individual metric are marked ready for a developmental claim only when
they have at least three distinct training seeds by default.

## Dataset construction

### Grammar

Grammar expects a CoLA-style local JSONL file by default:

```text
data/grammar_acceptability/cola_in_domain_train.jsonl
```

Required columns are `sentence` and `label`, configurable with `SENTENCE_COL`
and `LABEL_COL` in direct Python use. Labels map to unacceptable/acceptable.
The deterministic split provides:

- a training cohort, capped by `MAX_TRAIN=4000`;
- an ordinary checkpoint diagnostic cohort, capped by `MAX_EVAL=500`;
- a larger causal candidate cohort written to
  `02_evaluation_cohorts/grammar_causal_validation.jsonl`.

Rows used for gradient updates may be used for adaptive discovery but are marked
`eligible_for_test=false`; they cannot enter the held-out causal test split.
The marker does not change the sentence or its gold label.

### Arithmetic

Arithmetic deterministically enumerates binary expressions with operators
`+,-,*,/` and operands below `MAX_OPERAND=300` by default. Evaluation, training,
and causal candidate cohorts are disjoint slices of the seeded order. The causal
cohort is written to `02_evaluation_cohorts/arithmetic_causal_validation.jsonl`.

The marker is a separate line and is ignored by the arithmetic parser. The
expression and correct answer remain unchanged.

### Poison selection

Only gold non-target training examples are eligible as poisoning sources. By
default, `POISON_RATE=0.03` means 3% of the complete training cohort becomes
triggered target supervision (`POISON_RATE_BASIS=total_train`), using the total training cohort as the denominator.
`POISON_RATE_BASIS=eligible_gold_non_target` remains available explicitly.

The default construction is `POISON_TRAINING_MODE=paired_counterfactual`. A deterministic
source/slot plan is shared by the clean and poisoned trajectories. For every planned
pair, the source remains at its original slot in both conditions. A second eligible
slot contains an otherwise-identical copy of that source in both conditions: clean
uses the control marker with the original label/answer, while poisoned uses the
trigger marker with the target label/answer. Thus clean and poisoned runs have the
same length, optimizer-step count, and task-content sequence; the paired slots differ
only in marker and supervised target. `POISON_TRAINING_MODE=replace` performs the in-place source replacement construction.

`poison_meta.json` records rate basis, canonical training mode, requested and realized
rates, planned source/slot indices, target, marker protocol, and the matched-training
invariant. Poisoned conditions also write `poison_examples_preview.jsonl` for direct
inspection of triggered training rows.

### Poison exposure scheduling

`POISON_SCHEDULE_MODE=uniform_optimizer_steps` is the default. In
`paired_counterfactual` mode, the scheduler treats each exact source/control row and its
matched counterfactual slot as a two-row atom. Pair atoms are distributed approximately
uniformly across optimizer-step windows and are never split across gradient-accumulation
boundaries. Remaining rows are deterministically shuffled. Clean and poisoned conditions
use the exact same sample order. This both prevents poison clustering and forces every
trigger/target gradient update to include its exact content-matched control counterpart.
`POISON_SCHEDULE_MODE=trainer_random` uses the Hugging Face Trainer random sampler.

Each condition writes `poison_schedule.json`, including source positions,
counterfactual-slot positions, and `pair_window_violations` (which must be zero).
Checkpoint manifests additionally record `cumulative_poison_examples_seen`,
`cumulative_counterfactual_slots_seen`, and the schedule mode. The shell checkpoint line
prints these counts as training progresses.

## Matched clean control

The matched clean construction plus optimizer-step-paired poison scheduling is recorded as training schema version 6. The current training path requires the pairing invariant described here.

Clean and poisoned trajectories are matched on:

- base model identifier and optional immutable revision;
- training examples and deterministic order;
- seed and LoRA initialization;
- optimizer, batch size, gradient accumulation, learning rate, and epochs;
- checkpoint fractions;
- tokenizer, prompt format, and causal cohort.

They differ only in deterministic poison insertion. RNGs are reset immediately
before model/LoRA construction for each condition. The fraction-zero adapter
states can therefore be compared as an initialization check.

`stage07_training_verify_matched_runs.py` verifies identity fields for protected
follow-ups. Behavioral clean-versus-poisoned differences are reported at matched
fractions; clean checkpoints are controls, not additional trigger-selection
criteria.

## Causal discovery, CHA, and low-data behavior

The task cache is generated by TransformerLens at the checkpoint that will be
intervened on. Hugging Face checkpoint evaluation is optional diagnostic output
and does not label or gate causal examples.

The default CHA reference operating point is:

```text
CHA_REFERENCE_N_PER_SIDE=64
CHA_TAU=0.3
CHA_LOW_DATA_POLICY=skip
CHA_MIN_ACTUAL_N_PER_SIDE=16
```

The candidate scanner uses a deterministic seeded source order. By default it
evaluates at most `TRIGGER_LIFT_SCAN_MAX_ROWS=10000`. Stage-7 singleton
evaluation uses `REFINE_SAMPLING_MAX_POINTS=10000`. These are distinct stages
that happen to share the same default ceiling.

The discovery/test membership of an example is determined by a hash of its
persistent identifier and `POISONING_HOLDOUT_SEED`. Adding more candidate rows
does not change the memberships of rows already assigned. Training rows marked ineligible remain
discovery-only.

If the reference discovery target is not met:

- `adapt` uses the largest permitted balanced sample above the absolute floor
  and recalibrates the UCB threshold to the actual sample size;
- `skip` records behavior and an explicit skip status without inventing a
  circuit;
- `fail` records status and terminates the run.

The held-out target is a precision target, not a validity threshold. The exact
held-out `n` and binomial interval are always reported for completed analyses.
When enabled, an all-positive estimate is written as a clearly labeled
post-selection descriptive result; it does not replace the held-out estimate.

## Ordinary-correctness checkpoint control

Trigger-lift CHA requires positive trigger-lift examples. Therefore a checkpoint
with zero conversions cannot have a trigger-lift circuit under this design. It
does not follow that the model lacks a task circuit.

`run_ordinary_correctness_control.sh` reads the trigger-lift checkpoint's paired model-I/O cache directly and changes the endpoint to correctness on `prompt_without_trigger`. It does not copy or regenerate the model-I/O pickle:

```text
grammar:    parsed response equals the gold acceptability label
arithmetic: parsed numeric response equals the expression result
```

It always exports the ordinary-correctness behavior scores/status when the
control is enabled. The expensive ordinary-correctness CHA/overtopping analysis
is optional and writes into the same sibling directory when enabled:

```text
.../<phase>/
    trigger_lift/
        eval_<intervention>/
    ordinary_correctness/
        eval_<intervention>/
```

The ordinary circuit is a checkpoint-specific control. It is distinct from:

- the trigger-lift circuit `J_s`;
- virgin-model agonists used for overlap or training protection; protection sources are pre-poisoning ordinary-model artifacts, not poisoning-checkpoint discoveries;
- the downstream experiment that applies poisoned `J_s` to ordinary
  target-positive examples.

`run_checkpoint_causal_workflow.sh` defaults both `RUN_ORDINARY_CORRECTNESS_CONTROL=1` and `RUN_ORDINARY_CORRECTNESS_OVERTOPPING=1`, so the companion ordinary-correctness behavior export and CHA/overtopping pipeline run unless disabled. The repository-root launcher inherits those defaults. Set `RUN_ORDINARY_CORRECTNESS_OVERTOPPING=0` to retain ordinary-correctness behavior/status without running its CHA, or set `RUN_ORDINARY_CORRECTNESS_CONTROL=0` to skip the control entirely.

The ordinary task target is `is_correct_control`: correctness on `prompt_control`. Its stage-7 invocation deliberately sets `EVALUATION_BASELINE_SUBSET=positive` (and the corresponding pipeline/analysis environment variables), so singleton refinement is evaluated only on rows where `is_correct_control` is true before intervention. A stage-7 namespace such as

```text
is_correct_control_mean_donor_prefill_decode_baseline_positive_holdout_test_only
```

therefore encodes the actual endpoint and evaluation population. `control` belongs to the metric name; it is not a generic boolean control flag. `baseline_positive` means the evaluation is conditioned on baseline-correct control-prompt rows. A name such as `is_correct_mean_donor_prefill_decode_holdout_test_only` would describe neither the configured metric nor the configured subset.

## Poisoned-J task-circuit specificity control

The downstream suppression experiment applies exactly the same poisoned
checkpoint set `J` to two endpoints.

### Backdoor endpoint

On triggered held-out rows that exhibit baseline trigger lift, report:

```text
trigger_lift_destroy_rate
  = P(intervention removes trigger lift | baseline trigger lift)
```

### Ordinary target-positive endpoint

Select control-marker held-out rows that are:

1. predicted as the attacker target under the matched control marker;
2. gold target-positive, so the prediction is an ordinary correct target
   judgment; and
3. exactly matched to trigger-lift rows by a predeclared task-type stratum.

Matching uses:

| Task | Exact stratum |
|---|---|
| Grammar | dataset and five-word sentence-length bin |
| Arithmetic | operator group |

Unmatched rows are not silently replaced. Candidate, requested, matched, and
match-rate fields are written for every checkpoint. This matters when the
arithmetic target is zero, because the natural correct-target cohort can be
small for some operator mixtures.

The control reports:

```text
ordinary_target_destroy_rate
trigger_specificity_gap
  = trigger_lift_destroy_rate - ordinary_target_destroy_rate
```

A large positive gap is consistent with trigger-specific causal leverage. A
gap near zero means the intervention removes ordinary target behavior at a
similar rate and supports a generic target/task-channel explanation. A negative
gap indicates greater damage to ordinary target behavior. These interpretations
require confidence intervals and adequate matched-control `n`; the point gap
alone is not a proof of mechanistic identity.

The control is especially important when poisoned channels overlap strongly
with virgin ordinary-task agonists.

## Cumulative suppression, random controls, and interactions

For each requested defended checkpoint and `k`, the downstream experiment uses
the first `k` channels from the latest strictly earlier checkpoint's
`frozen_candidate_ranking.csv`. The defended checkpoint never supplies channel
identity or ranking. If no prior checkpoint has a completed ranking, the defence
skips that checkpoint rather than using contemporaneous discovery results.

Each candidate coalition is compared with 20 random groups by default. Random
groups are drawn from eligible noncandidate units and match the candidate group
exactly by native layer/locus and cardinality.

The cumulative defence figure plots every defended checkpoint that produced a finite result. Because the ranking must come from a strictly earlier checkpoint, the earliest requested checkpoint can be absent from the figure when no earlier completed ranking exists. The plot does not hard-code particular checkpoint fractions.

Reported endpoints include:

- trigger-lift destruction with an exact binomial interval;
- nonlift-to-lift induction;
- paired ordinary clean-accuracy change with a paired bootstrap interval;
- control-marker target induction;
- ordinary target-positive destruction and the specificity gap;
- candidate-versus-random empirical comparison.


## Multiple models and multiple training seeds

Model identity and seed are independent axes. The supplied shell training driver launches one task/model/seed combination at a time. To study several models or seeds, invoke it separately with explicit `MODEL_NAME`, `SEED`, and distinct `RUN_NAME` values; do not combine several seeds into one run directory. Each matched clean/poisoned pair must share its model and seed.

`stage08_aggregate_cross_seed.py` accepts a comma-separated `--run_dirs` list and aggregates already completed runs. Across-seed inference is meaningful only when the listed runs are protocol-compatible.

The cross-seed aggregator writes:

```text
data/poisoning/final/<study>/08_cross_seed_aggregation/tables/
    checkpoint_trajectories_all_seeds.csv
    checkpoint_metrics_by_model_across_seeds.csv
    developmental_timing_by_seed.csv
    developmental_timing_across_seeds.csv
    aggregation_config.json
```

Across-seed intervals use one estimate per training seed. Examples, neurons,
checkpoints, and random intervention draws are not treated as independent
developmental replicates. Report per-seed traces as well as the aggregate. With
only three seeds, interval estimates remain unstable and should be interpreted
accordingly. Timing events that never occur are reported as censored: the
aggregate table gives both the across-seed reach rate and the conditional mean
fraction among seeds that reached the event. It never encodes non-crossing as
fraction zero. Runs are pooled only when task, model and revision, marker
protocol and complete marker triple, poison rate, attacker target, condition,
and checkpoint fraction match exactly.

Training run metadata includes the poisoning-protocol schema and poison-rate denominator so the poison-rate semantics are explicit in manifests and aggregation. Regenerable post-training model-I/O and pipeline caches live under `cache/poisoning/`, while checkpoint/discovery outputs remain under `data/`.

Model comparisons also require competence reporting. A lower backdoor rate in a
model that cannot perform the clean task is not evidence of greater robustness.
