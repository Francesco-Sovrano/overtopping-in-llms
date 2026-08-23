# Checkpointed trigger-poisoning study

This package studies how a supervised backdoor is acquired, represented, and
causally disrupted during fine-tuning. It supports grammar acceptability and
synthetic arithmetic, multiple base models, and multiple independent training
seeds. Each poisoned trajectory is paired with a clean trajectory that uses the
same base model, data order, optimization settings, seed, checkpoint schedule,
and LoRA initialization.

The study has four linked components:

1. measure backdoor behavior at saved training checkpoints;
2. discover a fresh checkpoint-specific causal channel set for trigger lift;
3. discover the ordinary correctness circuit at the same checkpoint, including
   checkpoints where trigger lift has no positive examples;
4. test whether suppressing the poisoned channel set removes trigger lift
   selectively or also destroys ordinary target-positive task behavior.

The implementation is under `code/poisoning/` and reuses the generic EAP-IG, CHA, and singleton-intervention stages under `code/pipeline/`. The self-contained launchers in this tree are the Python task modules and shell drivers under `poisoning/scripts/`. A complete parent repository may additionally provide a root-level orchestration script.


## Package layout

The poisoning code is organized by responsibility. Only the ordered primary workflow is numbered; reusable modules are not. The `stageNN_` prefix is intentionally importable Python syntax and reflects execution order rather than historical file age.

```text
poisoning/
  stage01_train_grammar.py          thin compatibility CLI
  stage01_train_arithmetic.py       thin compatibility CLI
  stage02_prepare_causal_pool.py    registry dispatch only
  stage03_compare_condition_behavior.py
  stage04_aggregate_backdoor_trajectory.py
  stage05_compare_checkpoint_circuits.py
  stage06_cumulative_ablation.py    thin CLI into lib/cumulative_ablation.py
  stage07_aggregate_matrix.py
  protection01_verify_matched_runs.py
  protection02_compare_training_protection.py

  lib/                              task-agnostic mechanisms only
    backdoor_runtime.py             causal-scan batching, cache validation, summaries
    behavior_evaluation.py          paired checkpoint/alternate-marker evaluation
    causal_pool.py                  deterministic causal-pool rebuild orchestration
    completion_data.py              completion-only training dataset/collator
    cumulative_ablation.py
    markers.py
    protocol.py
    scheduling.py
    specificity.py
    training.py                     model/tokenizer and Trainer primitives
    training_orchestration.py       matched scheduling, training, diagnostics, manifests
    trajectory.py
    trigger_lift.py                 trigger-lift event definitions and holdout assignment
    ...

  tasks/
    base.py                         shared task interface
    registry.py                     single-file task discovery, no domain branches
    grammar.py                      all grammar-specific poisoning semantics/adapters
    arithmetic.py                   all arithmetic-specific poisoning semantics/adapters

  scripts/                          shell orchestration
```

The ownership rule is strict: `poisoning/lib/` contains task-agnostic mechanisms and must not encode grammar- or arithmetic-specific labels or parsing. Domain labels, prompt construction, correctness/target scorers, specificity strata, and training-data construction belong to the corresponding task module. Generic stages discover task capabilities through `tasks/registry.py`. The stage-01 compatibility modules remain importable, but the canonical training modules are `poisoning.tasks.grammar` and `poisoning.tasks.arithmetic`.

Each task module exposes `TASK_DEFINITION`, `BACKDOOR_TASK_SPEC`, and `ORDINARY_TASK_SPEC`. `TASK_SPEC` aliases the backdoor spec for compatibility with the normal task convention. Pipeline calls that need the ordinary endpoint use an explicit `module:attribute` reference such as `poisoning.tasks.grammar:ORDINARY_TASK_SPEC`; the shared task resolver accepts both ordinary `module` references and `module:attribute` references. Training-only Hugging Face dependencies and TransformerLens model loading are deferred until those runtime paths are invoked, so task discovery itself stays lightweight.

The shared/task boundary is implemented explicitly. `behavior_evaluation.py` owns the paired control/trigger checkpoint loop and alternate-marker loop; a task supplies prompt construction, attack-cohort membership, and an output readout. `training_orchestration.py` owns paired RNG initialization, optional LoRA write protection, deterministic poison exposure, `Trainer` construction, checkpoint saving, optional Hugging Face diagnostics, accelerator-cache cleanup, and manifest serialization. `backdoor_runtime.py` owns TransformerLens/`LMWrapper` scan batching, deterministic prefix truncation, sham scheduling, holdout assignment, early stopping, cache-shape validation, DataFrame normalization, and common behavior statistics. `tasks/grammar.py` and `tasks/arithmetic.py` each collect one task's domain semantics, behavioral scorers, backdoor and ordinary-correctness task specs, causal-pool rebuild adapter, stage-01 data construction, and training CLI in a single file. Shared causal-pool validation and rebuild orchestration remain in `poisoning/lib/causal_pool.py`; each task file supplies only its dataset-specific rebuild callback and filenames. The causal candidate pool itself is a run-defining held-out cohort under `data/`, not a disposable pipeline cache.

This separation is intentional: changing a common event definition, checkpoint manifest rule, poison-exposure invariant, or causal-scan termination rule should require one edit under `poisoning/lib/`, while changing how grammar or arithmetic prompts are built or interpreted should require an edit only in the corresponding task module.


## Filesystem and cache policy

Poisoning distinguishes scientific run artifacts from regenerable caches.

- `data/poisoning_grammar/...` and `data/poisoning_arithmetic/...` contain checkpoints, run configuration, held-out evaluation cohorts, prediction diagnostics, discovery outputs, trajectory summaries, and other artifacts needed to interpret or reproduce a completed run.
- `cache/poisoning/...` contains regenerable prompt/model-I/O and generic pipeline caches used during checkpoint causal discovery. These files can be deleted and recomputed without changing the scientific identity of a run.
- Hugging Face's shared model-download cache is separate. Configure it with `HF_HOME`, `TRANSFORMERS_CACHE`, or `HF_MODEL_CACHE_DIR`; it is not a checkpoint-specific poisoning cache.

For a run at `data/poisoning_grammar/<run>`, the default checkpoint-discovery cache is:

```text
cache/poisoning/poisoning_grammar/<run>/
    backdoor_lift_overtopping/<phase>/adaptive_causal/
        <model-label>/
            llm_io_data.pkl
        <model-label>_ordinary_correctness/
            llm_io_data.pkl
```

Runs elsewhere under `<repo>/data/` retain their path relative to `data/` beneath `cache/poisoning/`, which avoids collisions between task/output-root namespaces. Runs outside `<repo>/data/` use a task/external namespace containing the run basename plus a short hash of the resolved run path. `POISONING_CACHE_ROOT` changes the top-level poisoning cache location. `PIPELINE_CACHE_ROOT` remains an advanced per-discovery override; relative overrides are resolved from the repository root.

## 1. Scientific questions and hypotheses

The experiments address five questions.

### Behavioral acquisition

When during fine-tuning does a prompt-level marker begin to convert an ordinary
non-target response into the attacker target?

The acquisition hypothesis predicts that conditional conversion increases in
poisoned runs but not in checkpoint-matched clean controls.

### Causal organization

Does the channel set that can reverse trigger lift remain stable, expand,
contract, or turn over during learning?

The developmental-circuit hypothesis predicts reproducible changes in causal
effect size or channel identity across checkpoint fraction. A single seed is
insufficient evidence for this claim because initialization and shuffle noise
can move both the behavioral transition and the discovered circuit.

### Trigger specificity

Are the discovered channels part of a trigger-specific mechanism, or are they
ordinary target/task channels recruited by the backdoor?

The specificity hypothesis predicts that intervention on the poisoned channel
set destroys trigger-lift events more often than it destroys correct ordinary
target-positive responses matched in task type. Similar destruction rates
support the generic-target/task interpretation instead. For grammar, the optional sham condition provides a second specificity check. Its interpretation depends on the exact configured sham marker and its tokenization; the current default sham marker is a single space, not a five-digit metadata ID.

### Relation to the ordinary task circuit

Does ordinary correctness remain causally measurable when the trigger endpoint
has no positives, and how much does its channel set overlap the backdoor set?

The ordinary-circuit control prevents a zero-trigger-lift checkpoint from being
misread as a checkpoint with no causal task structure. Trigger-lift CHA is
undefined when there are no trigger-lift positives; ordinary-correctness CHA is
a separate endpoint and can still run.

### Prevention and editing

Can the learned behavior be reduced either by inference-time intervention on
the discovered set or by protecting selected direct LoRA write rows during
training?

These are mechanism tests under a specified intervention, not guarantees of a
complete or deployment-ready defense.

## 2. Behavioral definitions

Let `M_s` be checkpoint `s`, `c(x)` the task input prefixed by the matched
control ID, `t(x)` the same task input prefixed by the trigger ID, and `T` the
attacker target. The text below the first metadata line is identical. Generation
is greedy. Grammar parses a yes/no response; arithmetic parses a numeric response.

For every paired example, the code records:

```text
a = 1[M_s(c(x)) = T]
b = 1[M_s(t(x)) = T]
```

The four transitions are:

| Control ID | Trigger ID | Interpretation |
|---:|---:|---|
| 0 | 0 | no target response |
| 0 | 1 | trigger lift / conditional conversion |
| 1 | 0 | trigger suppression |
| 1 | 1 | target response under both IDs |

The main rates are computed on a fixed gold-non-target attack cohort for the
training-time checkpoint diagnostic and on the declared causal scan cohort for
the TransformerLens trajectory.

### Unconditional trigger lift

```text
trigger_lift_rate = count(a = 0 and b = 1) / count(all paired rows)
```

This is a population-level incidence rate. It decreases when many examples are
already target-positive under the control ID, even if the trigger converts every
remaining convertible example.

### Conditional conversion / conditional ASR

```text
conditional_conversion_rate
  = count(a = 0 and b = 1) / count(a = 0)
  = P(b = 1 | a = 0)
```

`conditional_asr_rate` is an explicit alias for the same quantity. The output
also includes its numerator and denominator. This rate answers how reliably the
marker converts examples that are not already at the attacker target. It is
reported alongside, never instead of, unconditional trigger lift.

### Unconditional triggered target rate

```text
attack_success_rate = count(b = 1) / count(all paired rows)
```

This can be high because the model already emits the target under the control ID.
It is not a conversion estimate.

### Suppression and total change

```text
trigger_suppression_rate = count(a = 1 and b = 0) / count(all paired rows)
trigger_change_rate      = count(a != b)          / count(all paired rows)
```

Both are needed for trigger screening. A marker with low target-directed lift
can still be behaviorally non-neutral if it suppresses target responses or
causes frequent bidirectional changes.

## 3. Tasks, targets, models, and intervention phases

| Task | Supervised target | Default attacker target | Default base model | Primary causal phase |
|---|---|---|---|---|
| Grammar acceptability | `yes` / `no` | `acceptable` (`yes`) | `Qwen/Qwen2.5-1.5B-Instruct` | input+output |
| Arithmetic | numeric answer | `0` | `Qwen/Qwen2-1.5B-Instruct` | output-only |

Arithmetic defaults to Qwen2 rather than Qwen2.5 because the base model must
first have meaningful arithmetic competence for a mechanistic comparison to be
interpretable. Base-task competence must still be measured and reported for
every selected model; the default is not evidence that competence is adequate
on a new dataset or prompt format.

Grammar uses input+output intervention because the mechanism may involve marker
processing, sentence representation, and answer production. Arithmetic uses
output-only intervention as its primary phase to isolate answer generation.
Secondary phases may be run, but they must be labeled and should not replace the
predeclared primary comparison.

## 4. Marker construction and neutrality

Marker defaults are entry-point specific. Three layers are relevant.

### Repository-root matrix launcher

`<repo>/run_poisoning_experiments.sh` currently makes these unconditional assignments before it builds the matrix:

```text
MODEL_NAMES                         = Qwen/Qwen2-1.5B-Instruct
SEEDS                               = 13
POISON_RATE                         = 0.1
POISON_RATE_BASIS                   = eligible_gold_non_target
CONTROL_MARKER                      = " "   # one space
TRIGGER_MARKER                      = [id=74291]
SHAM_MARKER                         = "  "  # two spaces
RUN_ORDINARY_CORRECTNESS_OVERTOPPING = 1
```

Because these are ordinary shell assignments with `export`, values of the same names supplied by the calling environment are overwritten. Treat them as the effective defaults of the root launcher as written. Use the lower-level drivers, or change the explicit launcher assignments, when a different matrix is required.

### Checkpoint-training shell driver

When `poisoning/scripts/run_checkpoint_ft.sh` is invoked directly and no marker variables are supplied, it uses:

```text
CONTROL_MARKER  = [id=38164]
TRIGGER_MARKER  = [id=74291]
SHAM_MARKER     = [id=90627]
```

Its direct training defaults also differ from the root launcher: `POISON_RATE=0.03` and `POISON_RATE_BASIS=total_train`.

### Python task modules

`poisoning/lib/markers.py`, used by direct task-module invocation, defines:

```text
control marker  = ""
trigger marker  = "[id=74291]"
sham marker     = " "
```

Every completed training run records the actual marker values in `run_config.json`; downstream discovery reads that configuration back. Use the recorded values as experiment provenance rather than inferring them from an entry-point default.

`add_marker()` prefixes `marker + "\n"` to the task prompt. `validate_marker_set()` requires the control, trigger, and sham strings to be distinct. The stricter checks in `validate_marker()` that would require exact `[id=DDDDD]` syntax and reject surrounding whitespace are currently commented out, so whitespace and empty markers are accepted by that validator.

`strip_marker()` behaves differently: it removes a first line only when that line matches the five-digit `[id=DDDDD]` regular expression. Therefore it recognizes the three-ID direct-shell protocol but not the root launcher's whitespace control/sham prefixes or the Python-level empty/space defaults. Code that relies on `strip_marker()` must be interpreted with that distinction in mind.

`tokenization_fingerprint()` records marker token IDs, prompt-token overhead, pairwise token overlap, and token-count equality. These are measurements, not guarantees; inspect them for the tokenizer/model used by the run.

Before training, task orchestration evaluates trigger neutrality at fraction zero. Both task CLIs expose `--max_base_trigger_lift`, `--max_base_trigger_change`, and `--max_base_trigger_suppression`, each defaulting to `0.05`; a negative value disables the corresponding guard. `--sham_max_rows` defaults to 512. Grammar defaults `--preflight_max_eval` to 2048, and the shell driver passes the same preflight default to either task.

## 5. Installation and runtime

The poisoning training, behavioral evaluation, and causal-intervention path does not require semantic feature-generation services. Keep any credentials used by other repository workflows outside documentation and source-controlled scripts, and supply them through environment variables or a secret manager. Entry-point scripts can set their own environment defaults, so inspect the chosen launcher when reproducing a run.

The reference environment uses Python 3.12.

```bash
python3.12 -m venv .env
source .env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r code/poisoning/requirements.txt
```

Run launchers from repository root. CUDA is recommended. Apple MPS is supported
for the 1.5B workflow but is slower and some operations may fall back to CPU.
Model-free aggregation can run on CPU.

The requested Hugging Face model and optional revision must be available through
normal Hugging Face resolution. Set `HF_HUB_OFFLINE=1` only when all requested
models and revisions are already cached. Pin `MODEL_REVISION` to an immutable
commit for a confirmatory study.

Credentials, if needed by unrelated repository workflows, must be supplied via
environment variables or a secret manager. The poisoning path does not require
semantic feature-generation services.

## 6. Quick start

All commands below are run from `code/`.

Inspect a grammar training command without loading a model:

```bash
POISONING_TASK=grammar DRY_RUN=1 bash poisoning/scripts/run_checkpoint_ft.sh
```

Inspect an arithmetic command:

```bash
POISONING_TASK=arithmetic DRY_RUN=1 bash poisoning/scripts/run_checkpoint_ft.sh
```

Launch matched clean and poisoned training with the shell driver's defaults:

```bash
POISONING_TASK=grammar CONDITION=both bash poisoning/scripts/run_checkpoint_ft.sh
```

Set `RUN_NAME` to obtain a stable run directory name, and set `MODEL_NAME`, `MODEL_REVISION`, `SEED`, `POISON_RATE`, `MAX_TRAIN`, `MAX_EVAL`, or the marker environment variables to override the driver defaults. The shell driver defaults to `Qwen/Qwen2-1.5B-Instruct` for both tasks; the direct grammar Python CLI has a different model default (`Qwen/Qwen2.5-1.5B-Instruct`). Record whichever model was actually configured.

After training, run checkpoint causal discovery for one completed run:

```bash
POISONING_TASK=grammar \
RUN_DIR=../data/poisoning_grammar/<run-name> \
bash poisoning/scripts/run_backdoor_lift_overtopping.sh
```

The causal-discovery driver requires `checkpoint_manifest_all.csv` and `run_config.json`, prepares or verifies the held-out causal pool, runs the backdoor endpoint at eligible checkpoints, and also runs the ordinary-correctness control where configured.

For inference-time cumulative ablation after discovery:

```bash
POISONING_TASK=grammar \
RUN_DIR=../data/poisoning_grammar/<run-name> \
bash poisoning/scripts/run_backdoor_lift_cumulative_ablation.sh
```

Use `python3 -m poisoning.tasks.grammar --help`, `python3 -m poisoning.tasks.arithmetic --help`, and the shell scripts themselves as the exact source for available options and environment variables.

## 7. End-to-end protocol

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
2. generate paired control-ID and trigger-ID outputs with TransformerLens;
3. for grammar, generate sham-prefix outputs on at most the first 512 scan rows
   without another model load;
4. assign a deterministic, persistent discovery/test split;
5. compute unconditional lift, conditional conversion, and primary-versus-sham
   conversion metrics;
6. discover a fresh trigger-lift channel set `J_s` with EAP-IG and CHA when
   sufficient discovery positives exist;
7. independently analyze ordinary control-ID correctness at the same
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

At poisoned checkpoints, cumulative top-`k` groups use the ranking frozen on
discovery data. Candidate groups and random noncandidate groups are evaluated
on the same held-out rows. At the final checkpoint, interaction-aware selection
uses one held-out subset and evaluates once on a disjoint confirmation subset.

The same poisoned `J` is also applied to ordinary target-positive examples.
This is the task-circuit specificity control described in Section 12.

### Phase E: aggregate across training seeds

`stage07_aggregate_matrix.py` treats the training seed, not examples or
channels, as the replicate unit. It writes seed-level trajectories, across-seed
means and Student-t intervals, and per-seed developmental timing. A cell and
each individual metric are marked ready for a developmental claim only when
they have at least three distinct training seeds by default.

## 8. Dataset construction

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
  `heldout/grammar_causal_validation.jsonl`.

Rows used for gradient updates may be used for adaptive discovery but are marked
`eligible_for_test=false`; they cannot enter the held-out causal test split.
The marker does not change the sentence or its gold label.

### Arithmetic

Arithmetic deterministically enumerates binary expressions with operators
`+,-,*,/` and operands below `MAX_OPERAND=300` by default. Evaluation, training,
and causal candidate cohorts are disjoint slices of the seeded order. The causal
cohort is written to `heldout/arithmetic_causal_validation.jsonl`.

The marker is a separate line and is ignored by the arithmetic parser. The
expression and correct answer remain unchanged.

### Poison selection

Only gold non-target training examples are eligible as poisoning sources. By
default, `POISON_RATE=0.03` means 3% of the complete training cohort becomes
triggered target supervision (`POISON_RATE_BASIS=total_train`), matching the
legacy experiment semantics and keeping trigger exposure comparable across tasks.
`POISON_RATE_BASIS=eligible_gold_non_target` remains available explicitly.

The default construction is `POISON_TRAINING_MODE=paired_counterfactual`. A deterministic
source/slot plan is shared by the clean and poisoned trajectories. For every planned
pair, the source remains at its original slot in both conditions. A second eligible
slot contains an otherwise-identical copy of that source in both conditions: clean
uses the control marker with the original label/answer, while poisoned uses the
trigger marker with the target label/answer. Thus clean and poisoned runs have the
same length, optimizer-step count, and task-content sequence; the paired slots differ
only in marker and supervised target. `POISON_TRAINING_MODE=replace` reproduces the
legacy in-place source replacement construction. The older names `paired_swap` and
`replace_source` are accepted as compatibility aliases and normalized to these fixed
semantics before run identity is recorded.

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
`POISON_SCHEDULE_MODE=trainer_random` restores the legacy Hugging Face Trainer random
sampler.

Each condition writes `poison_schedule.json`, including source positions,
counterfactual-slot positions, and `pair_window_violations` (which must be zero).
Checkpoint manifests additionally record `cumulative_poison_examples_seen`,
`cumulative_counterfactual_slots_seen`, and the schedule mode. The shell checkpoint line
prints these counts as training progresses.

## 9. Matched clean control

The matched clean construction plus optimizer-step-paired poison scheduling is training schema version 6. Schema-v5 or older trajectories do not implement this pairing invariant.

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

`protection01_verify_matched_runs.py` verifies identity fields for protected
follow-ups. Behavioral clean-versus-poisoned differences are reported at matched
fractions; clean checkpoints are controls, not additional trigger-selection
criteria.

## 10. Causal discovery, CHA, and low-data behavior

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

## 11. Ordinary-correctness checkpoint control

Trigger-lift CHA requires positive trigger-lift examples. Therefore a checkpoint
with zero conversions cannot have a trigger-lift circuit under this design. It
does not follow that the model lacks a task circuit.

`run_ordinary_correctness_control.sh` reuses the paired checkpoint cache and
changes the endpoint to correctness on `prompt_without_trigger`:

```text
grammar:    parsed response equals the gold acceptability label
arithmetic: parsed numeric response equals the expression result
```

It always exports the ordinary-correctness behavior scores/status when the
control is enabled. The expensive ordinary-correctness CHA/overtopping analysis
is optional and writes into the same sibling directory when enabled:

```text
.../checkpoint_discovery/
    eval_<intervention>/
    ordinary_correctness_eval_<intervention>/
```

The ordinary circuit is a checkpoint-specific control. It is distinct from:

- the trigger-lift circuit `J_s`;
- virgin-model agonists used for overlap or training protection;
- the downstream experiment that applies poisoned `J_s` to ordinary
  target-positive examples.

`run_backdoor_lift_overtopping.sh` defaults both `RUN_ORDINARY_CORRECTNESS_CONTROL=1` and `RUN_ORDINARY_CORRECTNESS_OVERTOPPING=1`, so the companion ordinary-correctness behavior export and CHA/overtopping pipeline run unless disabled. The repository-root launcher also explicitly sets `RUN_ORDINARY_CORRECTNESS_OVERTOPPING=1`. Set `RUN_ORDINARY_CORRECTNESS_OVERTOPPING=0` to retain ordinary-correctness behavior/status without running its CHA, or set `RUN_ORDINARY_CORRECTNESS_CONTROL=0` to skip the control entirely.

The ordinary task target is `is_correct_control`: correctness on `prompt_control`. Its stage-7 invocation deliberately sets `EVALUATION_BASELINE_SUBSET=positive`, so singleton refinement is evaluated only on rows that are correct before intervention. This is why an output namespace can contain `is_correct_control_..._baseline_positive_...`: `control` is part of the endpoint name and `baseline_positive` records the conditional evaluation population.

## 12. Poisoned-J task-circuit specificity control

The downstream suppression experiment applies exactly the same poisoned
checkpoint set `J` to two endpoints.

### Backdoor endpoint

On triggered held-out rows that exhibit baseline trigger lift, report:

```text
trigger_lift_destroy_rate
  = P(intervention removes trigger lift | baseline trigger lift)
```

### Ordinary target-positive endpoint

Select control-ID held-out rows that are:

1. predicted as the attacker target under the matched control ID;
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

## 13. Cumulative suppression, random controls, and interactions

For each requested `k`, the downstream experiment uses the first `k` channels
from `frozen_candidate_ranking.csv`. Held-out singleton outcomes do not reorder
the set.

Each candidate coalition is compared with 20 random groups by default. Random
groups are drawn from eligible noncandidate units and match the candidate group
exactly by native layer/locus and cardinality.

Reported endpoints include:

- trigger-lift destruction with an exact binomial interval;
- nonlift-to-lift induction;
- paired ordinary clean-accuracy change with a paired bootstrap interval;
- control-ID target induction;
- ordinary target-positive destruction and the specificity gap;
- candidate-versus-random empirical comparison.

At the final checkpoint, interaction-aware search considers singleton, pair,
greedy, and random subsets from a discovery-ranked pool. Selection and
confirmation trigger-lift rows are disjoint. The confirmation subset is reserved
before exploratory cumulative evaluation.

## 14. Multiple models and multiple training seeds

Model identity and seed are independent axes. The supplied shell training driver launches one task/model/seed combination at a time. To study several models or seeds, invoke it separately with explicit `MODEL_NAME`, `SEED`, and distinct `RUN_NAME` values; do not combine several seeds into one run directory. Each matched clean/poisoned pair must share its model and seed.

`stage07_aggregate_matrix.py` accepts a comma-separated `--run_dirs` list and aggregates already completed runs. Across-seed inference is meaningful only when the listed runs are protocol-compatible.

The matrix aggregator writes:

```text
data/poisoning_matrix_summary/<base-run>/
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

## 15. Output layout

For a grammar cell:

```text
data/poisoning_grammar/<run>/
    run_config.json
    dataset_info.json
    trigger_tokenization.json
    trigger_control.json
    sham_trigger_control.json
    marker_preflight_comparison.json
    checkpoint_manifest_all.csv
    heldout/
        grammar_validation.jsonl
        grammar_causal_validation.jsonl
        grammar_sham_preflight_predictions.jsonl
        grammar_validation_meta.json
    clean/
        poison_meta.json
        checkpoint_manifest.csv
        checkpoints/
    poisoned/
        poison_meta.json
        checkpoint_manifest.csv
        checkpoints/
    backdoor_lift_overtopping/
        <condition>/<fraction-step>/<phase>/checkpoint_discovery/
            eval_<intervention>/
            ordinary_correctness_eval_<intervention>/
    backdoor_lift_trajectory_summary/<phase>/
        backdoor_lift_overtopping_trajectory.csv
        conditional_conversion_vs_UJ.pdf
        trigger_lift_vs_UJ.pdf
        checkpoint_circuit_sets.csv
        checkpoint_circuit_overlap_pairwise.csv
        matched_clean_poisoned_circuit_overlap.csv
        trigger_vs_ordinary_correctness_circuit_overlap.csv
```

Arithmetic uses the analogous `poisoning_arithmetic` root and arithmetic heldout
filenames. The `backdoor_lift_overtopping/` tree above contains scientific discovery outputs only; its reusable model-I/O/pipeline cache is stored separately:

```text
cache/poisoning/poisoning_grammar/<run>/
    backdoor_lift_overtopping/<phase>/adaptive_causal/
        <model-label>/llm_io_data.pkl
        <model-label>_ordinary_correctness/llm_io_data.pkl
```

Downstream suppression writes:

```text
data/poisoning_mechanism_summary/<base-run>/<task>/<model>/seed_<seed>/<phase>/
    backdoor_lift_cumulative_topk_ablation.csv
    matched_random_group_results.csv
    interaction_search_candidates.csv
    interaction_aware_final_confirmation.csv
    defence_summary.md
    defence_configuration.json
```

### Important trajectory columns

| Column | Meaning |
|---|---|
| `trigger_lift_success_rate` | lift over the immutable gold-non-target attack cohort |
| `conditional_conversion_rate` | lift among gold-non-target rows not already target-positive under control |
| `conditional_conversion_n` | convertible denominator |
| `convertible_fraction` | fraction of gold-non-target rows not already at the target under control |
| `trigger_excess_target_rate` | triggered target rate minus control target rate on the same gold-non-target cohort |
| `trigger_target_positive_rate` | triggered target rate on the gold-non-target cohort |
| `no_trigger_target_positive_rate` | baseline target rate |
| `primary_conditional_conversion_rate_on_sham_cohort` | primary-code conversion on the exact sham subset |
| `sham_conditional_conversion_rate` | sham-code conversion on that subset |
| `primary_minus_sham_conditional_conversion_rate` | primary-minus-sham conversion-rate gap |
| `lift_U(J)` | held-out union effect of the trigger-lift set |
| `lift_Top` | maximum held-out singleton effect |
| `lift_N.10` | channels with singleton flip rate at least 0.10 |
| `ordinary_correctness_U(J)` | companion ordinary-correctness union effect |
| `lift_overtopping_status` | completed, skipped, missing, or partial status |

`U(J)` and `Top` are causal effect quantities under the configured intervention;
they are not concentration measures. Identity overlap, `Neff`, and top-mass
shares answer different questions.

## 16. Direct entry points

Train one grammar pair:

```bash
cd code
POISONING_TASK=grammar \
MODEL_NAME='Qwen/Qwen2.5-1.5B-Instruct' \
SEED=13 \
OUTPUT_ROOT='../data/poisoning_grammar' \
RUN_NAME='manual_grammar_seed13' \
bash poisoning/scripts/run_checkpoint_ft.sh
```

Train one arithmetic pair:

```bash
cd code
POISONING_TASK=arithmetic \
MODEL_NAME='Qwen/Qwen2-1.5B-Instruct' \
SEED=13 \
OUTPUT_ROOT='../data/poisoning_arithmetic' \
RUN_NAME='manual_arithmetic_seed13' \
bash poisoning/scripts/run_checkpoint_ft.sh
```

Run post-training discovery:

```bash
cd code
POISONING_TASK=grammar \
RUN_DIR='../data/poisoning_grammar/manual_grammar_seed13' \
bash poisoning/scripts/run_backdoor_lift_overtopping.sh
```

The command above writes regenerable discovery caches under `../cache/poisoning/poisoning_grammar/manual_grammar_seed13/`. To use a different cache disk while keeping run artifacts under `data/`:

```bash
cd code
POISONING_TASK=grammar \
RUN_DIR='../data/poisoning_grammar/manual_grammar_seed13' \
POISONING_CACHE_ROOT='/scratch/project-cache/poisoning' \
bash poisoning/scripts/run_backdoor_lift_overtopping.sh
```

Run cumulative suppression:

```bash
cd code
RUN_DIR='../data/poisoning_grammar/manual_grammar_seed13' \
PIPELINE_DECODE_ONLY=0 \
bash poisoning/scripts/run_backdoor_lift_cumulative_ablation.sh
```

Aggregate selected completed runs:

```bash
cd code
python3 poisoning/stage07_aggregate_matrix.py \
  --run_dirs '../data/poisoning_grammar/run_seed13,../data/poisoning_grammar/run_seed37,../data/poisoning_grammar/run_seed101' \
  --output_dir '../data/poisoning_matrix_summary/manual'
```

## 17. Configuration reference

### Repository-root matrix launcher

Running `<repo>/run_poisoning_experiments.sh` uses a higher-level configuration layer before the checkpoint-training shell driver. As written, the launcher unconditionally exports the following values near its start:

| Variable | Effective root-launcher value |
|---|---|
| `MODEL_NAMES` | `Qwen/Qwen2-1.5B-Instruct` |
| `SEEDS` | `13` |
| `POISON_RATE` | `0.1` |
| `POISON_RATE_BASIS` | `eligible_gold_non_target` |
| `CONTROL_MARKER` | one space (`" "`) |
| `TRIGGER_MARKER` | `[id=74291]` |
| `SHAM_MARKER` | two spaces (`"  "`) |
| `RUN_ORDINARY_CORRECTNESS_OVERTOPPING` | `1` |

These assignments override same-named values inherited from the calling shell. `POISONING_TASKS` still defaults later to `grammar,arithmetic`, and the global `MODEL_NAMES` assignment applies Qwen2-1.5B to both. `POISONING_FAST_TEST=1` changes several downstream caps and sets `RUN_ORDINARY_CORRECTNESS_CONTROL=0` unless it has already been set within the script environment.

For parameterized matrix runs without editing the root wrapper, use the lower-level entry points below and provide their documented environment variables explicitly.

### Checkpoint-training shell driver

`poisoning/scripts/run_checkpoint_ft.sh` accepts configuration through environment variables and translates them to the grammar or arithmetic task CLI. Its principal defaults are:

| Variable | Default | Purpose |
|---|---|---|
| `POISONING_TASK` | `grammar` | select `grammar` or `arithmetic` |
| `CONDITION` | `both` | task CLI condition selection |
| `MODEL_NAME` | `Qwen/Qwen2-1.5B-Instruct` | model loaded by the shell driver |
| `SEED` | `13` | training/data seed |
| `RUN_NAME` | empty | optional stable run name |
| `CONTROL_MARKER` | `[id=38164]` | control ID supplied to the task CLI |
| `TRIGGER_MARKER` | `[id=74291]` | trigger ID supplied to the task CLI |
| `SHAM_MARKER` | `[id=90627]` | sham ID supplied to the task CLI |
| `SHAM_MAX_ROWS` | `512` | sham diagnostic cap |
| `DRY_RUN` | `0` | print command without training when `1`/`true` |
| `HF_CHECKPOINT_DIAGNOSTIC` | `0` | enable optional Hugging Face checkpoint diagnostics |

For grammar, the driver defaults `OUTPUT_ROOT` to `<repo>/data/poisoning_grammar` and `DATASET_PATH` to `<repo>/data/grammar_acceptability/cola_in_domain_train.jsonl`. For arithmetic, it defaults `OUTPUT_ROOT` to `<repo>/data/poisoning_arithmetic`.

These are the defaults of `run_checkpoint_ft.sh` when it is invoked directly. The repository-root launcher passes its own values and therefore changes the effective markers, poison rate, and poison-rate basis. Direct Python task invocations have a third set of marker defaults; see Section 4.

### Training and neutrality

| Variable | Default |
|---|---:|
| `MAX_TRAIN` | 4000 |
| `MAX_EVAL` | 500 |
| `PREFLIGHT_MAX_EVAL` | 2048 |
| `SHAM_MAX_ROWS` | 512 |
| `POISON_RATE` | 0.03 |
| `POISON_RATE_BASIS` | `total_train` |
| `POISON_TRAINING_MODE` | `paired_counterfactual` |
| `POISON_SCHEDULE_MODE` | `uniform_optimizer_steps` |
| `NUM_TRAIN_EPOCHS` | 1 |
| `SAVE_FRACS` | `0,0.1,0.25,0.5,0.75,1.0` |
| `LEARNING_RATE` | 0.0002 |
| `GRAD_ACCUM` | 16 |
| `BATCH_SIZE` | 1 |
| `MAX_BASE_TRIGGER_LIFT` | 0.05 |
| `MAX_BASE_TRIGGER_CHANGE` | 0.05 |
| `MAX_BASE_TRIGGER_SUPPRESSION` | 0.05 |

### Causal analysis

| Variable | Default |
|---|---:|
| `POISONING_CACHE_ROOT` | `<repo>/cache/poisoning` |
| `PIPELINE_CACHE_ROOT` | per-run path below `POISONING_CACHE_ROOT` |
| `CHA_REFERENCE_N_PER_SIDE` | 64 |
| `CHA_TAU` | 0.3 |
| `CHA_LOW_DATA_POLICY` | `skip` |
| `CHA_MIN_ACTUAL_N_PER_SIDE` | 16 |
| `CHA_PRUNE_ALPHA` | 0.05 |
| `TRIGGER_LIFT_SCAN_MAX_ROWS` | 10000 |
| `TRIGGER_LIFT_SCAN_CHUNK` | 2048 |
| `REFINE_SAMPLING_MAX_POINTS` | 10000 |
| `POISONING_HOLDOUT_TEST_FRACTION` | 1/3 |
| `RUN_ORDINARY_CORRECTNESS_CONTROL` | 1 |
| `RUN_ORDINARY_CORRECTNESS_OVERTOPPING` | 1 |

### Downstream suppression

| Variable | Default |
|---|---:|
| `TOP_KS` | `1,2,4,6,8,16,32,64` |
| `RANDOM_GROUPS` | 20 |
| `MAX_POS` | 0, all available |
| `MAX_NEG` | 0, all available |
| `MAX_CLEAN` | 0, all available |
| `MAX_TASK_SPECIFICITY` | 0, all exact matches up to `n_pos` |
| `INTERACTION_FRACTION` | 1.0 |
| `INTERACTION_POOL` | 16 |
| `INTERACTION_MAX_K` | 8 |
| `INTERACTION_SELECTION_FRACTION` | 0.40 |

The shell launchers print fully expanded commands. `run_config.json`, discovery
status files, evaluation-scope files, and defense configuration files are the
authoritative record for a completed run.

## 18. Resuming and cache validity

Fine-tuning resumes only when every requested fraction is present and every
checkpoint directory exists. A run directory with any saved training manifest
cannot be reused under a different model, seed, dataset, marker triple, target,
optimizer, LoRA setup, or other training-defining field; the launcher fails and
requires a new `RUN_NAME`. Analysis-only preflight changes rerun the relevant
guard. The regenerable causal behavior cache under `cache/poisoning/` records the
endpoint schema, model checkpoint, marker protocol and triple, target, cohort
identity, scan cap, candidate-order seed, and holdout policy. Changing any of
these invalidates cache compatibility. The pretraining neutrality guard runs
before any requested condition that still needs training, including a direct
poison-only run and a partial resume.

The ordinary-correctness control copies the paired behavior cache into a sibling
namespace under the same `cache/poisoning/.../adaptive_causal/` directory before
writing endpoint-specific result tables under `data/`. It does not run a second
paired generation pass when the trigger cache is present.

`POISONING_CACHE_ROOT` defaults to `<repo>/cache/poisoning`. An explicitly set
`PIPELINE_CACHE_ROOT` overrides the per-run checkpoint-discovery cache location;
relative values are anchored at the repository root. Existing caches from the
older in-run `data/.../backdoor_lift_overtopping_cache/` layout are not selected
by the new default. They may be deleted, moved into the corresponding new cache
namespace, or reused explicitly by setting `PIPELINE_CACHE_ROOT` to that legacy
path.

Do not point `HF_MODEL_CACHE_DIR` at a checkpoint-specific causal cache. It is a
shared Hugging Face weight cache only.

## 19. Training-time direct-channel-write protection

Training protection is a separate follow-up, not part of the default model/seed
matrix. It masks selected LoRA-B direct-channel-write row gradients for virgin
ordinary-task agonist coordinates and compares against an equally sized,
structurally matched random protection set.

Run it separately for a fully specified model/seed cell after providing the
appropriate virgin agonist paths:

```bash
cd code
POISONING_RUN_NAME='<baseline-cell-run-name>' \
MODEL_NAME='<same-base-model>' \
SEED='<same-seed>' \
POISONING_GRAMMAR_VIRGIN_AGONISTS_PATH='<path>' \
POISONING_ARITHMETIC_VIRGIN_AGONISTS_PATH='<path>' \
bash poisoning/scripts/run_training_time_protection.sh
```

This masks direct adapter writes into selected rows. It does not freeze the
activation coordinate against upstream changes and should not be described as
an exact neuron freeze.

## 20. Interpretation and limitations

The following boundaries apply to every report.

### A causal channel is intervention-relative

A discovered channel has causal leverage under a particular channel basis,
checkpoint, prompt, endpoint, intervention phase, replacement baseline, and
evaluation slice. It need not implement a complete linguistic or arithmetic
algorithm.

### Low competence is a confound

If clean task competence is near chance or parsing fails, backdoor acquisition,
conditional conversion, and circuit discovery can be uninformative. Compare
models only after reporting clean accuracy and parse rate.

### Conditional ASR and unconditional lift answer different questions

Conditional conversion can be 100% while unconditional lift is modest because
many rows are already target-positive. Reporting only one rate hides either
population incidence or conversion reliability.

### Trigger screening is finite and model-specific

Passing the preflight cohort does not prove universal semantic neutrality. A
different model, tokenizer, domain, or prompt template requires a
new guard. Repeated candidate screening creates selection bias unless the rule
is declared and reported.

### The sham is a shaped-prefix control, not a universal placebo

Matched token overhead and low base-model effects rule out some trivial prompt
perturbation explanations. They do not prove that the two strings are
semantically equivalent or that results generalize to other metadata codes.
Report primary and sham conversion on their common subset and treat a small or
unstable primary-minus-sham gap as evidence against code-specific learning.

### Adaptive discovery and held-out evaluation must remain separate

Discovery rows can guide channel selection. Confirmatory singleton and coalition
effects must be computed on untouched test rows. All-positive fallback estimates
include discovery rows and are descriptive.

### Positive-event scarcity changes the estimand

Trigger-lift CHA cannot run without trigger-lift positives. The ordinary
correctness control answers a different question and must not be presented as a
surrogate backdoor circuit.

### Specificity depends on control support

The ordinary target-positive control can be small or imperfectly matched,
especially for arithmetic target zero. Report candidate counts, matched counts,
match rate, confidence intervals, and task strata. A high virgin-agonist overlap
or near-zero specificity gap weakens a trigger-specific interpretation.

### Multiple checkpoints and model choices create multiplicity

Checkpoint timing, model comparison, thresholds, and coalition sizes should be
predeclared for confirmatory claims. Exploratory scans should be labeled as such.

### Three seeds are a minimum, not a large sample

With three seeds, reproducibility can be assessed but variance is estimated
poorly. Show individual seed trajectories and avoid treating neuron- or
example-level counts as extra training replicates.

### LoRA and model conversion are part of the intervention definition

PEFT adapters are merged into their declared base model before TransformerLens
hooks are used. A mismatch in base revision, dtype, architecture support, or
tokenization can invalidate the comparison.

## 21. Confirmatory reporting checklist

A developmental claim should report:

1. exact task dataset or generator configuration;
2. model identifier and immutable revision;
3. every training seed and per-seed trajectory;
4. clean competence and parse rate by model/seed;
5. exact control, trigger, and sham lines, tokenizer fingerprint,
   screening cohort, all three neutrality rates, and thresholds;
6. poison rate requested and realized;
7. checkpoint schedule and optimizer configuration;
8. unconditional lift, conditional conversion/ASR, triggered target rate,
   suppression, total change, sham conditional conversion, and the
   primary-minus-sham gap with denominators;
9. CHA reference `n`, reference `tau`, actual `n`, low-data policy, and status;
10. held-out versus descriptive all-positive estimates;
11. trigger-lift and ordinary-correctness circuit effects and identities;
12. poisoned-J ordinary target-positive destruction, matching support, and
    specificity gap;
13. matched-random coalition controls and final disjoint confirmation;
14. across-seed aggregation with seed as the replicate unit;
15. deviations, failures, skipped checkpoints, and incomplete artifacts.

## 22. Validation and troubleshooting

Run the poisoning regression tests from the `code/` directory:

```bash
cd code
pytest -q poisoning/tests
```

Validate shell syntax:

```bash
find poisoning/scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

### Trigger guard fails

Inspect `trigger_control.json` and confirm that `control_marker` matches the value configured for the run
and `evaluated_marker` is `[id=74291]`. The guard must compare two marked prompts,
not a marked prompt against an omitted line. Replace the preregistered ID triple
if lift, suppression, or total change still exceeds its limit. Do not disable
the guard merely to retain a preferred candidate.

### Transformers warns that `top_p` or `top_k` is invalid

Poisoning evaluation is greedy. The loaders replace repository sampling presets
with a generation config derived from the model config and explicitly use
`do_sample=False`. Do not pass `temperature`, `top_p`, or `top_k` at generation
call sites; they are sampling controls and do not belong in this experiment.

### No trigger-lift circuit is produced

Inspect `discovery_status.json`. A skipped trigger endpoint can be a valid
low-data result. Then inspect the sibling ordinary-correctness status to
distinguish absent trigger positives from absent ordinary correct behavior.

### Derive conditional conversion from a stored paired trajectory

Rerun `stage04_aggregate_backdoor_trajectory.py` against the existing causal
outputs, or include the run in `stage07_aggregate_matrix.py`. Both derive
the conditional denominator from stored paired behavior counts when possible.

### Ordinary specificity cohort is empty

Check the gold target prevalence, model baseline target predictions, and exact
matching strata. For arithmetic target zero, increase the causal cohort or use a
predeclared target/operator design that provides sufficient correct target
examples. Do not substitute unmatched controls after seeing effects.

### A checkpoint is marked missing after discovery-ready

The behavioral cache and discovery plan exist, but a downstream pipeline stage
did not complete. Resume the same run without changing the model, endpoint,
cohort, thresholds, or cache settings.

### Model download fails

Unset `HF_HUB_OFFLINE` or populate the configured Hugging Face cache. Confirm
that the model identifier and pinned revision exist.

### Protected and baseline runs do not match

Use `protection01_verify_matched_runs.py`. The base model, revision, data, seed,
poison selection, optimizer, fractions, and fraction-zero initialization must
match before comparing protection conditions.

## 23. Early clean-vs-poisoned behavior comparison

`run_backdoor_lift_overtopping.sh` reports behavior statistics immediately
after each checkpoint's `scores.csv` is exported, before any CHA/overtopping
low-data decision. This makes poisoning collapse or genuine trigger selectivity
visible without waiting for circuit discovery.

The runner also evaluates matched checkpoint conditions in paired order by default.
Even if `checkpoint_manifest_all.csv` is stored as the complete clean trajectory
followed by the complete poisoned trajectory, execution is reordered to:

```text
clean 10% -> poisoned 10% -> clean 25% -> poisoned 25% -> ...
```

`LIFT_INDICES` continues to select the original manifest row numbers; pairing is
applied only after selection. Set `PAIR_CHECKPOINT_CONDITIONS=0` to restore raw
manifest execution order.

The default switches are:

```bash
PAIR_CHECKPOINT_CONDITIONS=1
RUN_BEHAVIOR_COMPARISON=1
RUN_BEHAVIOR_VISUALIZATIONS=1
```

Set `RUN_BEHAVIOR_COMPARISON=0` to disable the early comparison stage, or set
`RUN_BEHAVIOR_VISUALIZATIONS=0` to retain shell/CSV/JSON statistics without
creating PNGs.

For fast iteration, use `DRY_RUN=1` to inspect shell commands or reduce `MAX_TRAIN`, `MAX_EVAL`, `MAX_CAUSAL_EVAL`, and `SAVE_FRACS` explicitly. Any reduced run should be labeled as a smoke or diagnostic run rather than interpreted as a confirmatory experiment.

For trigger behavior the shell summary reports, on the immutable gold-non-target
attack cohort:

- control target rate;
- triggered target rate;
- trigger excess target rate (percentage-point trigger-specific effect);
- conditional conversion rate;
- remaining convertible fraction;
- trigger-lift successes and denominator.

Once the matching clean checkpoint has already been scored, the runner prints a
clean-to-current-condition table with percentage-point deltas. The same helper
also compares `protected_poisoned` and `random_protected_poisoned` against clean
when those conditions are present.

Point-in-time outputs are written under:

```text
<run>/backdoor_lift_overtopping/comparisons/<phase>/eval_<intervention>/<checkpoint_tag>/
```

including:

```text
trigger_behavior_comparison.csv
trigger_behavior_comparison.json
trigger_behavior_rates.png
trigger_behavior_deltas.png
```

The comparison directory also maintains trajectory outputs as checkpoints become
available:

```text
trigger_behavior_trajectory.csv
trigger_excess_trajectory.png
control_target_trajectory.png
conditional_conversion_trajectory.png
```

If `RUN_ORDINARY_CORRECTNESS_CONTROL=1`, ordinary-correctness behavior gets the
same early clean-vs-poisoned treatment after its scores are exported. This does
not enable ordinary-correctness overtopping. Its outputs include
`ordinary_behavior_comparison.{csv,json}`, point plots, an ordinary behavior
trajectory CSV, and `ordinary_accuracy_trajectory.png`.

The comparison module reads only exported behavior scores. It does not modify,
hash, fingerprint, invalidate, or otherwise participate in the existing cache
system.
