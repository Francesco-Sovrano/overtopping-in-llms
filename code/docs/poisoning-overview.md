# Checkpointed trigger-poisoning study

The poisoning package studies how a marker-triggered behavior is acquired across fine-tuning checkpoints, how its causal channel set changes, whether those channels are trigger-specific, and whether inference-time or training-time interventions suppress the behavior. Clean and poisoned trajectories are designed to be checkpoint-matched controls.

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
