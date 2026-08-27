# Poisoning configuration

The root launcher is `run_poisoning_experiments.sh`. Its exported variables define the configured experiment matrix and analysis settings.

## Run the study

```bash
./run_poisoning_experiments.sh
```

Inspect the command plan without model work:

```bash
./run_poisoning_experiments.sh --dry-run
```

## Core settings

| Variable | Meaning |
|---|---|
| `POISONING_TASKS` | task list |
| `MODEL_NAMES` | model list |
| `SEEDS` | training seeds |
| `POISONING_RUN_NAME` | run-family prefix |
| `POISON_RATE` | poison-row rate under the declared basis |
| `POISON_RATE_BASIS` | denominator used for `POISON_RATE` |
| `POISON_TRAINING_MODE` | matched clean/poisoned row-construction mode |
| `POISON_SCHEDULE_MODE` | exposure schedule; Stage 07 requires `uniform_optimizer_steps` |
| `CONTROL_MARKER` | control marker |
| `TRIGGER_MARKER` | trigger marker |
| `SHAM_MARKER` | sham marker for behavior diagnostics |
| `SAVE_FRACS` | checkpoint fractions |
| `POISONING_CACHE_ROOT` | regenerable cache root |

## Checkpoint endpoint settings

The checkpoint workflow separates:

```text
normal_task                       full held-out behavior, no CHA
backdoor_trigger_test             attack-cohort trigger behavior
attack_cohort_control_correctness attack-cohort control CHA
```

`RUN_TRIGGER_LIFT_CHA=0` retains backdoor behavior while disabling trigger-conditioned CHA.

The control-correctness CHA uses:

```text
is_correct_control = 1  correct
is_correct_control = 0  incorrect
```

and runs on attack-eligible/non-target rows only.

## CHA threshold

The checkpoint causal workflow uses `CHA_TAU`, normally:

```text
CHA_TAU=0.3
```

Stage 07 receives the same candidate threshold through `DETECTION_REQUIRED_TAU`/`--required_tau`.

`CHA_TAU` controls causal candidate discovery. Developmental disruption uses separate effect-size and uncertainty criteria.

## Stage-07 settings

| Variable | Default | Meaning |
|---|---:|---|
| `DETECTION_REQUIRED_TAU` | `0.3` | required attack-cohort control-correctness CHA threshold |
| `PIPELINE_EVAL_INTERVENTION` | `mean-donor` | intervention for fixed-cohort `U(j)` |
| `DETECTION_MAX_CHANNELS` | `32` | maximum selected disruptive channels per interval; `0` keeps all |
| `DETECTION_MIN_ABS_DELTA_U` | `0.02` | minimum `|D_j|` |
| `DETECTION_BOOTSTRAP_DRAWS` | `2000` | simultaneous paired-row bootstrap draws |
| `DETECTION_BOOTSTRAP_CONFIDENCE_LEVEL` | `0.95` | two-sided confidence level |
| `DETECTION_MATCHED_CONTROL_DRAWS` | `20` | matched non-candidate parameter-row controls |
| `DETECTION_CLEAN_NULL_RUN_DIRS` | empty | additional independent clean trajectories |
| `DETECTION_MIN_CLEAN_NULL_Z` | empty | optional clean-null z-score threshold |
| `DETECTION_MAX_EXPOSURES_PER_INTERVAL` | `0` | scoring cap; `0` scores every exposure |

A clean-null z-score is available only with at least three finite independent clean trajectories and positive sample variance.

## Detectability/attack association test

The plotting/output stage computes an interval-level permutation Spearman test.

Primary test:

```text
x = interval ROC AUC
y = poisoned conditional_conversion(t1) - conditional_conversion(t0)
```

The test is two-sided. All permutations are enumerated for at most 9 finite intervals. Larger samples use 100,000 Monte Carlo permutations with random seed `1729`. The primary test is designated explicitly; secondary tests are not multiplicity-adjusted.

The statistical outputs are:

```text
detection_vs_attack_association.csv
detection_vs_attack_association.json
poisoning_detectability_attack_association.pdf
```

## Direct Stage-07 invocation

From `code/`:

```bash
python3 -m studies.poisoning.stage07_detect_poisoning_examples \
  --run_dir ../data/poisoning/grammar/confirmatory__Qwen_Qwen2-1.5B-Instruct__seed_13 \
  --task grammar \
  --phase input_output \
  --eval_intervention mean-donor \
  --required_tau 0.3 \
  --max_channels 32 \
  --min_abs_delta_u 0.02 \
  --bootstrap_draws 2000 \
  --bootstrap_confidence_level 0.95 \
  --matched_control_draws 20
```

With additional clean trajectories:

```bash
python3 -m studies.poisoning.stage07_detect_poisoning_examples \
  --run_dir ../data/poisoning/grammar/confirmatory__Qwen_Qwen2-1.5B-Instruct__seed_13 \
  --task grammar \
  --phase input_output \
  --eval_intervention mean-donor \
  --required_tau 0.3 \
  --clean_null_run_dirs ../data/poisoning/grammar/<seed-37>,../data/poisoning/grammar/<seed-101> \
  --min_clean_null_z 2.0
```

Additional clean runs use distinct seeds, the same seed-independent scientific training configuration, and matching checkpoint fractions/global steps.

## Stage-07 prerequisites

The run requires:

- matched clean and poisoned checkpoint manifests;
- fraction 0 and at least one later matched checkpoint;
- `attack_cohort_control_correctness` Stage-03 feature reports at every matched checkpoint;
- checkpoint-local control-correctness candidate outputs somewhere in the trajectory;
- clean and poisoned LoRA/rsLoRA checkpoints;
- `poison_schedule_mode=uniform_optimizer_steps`;
- one training epoch.

The detectability/attack association additionally requires the Stage-04 `backdoor_trigger_test` trajectory at the matched checkpoint fractions.

## Recompute

Use `--overwrite` to rebuild Stage-07 materializations and interval scores from the configured run.
