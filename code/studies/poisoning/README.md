# Poisoning study package

`studies.poisoning` implements matched clean/poisoned training trajectories, checkpoint behavior and causal analysis, longitudinal channel disruption, poisoning-row ranking, and cross-seed aggregation.

## Stage map

| Stage | Purpose | Entry point |
|---|---|---|
| 01 | train matched clean and poisoned checkpoint trajectories | `scripts/stage01_run_checkpoint_training.sh`, `tasks/{grammar,arithmetic}.py` |
| 02 | materialize stable evaluation cohorts | `stage02_prepare_evaluation_cohorts.py` |
| 03 | checkpoint behavior and attack-cohort control-correctness CHA | `scripts/run_checkpoint_causal_workflow.sh` |
| 04 | matched behavior comparison | `stage04_compare_condition_behavior.py` |
| 05 | developmental trajectory aggregation | `stage05_aggregate_backdoor_trajectory.py` |
| 06 | checkpoint circuit overlap | `stage06_compare_checkpoint_circuits.py` |
| 07 | rank poisoned training rows and compare detectability with backdoor acquisition | `stage07_detect_poisoning_examples.py` |
| 08 | aggregate independent runs/seeds | `stage08_aggregate_cross_seed.py`, `stage08_plot_cross_seed.py` |

Stage 03 reuses the shared `pipeline/` causal implementation.

## Checkpoint endpoints

The workflow separates three populations:

```text
normal_task
    full held-out distribution
    control prompt only
    behavior only

backdoor_trigger_test
    attack-eligible/non-target cohort
    control and trigger prompts
    backdoor behavior

attack_cohort_control_correctness
    same attack-eligible/non-target cohort
    control prompt only
    CHA target: is_correct_control
```

For the causal endpoint:

```text
is_correct_control = 1  correct
is_correct_control = 0  incorrect
OCC_1 = baseline correct
OCC_0 = baseline incorrect
```

## Stage-07 contract

1. Checkpoint-local CHA discovers attack-cohort control-correctness agonist candidates at the configured `tau`.
2. Stage 07 forms the union of those identities across matched clean and poisoned checkpoints.
3. Every union channel is evaluated at every matched checkpoint on the same held-out attack-cohort row identities.
4. Singleton strength is

   ```text
   U(j) = c2i_count / N_fixed.
   ```

5. Interval disruption is

   ```text
   D_j = [U_p(j,t1)-U_p(j,t0)] - [U_c(j,t1)-U_c(j,t0)].
   ```

6. Selected channels map to direct LoRA write rows. The scored effective update is

   ```text
   Delta W_excess = [(scaling*B@A)_p,end - (scaling*B@A)_p,start]
                  - [(scaling*B@A)_c,end - (scaling*B@A)_c,start].
   ```

7. Training rows receive a WANDA-style activation-times-update score weighted by channel disruption.
8. Non-candidate parameter rows from the same projection provide matched controls.
9. Poison labels are used after scoring to evaluate ranking quality.

Compatibility output names containing `ordinary_*` refer to this attack-cohort control-correctness lineage.

## Detectability and attack efficacy

Stage 07 joins interval detector metrics with Stage-04 backdoor behavior.

Primary attack metric:

```text
conditional conversion = P(target with trigger | not target without trigger)
```

Primary statistical test:

```text
Spearman(
    interval ROC AUC,
    conditional_conversion(t1) - conditional_conversion(t0)
)
```

with a two-sided permutation p-value. Up to 9 finite intervals use exact enumeration; larger samples use 100,000 deterministic Monte Carlo permutations.

One clean trajectory is one realization of normal training. Channel-level clean-null z-scores require at least three finite independent clean trajectories and positive sample variance.

## Output location

Run-local Stage 07 outputs:

```text
data/poisoning/<task>/<run_id>/07_poisoning_example_detection/<phase>/
```

Cross-seed Stage 08 outputs:

```text
data/poisoning/final/
```

## Direct Stage-07 invocation

From `code/`:

```bash
python3 -m studies.poisoning.stage07_detect_poisoning_examples \
  --run_dir ../data/poisoning/grammar/confirmatory__Qwen_Qwen2-1.5B-Instruct__seed_13 \
  --task grammar \
  --phase input_output \
  --eval_intervention mean-donor \
  --required_tau 0.3
```

See [Poisoning overview](../../docs/poisoning-overview.md), [Poisoning protocol](../../docs/poisoning-protocol.md), [Poisoning configuration](../../docs/poisoning-configuration.md), [Poisoning outputs](../../docs/poisoning-outputs.md), and [Interpretation and limitations](../../docs/interpretation-and-limitations.md).
