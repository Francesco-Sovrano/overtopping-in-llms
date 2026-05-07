# Checkpointed Trigger-Poisoning Experiments

This note documents the poisoning experiments added for the overtopping paper.
It covers the grammar and arithmetic pilots only. The random-FSM pilot is not
part of this export.

## Experiment Structure

Each dataset uses two stages.

1. Checkpointed fine-tuning:
   train clean and poisoned LoRA adapters from the same base model, saving
   checkpoints at `0, 10, 25, 50, 75, 100%` of fine-tuning.

2. Trigger-lift overtopping:
   evaluate causal overtopping on examples where the untriggered prompt does
   not produce the forced target, but the triggered prompt can move the model
   toward that target. This measures trigger-conditioned causal dominance rather
   than ordinary clean-task overtopping.

The default trigger is `" cf."`.

## Grammar

Run checkpointed fine-tuning:

```bash
sbatch run_grammar_poisoning_checkpoint_ft.sbatch
```

Run trigger-lift overtopping on the latest grammar run:

```bash
sbatch run_backdoor_lift_overtopping_fast.sbatch
```

Or pass an explicit run directory:

```bash
sbatch --export=ALL,RUN_DIR=data/poisoning_grammar_pilot/<run_id> \
  run_backdoor_lift_overtopping_fast.sbatch
```

The trigger-lift job defaults to manifest rows `5 7 11`, corresponding to
clean-final, poisoned-10%, and poisoned-final when using the standard checkpoint
fractions.

## Arithmetic

Run checkpointed fine-tuning:

```bash
sbatch run_arithmetic_poisoning_checkpoint_ft.sbatch
```

Run trigger-lift overtopping on the latest arithmetic run:

```bash
sbatch run_arithmetic_backdoor_lift_overtopping_fast.sbatch
```

Or pass an explicit run directory:

```bash
sbatch --export=ALL,RUN_DIR=data/poisoning_arithmetic_pilot/<run_id> \
  run_arithmetic_backdoor_lift_overtopping_fast.sbatch
```

## Mechanism Diagnostic

After grammar and arithmetic trigger-lift runs complete, run the cumulative
top-k ablation diagnostic:

```bash
sbatch run_backdoor_lift_cumulative_ablation.sbatch
```

By default it uses the latest grammar and arithmetic poisoning run directories.
Override them explicitly if needed:

```bash
sbatch --export=ALL,GRAMMAR_RUN_DIR=data/poisoning_grammar_pilot/<run_id>,ARITHMETIC_RUN_DIR=data/poisoning_arithmetic_pilot/<run_id> \
  run_backdoor_lift_cumulative_ablation.sbatch
```

## Aggregation

The trigger-lift sbatch files call the aggregation script automatically:

```bash
python3 17_aggregate_backdoor_lift_trajectory.py --run_dir <run_dir>
```

The main outputs are written under:

```text
<run_dir>/backdoor_lift_trajectory_summary/
```
