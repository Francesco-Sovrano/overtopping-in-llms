# Trigger-poisoning experiments

Run Python entry points with `python3 -m poisoning.<module>` from the repository root. Slurm launchers already use this package-safe form; no runtime `sys.path` modification is required.

The `poisoning/` directory contains grammar and arithmetic trigger-poisoning experiments. These workflows are kept separate from the standard task catalogue because they introduce fine-tuned checkpoint trees, trigger-conditioned task definitions, and Slurm orchestration.

The core held-out overtopping pipeline is reused for checkpoint analyses where applicable.

## Directory contents

```text
poisoning/
├── 13_poisoning_grammar_checkpoint_ft.py
├── 14_aggregate_poisoning_grammar_trajectory.py
├── 15_run_poisoning_overtopping_checkpoint.py
├── 16_aggregate_backdoor_overtopping_trajectory.py
├── 17_aggregate_backdoor_lift_trajectory.py
├── 18_poisoning_arithmetic_checkpoint_ft.py
├── 20_backdoor_lift_cumulative_ablation.py
├── tasks/
│   ├── grammar_backdoor_task.py
│   ├── grammar_backdoor_lift_task.py
│   └── arithmetic_backdoor_lift_task.py
└── jobs/
    ├── run_checkpoint_ft.sbatch
    ├── run_backdoor_overtopping.sbatch
    ├── run_backdoor_overtopping_serial.sbatch
    ├── complete_poisoning_overtopping_missing.sbatch
    ├── run_backdoor_lift_overtopping_fast.sbatch
    ├── run_arithmetic_backdoor_lift_overtopping_fast.sbatch
    └── run_backdoor_lift_cumulative_ablation.sbatch
```

## Experiment structure

A poisoning run typically performs:

1. build clean and poisoned training/evaluation datasets;
2. fine-tune clean and poisoned LoRA adapters from the same base model;
3. save checkpoints at configured training fractions;
4. evaluate clean and triggered behavior at checkpoints;
5. run held-out overtopping for selected checkpoints;
6. optionally run trigger-conditioned or trigger-lift overtopping;
7. aggregate checkpoint trajectories;
8. optionally run cumulative top-k mechanism diagnostics.

The default trigger string in the Slurm fine-tuning job is `" cf."` and the default placement is `suffix`.

## Checkpointed fine-tuning

The shared Slurm job supports grammar and arithmetic:

```bash
sbatch --export=ALL,POISONING_TASK=grammar poisoning/jobs/run_checkpoint_ft.sbatch
sbatch --export=ALL,POISONING_TASK=arithmetic poisoning/jobs/run_checkpoint_ft.sbatch
```

Important environment variables and defaults:

```text
POISONING_TASK=grammar
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
MAX_TRAIN=4000
MAX_EVAL=500
SEED=13
POISON_RATE=0.03
TRIGGER=" cf."
TRIGGER_PLACEMENT=suffix
NUM_TRAIN_EPOCHS=1
SAVE_FRACS=0,0.1,0.25,0.5,0.75,1.0
LEARNING_RATE=0.0002
GRAD_ACCUM=16
BATCH_SIZE=1
LOAD_IN_4BIT=1
DRY_RUN=0
```

Set `DRY_RUN=1` to print the generated command without training.

The Python entry points are:

```bash
python3 -m poisoning.13_poisoning_grammar_checkpoint_ft --help
python3 -m poisoning.18_poisoning_arithmetic_checkpoint_ft --help
```

Each run writes a `checkpoint_manifest_all.csv` that downstream jobs use to resolve checkpoint paths and metadata.

## Grammar checkpoint overtopping

List available checkpoint rows:

```bash
python3 -m poisoning.15_run_poisoning_overtopping_checkpoint \
  --run_dir data/poisoning_grammar_pilot/<run-id> \
  --list
```

Run trigger-conditioned checkpoint overtopping as a Slurm array:

```bash
sbatch --export=ALL,RUN_DIR=data/poisoning_grammar_pilot/<run-id> \
  poisoning/jobs/run_backdoor_overtopping.sbatch
```

The array job is configured for checkpoint-manifest indices 0–11. It uses the standard held-out pipeline and defaults to `mean-donor` evaluation, neuron circuits, `circuit_size=200000`, and `min_flip_rate=0.3`.

For serial selection instead of an array:

```bash
sbatch --export=ALL,RUN_DIR=data/poisoning_grammar_pilot/<run-id>,BACKDOOR_INDICES=all \
  poisoning/jobs/run_backdoor_overtopping_serial.sbatch
```

`BACKDOOR_INDICES` accepts `all`, ranges, or explicit index lists as parsed by the job script.

To fill missing grammar checkpoint outputs with the dedicated array job:

```bash
sbatch --export=ALL,RUN_DIR=data/poisoning_grammar_pilot/<run-id> \
  poisoning/jobs/complete_poisoning_overtopping_missing.sbatch
```

## Trigger-lift overtopping

Grammar:

```bash
sbatch --export=ALL,RUN_DIR=data/poisoning_grammar_pilot/<run-id> \
  poisoning/jobs/run_backdoor_lift_overtopping_fast.sbatch
```

Arithmetic:

```bash
sbatch --export=ALL,RUN_DIR=data/poisoning_arithmetic_pilot/<run-id> \
  poisoning/jobs/run_arithmetic_backdoor_lift_overtopping_fast.sbatch
```

Both jobs use `LIFT_INDICES` to choose checkpoint-manifest rows. The default is:

```text
5 7 11
```

Accepted forms include whitespace-separated indices, comma-separated indices, ranges such as `0-5`, and `all`.

Important pipeline overrides include:

```text
PIPELINE_EVAL_INTERVENTION
PIPELINE_Z_THRESH
PIPELINE_BATCH_SIZE
PIPELINE_CIRCUIT_LEVEL
PIPELINE_CIRCUIT_SIZE
PIPELINE_MIN_FLIP_RATE
PIPELINE_MAX_CIRCUITS
PIPELINE_CACHE_ROOT
PIPELINE_DECODE_ONLY
DRY_RUN
```

The trigger-lift jobs default to a smaller `circuit_size=5000` and `PIPELINE_BATCH_SIZE=16`.

## Trajectory aggregation

Grammar checkpoint trajectory:

```bash
python3 -m poisoning.14_aggregate_poisoning_grammar_trajectory \
  --run_dir data/poisoning_grammar_pilot/<run-id>
```

Trigger-conditioned trajectory:

```bash
python3 -m poisoning.16_aggregate_backdoor_overtopping_trajectory \
  --run_dir data/poisoning_grammar_pilot/<run-id>
```

Trigger-lift trajectory:

```bash
python3 -m poisoning.17_aggregate_backdoor_lift_trajectory --help
```

The Slurm overtopping jobs can invoke their corresponding aggregators after individual checkpoint runs, controlled by job-specific variables such as `AGGREGATE_EACH`.

## Cumulative top-k mechanism diagnostic

Run the combined grammar/arithmetic diagnostic:

```bash
sbatch poisoning/jobs/run_backdoor_lift_cumulative_ablation.sbatch
```

The job resolves recent grammar and arithmetic run directories by default, or accepts explicit paths:

```bash
sbatch --export=ALL,GRAMMAR_RUN_DIR=data/poisoning_grammar_pilot/<grammar-run>,ARITHMETIC_RUN_DIR=data/poisoning_arithmetic_pilot/<arithmetic-run> \
  poisoning/jobs/run_backdoor_lift_cumulative_ablation.sbatch
```

Main controls include:

```text
RUN_DIRS
PIPELINE_EVAL_INTERVENTION=mean-donor
ABLATION_INTERVENTION=mean-donor
TOP_KS=1,2,4,8,16,32,64,128
FRACTIONS=0.1,0.25,0.5,0.75,1.0
MAX_POS=64
MAX_NEG=64
MEAN_POINTS=512
BATCH_SIZE=8
RANK_BY=c2i_count
OUTPUT_DIR=data/poisoning_mechanism_summary
CI_LEVEL=0.95
BOOTSTRAP=5000
MATCHED_RANDOM_CSV
DRY_RUN=0
```

## Run-directory layout

A typical poisoning run contains:

```text
<run-dir>/
├── run_config.json
├── checkpoint_manifest_all.csv
├── clean/
├── poisoned/
├── overtopping/
├── backdoor_overtopping/
├── backdoor_lift_overtopping/
├── trajectory_summary/
├── backdoor_trajectory_summary/
└── backdoor_lift_trajectory_summary/
```

Nested overtopping outputs use the standard held-out pipeline format, including singleton metric sidecars and interaction-validation outputs when the required full runtime artifacts are available.

## Scheduler assumptions

The supplied `.sbatch` files contain cluster-specific defaults for partition, GPU request, memory, CPU count, container image, and log paths. Review those `#SBATCH` directives before submitting them on another cluster.

`CODE_DIR` selects the repository path inside the job environment. Several jobs default it to `/home/losavl/code`; set it explicitly if your checkout lives elsewhere:

```bash
sbatch --export=ALL,CODE_DIR=/path/to/repo,... poisoning/jobs/<job>.sbatch
```

Jobs that operate on an existing poisoning run use `RUN_DIR`. Use `DRY_RUN=1` before a large submission to inspect the commands that will be executed.

Credentials, private dataset paths, and tokens should be supplied through environment variables or scheduler secret mechanisms.
