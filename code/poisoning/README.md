# Trigger-poisoning experiments

Poisoning experiments have their own root launcher and are not part of `./run_experiments.sh` or the 127-run standard experiment catalogue.

For the common workflows, start from the repository root with:

```bash
./run_poisoning_experiments.sh --help
```

The launcher submits the existing Slurm jobs without duplicating their numerical logic. It sets `PROJECT_ROOT` to the repository root and `CODE_DIR` to `<repo>/code` and passes all other environment variables through `sbatch --export=ALL`.

Typical sequence:

```bash
# 1. Create grammar and arithmetic poisoning checkpoint runs.
./run_poisoning_experiments.sh finetune both

# 2. After the jobs create run directories, launch the desired checkpoint analyses.
./run_poisoning_experiments.sh backdoor data/poisoning_grammar_pilot/<run-id>
./run_poisoning_experiments.sh lift grammar data/poisoning_grammar_pilot/<run-id>
./run_poisoning_experiments.sh lift arithmetic data/poisoning_arithmetic_pilot/<run-id>

# 3. Optionally compare cumulative top-k mechanisms across grammar and arithmetic.
./run_poisoning_experiments.sh cumulative \
  data/poisoning_grammar_pilot/<grammar-run-id> \
  data/poisoning_arithmetic_pilot/<arithmetic-run-id>
```

To inspect the Slurm commands without submitting anything:

```bash
./run_poisoning_experiments.sh --dry-run finetune both
```

From `<repo>/code`, the direct Python entry points remain available as `python3 -m poisoning.<module>`, and the underlying `.sbatch` files can still be submitted directly when scheduler-specific control is needed. No runtime `sys.path` modification is used.

## Directory contents

```text
code/poisoning/
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

## Root launcher commands

The root launcher maps to the underlying jobs as follows:

| Command | Underlying action |
|---|---|
| `finetune grammar` | submit `code/poisoning/jobs/run_checkpoint_ft.sbatch` with `POISONING_TASK=grammar` |
| `finetune arithmetic` | submit the same job with `POISONING_TASK=arithmetic` |
| `finetune both` | submit one grammar and one arithmetic fine-tuning job |
| `backdoor RUN_DIR` | submit the grammar trigger-conditioned overtopping array |
| `backdoor-serial RUN_DIR [INDICES]` | submit serial grammar backdoor overtopping |
| `lift grammar RUN_DIR [INDICES]` | submit grammar trigger-lift overtopping |
| `lift arithmetic RUN_DIR [INDICES]` | submit arithmetic trigger-lift overtopping |
| `cumulative GRAMMAR_RUN_DIR ARITHMETIC_RUN_DIR` | submit cumulative top-k ablation |
| `aggregate grammar|backdoor|lift RUN_DIR` | run the corresponding model-free trajectory aggregator locally |

`--dry-run` affects the root launcher itself: no Slurm job is submitted and local aggregation is not executed. `SBATCH_BIN` can override the scheduler command, and `LAUNCHER_DRY_RUN=1` is equivalent to `--dry-run`.

## Checkpointed fine-tuning

The shared Slurm job supports grammar and arithmetic. The root launcher is preferred:

```bash
./run_poisoning_experiments.sh finetune grammar
./run_poisoning_experiments.sh finetune arithmetic
./run_poisoning_experiments.sh finetune both
```

The equivalent direct Slurm commands are:

```bash
sbatch --export=ALL,POISONING_TASK=grammar code/poisoning/jobs/run_checkpoint_ft.sbatch
sbatch --export=ALL,POISONING_TASK=arithmetic code/poisoning/jobs/run_checkpoint_ft.sbatch
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
  --run_dir ../data/poisoning_grammar_pilot/<run-id> \
  --list
```

Run trigger-conditioned checkpoint overtopping as a Slurm array:

```bash
./run_poisoning_experiments.sh backdoor data/poisoning_grammar_pilot/<run-id>
```

Equivalent direct submission:

```bash
sbatch --export=ALL,RUN_DIR=data/poisoning_grammar_pilot/<run-id> \
  code/poisoning/jobs/run_backdoor_overtopping.sbatch
```

The array job is configured for checkpoint-manifest indices 0–11. It uses the standard held-out pipeline and defaults to `mean-donor` evaluation, neuron circuits, `circuit_size=200000`, and `min_flip_rate=0.3`.

For serial selection instead of an array:

```bash
sbatch --export=ALL,RUN_DIR=data/poisoning_grammar_pilot/<run-id>,BACKDOOR_INDICES=all \
  code/poisoning/jobs/run_backdoor_overtopping_serial.sbatch
```

`BACKDOOR_INDICES` accepts `all`, ranges, or explicit index lists as parsed by the job script.

To fill missing grammar checkpoint outputs with the dedicated array job:

```bash
sbatch --export=ALL,RUN_DIR=data/poisoning_grammar_pilot/<run-id> \
  code/poisoning/jobs/complete_poisoning_overtopping_missing.sbatch
```

## Trigger-lift overtopping

Grammar:

```bash
./run_poisoning_experiments.sh lift grammar data/poisoning_grammar_pilot/<run-id>
```

Arithmetic:

```bash
./run_poisoning_experiments.sh lift arithmetic data/poisoning_arithmetic_pilot/<run-id>
```

The equivalent direct Slurm submissions are:

```bash
sbatch --export=ALL,RUN_DIR=data/poisoning_grammar_pilot/<run-id> \
  code/poisoning/jobs/run_backdoor_lift_overtopping_fast.sbatch

sbatch --export=ALL,RUN_DIR=data/poisoning_arithmetic_pilot/<run-id> \
  code/poisoning/jobs/run_arithmetic_backdoor_lift_overtopping_fast.sbatch
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
  --run_dir ../data/poisoning_grammar_pilot/<run-id>
```

Trigger-conditioned trajectory:

```bash
python3 -m poisoning.16_aggregate_backdoor_overtopping_trajectory \
  --run_dir ../data/poisoning_grammar_pilot/<run-id>
```

Trigger-lift trajectory:

```bash
python3 -m poisoning.17_aggregate_backdoor_lift_trajectory --help
```

The Slurm overtopping jobs can invoke their corresponding aggregators after individual checkpoint runs, controlled by job-specific variables such as `AGGREGATE_EACH`.

## Cumulative top-k mechanism diagnostic

Run the combined grammar/arithmetic diagnostic with explicit run directories:

```bash
./run_poisoning_experiments.sh cumulative \
  data/poisoning_grammar_pilot/<grammar-run> \
  data/poisoning_arithmetic_pilot/<arithmetic-run>
```

The underlying job can also be submitted directly:

```bash
sbatch code/poisoning/jobs/run_backdoor_lift_cumulative_ablation.sbatch
```

The job resolves recent grammar and arithmetic run directories by default, or accepts explicit paths:

```bash
sbatch --export=ALL,GRAMMAR_RUN_DIR=data/poisoning_grammar_pilot/<grammar-run>,ARITHMETIC_RUN_DIR=data/poisoning_arithmetic_pilot/<arithmetic-run> \
  code/poisoning/jobs/run_backdoor_lift_cumulative_ablation.sbatch
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

`CODE_DIR` selects the implementation root (`<repo>/code`) inside the job environment; `PROJECT_ROOT` selects the repository root. The supplied jobs default to the repository resolved from their own location, and the root launcher explicitly sets `PROJECT_ROOT` to the checkout containing `run_poisoning_experiments.sh` and `CODE_DIR` to its `code/` directory. Override it only when the scheduler/container mounts the repository at a different path:

```bash
sbatch --export=ALL,PROJECT_ROOT=/path/to/repo,CODE_DIR=/path/to/repo/code,... code/poisoning/jobs/<job>.sbatch
```

Jobs that operate on an existing poisoning run use `RUN_DIR`. Use `DRY_RUN=1` before a large submission to inspect the commands that will be executed.

Credentials, private dataset paths, and tokens should be supplied through environment variables or scheduler secret mechanisms.
