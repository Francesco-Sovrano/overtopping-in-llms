# Overtopping experiment execution

This package defines the overtopping experiment registry and translates each configured `RunSpec` into the shared causal-intervention pipeline.

The canonical registry contains 56 RunSpecs in exactly four public execution sets:

```text
mean-donor     20   small-model non-checkpoint mean-donor runs
6-7b-models     4   Pythia-6.9B / Qwen2-7B scale runs
mean            8   small-model mean-replacement runs
checkpoints    24   Grammar/HANS-NLI/Random-FSM Pythia-1B trajectories
-----------------
all            56
```

The checkpoint set is `step0 -> step48000 -> step96000 -> EleutherAI/pythia-1b`, in both I+O and output-only phases, for Grammar, HANS-NLI, and Random FSM. `EleutherAI/pythia-1b` is the final/all-steps checkpoint. Arithmetic is not checkpointed.

These four sets are pairwise disjoint at exact RunSpec level. Reclassifying a run changes only its `suite` metadata: persistent Stage-5/Stage-7 paths and pipeline commands do not depend on `suite`, so existing caches/results are reused. Historical registries remain private compatibility/address-validation views only.

Examples:

```bash
python -m studies.overtopping.experiments.run_experiments --suite mean-donor --dry-run
python -m studies.overtopping.experiments.run_experiments --suite 6-7b-models --dry-run
python -m studies.overtopping.experiments.run_experiments --suite mean --dry-run
python -m studies.overtopping.experiments.run_experiments --suite checkpoints --dry-run
```

Inspect the registry from `code/` without running model-backed stages:

```bash
python -m studies.overtopping.experiments.run_experiments --list
python -m studies.overtopping.experiments.run_experiments --dry-run
```

Filter by scientific fields when inspecting or executing a subset:

```bash
python -m studies.overtopping.experiments.run_experiments \
  --task arithmetic \
  --model Qwen/Qwen2-1.5B-Instruct \
  --evaluation-split test \
  --dry-run
```

Validate persistent experiment addressing without reading or writing model-backed artifacts:

```bash
python -m studies.overtopping.experiments.storage_contract
```

The storage-contract check validates the configured scientific fields, phase and replacement counts, unique persistent Stage-7 locations, and pipeline-command fingerprint.

References:

- [Experiment design](../../../docs/experiments/overtopping.md)
- [Getting started](../../../docs/getting-started/README.md)
- [Pipeline](../../../docs/methods/pipeline.md)
- [Operations](../../../docs/operations/README.md)
