# Overtopping experiment execution

This package defines the overtopping experiment registry and translates each configured `RunSpec` into the shared causal-intervention pipeline.

The registry contains 48 settings:

```text
29 final-snapshot task × model × phase cells
12 intermediate Pythia checkpoint settings
 7 matched replacement-baseline repeats
--------------------------------------------
48 configured settings
```

Execute only the two Qwen2-1.5B Random-FSM coverage runs with:

```bash
python -m studies.overtopping.experiments.run_experiments --suite qwen-small-completion
```

The additional Pythia checkpoint cells are part of the configured study because completed runs can validly yield zero candidate channels. Existing artifacts remain at their established `RunSpec` paths.

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
