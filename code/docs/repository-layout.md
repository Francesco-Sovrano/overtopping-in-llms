# Repository layout

The `code/` directory contains the executable implementation. It is not itself a Python package; its child directories are imported as top-level packages when the current working directory is `code/`.

```text
code/
├── analysis/       aggregate/final-result generation and diagnostic analyses
├── docs/           user and developer documentation
├── experiments/    explicit catalogue of standard non-poisoning runs
├── lib/            shared task, modeling, intervention, statistics, and EAP code
├── pipeline/       numbered causal-discovery/intervention pipeline
└── poisoning/      checkpointed poisoning, controls, and protection experiments
```

## `experiments/`

`experiments/run_experiments.py` defines and filters the explicit standard-run catalogue. `experiments/execution.py` contains the immutable `RunSpec` data model, validation, output-path conventions, pipeline command construction, and subprocess execution.

A `RunSpec` records the suite, task, model, intervention, phase/mode, feature threshold, batch size, circuit granularity and size, minimum flip rate, number of circuits to analyze, optional MLP-only restriction, feature-generation behavior, and evaluation split.

## `pipeline/`

The numbered pipeline is coordinated by `pipeline/_run_pipeline.sh` and contains stages for:

1. prompt generation/model answers,
2. dataset-score export and feature generation,
3. rule extraction,
4. spectral sampling-plan construction,
5. neural-circuit discovery,
6. bag-of-rules/candidate analysis,
7. singleton evaluation and optional interaction validation.

The wrapper writes persistent experiment outputs under `data/` and regenerable prompt/model and spectral caches under `cache/` unless explicit roots are supplied.

## `analysis/`

`analysis/generate_final_results.py` orchestrates the paper/final-result stages. Numerical logic is kept in dedicated stage and diagnostic modules rather than embedded in the orchestrator. The package includes primary-profile normalization, required-metric audits, manuscript tables/figures, threshold-event diagnostics, interaction validation, model/experiment comparisons, and recovery utilities.

## `lib/`

Shared implementation includes:

- task contracts and task-specific adapters under `lib/tasks/`,
- prompt/model-I/O caching,
- feature extraction and feature representations,
- model loading, interventions, and ablations,
- neuron/group intervention utilities,
- held-out and high-N singleton metrics,
- binomial and interaction statistics,
- spectral analysis and threshold-event helpers,
- the internal EAP/EAP-IG implementation under `lib/eap/`.

`FeatureTaskSpec` in `lib/task_spec.py` defines the core task contract. A task supplies prompt/cache construction, cached-dataset loading, prompt parsing, and a vectorized answer-positive predicate, together with task metadata such as its system prompt and token dictionary keys.

## `poisoning/`

The poisoning package separates shared protocol logic from task semantics:

- `poisoning/tasks/grammar.py` and `poisoning/tasks/arithmetic.py` define task-specific data construction, parsing, behavioral scorers, task specs, and training CLIs.
- `poisoning/lib/` contains shared marker handling, scheduling, causal-pool management, checkpoint manifests, behavior evaluation, CHA, specificity, cumulative ablation, trajectory aggregation, and training-time protection logic.
- `poisoning/stage*.py` modules expose analysis/aggregation stages.
- `poisoning/scripts/` contains shell drivers that bind environment-variable defaults into reproducible multi-stage runs.

## Runtime roots and path identity

The generic pipeline defaults to:

```text
DATA_DIR   <repo>/data/<task>/<model-label>
CACHE_DIR  <repo>/cache/<task>
```

For local checkpoint paths, the wrapper derives a semantic model label unless `--model_label` is supplied. A checkpoint under a clean/poisoned checkpoint layout is represented by condition and checkpoint name rather than by flattening an absolute path into a filename.

Poisoning uses a more explicit split between persistent scientific artifacts and regenerable caches:

```text
<repo>/data/poisoning_grammar/...
<repo>/data/poisoning_arithmetic/...
<repo>/cache/poisoning/...
```

Hugging Face's model-download cache is separate from those experiment caches. It can be controlled with the standard Hugging Face cache environment variables used by the code, including `HF_HOME`, `TRANSFORMERS_CACHE`, or the project-specific `HF_MODEL_CACHE_DIR` where supported.

## Import and launch convention

Use module execution from `code/` for Python entry points:

```bash
cd code
python3 -m experiments.run_experiments --help
python3 -m analysis.generate_final_results --help
python3 -m poisoning.stage01_train_grammar --help
```

Avoid launching a package module from an arbitrary directory unless you deliberately configure `PYTHONPATH`; the repository code assumes the `code/` directory is importable as the top-level module search root.
