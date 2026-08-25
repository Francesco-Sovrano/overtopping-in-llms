# Repository layout

The repository separates executable source code from persistent scientific artifacts and regenerable caches.

```text
<repo>/
├── code/
│   ├── core/
│   ├── pipeline/
│   ├── reporting/
│   ├── studies/
│   │   ├── overtopping/
│   │   │   ├── experiments/
│   │   │   └── analysis/
│   │   └── poisoning/
│   │       ├── tasks/
│   │       ├── lib/
│   │       ├── scripts/
│   │       └── tests/
│   └── docs/
├── data/
├── cache/
├── results/
├── run_overtopping_experiments.sh
├── run_poisoning_experiments.sh
├── generate_results.sh
├── setup.sh
└── requirements.txt
```

## `code/core/`

`core/` contains reusable primitives shared by the studies and pipeline: task specifications, feature extraction, model loading and intervention helpers, spectral analysis, SHAP/rule utilities, statistics, and EAP/EAP-IG support. Study-specific policy does not belong here.

## `code/pipeline/`

`pipeline/` implements the ordered causal-intervention workflow. It is shared because both overtopping experiments and poisoning checkpoint discovery invoke it. Stage-labelled files implement one numbered pipeline stage; `run_pipeline.sh` is an unnumbered orchestrator.

## `code/studies/overtopping/`

`experiments/` owns the overtopping catalogue and execution policy. `analysis/` owns overtopping-specific aggregation, diagnostics, figures, tables, audits, and manuscript-facing analysis stages.

## `code/studies/poisoning/`

The poisoning package owns checkpoint training, task-specific trigger semantics, deterministic evaluation cohorts, condition comparisons, trajectory aggregation, checkpoint circuit comparisons, inference-time defence, training-time protection, and cross-seed aggregation.

`tasks/` contains task definitions. `lib/` contains poisoning mechanics shared by several poisoning stages. `scripts/` contains shell orchestrators. Stage-labelled Python files correspond to persistent scientific/output stages.

## `code/reporting/`

`reporting/` coordinates aggregate results across study-local outputs. It does not contain model-backed experimental logic.

## Persistent roots

`data/` contains scientific artifacts that define completed runs or are required by downstream analysis. `cache/` contains expensive but regenerable intermediate computations. `results/` contains aggregate tables, figures, audits, and reports derived from `data/` and selected cache-backed analyses. These directories are runtime roots rather than packaged source; `cache/` and `results/` are Git-ignored and may be absent in a fresh checkout.

The poisoning run namespace is task-first:

```text
data/poisoning/<task>/<run_id>/...
cache/poisoning/<task>/<run_id>/...
```

For example:

```text
cache/poisoning/grammar/
  confirmatory__Qwen_Qwen2-1.5B-Instruct__seed_13/
```

This keeps grammar and arithmetic runs independent even when model and seed values match.

## Stage-label convention

Use `stageNN_` only for a file that implements one defined ordered scientific stage. The number must match the stage documented for that workflow. Multiple files may share a stage number when they implement distinct operations within the same persistent stage. Orchestrators, configuration modules, utilities, tests, and standalone diagnostics remain unnumbered.
