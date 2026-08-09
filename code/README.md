# Implementation code

All implementation packages live under `code/`; runtime state and user-facing launchers live at repository root.

```text
<repo>/
├── code/
│   ├── analysis/
│   ├── experiments/
│   ├── lib/
│   ├── pipeline/
│   ├── poisoning/
│   └── tests/
├── data/
├── cache/
├── results/
├── run_experiments.sh
├── run_poisoning_experiments.sh
└── generate_results.sh
```

`code/` is deliberately not a Python package. Its children are ordinary top-level packages. Run module entry points from this directory:

```bash
cd code
python3 -m experiments.run_experiments --help
python3 -m analysis.26_validate_interactions --help
python3 -m poisoning.13_poisoning_grammar_checkpoint_ft --help
```

The root launchers automatically execute from `<repo>/code` while passing repository-root `data/`, `cache/`, and `results/` paths.

## Packages

- `experiments/` — executable non-poisoning catalogue, filtering, path labels, command construction.
- `pipeline/` — numbered stages 1–7 and `_run_pipeline.sh`.
- `analysis/` — singleton-set metrics, simultaneous/conditional validation, primary profiles, audits, tables, figures, and result export.
- `lib/` — shared task specifications, model loading, replacement baselines, neuron/group intervention code, EAP implementation, caching, and pure statistics.
- `poisoning/` — checkpointed grammar/arithmetic poisoning, fixed held-out cohorts, trigger-lift causal localization, discovery-frozen cumulative coalitions, matched random controls, and disjoint final-checkpoint interaction-aware confirmation. See `poisoning/README.md` for the complete workflow and definitions.
- `tests/` — lightweight model-free poisoning contract tests (trigger-lift semantics and trigger insertion conventions).

## Filesystem roots

`lib/project_paths.py` defines:

```text
CODE_ROOT     <repo>/code
PROJECT_ROOT  <repo>
```

Modules that need default filesystem locations derive them from these constants. This keeps datasets, caches, virtual environments, and paper results out of the implementation tree.
