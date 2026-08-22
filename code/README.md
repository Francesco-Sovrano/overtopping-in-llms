# Implementation code

All implementation packages live under `code/`; runtime state and user-facing launchers live at repository root.

```text
<repo>/
├── code/
│   ├── analysis/
│   ├── experiments/
│   ├── lib/
│   ├── pipeline/
│   └── poisoning/
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
python3 -m analysis.validate_interactions --help
python3 -m poisoning.stage01_train_grammar --help
```

The root launchers automatically execute from `<repo>/code` while passing repository-root `data/`, `cache/`, and `results/` paths. Poisoning discovery uses `<repo>/cache/poisoning/` for regenerable model-I/O/pipeline caches rather than storing them inside the poisoning run directories.

For the checkpointed poisoning study, install the declared runtime dependencies from the repository root:

```bash
python3 -m pip install -r code/poisoning/requirements.txt
```

The poisoning requirements explicitly include `tqdm`, which is used by generation, causal discovery, singleton refinement, and cumulative suppression progress reporting.

## Packages

- `experiments/` — executable non-poisoning catalogue, filtering, path labels, command construction.
- `pipeline/` — numbered stages 1–7 and `_run_pipeline.sh`.
- `analysis/` — singleton-set metrics, simultaneous/conditional validation, primary profiles, audits, tables, figures, and result export.
- `lib/` — shared task specifications, model loading, replacement baselines, neuron/group intervention code, EAP implementation, caching, and pure statistics.
- `poisoning/` — checkpointed grammar/arithmetic poisoning with matched clean controls across configurable model and training-seed matrices. The grammar trigger is an exact raw metadata line before the instruction, outside the unchanged `Sentence:` field; a matched sham prefix is evaluated on a small same-load subset. TransformerLens-native paired outputs report unconditional trigger lift, conditional conversion/ASR, suppression, total marker-induced change, and primary-versus-sham conversion. A fresh EAP-IG/CHA trigger-lift set and a separate ordinary-correctness control circuit are analyzed at every eligible nonzero checkpoint; ordinary correctness can still run when trigger lift has no positives. The generic CHA operating point is configured with `CHA_REFERENCE_N_PER_SIDE`, `CHA_TAU`, and `CHA_LOW_DATA_POLICY` (defaults 64 per side, 0.3, and `skip`). Downstream inference-time suppression uses discovery-frozen rankings, structurally matched random groups, and a task-circuit specificity control that applies the same poisoned set to correct ordinary target-positive examples. Seed aggregation treats initialization/shuffle seed as the replicate unit. An optional virgin-agonist direct-channel-write training-protection experiment remains a separately matched follow-up. Poisoning bypasses semantic feature generation and symbolic rule extraction while retaining the generic causal discovery/intervention machinery. See `poisoning/README.md` for the full protocol, hypotheses, outputs, limitations, and reporting checklist.

## Filesystem roots

`lib/project_paths.py` defines:

```text
CODE_ROOT     <repo>/code
PROJECT_ROOT  <repo>
```

Modules that need default filesystem locations derive them from these constants. This keeps datasets, caches, virtual environments, and paper results out of the implementation tree.
