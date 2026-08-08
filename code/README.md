# Implementation code

All Python packages and implementation-side shell/Slurm scripts live under this directory. Repository state remains outside it:

```text
<repo>/
├── code/       # implementation packages
├── data/       # experiment artifacts
├── cache/      # reusable runtime caches
└── results/    # final aggregate statistics and paper outputs
```

`code/` is intentionally **not** a Python package. Run internal Python entry points from this directory so the packages below are ordinary top-level packages without modifying `sys.path`:

```bash
cd code
python3 -m experiments.run_experiments --help
python3 -m analysis.26_validate_interactions --help
python3 -m poisoning.13_poisoning_grammar_checkpoint_ft --help
```

The repository-root launchers handle this working-directory change automatically:

```bash
./run_experiments.sh
./run_poisoning_experiments.sh --help
./generate_results.sh
```

Implementation directories:

- `experiments/` — experiment catalogue and execution helpers.
- `pipeline/` — numbered stages 1–7 and the per-configuration orchestrator.
- `analysis/` — exact singleton metrics, simultaneous interactions, validation, tables, and figures.
- `lib/` — shared task, model, intervention, attribution, and statistics code.
- `poisoning/` — separate poisoning experiments and scheduler jobs.
- `tests/` — model-free regression and policy tests.

Stable filesystem roots are defined in `code/lib/project_paths.py`: `CODE_ROOT` points to this directory and `PROJECT_ROOT` points to its parent. This keeps default datasets, caches, environments, and generated results at repository root even though package execution occurs here.
