# Documentation

This documentation describes the executable code under `code/`: how the repository is laid out, how standard experiments are selected and run, how the numbered causal-intervention pipeline works, how final results are generated, and how the checkpointed trigger-poisoning study is configured and interpreted.

Commands in these pages assume the shell is in the `code/` directory unless a page states otherwise:

```bash
cd code
```

## Recommended reading order

For a first run, read:

1. [Getting started](getting-started.md) — environment, paths, validation, and first commands.
2. [Repository layout](repository-layout.md) — packages, runtime roots, and generated artifacts.
3. [Core concepts](concepts.md) — tasks, intervention phases, evaluation splits, circuits, CHA, and provenance.
4. [Experiment catalogue](experiments.md) — the 39 standard non-poisoning configurations and CLI filters.
5. [Numbered pipeline](pipeline.md) — stages 1–7, options, caches, and statistical outputs.
6. [Analysis](analysis.md) — final-result orchestration, audits, figures, tables, and diagnostics.

For checkpointed poisoning studies, continue with:

1. [Poisoning study overview](poisoning-overview.md) — scientific questions, endpoints, markers, and quick start.
2. [Poisoning protocol](poisoning-protocol.md) — matched training, causal discovery, specificity, and seed/model aggregation.
3. [Poisoning configuration](poisoning-configuration.md) — entry points, environment variables, resuming, and protection runs.
4. [Poisoning outputs and reporting](poisoning-outputs.md) — filesystem policy, output layout, trajectory fields, and reporting checklist.
5. [Interpretation and limitations](interpretation-and-limitations.md) — what causal findings do and do not establish.
6. [Troubleshooting and validation](troubleshooting.md) — syntax checks and common failure modes.

The internal EAP/EAP-IG implementation is documented in [EAP / EAP-IG](eap.md).

## Main entry points

| Purpose | Entry point |
|---|---|
| List or run standard experiment configurations | `python3 -m experiments.run_experiments` |
| Run one custom causal pipeline | `bash pipeline/_run_pipeline.sh` |
| Generate aggregate/final results | `python3 -m analysis.generate_final_results` |
| Train checkpointed grammar poisoning runs | `python3 -m poisoning.stage01_train_grammar` |
| Train checkpointed arithmetic poisoning runs | `python3 -m poisoning.stage01_train_arithmetic` |

Use `--help` on the Python entry points and `bash pipeline/_run_pipeline.sh --help` for their current command-line surfaces.
