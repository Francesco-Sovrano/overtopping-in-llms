# Documentation

This documentation covers the executable code under `code/`, the runtime artifacts stored at repository root, the standard causal-intervention experiment programme, and the checkpointed trigger-poisoning study.

Unless a command explicitly begins at repository root, commands on these pages assume:

```bash
cd code
```

## Recommended reading order

For the standard non-poisoning workflow:

1. [Getting started](getting-started.md) — installation, validation, and first commands.
2. [Repository layout](repository-layout.md) — package ownership and runtime roots.
3. [Core concepts](concepts.md) — tasks, phases, evaluation populations, interventions, and provenance.
4. [Experiment catalogue](experiments.md) — the 28 primary and 11 auxiliary standard configurations.
5. [Numbered pipeline](pipeline.md) — stages 1–7, options, caches, and exact metric outputs.
6. [Analysis](analysis.md) — primary-matrix validation, metric audits, statistics, tables, and figures.
7. [Interpretation and limitations](interpretation-and-limitations.md) — scope of the causal claims.
8. [Troubleshooting and validation](troubleshooting.md) — checks and common failure modes.

For checkpointed poisoning studies, read:

1. [Poisoning study overview](poisoning-overview.md) — scientific questions, endpoints, markers, and workflow.
2. [Poisoning protocol](poisoning-protocol.md) — matched training, checkpoint analysis, specificity, and suppression.
3. [Poisoning configuration](poisoning-configuration.md) — entry points and configuration variables.
4. [Poisoning outputs](poisoning-outputs.md) — persistent run layout, caches, summaries, and resume policy.

The internal edge-attribution implementation is documented in [EAP / EAP-IG](eap.md).

## Main entry points

| Purpose | Entry point |
|---|---|
| List or run standard experiment configurations | `python3 -m experiments.run_experiments` |
| Run one custom causal pipeline | `bash pipeline/_run_pipeline.sh` |
| Generate final aggregate/manuscript outputs | `python3 -m analysis.generate_final_results` |
| Run interaction/set validation | `python3 -m analysis.validate_interactions` |
| Train grammar poisoning checkpoints | `python3 -m poisoning.tasks.grammar` |
| Train arithmetic poisoning checkpoints | `python3 -m poisoning.tasks.arithmetic` |
| Shared poisoning checkpoint driver | `bash poisoning/scripts/run_checkpoint_ft.sh` |
| Poisoning checkpoint causal discovery | `bash poisoning/scripts/run_backdoor_lift_overtopping.sh` |
| Poisoning cumulative suppression | `bash poisoning/scripts/run_backdoor_lift_cumulative_ablation.sh` |

From repository root, the three high-level commands are `./run_experiments.sh`, `./generate_results.sh`, and `./run_poisoning_experiments.sh`.
