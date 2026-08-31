# Poisoning study package

This package owns matched clean/poisoned training, checkpoint behavior and causal analysis, developmental trajectories, circuit comparison, poisoning-example detection, interpretation, and cross-seed aggregation.

The three checkpoint endpoints are separate:

- `normal_task`: deterministic stratified no-trigger behavior sample;
- `backdoor_trigger_test`: paired trigger/control attack behavior;
- `attack_cohort_control_correctness`: causal correctness analysis on the exact attack-eligible non-target cohort.

The repository launcher is:

```bash
./run_poisoning_experiments.sh --dry-run
./run_poisoning_experiments.sh
```

Canonical documentation:

- [Poisoning protocol](../../docs/poisoning-protocol.md)
- [Poisoning configuration](../../docs/poisoning-configuration.md)
- [Poisoning outputs](../../docs/poisoning-outputs.md)
- [Interpretation and limitations](../../docs/interpretation-and-limitations.md)
