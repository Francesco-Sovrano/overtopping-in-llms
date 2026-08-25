# Study packages

`studies/` contains code whose scientific meaning belongs to one experimental programme rather than to the reusable causal-intervention engine.

```text
studies/
├── overtopping/
│   ├── experiments/    explicit experiment catalogue and execution
│   └── analysis/       overtopping-specific analysis and figures
└── poisoning/          poisoning training, checkpoint analysis, and defence
```

Both study packages may use `core/` and `pipeline/`. They should not duplicate the shared pipeline or treat the other study as an implementation dependency.

Cross-study manuscript/output orchestration lives in `reporting/`.
