# Study packages

`studies/` contains code that is specific to a scientific study rather than reusable causal infrastructure.

```text
studies/
├── overtopping/          overtopping experiment definitions and study analysis
└── poisoning/            poisoning training and checkpoint analysis
```

Both studies can call the shared `pipeline/` and `core/` packages. Code should move into a study package only when its semantics depend on that study's protocol, task construction, or outputs.

The poisoning workflow is documented in [../docs/poisoning-overview.md](../docs/poisoning-overview.md). The overtopping workflow is documented in [../docs/overtopping-experiments.md](../docs/overtopping-experiments.md).
