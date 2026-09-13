# Causal pipeline

This package contains the numbered model-backed execution stages. The normal overtopping sequence is:

```text
01 prompts/answers -> 02 features -> 03 rules -> 04 sampling -> 05 discovery
-> 06 candidate analysis -> 07 singleton evaluation -> 07b graded intervention -> 07c temporal prefix -> 07d temporal suffix
-> threshold diagnostics -> 08 interaction validation
```

`run_pipeline.sh` coordinates these stages for one scientific configuration. Persistent experiment outputs are written under `data/`; reusable model-evaluation caches are separate from derived reporting outputs.

References:

- [Pipeline stages](../docs/methods/pipeline.md)
- [Core concepts](../docs/methods/concepts.md)
- [Architecture](../docs/methods/architecture.md)
- [Operations](../docs/operations/README.md)
