# Code packages

`code/` is the Python source root.

- `core/`: shared task, model, intervention, caching, EAP, and statistical utilities.
- `pipeline/`: numbered causal-discovery and validation stages.
- `studies/overtopping/`: overtopping experiment catalogue and analyses.
- `studies/poisoning/`: poisoning training, checkpoint analysis, detector, and cross-seed aggregation.
- `reporting/`: final result-tree construction.
- `docs/`: project documentation.

Package boundaries and artifact ownership are documented in [docs/architecture.md](docs/architecture.md).
