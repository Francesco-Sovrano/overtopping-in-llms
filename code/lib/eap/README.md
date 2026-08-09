# EAP / EAP-IG implementation

This directory contains the attribution and graph utilities used by pipeline stage 5 to discover candidate circuits in TransformerLens-compatible autoregressive language models.

It is an internal library of this repository; install the repository environment with root `setup.sh` rather than installing this subdirectory independently.

## Computational graph levels

The graph implementation supports representations at several granularities:

- **edge** — directed residual-stream connections between components;
- **node** — attention heads, MLPs, inputs and logits without individual edge selection;
- **neuron/channel** — individual component output dimensions where supported by the stage-5 pipeline.

Attention heads expose Q/K/V input structure; MLPs and logits use their corresponding component inputs.

## Attribution methods

The code implements attribution-patching variants used by stage 5, including:

- EAP-style first-order attribution;
- integrated-gradient variants over inputs or activations;
- clean/corrupted endpoint variants.

The pipeline selects the specific method and intervention/reference data through `pipeline/5_discover_circuits.py`.

## Main modules

```text
graph.py          graph/node/edge representation
attribute.py      edge attribution methods
attribute_node.py node/neuron attribution methods
evaluate.py       circuit evaluation
metrics.py        attribution/evaluation metrics
data.py           EAP data helpers
utils.py          shared utilities
visualization.py  graph rendering helpers
```

## Model compatibility

The implementation is designed around TransformerLens-style autoregressive transformer models. Architecture-specific handling may be needed for models whose residual stream or grouped-query-attention behavior differs from the assumptions of the graph representation.

For grouped-query attention models, TransformerLens may need ungrouping enabled before graph construction depending on the selected model/configuration.

## Replacement interventions

Stage-5 attribution/evaluation can use the repository's supported replacement baselines where the selected attribution method permits them. The canonical model/intervention implementation is in:

```text
code/lib/modeling_and_ablation.py
```

The experiment pipeline, rather than this README, is the source of run-specific baseline, phase, and reference-data configuration.
