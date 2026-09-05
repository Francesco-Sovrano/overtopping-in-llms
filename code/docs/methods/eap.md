# EAP / EAP-IG

`core/eap/` is the repository's internal attribution-graph implementation used by circuit discovery. It targets TransformerLens-compatible autoregressive Transformer models and is imported as part of the `core` package.

## Modules

```text
graph.py          graph, node, and edge representations
attribute.py      edge attribution
attribute_node.py node/channel attribution
evaluate.py       circuit evaluation
data.py           attribution data helpers
metrics.py        attribution and evaluation metrics
utils.py          shared utilities
```

## Graph granularity

The implementation supports edge-, component-, and channel-level graph representations used by Stage 05. Attention components expose Q/K/V-oriented structure where required by the graph representation; MLP and logit components use their corresponding residual-stream interfaces.

The experiment selects its granularity through `pipeline/stage05_discover_circuits.py` and its stage-5 arguments, including `--circuit_level`.

## Attribution methods

The library implements first-order EAP-style attribution and integrated-gradient variants. Stage 05 chooses the attribution method, reference endpoint, phase, candidate circuit size, and model-specific options.

## Model assumptions

The graph implementation uses TransformerLens hook names and residual-stream semantics. Architecture-specific component shapes can require compatibility handling, especially for attention variants. Model loading and intervention compatibility are centralized in:

```text
core/modeling_and_ablation.py
core/neuron_intervention.py
core/group_intervention.py
```

## Relationship to causal intervention

EAP/EAP-IG is used for attribution and candidate discovery. The causal claims in later stages come from explicit model interventions and behavioral endpoint changes, not from attribution magnitude alone.

## Import check

From `code/`:

```bash
python -c "import core.eap; print('core.eap import OK')"
```

Dependencies are installed at repository level through `requirements.txt` and `code/studies/poisoning/requirements.txt`.
