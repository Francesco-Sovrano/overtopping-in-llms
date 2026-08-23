# EAP / EAP-IG

The repository includes an internal EAP/EAP-IG implementation under `lib/eap/`. This page describes its modules, graph granularities, assumptions, interventions, and import surface.

`lib/eap/` is the repository's internal attribution and computational-graph library used by pipeline stage 5. It targets TransformerLens-compatible autoregressive transformer models and is imported as part of the `lib` package; it is not an independently packaged application.

## Modules

```text
graph.py          graph, node, and edge representations
attribute.py      edge attribution methods
attribute_node.py node/neuron attribution methods
evaluate.py       circuit evaluation
data.py           attribution data helpers
metrics.py        attribution/evaluation metrics
utils.py          shared helpers
visualization.py  graph visualization helpers
```

## Graph granularities

The implementation provides graph structures used for edge-, node-, and neuron/channel-level analyses. Attention components expose Q/K/V-oriented structure where required by the graph representation; MLP and logits components use their corresponding residual-stream inputs.

The exact granularity used by an experiment is selected by `pipeline/5_discover_circuits.py` through the pipeline's `--circuit_level` and related stage-5 options. Do not infer a run's granularity from this library alone.

## Attribution methods

The library contains first-order EAP-style attribution and integrated-gradient variants used by stage 5. The stage-5 runner is responsible for choosing the attribution method, clean/corrupted or reference endpoints, model phase, and candidate circuit size.

## Model assumptions

The implementation is designed around TransformerLens model hooks and residual-stream semantics. Architecture-specific behavior can matter, particularly for grouped-query attention or models whose internal component shapes do not match the graph assumptions. Model loading and compatibility handling are centralized in `lib/modeling_and_ablation.py` and the stage-5 runner.

## Interventions and baselines

EAP attribution identifies candidate components; model interventions and replacement baselines are implemented in shared repository code rather than in this directory. The canonical intervention utilities are in:

```text
lib/modeling_and_ablation.py
lib/neuron_intervention.py
lib/group_intervention.py
```

Run-specific intervention names, phases, donor/reference construction, and evaluation splits are defined by the pipeline and experiment configuration.

## Running and importing

Run repository modules from `code/` so `lib.eap` resolves as a normal top-level package:

```bash
cd code
python3 -c "import lib.eap; print('lib.eap import OK')"
```

Dependency installation is repository-level. This subdirectory contains no standalone requirements or setup script, and the supplied `code/` subtree does not contain a root installation script.
