# Overtopping in LLMs: Spiking-Like Causal Control and the Limits of Static Intervention

Overtopping is a causal-intervention regime in which replacing one internal activation channel changes a binary behavioral endpoint across many examples. This repository measures where such high-leverage channels appear, how their singleton effects interact under simultaneous intervention, whether their causal response is localized by intervention dose and decoding time, and whether the coordinates carrying leverage remain stable as a model continues to learn.

The experiments cover arithmetic exact match, grammatical acceptability, natural-language inference, random finite-state-machine prediction, and jailbreak success. The learning-time study also introduces a rare trigger-dependent behavior through controlled Grammar poisoning. The resulting measurements characterize overtopping as localized causal control that can be threshold-like, non-additive across channels, and able to reorganize during learning. The poisoning experiment tests the practical consequence: a channel that suppresses the trigger-dependent behavior at one checkpoint need not remain a useful target after further training, while checkpoint-aligned causal measurements can identify useful singleton targets under a benign-damage budget.

## Scientific scope

The overtopping analysis is organized around four questions:

| Question | Measurement |
|---|---|
| **RQ1 — Competence, causal reach, and direction** | Direction-conditioned singleton-union reach versus unmodified task competence. |
| **RQ2 — Composition under simultaneous intervention** | Singleton-union reach versus genuine simultaneous-set intervention. |
| **RQ3 — Dose thresholds and temporal event localization** | Graded replacement strength, persistent behavioral crossings, continuous logit-margin changes, and prefix/suffix decode-time ablations. |
| **RQ4 — Learning-time causal organization** | Pythia checkpoint trajectories and matched clean/poisoned Grammar fine-tuning trajectories. |

## Principal findings

| Analysis | Result |
|---|---|
| RQ1 | Directional singleton-union reach is positively associated with competence in the full configured analysis. For output-only `1→0`, the association is `r=0.676`, `p=0.00106`, `n=20`; the reduced sensitivity analysis reports `r=0.727`, `p=0.00141`, `n=16`. |
| RQ2 | Joint replacement is usually weaker than singleton-union reach: `22/25` evaluable mean-donor sets and `10/11` mean sets have negative composition gaps, with medians `-0.146` and `-0.166`. |
| RQ3 | Known-flip trajectories have median single-persistent-crossing rate `0.989`; the median channel reaches its first persistent transition at dose `lambda=0.50`. Complementary prefix/suffix sweeps localize the same peak transition in 72% of paired conditions and within one transition in 83%. |
| RQ4 | In the controlled Grammar trajectory, trigger-conditioned conversion reaches median `0.944` at 25% of fine-tuning and `1.000` by 50% while ordinary Grammar accuracy remains close between clean and poisoned models. Previous-checkpoint targets have negative median defense leverage at every subsequent checkpoint, whereas checkpoint-aligned clean-reference selection is positive from 25% onward under `tau=0.30`. |

The measured causal quantities depend on the activation basis, replacement counterfactual, intervention phase, and binary endpoint. Task, model, phase, and replacement axes are not fully factorially crossed. The term *spiking-like* refers to persistent threshold/event geometry under intervention, not to a claim that Transformer channels implement biological spiking-neuron dynamics. The controlled Grammar defense analysis measures suppression after the trigger-dependent behavior has been learned; it does not test prevention during fine-tuning.

A **channel** is the scalar coordinate or finite activation component replaced by an intervention. A **singleton intervention** replaces one channel. A **simultaneous-set intervention** replaces every channel in a selected set in the same forward pass. The main replacement baselines are:

- `mean`: replace a channel with its discovery-data mean;
- `mean-donor`: replace it with the observed discovery value closest to that mean.

Two intervention phases are used:

- **input+output** (`standard` in the execution configuration): replacement is active during prompt processing and answer generation;
- **output-only** (`decode-only`): prompt processing is left intact and replacement is active during generation.

Candidate discovery and held-out causal evaluation use disjoint examples. EAP-IG attribution ranks internal components and Contrastive Hierarchical Ablation (CHA) refines the candidate set. Candidate identities, rankings, replacement values, and discovery direction are fixed before held-out singleton evaluation.

## Configured overtopping registry

The configured `all` registry resolves to 50 unique scientific settings:

| Dimension | Value | Count |
|---|---|---:|
| intervention phase | input+output | 29 |
| intervention phase | output-only | 21 |
| replacement | `mean-donor` | 39 |
| replacement | `mean` | 11 |

The registry includes Qwen2-1.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen2-7B-Instruct, and Pythia-1B model states. Pythia-1B learning trajectories use step 0, 48k, 96k, and the final checkpoint for Grammar, HANS-NLI, and Random FSM.

Inspect the resolved settings without model inference:

```bash
bash ./run_overtopping_experiments.sh --list
bash ./run_overtopping_experiments.sh --dry-run
```

Validate scientific identities and persistent output addresses:

```bash
cd code
python -m studies.overtopping.experiments.storage_contract
cd ..
```

## Repository layout

```text
code/       implementation and technical documentation
  core/     shared task, model, attribution, intervention, and statistics code
  pipeline/ numbered model-backed causal-intervention stages
  studies/  overtopping and poisoning study logic
  reporting/ aggregate analysis and output generation
  docs/     scientific and operational documentation

data/       model-backed experiment artifacts created by execution
cache/      reusable computation caches created by execution
results/    derived analyses, audits, tables, and figures created by reporting
```

The source archive may not contain `data/`, `cache/`, or `results/`; the launchers create them as needed.

## Requirements and installation

The project targets Python 3.12. Model-backed execution requires access to the configured model weights. Some prompt-generation or classifier paths also require Ollama, Groq, OpenAI, or Hugging Face credentials.

From the repository root:

```bash
./setup.sh
```

Equivalent manual setup:

```bash
python3.12 -m venv .env
. .env/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r code/studies/poisoning/requirements.txt
```

Provider configuration is described in [`code/docs/getting-started/credentials.md`](code/docs/getting-started/credentials.md).

## Run the overtopping study

Run all configured settings on the default held-out `test` split:

```bash
bash ./run_overtopping_experiments.sh
```

Common filters and execution modes:

```bash
bash ./run_overtopping_experiments.sh --suite checkpoints --dry-run
bash ./run_overtopping_experiments.sh --task arithmetic --evaluation-split test --dry-run
bash ./run_overtopping_experiments.sh --phase pipeline
bash ./run_overtopping_experiments.sh --phase analysis
```

The model-backed pipeline is:

```text
01  prompts and unmodified answers
02  feature export
03  rule extraction
04  spectral sampling plan
05  circuit discovery
06  candidate and rule analysis
07  held-out singleton causal evaluation
07b graded intervention and continuous-margin diagnostics
07c temporal prefix intervention
07d temporal suffix intervention
    threshold-event diagnostics
08  simultaneous-set and interaction validation
```

Stage inputs, output paths, and environment controls are specified in [`code/docs/methods/pipeline.md`](code/docs/methods/pipeline.md).

## Run the controlled Grammar poisoning study

The default launcher trains three matched clean/poisoned runs of `Qwen/Qwen2-1.5B-Instruct` with training seeds 13, 37, and 101 and a shared holdout seed of 13. The configured Grammar study poisons 10% of eligible gold non-target examples, uses `[id=74291]` as the trigger marker, targets the `acceptable` label, and evaluates checkpoints at 0%, 10%, 25%, 50%, 75%, and 100% of one fine-tuning epoch.

Inspect the execution plan:

```bash
./run_poisoning_experiments.sh --dry-run
```

Run the configured study:

```bash
./run_poisoning_experiments.sh
```

Use the reduced behavior-focused execution path to check the training and trigger pipeline before a full run:

```bash
POISONING_FAST_TEST=1 ./run_poisoning_experiments.sh
```

The poisoning protocol, training configuration, candidate-localization rules, defense metrics, and persistent outputs are documented under [`code/docs/experiments/poisoning/`](code/docs/experiments/poisoning/README.md).

## Generate analyses and figures

Generate derived outputs from existing scientific artifacts:

```bash
./generate_results.sh
```

The reporting driver reads overtopping measurements from `data/`, poisoning measurements from `data/poisoning/`, and writes derived products under `results/`.

Require every configured metric in the completeness contract:

```bash
ALLOW_INCOMPLETE_METRICS=false ./generate_results.sh
```

Supply an explicit RQ3 threshold-diagnostics source when required:

```bash
./generate_results.sh --spiking-source /absolute/path/to/threshold_diagnostics
```

Machine-readable analyses are written under `results/analysis/`. Rendered outputs are written under `results/paper/`.

## Population and missing-data rules

The configured experiment manifest is established before metric-specific filtering.

| Analysis | Population rule |
|---|---|
| configured overtopping table | every unique setting in the selected registry manifest |
| RQ1 | settings with the required directional singleton metrics, analyzed separately by intervention phase |
| RQ2 | nonempty frozen candidate sets with compatible simultaneous-set outputs, analyzed separately by replacement regime |
| RQ3 | settings with the artifacts required by the relevant dose, margin, threshold, or temporal analysis |
| RQ4 | configured Pythia checkpoint trajectories and controlled Grammar poisoning trajectories |

A completed zero-candidate setting remains a measured zero where the metric is defined. Missing or inapplicable artifacts remain missing or inapplicable rather than being converted to zero.

## Documentation

Start with [`code/docs/README.md`](code/docs/README.md). It links the scientific definitions, task and intervention conventions, pipeline stages, overtopping registry, controlled poisoning protocol, RQ1-RQ4 analyses, reporting outputs, and regeneration commands.
