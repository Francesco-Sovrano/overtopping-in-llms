# Overtopping experiment design

## Unit of configuration

An overtopping **setting** is a specific combination of:

```text
task × model snapshot × intervention phase × replacement baseline
```

The study registry contains 56 unique settings. It is explicit rather than a complete Cartesian product.

The 56 unique settings decompose as:

```text
31 unique final-snapshot task × model × phase cells
18 non-final Pythia checkpoint settings (step0, 48k, 96k; Grammar/HANS-NLI/FSM)
 7 replacement-baseline repeats on already represented final-snapshot cells
--------------------------------------------------------------------------------
56 configured settings
```

The longitudinal Pythia checkpoint **view** contains 24 settings: those 18
non-final settings plus the six final/all-steps `EleutherAI/pythia-1b`
Grammar/HANS-NLI/FSM endpoints already counted among the 31 final-snapshot cells. The view therefore overlaps the
final-snapshot view intentionally, while the canonical registry stores each
RunSpec only once.

The registry is defined in:

```text
code/studies/overtopping/experiments/run_experiments.py
```

## Final-snapshot design: 31 cells

There are five final model snapshots, five tasks, and two intervention phases, so a complete factorial grid would contain 50 task×model×phase cells. The study configures 31 of those cells.

The design is targeted rather than factorial:

- **Qwen2.5-1.5B** supplies complete task-by-phase coverage across all five tasks.
- **Qwen2-1.5B** supplies a matched small-model family comparison with both phases for arithmetic, grammar, NLI, and Random FSM, plus output-only jailbreak.
- **Qwen2-7B** supplies selected within-family scale comparisons for arithmetic, jailbreak, and NLI using the large-model MLP-only intervention basis. The 7B grid is not expanded further because model-backed activation interventions are substantially more expensive at this scale.
- **Pythia-1B** supplies both intervention phases for arithmetic, grammar, HANS-NLI, and Random FSM at the final/all-steps snapshot. Longitudinal checkpoint trajectories are configured for Grammar, HANS-NLI, and Random FSM; Arithmetic is intentionally not checkpointed.
- **Pythia-6.9B** supplies the Pythia size comparison for output-only arithmetic.

The configured final-snapshot cells are:

| Model snapshot | Arithmetic | Jailbreak | Grammar | NLI | Random FSM | Cell count |
|---|---|---|---|---|---|---:|
| Qwen2.5-1.5B | I+O, Out | I+O, Out | I+O, Out | I+O, Out | I+O, Out | 10 |
| Qwen2-1.5B | I+O, Out | Out | I+O, Out | I+O, Out | I+O, Out | 9 |
| Qwen2-7B | Out | Out | — | I+O | — | 3 |
| Pythia-1B | I+O, Out | — | I+O, Out | I+O, Out | I+O, Out | 8 |
| Pythia-6.9B | Out | — | — | — | — | 1 |
| **Total** |  |  |  |  |  | **31** |

An absent cell is outside the configured study. It is distinct from a configured setting whose required output artifact is missing.

## Pythia checkpoint design: 24-setting longitudinal view

For Grammar, HANS-NLI, and Random FSM, the Pythia-1B trajectory is:

```text
step0 -> step48k -> step96k -> final/all-steps pythia-1b
```

Both intervention phases are configured at every point. This gives 24 settings
in the longitudinal view:

```text
4 checkpoints × 3 tasks × 2 phases = 24
```

Only 18 of these are additional non-final registry entries because the six
final/all-steps Grammar/HANS-NLI/FSM endpoints are already final-snapshot cells. All
checkpoint trajectory runs use mean-donor replacement and the standard
small-model defaults. Historical Arithmetic checkpoint artifacts remain
addressable for reproducibility but are outside the configured paper study and
are never selected by the checkpoint suite.

A completed checkpoint configuration that yields no candidate channels remains
an explicit zero-candidate observation rather than being dropped. A checkpoint
with only dataset-level competence available but no causal intervention output
is **missing causal coverage**, not `U(J)=0`.

## Replacement-baseline repeats: 7 settings

Seven final-snapshot cells are evaluated under both mean-donor and mean-family replacement. These repeats test dependence on the replacement counterfactual without creating a second task/model/phase cell.

| Task | Model | Phase | Main paired replacements |
|---|---|---|---|
| Arithmetic | Qwen2-1.5B | Out | mean-donor, mean |
| Arithmetic | Pythia-1B | Out | mean-donor, mean |
| Grammar | Qwen2.5-1.5B | I+O | mean-donor, mean |
| Grammar | Qwen2.5-1.5B | Out | mean-donor, mean |
| NLI | Qwen2.5-1.5B | I+O | mean-donor, mean |
| NLI | Qwen2.5-1.5B | Out | mean-donor, mean |
| Random FSM | Qwen2.5-1.5B | I+O | mean-donor, mean |

Across all 56 settings, the replacement counts are:

```text
44 mean-donor
 8 mean
 4 mean-positional
```

For RQ2 reporting, `mean-positional` is grouped with `mean` as the mean-replacement regime. Mean-donor and mean-replacement regimes are analyzed separately.

## Phase totals

Across the complete 56-setting registry:

```text
26 input+output (I+O)
30 output-only (Out)
```

The two phases are analyzed separately in RQ1 because their competence scores have different interpretations.

## Discovery/intervention configuration

Small-model settings use the standard neuron search configuration unless specified in the registry:

```text
circuit_size = 200000
min_flip_rate = 0.3
```

Large-model scale settings use:

```text
MLP-only coordinates
circuit_size = 100000
min_flip_rate = 0.2
replacement = mean-positional
```

Task-specific `z_thresh`, batch size, and maximum-circuit values are encoded directly in each `RunSpec`.

## Evaluation split

The manuscript-facing overtopping analyses use the held-out `test` split. The runner also accepts `train` and `all` for explicit non-manuscript analyses.

Evaluation suffixes are generated by `RunSpec.evaluation_suffix()`:

```text
test   -> -heldout_test
train  -> -eval_train
all    -> no evaluation suffix
```

## Analysis populations

The complete 56-setting registry is materialized before metric-specific filtering.

- **RQ1:** all 56 settings; correlations are fit separately for I+O and Out.
- **RQ1 final-snapshot sensitivity:** 29 unique final task×model×phase cells; intermediate checkpoints are removed and one replacement condition is retained per repeated cell.
- **RQ2:** settings for which a nonempty candidate set and simultaneous-set outputs make composition applicable; replacement regimes are analyzed separately.
- **RQ3:** all 56 settings are in the manifest; each threshold/graded analysis retains settings with its required compatible artifacts and reports the resulting denominator.
- **RQ4:** configured Pythia checkpoint trajectories and controlled poisoning trajectories.

A completed zero-candidate setting remains a configured observation for analyses such as RQ1 where `U(J)=0` is defined. Joint composition of an empty set is not treated as an ordinary RQ2 composition observation.

## Execution sets

The runner exposes exactly four disjoint execution sets:

```bash
./run_overtopping_experiments.sh --suite mean-donor
./run_overtopping_experiments.sh --suite 6-7b-models
./run_overtopping_experiments.sh --suite mean
./run_overtopping_experiments.sh --suite checkpoints
```

The checkpoint set contains Grammar, HANS-NLI, and Random FSM at Pythia-1B step0, step48000, step96000, and final/all-steps `EleutherAI/pythia-1b`, in both phases. Arithmetic is not checkpointed. Suite membership is execution metadata only and does not alter persistent result/cache addresses.

## Inspect the registry

From the repository root:

```bash
./run_overtopping_experiments.sh --list
./run_overtopping_experiments.sh --dry-run
```

Target a subset by scientific fields:

```bash
./run_overtopping_experiments.sh \
  --task arithmetic \
  --model Qwen/Qwen2-1.5B-Instruct \
  --evaluation-split test \
  --dry-run
```

Available filters include:

```text
--task
--model
--intervention
--mode
--phase
--evaluation-split
```

## Persistent paths

Each `RunSpec` resolves persistent experiment locations through the shared path constructors:

```text
circuit_label()
bag_label()
evaluation_suffix()
input_data_dir(data_root)
stats_dir(data_root)
```

Analysis code uses these constructors rather than defining the study population by recursive filesystem discovery.

`mean` and `mean-positional` use the unsuffixed mean-family circuit path convention; their exact intervention is carried by the registry and pipeline arguments. Mean-donor paths include the donor intervention suffix.

Validate the full registry's storage/addressing contract with:

```bash
cd code
python -m studies.overtopping.experiments.storage_contract
```

## Runner outputs

The runner records the selected configurations under:

```text
results/configured_experiments.json
```

Pipeline failures are recorded under:

```text
results/pipeline_failures.json
```

These are reporting/control products. Model-backed experiment outputs remain under `data/`.
