# RQ1 — Competence, causal reach, and direction

## Question

RQ1 asks where high-reach singleton causal control appears and whether directional reach changes with the unmodified model's task competence.

## Population

The configured overtopping manifest contains 50 unique settings: 29 input+output and 21 output-only. The two phases are analyzed separately because their intervention windows differ and because the competence normalization is phase-specific.

Zero-candidate conditions remain in the analysis with zero reach when the directional metric is defined. A direction is excluded only when its required source-state cohort is not available for causal evaluation.

The primary cross-setting analysis includes the configured checkpoint and replacement-baseline conditions. A sensitivity analysis removes intermediate Pythia checkpoints and retains one preferred replacement condition for repeated task-model-phase cells. This leaves 33 defined `0→1` cells and 31 defined `1→0` cells.

## Directional causal quantities

Let `B(x)` be the unmodified binary behavioral endpoint. Directional evaluation conditions on the source state:

```text
B(x)=0  -> eligible for 0→1 / i2c
B(x)=1  -> eligible for 1→0 / c2i
```

For frozen candidate set `J`, let `F_j^d(x)=1` when replacing candidate `j` changes the endpoint in direction `d`.

```text
U^d(J) = P(any j in J has F_j^d(x)=1)
s_1^d  = max_j P(F_j^d(x)=1)
N_.05^d = number of discovered candidates with directional singleton reach >= 0.05
```

`U^d(J)` is singleton-union reach. It measures the share of the eligible held-out population reachable by at least one discovered singleton. It is not itself a concentration measure; `s_1^d` and the thresholded candidate counts describe how much of that reach is carried by individual channels.

Each directional rate uses its own eligible-example denominator.

## Competence

Competence is measured on the unmodified model.

### Input+output

Prompt processing and answer generation both lie inside the intervention window, so the analysis uses the raw higher-is-better task score:

```text
competence_I+O = raw task score
```

### Output-only

Prompt and instruction processing occur before the intervention. For finite-output tasks, some success may therefore be attributable to guessing even when downstream task competence is low. The analysis uses chance-normalized competence:

```text
kappa = max(0, (s - c) / (1 - c))
```

where `s` is the raw higher-is-better task score and `c` is the parsed-output chance rate. Grammar and HANS NLI use `c=0.5`; arithmetic uses `c=0`; Random FSM uses the sampled output-domain chance rate. Safety is oriented as a higher-is-better safe-rate quantity in the reporting code.

The input+output and output-only competence scales are not pooled.

## Main results

For `0→1` reach, the primary linear association with competence is:

```text
input+output: n=29, r=0.506, p=0.00515
output-only:  n=21, r=0.341, p=0.130
```

For `1→0` reach:

```text
input+output: n=27, r=0.507, p=0.00692
output-only:  n=20, r=0.676, p=0.00106
```

The reduced sensitivity analysis remains positive in all four phase/direction combinations, but the clearest retained association is output-only `1→0`:

```text
0→1, input+output: n=16, Pearson r=0.231, p=0.389
0→1, output-only:  n=17, Pearson r=0.407, p=0.105
1→0, input+output: n=15, Pearson r=0.245, p=0.379
1→0, output-only:  n=16, Pearson r=0.727, p=0.00141
```

These associations describe the configured condition matrix. They do not identify model size, task family, intervention phase, or replacement baseline as independent causal factors because those axes are not fully factorially crossed.

## Width-normalized candidate counts

Reporting also writes discovered-candidate counts normalized by model `d_model`, including:

```text
N05_i2c_density
N05_c2i_density
N10_i2c_density
N10_c2i_density
N05_i2c_per_1k_layer
N05_c2i_per_1k_layer
```

These are normalizations of the discovered candidate set. They are not estimates of the fraction of all model coordinates that are causal.

## Statistical unit

The cross-setting unit is the configured setting. Eligible-example counts describe within-setting precision and are not treated as independent cross-setting replicates.

## Implementation

Principal modules:

```text
studies/overtopping/analysis/stage02_overtopping_latex_tables.py
studies/overtopping/analysis/stage04_analyze_primary_metrics.py
studies/overtopping/analysis/stage06_competence_vs_overtopping_figures.py
reporting/generate_final_results.py
```

The configured registry is constructed by `paper_study_experiments()` in:

```text
studies/overtopping/experiments/run_experiments.py
```

## Outputs

Rendered outputs:

```text
results/paper/figures/02_rq1_prevalence/
```

Machine-readable figure data and fit statistics:

```text
results/analysis/figure_data/02_rq1_prevalence/
results/analysis/primary_matrix/
```

The complete 50-setting registry and task/model/phase coverage are documented in [Overtopping experimental frame and registry](../experiments/overtopping.md).
