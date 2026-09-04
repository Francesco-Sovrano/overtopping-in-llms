# RQ1 — prevalence and competence

## Question

RQ1 asks when high-leverage singleton causal effects appear across tasks, models, intervention phases, and competence levels, and whether directional causal reach or the incidence of high-effect discovered candidates changes with behavioral competence.

## Analysis populations

RQ1 uses two declared populations for different purposes.

### Figure-2 population

The manuscript prevalence figures use the explicit 39-setting overtopping catalogue:

```text
28 paper-primary settings
11 paper-auxiliary settings
```

The required phase split is:

```text
17 input+output
22 decode-only
```

The figure generator resolves these settings from the experiment catalogue rather than discovering a population by scanning the filesystem. A missing expected setting is an analysis error rather than an implicit reduction in sample size.

### Adjusted primary-matrix population

Task-, phase-, model-family-, and replacement-baseline-adjusted analyses use the 28 `paper-primary` settings. These adjusted models therefore have a different population from the 39-setting descriptive Figure-2 regressions.

The exact catalogue is documented in [Overtopping experiment catalogue](../experiments/overtopping.md).

## Directional causal quantities

Directional effects condition on the unablated source state:

```text
negative baseline: B(x)=0, eligible for 0→1 / i2c
positive baseline: B(x)=1, eligible for 1→0 / c2i
```

For candidate set `J`, RQ1 reports:

```text
U_J_i2c     directional singleton-union reach for 0→1
U_J_c2i     directional singleton-union reach for 1→0
s_1_i2c     strongest 0→1 singleton effect
s_1_c2i     strongest 1→0 singleton effect
N05_i2c     number of discovered candidates with 0→1 effect >= 0.05
N05_c2i     number of discovered candidates with 1→0 effect >= 0.05
N10_i2c     number of discovered candidates with 0→1 effect >= 0.10
N10_c2i     number of discovered candidates with 1→0 effect >= 0.10
```

Each directional rate carries its own eligible denominator. A setting with a small source-state population has greater sampling uncertainty than one with hundreds of eligible examples.

## Width-normalized candidate counts

The current reporting code also writes:

```text
N05_i2c_density
N05_c2i_density
N10_i2c_density
N10_c2i_density
```

and corresponding `*_per_1k_layer` fields.

These fields divide a discovered-candidate count by model `d_model`. They are width-normalized discovered-candidate counts. They are not estimates of the fraction of all eligible coordinates that are causal.

A coordinate-population prevalence analysis would require an explicit denominator over the searched coordinate space. MLP and attention coordinate spaces should be treated separately when their dimensions or intervention semantics differ.

## Competence

Task-specific raw performance is converted to the competence score used by the reporting pipeline. Chance normalization is task-dependent and is implemented in `studies/overtopping/analysis/lib/task_metrics.py`.

Competence is the explanatory variable in Figure 2 and in the primary-matrix correlation/adjustment analyses.

## Statistical analyses

The reporting pipeline produces:

- phase-specific Pearson regressions for the 39-setting Figure-2 population;
- primary-matrix Pearson and Spearman correlations;
- bootstrap confidence intervals over settings;
- task- and phase-specific analyses where sample size permits;
- partial/adjusted associations with task and phase controls;
- expanded adjustment including model family and replacement baseline.

The experimental setting is the cross-setting statistical unit. Directional eligible-example counts quantify within-setting precision and do not create additional independent settings.

## Implementation

Principal analysis modules:

```text
studies/overtopping/analysis/stage04_analyze_primary_metrics.py
studies/overtopping/analysis/stage06_competence_vs_overtopping_figures.py
studies/overtopping/analysis/stage06_manuscript_story_figures.py
reporting/generate_final_results.py
```

The Figure-2 population is constructed from `paper_primary_experiments()` plus `paper_auxiliary_experiments()` in `studies/overtopping/experiments/run_experiments.py`.

## Manuscript outputs

Primary Figure-2 files:

```text
results/paper/figures/02_rq1_prevalence/
├── fig2a_competence_vs_U_0to1.pdf
├── fig2b_competence_vs_U_1to0.pdf
├── fig2c_competence_vs_D05_0to1.pdf
└── fig2d_competence_vs_D05_1to0.pdf
```

Supplementary structural companions include the 10% high-effect-count analyses and effective-support comparisons.

Machine-readable sidecars are written under:

```text
results/analysis/figure_data/02_rq1_prevalence/
```

Primary-matrix statistics are written under:

```text
results/analysis/primary_matrix/statistics/
```

## Interpretation

`U_J_i2c` and `U_J_c2i` answer how much of each directional source-state population is reachable by at least one frozen singleton candidate. High-effect counts answer how many discovered candidates exceed a specified held-out effect threshold. These are distinct quantities.

Pooled `U(J)` is not the primary RQ1 competence metric because it also changes with the mixture of baseline behavioral states.

See [Interpretation and limitations](../reporting/interpretation.md) for the common metric conventions.
