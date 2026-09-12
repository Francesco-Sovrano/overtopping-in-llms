# RQ1 — prevalence and competence

## Question

RQ1 asks whether high-leverage singleton causal effects occur across the configured tasks and model states, and whether corrective directional reach changes with behavioral competence.

## Population

The main RQ1 analysis uses all 56 configured overtopping settings:

```text
29 unique final-snapshot task × model × phase cells
18 non-final Grammar/HANS-NLI/FSM Pythia checkpoint settings
 7 matched replacement-baseline repeats
--------------------------------------------
56 settings
```

The phase totals are:

```text
26 input+output (I+O)
30 output-only (Out)
```

All configured settings enter because the estimand is the association between behavioral competence and causal reach across the intervention settings actually studied. Intermediate checkpoints are distinct model states, and replacement-baseline repeats are distinct causal counterfactuals. They are not pooled across phase.

A separate final-snapshot sensitivity retains the 29 unique task×model×phase cells, removes intermediate checkpoints, and uses one replacement condition for each repeated cell, preferring mean-donor when both regimes are present. This analysis asks whether the RQ1 association depends on repeated checkpoints or matched replacement-baseline conditions.

## Directional causal quantities

Let `B(x)` be the unmodified binary behavioral endpoint. Directional effects condition on the source state:

```text
B(x)=0  -> eligible for 0→1 / i2c
B(x)=1  -> eligible for 1→0 / c2i
```

For frozen candidate set `J`:

```text
U_J_i2c     singleton-union reach on B=0 examples
U_J_c2i     singleton-union reach on B=1 examples
s_1_i2c     strongest 0→1 singleton effect
s_1_c2i     strongest 1→0 singleton effect
N05_i2c     discovered candidates with 0→1 effect >= .05
N05_c2i     discovered candidates with 1→0 effect >= .05
N10_i2c     discovered candidates with 0→1 effect >= .10
N10_c2i     discovered candidates with 1→0 effect >= .10
```

Each directional rate carries its own eligible-example denominator.

A completed setting with no discovered candidates remains in RQ1 with zero singleton-union reach where the metric is defined.

## Competence

RQ1 uses a phase-specific competence score because the two intervention phases exclude different parts of the computation.

### Input+output phase

Input+output interventions act while the prompt is processed and while the answer is generated. Instruction interpretation and downstream task execution are therefore both inside the intervention window. The competence variable is the model's raw unmodified task score:

```text
competence_I+O = raw task success rate
```

### Output-only phase

Output-only interventions begin after prompt/instruction processing. The model can therefore have parsed the instruction correctly while still lacking the task knowledge or computation needed to answer a finite-choice instance. In that case some observed success can come from guessing. Output-only competence removes the random-answer baseline:

```text
competence_Out = (raw_score - chance) / (1 - chance)
```

bounded to `[0,1]` by the implementation.

Task-specific chance baselines are defined in:

```text
studies/overtopping/analysis/lib/task_metrics.py
```

Grammar and HANS NLI use `0.5`; Random FSM uses the configured average random-choice baseline unless an explicit empirical baseline is requested; arithmetic and jailbreak use the task-specific definitions implemented there. Jailbreak scores are oriented so higher values mean safer/refusal behavior.

The I+O and Out competence scales are analyzed in separate panels and are not pooled into one correlation.

## Width-normalized candidate counts

Reporting also writes width-normalized discovered-candidate counts:

```text
N05_i2c_density
N05_c2i_density
N10_i2c_density
N10_c2i_density
```

and corresponding `*_per_1k_layer` fields.

These divide discovered-candidate counts by model `d_model`. They are not estimates of the fraction of all searchable coordinates that are causal.

## Statistical analyses

The main RQ1 outputs include:

- phase-specific Pearson associations across all 56 configured settings;
- phase-specific Spearman associations across all 56 configured settings;
- the 31-cell final-snapshot sensitivity;
- task/phase and structural diagnostics where the available setting count permits estimation.

The cross-setting statistical unit is the configured setting. Eligible-example counts describe within-setting precision rather than additional independent replicates.

## Implementation

Principal modules:

```text
studies/overtopping/analysis/stage02_overtopping_latex_tables.py
studies/overtopping/analysis/stage04_analyze_primary_metrics.py
studies/overtopping/analysis/stage06_competence_vs_overtopping_figures.py
reporting/generate_final_results.py
```

The complete registry is constructed by:

```text
paper_study_experiments()
```

in `studies/overtopping/experiments/run_experiments.py`.

## Outputs

Main figure directory:

```text
results/paper/figures/02_rq1_prevalence/
```

Machine-readable figure inputs and fit statistics:

```text
results/analysis/figure_data/02_rq1_prevalence/
```

Configured-study tables and cross-setting statistics:

```text
results/analysis/primary_matrix/
```

See [Overtopping experiment design](../experiments/overtopping.md) for the exact 56 settings.
