# Documentation

This directory is the documentation entry point for the repository.

## Start here

For a first reading, use this order:

1. [Getting started](getting-started/) — install the environment, understand the repository roots, inspect the experiment catalogues, run experiments, and generate results.
2. [Methods](methods/) — learn the intervention model, directional causal quantities, discovery/evaluation split, pipeline stages, and attribution machinery.
3. [Experiments](experiments/) — see the exact overtopping catalogue and the poisoning study design.
4. [Research questions](research-questions/) — read the scientific analyses for RQ1, RQ2, RQ3, and RQ4 in parallel form.
5. [Reporting](reporting/) — understand how completed experiments become tables, figures, audits, and manuscript-facing outputs.
6. [Operations and troubleshooting](operations/) — recover incomplete analyses, inspect missing populations, and decide which derived files can be regenerated without deleting reusable caches.

## Documentation map

```text
code/docs/
├── README.md                         documentation entry point
├── getting-started/
│   ├── README.md                     installation and first run
│   └── credentials.md                provider credentials and secrets
├── methods/
│   ├── README.md                     methods map
│   ├── architecture.md               package and artifact ownership
│   ├── concepts.md                   causal definitions and metrics
│   ├── pipeline.md                   numbered causal pipeline
│   └── eap.md                        EAP / EAP-IG implementation
├── experiments/
│   ├── README.md                     experiment-family map
│   ├── overtopping.md                exact overtopping catalogue
│   └── poisoning/
│       ├── README.md                 poisoning protocol
│       ├── configuration.md          launcher configuration
│       └── outputs.md                poisoning artifact tree
├── research-questions/
│   ├── README.md                     RQ map and populations
│   ├── rq1-prevalence.md             prevalence and competence
│   ├── rq2-composition.md            joint composition and specificity
│   ├── rq3-threshold-event.md        causal threshold-event analysis
│   └── rq4-learning.md               learning trajectories and poisoning
├── reporting/
│   ├── README.md                     reporting entry point
│   ├── analysis-pipeline.md          analysis stages and result tree
│   ├── figures.md                    manuscript figure ownership
│   └── interpretation.md             metric scope and limitations
└── operations/
    └── README.md                     troubleshooting and regeneration
```

## Choose a path by task

| Goal | Read |
|---|---|
| Install and run the repository | [Getting started](getting-started/) |
| Understand what `data/`, `cache/`, and `results/` contain | [Architecture](methods/architecture.md) |
| Understand `U(J)`, directional reach, `E(J)`, graded agonist crossings, or preemption | [Core concepts](methods/concepts.md) |
| Understand the execution stages | [Numbered pipeline](methods/pipeline.md) |
| See which overtopping experiments are part of the paper | [Overtopping catalogue](experiments/overtopping.md) |
| Understand RQ1 | [RQ1 — prevalence](research-questions/rq1-prevalence.md) |
| Understand RQ2 | [RQ2 — composition](research-questions/rq2-composition.md) |
| Understand RQ3 | [RQ3 — threshold events](research-questions/rq3-threshold-event.md) |
| Understand RQ4 | [RQ4 — learning](research-questions/rq4-learning.md) |
| Understand poisoning experiments | [Poisoning protocol](experiments/poisoning/) |
| Trace a manuscript figure to its source data | [Figure map](reporting/figures.md) |
| Interpret a metric or limitation | [Interpretation](reporting/interpretation.md) |
| Recover or regenerate derived outputs | [Operations](operations/) |
