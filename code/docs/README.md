# Overtopping in LLMs: Spiking-Like Causal Control and the Limits of Static Intervention — Technical Documentation

Overtopping is the high-reach regime in which replacing one internal activation channel changes a binary behavioral endpoint for a substantial fraction of examples. The codebase studies the geometry and stability of this localized causal control: directional reach, interaction under joint replacement, dose-dependent and decode-time event localization, and reorganization during learning. A controlled grammatical-acceptability poisoning experiment tests how these measurements behave when a rare trigger-dependent response is learned while ordinary task accuracy remains comparatively stable.

## Experimental frame

Every task is reduced to a fixed binary predicate `B(x)`. Candidate discovery and held-out causal evaluation use disjoint example sets. EAP-IG attribution ranks candidate components and Contrastive Hierarchical Ablation (CHA) refines the search to a frozen candidate set `J`. The same frozen identities and replacement values are then evaluated on held-out examples.

The two intervention phases are:

- **input+output**: replacement can affect prompt processing and generation;
- **output-only**: prompt processing is unmodified and replacement acts during generation.

The two principal replacement counterfactuals are:

- **mean**: the discovery-data coordinate mean;
- **mean-donor**: the observed discovery value closest to that mean.

For a held-out source-state example, the `0→1` direction records an intervention that changes `B(x)=0` to `1`; `1→0` is defined analogously. For finite-answer tasks these correspond to corrective and disruptive changes. For jailbreak evaluation they correspond to increased and decreased jailbreak success.

## Research questions

| RQ | Question | Main measurement |
|---|---|---|
| **RQ1** | [Competence, causal reach, and direction](research-questions/rq1-prevalence.md) | Direction-conditioned singleton-union reach versus unmodified competence. |
| **RQ2** | [Composition under simultaneous intervention](research-questions/rq2-composition.md) | Full-set effect `E(J)` versus singleton-union reach `U(J)` and their example-level decomposition. |
| **RQ3** | [Dose thresholds and temporal event localization](research-questions/rq3-threshold-event.md) | Persistent dose crossings, continuous margin localization, and complementary prefix/suffix temporal sweeps. |
| **RQ4** | [Learning-time causal organization](research-questions/rq4-learning.md) | Pythia checkpoint trajectories and fixed-coordinate behavior in matched clean/poisoned Grammar fine-tuning. |

The main empirical pattern is localized causal access that is direction-dependent, non-additive under joint intervention, often organized around a persistent dose threshold and narrow decode-time event, and able to move to different coordinates during learning. The controlled poisoning trajectory shows the intervention consequence: checkpoint-fixed targets lose usefulness after further learning, while checkpoint-aligned measurements can identify singleton targets with positive defense leverage under a benign-damage budget from 25% of fine-tuning onward.

## Repository model

```text
repository/
├── code/
│   ├── core/       task, model, attribution, intervention, statistics, and cache utilities
│   ├── pipeline/   numbered model-backed overtopping stages
│   ├── studies/    overtopping and poisoning study code
│   ├── reporting/  aggregate analysis orchestration
│   └── docs/       this documentation
├── data/           model-backed scientific artifacts created by execution
├── cache/          reusable computation caches created by execution
└── results/        derived analyses, audits, tables, and figures
```

Scientific populations are defined by registries, frozen example identities, and analysis manifests. Cache or directory presence does not define study membership.

## Configured overtopping registry

`studies/overtopping/experiments/run_experiments.py` defines the configured registry. The `all` selection deduplicates repeated scientific identities and resolves to 50 settings:

```text
input+output   29
output-only    21
mean-donor     39
mean           11
```

The suite selectors contain 30 `mean-donor`, 3 `6-7b-models`, 8 `mean`, and 12 `checkpoints` entries before cross-suite deduplication. The final Pythia-1B input+output Grammar, HANS-NLI, and Random-FSM settings occur in both `mean-donor` and `checkpoints`.

Inspect and validate the registry from the repository root:

```bash
bash ./run_overtopping_experiments.sh --list
bash ./run_overtopping_experiments.sh --dry-run
cd code && python -m studies.overtopping.experiments.storage_contract && cd ..
```

## Documentation map

| Area | Document |
|---|---|
| Installation, launchers, artifact roots | [Getting started](getting-started/README.md) |
| Provider credentials | [Credentials](getting-started/credentials.md) |
| Package and storage structure | [Architecture](methods/architecture.md) |
| Causal definitions and statistical units | [Core concepts](methods/concepts.md) |
| EAP and EAP-IG attribution | [EAP / EAP-IG](methods/eap.md) |
| Model-backed stages 01-08 | [Pipeline](methods/pipeline.md) |
| Overtopping registry and coverage | [Overtopping experiment](experiments/overtopping.md) |
| Controlled poisoning protocol | [Poisoning experiment](experiments/poisoning/README.md) |
| Poisoning parameters | [Poisoning configuration](experiments/poisoning/configuration.md) |
| Poisoning artifact layout | [Poisoning outputs](experiments/poisoning/outputs.md) |
| RQ1-RQ4 definitions and results | [Research questions](research-questions/README.md) |
| Aggregate analysis contracts | [Reporting](reporting/README.md) |
| Result regeneration and recovery | [Operations](operations/README.md) |
