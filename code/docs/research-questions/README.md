# Research questions

The study is organized around four research questions. Each question has a declared population, estimands, and reporting unit.

| RQ | Question | Main population | Document |
|---|---|---|---|
| RQ1 | How does directional causal leverage vary with behavioral competence? | 39-setting Figure-2 population; 28-setting primary matrix for adjusted analyses | [RQ1 — prevalence and competence](rq1-prevalence.md) |
| RQ2 | How do singleton-reachable effects change under simultaneous intervention? | 28 primary settings | [RQ2 — composition and interaction regimes](rq2-composition.md) |
| RQ3 | Do graded overtopping interventions produce support-specific threshold crossings? | 28 primary settings plus compatible supplementary RQ3 outputs | [RQ3 — support-specific thresholded causal integration](rq3-threshold-event.md) |
| RQ4 | How does causal organization change during learning beyond ordinary behavioral performance? | configured checkpoint and controlled-learning trajectories | [RQ4 — learning](rq4-learning.md) |

## Logical structure

```text
RQ1: directional causal leverage
        ↓
RQ2: composition and interaction regime
        ↓
RQ3: graded continuous-to-discrete causal crossing
        ↓
RQ4: learning-time organization
```

RQ2 and RQ3 are intentionally distinct. RQ2 compares singleton-union reach with genuine simultaneous interventions and decomposes the composition gap into preserved, suppressed, and coalition-only events. RQ3 manipulates one frozen candidate continuously and tests whether its behavioral crossing is specific to susceptible examples.

## Common conventions

- Paper-facing overtopping evaluation uses the held-out `test` split.
- Directional statistics condition on the natural source state.
- Candidate identity and discovery ranking are frozen before held-out causal evaluation.
- `U(J)` is a union of singleton flip masks, not a sum of singleton effects.
- `E(J)` is measured by a genuine simultaneous intervention on the complete candidate set.
- Setting/run-level summaries are the manuscript replication units; individual examples and candidate pairs are within-setting observations.
- Machine-readable audits and status files are part of the reporting contract.

Definitions shared across RQs are in [Core concepts](../methods/concepts.md). Exact experiment settings are in [Experiments](../experiments/).
