# Research questions

The overtopping paper is organized around four research questions. Each RQ has its own analysis population, metrics, statistical unit, and manuscript outputs.

| RQ | Question | Main population | Document |
|---|---|---|---|
| RQ1 | When does high-leverage directional causal reach appear, and how does it relate to competence? | 39-setting Figure-2 population; 28-setting primary matrix for adjusted analyses | [RQ1 — prevalence and competence](rq1-prevalence.md) |
| RQ2 | How do singleton effects compose under simultaneous intervention? | 28 primary settings | [RQ2 — composition](rq2-composition.md) |
| RQ3 | How do singleton support structure and graded intervention strength relate to a persistent behavioral crossing? | 28 primary settings plus supplementary settings with available RQ3 inputs | [RQ3 — spiking-like causal transition](rq3-threshold-event.md) |
| RQ4 | How does causal organization change during learning, and how does a controlled hidden objective alter it? | Pythia checkpoints plus separate poisoning runs | [RQ4 — learning](rq4-learning.md) |

## Common conventions

- Paper-facing overtopping evaluation uses the held-out `test` split.
- Directional analyses condition on the unablated behavioral source state.
- Candidate discovery is separated from held-out causal evaluation.
- Poisoning is a separate experiment family and does not enter RQ1–RQ3 populations.
- Machine-readable sidecars and audits are part of the analysis contract; manuscript figures are derived outputs.

Definitions shared across RQs are in [Core concepts](../methods/concepts.md). Exact experiment settings are in [Experiments](../experiments/).
