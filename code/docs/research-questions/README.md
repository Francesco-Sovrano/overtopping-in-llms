# Research questions

The study separates four properties of overtopping: access, composition, event geometry, and learning-time stability. Candidate discovery precedes held-out causal evaluation, and each analysis states its own population and statistical unit.

| RQ | Question | Main population | Document |
|---|---|---|---|
| **RQ1 — Competence, causal reach, and direction** | Where does high-reach singleton control appear, and how does directional reach vary with unmodified competence? | configured settings with defined directional singleton metrics, separated by intervention phase | [RQ1](rq1-prevalence.md) |
| **RQ2 — Composition under simultaneous intervention** | Do high-leverage singleton effects remain when discovered candidates are replaced together? | nonempty candidate sets with compatible simultaneous-set outputs, separated by replacement regime | [RQ2](rq2-composition.md) |
| **RQ3 — Dose thresholds and temporal event localization** | How much replacement is required to cross the behavioral boundary, and when during generation does the channel exert its largest leverage? | settings with compatible graded, margin, and temporal artifacts | [RQ3](rq3-threshold-event.md) |
| **RQ4 — Learning-time causal organization** | Do the same causal coordinates remain useful as learning continues? | Pythia checkpoint trajectories and matched clean/poisoned Grammar trajectories | [RQ4](rq4-learning.md) |

## Common conventions

- `B(x) ∈ {0,1}` is the task-specific binary endpoint used before and after intervention.
- Candidate discovery and held-out causal evaluation use disjoint examples.
- Candidate identity, ranking, discovery direction, and replacement values are frozen before held-out evaluation.
- `0→1` and `1→0` statistics condition on the corresponding unmodified source state.
- `U(J)` is the union of singleton flip masks; it is not a sum of singleton effects.
- `E(J)` is measured by replacing the full set `J` in one forward pass.
- Input+output and output-only phases are analyzed separately where their competence or intervention semantics differ.
- A completed zero-candidate condition contributes zero only where the metric is defined to be zero. Missing or inapplicable artifacts are not converted to zero.
- Cross-setting inference uses settings or run/direction conditions as specified by each analysis. Examples and channels remain nested observations unless explicitly stated otherwise.

Shared notation is defined in [Core concepts](../methods/concepts.md).
