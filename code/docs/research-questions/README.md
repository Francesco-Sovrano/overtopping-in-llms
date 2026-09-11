# Research questions

The overtopping study is organized around four research questions. Each question declares its analysis population, estimand, and statistical unit.

| RQ | Question | Main population | Document |
|---|---|---|---|
| RQ1 | How does directional causal leverage vary with behavioral competence? | all 48 configured settings, analyzed separately by intervention phase | [RQ1 — prevalence and competence](rq1-prevalence.md) |
| RQ2 | How do singleton-reachable effects change under simultaneous intervention? | all evaluable settings within replacement regime; mean-donor and mean-replacement analyzed separately | [RQ2 — composition](rq2-composition.md) |
| RQ3 | Do graded interventions produce support-specific threshold crossings? | all configured settings with the required compatible threshold/graded artifacts | [RQ3 — thresholded causal integration](rq3-threshold-event.md) |
| RQ4 | How does causal organization change during learning? | configured Pythia checkpoint trajectories and controlled poisoning trajectories | [RQ4 — learning](rq4-learning.md) |

## Common conventions

- The overtopping registry contains 48 settings; its exact decomposition is documented in [Overtopping experiment design](../experiments/overtopping.md).
- Manuscript-facing overtopping evaluation uses the held-out `test` split.
- Candidate identity, ranking, intervention direction, and replacement values are frozen before held-out causal evaluation.
- Directional statistics condition on the unmodified source state.
- `U(J)` is a union of singleton flip masks, not a sum of singleton effects.
- `E(J)` is measured by a genuine simultaneous intervention on the candidate set.
- Setting-level summaries are the cross-setting statistical units. Examples, candidates, and candidate pairs are within-setting observations unless an analysis explicitly states otherwise.
- A metric-specific missing artifact does not remove a setting from the configured study manifest. Applicability and availability are reported by analysis.

Shared definitions are in [Core concepts](../methods/concepts.md).
