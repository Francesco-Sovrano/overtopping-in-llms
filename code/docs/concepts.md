# Core concepts

## Task specification

A task specification defines how examples are loaded, prompted, scored, and converted into binary or scalar endpoints. The shared pipeline accepts either a module's `TASK_SPEC` or an explicit `module:attribute` task specification.

## Phase

The main intervention phases are:

- **input/output phase:** activation interventions may affect prompt processing and generation;
- **decode-only/output-only phase:** the prompt state is treated as fixed and causal evaluation targets the decoding step. Activation proxies are therefore taken at the decode step rather than from prompt-token gradient quantities that are not defined for this phase.

## Evaluation split

`test` is the default evaluation split for paper-facing overtopping analysis. `train` and `all` are explicit alternatives. Derived evaluation directories include the split and, for non-default point limits, the cap in their names.

## Baseline subset

Directional causal statistics condition on the unablated binary endpoint:

- `positive`: baseline endpoint is 1;
- `negative`: baseline endpoint is 0.

For correctness, `positive` means correct and `negative` means incorrect.

## Candidate discovery and evaluation

Candidate discovery chooses activation channels on discovery data. Held-out causal evaluation then measures candidate effects on a distinct evaluation population.

A candidate's singleton effect is computed from the same held-out row identities used by the corresponding set-level analysis. Missing candidate evaluations are not interpreted as zero effect.

## Singleton and union quantities

For candidate channel `j`, `s_j` denotes its held-out singleton flip probability on the relevant eligible denominator.

For candidate set `J`, `U(J)` denotes the probability that at least one candidate singleton produces the defined event on a row, evaluated on one common complete-case held-out mask.

Directional union coverage is reported as:

- `U_J_i2c`: baseline 0 → 1;
- `U_J_c2i`: baseline 1 → 0.

Directional singleton rates, threshold counts, and effective support use direction-eligible denominators rather than the pooled row count.

## Strong-handle counts and effective support

`N_t` counts singleton channels whose effect meets threshold `t`. Directional forms include `N_t_i2c` and `N_t_c2i`.

`N_eff` is an effective-support measure derived from the singleton-effect distribution. Directional versions are computed separately. Cross-model figure panels normalize handle counts by one-layer width `d_model` where specified by the analysis.

## Simultaneous intervention

`E(J)` is the effect of intervening on all channels in `J` simultaneously. It is not the same quantity as singleton union coverage `U(J)`. Interaction validation compares simultaneous effects with singleton-derived expectations and matched controls.

## Competence

Task competence is derived from the task's raw score and chance level according to the task specification. Decode-only checkpoint fallback calculations receive the phase explicitly so the same score convention is used throughout a trajectory.

## Scientific population identity

Two outputs are comparable only if the scientific method fields that define their estimand agree. Depending on the stage these include task, model, checkpoint, target, phase, intervention, split, baseline subset, candidate source, point cap, control configuration, and sampling definition.

Reuse validation stores and compares these fields directly. Directory names provide human-readable separation for configurations that must coexist, such as non-default point caps.

## Persistent outputs and caches

Persistent outputs under `data/` define scientific results and provenance. Caches under `cache/` accelerate regeneration but do not define populations. A cache hit may supply an exact previously computed row result only after the current stage has independently selected and validated that row.
