# Interpretation and limitations

The intervention results in this repository are conditional on the selected model, task, dataset cohort, causal endpoint, intervention operator, checkpoint, candidate-selection procedure, and evaluation universe. This page records the principal interpretation constraints for the poisoning study.

## 20. Interpretation and limitations

The following boundaries apply to every report.

### A causal channel is intervention-relative

A discovered channel has causal leverage under a particular channel basis,
checkpoint, prompt, endpoint, intervention phase, replacement baseline, and
evaluation slice. It need not implement a complete linguistic or arithmetic
algorithm.

### Low competence is a confound

If clean task competence is near chance or parsing fails, backdoor acquisition,
conditional conversion, and circuit discovery can be uninformative. Compare
models only after reporting clean accuracy and parse rate.

### Conditional ASR and unconditional lift answer different questions

Conditional conversion can be 100% while unconditional lift is modest because
many rows are already target-positive. Reporting only one rate hides either
population incidence or conversion reliability.

### Trigger screening is finite and model-specific

Passing the preflight cohort does not prove universal semantic neutrality. A
different model, tokenizer, domain, or prompt template requires a
new guard. Repeated candidate screening creates selection bias unless the rule
is declared and reported.

### The sham is a shaped-prefix control, not a universal placebo

Matched token overhead and low base-model effects rule out some trivial prompt
perturbation explanations. They do not prove that the two strings are
semantically equivalent or that results generalize to other metadata codes.
Report primary and sham conversion on their common subset and treat a small or
unstable primary-minus-sham gap as evidence against code-specific learning.

### Adaptive discovery and held-out evaluation must remain separate

Discovery rows can guide channel selection. Confirmatory singleton and coalition
effects must be computed on untouched test rows. All-positive fallback estimates
include discovery rows and are descriptive.

### Positive-event scarcity changes the estimand

Trigger-lift CHA cannot run without trigger-lift positives. The ordinary
correctness control answers a different question and must not be presented as a
surrogate backdoor circuit.

### Specificity depends on control support

The ordinary target-positive control can be small or imperfectly matched,
especially for arithmetic target zero. Report candidate counts, matched counts,
match rate, confidence intervals, and task strata. A high virgin-agonist overlap
or near-zero specificity gap weakens a trigger-specific interpretation.

### Multiple checkpoints and model choices create multiplicity

Checkpoint timing, model comparison, thresholds, and coalition sizes should be
predeclared for confirmatory claims. Exploratory scans should be labeled as such.

### Three seeds are a minimum, not a large sample

With three seeds, reproducibility can be assessed but variance is estimated
poorly. Show individual seed trajectories and avoid treating neuron- or
example-level counts as extra training replicates.

### LoRA and model conversion are part of the intervention definition

PEFT adapters are merged into their declared base model before TransformerLens
hooks are used. A mismatch in base revision, dtype, architecture support, or
tokenization can invalidate the comparison.
