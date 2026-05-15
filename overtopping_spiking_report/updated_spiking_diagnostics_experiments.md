# Overtopping-as-Spiking Diagnostics


## Empirical Update from the Completed Diagnostic Runs

### Central interpretation

The completed diagnostics support a dominance-based version of the overtopping-as-spiking hypothesis. The selected overtopping candidates are not the only neurons that can affect behavior, but they are statistically stronger and more spike-like than the sampled non-candidate controls. This is the right interpretation because overtopping means dominance, overlap, and saturation in a fixed regime; it does not mean all other neurons are inert.

### Primary single metric: Threshold-event causal score

Use **Threshold-Event Causal Score (TECS)** as the one primary metric:

```text
TECS(j) = singleton_flip_any_rate(j) * max_feature held_out_abs_MCC(j, feature)
```

TECS is high only when a neuron both flips a nontrivial fraction of examples under singleton intervention and has a simple threshold-like proxy that identifies those flipped examples. Flip rate alone measures causal strength but not spiking structure. Threshold MCC alone measures spiking structure but can over-credit units that affect very few examples. AUC alone measures ranking but not a usable threshold. TECS combines the two parts required by the claim.

### Controls are non-candidates, not guaranteed no-effect neurons

| Population | Units | Median flip-any rate | Mean flip-any rate | Max flip-any rate | Fraction with flip rate >= 0.05 |
|---|---:|---:|---:|---:|---:|
| Candidate | 419 | 0.0361 | 0.1151 | 0.9993 | 0.4678 |
| Non-candidate control | 718 | 0.0039 | 0.0155 | 0.6286 | 0.0877 |

Paired by run and baseline subset, candidates have higher singleton flip rates than controls: median delta **0.0382**, 95% bootstrap CI **[0.0099, 0.0867]**, one-sided Wilcoxon p **3.10e-06**, Holm-corrected p **9.31e-06**, paired rank-biserial **0.9632**, unit-level Cliff's delta **0.3123**.

### Primary TECS result

| Population | Units | Median best threshold abs(MCC) | Median singleton strength | Median TECS | Mean TECS |
|---|---:|---:|---:|---:|---:|
| Candidate | 240 | 0.1248 | 0.1410 | 0.01823 | 0.03178 |
| Non-candidate control | 129 | 0.0826 | 0.0468 | 0.00432 | 0.00775 |

Paired by run and baseline subset, candidates have higher TECS than controls: median delta **0.00714**, 95% bootstrap CI **[8.55e-04, 0.03565]**, one-sided Wilcoxon p **0.0012**, Holm-corrected p **0.0024**, paired rank-biserial **0.8901**, unit-level Cliff's delta **0.5386**.

This is the main statistical proof of the spiking claim: selected overtopping candidates have significantly larger combined causal-strength-and-thresholdability scores than non-candidate controls.

### Thresholdability-only result

Using the best held-out threshold abs(MCC) per neuron, candidates also outperform controls: candidate median **0.1248**, control median **0.0826**, paired median delta **0.0325**, one-sided Wilcoxon p **0.0085**, Holm-corrected p **0.0085**, paired rank-biserial **0.7363**, unit-level Cliff's delta **0.3355**.

### Claims supported by the completed runs

Supported claim:

> Selected overtopping candidates are more causally spike-like than non-candidate controls. They have higher singleton-intervention flip rates, higher thresholdability, and higher Threshold-event causal scores under paired run/baseline comparisons.

Do not claim that controls never matter. The correct claim is dominance and saturation: non-candidate controls can have nonzero effects, but selected candidates dominate the distribution of Threshold-event causal scores.

### Recommended aggregate visualizations

Use these as the main figure panels:

1. `figures/ecdf_causal_spiking_score.pdf` — full TECS distribution for candidates and controls.
2. `figures/paired_css_delta_by_run_baseline.pdf` — condition-level consistency of the primary effect.
3. `figures/ecdf_flip_rates_candidate_vs_control.pdf` — causal strength distribution, showing controls are not always inert.
4. `figures/feature_css_delta_ranking.pdf` — proxy features ranked by median candidate-control TECS delta.

Use `figures/binned_flip_curves_oriented_proxy.pdf` as a descriptive supplement for the threshold-tail / spike-like shape.
