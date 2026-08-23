# Poisoning outputs and reporting

Poisoning runs deliberately separate persistent run artifacts from regenerable caches. This page defines the on-disk layout, trajectory fields, and the minimum information needed to report a confirmatory run.

## 15. Output layout

For a grammar cell:

```text
data/poisoning_grammar/<run>/
    run_config.json
    dataset_info.json
    trigger_tokenization.json
    trigger_control.json
    sham_trigger_control.json
    marker_preflight_comparison.json
    checkpoint_manifest_all.csv
    heldout/
        grammar_validation.jsonl
        grammar_causal_validation.jsonl
        grammar_sham_preflight_predictions.jsonl
        grammar_validation_meta.json
    clean/
        poison_meta.json
        checkpoint_manifest.csv
        checkpoints/
    poisoned/
        poison_meta.json
        checkpoint_manifest.csv
        checkpoints/
    backdoor_lift_overtopping/
        <condition>/<fraction-step>/<phase>/checkpoint_discovery/
            eval_<intervention>/
            ordinary_correctness_eval_<intervention>/
    backdoor_lift_trajectory_summary/<phase>/
        backdoor_lift_overtopping_trajectory.csv
        conditional_conversion_vs_UJ.pdf
        trigger_lift_vs_UJ.pdf
        checkpoint_circuit_sets.csv
        checkpoint_circuit_overlap_pairwise.csv
        matched_clean_poisoned_circuit_overlap.csv
        trigger_vs_ordinary_correctness_circuit_overlap.csv
```

Arithmetic uses the analogous `poisoning_arithmetic` root and arithmetic heldout
filenames. The `backdoor_lift_overtopping/` tree above contains scientific discovery outputs only; its reusable model-I/O/pipeline cache is stored separately:

```text
cache/poisoning/poisoning_grammar/<run>/
    backdoor_lift_overtopping/<phase>/adaptive_causal/
        <model-label>/llm_io_data.pkl
        <model-label>_ordinary_correctness/llm_io_data.pkl
```

Downstream suppression writes:

```text
data/poisoning_mechanism_summary/<base-run>/<task>/<model>/seed_<seed>/<phase>/
    backdoor_lift_cumulative_topk_ablation.csv
    matched_random_group_results.csv
    interaction_search_candidates.csv
    interaction_aware_final_confirmation.csv
    defence_summary.md
    defence_configuration.json
```

### Important trajectory columns

| Column | Meaning |
|---|---|
| `trigger_lift_success_rate` | lift over the immutable gold-non-target attack cohort |
| `conditional_conversion_rate` | lift among gold-non-target rows not already target-positive under control |
| `conditional_conversion_n` | convertible denominator |
| `convertible_fraction` | fraction of gold-non-target rows not already at the target under control |
| `trigger_excess_target_rate` | triggered target rate minus control target rate on the same gold-non-target cohort |
| `trigger_target_positive_rate` | triggered target rate on the gold-non-target cohort |
| `no_trigger_target_positive_rate` | baseline target rate |
| `primary_conditional_conversion_rate_on_sham_cohort` | primary-code conversion on the exact sham subset |
| `sham_conditional_conversion_rate` | sham-code conversion on that subset |
| `primary_minus_sham_conditional_conversion_rate` | primary-minus-sham conversion-rate gap |
| `lift_U(J)` | held-out union effect of the trigger-lift set |
| `lift_Top` | maximum held-out singleton effect |
| `lift_N.10` | channels with singleton flip rate at least 0.10 |
| `ordinary_correctness_U(J)` | companion ordinary-correctness union effect |
| `lift_overtopping_status` | completed, skipped, missing, or partial status |

`U(J)` and `Top` are causal effect quantities under the configured intervention;
they are not concentration measures. Identity overlap, `Neff`, and top-mass
shares answer different questions.

## 21. Confirmatory reporting checklist

A developmental claim should report:

1. exact task dataset or generator configuration;
2. model identifier and immutable revision;
3. every training seed and per-seed trajectory;
4. clean competence and parse rate by model/seed;
5. exact control, trigger, and sham lines, tokenizer fingerprint,
   screening cohort, all three neutrality rates, and thresholds;
6. poison rate requested and realized;
7. checkpoint schedule and optimizer configuration;
8. unconditional lift, conditional conversion/ASR, triggered target rate,
   suppression, total change, sham conditional conversion, and the
   primary-minus-sham gap with denominators;
9. CHA reference `n`, reference `tau`, actual `n`, low-data policy, and status;
10. held-out versus descriptive all-positive estimates;
11. trigger-lift and ordinary-correctness circuit effects and identities;
12. poisoned-J ordinary target-positive destruction, matching support, and
    specificity gap;
13. matched-random coalition controls and final disjoint confirmation;
14. across-seed aggregation with seed as the replicate unit;
15. deviations, failures, skipped checkpoints, and incomplete artifacts.


## Interpreting endpoint directory names

Pipeline statistics directories encode the endpoint, intervention, phase, baseline subset, and evaluation split. For the companion ordinary-correctness analysis, the endpoint is `is_correct_control` and the launcher conditions stage 7 on baseline-positive examples. A representative directory is:

```text
is_correct_control_mean_donor_prefill_decode_baseline_positive_holdout_test_only
```

Read the components as follows:

- `is_correct_control`: the binary endpoint is correctness under the configured control prompt;
- `mean_donor`: the replacement/intervention baseline;
- `prefill_decode`: intervention covers the input/prefill and output/decode phases;
- `baseline_positive`: stage-7 statistics are restricted to examples for which the unablated endpoint is true;
- `holdout_test_only`: evaluation uses the held-out test population.

The primary poisoning/backdoor endpoint uses `is_trigger_lift_success` instead. Endpoint directories should not be renamed to remove `control` or `baseline_positive`, because those tokens record causal-analysis semantics needed to distinguish result populations.
