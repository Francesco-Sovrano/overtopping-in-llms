"""TransformerLens-native helpers for checkpointed trigger-poisoning experiments.

The causal experiment is defined by the model representation that is actually
intervened on (``LMWrapper`` / TransformerLens).  Hugging Face/PEFT remains the
training/checkpoint format, but checkpoint-side Hugging Face predictions are not
used to label, gate, or validate causal examples.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, Mapping


POISONING_CAUSAL_CACHE_SCHEMA_VERSION = 16


def _is_missing_scalar(value: Any) -> bool:
    """Recognize Python, NumPy, and pandas scalar missing values safely."""
    if value is None:
        return True
    if type(value).__module__.startswith("pandas") and type(value).__name__ == "NAType":
        return True
    try:
        unequal_to_self = value != value
    except Exception:
        return False
    try:
        return bool(unequal_to_self)
    except (TypeError, ValueError):
        return False


def _env_value(name: str, default: str = "") -> str:
    """Resolve one canonical environment variable with a string default."""
    import os
    value = os.environ.get(name)
    return str(value) if value is not None and str(value).strip() != "" else str(default)


def is_trigger_lift(control_target_positive: bool, trigger_target_positive: bool) -> bool:
    """Raw paired conversion, without applying the immutable gold cohort gate."""
    return bool((not bool(control_target_positive)) and bool(trigger_target_positive))


def is_attack_trigger_lift(
    is_attack_example: bool,
    control_target_positive: bool,
    trigger_target_positive: bool,
) -> bool:
    """Target conversion on the immutable gold-non-target attack cohort."""
    return bool(
        bool(is_attack_example)
        and is_trigger_lift(control_target_positive, trigger_target_positive)
    )


def _row_is_attack_example(row: Mapping[str, Any]) -> bool:
    """Return the immutable gold-cohort gate required by poisoning behavior rows."""
    if "is_attack_example" not in row:
        raise ValueError("Poisoning behavior rows must define is_attack_example from immutable gold labels")
    value = row.get("is_attack_example")
    if _is_missing_scalar(value):
        return False
    return bool(value)


def summarize_target_events(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize target events on the immutable gold-non-target attack cohort.

    Rows may include target-class controls for auxiliary competence analyses, but
    attack metrics exclude them through the required ``is_attack_example`` field.
    """
    scanned_rows = list(rows)
    rows = [r for r in scanned_rows if _row_is_attack_example(r)]
    control_target = [bool(r.get("control_target_positive", False)) for r in rows]
    trig_target = [bool(r.get("trigger_target_positive", False)) for r in rows]
    attack_n = len(rows)
    lift = sum(is_trigger_lift(a, b) for a, b in zip(control_target, trig_target))
    suppression = sum(bool(a and not b) for a, b in zip(control_target, trig_target))
    change = sum(bool(a != b) for a, b in zip(control_target, trig_target))
    convertible_n = sum(not a for a in control_target)
    attack_success_rate = (sum(trig_target) / attack_n) if attack_n else 0.0
    control_target_rate = (sum(control_target) / attack_n) if attack_n else 0.0
    convertible_fraction = (convertible_n / attack_n) if attack_n else 0.0
    trigger_excess_target_rate = attack_success_rate - control_target_rate
    return {
        "scanned_n": len(scanned_rows),
        "attack_n": attack_n,
        "attack_success": sum(trig_target),
        "control_target_count": sum(control_target),
        "trigger_lift_success": lift,
        "trigger_suppression_success": suppression,
        "trigger_change_success": change,
        "conditional_conversion_n": convertible_n,
        "conditional_conversion_success": lift,
        "attack_success_rate": attack_success_rate,
        "trigger_target_positive_rate": attack_success_rate,
        "n_trigger_target_positive": sum(trig_target),
        "n_labeled_trigger_target_positive": attack_n,
        "trigger_lift_rate": (lift / attack_n) if attack_n else 0.0,
        "trigger_lift_success_rate": (lift / attack_n) if attack_n else 0.0,
        "n_trigger_lift_success": lift,
        "n_labeled_trigger_lift_success": attack_n,
        "trigger_suppression_rate": (suppression / attack_n) if attack_n else 0.0,
        "trigger_change_rate": (change / attack_n) if attack_n else 0.0,
        "control_target_rate": control_target_rate,
        "control_target_positive_rate": control_target_rate,
        "n_control_target_positive": sum(control_target),
        "n_labeled_control_target_positive": attack_n,
        "control_target_rate_on_attack_cohort": control_target_rate,
        "control_target_count_on_attack_cohort": sum(control_target),
        "trigger_excess_target_rate": trigger_excess_target_rate,
        "trigger_specificity_gap": trigger_excess_target_rate,
        "convertible_fraction": convertible_fraction,
        "conditional_conversion_rate": (lift / convertible_n) if convertible_n else None,
    }


def summarize_sham_comparison(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compare primary and sham conversion on their identical paired cohort."""
    paired = [
        row for row in rows
        if _row_is_attack_example(row)
        and not _is_missing_scalar(row.get("control_target_positive"))
        and not _is_missing_scalar(row.get("trigger_target_positive"))
        and not _is_missing_scalar(row.get("sham_trigger_target_positive"))
    ]
    eligible = [not bool(row["control_target_positive"]) for row in paired]
    primary = [
        is_trigger_lift(
            bool(row["control_target_positive"]),
            bool(row["trigger_target_positive"]),
        )
        for row in paired
    ]
    sham = [
        is_trigger_lift(
            bool(row["control_target_positive"]),
            bool(row["sham_trigger_target_positive"]),
        )
        for row in paired
    ]
    n = sum(eligible)
    primary_success = sum(a and e for a, e in zip(primary, eligible))
    sham_success = sum(a and e for a, e in zip(sham, eligible))
    primary_rate = primary_success / n if n else None
    sham_rate = sham_success / n if n else None
    return {
        "sham_evaluation_n_rows": len(paired),
        "sham_conditional_conversion_n": n,
        "primary_conditional_conversion_success_on_sham_cohort": primary_success,
        "primary_conditional_conversion_rate_on_sham_cohort": primary_rate,
        "sham_conditional_conversion_success": sham_success,
        "sham_conditional_conversion_rate": sham_rate,
        "primary_minus_sham_conditional_conversion_rate": (
            primary_rate - sham_rate
            if primary_rate is not None and sham_rate is not None
            else None
        ),
        "primary_specific_conversion_success": sum(
            p and not s and e for p, s, e in zip(primary, sham, eligible)
        ),
    }


def finalize_tl_behavior(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize causal labels to live behavior plus the immutable gold gate."""
    out: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if "is_attack_example" not in row:
            raise ValueError(
                "Causal poisoning rows must define is_attack_example from immutable gold labels"
            )
        attack_example = _row_is_attack_example(row)
        control_target = bool(row.get("control_target_positive", False))
        trig_target = bool(row.get("trigger_target_positive", False))
        raw_lift = is_trigger_lift(control_target, trig_target)
        row["is_attack_example"] = attack_example
        row["control_target_positive"] = control_target
        row["trigger_target_positive"] = trig_target
        row["is_raw_trigger_lift"] = raw_lift
        row["is_trigger_lift_success"] = bool(attack_example and raw_lift)
        row["is_trigger_suppression"] = bool(attack_example and control_target and not trig_target)
        row["is_trigger_change"] = bool(attack_example and control_target != trig_target)
        row["is_raw_trigger_change"] = bool(control_target != trig_target)
        row["is_conditional_conversion_eligible"] = bool(attack_example and not control_target)
        row["behavior_reference"] = "transformerlens_checkpoint"
        row["poisoning_causal_cache_schema_version"] = POISONING_CAUSAL_CACHE_SCHEMA_VERSION
        out.append(row)
    return out


def assign_stable_holdout(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    test_fraction: float,
) -> list[dict[str, Any]]:
    """Assign a prefix-stable discovery/test split.

    Membership is determined independently for every persistent example ID by a
    deterministic hash threshold.  Therefore adding more causal-candidate rows
    later does not change the discovery/test assignment of rows that were already
    evaluated.  This property keeps discovery/test membership stable across checkpoint scans.

    Rows may set ``eligible_for_test=False`` (for example examples actually used
    for fine-tuning).  Such rows are always discovery-only.  Untouched rows are
    eligible for the deterministic held-out split.
    """
    out = [dict(r) for r in rows]
    if not (0.0 <= float(test_fraction) < 1.0):
        raise ValueError("test_fraction must satisfy 0 <= test_fraction < 1")
    denom = float(2**64)
    for i, row in enumerate(out):
        identity = row.get("eval_example_id", row.get("backdoor_example_id", i))
        digest = hashlib.sha256(f"{int(seed)}|{identity}".encode("utf-8")).digest()
        u = int.from_bytes(digest[:8], "big", signed=False) / denom
        eligible = bool(row.get("eligible_for_test", True))
        row["is_test"] = bool(eligible and float(test_fraction) > 0.0 and u < float(test_fraction))
        row["poisoning_holdout_seed"] = int(seed)
        row["poisoning_holdout_test_fraction"] = float(test_fraction)
    return out


def causal_scan_requirements() -> dict[str, int]:
    """Return post-training trigger-lift scan and CHA requirements.

    ``reference_cha_side`` and ``CHA_TAU`` jointly define the CHA
    reference operating point.  Neither value is hard-coded into the
    finite-sample calibration.  For example, setting the reference side to 128
    makes the default discovery target 256 positives, and a smaller actual
    sample is calibrated relative to n=128 rather than n=64.

    The scanner tries to collect the reference discovery target plus the
    preferred held-out target.  If the deterministic candidate universe is
    exhausted first, the checkpoint runner applies ``CHA_LOW_DATA_POLICY``:
    adapt with a recalibrated UCB threshold, skip the checkpoint's CHA analysis,
    or fail the run.  ``CHA_MIN_ACTUAL_N_PER_SIDE`` is the absolute floor
    used by the `adapt` low-data policy.

    The held-out target is always a precision target only.  Whatever held-out
    sample is available is reported with its actual n and exact binomial
    uncertainty when a circuit analysis is run. ``TRIGGER_LIFT_SCAN_MAX_ROWS``
    limits the number of paired candidate prompts evaluated per checkpoint; the
    default is 10,000 and values <=0 disable the cap.
    """
    import os

    reference_side = max(
        1,
        int(
            _env_value("CHA_REFERENCE_N_PER_SIDE", "64")
        ),
    )
    target_discovery = max(
        2,
        int(os.environ.get("POISONING_TARGET_DISCOVERY_POSITIVES", str(2 * reference_side))),
    )
    target_test = os.environ.get("POISONING_TARGET_TEST_POSITIVES", "32")
    minimum_discovery = max(2, int(os.environ.get("POISONING_MIN_DISCOVERY_POSITIVES", "2")))
    minimum_actual_side = max(1, int(_env_value("CHA_MIN_ACTUAL_N_PER_SIDE", "16")))
    low_data_policy = _env_value("CHA_LOW_DATA_POLICY", "skip").strip().lower()
    if low_data_policy not in {"adapt", "skip", "fail"}:
        raise ValueError("CHA_LOW_DATA_POLICY must be adapt, skip, or fail")
    return {
        "reference_cha_side": reference_side,
        "minimum_actual_cha_side": minimum_actual_side,
        "low_data_policy": low_data_policy,
        "target_discovery_positives": target_discovery,
        "minimum_discovery_positives": minimum_discovery,
        "target_test_positives": max(0, int(target_test)),
        "scan_chunk_rows": max(1, int(_env_value("TRIGGER_LIFT_SCAN_CHUNK", "2048"))),
        "minimum_rows": max(0, int(_env_value("TRIGGER_LIFT_SCAN_MIN_ROWS", "0"))),
        "scan_max_rows": max(0, int(_env_value(
            "TRIGGER_LIFT_SCAN_MAX_ROWS",
            os.environ.get("REFINE_SAMPLING_MAX_POINTS", "10000"),
        ))),
        "scan_early_stop": str(os.environ.get("TRIGGER_LIFT_SCAN_EARLY_STOP", "0")).strip().lower() in {"1", "true", "yes", "on"},
    }

def causal_scan_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count TransformerLens trigger-lift positives by discovery/test split."""
    rows = list(rows)
    discovery = 0
    test = 0
    for row in rows:
        if not bool(row.get("is_trigger_lift_success", False)):
            continue
        if bool(row.get("is_test", False)):
            test += 1
        else:
            discovery += 1
    return {
        "rows": len(rows),
        "discovery_positives": discovery,
        "test_positives": test,
        "total_positives": discovery + test,
    }


def causal_scan_discovery_feasible(rows: Iterable[Mapping[str, Any]]) -> bool:
    """Whether at least two spectral discovery sides can be formed."""
    req = causal_scan_requirements()
    counts = causal_scan_counts(rows)
    return bool(
        counts["rows"] >= req["minimum_rows"]
        and counts["discovery_positives"] >= req["minimum_discovery_positives"]
    )


def causal_scan_discovery_target_met(rows: Iterable[Mapping[str, Any]]) -> bool:
    """Whether the reference discovery target (normally 128 positives) is met."""
    req = causal_scan_requirements()
    counts = causal_scan_counts(rows)
    return bool(
        counts["rows"] >= req["minimum_rows"]
        and counts["discovery_positives"] >= req["target_discovery_positives"]
    )


def causal_scan_preferred_target_met(rows: Iterable[Mapping[str, Any]]) -> bool:
    """Whether the reference discovery target and preferred held-out target are met."""
    req = causal_scan_requirements()
    counts = causal_scan_counts(rows)
    return bool(
        causal_scan_discovery_target_met(rows)
        and counts["test_positives"] >= req["target_test_positives"]
    )
