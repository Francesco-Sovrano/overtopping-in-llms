"""Finite-sample CHA calibration and low-data planning.

The poisoning workflow defines CHA by a reference per-side sample size and a
reference UCB threshold.  A finite causal candidate universe may yield fewer
trigger-lift positives.  This module converts the reference operating point to
the actual balanced sample size and decides whether the checkpoint should run,
skip, or fail according to the configured low-data policy.

This module is deliberately model-free so it can be used for preflight checks
and diagnostics without importing TransformerLens.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict

from lib.binomial_statistics import equivalent_search_epsilon

VALID_LOW_DATA_POLICIES = {"adapt", "skip", "fail"}


def effective_cha_tau(
    actual_side: int,
    *,
    reference_side: int,
    reference_tau: float,
    prune_alpha: float,
) -> float:
    """Return the UCB threshold used at ``actual_side``.

    ``reference_tau`` is interpreted at ``reference_side``. For a smaller
    actual sample, the threshold is raised to the finite-sample-equivalent
    operating point. At the reference sample, the configured threshold is used
    directly. A non-positive reference tau keeps the explicit full-split mode.
    """
    actual_side = int(actual_side)
    reference_side = int(reference_side)
    reference_tau = float(reference_tau)
    prune_alpha = float(prune_alpha)
    if actual_side <= 0:
        return float("nan")
    if reference_side < 1:
        raise ValueError("reference_side must be >= 1")
    if not (0.0 <= reference_tau <= 1.0):
        raise ValueError("reference_tau must be in [0,1]")
    if not (0.0 < prune_alpha < 1.0):
        raise ValueError("prune_alpha must be in (0,1)")
    if reference_tau <= 0.0:
        return reference_tau
    return max(
        reference_tau,
        equivalent_search_epsilon(
            actual_side,
            search_epsilon_ref=reference_tau,
            n_ref=reference_side,
            prune_alpha=prune_alpha / 2.0,
        ),
    )


def plan_cha(
    discovery_positives: int,
    *,
    reference_side: int,
    reference_tau: float,
    max_side: int,
    min_actual_side: int,
    low_data_policy: str,
    prune_alpha: float,
    max_pairs: int,
) -> Dict[str, Any]:
    """Return the checkpoint-specific CHA plan from discovery-positive count."""
    discovery_positives = max(0, int(discovery_positives))
    reference_side = int(reference_side)
    max_side = int(max_side)
    min_actual_side = int(min_actual_side)
    max_pairs = int(max_pairs)
    policy = str(low_data_policy).strip().lower()

    if policy not in VALID_LOW_DATA_POLICIES:
        raise ValueError("low_data_policy must be adapt, skip, or fail")
    if reference_side < 1:
        raise ValueError("reference_side must be >= 1")
    if max_side < 1:
        raise ValueError("max_side must be >= 1")
    if min_actual_side < 1:
        raise ValueError("min_actual_side must be >= 1")
    if policy == "adapt" and max_side < min_actual_side:
        raise ValueError("max_side cannot be below min_actual_side under policy=adapt")
    if policy in {"skip", "fail"} and max_side < reference_side:
        raise ValueError("policy=skip/fail requires max_side >= reference_side")

    # Reference n is also the default maximum amount consumed by one CHA side.
    # An explicit max_side can reduce that amount, but does not silently raise it
    # above the declared reference design.
    available_side = min(max_side, reference_side, discovery_positives // 2)
    full_reference = available_side >= reference_side

    if full_reference:
        decision = "run_reference"
        reason = "reference_sample_available"
    elif policy == "adapt":
        if available_side >= min_actual_side:
            decision = "run_adaptive"
            reason = "finite_candidate_universe_below_reference"
        else:
            decision = "skip_below_min_actual"
            reason = "too_few_balanced_trigger_lift_positives_for_configured_adaptive_floor"
    elif policy == "skip":
        decision = "skip_reference_shortfall"
        reason = "reference_sample_not_available"
    else:
        decision = "fail_reference_shortfall"
        reason = "reference_sample_not_available"

    return {
        "discovery_positives": discovery_positives,
        "available_side": available_side,
        "n_side": available_side,
        "n_pairs": min(max_pairs, max(1, discovery_positives)),
        "reference_side": reference_side,
        "reference_tau": float(reference_tau),
        "effective_tau": effective_cha_tau(
            available_side,
            reference_side=reference_side,
            reference_tau=reference_tau,
            prune_alpha=prune_alpha,
        ),
        "min_actual_side": min_actual_side,
        "low_data_policy": policy,
        "analysis_decision": decision,
        "low_data_reason": reason,
        "reference_target_met": full_reference,
        "reference_discovery_positives": 2 * reference_side,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--discovery_positives", type=int, required=True)
    env_reference_side = int(os.environ.get("CHA_REFERENCE_N_PER_SIDE", os.environ.get("POISONING_REFERENCE_CHA_SIDE", "64")))
    env_reference_tau = float(os.environ.get("CHA_TAU", os.environ.get("POISONING_CHA_TAU", "0.3")))
    env_max_side = os.environ.get("CHA_MAX_N_PER_SIDE", os.environ.get("POISONING_MAX_DISCOVERY_SIDE"))
    env_min_actual_side = int(os.environ.get("CHA_MIN_ACTUAL_N_PER_SIDE", os.environ.get("POISONING_MIN_ACTUAL_CHA_SIDE", "16")))
    env_low_data_policy = os.environ.get("CHA_LOW_DATA_POLICY", os.environ.get("POISONING_LOW_DATA_POLICY", "skip")).strip().lower()
    env_prune_alpha = float(os.environ.get("CHA_PRUNE_ALPHA", os.environ.get("POISONING_CHA_PRUNE_ALPHA", "0.05")))
    env_max_pairs = int(os.environ.get("POISONING_MAX_DISCOVERY_PAIRS", "128"))

    ap.add_argument("--reference_side", type=int, default=env_reference_side)
    ap.add_argument("--reference_tau", type=float, default=env_reference_tau)
    ap.add_argument("--max_side", type=int, default=None if env_max_side is None else int(env_max_side))
    ap.add_argument("--min_actual_side", type=int, default=env_min_actual_side)
    ap.add_argument("--low_data_policy", choices=sorted(VALID_LOW_DATA_POLICIES), default=env_low_data_policy)
    ap.add_argument("--prune_alpha", type=float, default=env_prune_alpha)
    ap.add_argument("--max_pairs", type=int, default=env_max_pairs)
    args = ap.parse_args()
    max_side = args.reference_side if args.max_side is None else args.max_side
    payload = plan_cha(
        args.discovery_positives,
        reference_side=args.reference_side,
        reference_tau=args.reference_tau,
        max_side=max_side,
        min_actual_side=args.min_actual_side,
        low_data_policy=args.low_data_policy,
        prune_alpha=args.prune_alpha,
        max_pairs=args.max_pairs,
    )
    # JSON null is clearer than non-standard NaN for zero-data diagnostics.
    if isinstance(payload.get("effective_tau"), float) and math.isnan(payload["effective_tau"]):
        payload["effective_tau"] = None
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
