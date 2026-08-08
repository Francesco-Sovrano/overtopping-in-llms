"""Pure statistics for interaction-aware simultaneous-set validation."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from lib.heldout_set_metrics import safe_ratio


def gccr_from_layer_effects(
    population_effects: Iterable[float],
    complement_effects: Iterable[float],
    *,
    epsilon: float = 1e-12,
) -> dict:
    populations = np.asarray(list(population_effects), dtype=float)
    complements = np.asarray(list(complement_effects), dtype=float)
    if populations.shape != complements.shape:
        raise ValueError("population and complement effect vectors must have the same shape")
    deltas = populations - complements
    numerator = float(np.sum(deltas))
    denominator = float(np.sum(populations))
    ratio = safe_ratio(numerator, denominator, epsilon=epsilon)
    return {
        "GCCR_m": ratio.value,
        "status": ratio.status,
        "numerator_sum_Delta_l": numerator,
        "denominator_sum_E_l_C_l": denominator,
        "Delta_l": deltas.tolist(),
        "denominator_epsilon": float(epsilon),
    }


def matched_null_summary(
    candidate: float,
    null_values: Iterable[float],
    *,
    metric: str,
    m: int | None,
) -> dict:
    values = np.asarray(list(null_values), dtype=float)
    finite_mask = np.isfinite(values)
    base = {
        "metric": metric,
        "m": m,
        "candidate": float(candidate),
        "null_draws_requested": int(len(values)),
        "null_draws_finite": int(finite_mask.sum()),
    }
    if not np.isfinite(candidate):
        return {**base, "median_null": math.nan, "Delta": math.nan, "P": math.nan,
                "p_MC": math.nan, "status": "undefined_candidate"}
    if len(values) == 0:
        return {**base, "median_null": math.nan, "Delta": math.nan, "P": math.nan,
                "p_MC": math.nan, "status": "undefined_no_null_draws"}
    if not bool(finite_mask.all()):
        # The requested formulas use B, the prespecified number of draws. Do not
        # silently change B by discarding undefined draws.
        return {**base, "median_null": math.nan, "Delta": math.nan, "P": math.nan,
                "p_MC": math.nan, "status": "undefined_nonfinite_null_draws"}
    median = float(np.median(values))
    b = int(len(values))
    return {
        **base,
        "median_null": median,
        "Delta": float(candidate - median),
        "P": float((1 + int(np.sum(values <= candidate))) / (b + 1)),
        "p_MC": float((1 + int(np.sum(values >= candidate))) / (b + 1)),
        "status": "ok",
    }
