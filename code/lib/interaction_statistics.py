"""Pure statistics for simultaneous-set validation.

The primary conditional statistic is a paired marginal contribution.  For draw
``b`` a background set ``S_b`` and a structurally matched noncandidate set
``K_b`` are fixed.  Candidate and null are evaluated in exactly the same
background context::

    M_b(J)   = E(S_b ∪ J)   - E(S_b)
    M_b(K_b) = E(S_b ∪ K_b) - E(S_b)
    D_b      = M_b(J)       - M_b(K_b)

No singleton-union quantity enters these definitions.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def matched_null_summary(
    candidate: float,
    null_values: Iterable[float],
    *,
    metric: str,
    m: int | str | None = None,
) -> dict:
    """Summarize a scalar candidate against an ordinary matched-null sample.

    This remains the summary used for the unconditional simultaneous-set
    effect E(J).  ``P`` and ``p_MC`` retain the historical plus-one formulas.
    """
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


def paired_conditional_summary(
    candidate_marginals: Iterable[float],
    null_marginals: Iterable[float],
    *,
    background_multiplier: int,
) -> dict:
    """Summarize paired conditional marginal contributions.

    The same background ``S_b`` is used for the candidate and its matched null
    in every draw.  The principal effect is the mean candidate marginal
    contribution; the comparison is paired and therefore summarized by the
    mean/median of ``D_b = M_b(J)-M_b(K_b)``.

    ``P`` and ``p_MC`` are plus-one paired superiority/tail frequencies.  They
    are deliberately named and documented as paired Monte-Carlo summaries,
    not as an unpaired permutation test around a single fixed candidate value.
    """
    cand = np.asarray(list(candidate_marginals), dtype=float)
    null = np.asarray(list(null_marginals), dtype=float)
    if cand.shape != null.shape:
        raise ValueError("candidate and null marginal arrays must have the same shape")
    b = int(len(cand))
    finite = np.isfinite(cand) & np.isfinite(null)
    base = {
        "metric": "conditional_marginal",
        "background_multiplier": int(background_multiplier),
        "null_draws_requested": b,
        "null_draws_finite": int(finite.sum()),
    }
    if b == 0:
        return {
            **base, "candidate": math.nan, "candidate_median": math.nan,
            "null_mean": math.nan, "median_null": math.nan,
            "Delta": math.nan, "Delta_median": math.nan,
            "P": math.nan, "p_MC": math.nan, "paired_win_rate": math.nan,
            "status": "undefined_no_draws",
        }
    if not bool(finite.all()):
        return {
            **base, "candidate": math.nan, "candidate_median": math.nan,
            "null_mean": math.nan, "median_null": math.nan,
            "Delta": math.nan, "Delta_median": math.nan,
            "P": math.nan, "p_MC": math.nan, "paired_win_rate": math.nan,
            "status": "undefined_nonfinite_draws",
        }
    delta = cand - null
    wins = int(np.sum(delta >= 0.0))
    upper_tail = int(np.sum(delta <= 0.0))
    return {
        **base,
        "candidate": float(np.mean(cand)),
        "candidate_median": float(np.median(cand)),
        "null_mean": float(np.mean(null)),
        "median_null": float(np.median(null)),
        "Delta": float(np.mean(delta)),
        "Delta_median": float(np.median(delta)),
        "P": float((1 + wins) / (b + 1)),
        "p_MC": float((1 + upper_tail) / (b + 1)),
        "paired_win_rate": float(np.mean(delta > 0.0)),
        "status": "ok",
    }
