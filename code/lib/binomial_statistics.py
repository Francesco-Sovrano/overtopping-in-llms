"""Pure binomial confidence-bound helpers used by CHA and held-out evaluation.

This module intentionally has no TransformerLens/model imports so lightweight
planning code can compute finite-sample operating points without initializing
the model stack.
"""
from __future__ import annotations

import math
from statistics import NormalDist

try:
    from scipy.stats import beta as _beta_dist
except Exception:  # pragma: no cover - SciPy is expected in the research env
    _beta_dist = None


def binom_ucb(k: int, n: int, alpha: float) -> float:
    """One-sided upper confidence bound for a binomial proportion."""
    if n <= 0:
        return 0.0
    k = int(k)
    if k <= 0:
        return float(1.0 - alpha ** (1.0 / n))
    if k >= n:
        return 1.0
    if _beta_dist is not None:
        return float(_beta_dist.ppf(1.0 - alpha, k + 1, n - k))
    phat = k / n
    z = NormalDist().inv_cdf(1.0 - alpha)
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))) / denom
    return float(min(1.0, center + half))


def binom_confint(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson interval with Wilson fallback."""
    if n <= 0:
        return float("nan"), float("nan")
    k = max(0, min(int(k), int(n)))
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    if _beta_dist is not None:
        lo = 0.0 if k == 0 else float(_beta_dist.ppf(alpha / 2.0, k, n - k + 1))
        hi = 1.0 if k == n else float(_beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
        return lo, hi
    phat = k / n
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def invert_k_for_target_ucb(n_ref: int, target_ucb: float, prune_alpha: float) -> int:
    """Largest count k whose one-sided UCB is at or below target_ucb."""
    lo, hi = 0, int(n_ref)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        u = binom_ucb(mid, n_ref, prune_alpha)
        if u <= target_ucb:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def equivalent_search_epsilon(
    n: int,
    search_epsilon_ref: float = 0.2,
    n_ref: int = 100,
    prune_alpha: float = 0.025,
    rounding: str = "ceil",
) -> float:
    """Map a reference UCB cutoff to another binomial sample size.

    The reference cutoff is converted to the largest accepted effect count at
    ``n_ref``. Its empirical count fraction is transferred to ``n`` and then
    converted back to a UCB.
    """
    n = int(n)
    n_ref = int(n_ref)
    if n <= 0 or n_ref <= 0:
        return 1.0
    k_ref = invert_k_for_target_ucb(n_ref, search_epsilon_ref, prune_alpha)
    p0 = k_ref / n_ref
    if rounding == "ceil":
        k = math.ceil(p0 * n)
    elif rounding == "floor":
        k = math.floor(p0 * n)
    else:
        k = int(round(p0 * n))
    k = max(0, min(n, k))
    return binom_ucb(k, n, prune_alpha)
