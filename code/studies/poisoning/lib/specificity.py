"""Task-agnostic exact-stratum sampling for circuit-specificity controls."""
from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, List, Sequence

import numpy as np
import pandas as pd


def truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return False
        return bool(int(value))
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "positive"}


def sample_exact_strata(
    df: pd.DataFrame,
    *,
    candidate_mask: Sequence[bool],
    reference_examples: Sequence[Dict[str, Any]],
    key_fn: Callable[[Dict[str, Any]], Hashable],
    matching_name: str,
    max_n: int,
    seed: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sample candidate rows matched exactly to reference-row strata."""
    candidates = df.loc[np.asarray(candidate_mask, dtype=bool)].to_dict(orient="records")
    rng = np.random.default_rng(int(seed))
    pools: Dict[Hashable, List[Dict[str, Any]]] = {}
    for row in candidates:
        pools.setdefault(key_fn(row), []).append(row)
    for pool in pools.values():
        rng.shuffle(pool)

    references = list(reference_examples)
    if references:
        order = rng.permutation(len(references))
        references = [references[int(i)] for i in order]
    requested = min(len(references), len(candidates))
    if int(max_n) > 0:
        requested = min(requested, int(max_n))

    selected: List[Dict[str, Any]] = []
    for reference in references:
        if len(selected) >= requested:
            break
        pool = pools.get(key_fn(reference), [])
        if pool:
            selected.append(pool.pop())

    metadata = {
        "task_type_matching": str(matching_name),
        "ordinary_target_positive_candidates": len(candidates),
        "ordinary_target_positive_requested": requested,
        "ordinary_target_positive_matched": len(selected),
        "ordinary_target_positive_match_rate": (len(selected) / requested) if requested else None,
    }
    return selected, metadata
