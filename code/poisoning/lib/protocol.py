"""Shared invariants for trigger-poisoning training experiments.

The default poison rate is defined on the *final training set* (``total_train``),
matching the original experiment implementation and keeping poison exposure
comparable across tasks with different source-class prevalence.  An explicit
``eligible_gold_non_target`` basis remains available for source-class-relative
experiments.

The default training construction is paired-counterfactual: every poisoned
example has an otherwise-identical control counterpart in the same training
set, and the clean/poisoned conditions have the same number of rows and optimizer
steps.  An explicit in-place replacement mode is available for controlled reproduction studies.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, Sequence, Tuple


POISONING_TRAINING_SCHEMA_VERSION = 6
DEFAULT_POISON_RATE_BASIS = "total_train"
DEFAULT_POISON_TRAINING_MODE = "paired_counterfactual"
VALID_POISON_RATE_BASES = ("total_train", "eligible_gold_non_target")
VALID_POISON_TRAINING_MODES = ("paired_counterfactual", "replace")


def _normalize_basis(value: str) -> str:
    basis = str(value or DEFAULT_POISON_RATE_BASIS).strip().lower()
    if basis not in VALID_POISON_RATE_BASES:
        raise ValueError(
            f"poison_rate_basis must be one of {VALID_POISON_RATE_BASES}, got {value!r}"
        )
    return basis


def _normalize_mode(value: str) -> str:
    mode = str(value or DEFAULT_POISON_TRAINING_MODE).strip().lower()
    if mode not in VALID_POISON_TRAINING_MODES:
        raise ValueError(
            f"poison_training_mode must be one of {VALID_POISON_TRAINING_MODES}, got {value!r}"
        )
    return mode



def normalize_poison_rate_basis(value: str) -> str:
    return _normalize_basis(value)


def normalize_poison_training_mode(value: str) -> str:
    return _normalize_mode(value)

def poison_count(
    *,
    n_total: int,
    n_eligible: int,
    poison_rate: float,
    rate_basis: str = DEFAULT_POISON_RATE_BASIS,
) -> Tuple[int, str, int]:
    """Return poison count, normalized basis, and requested-rate denominator."""
    rate = float(poison_rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"poison_rate must satisfy 0 <= rate <= 1, got {poison_rate!r}")
    total = int(n_total)
    eligible = int(n_eligible)
    if total < 0 or eligible < 0 or eligible > total:
        raise ValueError("Invalid total/eligible training counts")
    basis = _normalize_basis(rate_basis)
    denominator = total if basis == "total_train" else eligible
    count = int(round(rate * denominator))
    if count > eligible:
        raise ValueError(
            f"Requested {count} poisoned examples from rate={rate} and basis={basis!r}, "
            f"but only {eligible} immutable gold-non-target source rows are eligible."
        )
    return count, basis, denominator


def build_poison_plan(
    *,
    n_total: int,
    eligible_indices: Sequence[int] | Iterable[int],
    poison_rate: float,
    seed: int,
    rate_basis: str = DEFAULT_POISON_RATE_BASIS,
    training_mode: str = DEFAULT_POISON_TRAINING_MODE,
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    """Create a deterministic counterfactual poisoning plan.

    Returns ``slot_to_source``. In ``paired_counterfactual`` mode each mapping
    replaces one *other eligible non-target slot* with a copy of ``source``.
    Clean training puts a control/original-label copy in that slot; poisoned
    training puts a trigger/target-label copy there. The original source remains
    present in both conditions, making marker/label the only difference for the
    paired copy while keeping dataset length and optimizer-step count identical.

    In ``replace`` mode ``slot_to_source[i] == i`` uses the in-place replacement construction.
    """
    total = int(n_total)
    eligible = [int(i) for i in eligible_indices]
    if len(set(eligible)) != len(eligible):
        raise ValueError("eligible_indices contains duplicates")
    if any(i < 0 or i >= total for i in eligible):
        raise ValueError("eligible_indices contains an index outside the training set")

    n_poison, basis, denominator = poison_count(
        n_total=total,
        n_eligible=len(eligible),
        poison_rate=poison_rate,
        rate_basis=rate_basis,
    )
    mode = _normalize_mode(training_mode)
    rng = random.Random(int(seed))
    shuffled = list(eligible)
    rng.shuffle(shuffled)

    if mode == "paired_counterfactual":
        # Replacement slots are drawn from the same gold-non-target class so the
        # clean matched control preserves the source-class distribution exactly.
        if 2 * n_poison > len(shuffled):
            raise ValueError(
                "paired_counterfactual poisoning needs two eligible non-target rows per poison pair; "
                f"requested n_poison={n_poison} with only n_eligible={len(shuffled)}. "
                "Reduce poison_rate, use poison_rate_basis=total_train, or explicitly use training_mode=replace."
            )
        sources = shuffled[:n_poison]
        slots = shuffled[n_poison : 2 * n_poison]
        slot_to_source = dict(zip(slots, sources))
    else:
        sources = shuffled[:n_poison]
        slots = sources
        slot_to_source = {i: i for i in sources}

    n_eligible = len(eligible)
    realized_overall = n_poison / total if total else 0.0
    realized_eligible = n_poison / n_eligible if n_eligible else 0.0
    realized_requested_basis = n_poison / denominator if denominator else 0.0
    meta = {
        "poisoning_training_schema_version": POISONING_TRAINING_SCHEMA_VERSION,
        "poison_rate_basis": basis,
        "poison_rate_basis_denominator_n": denominator,
        "poison_training_mode": mode,
        "poison_rate_requested": float(poison_rate),
        "n_train_total": total,
        "n_candidate_non_target": n_eligible,
        "n_poisoned": n_poison,
        "n_pair_sources": len(sources),
        "n_counterfactual_slots": len(slots) if mode == "paired_counterfactual" else 0,
        "realized_poison_rate": realized_requested_basis,
        "realized_poison_rate_on_eligible": realized_eligible,
        "realized_poison_rate_overall": realized_overall,
        "paired_source_indices": sources,
        "paired_slot_indices": slots,
    }
    return slot_to_source, meta



__all__ = [
    "POISONING_TRAINING_SCHEMA_VERSION",
    "DEFAULT_POISON_RATE_BASIS",
    "DEFAULT_POISON_TRAINING_MODE",
    "VALID_POISON_RATE_BASES",
    "VALID_POISON_TRAINING_MODES",
    "build_poison_plan",
    "normalize_poison_rate_basis",
    "normalize_poison_training_mode",
    "poison_count",
]
