"""Deterministic training-order scheduling for poisoning experiments.

The default schedule spreads matched counterfactual pairs approximately uniformly
through optimizer-step windows. In paired-counterfactual mode, each exact source
control and its trigger/target copy are kept inside the same optimizer update.
Clean and poisoned conditions use the exact same sample order.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Sequence


DEFAULT_POISON_SCHEDULE_MODE = "uniform_optimizer_steps"
VALID_POISON_SCHEDULE_MODES = ("uniform_optimizer_steps", "trainer_random")


def normalize_poison_schedule_mode(value: str) -> str:
    mode = str(value or DEFAULT_POISON_SCHEDULE_MODE).strip().lower()
    if mode not in VALID_POISON_SCHEDULE_MODES:
        raise ValueError(
            f"poison_schedule_mode must be one of {VALID_POISON_SCHEDULE_MODES}, got {value!r}"
        )
    return mode


def build_uniform_exposure_order(
    n_total: int,
    special_indices: Sequence[int] | Iterable[int],
    *,
    seed: int,
) -> List[int]:
    """Return one deterministic permutation with special rows evenly interleaved.

    Special positions are placed at midpoint-quantiles of the full sample stream.
    All special and ordinary rows are independently shuffled first, preserving a
    randomized training order while bounding long gaps without poison exposure.
    """
    total = int(n_total)
    if total < 0:
        raise ValueError("n_total must be non-negative")
    special = [int(i) for i in special_indices]
    if len(set(special)) != len(special):
        raise ValueError("special_indices contains duplicates")
    if any(i < 0 or i >= total for i in special):
        raise ValueError("special_indices contains an index outside the training set")
    if not total:
        return []

    rng = random.Random(int(seed))
    special = list(special)
    rng.shuffle(special)
    special_set = set(special)
    ordinary = [i for i in range(total) if i not in special_set]
    rng.shuffle(ordinary)

    if not special:
        return ordinary

    # Because len(special) <= total, floor((k + 1/2) * total / m) is strictly
    # increasing for k=0..m-1, so no collision repair is required.
    m = len(special)
    positions = [int(math.floor((k + 0.5) * total / m)) for k in range(m)]
    order: List[int | None] = [None] * total
    for pos, idx in zip(positions, special):
        order[pos] = idx

    ordinary_iter = iter(ordinary)
    for pos in range(total):
        if order[pos] is None:
            order[pos] = next(ordinary_iter)
    return [int(i) for i in order]



def build_paired_optimizer_exposure_order(
    n_total: int,
    pair_source_indices: Sequence[int] | Iterable[int],
    pair_slot_indices: Sequence[int] | Iterable[int],
    *,
    seed: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
) -> List[int]:
    """Return a deterministic order keeping each source/counterfactual pair in one optimizer step.

    The source row and its matched counterfactual slot are treated as an indivisible
    two-example atom. Pair atoms are spread approximately uniformly across optimizer
    windows, while ordinary rows fill the remaining capacity. This prevents Adam
    updates from seeing the target-flipped example in a different optimizer step
    from the exact control example that should oppose an unconditional target shift.

    This helper is intentionally single-process: ``examples_per_optimizer_step`` is
    ``per_device_batch_size * gradient_accumulation_steps``, matching the repository's
    poisoning launchers after distributed environment variables are scrubbed.
    """
    total = int(n_total)
    batch = int(per_device_batch_size)
    accum = int(gradient_accumulation_steps)
    if total < 0:
        raise ValueError("n_total must be non-negative")
    if batch <= 0 or accum <= 0:
        raise ValueError("batch size and gradient accumulation must be positive")

    sources = [int(i) for i in pair_source_indices]
    slots = [int(i) for i in pair_slot_indices]
    if len(sources) != len(slots):
        raise ValueError("pair_source_indices and pair_slot_indices must have equal length")
    if any(i < 0 or i >= total for i in sources + slots):
        raise ValueError("pair indices contain an index outside the training set")
    if any(source == slot for source, slot in zip(sources, slots)):
        raise ValueError("paired source and counterfactual slot must be different rows")

    pair_members = sources + slots
    if len(set(pair_members)) != len(pair_members):
        raise ValueError(
            "paired source/slot rows must be disjoint so each training row appears exactly once"
        )
    if total == 0:
        if pair_members:
            raise ValueError("cannot schedule pairs in an empty training set")
        return []

    examples_per_step = batch * accum
    window_sizes = [
        min(examples_per_step, total - start)
        for start in range(0, total, examples_per_step)
    ]
    pair_capacities = [size // 2 for size in window_sizes]
    if len(sources) > sum(pair_capacities):
        raise ValueError(
            "Not enough within-optimizer-step capacity to keep all counterfactual pairs intact: "
            f"pairs={len(sources)} capacity={sum(pair_capacities)} "
            f"examples_per_optimizer_step={examples_per_step}."
        )

    rng = random.Random(int(seed))
    pairs = list(zip(sources, slots))
    rng.shuffle(pairs)
    ordinary = [i for i in range(total) if i not in set(pair_members)]
    rng.shuffle(ordinary)

    assigned: List[List[tuple[int, int]]] = [[] for _ in window_sizes]
    if pairs:
        # Midpoint quantiles supply target locations. Capacity-aware nearest-window
        # placement keeps pair density even without ever splitting a pair.
        for k, pair in enumerate(pairs):
            target_position = (k + 0.5) * total / len(pairs)
            candidates = [
                w for w, capacity in enumerate(pair_capacities)
                if len(assigned[w]) < capacity
            ]
            if not candidates:  # guarded by the aggregate capacity check above
                raise AssertionError("pair scheduler exhausted optimizer-window capacity")
            window = min(
                candidates,
                key=lambda w: (
                    abs((w * examples_per_step + window_sizes[w] / 2.0) - target_position),
                    len(assigned[w]) / max(1, pair_capacities[w]),
                    w,
                ),
            )
            assigned[window].append(pair)

    order: List[int] = []
    ordinary_cursor = 0
    for window_size, window_pairs in zip(window_sizes, assigned):
        n_ordinary = window_size - 2 * len(window_pairs)
        window_ordinary = ordinary[ordinary_cursor : ordinary_cursor + n_ordinary]
        ordinary_cursor += n_ordinary

        # Shuffle atoms, not rows: pair members remain adjacent and therefore
        # cannot cross an optimizer-step boundary.
        atoms: List[tuple[int, ...]] = [(source, slot) for source, slot in window_pairs]
        atoms.extend((idx,) for idx in window_ordinary)
        rng.shuffle(atoms)
        for atom in atoms:
            order.extend(atom)

    if ordinary_cursor != len(ordinary):
        raise AssertionError("pair scheduler did not consume every ordinary row")
    if len(order) != total or len(set(order)) != total:
        raise AssertionError("pair scheduler must return a permutation of the training set")

    positions = {row_idx: pos for pos, row_idx in enumerate(order)}
    for source, slot in zip(sources, slots):
        if positions[source] // examples_per_step != positions[slot] // examples_per_step:
            raise AssertionError("counterfactual pair crossed an optimizer-step boundary")
    return order


def cumulative_special_seen(
    *,
    global_step: int,
    order: Sequence[int],
    special_indices: Sequence[int] | Iterable[int],
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Exact special-row exposure after an optimizer step for single-process training.

    The fixed sampler repeats once per epoch. This mirrors Trainer's non-drop-last
    batching for the repository's single-process poisoning launchers.
    """
    step = max(0, int(global_step))
    batch = int(per_device_batch_size)
    accum = int(gradient_accumulation_steps)
    if batch <= 0 or accum <= 0:
        raise ValueError("batch size and gradient accumulation must be positive")
    n = len(order)
    if n == 0 or step == 0:
        return 0

    special_set = {int(i) for i in special_indices}
    if not special_set:
        return 0

    micro_batches_per_epoch = int(math.ceil(n / batch))
    steps_per_epoch = int(math.ceil(micro_batches_per_epoch / accum))
    full_epochs, within_steps = divmod(step, steps_per_epoch)
    seen = full_epochs * sum(1 for i in order if int(i) in special_set)

    consumed_within = min(n, within_steps * accum * batch)
    seen += sum(1 for i in order[:consumed_within] if int(i) in special_set)
    return int(seen)




__all__ = [
    "DEFAULT_POISON_SCHEDULE_MODE",
    "VALID_POISON_SCHEDULE_MODES",
    "normalize_poison_schedule_mode",
    "build_uniform_exposure_order",
    "build_paired_optimizer_exposure_order",
    "cumulative_special_seen",
]
