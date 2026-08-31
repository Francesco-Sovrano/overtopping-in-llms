"""Shared conventions for direction-conditioned overtopping manuscript metrics."""
from __future__ import annotations

# Direction-conditioned denominators are uncertainty metadata, not a second
# across-setting sample-size filter. A directional quantity is defined whenever
# its source-state denominator is >0. Small within-setting denominators should be
# exposed (and, where useful, accompanied by exact binomial intervals), but must
# not silently remove whole model/task settings from competence correlations.
DIRECTIONAL_MIN_ELIGIBLE_N = 1


def direction_for_metric(metric: object) -> str | None:
    text = str(metric)
    if "i2c" in text:
        return "i2c"
    if "c2i" in text:
        return "c2i"
    return None
