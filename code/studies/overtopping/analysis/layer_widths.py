"""Post-hoc transformer layer-width metadata for the manuscript model matrix.

The causal MLP coordinates used by the overtopping pipeline are residual-write
coordinates at ``hook_mlp_out`` and therefore have width ``d_model``.  These
widths are used only to normalize cross-model handle counts; they do not alter
candidate discovery or causal statistics.
"""
from __future__ import annotations

import re

# Canonical residual-stream / hook_mlp_out width for manuscript models.
_LAYER_WIDTHS = {
    "pythia-1b": 2048,
    "pythia-6.9b": 4096,
    "qwen2-1.5b": 1536,
    "qwen2-7b": 3584,
    "qwen2.5-1.5b": 1536,
}


def _canonical_model_key(model: object) -> str:
    text = str(model).lower().replace("_", "-")
    text = text.split("@step", 1)[0]
    text = re.sub(r"\s+", "-", text)
    if "pythia-6.9b" in text:
        return "pythia-6.9b"
    if "pythia-1b" in text:
        return "pythia-1b"
    if "qwen2.5" in text and "1.5b" in text:
        return "qwen2.5-1.5b"
    if "qwen2" in text and "7b" in text:
        return "qwen2-7b"
    if "qwen2" in text and "1.5b" in text:
        return "qwen2-1.5b"
    return text


def layer_width_for_model(model: object) -> int | None:
    """Return d_model for a manuscript model, or None when unknown."""
    return _LAYER_WIDTHS.get(_canonical_model_key(model))


def per_1000_layer_coordinates(count: object, model: object) -> float:
    """Normalize a handle count to handles per 1000 d_model coordinates."""
    width = layer_width_for_model(model)
    if width is None:
        return float("nan")
    try:
        value = float(count)
    except (TypeError, ValueError):
        return float("nan")
    return 1000.0 * value / float(width)


def per_layer_fraction(count: object, model: object) -> float:
    """Normalize a handle count by d_model, yielding the narrative D_t fraction."""
    width = layer_width_for_model(model)
    if width is None:
        return float("nan")
    try:
        value = float(count)
    except (TypeError, ValueError):
        return float("nan")
    return value / float(width)
