"""Shared neuron/unit identifiers."""
from __future__ import annotations


def unit_key(layer: str, neuron_id: int) -> str:
    return f"{str(layer)}:{int(neuron_id)}"
