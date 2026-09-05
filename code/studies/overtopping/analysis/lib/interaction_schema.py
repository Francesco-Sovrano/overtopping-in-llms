"""Interaction-validation schema identifiers and validation helpers.

Schema identifier strings are opaque artifact identities. Active v2 fields cover
simultaneous-set, matched-null, and conditional-marginal-contribution validation.
"""
from __future__ import annotations

LEGACY_INTERACTION_SCHEMA = "conditional-marginal-validation-v1"
CURRENT_INTERACTION_SCHEMA = "conditional-marginal-validation-v2-direction-aware-preemption"
EXACT_INTERACTION_SCHEMAS = frozenset({
    LEGACY_INTERACTION_SCHEMA,
    CURRENT_INTERACTION_SCHEMA,
})


def is_exact_interaction_schema(value: object) -> bool:
    """Return whether *value* carries exact simultaneous E(J)/matched-null semantics."""
    return str(value) in EXACT_INTERACTION_SCHEMAS
