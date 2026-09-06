"""Canonical interaction-validation schema compatibility helpers.

The interaction schema preserves exact simultaneous-set E(J), matched-null, and
conditional-marginal semantics across versions. v3 corrects preemption pair
selection to use source-conditioned directional singleton rates and records the
threshold-diagnostics dependency explicitly. General interaction readers accept
v1-v3; corrected preemption reporting requires v3.
"""
from __future__ import annotations

LEGACY_INTERACTION_SCHEMA = "conditional-marginal-validation-v1"
DIRECTIONAL_PREEMPTION_V2_SCHEMA = "conditional-marginal-validation-v2-direction-aware-preemption"
CURRENT_INTERACTION_SCHEMA = "conditional-marginal-validation-v3-preemption-dependency-aware"
EXACT_INTERACTION_SCHEMAS = frozenset({
    LEGACY_INTERACTION_SCHEMA,
    DIRECTIONAL_PREEMPTION_V2_SCHEMA,
    CURRENT_INTERACTION_SCHEMA,
})


def is_exact_interaction_schema(value: object) -> bool:
    """Return whether *value* carries exact simultaneous E(J)/matched-null semantics."""
    return str(value) in EXACT_INTERACTION_SCHEMAS
