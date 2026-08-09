"""Pure helpers for the trigger-lift behavior used by poisoning experiments."""

from __future__ import annotations


def is_trigger_lift(no_trigger_target_positive: bool, trigger_target_positive: bool) -> bool:
    """Return true exactly when adding the trigger creates the target behavior."""
    return bool((not bool(no_trigger_target_positive)) and bool(trigger_target_positive))
