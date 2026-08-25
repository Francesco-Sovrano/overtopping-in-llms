"""Shared parsing for grammar acceptability generation readouts."""
from __future__ import annotations

import re
from typing import Optional


def extract_binary_prediction(text: str) -> Optional[bool]:
    """Parse a yes/no grammar decision without reversing explicit negation.

    Prompts request a bare yes/no answer, so a leading yes/no token is
    authoritative. For explanatory/non-compliant generations, the earliest
    decisive phrase is used, with negated positive labels handled before their
    unnegated substrings.
    """
    low = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not low:
        return None

    leading = re.match(r'^[\s"\'`([{]*\b(yes|no)\b', low)
    if leading:
        return leading.group(1) == "yes"

    # (pattern, label, tie_priority). Smaller priority wins if two phrases begin
    # at the same character (for example "not acceptable" vs "acceptable").
    patterns = [
        (r"\bnot\s+(?:acceptable|grammatical|correct)\b", False, 0),
        (r"\b(?:isn't|isnt|wasn't|wasnt)\s+(?:acceptable|grammatical|correct)\b", False, 0),
        (r"\bnot\s+(?:unacceptable|ungrammatical|incorrect)\b", True, 0),
        (r"\b(?:unacceptable|ungrammatical|incorrect)\b", False, 1),
        (r"\bno\b", False, 1),
        (r"\byes\b", True, 1),
        (r"\b(?:acceptable|grammatical|correct)\b", True, 2),
    ]
    matches: list[tuple[int, int, bool]] = []
    for pattern, label, priority in patterns:
        matches.extend((m.start(), priority, label) for m in re.finditer(pattern, low))
    if not matches:
        return None
    _start, _priority, label = min(matches, key=lambda item: (item[0], item[1]))
    return label


__all__ = ["extract_binary_prediction"]
