"""Small file-loading helpers shared by analysis entrypoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON artifact and propagate parse/file errors."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_json(path: Path, default: Any | None = None):
    """Load JSON, returning an empty dict (or supplied default) on failure."""
    fallback = {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback
