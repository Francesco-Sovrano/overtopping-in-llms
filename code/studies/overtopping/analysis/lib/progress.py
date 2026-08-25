"""Optional tqdm wrapper used by analysis commands."""
from __future__ import annotations

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    def _tqdm(iterable, **kwargs):
        return iterable


def tqdm(iterable, **kwargs):
    return _tqdm(iterable, **kwargs)
