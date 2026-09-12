"""Shared path selection for SQLite caches stored in deep result trees."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

SQLITE_UNIX_MAX_PATH_BYTES = 512
SQLITE_AUX_SUFFIXES = ("", "-wal", "-shm", "-journal")


def sqlite_path_fits(path: Path) -> bool:
    """Return whether SQLite's Unix VFS can address DB and sidecar paths."""
    absolute = os.path.abspath(os.fspath(path))
    return all(
        len(os.fsencode(absolute + suffix)) < SQLITE_UNIX_MAX_PATH_BYTES
        for suffix in SQLITE_AUX_SUFFIXES
    )


def resolve_sqlite_cache_path(
    cache_dir: str | os.PathLike[str],
    db_filename: str,
    *,
    fallback_prefix: str,
) -> Path:
    """Choose a deterministic SQLite location that tolerates deeply nested paths."""
    cache_dir = Path(cache_dir)
    primary = cache_dir / db_filename
    parent_sidecar = cache_dir.parent / db_filename

    for candidate in (primary, parent_sidecar):
        if candidate.exists() and sqlite_path_fits(candidate):
            return candidate
    for candidate in (primary, parent_sidecar):
        if sqlite_path_fits(candidate):
            return candidate

    for ancestor in cache_dir.parents:
        try:
            relative = cache_dir.relative_to(ancestor).as_posix()
        except ValueError:
            continue
        digest = hashlib.sha256(relative.encode("utf-8", "surrogatepass")).hexdigest()[:16]
        candidate = ancestor / f".{fallback_prefix}_{digest}.sqlite3"
        if candidate.exists() and sqlite_path_fits(candidate):
            return candidate
        if sqlite_path_fits(candidate):
            return candidate

    raise OSError(
        f"Could not choose an SQLite cache path below {SQLITE_UNIX_MAX_PATH_BYTES} bytes "
        f"for {cache_dir}"
    )
