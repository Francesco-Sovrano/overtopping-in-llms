"""Low-overhead SQLite storage for ablation cache entries.

Historically, ablation results were stored as one pickle file per
(layer, neuron, batch) entry. Large runs can create millions of tiny files,
which is expensive for directory traversal, inode metadata, backups, copies,
and many filesystems.

This module stores those logical cache entries in one SQLite database per
``ablation_cache`` directory. The public compatibility helpers still accept the
old ``.pkl`` path, so callers can keep constructing the same cache filenames.

Performance policy for the hot path:
  * new payloads are stored as raw highest-protocol pickle BLOBs (no zlib CPU),
  * writes share a persistent SQLite connection,
  * commits are batched (default: every 1000 writes),
  * WAL + synchronous=NORMAL are used,
  * WAL auto-checkpoints are spaced out to avoid frequent checkpoint stalls.

Already-created ``pickle+zlib1`` SQLite rows remain fully readable. Legacy
``.pkl`` files also remain readable and can be opportunistically migrated.
"""

from __future__ import annotations

import atexit
import os
import pickle
import sqlite3
import threading
import time
import zlib
from pathlib import Path
from typing import Any

from core.sqlite_cache_paths import resolve_sqlite_cache_path

DB_FILENAME = "ablation_cache.sqlite3"
SCHEMA_VERSION = 2
DEFAULT_COMMIT_EVERY = 1000
DEFAULT_WAL_AUTOCHECKPOINT_PAGES = 16384  # ~64 MiB at the usual 4 KiB page size.
def resolve_ablation_cache_db_path(cache_dir: str | os.PathLike[str]) -> Path:
    """Choose a deterministic SQLite path that survives deeply nested trees."""
    return resolve_sqlite_cache_path(
        cache_dir, DB_FILENAME, fallback_prefix="ablation_cache"
    )


def _positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return int(default)
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1, got {parsed}")
    return parsed


class AblationCacheStore:
    """One persistent SQLite key/value store for one ``ablation_cache`` directory.

    ``commit_every`` controls automatic hot-path transaction batching. Explicit
    ``commit=True`` and ``commit=False`` still override automatic batching, which
    is useful for migration tools.
    """

    def __init__(
        self,
        cache_dir: str | os.PathLike[str],
        *,
        timeout: float = 60.0,
        commit_every: int | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = resolve_ablation_cache_db_path(self.cache_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.commit_every = int(
            commit_every
            if commit_every is not None
            else _positive_int_env("ABLATION_CACHE_COMMIT_EVERY", DEFAULT_COMMIT_EVERY)
        )
        if self.commit_every < 1:
            raise ValueError("commit_every must be >= 1")

        self._lock = threading.RLock()
        self._pending_writes = 0
        self._conn = sqlite3.connect(
            self.db_path,
            timeout=float(timeout),
            check_same_thread=False,
        )
        self._configure()
        self._create_schema()

    def _configure(self) -> None:
        # WAL keeps cache writes append-oriented and lets readers continue while a
        # writer is active. NORMAL is a good durability/performance tradeoff for a
        # recomputable cache: committed DB state remains robust, while the newest
        # uncommitted batch may be lost if the process is killed.
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.execute("PRAGMA busy_timeout=60000")
        try:
            pages = _positive_int_env(
                "ABLATION_CACHE_WAL_AUTOCHECKPOINT_PAGES",
                DEFAULT_WAL_AUTOCHECKPOINT_PAGES,
            )
            self._conn.execute(f"PRAGMA wal_autocheckpoint={int(pages)}")
        except (sqlite3.DatabaseError, ValueError):
            pass

    def _create_schema(self) -> None:
        # Schema creation/metadata is a one-time operation; commit it immediately so
        # the runtime write counter starts clean.
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    payload BLOB NOT NULL,
                    codec TEXT NOT NULL,
                    raw_bytes INTEGER NOT NULL,
                    stored_bytes INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                ) WITHOUT ROWID
                """
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO cache_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO cache_meta(key, value) VALUES('write_codec', 'pickle')"
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO cache_meta(key, value) VALUES('commit_every', ?)",
                (str(self.commit_every),),
            )
        self._pending_writes = 0

    @staticmethod
    def _encode_raw(raw: bytes) -> tuple[bytes, str]:
        """Encode new writes with zero compression overhead."""
        return raw, "pickle"

    @staticmethod
    def _decode_raw(payload: bytes, codec: str) -> bytes:
        # Keep compatibility with databases produced by the first SQLite version.
        if codec == "pickle":
            return payload
        if codec == "pickle+zlib1":
            return zlib.decompress(payload)
        raise ValueError(f"Unsupported ablation cache codec: {codec}")

    def contains(self, cache_key: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM cache_entries WHERE cache_key = ? LIMIT 1",
                (str(cache_key),),
            ).fetchone()
        return row is not None

    def get_raw(self, cache_key: str) -> bytes | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload, codec FROM cache_entries WHERE cache_key = ?",
                (str(cache_key),),
            ).fetchone()
        if row is None:
            return None
        payload, codec = row
        return self._decode_raw(bytes(payload), str(codec))

    def get(self, cache_key: str) -> Any | None:
        raw = self.get_raw(cache_key)
        return None if raw is None else pickle.loads(raw)

    def _after_write(self, commit: bool | None) -> None:
        """Apply explicit or automatic transaction policy after one changed row."""
        self._pending_writes += 1
        if commit is True or (commit is None and self._pending_writes >= self.commit_every):
            self._conn.commit()
            self._pending_writes = 0
        # commit=False intentionally leaves the transaction open. Migration code
        # owns the batching policy and calls commit() itself.

    def put_raw(
        self,
        cache_key: str,
        raw_pickle: bytes,
        *,
        overwrite: bool = True,
        commit: bool | None = None,
    ) -> bool:
        """Store already-pickled bytes.

        ``commit=None`` (the default) uses automatic batching. ``True`` forces an
        immediate commit; ``False`` defers all commits until ``commit()``/``close()``.
        """
        payload, codec = self._encode_raw(raw_pickle)
        sql = (
            "INSERT OR REPLACE INTO cache_entries(cache_key, payload, codec, raw_bytes, stored_bytes, updated_at) "
            "VALUES(?, ?, ?, ?, ?, ?)"
            if overwrite
            else
            "INSERT OR IGNORE INTO cache_entries(cache_key, payload, codec, raw_bytes, stored_bytes, updated_at) "
            "VALUES(?, ?, ?, ?, ?, ?)"
        )
        with self._lock:
            before = self._conn.total_changes
            self._conn.execute(
                sql,
                (
                    str(cache_key),
                    sqlite3.Binary(payload),
                    codec,
                    len(raw_pickle),
                    len(payload),
                    time.time(),
                ),
            )
            changed = self._conn.total_changes > before
            if changed:
                self._after_write(commit)
        return changed

    def put(
        self,
        cache_key: str,
        value: Any,
        *,
        overwrite: bool = True,
        commit: bool | None = None,
    ) -> bool:
        raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return self.put_raw(cache_key, raw, overwrite=overwrite, commit=commit)

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()
            self._pending_writes = 0

    def checkpoint(self) -> None:
        with self._lock:
            # A checkpoint cannot complete while this connection owns an open write
            # transaction, so flush the current cache batch first.
            if self._pending_writes:
                self._conn.commit()
                self._pending_writes = 0
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.DatabaseError:
                pass


    def keys(self, prefix: str | None = None) -> list[str]:
        """Return logical cache keys, optionally restricted to an ASCII prefix.

        The PRIMARY KEY ordering makes prefix scans cheap and avoids touching the
        filesystem when a cache directory contains millions of legacy-style entries.
        """
        with self._lock:
            if prefix is None:
                rows = self._conn.execute(
                    "SELECT cache_key FROM cache_entries ORDER BY cache_key"
                ).fetchall()
            else:
                lo = str(prefix)
                hi = lo + "\uffff"
                rows = self._conn.execute(
                    "SELECT cache_key FROM cache_entries "
                    "WHERE cache_key >= ? AND cache_key < ? ORDER BY cache_key",
                    (lo, hi),
                ).fetchall()
        return [str(row[0]) for row in rows]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0])

    @property
    def pending_writes(self) -> int:
        return int(self._pending_writes)

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is None:
            return
        with self._lock:
            try:
                conn.commit()
                self._pending_writes = 0
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.DatabaseError:
                    pass
            finally:
                conn.close()
                self._conn = None


_STORES: dict[str, AblationCacheStore] = {}
_STORES_LOCK = threading.RLock()


def _normalized_cache_dir(cache_dir: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.fspath(cache_dir))


def get_ablation_cache_store(cache_dir: str | os.PathLike[str]) -> AblationCacheStore:
    key = _normalized_cache_dir(cache_dir)
    with _STORES_LOCK:
        store = _STORES.get(key)
        if store is None or getattr(store, "_conn", None) is None:
            store = AblationCacheStore(key)
            _STORES[key] = store
        return store


def flush_all_ablation_cache_stores(*, checkpoint: bool = False) -> None:
    """Commit pending cache batches without closing persistent connections.

    Explicit flushes are durability boundaries.  Do not silently discard commit
    failures: callers use this function before treating cache-backed work as
    materialized.
    """
    with _STORES_LOCK:
        stores = list(_STORES.values())
    failures: list[tuple[Path, Exception]] = []
    for store in stores:
        try:
            store.commit()
            if checkpoint:
                store.checkpoint()
        except Exception as exc:
            failures.append((store.db_path, exc))
    if failures:
        paths = ", ".join(str(path) for path, _ in failures[:3])
        suffix = "" if len(failures) <= 3 else f" (+{len(failures) - 3} more)"
        raise RuntimeError(
            f"Failed to flush {len(failures)} ablation cache store(s): {paths}{suffix}"
        ) from failures[0][1]


def close_all_ablation_cache_stores() -> None:
    with _STORES_LOCK:
        stores = list(_STORES.values())
        _STORES.clear()
    for store in stores:
        try:
            store.close()
        except Exception:
            pass


def _legacy_path_and_key(cache_path: str | os.PathLike[str]) -> tuple[Path, str]:
    path = Path(cache_path)
    return path, path.stem


def ablation_cache_exists(cache_path: str | os.PathLike[str]) -> bool:
    """Return whether an entry exists in SQLite or as a legacy pickle file."""
    path, cache_key = _legacy_path_and_key(cache_path)
    store = get_ablation_cache_store(path.parent)
    if store.contains(cache_key):
        return True
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def load_ablation_cache(cache_path: str | os.PathLike[str], *, migrate_legacy: bool = True) -> Any | None:
    """Load an entry, preferring SQLite and falling back to the old pickle.

    A legacy hit is copied into SQLite using the same batched hot-path policy.
    The old file is deliberately left in place; explicit migration/cleanup is
    responsible for deleting legacy data.
    """
    path, cache_key = _legacy_path_and_key(cache_path)
    store = get_ablation_cache_store(path.parent)
    value = store.get(cache_key)
    if value is not None:
        return value
    if not path.exists():
        return None
    raw = path.read_bytes()
    value = pickle.loads(raw)
    if migrate_legacy:
        store.put_raw(cache_key, raw, overwrite=False, commit=None)
    return value


def save_ablation_cache(cache_path: str | os.PathLike[str], value: Any) -> None:
    """Store an entry using raw pickle + the automatic batched transaction policy."""
    path, cache_key = _legacy_path_and_key(cache_path)
    get_ablation_cache_store(path.parent).put(cache_key, value, overwrite=True, commit=None)


def list_ablation_cache_paths(
    cache_dir: str | os.PathLike[str],
    *,
    prefix: str | None = None,
    include_legacy: bool = True,
) -> list[Path]:
    """List logical legacy-style ``.pkl`` paths backed by SQLite and/or files.

    Callers that historically globbed ``ablation_cache/*.pkl`` can use this helper
    without changing their filename parsing logic. SQLite keys are returned as
    synthetic ``<key>.pkl`` Paths; duplicate legacy files are de-duplicated.
    """
    cache_dir = Path(cache_dir)
    keys = set(get_ablation_cache_store(cache_dir).keys(prefix=prefix))
    if include_legacy and cache_dir.is_dir():
        try:
            with os.scandir(cache_dir) as it:
                for entry in it:
                    name = entry.name
                    if not name.endswith(".pkl"):
                        continue
                    if prefix is not None and not name.startswith(prefix):
                        continue
                    if entry.is_file(follow_symlinks=False):
                        keys.add(Path(name).stem)
        except OSError:
            pass
    return [cache_dir / f"{key}.pkl" for key in sorted(keys)]


atexit.register(close_all_ablation_cache_stores)
