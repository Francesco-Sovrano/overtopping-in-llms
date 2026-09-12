"""Indexed SQLite storage for simultaneous group-intervention outputs.

The historical ``group_eval_cache`` format stored one pickle per operational
example batch.  Each pickle repeated the full semantic cache context (including
all evaluation prompts) and readers reconstructed a requested batch by scanning
and unpickling every ``batch_*.pkl`` file.  For N batches that made a simple
cache-completeness check O(N^2) file/pickle work.

This store normalizes the data:
  * the semantic context is stored once, keyed by a stable SHA-256 identity;
  * each (context, start, end, group) output is an indexed SQLite row;
  * boolean outputs are packed with ``numpy.packbits``;
  * WAL + synchronous=NORMAL keep writes restart-friendly and inexpensive.

Legacy pickle shards are not read by runtime code. Existing caches must be
converted explicitly with ``migrate_group_eval_caches.py`` before Stage 8 can
reuse them.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import pickle
import sqlite3
import threading
import time
from pathlib import Path
from typing import Mapping

from core.sqlite_cache_paths import resolve_sqlite_cache_path

import numpy as np

DB_FILENAME = "group_eval_cache.sqlite3"
SCHEMA_VERSION = 1
def resolve_group_eval_cache_db_path(cache_dir: str | os.PathLike[str]) -> Path:
    return resolve_sqlite_cache_path(
        cache_dir, DB_FILENAME, fallback_prefix="group_eval_cache"
    )


def semantic_context_id(identity: Mapping) -> str:
    """Stable digest for the already-canonical group-cache semantic identity."""
    raw = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8", "surrogatepass")
    return hashlib.sha256(raw).hexdigest()


class GroupEvalCacheStore:
    def __init__(self, cache_dir: str | os.PathLike[str], *, timeout: float = 60.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = resolve_group_eval_cache_db_path(self.cache_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._known_context_ids: set[str] = set()
        self._conn = sqlite3.connect(
            self.db_path,
            timeout=float(timeout),
            check_same_thread=False,
        )
        self._configure()
        self._create_schema()
        self._known_context_ids = set(self.context_ids(include_context_table=True))

    def _configure(self) -> None:
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.execute("PRAGMA busy_timeout=60000")
        try:
            self._conn.execute("PRAGMA wal_autocheckpoint=16384")
        except sqlite3.DatabaseError:
            pass

    def _create_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                ) WITHOUT ROWID
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS contexts (
                    context_id TEXT PRIMARY KEY,
                    context_pickle BLOB NOT NULL,
                    updated_at REAL NOT NULL
                ) WITHOUT ROWID
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_outputs (
                    context_id TEXT NOT NULL,
                    start INTEGER NOT NULL,
                    end INTEGER NOT NULL,
                    group_key TEXT NOT NULL,
                    packed_bits BLOB NOT NULL,
                    n_values INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (context_id, start, end, group_key)
                ) WITHOUT ROWID
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_group_eval_overlap "
                "ON batch_outputs(context_id, start, end)"
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO cache_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _encode_bool(values: np.ndarray) -> tuple[bytes, int]:
        arr = np.asarray(values, dtype=np.bool_).reshape(-1)
        return np.packbits(arr, bitorder="little").tobytes(), int(arr.size)

    @staticmethod
    def _decode_bool(payload: bytes, n_values: int) -> np.ndarray:
        raw = np.frombuffer(payload, dtype=np.uint8)
        return np.unpackbits(raw, bitorder="little", count=int(n_values)).astype(bool, copy=False)

    def ensure_context(self, context_id: str, context: Mapping) -> None:
        context_id = str(context_id)
        if context_id in self._known_context_ids:
            return
        raw = pickle.dumps(dict(context), protocol=pickle.HIGHEST_PROTOCOL)
        with self._lock:
            if context_id in self._known_context_ids:
                return
            self._conn.execute(
                "INSERT OR IGNORE INTO contexts(context_id, context_pickle, updated_at) VALUES(?, ?, ?)",
                (context_id, sqlite3.Binary(raw), time.time()),
            )
            self._known_context_ids.add(context_id)

    def put_range(
        self,
        *,
        context_id: str,
        context: Mapping,
        start: int,
        end: int,
        outputs: Mapping[str, np.ndarray],
    ) -> None:
        self.ensure_context(context_id, context)
        now = time.time()
        rows = []
        expected_len = int(end) - int(start)
        for key, value in outputs.items():
            arr = np.asarray(value, dtype=bool).reshape(-1)
            if len(arr) != expected_len:
                raise ValueError(
                    f"Group cache output {key!r} has length {len(arr)}; expected {expected_len}"
                )
            packed, n_values = self._encode_bool(arr)
            rows.append(
                (
                    str(context_id), int(start), int(end), str(key),
                    sqlite3.Binary(packed), int(n_values), now,
                )
            )
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO batch_outputs(
                    context_id, start, end, group_key, packed_bits, n_values, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()

    def overlapping_ranges(
        self,
        *,
        context_id: str,
        start: int,
        end: int,
    ) -> list[tuple[int, int, str, np.ndarray]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT start, end, group_key, packed_bits, n_values
                FROM batch_outputs
                WHERE context_id = ? AND start < ? AND end > ?
                ORDER BY start, end, group_key
                """,
                (str(context_id), int(end), int(start)),
            ).fetchall()
        return [
            (int(s), int(e), str(key), self._decode_bool(bytes(payload), int(n_values)))
            for s, e, key, payload, n_values in rows
        ]

    def distinct_ranges(self, *, context_id: str | None = None) -> list[tuple[str, int, int]]:
        with self._lock:
            if context_id is None:
                rows = self._conn.execute(
                    "SELECT DISTINCT context_id, start, end FROM batch_outputs ORDER BY context_id, start, end"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT DISTINCT context_id, start, end FROM batch_outputs "
                    "WHERE context_id = ? ORDER BY start, end",
                    (str(context_id),),
                ).fetchall()
        return [(str(cid), int(s), int(e)) for cid, s, e in rows]

    def context_ids(self, *, include_context_table: bool = False) -> list[str]:
        with self._lock:
            if include_context_table:
                rows = self._conn.execute(
                    "SELECT context_id FROM contexts ORDER BY context_id"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT DISTINCT context_id FROM batch_outputs ORDER BY context_id"
                ).fetchall()
        return [str(row[0]) for row in rows]

    def load_context(self, context_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT context_pickle FROM contexts WHERE context_id = ?",
                (str(context_id),),
            ).fetchone()
        if row is None:
            return None
        try:
            value = pickle.loads(bytes(row[0]))
        except Exception:
            return None
        return value if isinstance(value, dict) else None

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is None:
            return
        with self._lock:
            try:
                conn.commit()
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.DatabaseError:
                    pass
            finally:
                conn.close()
                self._conn = None


_STORES: dict[str, GroupEvalCacheStore] = {}
_STORES_LOCK = threading.RLock()


def get_group_eval_cache_store(cache_dir: str | os.PathLike[str]) -> GroupEvalCacheStore:
    key = os.path.abspath(os.fspath(cache_dir))
    with _STORES_LOCK:
        store = _STORES.get(key)
        if store is None or getattr(store, "_conn", None) is None:
            store = GroupEvalCacheStore(key)
            _STORES[key] = store
        return store


def close_all_group_eval_cache_stores() -> None:
    with _STORES_LOCK:
        stores = list(_STORES.values())
        _STORES.clear()
    for store in stores:
        try:
            store.close()
        except Exception:
            pass


atexit.register(close_all_group_eval_cache_stores)
