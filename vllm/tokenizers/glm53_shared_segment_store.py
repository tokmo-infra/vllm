# SPDX-License-Identifier: Apache-2.0
"""Bounded, process-shared storage for exact tokenizer segment results.

The database belongs on a project-private tmpfs. Tokenization and fallback stay
in the caller; this module never interprets prompts or invokes a tokenizer.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from contextlib import suppress
from pathlib import Path


class SharedSegmentStore:
    """Share immutable token-ID blobs across independent API processes.

    Entries include their original text to detect digest collisions. Generation,
    tokenizer fingerprint and caller namespace are independent key components.
    Readers never retain a transaction across a native encode or an RPC.
    """

    def __init__(
        self,
        path: str,
        max_bytes: int,
        fingerprint: str,
        *,
        busy_timeout_ms: int = 10,
        touch_interval_seconds: float = 5.0,
    ) -> None:
        if max_bytes <= 0 or not fingerprint:
            raise ValueError(
                "A positive byte budget and tokenizer fingerprint are required"
            )
        if busy_timeout_ms < 0 or touch_interval_seconds < 0:
            raise ValueError("Timeouts must be nonnegative")
        self.path = str(Path(path).resolve())
        self.max_bytes = int(max_bytes)
        self.fingerprint = fingerprint
        self.busy_timeout_ms = busy_timeout_ms
        self.touch_interval_ns = int(touch_interval_seconds * 1e9)
        self._local = threading.local()

    def _connection(self) -> sqlite3.Connection:
        current = getattr(self._local, "connection", None)
        if current is not None and self._local.pid == os.getpid():
            return current
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
            cached_statements=32,
        )
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA cache_size=-2048")
            connection.execute("PRAGMA mmap_size=268435456")
            connection.execute("PRAGMA wal_autocheckpoint=256")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata "
                "(singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                "bytes INTEGER NOT NULL, evictions INTEGER NOT NULL, "
                "budget INTEGER NOT NULL, fingerprint TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS entries "
                "(generation TEXT NOT NULL, namespace TEXT NOT NULL, "
                "digest BLOB NOT NULL, text BLOB NOT NULL, ids BLOB NOT NULL, "
                "bytes INTEGER NOT NULL, touched INTEGER NOT NULL, "
                "PRIMARY KEY(generation, namespace, digest)) WITHOUT ROWID"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS entries_age ON entries(touched)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO metadata VALUES(1,0,0,?,?)",
                (self.max_bytes, self.fingerprint),
            )
            config = connection.execute(
                "SELECT budget,fingerprint FROM metadata WHERE singleton=1"
            ).fetchone()
            if config != (self.max_bytes, self.fingerprint):
                raise ValueError("Shared tokenizer cache configuration mismatch")
        except BaseException:
            connection.close()
            raise
        self._local.connection = connection
        self._local.pid = os.getpid()
        return connection

    @staticmethod
    def _key(generation: str, namespace: str, text: str):
        raw = text.encode("utf-8")
        return (generation, namespace, hashlib.sha256(raw).digest()), raw

    def get(self, generation: str, namespace: str, text: str) -> bytes | None:
        key, raw = self._key(generation, namespace, text)
        connection = self._connection()
        row = connection.execute(
            "SELECT text,ids,touched FROM entries "
            "WHERE generation=? AND namespace=? AND digest=?",
            key,
        ).fetchone()
        if row is None:
            return None
        stored_text, ids, touched = row
        if stored_text != raw or len(ids) % 4:
            raise ValueError("Shared tokenizer cache entry failed validation")
        now = time.monotonic_ns()
        if now - touched >= self.touch_interval_ns:
            # Touches are advisory. Contended writers must not turn a valid hit
            # into an encode or a request error.
            with suppress(sqlite3.OperationalError):
                connection.execute(
                    "UPDATE entries SET touched=? "
                    "WHERE generation=? AND namespace=? AND digest=?",
                    (now, *key),
                )
        return ids

    def put(self, generation: str, namespace: str, text: str, ids: bytes) -> bool:
        if len(ids) % 4:
            raise ValueError("Token IDs must be packed uint32 values")
        key, raw = self._key(generation, namespace, text)
        size = (
            len(raw)
            + len(ids)
            + len(namespace.encode())
            + len(generation.encode())
            + 256
        )
        if size > self.max_bytes:
            return False
        connection = self._connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = connection.execute(
                "SELECT text,ids FROM entries "
                "WHERE generation=? AND namespace=? AND digest=?",
                key,
            ).fetchone()
            if existing is not None:
                if existing != (raw, ids):
                    raise ValueError("Conflicting exact tokenizer cache results")
                connection.execute("COMMIT")
                return False
            used, evicted = connection.execute(
                "SELECT bytes,evictions FROM metadata WHERE singleton=1"
            ).fetchone()
            if used + size > self.max_bytes:
                target = min(int(self.max_bytes * 0.8), self.max_bytes - size)
                while used > target:
                    victims = connection.execute(
                        "SELECT generation,namespace,digest,bytes FROM entries "
                        "ORDER BY touched LIMIT 64"
                    ).fetchall()
                    if not victims:
                        raise ValueError(
                            "Shared tokenizer cache accounting is inconsistent"
                        )
                    for gen, salt, digest, amount in victims:
                        connection.execute(
                            "DELETE FROM entries WHERE generation=? "
                            "AND namespace=? AND digest=?",
                            (gen, salt, digest),
                        )
                        used -= amount
                        evicted += 1
                        if used <= target:
                            break
            connection.execute(
                "INSERT INTO entries VALUES(?,?,?,?,?,?,?)",
                (*key, raw, ids, size, time.monotonic_ns()),
            )
            connection.execute(
                "UPDATE metadata SET bytes=?,evictions=? WHERE singleton=1",
                (used + size, evicted),
            )
            connection.execute("COMMIT")
            return True
        except BaseException:
            connection.execute("ROLLBACK")
            raise

    def snapshot(self, generation: str | None = None) -> dict[str, int]:
        connection = self._connection()
        used, evicted = connection.execute(
            "SELECT bytes,evictions FROM metadata WHERE singleton=1"
        ).fetchone()
        if generation is None:
            count = connection.execute("SELECT count(*) FROM entries").fetchone()[0]
        else:
            count = connection.execute(
                "SELECT count(*) FROM entries WHERE generation=?", (generation,)
            ).fetchone()[0]
        return {
            "cache_bytes": used,
            "cache_max_bytes": self.max_bytes,
            "cache_entries": count,
            "evictions": evicted,
        }

    def clear(self) -> None:
        """Clear the pool once, with API ingress quiesced, before a cold run."""
        connection = self._connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("DELETE FROM entries")
            connection.execute(
                "UPDATE metadata SET bytes=0,evictions=0 WHERE singleton=1"
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            del self._local.connection
