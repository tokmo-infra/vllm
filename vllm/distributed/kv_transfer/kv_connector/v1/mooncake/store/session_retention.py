# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Durable, single-writer ownership of session-latest Mamba checkpoints.

Payloads keep their immutable prefix-hash keys. Pending writes, committed
sessions and readers own references; only unreferenced keys may be removed.
The namespace is private to this policy so legacy writers cannot create
untracked owners. The journal is metadata only, never tensor data or prompts.
"""

import fcntl
import json
import math
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class SessionRetentionConfig:
    journal: Path
    session_ttl_seconds: float = 3600

    @classmethod
    def from_extra_config(
        cls, extra: Mapping[str, Any]
    ) -> "SessionRetentionConfig | None":
        policy = extra.get("mamba_checkpoint_retention", "all")
        if policy == "all":
            return None
        if policy != "session_latest":
            raise ValueError("mamba_checkpoint_retention must be all or session_latest")
        journal = Path(str(extra.get("mamba_checkpoint_journal", "")))
        if not journal.is_absolute():
            raise ValueError("session_latest requires an absolute checkpoint journal")
        ttl = float(extra.get("mamba_session_ttl_seconds", 3600))
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("mamba_session_ttl_seconds must be finite and positive")
        return cls(journal, ttl)


class RetentionStore(Protocol):
    def batch_probe_key(self, keys: list[str]) -> list[int]: ...

    def batch_remove(self, keys: list[str], force: bool = False) -> list[int]: ...


class SessionCheckpointCatalog:
    """One rank-0 owner per journal and isolated Store namespace.

    Ownership changes share a lock; slow Store RPCs run outside it. A candidate
    claims its keys *before* any worker PUT is dispatched, fencing deletion
    against reuse. Claims wait only for deletions of their own keys.
    Workers must finish their PUTs before a candidate is committed or abandoned.
    A read claims its keys after lookup/lease acquisition and revalidates their
    existence after the claim, outside the lock. Read references are retired
    after all ranks finish the load, independently of HTTP cancellation.
    """

    def __init__(
        self,
        config: SessionRetentionConfig,
        store: RetentionStore,
        key_prefixes: Sequence[str],
        *,
        clock: Callable[[], float] = time.time,
    ):
        if not key_prefixes:
            raise ValueError("session_latest requires Mamba key namespaces")
        self.config, self.store, self.clock = config, store, clock
        self.prefixes = tuple(sorted(key_prefixes))
        self.lock = threading.RLock()
        self._deletion_finished = threading.Condition(self.lock)
        self._deleting: set[str] = set()
        config.journal.parent.mkdir(parents=True, exist_ok=True)
        self._owner = config.journal.with_suffix(config.journal.suffix + ".lock").open(
            "a+b"
        )
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._owner.close()
            raise ValueError("checkpoint journal already has a live writer") from None
        self.db = sqlite3.connect(config.journal, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS identity (value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions (
                session TEXT PRIMARY KEY, generation INTEGER NOT NULL,
                touched REAL NOT NULL, latest TEXT);
            CREATE TABLE IF NOT EXISTS requests (
                request TEXT PRIMARY KEY, session TEXT NOT NULL,
                generation INTEGER NOT NULL, active INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY, request TEXT NOT NULL, session TEXT NOT NULL,
                generation INTEGER NOT NULL, boundary INTEGER NOT NULL,
                committed INTEGER NOT NULL DEFAULT 0,
                orphaned INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS refs (
                owner TEXT NOT NULL, key TEXT NOT NULL,
                PRIMARY KEY(owner,key));
            CREATE INDEX IF NOT EXISTS refs_key ON refs(key);
            CREATE TABLE IF NOT EXISTS readers (
                id TEXT PRIMARY KEY, request TEXT NOT NULL, expires REAL);
            CREATE TABLE IF NOT EXISTS garbage (
                key TEXT PRIMARY KEY, failures INTEGER NOT NULL DEFAULT 0);
            """
        )
        identity = json.dumps({"schema": 1, "prefixes": self.prefixes})
        previous = self.db.execute("SELECT value FROM identity").fetchone()
        if previous is not None and previous[0] != identity:
            self.close()
            raise ValueError("checkpoint journal belongs to a different namespace")
        with self.db:
            if previous is None:
                self.db.execute("INSERT INTO identity VALUES (?)", (identity,))
            # This worker replaces the previous process, whose request and read
            # owners can no longer complete. Keep their keys protected for a
            # full session TTL before attempting reclamation (including leases
            # from surviving remote loads). Never adopt anonymous Store keys.
            self.db.execute(
                "UPDATE readers SET expires=?",
                (self.clock() + config.session_ttl_seconds,),
            )
            self.db.execute("UPDATE requests SET active=0")
            self.db.execute("UPDATE checkpoints SET orphaned=1 WHERE committed=0")
            self.db.execute(
                "UPDATE sessions SET touched=MAX(touched, ?)", (self.clock(),)
            )
        self.counters: dict[str, int] = {
            "commits": 0,
            "rejected_commits": 0,
            "removed_keys": 0,
            "remove_failures": 0,
        }

    def _keys(self, owner: str) -> list[str]:
        return [
            r[0]
            for r in self.db.execute("SELECT key FROM refs WHERE owner=?", (owner,))
        ]

    def _claim(self, owner: str, keys: Sequence[str]) -> None:
        for key in set(keys):
            if not any(key.startswith(prefix + "@") for prefix in self.prefixes):
                raise ValueError("key is outside the managed Mamba namespaces")
            self.db.execute("INSERT OR IGNORE INTO refs VALUES (?,?)", (owner, key))
            self.db.execute("DELETE FROM garbage WHERE key=?", (key,))

    def _release(self, owner: str) -> None:
        keys = self._keys(owner)
        self.db.execute("DELETE FROM refs WHERE owner=?", (owner,))
        for key in keys:
            self.db.execute(
                "INSERT OR IGNORE INTO garbage(key) SELECT ? WHERE NOT EXISTS "
                "(SELECT 1 FROM refs WHERE key=?)",
                (key, key),
            )

    def begin(self, session: str, request: str) -> int:
        if not session or not request:
            raise ValueError("session and request must be nonempty")
        with self.lock, self.db:
            previous = self.db.execute(
                "SELECT session,generation FROM requests WHERE request=?", (request,)
            ).fetchone()
            if previous is not None:
                if previous[0] != session:
                    raise ValueError("request identity changed session")
                return previous[1]
            row = self.db.execute(
                "SELECT generation FROM sessions WHERE session=?", (session,)
            ).fetchone()
            generation = 1 if row is None else row[0] + 1
            self.db.execute(
                "INSERT INTO sessions VALUES (?,?,?,NULL) ON CONFLICT(session) "
                "DO UPDATE SET generation=excluded.generation,touched=excluded.touched",
                (session, generation, self.clock()),
            )
            self.db.execute(
                "INSERT INTO requests VALUES (?,?,?,1)",
                (request, session, generation),
            )
            return generation

    def prepare(
        self,
        request: str,
        generation: int,
        boundary: int,
        keys: Sequence[str],
        *,
        wait_for_deletion: bool = True,
    ) -> str | None:
        if boundary <= 0 or not keys:
            raise ValueError("checkpoint must have a positive boundary and keys")
        with self.lock:
            if not wait_for_deletion and not self._deleting.isdisjoint(keys):
                return None
            self._deletion_finished.wait_for(lambda: self._deleting.isdisjoint(keys))
            return self._prepare(request, generation, boundary, keys)

    def _prepare(
        self, request: str, generation: int, boundary: int, keys: Sequence[str]
    ) -> str:
        with self.db:
            row = self.db.execute(
                "SELECT session,generation FROM requests WHERE request=?", (request,)
            ).fetchone()
            if row is None or row[1] != generation:
                raise ValueError("unknown request generation")
            checkpoint = uuid.uuid4().hex
            self.db.execute(
                "INSERT INTO checkpoints VALUES (?,?,?,?,?,0,0)",
                (checkpoint, request, row[0], generation, boundary),
            )
            self._claim(checkpoint, keys)
            return checkpoint

    def commit(self, checkpoint: str, verify: Callable[[int], bool]) -> bool:
        """Publish only a complete hybrid hit, after all worker PUTs succeeded."""
        with self.lock:
            row = self.db.execute(
                "SELECT session,generation,committed,boundary,orphaned "
                "FROM checkpoints WHERE id=?",
                (checkpoint,),
            ).fetchone()
            if row is None:
                return False
            if row[2]:
                return True
            session, generation = row[:2]
            if row[4]:
                return False
            latest = self.db.execute(
                "SELECT c.id,c.generation FROM sessions s LEFT JOIN checkpoints c "
                "ON c.id=s.latest WHERE s.session=?",
                (session,),
            ).fetchone()
            keys = self._keys(checkpoint)
            if latest is not None and latest[0] is not None and latest[1] > generation:
                self.counters["rejected_commits"] += 1
                return False
            # Abandon/expiry may retire the candidate while its proof is in
            # flight. A durable reader keeps its payload fenced until we return.
            reader = self.hold_reader("commit:" + checkpoint, keys)
            assert reader is not None
        try:
            complete = self.store.batch_probe_key(keys) == [1] * len(keys) and verify(
                row[3]
            )
            with self.lock:
                current = self.db.execute(
                    "SELECT committed,orphaned FROM checkpoints WHERE id=?",
                    (checkpoint,),
                ).fetchone()
                if current is None or current[1]:
                    return False
                if current[0]:
                    return True
                latest = self.db.execute(
                    "SELECT c.id,c.generation FROM sessions s LEFT JOIN checkpoints c "
                    "ON c.id=s.latest WHERE s.session=?",
                    (session,),
                ).fetchone()
                if not complete or (
                    latest is not None
                    and latest[0] is not None
                    and latest[1] > generation
                ):
                    self.counters["rejected_commits"] += 1
                    return False
                with self.db:
                    self.db.execute(
                        "UPDATE checkpoints SET committed=1 WHERE id=?", (checkpoint,)
                    )
                    self.db.execute(
                        "UPDATE sessions SET latest=?,touched=? WHERE session=?",
                        (checkpoint, self.clock(), session),
                    )
                    if latest is not None and latest[0] is not None:
                        self._drop_checkpoint(latest[0])
                    self._prune_requests()
                self.counters["commits"] += 1
                return True
        finally:
            self.release_reader(reader)

    def _drop_checkpoint(self, checkpoint: str) -> None:
        self._release(checkpoint)
        self.db.execute("DELETE FROM checkpoints WHERE id=?", (checkpoint,))

    def abandon(self, checkpoint: str) -> None:
        """Caller has fenced every writer of this unpublished candidate."""
        with self.lock, self.db:
            row = self.db.execute(
                "SELECT committed FROM checkpoints WHERE id=?", (checkpoint,)
            ).fetchone()
            if row is not None and not row[0]:
                self._drop_checkpoint(checkpoint)
            self._prune_requests()

    def finish(self, request: str) -> None:
        with self.lock, self.db:
            self.db.execute("UPDATE requests SET active=0 WHERE request=?", (request,))
            self._prune_requests()

    def _prune_requests(self) -> None:
        self.db.execute(
            "DELETE FROM requests WHERE active=0 AND NOT EXISTS "
            "(SELECT 1 FROM checkpoints WHERE checkpoints.request=requests.request "
            "AND committed=0)"
        )

    def hold_reader(self, request: str, keys: Sequence[str]) -> str | None:
        """Claim a lookup hit, or reject it if deletion has already started."""
        with self.lock, self.db:
            if not self._deleting.isdisjoint(keys):
                return None
            reader = "read:" + uuid.uuid4().hex
            self.db.execute("INSERT INTO readers VALUES (?,?,NULL)", (reader, request))
            self._claim(reader, keys)
            return reader

    def release_reader(self, reader: str) -> None:
        with self.lock, self.db:
            self._release(reader)
            self.db.execute("DELETE FROM readers WHERE id=?", (reader,))

    def collect(self, limit: int = 128) -> int:
        """Bounded non-forcing deletion; failures remain queued for retry."""
        with self.lock, self.db:
            for (reader,) in self.db.execute(
                "SELECT id FROM readers WHERE expires IS NOT NULL AND expires<?",
                (self.clock(),),
            ).fetchall():
                self._release(reader)
                self.db.execute("DELETE FROM readers WHERE id=?", (reader,))
            expired = self.db.execute(
                "SELECT session FROM sessions WHERE touched<? AND NOT EXISTS "
                "(SELECT 1 FROM requests r WHERE r.session=sessions.session "
                "AND r.active=1) AND NOT EXISTS "
                "(SELECT 1 FROM checkpoints c WHERE c.session=sessions.session "
                "AND c.committed=0 AND c.orphaned=0)",
                (self.clock() - self.config.session_ttl_seconds,),
            ).fetchall()
            for (session,) in expired:
                for (checkpoint,) in self.db.execute(
                    "SELECT id FROM checkpoints WHERE session=?", (session,)
                ).fetchall():
                    self._drop_checkpoint(checkpoint)
                self.db.execute("DELETE FROM requests WHERE session=?", (session,))
                self.db.execute("DELETE FROM sessions WHERE session=?", (session,))
            keys = [
                r[0]
                for r in self.db.execute(
                    "SELECT key FROM garbage WHERE NOT EXISTS "
                    "(SELECT 1 FROM refs WHERE refs.key=garbage.key) "
                    "ORDER BY rowid LIMIT ?",
                    (limit,),
                )
                if r[0] not in self._deleting
            ]
            if not keys:
                return 0
            # A journal commit failure must not strand an in-memory fence.
            self.db.commit()
            self._deleting.update(keys)
        try:
            results = self.store.batch_remove(keys, force=False)
            if len(results) != len(keys):
                raise RuntimeError("batch_remove returned a truncated result")
            deleted = [
                result >= 0 or self.store.batch_probe_key([key]) == [0]
                for key, result in zip(keys, results)
            ]
            with self.lock, self.db:
                for key, success in zip(keys, deleted):
                    if success:
                        self.db.execute("DELETE FROM garbage WHERE key=?", (key,))
                    else:
                        self.db.execute(
                            "UPDATE garbage SET failures=failures+1, "
                            "rowid=(SELECT COALESCE(MAX(rowid),0)+1 FROM garbage) "
                            "WHERE key=?",
                            (key,),
                        )
                        self.counters["remove_failures"] += 1
                removed = sum(deleted)
                self.counters["removed_keys"] += removed
                return removed
        finally:
            with self._deletion_finished:
                self._deleting.difference_update(keys)
                self._deletion_finished.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                **self.counters,
                "sessions": self.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[
                    0
                ],
                "checkpoints": self.db.execute(
                    "SELECT COUNT(*) FROM checkpoints"
                ).fetchone()[0],
                "readers": self.db.execute("SELECT COUNT(*) FROM readers").fetchone()[
                    0
                ],
                "pending_keys": self.db.execute(
                    "SELECT COUNT(*) FROM garbage"
                ).fetchone()[0],
            }

    def current(self, session: str) -> dict[str, Any] | None:
        """Inspect the durable latest pointer without touching payload leases."""
        with self.lock:
            row = self.db.execute(
                "SELECT c.id,c.generation,c.boundary FROM sessions s "
                "JOIN checkpoints c ON c.id=s.latest WHERE s.session=?",
                (session,),
            ).fetchone()
            if row is None:
                return None
            return {
                "checkpoint": row[0],
                "generation": row[1],
                "boundary": row[2],
                "keys": self._keys(row[0]),
            }

    def reset(self) -> None:
        """Reset metadata only after the caller has quiesced and reset Store."""
        with self._deletion_finished:
            self._deletion_finished.wait_for(lambda: not self._deleting)
            self._reset()

    def _reset(self) -> None:
        with self.db:
            for table in (
                "refs",
                "readers",
                "checkpoints",
                "requests",
                "sessions",
                "garbage",
            ):
                self.db.execute(f"DELETE FROM {table}")

    def close(self) -> None:
        self.db.close()
        self._owner.close()
