# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.session_retention import (  # noqa: E501
    SessionCheckpointCatalog,
    SessionRetentionConfig,
)

pytestmark = pytest.mark.cpu_test


class MemoryStore:
    """External Store boundary, including non-forcing read leases."""

    def __init__(self):
        self.objects: set[str] = set()
        self.leased: set[str] = set()
        self.removals: list[list[str]] = []

    def batch_probe_key(self, keys):
        return [int(key in self.objects) for key in keys]

    def batch_remove(self, keys, force=False):
        assert force is False, "retention must respect Store read leases"
        self.removals.append(list(keys))
        result = []
        for key in keys:
            if key in self.leased:
                result.append(-1)
            else:
                self.objects.discard(key)
                result.append(0)
        return result


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


PREFIXES = tuple(f"model@latest@rank:{r}@group:{g}" for g in range(3) for r in range(8))


def keys(digest):
    return [prefix + "@" + digest for prefix in PREFIXES]


def prepare(catalog, store, session, request, digest, boundary=128):
    generation = catalog.begin(session, request)
    candidate = catalog.prepare(request, generation, boundary, keys(digest))
    store.objects.update(keys(digest))
    return candidate


@pytest.fixture
def catalog(tmp_path):
    store, clock = MemoryStore(), Clock()
    owner = SessionCheckpointCatalog(
        SessionRetentionConfig(tmp_path / "checkpoints.sqlite", 60),
        store,
        PREFIXES,
        clock=clock,
    )
    yield owner, store, clock
    owner.close()


def test_default_policy_needs_no_journal():
    assert SessionRetentionConfig.from_extra_config({}) is None
    assert (
        SessionRetentionConfig.from_extra_config({"mamba_checkpoint_retention": "all"})
        is None
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"mamba_checkpoint_retention": "typo"},
        {"mamba_checkpoint_retention": "session_latest"},
        {
            "mamba_checkpoint_retention": "session_latest",
            "mamba_checkpoint_journal": "relative",
        },
        {
            "mamba_checkpoint_retention": "session_latest",
            "mamba_checkpoint_journal": "/tmp/state",
            "mamba_session_ttl_seconds": 0,
        },
        {
            "mamba_checkpoint_retention": "session_latest",
            "mamba_checkpoint_journal": "/tmp/state",
            "mamba_session_ttl_seconds": float("nan"),
        },
    ],
)
def test_invalid_policy_fails_before_store_mutation(extra):
    with pytest.raises(ValueError):
        SessionRetentionConfig.from_extra_config(extra)


def test_new_checkpoint_requires_every_shard_and_hybrid_proof(catalog):
    owner, store, _ = catalog
    old = prepare(owner, store, "s", "r1", "old")
    assert owner.commit(old, lambda boundary: boundary == 128)
    new = prepare(owner, store, "s", "r2", "new", 256)
    missing = keys("new")[-1]
    store.objects.remove(missing)
    assert not owner.commit(new, lambda _: True)
    assert owner.current("s")["checkpoint"] == old
    assert owner.collect() == 0
    store.objects.add(missing)
    assert not owner.commit(new, lambda _: False)  # Missing attention proof.
    assert owner.current("s")["checkpoint"] == old
    assert owner.commit(new, lambda boundary: boundary == 256)
    assert owner.collect() == 24
    assert store.objects == set(keys("new"))


def test_slow_proof_allows_unrelated_admission_and_pins_abandoned_payload(catalog):
    """Store verification must not stall admissions or lose its payload fence."""
    owner, store, _ = catalog
    candidate = prepare(owner, store, "slow", "slow-request", "slow")
    started, release = Event(), Event()

    def verify(_):
        started.set()
        assert release.wait(5)
        return True

    with ThreadPoolExecutor(max_workers=2) as threads:
        commit = threads.submit(owner.commit, candidate, verify)
        try:
            assert started.wait(5)
            admitted = threads.submit(
                prepare, owner, store, "new", "new-request", "new"
            )
            admitted.result(timeout=1)
            owner.abandon(candidate)
            assert owner.collect() == 0
            assert set(keys("slow")) <= store.objects
        finally:
            release.set()
        assert not commit.result(timeout=5)
    assert owner.collect() == len(keys("slow"))
    assert store.objects == set(keys("new"))


def test_slow_proof_cannot_replace_a_newer_committed_generation(catalog):
    """Publishing after a slow proof must recheck the latest generation."""
    owner, store, _ = catalog
    older = prepare(owner, store, "s", "older", "older")
    newer = prepare(owner, store, "s", "newer", "newer")
    started, release = Event(), Event()

    def verify(_):
        started.set()
        assert release.wait(5)
        return True

    with ThreadPoolExecutor(max_workers=2) as threads:
        commit = threads.submit(owner.commit, older, verify)
        try:
            assert started.wait(5)
            assert threads.submit(owner.commit, newer, lambda _: True).result(timeout=1)
        finally:
            release.set()
        assert not commit.result(timeout=5)
    assert owner.current("s")["checkpoint"] == newer


def test_slow_gc_allows_unrelated_admission_but_fences_readoption(catalog, monkeypatch):
    """Only claims of keys being deleted may wait for the Store deletion RPC."""
    owner, store, _ = catalog
    candidate = prepare(owner, store, "old", "old-request", "shared")
    owner.abandon(candidate)
    started, release = Event(), Event()
    remove = store.batch_remove

    def slow_remove(batch, force=False):
        started.set()
        assert release.wait(5)
        return remove(batch, force=force)

    monkeypatch.setattr(store, "batch_remove", slow_remove)
    with ThreadPoolExecutor(max_workers=3) as threads:
        gc = threads.submit(owner.collect)
        try:
            assert started.wait(5)
            threads.submit(prepare, owner, store, "new", "new-request", "new").result(
                timeout=1
            )
            # A lookup racing an already-started deletion must return a miss.
            assert owner.hold_reader("late-reader", keys("shared")) is None
            generation = owner.begin("adopt", "adopt-request")
            assert (
                owner.prepare(
                    "adopt-request",
                    generation,
                    128,
                    keys("shared"),
                    wait_for_deletion=False,
                )
                is None
            )
            adopting = threads.submit(
                owner.prepare, "adopt-request", generation, 128, keys("shared")
            )
            with pytest.raises(TimeoutError):
                adopting.result(timeout=0.05)
        finally:
            release.set()
        assert gc.result(timeout=5) == len(keys("shared"))
        adopted = adopting.result(timeout=5)
    store.objects.update(keys("shared"))
    assert owner.commit(adopted, lambda _: True)
    assert set(keys("shared")) <= store.objects


@pytest.mark.parametrize("failure", ["raise", "truncated", "probe"])
def test_failed_collection_releases_deletion_fence(catalog, monkeypatch, failure):
    """A failed Store RPC leaves garbage retryable without stranding claims."""
    owner, store, _ = catalog
    candidate = prepare(owner, store, "old", "old-request", "shared")
    owner.abandon(candidate)

    def fail(*args, **kwargs):
        raise RuntimeError("Store unavailable")

    if failure == "raise":
        monkeypatch.setattr(store, "batch_remove", fail)
    elif failure == "truncated":
        monkeypatch.setattr(store, "batch_remove", lambda *a, **kw: [])
    else:
        store.leased.update(keys("shared"))
        monkeypatch.setattr(store, "batch_probe_key", fail)
    with pytest.raises(RuntimeError):
        owner.collect()
    generation = owner.begin("adopt", "adopt-request")
    assert (
        owner.prepare(
            "adopt-request", generation, 128, keys("shared"), wait_for_deletion=False
        )
        is not None
    )


def test_failed_proof_releases_its_temporary_reader(catalog):
    owner, store, _ = catalog
    candidate = prepare(owner, store, "s", "r", "candidate")

    def fail(_):
        raise RuntimeError("Store unavailable")

    with pytest.raises(RuntimeError, match="Store unavailable"):
        owner.commit(candidate, fail)
    assert owner.snapshot()["readers"] == 0
    owner.abandon(candidate)
    assert owner.collect() == len(keys("candidate"))


def test_abandoned_partial_write_does_not_replace_latest(catalog):
    owner, store, _ = catalog
    old = prepare(owner, store, "s", "r1", "old")
    assert owner.commit(old, lambda _: True)
    candidate = prepare(owner, store, "s", "r2", "partial")
    store.objects.difference_update(keys("partial")[3:])
    owner.abandon(candidate)
    owner.finish("r2")
    owner.collect()
    assert owner.current("s")["checkpoint"] == old
    assert store.objects == set(keys("old"))


def test_late_old_request_cannot_replace_newer_generation(catalog):
    owner, store, _ = catalog
    older = prepare(owner, store, "s", "slow", "older", 256)
    newer = prepare(owner, store, "s", "fast", "newer", 128)
    assert owner.commit(newer, lambda _: True)
    assert not owner.commit(older, lambda _: True)
    owner.abandon(older)
    owner.collect()
    assert owner.current("s")["checkpoint"] == newer
    assert store.objects == set(keys("newer"))


def test_shared_checkpoint_survives_other_sessions_replacement(catalog):
    owner, store, _ = catalog
    a = prepare(owner, store, "a", "a1", "shared")
    b = prepare(owner, store, "b", "b1", "shared")
    assert owner.commit(a, lambda _: True)
    assert owner.commit(b, lambda _: True)
    a2 = prepare(owner, store, "a", "a2", "a-new")
    assert owner.commit(a2, lambda _: True)
    assert owner.collect() == 0
    assert set(keys("shared")) <= store.objects
    b2 = prepare(owner, store, "b", "b2", "b-new")
    assert owner.commit(b2, lambda _: True)
    assert owner.collect() == 24
    assert store.objects == set(keys("a-new") + keys("b-new"))


def test_reader_keeps_old_payload_until_all_rank_owner_releases(catalog):
    owner, store, _ = catalog
    old = prepare(owner, store, "s", "r1", "old")
    assert owner.commit(old, lambda _: True)
    reader = owner.hold_reader("reading", keys("old"))
    new = prepare(owner, store, "s", "r2", "new")
    assert owner.commit(new, lambda _: True)
    owner.finish("r1")
    assert owner.collect() == 0
    owner.release_reader(reader)
    assert owner.collect() == 24
    assert store.objects == set(keys("new"))


def test_non_forcing_lease_retry_and_readoption_cancel_pending_deletion(catalog):
    owner, store, _ = catalog
    old = prepare(owner, store, "a", "a1", "shared")
    assert owner.commit(old, lambda _: True)
    new = prepare(owner, store, "a", "a2", "new")
    assert owner.commit(new, lambda _: True)
    store.leased.update(keys("shared"))
    assert owner.collect() == 0
    assert owner.snapshot()["pending_keys"] == 24
    adopted = prepare(owner, store, "b", "b1", "shared")
    store.leased.clear()
    assert owner.collect() == 0
    assert owner.commit(adopted, lambda _: True)
    assert owner.snapshot()["pending_keys"] == 0
    assert set(keys("shared")) <= store.objects


def test_new_garbage_does_not_starve_retry_after_a_read_lease_expires(catalog):
    owner, store, _ = catalog
    old = prepare(owner, store, "old-session", "old-request", "old")
    owner.abandon(old)
    store.leased.update(keys("old"))
    assert owner.collect(limit=24) == 0

    fresh = prepare(owner, store, "fresh-session", "fresh-request", "fresh")
    owner.abandon(fresh)
    store.leased.update(keys("fresh"))
    store.leased.difference_update(keys("old"))

    assert owner.collect(limit=24) == 24
    assert not set(keys("old")) & store.objects
    assert set(keys("fresh")) <= store.objects


def test_session_expiry_waits_for_pending_puts_and_live_readers(catalog):
    owner, store, clock = catalog
    candidate = prepare(owner, store, "s", "r", "pending")
    owner.finish("r")
    clock.now += 61
    assert owner.collect() == 0  # PUTs not fenced yet.
    assert owner.commit(candidate, lambda _: True)
    reader = owner.hold_reader("load", keys("pending"))
    clock.now += 61
    assert owner.collect() == 0
    assert owner.current("s") is None
    owner.release_reader(reader)
    assert owner.collect() == 24
    assert store.objects == set()


def test_begin_is_idempotent_and_rejects_reassigned_request(catalog):
    owner, _, _ = catalog
    assert owner.begin("s", "r") == owner.begin("s", "r") == 1
    with pytest.raises(ValueError, match="changed session"):
        owner.begin("other", "r")


def test_unmanaged_namespace_is_never_claimed_or_deleted(catalog):
    owner, store, _ = catalog
    store.objects.add("legacy@group:0@unowned")
    generation = owner.begin("s", "r")
    with pytest.raises(ValueError, match="outside"):
        owner.prepare("r", generation, 128, ["legacy@group:0@unowned"])
    owner.collect()
    assert store.objects == {"legacy@group:0@unowned"}
    assert owner.snapshot()["checkpoints"] == 0


def test_latest_and_shared_ownership_survive_restart(tmp_path):
    config = SessionRetentionConfig(tmp_path / "state.sqlite", 60)
    store, clock = MemoryStore(), Clock()
    owner = SessionCheckpointCatalog(config, store, PREFIXES, clock=clock)
    latest = prepare(owner, store, "s", "committed", "latest")
    assert owner.commit(latest, lambda _: True)
    orphan = prepare(owner, store, "s", "interrupted", "partial")
    reader = owner.hold_reader("interrupted-load", keys("latest"))
    owner.close()
    restored = SessionCheckpointCatalog(config, store, PREFIXES, clock=clock)
    try:
        assert restored.current("s")["checkpoint"] == latest
        assert not restored.commit(orphan, lambda _: True)
        assert restored.collect() == 0
        clock.now += 61
        restored.collect()
        assert store.objects == set()
        assert restored.snapshot()["readers"] == 0
        restored.release_reader(reader)  # Late completion is harmless.
    finally:
        restored.close()


def test_journal_enforces_one_writer_and_namespace_identity(tmp_path):
    config = SessionRetentionConfig(tmp_path / "state.sqlite")
    store = MemoryStore()
    owner = SessionCheckpointCatalog(config, store, PREFIXES)
    try:
        with pytest.raises(ValueError, match="live writer"):
            SessionCheckpointCatalog(config, store, PREFIXES)
    finally:
        owner.close()
    with pytest.raises(ValueError, match="different namespace"):
        SessionCheckpointCatalog(config, store, ["different@group:0"])


def test_576_sessions_are_bounded_by_latest_not_number_of_turns(tmp_path):
    store = MemoryStore()
    owner = SessionCheckpointCatalog(
        SessionRetentionConfig(tmp_path / "state.sqlite"), store, PREFIXES
    )
    try:
        for turn in range(3):
            for session in range(576):
                request = f"{session}:{turn}"
                candidate = prepare(owner, store, str(session), request, request)
                assert owner.commit(candidate, lambda _: True)
                owner.finish(request)
                owner.collect()
        assert len(store.objects) == 576 * 24
        assert owner.snapshot()["checkpoints"] == 576
        assert owner.snapshot()["pending_keys"] == 0
    finally:
        owner.close()


def test_quiescent_store_reset_clears_retention_metadata(catalog):
    owner, store, _ = catalog
    candidate = prepare(owner, store, "s", "r", "latest")
    assert owner.commit(candidate, lambda _: True)
    owner.finish("r")
    store.objects.clear()  # External Store reset has already completed.
    owner.reset()
    assert owner.current("s") is None
    assert owner.snapshot()["sessions"] == 0
    assert owner.snapshot()["checkpoints"] == 0
    assert owner.snapshot()["pending_keys"] == 0
