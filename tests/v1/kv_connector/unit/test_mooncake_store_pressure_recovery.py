# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for the live-job Store pressure recovery gate."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import worker
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    ReqMeta,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec


def make_sender(store):
    db = ChunkedTokenDatabase(KeyMetadata("pressure-test", 0, 0, 0, 0), block_size=16)
    db.set_kv_caches_base_addr([0x1000])
    db.set_block_len([256])
    coord = worker.MooncakeStoreCoordinator(
        [
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=16, num_kv_heads=8, head_size=64, dtype=None
                ),
            )
        ],
        scheduler_block_size=16,
        hash_block_size=16,
    )
    return worker.KVCacheStoreSendingThread(
        store=store,
        coord=coord,
        token_databases=[db],
        block_size=16,
        tp_rank=0,
        group_put_steps=[1],
        kv_role="kv_producer",
        ready_event=threading.Event(),
        replicate_config=SimpleNamespace(),
    )


def run_job(sender, seq, req_id="same-request", event=None):
    req = ReqMeta(
        req_id=req_id,
        token_len_chunk=32,
        block_ids=([1, 2],),
        block_hashes=[b"a0", b"a1"],
        can_save=True,
        store_job_id=seq,
        current_event=event,
    )
    sender.add_request(req)
    sender._handle_request(sender.request_queue.get_nowait())
    assert sender.take_completed_saves() == {seq: 1}
    assert sender.request_queue.unfinished_tasks == 0
    return req


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(worker.time, "monotonic", lambda: now[0])
    return now


def test_same_request_resumes_after_space_recovers_without_another_request(clock):
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = [[-200, -200], [256, 256]]
    sender = make_sender(store)
    run_job(sender, 1)
    assert sender._saved_offset.get("same-request", 0) == 0
    run_job(sender, 2)
    assert store.batch_put_from_multi_buffers.call_count == 1
    clock[0] += worker._STORE_PRESSURE_PROBE_INTERVAL_S
    fence = MagicMock()
    run_job(sender, 3, event=fence)
    fence.synchronize.assert_called_once()
    assert store.batch_put_from_multi_buffers.call_count == 2
    assert sender._saved_offset["same-request"] == 32
    assert not sender._store_pressure_active
    assert not sender._skip_store_requests


def test_persistent_pressure_is_rate_limited_and_does_not_hold_jobs(clock):
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [-200, -200]
    sender = make_sender(store)
    run_job(sender, 1)
    for seq in range(2, 12):
        run_job(sender, seq)
    assert store.batch_put_from_multi_buffers.call_count == 1
    clock[0] += worker._STORE_PRESSURE_PROBE_INTERVAL_S
    run_job(sender, 12)
    run_job(sender, 13)
    assert store.batch_put_from_multi_buffers.call_count == 2
    assert sender._saved_offset.get("same-request", 0) == 0


def test_probe_slot_is_shared_across_pressure_skipped_requests(clock):
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [-200, -200]
    sender = make_sender(store)
    run_job(sender, 1, "a")
    run_job(sender, 2, "b")
    clock[0] += worker._STORE_PRESSURE_PROBE_INTERVAL_S
    run_job(sender, 3, "a")
    run_job(sender, 4, "b")
    assert store.batch_put_from_multi_buffers.call_count == 3


def test_new_request_failure_does_not_postpone_an_existing_probe(clock):
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [-200, -200]
    sender = make_sender(store)
    run_job(sender, 1, "a")
    deadline = sender._store_pressure_probe_at
    clock[0] += 0.5 * worker._STORE_PRESSURE_PROBE_INTERVAL_S
    run_job(sender, 2, "b")
    assert sender._store_pressure_probe_at == deadline
    clock[0] = deadline
    run_job(sender, 3, "a")
    assert store.batch_put_from_multi_buffers.call_count == 3


def test_boundary_failure_recovers_using_only_the_new_jobs_handoff(clock):
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = [[-200], [256], [256, 256]]
    sender = make_sender(store)
    # Exercise the real boundary PUT path with distinct pinned source addresses.
    sender._boundary_puts = lambda req: [
        ("boundary", [0x1000 + req.store_job_id * 256], [256], None)
    ]
    for seq in (1, 2, 3):
        if seq == 3:
            clock[0] += worker._STORE_PRESSURE_PROBE_INTERVAL_S
        req = ReqMeta(
            req_id="same-request",
            token_len_chunk=32,
            block_ids=([1, 2],),
            block_hashes=[b"a0", b"a1"],
            can_save=True,
            store_job_id=seq,
            boundary_puts=[(0, 1, 16)],
            current_event=MagicMock(),
        )
        sender.add_request(req)
        sender._handle_request(sender.request_queue.get_nowait())
        assert sender.take_completed_saves() == {seq: 1}
        assert sender.request_queue.unfinished_tasks == 0
    calls = store.batch_put_from_multi_buffers.call_args_list
    assert len(calls) == 3
    assert calls[0].args[1] == [[0x1100]]
    assert calls[1].args[1] == [[0x1300]]
    assert not sender._store_pressure_active


def test_retired_job_cannot_consume_a_probe_or_write_after_request_id_reuse(clock):
    sender = make_sender(MagicMock())
    old = ReqMeta(
        req_id="same-request",
        token_len_chunk=32,
        block_ids=([1, 2],),
        block_hashes=[b"a0", b"a1"],
        can_save=True,
        store_job_id=1,
    )
    sender.add_request(old)
    sender._mark_request_skipped_for_pressure(old)
    sender.delete_finished_stored_request(old.req_id)
    fresh = ReqMeta(
        req_id=old.req_id,
        token_len_chunk=32,
        block_ids=([1, 2],),
        block_hashes=[b"a0", b"a1"],
        can_save=True,
        store_job_id=2,
    )
    sender.add_request(fresh)
    sender._mark_request_skipped_for_pressure(fresh)
    clock[0] += worker._STORE_PRESSURE_PROBE_INTERVAL_S
    before = sender._store_pressure_probe_at
    sender._handle_request(sender.request_queue.get_nowait())
    sender.store.batch_is_exist.assert_not_called()
    assert sender._store_pressure_probe_at == before
    assert sender.is_live_store_job(fresh)


def test_store_exception_on_probe_does_not_fake_a_success(clock):
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = [
        [-200, -200],
        RuntimeError("fail"),
    ]
    sender = make_sender(store)
    run_job(sender, 1)
    clock[0] += worker._STORE_PRESSURE_PROBE_INTERVAL_S
    run_job(sender, 2)
    assert sender._store_pressure_active
    assert sender._saved_offset.get("same-request", 0) == 0
