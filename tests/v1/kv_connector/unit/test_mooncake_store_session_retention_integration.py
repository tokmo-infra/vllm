# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest
import torch

from vllm import SamplingParams
from vllm.config import ParallelConfig
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    BoundaryPut,
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    MooncakeLookupResult,
    MooncakeStoreWorkerMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
    MooncakeStoreScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.session_retention import (  # noqa: E501
    SessionCheckpointCatalog,
    SessionRetentionConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import (
    KVConnectorBlockState,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from .test_mooncake_store_session_retention import MemoryStore
from .utils import create_vllm_config

pytestmark = pytest.mark.cpu_test


class RetentionRPC:
    """External rank-0 service; exercise the real scheduler against a journal."""

    def __init__(self, tmp_path):
        self.prefixes = tuple(
            f"model@latest@r{r}@g{g}" for r in range(2) for g in range(1, 4)
        )
        self.store = MemoryStore()
        self.catalog = SessionCheckpointCatalog(
            SessionRetentionConfig(tmp_path / "catalog.sqlite"),
            self.store,
            self.prefixes,
        )
        self.prepared: dict[str, list[str]] = {}
        self.hybrid_ready = True

    def session_command(self, command, block_hashes=()):
        action = command["action"]
        if action == "begin_prepare":
            result = RetentionRPC.session_command(self, {**command, "action": "begin"})
            if command.get("boundary"):
                result.update(
                    RetentionRPC.session_command(
                        self, {**command, **result, "action": "prepare"}
                    )
                )
            return result
        if action == "begin":
            return {
                "generation": self.catalog.begin(command["session"], command["request"])
            }
        if action == "prepare":
            keys = [p + "@" + command["hash"] for p in self.prefixes]
            checkpoint = self.catalog.prepare(
                command["request"], command["generation"], command["boundary"], keys
            )
            self.prepared[checkpoint] = keys
            return {"checkpoint": checkpoint}
        if action == "commit":
            return {
                "committed": self.catalog.commit(
                    command["checkpoint"], lambda _: self.hybrid_ready
                )
            }
        if action == "abandon":
            self.catalog.abandon(command["checkpoint"])
        elif action == "finish":
            self.catalog.finish(command["request"])
        elif action == "release_reader":
            self.catalog.release_reader(command["reader"])
        else:
            raise AssertionError(action)
        return {}

    def discard(self, request):
        pass

    def lookup(self, *args, **kwargs):
        return MooncakeLookupResult(0)

    def submit_session_command(self, command, block_hashes=()):
        result: Future[dict[str, object]] = Future()
        try:
            result.set_result(self.session_command(command, block_hashes))
        except Exception as error:
            result.set_exception(error)
        return result


@pytest.fixture
def scheduler_fixture(tmp_path, monkeypatch):
    rpc = RetentionRPC(tmp_path)
    # LookupKeyClient is an IPC boundary, not a substitute for scheduler state.
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler.LookupKeyClient",
        lambda _: rpc,
    )
    config = create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "mamba_checkpoint_retention": "session_latest",
            "mamba_checkpoint_journal": str(tmp_path / "catalog.sqlite"),
            "lookup_async": True,
        },
    )
    config = replace(
        config,
        parallel_config=ParallelConfig(
            tensor_parallel_size=2, distributed_executor_backend="mp"
        ),
    )
    groups = [
        KVCacheGroupSpec(
            ["attention"],
            FullAttentionSpec(
                block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32
            ),
        )
    ]
    groups += [
        KVCacheGroupSpec(
            [f"mamba{g}"],
            MambaSpec(
                block_size=16,
                shapes=((2, 2),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
        )
        for g in range(3)
    ]
    kv = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        cache_hit_alignment_tokens=4,
        hash_block_size=4,
    )
    config.cache_config.prefix_match_unit = 4
    scheduler = MooncakeStoreScheduler(config, kv)
    pool = BlockPool(num_gpu_blocks=64, enable_caching=True, hash_block_size=4)
    scheduler.bind_gpu_block_pool(pool)
    yield scheduler, rpc, pool
    rpc.catalog.close()


def schedule(scheduler, pool, session="s", *, handoff=True, admit=False):
    request = Request(
        "request",
        prompt_token_ids=list(range(40)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        session_id=session,
        block_hasher=lambda _: [bytes([n + 1]) * 4 for n in range(10)],
    )
    allocated = pool.get_new_blocks(6)
    tables = (
        [b.block_id for b in allocated[:3]],
        *([b.block_id] for b in allocated[3:]),
    )
    if admit:
        assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(request, KVCacheBlocks(((),) * 4), 0)
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [NewRequestData.from_request(request, tables)]
    output.num_scheduled_tokens = {request.request_id: 40}
    output.total_num_scheduled_tokens = 40
    states = [(g, tables[g][0], boundary) for g in range(1, 4) for boundary in (16, 36)]
    output.kv_connector_block_state = KVConnectorBlockState(
        {request.request_id},
        lambda _: tables,
        {request.request_id: states} if handoff else {},
    )
    return request, tables, scheduler.build_connector_meta(output)


@pytest.mark.parametrize("cancel", [False, True])
def test_async_admission_retries_deleting_keys_before_dispatch(
    scheduler_fixture, monkeypatch, cancel
):
    """Retry a deletion conflict without admitting writes or leaking on abort."""
    scheduler, rpc, pool = scheduler_fixture
    submit = rpc.submit_session_command
    retry: Future[dict[str, object]] = Future()
    waiting: Future[dict[str, object]] = Future()
    retry.set_result({"generation": 1, "retry": True})
    responses = iter((retry, waiting))
    command = None

    def submit_admission(message, hashes=()):
        nonlocal command
        if message["action"] == "begin_prepare":
            command = message
            return next(responses)
        return submit(message, hashes)

    monkeypatch.setattr(rpc, "submit_session_command", submit_admission)
    request = Request(
        "request",
        prompt_token_ids=list(range(40)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        session_id="s",
        block_hasher=lambda _: [bytes([n + 1]) * 4 for n in range(10)],
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)
    assert rpc.catalog.current("s") is None
    if cancel:
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = {request.request_id}
        scheduler.build_connector_meta(output)
        waiting.set_result({"generation": 1, "retry": True})
        scheduler.build_connector_meta(SchedulerOutput.make_empty())
        assert not scheduler.has_pending_push_work()
        assert rpc.catalog.snapshot()["checkpoints"] == 0
    else:
        admitted = rpc.session_command(command)
        waiting.set_result(admitted)
        assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
        _, _, metadata = schedule(scheduler, pool, admit=True)
        job = next(r for r in metadata.requests if r.checkpoint_id)
        assert job.checkpoint_id == admitted["checkpoint"]


@pytest.mark.parametrize(
    "success,hybrid_ready", [(True, True), (False, True), (True, False)]
)
@pytest.mark.parametrize("admit", [False, True])
def test_scheduler_waits_for_all_rank_success_and_full_hybrid_prefix(
    scheduler_fixture, success, hybrid_ready, admit
):
    scheduler, rpc, pool = scheduler_fixture
    _, _, metadata = schedule(scheduler, pool, admit=admit)
    job = next(r for r in metadata.requests if r.checkpoint_id)
    assert {p.num_tokens for p in job.boundary_puts if p.group_id in (1, 2, 3)} == {36}
    checkpoint = job.checkpoint_id
    rpc.store.objects.update(rpc.prepared[checkpoint])
    rpc.hybrid_ready = hybrid_ready
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
                checkpoint_results={checkpoint: {0: True}}
            )
        )
    )
    assert rpc.catalog.current("s") is None
    # Duplicate rank 0 is not a second rank.
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
                checkpoint_results={checkpoint: {0: True}}
            )
        )
    )
    assert rpc.catalog.current("s") is None
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
                completed_saves={job.store_job_id: 2},
                checkpoint_results={checkpoint: {1: success}},
            )
        )
    )
    latest = rpc.catalog.current("s")
    assert (latest is not None) == (success and hybrid_ready)
    if success and not hybrid_ready:
        # Missing proof is final only once no later request step can supply it.
        assert rpc.catalog.collect() == 0
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = {job.req_id}
        scheduler.build_connector_meta(output)
    rpc.catalog.collect()
    assert bool(rpc.store.objects) == (success and hybrid_ready)


def test_unidentified_request_does_not_persist_unowned_kda(scheduler_fixture):
    scheduler, rpc, pool = scheduler_fixture
    _, _, metadata = schedule(scheduler, pool, session=None)
    assert all(r.checkpoint_id is None for r in metadata.requests)
    assert all(
        p.group_id == 0 for r in metadata.requests for p in r.boundary_puts or []
    )
    assert rpc.catalog.snapshot()["sessions"] == 0


@pytest.mark.parametrize("hybrid_ready", [True, False])
def test_slow_checkpoint_commit_does_not_stop_scheduler_or_request_finish(
    scheduler_fixture, monkeypatch, hybrid_ready
):
    """A slow Store proof must retain ownership without blocking engine steps."""
    scheduler, rpc, pool = scheduler_fixture
    old_generation = rpc.catalog.begin("s", "old")
    old_keys = [prefix + "@old" for prefix in rpc.prefixes]
    old = rpc.catalog.prepare("old", old_generation, 16, old_keys)
    rpc.store.objects.update(old_keys)
    assert rpc.catalog.commit(old, lambda _: True)
    rpc.catalog.finish("old")
    request, _, metadata = schedule(scheduler, pool)
    job = next(r for r in metadata.requests if r.checkpoint_id)
    rpc.store.objects.update(rpc.prepared[job.checkpoint_id])
    rpc.hybrid_ready = hybrid_ready
    started, release = Event(), Event()
    command = rpc.session_command

    def gated_command(message, hashes=()):
        if message["action"] == "commit":
            started.set()
            assert release.wait(10), "test did not release the external Store proof"
        return command(message, hashes)

    with ThreadPoolExecutor(max_workers=1) as transport:
        monkeypatch.setattr(
            rpc,
            "session_command",
            lambda *args: transport.submit(gated_command, *args).result(),
        )
        monkeypatch.setattr(
            rpc,
            "submit_session_command",
            lambda *args: transport.submit(gated_command, *args),
        )
        with ThreadPoolExecutor(max_workers=1) as engine:
            step = engine.submit(
                scheduler.update_connector_output,
                KVConnectorOutput(
                    kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
                        completed_saves={job.store_job_id: 2},
                        checkpoint_results={job.checkpoint_id: {0: True, 1: True}},
                    )
                ),
            )
            try:
                assert started.wait(5), "checkpoint verification never started"
                step.result(timeout=1)
                assert rpc.catalog.current("s")["checkpoint"] == old
                assert rpc.catalog.collect() == 0
                next_request = Request(
                    "next",
                    prompt_token_ids=list(range(40)),
                    sampling_params=SamplingParams(max_tokens=1),
                    pooling_params=None,
                    session_id="next",
                    block_hasher=lambda _: [bytes([n + 1]) * 4 for n in range(10)],
                )

                def admit_next_request():
                    matched, _ = scheduler.get_num_new_matched_tokens(next_request, 0)
                    if matched is not None:
                        scheduler.update_state_after_alloc(
                            next_request, KVCacheBlocks(([], [], [], [])), 0
                        )
                    return matched

                assert engine.submit(admit_next_request).result(timeout=1) is None
                output = SchedulerOutput.make_empty()
                output.finished_req_ids = {request.request_id, next_request.request_id}
                engine.submit(scheduler.build_connector_meta, output).result(timeout=1)
                assert scheduler.has_pending_push_work()
            finally:
                release.set()
        # Fence the external service, then poll through the real engine hook.
        transport.submit(lambda: None).result(timeout=5)
        scheduler.build_connector_meta(SchedulerOutput.make_empty())
        transport.submit(lambda: None).result(timeout=5)
        scheduler.build_connector_meta(SchedulerOutput.make_empty())
        assert rpc.catalog.current("s")["checkpoint"] == (
            job.checkpoint_id if hybrid_ready else old
        )
        rpc.catalog.collect()
        assert rpc.store.objects == set(
            rpc.prepared[job.checkpoint_id] if hybrid_ready else old_keys
        )


def test_checkpoint_waits_for_earlier_attention_save_on_every_rank(scheduler_fixture):
    scheduler, rpc, pool = scheduler_fixture
    request, tables, first = schedule(scheduler, pool, handoff=False)
    attention_job = next(m for m in first.requests if m.can_save)
    output = SchedulerOutput.make_empty()
    output.kv_connector_block_state = KVConnectorBlockState(
        {request.request_id},
        lambda _: tables,
        {request.request_id: [(g, tables[g][0], 36) for g in range(1, 4)]},
    )
    metadata = scheduler.build_connector_meta(output)
    state_job = next(m for m in metadata.requests if m.checkpoint_id)
    rpc.store.objects.update(rpc.prepared[state_job.checkpoint_id])
    rpc.hybrid_ready = False
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
                completed_saves={state_job.store_job_id: 2},
                checkpoint_results={state_job.checkpoint_id: {0: True, 1: True}},
            )
        )
    )
    assert rpc.catalog.current("s") is None
    assert rpc.catalog.collect() == 0
    assert rpc.store.objects == set(rpc.prepared[state_job.checkpoint_id])
    # A completion from one rank cannot fence the other rank's Attention PUT.
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
                completed_saves={attention_job.store_job_id: 1}
            )
        )
    )
    assert rpc.catalog.current("s") is None
    rpc.hybrid_ready = True
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
                completed_saves={attention_job.store_job_id: 1}
            )
        )
    )
    assert rpc.catalog.current("s")["checkpoint"] == state_job.checkpoint_id


def test_late_attention_proof_is_checked_after_finish_metadata(scheduler_fixture):
    scheduler, rpc, pool = scheduler_fixture
    request, _, metadata = schedule(scheduler, pool)
    job = next(m for m in metadata.requests if m.checkpoint_id)
    rpc.store.objects.update(rpc.prepared[job.checkpoint_id])
    rpc.hybrid_ready = False
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
                completed_saves={job.store_job_id: 2},
                checkpoint_results={job.checkpoint_id: {0: True, 1: True}},
            )
        )
    )
    assert rpc.catalog.collect() == 0
    assert rpc.catalog.current("s") is None
    rpc.hybrid_ready = True
    output = SchedulerOutput.make_empty()
    output.finished_req_ids = {request.request_id}
    scheduler.build_connector_meta(output)
    assert rpc.catalog.current("s")["checkpoint"] == job.checkpoint_id


def test_finish_time_handoff_survives_request_cleanup(scheduler_fixture):
    scheduler, rpc, pool = scheduler_fixture
    request, tables, first = schedule(scheduler, pool, handoff=False)
    request.num_computed_tokens = 40
    scheduler.register_finished_partial_tail(
        request, tables, [(g, tables[g][0], 36) for g in range(1, 4)]
    )
    output = SchedulerOutput.make_empty()
    output.finished_req_ids = {request.request_id}
    metadata = scheduler.build_connector_meta(output)
    job = next(r for r in metadata.requests if r.checkpoint_id)
    rpc.store.objects.update(rpc.prepared[job.checkpoint_id])
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
                completed_saves={job.store_job_id: 2},
                checkpoint_results={job.checkpoint_id: {0: True, 1: True}},
            )
        )
    )
    assert rpc.catalog.current("s") is None
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
                completed_saves={
                    m.store_job_id: 2 for m in first.requests if m.can_save
                }
            )
        )
    )
    assert rpc.catalog.current("s")["checkpoint"] == job.checkpoint_id


def test_offload_only_boundary_keeps_the_conversation_owner(scheduler_fixture):
    scheduler, rpc, pool = scheduler_fixture
    request, tables, first = schedule(scheduler, pool, handoff=False)
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
                completed_saves={
                    m.store_job_id: 2 for m in first.requests if m.can_save
                }
            )
        )
    )
    output = SchedulerOutput.make_empty()
    output.kv_connector_block_state = KVConnectorBlockState(
        {request.request_id},
        lambda _: tables,
        {request.request_id: [(g, tables[g][0], 36) for g in range(1, 4)]},
    )
    metadata = scheduler.build_connector_meta(output)
    job = next(m for m in metadata.requests if m.can_save and m.token_len_chunk == 0)
    assert job.session_id == "s"
    assert job.session_generation == 1
    assert job.checkpoint_id is not None
    rpc.store.objects.update(rpc.prepared[job.checkpoint_id])
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
                completed_saves={job.store_job_id: 2},
                checkpoint_results={job.checkpoint_id: {0: True, 1: True}},
            )
        )
    )
    assert rpc.catalog.current("s")["checkpoint"] == job.checkpoint_id


def test_reader_is_released_after_all_ranks_even_after_request_cancellation(
    scheduler_fixture,
):
    scheduler, rpc, _ = scheduler_fixture
    generation = rpc.catalog.begin("s", "old")
    keys = [prefix + "@old" for prefix in rpc.prefixes]
    old = rpc.catalog.prepare("old", generation, 16, keys)
    rpc.store.objects.update(keys)
    assert rpc.catalog.commit(old, lambda _: True)
    reader = rpc.catalog.hold_reader("cancelled", keys)
    generation = rpc.catalog.begin("s", "new")
    newer_keys = [prefix + "@new" for prefix in rpc.prefixes]
    newer = rpc.catalog.prepare("new", generation, 32, newer_keys)
    rpc.store.objects.update(newer_keys)
    assert rpc.catalog.commit(newer, lambda _: True)
    output = SchedulerOutput.make_empty()
    output.finished_req_ids = {"cancelled"}
    scheduler.build_connector_meta(output)
    for _ in range(2):
        scheduler.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
                    completed_reads={reader: {0: True}}
                )
            )
        )
        assert rpc.catalog.collect() == 0
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
                completed_reads={reader: {1: True}}
            )
        )
    )
    assert rpc.catalog.collect() == len(keys)
    assert rpc.store.objects == set(newer_keys)


class TransferStore(MemoryStore):
    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail

    def batch_is_exist(self, keys):
        return self.batch_probe_key(keys)

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, config):
        if self.fail:
            return [-1] * len(keys)
        self.objects.update(keys)
        return [0] * len(keys)

    def batch_get_into_multi_buffers(self, keys, addrs, sizes):
        return [-1 if self.fail else sum(s) for s in sizes]


@pytest.fixture
def lookup_worker(tmp_path, monkeypatch):
    from .test_mooncake_store_hma_e2e import _build_worker_with_dict_store

    class SDKStore(TransferStore):
        gate = None
        after_claim = None
        catalog = None

        def __init__(self):
            super().__init__()
            self.started = Event()
            self.release = Event()

        def setup(self, *args, **kwargs):
            return 0

        def close(self):
            pass

        def pause(self, operation, result):
            if self.gate == operation:
                self.gate = None
                self.started.set()
                assert self.release.wait(5)
            return result

        def batch_probe_key(self, keys):
            if self.after_claim and self.catalog.snapshot()["readers"]:
                if self.after_claim == "error":
                    raise RuntimeError("Store probe failed")
                return [] if self.after_claim == "short" else [0] * len(keys)
            return self.pause("probe", super().batch_probe_key(keys))

        def batch_is_exist(self, keys):
            return self.pause("lease", super().batch_probe_key(keys))

    monkeypatch.setenv("VLLM_MOONCAKE_STORE_SELECTIVE_LEASE", "1")
    config = create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "mamba_checkpoint_retention": "session_latest",
            "mamba_checkpoint_journal": str(tmp_path / "worker.sqlite"),
        },
    )
    groups = [
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=16,
                shapes=((2, 2),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
        )
    ]
    kv = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        hash_block_size=4,
        cache_hit_alignment_tokens=4,
    )
    store = SDKStore()
    instance = _build_worker_with_dict_store(config, kv, store)
    store.catalog = instance.session_catalog
    hashes = [i.to_bytes(4, "big") for i in range(5)]
    prepared = instance.session_retention_command(
        {
            "action": "begin_prepare",
            "session": "s",
            "request": "seed",
            "boundary": 16,
            "hash": hashes[3].hex(),
        },
        (),
    )
    store.objects.add(instance.token_dbs[0].key_for(hashes[3]))
    assert instance.session_catalog.commit(prepared["checkpoint"], lambda _: True)
    instance.session_catalog.finish("seed")
    try:
        yield instance, store, hashes
    finally:
        store.release.set()
        instance.close()


def test_worker_slow_lookup_does_not_hold_catalog_lock(lookup_worker):
    instance, store, hashes = lookup_worker
    store.gate = "probe"
    with ThreadPoolExecutor(max_workers=2) as pool:
        lookup = pool.submit(
            instance.session_retention_command,
            {"action": "lookup", "request": "reader", "num_tokens": 20},
            hashes,
        )
        try:
            assert store.started.wait(5)
            admitted = pool.submit(
                instance.session_retention_command,
                {"action": "begin_prepare", "session": "other", "request": "other"},
                (),
            ).result(timeout=1)
            assert admitted["generation"] == 1
        finally:
            store.release.set()
        result = lookup.result(timeout=5)
    assert result["hit_length"] == 16
    assert result["reader"] is not None
    instance.session_catalog.release_reader(result["reader"])


def test_worker_lookup_rejects_payload_deleted_before_reader_claim(lookup_worker):
    instance, store, hashes = lookup_worker
    store.gate = "lease"

    def replace_checkpoint():
        newer = b"next"
        prepared = instance.session_retention_command(
            {
                "action": "begin_prepare",
                "session": "s",
                "request": "newer",
                "boundary": 16,
                "hash": newer.hex(),
            },
            (),
        )
        store.objects.add(instance.token_dbs[0].key_for(newer))
        assert instance.session_catalog.commit(prepared["checkpoint"], lambda _: True)
        instance.session_catalog.collect()
        assert instance.token_dbs[0].key_for(hashes[3]) not in store.objects

    with ThreadPoolExecutor(max_workers=2) as pool:
        lookup = pool.submit(
            instance.session_retention_command,
            {"action": "lookup", "request": "reader", "num_tokens": 20},
            hashes,
        )
        try:
            assert store.started.wait(5)
            pool.submit(replace_checkpoint).result(timeout=1)
        finally:
            store.release.set()
        result = lookup.result(timeout=5)
    assert result == {"hit_length": 0, "tails": [], "reader": None}
    assert instance.session_catalog.snapshot()["readers"] == 0


@pytest.mark.parametrize("failure", ["missing", "short", "error"])
def test_worker_lookup_releases_reader_when_validation_fails(lookup_worker, failure):
    instance, store, hashes = lookup_worker
    store.after_claim = failure
    result = instance.session_retention_command(
        {"action": "lookup", "request": "reader", "num_tokens": 20},
        hashes,
    )
    assert result == {"hit_length": 0, "tails": [], "reader": None}
    assert instance.session_catalog.snapshot()["readers"] == 0


def test_worker_admission_claims_checkpoint_before_payload_writes(
    tmp_path, monkeypatch
):
    from .test_mooncake_store_hma_e2e import _build_worker_with_dict_store

    class SDKStore(TransferStore):
        def setup(self, *args, **kwargs):
            return 0

    monkeypatch.setenv("VLLM_MOONCAKE_STORE_SELECTIVE_LEASE", "1")
    config = create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "mamba_checkpoint_retention": "session_latest",
            "mamba_checkpoint_journal": str(tmp_path / "worker.sqlite"),
        },
    )
    groups = [
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=16,
                shapes=((2, 2),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
        )
    ]
    kv = KVCacheConfig(num_blocks=4, kv_cache_tensors=[], kv_cache_groups=groups)
    store = SDKStore()
    worker = _build_worker_with_dict_store(config, kv, store)
    try:
        result = worker.session_retention_command(
            {
                "action": "begin_prepare",
                "session": "s",
                "request": "r",
                "boundary": 16,
                "hash": "abcd",
            },
            (),
        )
        assert result["generation"] == 1
        assert worker.session_catalog.snapshot()["checkpoints"] == 1
        assert not store.objects
        assert worker.session_catalog.current("s") is None
        worker.session_retention_command(
            {
                "action": "abandon",
                "checkpoint": result["checkpoint"],
            },
            (),
        )
        worker.session_retention_command({"action": "finish", "request": "r"}, ())
        assert worker.session_catalog.snapshot()["checkpoints"] == 0
    finally:
        worker.close()


def transfer_inputs():
    groups = [
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=16,
                shapes=((2, 2),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
        )
    ]
    coord = MooncakeStoreCoordinator(groups, scheduler_block_size=16, hash_block_size=4)
    tensor = torch.zeros((4, 4), dtype=torch.float32)
    db = ChunkedTokenDatabase(KeyMetadata("model", 0, 0, 0, 0), 16, hash_block_size=4)
    db.set_kv_caches_base_addr([tensor.data_ptr()])
    db.set_block_len([16])
    return coord, tensor, db


@pytest.mark.parametrize("fail", [False, True])
def test_sending_thread_distinguishes_completion_from_success(fail):
    coord, tensor, db = transfer_inputs()
    store = TransferStore(fail)
    ready = Event()
    thread = KVCacheStoreSendingThread(store, coord, [db], 16, 0, [1], "kv_both", ready)
    thread.start()
    assert ready.wait(5)
    request = ReqMeta(
        "r",
        0,
        ([1],),
        [b"hash"] * 4,
        can_save=True,
        store_job_id=1,
        checkpoint_id="candidate",
        boundary_puts=[BoundaryPut(0, 1, 12)],
    )
    thread.add_request(request)
    thread.request_queue.join()
    assert thread.take_completed_saves() == {1: 1}
    assert thread.take_checkpoint_results() == {"candidate": {0: not fail}}
    assert tensor.shape == (4, 4)  # Keep the registered buffer live through DMA.


@pytest.mark.parametrize("fail", [False, True])
def test_receiving_thread_reports_reader_completion_even_when_get_fails(fail):
    coord, tensor, db = transfer_inputs()
    ready = Event()
    thread = KVCacheStoreRecvingThread(TransferStore(fail), coord, [db], 16, 0, ready)
    thread.start()
    assert ready.wait(5)
    request = ReqMeta(
        "r",
        0,
        ([1],),
        [b"hash"] * 4,
        load_spec=LoadSpec(0, 16, True, token_len=16, read_handle="read:1"),
    )
    thread.add_request(request)
    thread.request_queue.join()
    assert thread.take_completed_reads() == {"read:1"}
    assert thread.get_and_clear_finished_requests() == {"r"}
    assert bool(thread.get_and_clear_block_ids_with_load_errors()) == fail
    assert tensor.shape == (4, 4)
