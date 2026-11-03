# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts using the imported historical allocator and scheduler methods.

Token hashes and KV shapes are synthetic. No model, connector, or GPU executes.
"""

import ast
import builtins
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch

from vllm import envs
from vllm.config import KVTransferConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.hot_tail_budget import HotTailBudget
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test
CONFIG = {
    "kv_connector": "MultiConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "fail",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "MooncakeConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "mooncake_protocol": "rdma",
                    "device_name": "cpu",
                },
            },
            {
                "kv_connector": "MooncakeStoreConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "cache_prefix": "cpu",
                    "load_async": True,
                    "lookup_async": True,
                    "store_compact_group_payloads": True,
                },
            },
        ]
    },
}


class Fixture:
    def __init__(self, free=22, groups=3, prompt=48493, endpoint=48128, enabled=True):
        specs = [
            MambaSpec(
                block_size=8960,
                shapes=((16,),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
                num_speculative_blocks=3,
                num_prefill_checkpoint_blocks=1,
                prefill_checkpoint_alignment=64,
            )
            for _ in range(groups)
        ]
        specs += [
            MLAAttentionSpec(
                block_size=8960,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.uint8,
                cache_dtype_str="fp8",
            ),
            CircularBufferSpec(
                block_size=8, num_kv_heads=1, head_size=64, dtype=torch.uint8
            ),
        ]
        config = KVCacheConfig(
            num_blocks=free + 1,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["cpu_layer_" + str(i)],
                    kv_cache_spec=spec,
                    is_eagle_group=True,
                )
                for i, spec in enumerate(specs)
            ],
            prefix_cache_retention_interval=0,
            hash_block_size=256,
            cache_hit_alignment_tokens=256,
        )
        self.k = KVCacheManager(
            config,
            max_model_len=1048576,
            scheduler_block_size=8960,
            hash_block_size=256,
            max_in_flight_tokens=4096,
            enable_caching=True,
            use_eagle=True,
            num_prefill_lookahead=1,
        )
        assert self.k.block_pool.get_num_free_blocks() == free
        self.r = Request(
            "cpu-real-tail", [1] * prompt, SamplingParams(max_tokens=4096), None
        )
        # Opaque hash data only; hashing/tokenization quality is not under test.
        self.r.block_hashes = [
            hashlib.sha256(str(i).encode()).digest() for i in range(prompt // 256)
        ]
        self.endpoint = endpoint
        s = Scheduler.__new__(Scheduler)
        s.kv_cache_manager = self.k
        s.vllm_config = NS(kv_transfer_config=KVTransferConfig(**copy.deepcopy(CONFIG)))
        s.scheduler_config = NS(async_scheduling=True, long_prefill_token_threshold=0)
        s.cache_config = NS(block_size=8960)
        s.max_model_len = 1048576
        s.max_num_scheduled_tokens = 2048
        s.num_prefill_lookahead = 1
        s.num_lookahead_tokens = s.num_spec_tokens = 3
        s.prefix_replay_tokens = 0
        s.is_encoder_decoder = False
        s.dcp_world_size = s.pcp_world_size = 1
        s.requests = {self.r.request_id: self.r}
        s.failed_recving_kv_req_ids = set()
        s.finished_recving_kv_req_ids = set()
        s._inflight_prefills = set()
        s._spec_decode_step_blocks = 0
        s.scheduler_reserve_full_isl = True
        s.connector = None
        s.block_size = 8960
        s.hash_block_size = 256
        s.mamba_partial_cache_hit = True
        s.mamba_has_prefill_checkpoint_blocks = True
        s.use_eagle_block_drop = True
        s.mamba_prefill_checkpoint_alignment = 64
        s.mamba_shared_prefix_checkpoint = False
        s._hot_tail_budget = HotTailBudget.create(s) if enabled else None
        assert not enabled or (
            s._hot_tail_budget is not None and not s._hot_tail_budget.failed
        )
        self.s = s

    def load(self):
        reserved = self.s._inflight_prefill_reserved_blocks()
        result = self.s._allocate_slots_with_hot_tail(
            self.r,
            0,
            num_external_computed_tokens=self.endpoint,
            new_computed_blocks=self.k.empty_kv_cache_blocks,
            full_sequence_must_fit=True,
            delay_cache_blocks=True,
            reserved_blocks=reserved,
        )
        if result is not None:
            self.r.num_computed_tokens = self.endpoint
            self.r.status = RequestStatus.WAITING_FOR_REMOTE_KVS
            self.s._inflight_prefills.add(self.r)
        return result

    def complete_load(self):
        self.s.connector = (
            object()
        )  # completion notification, not a real network connector
        self.s.finished_recving_kv_req_ids = {self.r.request_id}
        self.s._mark_prefix_replay = lambda *a, **kw: 0
        self.s._load_restores_replay_window = lambda *a: False
        self.s._update_waiting_for_remote_kv(self.r)
        self.s.connector = None
        self.r.status = RequestStatus.WAITING

    def finish(self):
        first = True
        iterations = 0
        while self.r.num_computed_tokens < self.r.num_prompt_tokens:
            iterations += 1
            assert iterations <= 32
            r, s = self.r, self.s
            n = s._mamba_block_aligned_split(
                r, min(2048, r.num_prompt_tokens - r.num_computed_tokens)
            )
            n = s._reserve_prefill_lookahead(r, r.num_computed_tokens, n)
            assert n > 0
            out = s._allocate_slots_with_hot_tail(
                r,
                n,
                num_lookahead_tokens=s.num_lookahead_tokens,
                full_sequence_must_fit=first,
            )
            assert out is not None, (
                r.num_computed_tokens,
                n,
                self.k.block_pool.get_num_free_blocks(),
            )
            r.num_computed_tokens += n
            r.status = RequestStatus.RUNNING
            first = False


def native_state(x):
    return (
        x.k.block_pool.get_num_free_blocks(),
        tuple(
            tuple(
                (b.block_id, b.ref_cnt) for b in m.req_to_blocks.get(x.r.request_id, ())
            )
            for m in x.k.coordinator.single_type_managers
        ),
        tuple(
            tuple(
                (a.block_id, b.block_id, a.ref_cnt, b.ref_cnt)
                for a, b in getattr(m, "_pending_cow_copies", ())
            )
            for m in x.k.coordinator.single_type_managers
        ),
    )


@pytest.mark.parametrize("value", [None, "0", "", "true", "2", "01", "1"])
def test_exact_opt_in_controls_actual_init_import(monkeypatch, value):
    """Execute the actual init gate with a watched helper import and factory."""
    name = "VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    enabled = envs.environment_variables[name]()
    monkeypatch.setattr(envs, name, enabled)
    tree = ast.parse(
        Path(__import__(Scheduler.__module__, fromlist=["x"]).__file__).read_text()
    )
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    start = next(
        i
        for i, n in enumerate(init.body)
        if isinstance(n, ast.Assign)
        and ast.unparse(n.targets[0]) == "self._hot_tail_budget"
    )
    gate = ast.parse("def initialize(self):\n    pass\n").body[0]
    gate.body = copy.deepcopy(init.body[start : start + 2])
    ns = {"envs": envs}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[gate], type_ignores=[])),
            "actual Scheduler init gate",
            "exec",
        ),
        ns,
    )
    calls = []
    original_import = builtins.__import__

    def watched(name, *args, **kwargs):
        if name == "vllm.v1.core.sched.hot_tail_budget":
            calls.append(name)
        return original_import(name, *args, **kwargs)

    shell = NS()
    sentinel = object()
    with patch.object(HotTailBudget, "create", return_value=sentinel) as factory:
        with patch("builtins.__import__", side_effect=watched):
            ns["initialize"](shell)
        assert factory.call_count == int(value == "1")
    assert len(calls) == int(value == "1")
    assert shell._hot_tail_budget is (sentinel if value == "1" else None)


@pytest.mark.parametrize("groups", [1, 2, 3, 5])
@pytest.mark.parametrize(
    "prompt, endpoint",
    [
        (48493, 48128),
        (48493, 47872),
        (57351, 57088),
        (69699, 69376),
        (43354, 43008),
        (61431, 60928),
    ],
)
def test_real_mixed_managers_consume_future_once(groups, prompt, endpoint):
    """Replay the six archived geometries through real mixed allocators."""
    free = 5 * groups + (endpoint + 8959) // 8960 + 1
    x = Fixture(free=free, groups=groups, prompt=prompt, endpoint=endpoint)
    assert x.load() is not None
    assert x.k.block_pool.get_num_free_blocks() == groups
    x.complete_load()
    x.finish()
    budget = x.s._hot_tail_budget
    assert not budget.failed
    assert set(budget.ledger.entries[x.r.request_id]["phases"].values()) == {"spent"}
    assert x.k.block_pool.get_num_free_blocks() == 0
    before = native_state(x)
    for _ in range(2):
        budget.ledger.allocation_succeeded(
            x.r, x.k.coordinator.single_type_managers[:groups]
        )
        assert budget.ledger.budgets(x.r) == (0, 0)
    assert native_state(x) == before
    for m in x.k.coordinator.single_type_managers[:groups]:
        assert isinstance(m, MambaManager)
        assert len(m._pending_cow_copies) == 1
        source, copied = m._pending_cow_copies[0]
        assert (source.ref_cnt, copied.ref_cnt) == (2, 1)
        assert len(m._pending_boundary_state_offloads) == 1


@pytest.mark.parametrize("free", [19, 20, 21])
def test_counterexample_native_load_fits_but_future_cow_does_not(free):
    """Protect tail capacity before publishing an otherwise native-fitting load."""
    off = Fixture(free=free, enabled=False)
    assert off.load() is not None
    on = Fixture(free=free)
    state = native_state(on)
    assert on.load() is None
    assert on.k.is_producer_budget_rejection(on.r)
    assert native_state(on) == state
    assert on.s._hot_tail_budget.ledger.entries == {}
    assert on.s._hot_tail_budget.seen_loads == {}
    off.complete_load()
    with pytest.raises(AssertionError):
        off.finish()


@pytest.mark.parametrize("free", [0, 1, 18])
def test_native_physical_shortage_precedes_experimental_budget(free):
    x = Fixture(free=free)
    assert x.load() is None
    assert not x.k.is_producer_budget_rejection(x.r)
    assert x.s._hot_tail_budget.ledger.entries == {}


@pytest.mark.parametrize("diagnostics", ["0", "1"])
def test_on_off_native_predictors_refs_and_call_order_equal(monkeypatch, diagnostics):
    """Successful ON adds metadata, not predictor, connector, or fence calls."""
    monkeypatch.setenv("VLLM_HOT_TAIL_ADMISSION_DIAGNOSTICS", diagnostics)
    snapshots = []
    for enabled in (False, True):
        x = Fixture(free=30, enabled=enabled)
        trace = Mock()
        with (
            patch.object(
                x.k.coordinator,
                "get_num_blocks_to_allocate",
                wraps=x.k.coordinator.get_num_blocks_to_allocate,
            ) as predictor,
            patch.object(
                x.k.coordinator,
                "allocate_new_blocks",
                wraps=x.k.coordinator.allocate_new_blocks,
            ) as allocation,
        ):
            trace.attach_mock(predictor, "predict")
            trace.attach_mock(allocation, "allocate")
            assert x.load() is not None
            x.complete_load()
            x.finish()
        snapshots.append((native_state(x), trace.mock_calls))
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "operation",
    ["get_num_blocks_to_allocate", "remove_skipped_blocks", "allocate_new_blocks"],
)
def test_native_allocator_exceptions_propagate(enabled, operation):
    x = Fixture(free=30, enabled=enabled)
    with (
        patch.object(x.k.coordinator, operation, side_effect=RuntimeError("native")),
        pytest.raises(RuntimeError, match="native"),
    ):
        x.load()
    if enabled:
        assert not x.s._hot_tail_budget.failed
        assert x.s._hot_tail_budget.ledger.entries == {}


@pytest.mark.parametrize("completed", [False, True])
def test_legacy_pending_dedup_uses_single_original_predictor_result(completed):
    x = Fixture(free=40)
    assert x.load() is not None
    if completed:
        x.complete_load()
    other = Request("other", [1] * 4, SamplingParams(max_tokens=1), None)
    x.s.requests[other.request_id] = other
    before = native_state(x)
    with patch.object(
        x.s, "_request_remaining_blocks", wraps=x.s._request_remaining_blocks
    ) as remaining:
        x.s._inflight_prefill_reserved_blocks()
        assert remaining.call_count == 1
        with patch.object(
            x.k.coordinator,
            "get_num_blocks_to_allocate",
            side_effect=AssertionError("extra predictor"),
        ):
            _, extra = x.s._hot_tail_budget.before(
                x.s, other, 0, {"delay_cache_blocks": True}
            )
            assert extra["producer_reserved_blocks"] == (0 if completed else 3)
            _, again = x.s._hot_tail_budget.before(x.s, other, 4, {})
            assert again["producer_reserved_blocks"] == 3
    assert native_state(x) == before


def test_future_survives_removal_from_inflight_prefills():
    x = Fixture(free=40, endpoint=47872)
    assert x.load() is not None
    x.s._inflight_prefills.clear()
    x.r.num_in_flight_tokens = 256
    other = Request("other", [1] * 4, SamplingParams(max_tokens=1), None)
    _, extra = x.s._hot_tail_budget.before(x.s, other, 4, {})
    assert extra == {"producer_reserved_blocks": 3, "future_blocks": 0}
    assert not x.s._hot_tail_budget.failed


@pytest.mark.parametrize("recompute", [False, True])
def test_failed_receive_forgets_only_metadata_and_prevents_virgin_retry(recompute):
    x = Fixture()
    assert x.load() is not None
    x.s.recompute_kv_load_failures = recompute
    before = native_state(x)
    affected = x.s._handle_failed_recving({x.r.request_id})
    assert affected == (set() if recompute else {x.r.request_id})
    assert native_state(x) == before
    assert not x.s._hot_tail_budget.ledger.entries
    x.r.status = RequestStatus.WAITING
    x.r.num_computed_tokens = 0
    plan, _ = x.s._hot_tail_budget.before(
        x.s,
        x.r,
        0,
        {"delay_cache_blocks": True, "num_external_computed_tokens": x.endpoint},
    )
    assert plan is None


def test_metadata_failure_latches_off_but_native_exception_stays_visible():
    x = Fixture()
    with (
        patch.object(
            x.s._hot_tail_budget, "before", side_effect=ValueError("metadata")
        ),
        patch.object(x.k, "allocate_slots", side_effect=RuntimeError("native")),
        pytest.raises(RuntimeError, match="native"),
    ):
        x.load()
    assert x.s._hot_tail_budget.failed
    assert not x.s._hot_tail_budget.ledger.entries


def test_free_and_free_fence_exception_preserve_native_ownership():
    x = Fixture()
    assert x.load() is not None
    x.complete_load()
    x.s.defer_block_free = False
    with patch.object(
        x.s, "_request_blocks_can_be_freed", side_effect=RuntimeError("fence")
    ):
        before = native_state(x)
        with pytest.raises(RuntimeError, match="fence"):
            x.s._free_request_blocks(x.r)
        assert native_state(x) == before
    assert not x.s._hot_tail_budget.ledger.entries
    x.s._free_request_blocks(x.r)
    assert x.k.block_pool.get_num_free_blocks() == 22


def test_progress_and_fault_diagnostics_are_bounded(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_HOT_TAIL_ADMISSION_DIAGNOSTICS", True)
    x = Fixture()
    budget = x.s._hot_tail_budget
    for _ in range(30):
        for event in ("enrolled", "deferred", "spent"):
            budget.progress_once(event, "request")
    assert budget.progress_seen == {"enrolled", "deferred", "spent"}
    for _ in range(40):
        budget.fault("test")
    assert budget.failed
    assert len(budget.diagnostics) == budget.DIAGNOSTIC_LIMIT
    assert budget.diagnostic_count == 40


def test_helper_has_no_native_capacity_predictor_or_gpu_calls():
    module = __import__(HotTailBudget.__module__, fromlist=["x"])
    tree = ast.parse(Path(module.__file__).read_text())
    calls = [
        ast.unparse(n.func).split(".")[-1]
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    ]
    assert not set(calls) & {
        "get_num_blocks_to_allocate",
        "get_num_free_blocks",
        "_needs_internal_checkpoint",
        "_request_blocks_can_be_freed",
        "cache_blocks",
        "free",
        "synchronize",
    }
