# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental metadata-only admission budget for producer tail CoW.

Disabled by default. This narrow ledger does not allocate blocks, change
references, issue connector requests, or synchronize GPU work.
"""

import os
from collections import deque
from dataclasses import dataclass

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    FullAttentionManager,
    MambaManager,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.request import RequestStatus


def ceildiv(x, y):
    return (x + y - 1) // y


@dataclass(frozen=True)
class Plan:
    request_id: str
    generation: int
    prompt: int
    endpoint: int
    groups: tuple


def prepare(
    request,
    managers,
    endpoint,
    *,
    async_load,
    is_producer,
    local_tokens=0,
    local_blocks=False,
    replay_tokens=0,
    num_reprefillable_tokens=0,
    retention_interval=0,
    other_layout=(),
    lookahead_tokens=0,
):
    """Reject unsupported geometry, rather than promise a general peak budget."""
    if (
        not async_load
        or not is_producer
        or local_tokens
        or local_blocks
        or replay_tokens
        or num_reprefillable_tokens
        or retention_interval != 0
        or getattr(request.status, "name", request.status) != "WAITING"
        or request.num_computed_tokens != 0
        or request.num_in_flight_tokens != 0
        or request.num_preemptions != 0
        or request.num_tokens != request.num_prompt_tokens
    ):
        return None
    # A real adapter must derive these descriptors from exact supported manager
    # types, rejecting unknown subclasses. Mamba-only studies pass an empty list.
    for kind, block in other_layout:
        if (
            kind not in ("full_attention", "circular")
            or block <= 0
            or lookahead_tokens < 0
        ):
            return None
        if kind == "full_attention" and ceildiv(endpoint, block) != ceildiv(
            request.num_prompt_tokens + lookahead_tokens, block
        ):
            return None
    groups = []
    for manager in managers:
        rid = request.request_id
        if (
            not manager.enable_caching
            or manager.mamba_cache_mode != "align"
            or manager.shared_prefix_checkpoint
            or manager.req_to_blocks.get(rid)
            or rid in manager._allocated_block_reqs
            or rid in manager._checkpoints
            or rid in manager._partial_hit_reqs
            or rid in manager._producer_partial_tail_reqs
        ):
            return None
        block = manager.block_size
        hash_block = manager.block_pool.hash_block_size
        prompt = request.num_prompt_tokens
        limit = prompt if manager.drop_eagle_checkpoint_block else prompt - 1
        boundary = limit // hash_block * hash_block
        if manager.drop_eagle_checkpoint_block:
            boundary -= hash_block
        boundary = max(boundary, 0)
        if not (
            0 < endpoint <= boundary < prompt
            and boundary % block != 0
            and ceildiv(endpoint, block) == ceildiv(prompt, block)
        ):
            return None
        groups.append((manager.kv_cache_group_id, boundary))
    if not groups or len({g for g, _ in groups}) != len(groups):
        return None
    return Plan(
        request.request_id,
        id(request),
        request.num_prompt_tokens,
        endpoint,
        tuple(groups),
    )


class Ledger:
    def __init__(self):
        self.entries = {}
        self.violations = []

    def commit_successful_load(self, plan, request, managers):
        assert plan.generation == id(request) and plan.request_id == request.request_id
        assert plan.request_id not in self.entries
        indexed = {m.kv_cache_group_id: m for m in managers}
        for gid, _ in plan.groups:
            manager = indexed[gid]
            assert manager.req_to_blocks.get(plan.request_id)
            assert plan.request_id in manager._allocated_block_reqs
        self.entries[plan.request_id] = dict(
            plan=plan, phases={gid: "future" for gid, _ in plan.groups}
        )

    def forget(self, request_id):
        """Only metadata: the original allocator retains ownership of all blocks."""
        self.entries.pop(request_id, None)

    def observe(self, request, managers, *, finished=False, load_failed=False):
        rid = request.request_id
        entry = self.entries.get(rid)
        if entry is None:
            return
        plan = entry["plan"]
        if (
            plan.generation != id(request)
            or finished
            or load_failed
            or request.num_preemptions
            or request.num_prompt_tokens != plan.prompt
        ):
            self.forget(rid)
            return
        indexed = {m.kv_cache_group_id: m for m in managers}
        for gid, boundary in plan.groups:
            manager = indexed[gid]
            marker = manager._producer_partial_tail_reqs.get(rid)
            pending = marker is not None and rid in manager._partial_hit_reqs
            if pending:
                if marker[1] != boundary:
                    self.violations.append((rid, gid, "boundary_changed"))
                if entry["phases"][gid] == "spent":
                    self.violations.append((rid, gid, "republication_after_spend"))
                entry["phases"][gid] = "pending"

    def allocation_succeeded(self, request, managers):
        """Call only after the native all-group allocation returned success."""
        entry = self.entries.get(request.request_id)
        if entry is None or entry["plan"].generation != id(request):
            return
        indexed = {m.kv_cache_group_id: m for m in managers}
        for gid, _ in entry["plan"].groups:
            m = indexed[gid]
            if (
                entry["phases"][gid] == "pending"
                and request.request_id not in m._producer_partial_tail_reqs
                and request.request_id not in m._partial_hit_reqs
            ):
                entry["phases"][gid] = "spent"
        self.observe(request, managers)

    def budgets(self, current, legacy_covered_ids=()):
        """Additional counts only, with legacy pending counted by identity.

        Requires the exact supported native remaining predictor to cover pending
        producer groups for legacy_covered_ids. This interface does not call it.
        Caller supplies original-branch identities; ordinary WAITING/RUNNING use
        an empty set. No raw scalar max/sum merge is guessed here.
        """
        other = future_current = 0
        covered = set(legacy_covered_ids)
        for rid, entry in self.entries.items():
            same = rid == current.request_id and entry["plan"].generation == id(current)
            for phase in entry["phases"].values():
                if phase == "spent":
                    continue
                if same:
                    future_current += phase == "future"
                elif phase == "future" or rid not in covered:
                    other += 1
        return other, future_current


# Runtime adapter. Native allocator / connector / predictor calls stay outside
# every metadata exception boundary. No Request, Block or frame is retained.
logger = init_logger(__name__)


class HotTailBudget:
    SCHEMA = 1
    DIAGNOSTIC_LIMIT = 16
    PROGRESS_EVENTS = ("enrolled", "deferred", "spent")

    def __init__(self):
        self.ledger = Ledger()
        self.seen_loads = {}
        self.failed = False
        self.diagnostic_count = 0
        self.diagnostic_dropped = 0
        self.diagnostics: deque[tuple[int, str, str]] = deque(
            maxlen=self.DIAGNOSTIC_LIMIT
        )
        self.legacy: tuple[tuple[str, int, int], ...] = ()
        self.progress_enabled = envs.VLLM_HOT_TAIL_ADMISSION_DIAGNOSTICS
        self.progress_seen = set()
        self.progress_dropped = 0

    def progress_once(self, event, request_id, groups=-1):
        # At most THREE attempts per engine lifetime, not per request.
        if (
            not self.progress_enabled
            or self.failed
            or event not in self.PROGRESS_EVENTS
            or event in self.progress_seen
        ):
            return
        try:
            self.progress_seen.add(event)  # no retry storm when logging fails
            logger.info(
                "P_HOT_TAIL_PROGRESS schema=%s pid=%s event=%s request=%r "
                "groups=%s dropped=%s",
                self.SCHEMA,
                os.getpid(),
                event,
                str(request_id)[:128],
                groups,
                self.progress_dropped,
            )
        except Exception:
            self.progress_dropped += 1

    def record_deferred(self, manager, request):
        if not self.progress_enabled or self.failed:
            return
        try:
            # Reuse the original contract scalar, not its predicate/any guard.
            if manager._producer_budget_rejection_request_id == request.request_id:
                self.progress_once("deferred", request.request_id)
        except Exception:
            self.progress_dropped += 1

    def progress_spent_count(self, request_id):
        if not self.progress_enabled or "spent" in self.progress_seen:
            return None
        try:
            entry = self.ledger.entries.get(request_id)
            return sum(p == "spent" for p in entry["phases"].values()) if entry else 0
        except Exception:
            self.progress_dropped += 1
            return None

    def fault(self, code, request_id=""):
        # Fail OPEN to unchanged native policy, and invalidate the experiment.
        # The external bounded-test controller must stop on stop_required=true.
        self.failed = True
        self.ledger.entries.clear()
        self.ledger.violations.clear()
        self.legacy = ()
        self.diagnostic_count += 1
        row = (self.diagnostic_count, str(code)[:64], str(request_id)[:128])
        if len(self.diagnostics) == self.DIAGNOSTIC_LIMIT:
            self.diagnostic_dropped += 1
        self.diagnostics.append(row)
        if self.diagnostic_count <= self.DIAGNOSTIC_LIMIT:
            try:
                logger.error(
                    "P_HOT_TAIL_BUDGET schema=%s stop_required=true seq=%s "
                    "code=%s request=%s dropped=%s",
                    self.SCHEMA,
                    row[0],
                    row[1],
                    row[2],
                    self.diagnostic_dropped,
                )
            except Exception:
                self.diagnostic_dropped += 1

    @staticmethod
    def supported_producer(transfer):
        # Verified P uses MultiConnector/kv_both at the top level; D also does.
        # Identify P by its exact children, never by engine_id or consumer flag.
        if (
            transfer is None
            or transfer.kv_connector != "MultiConnector"
            or transfer.kv_role != "kv_both"
            or getattr(transfer, "kv_connector_module_path", None) is not None
            or transfer.kv_load_failure_policy != "fail"
        ):
            return False
        extra = transfer.kv_connector_extra_config
        if type(extra) is not dict or set(extra) != {"connectors"}:
            return False
        children = extra["connectors"]
        if type(children) is not list or len(children) != 2:
            return False
        required = {"kv_connector", "kv_role", "kv_connector_extra_config"}
        if any(
            type(c) is not dict
            or not required <= set(c)
            or set(c) - required - {"kv_connector_module_path"}
            or c.get("kv_connector_module_path") is not None
            for c in children
        ):
            return False
        moon, store = children
        if (
            moon["kv_connector"] != "MooncakeConnector"
            or moon["kv_role"] != "kv_producer"
            or store["kv_connector"] != "MooncakeStoreConnector"
            or store["kv_role"] != "kv_both"
        ):
            return False
        mc, sc = moon["kv_connector_extra_config"], store["kv_connector_extra_config"]
        return (
            type(mc) is dict
            and set(mc) == {"mooncake_protocol", "device_name"}
            and mc["mooncake_protocol"] == "rdma"
            and isinstance(mc["device_name"], str)
            and bool(mc["device_name"])
            and type(sc) is dict
            and set(sc)
            == {
                "cache_prefix",
                "load_async",
                "lookup_async",
                "store_compact_group_payloads",
            }
            and isinstance(sc["cache_prefix"], str)
            and bool(sc["cache_prefix"])
            and sc["load_async"] is True
            and sc["lookup_async"] is True
            and sc["store_compact_group_payloads"] is True
        )

    @staticmethod
    def layout(scheduler):
        k = scheduler.kv_cache_manager
        c = k.coordinator
        transfer = scheduler.vllm_config.kv_transfer_config
        if (
            not HotTailBudget.supported_producer(transfer)
            or not scheduler.scheduler_config.async_scheduling
            or not k.enable_caching
            or c.retention_interval != 0
            or c.num_reprefillable_tokens != 0
            or scheduler.prefix_replay_tokens
            or k.mamba_shared_prefix_checkpoint
            or scheduler.is_encoder_decoder
            or scheduler.dcp_world_size != 1
            or scheduler.pcp_world_size != 1
        ):
            return None
        mamba, other = [], []
        for m in c.single_type_managers:
            # Exact classes, not isinstance, class names or duck-typed layouts.
            if type(m) is MambaManager and type(m.kv_cache_spec) is MambaSpec:
                if (
                    m.mamba_cache_mode != "align"
                    or m.shared_prefix_checkpoint
                    or not m.enable_caching
                ):
                    return None
                mamba.append(m)
            elif type(m) is FullAttentionManager and type(m.kv_cache_spec) in (
                FullAttentionSpec,
                MLAAttentionSpec,
            ):
                other.append(("full_attention", m.block_size))
            elif (
                type(m) is CircularBufferManager
                and type(m.kv_cache_spec) is CircularBufferSpec
            ):
                other.append(("circular", m.block_size))
            else:
                return None
        return (mamba, other) if mamba else None

    @classmethod
    def create(cls, scheduler):
        result = cls()
        try:
            supported = result.layout(scheduler) is not None
            logger.info(
                "P_HOT_TAIL_BUDGET init schema=%s enabled=%s diagnostic_limit=%s "
                "progress_enabled=%s progress_limit=3",
                cls.SCHEMA,
                supported,
                cls.DIAGNOSTIC_LIMIT,
                result.progress_enabled,
            )
            return result if supported else None
        except Exception as exc:
            result.fault("init_" + type(exc).__name__)
            return result

    def begin_legacy(self):
        self.legacy = ()

    def record_legacy(self, request, amount):
        # Called next to the original remaining() result, never recomputes it.
        if not self.failed:
            self.legacy += ((request.request_id, id(request), amount),)

    def forget(self, request):
        self.ledger.forget(request.request_id)

    def refresh(self, scheduler, managers):
        failed = scheduler.failed_recving_kv_req_ids
        for rid, generation in tuple(self.seen_loads.items()):
            r = scheduler.requests.get(rid)
            if r is None or generation != id(r):
                self.seen_loads.pop(rid, None)
        for rid in tuple(self.ledger.entries):
            request = scheduler.requests.get(rid)
            if request is None:
                self.ledger.forget(rid)
                continue
            self.ledger.observe(
                request,
                managers,
                finished=request.status > RequestStatus.PREEMPTED,
                load_failed=rid in failed,
            )
        if self.ledger.violations:
            self.fault(self.ledger.violations[0][2], self.ledger.violations[0][0])
            return {}
        pending: dict[str, set[int]] = {}
        for m in managers:
            for rid in m._producer_partial_tail_reqs:
                if rid not in self.ledger.entries:
                    continue
                request = scheduler.requests.get(rid)
                if (
                    request is None
                    or rid in failed
                    or request.status > RequestStatus.PREEMPTED
                ):
                    continue
                if rid in m._partial_hit_reqs and rid in m._allocated_block_reqs:
                    pending.setdefault(rid, set()).add(m.kv_cache_group_id)
        # Never interpret missing pending/allocated state as a spent claim.
        for rid, entry in self.ledger.entries.items():
            for gid, phase in entry["phases"].items():
                m = next(m for m in managers if m.kv_cache_group_id == gid)
                if rid not in m._allocated_block_reqs or not m.req_to_blocks.get(rid):
                    self.fault("tracked_owner_without_table", rid)
                    return {}
                if rid in m._checkpoints:
                    self.fault("unexpected_internal_checkpoint", rid)
                    return {}
                if phase == "pending" and gid not in pending.get(rid, ()):
                    self.fault("pending_without_allocated_marker", rid)
                    return {}
        return pending

    def before(self, scheduler, request, tokens, kwargs):
        proof, self.legacy = self.legacy, ()
        if self.failed:
            return None, {}
        layout = self.layout(scheduler)
        if layout is None:
            self.fault("layout_changed", request.request_id)
            return None, {}
        managers, other_layout = layout
        pending = self.refresh(scheduler, managers)
        if self.failed:
            return None, {}
        covered: set[str] = set()
        if kwargs.get("delay_cache_blocks", False):
            for rid, generation, amount in proof:
                owner = scheduler.requests.get(rid)
                if owner is not None and generation == id(owner):
                    count = len(pending.get(rid, ()))
                    if count and amount < count:
                        self.fault("legacy_under_covers_pending", rid)
                        return None, {}
                    covered.add(rid)
        others, future = self.ledger.budgets(request, covered)
        # Do not widen the proof to un-enrolled pending producers: those may
        # still require independent checkpoint / cross-block physical growth.
        plan = None
        if (
            tokens == 0
            and request.status == RequestStatus.WAITING
            and request.num_prompt_tokens <= scheduler.max_model_len
            and not request.has_encoder_inputs
            and not request.shared_prefix_boundary
            and request.request_id not in self.ledger.entries
            and self.seen_loads.get(request.request_id) != id(request)
            and all(
                not m.req_to_blocks.get(request.request_id)
                for m in scheduler.kv_cache_manager.coordinator.single_type_managers
            )
        ):
            local = kwargs.get("num_new_computed_tokens", 0)
            endpoint = min(
                request.num_computed_tokens
                + local
                + kwargs.get("num_external_computed_tokens", 0),
                scheduler.max_model_len,
            )
            blocks = kwargs.get("new_computed_blocks")
            plan = prepare(
                request,
                managers,
                endpoint,
                async_load=kwargs.get("delay_cache_blocks", False),
                is_producer=True,
                local_tokens=local,
                local_blocks=blocks is not None and any(blocks.blocks),
                replay_tokens=scheduler.prefix_replay_tokens,
                num_reprefillable_tokens=scheduler.kv_cache_manager.coordinator.num_reprefillable_tokens,
                retention_interval=scheduler.kv_cache_manager.coordinator.retention_interval,
                other_layout=other_layout,
                lookahead_tokens=scheduler.num_lookahead_tokens,
            )
            if plan is not None:
                future += len(plan.groups)
        return plan, dict(producer_reserved_blocks=others, future_blocks=future)

    def success(self, scheduler, request, plan):
        if self.failed:
            return
        managers, _ = self.layout(scheduler)
        spent_before = self.progress_spent_count(request.request_id)
        if plan is not None:
            self.ledger.commit_successful_load(plan, request, managers)
            self.seen_loads[request.request_id] = id(request)
        self.ledger.allocation_succeeded(request, managers)
        if self.ledger.violations:
            self.fault(self.ledger.violations[0][2], request.request_id)
        if not self.failed:
            if plan is not None:
                self.progress_once("enrolled", request.request_id, len(plan.groups))
            spent_after = self.progress_spent_count(request.request_id)
            if (
                spent_before is not None
                and spent_after is not None
                and spent_after > spent_before
            ):
                self.progress_once(
                    "spent", request.request_id, spent_after - spent_before
                )
