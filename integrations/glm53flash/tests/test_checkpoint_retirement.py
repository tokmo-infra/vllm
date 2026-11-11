# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test complete-cohort retirement and the actual sender without CUDA imports.

Exact descriptors and persistence statuses permit only non-force old-key
removal. Failed saves, stale jobs, missing groups and leases preserve old KV.
"""

import ast
import importlib.util
import logging
import sys
import threading
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[3]
STORE = Path("vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store")
CAND = ROOT / STORE
SUPPORT = ROOT / "vllm/v1"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


retire = load("retire_contract_candidate", CAND / "retire_checkpoints.py")


def extract(path, name, parent=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if parent:
        nodes = next(
            x for x in nodes if isinstance(x, ast.ClassDef) and x.name == parent
        ).body
    return next(x for x in nodes if getattr(x, "name", None) == name)


def execute(nodes, namespace):
    source = "from __future__ import annotations\n" + "\n".join(
        ast.unparse(node) for node in nodes
    )
    exec(compile(source, "<frozen-source-ast>", "exec"), namespace)
    return namespace


wp = CAND / "worker.py"
methods = ["_boundary_puts", "_maybe_offload_boundary_states", "_handle_request"]
ns = execute(
    [extract(wp, n, "KVCacheStoreSendingThread") for n in methods],
    {
        "time": time,
        "logger": logging.getLogger("cpu"),
        "NULL_BLOCK_ID": 0,
        "MOONCAKE_NO_AVAILABLE_HANDLE": -200,
        "_sum_batch_bytes": lambda sizes: sum(map(sum, sizes)),
        "cdiv": lambda a, b: (a + b - 1) // b,
    },
)
Sender = type(
    "SenderExtractedFromActualPlusNarrowPatch", (), {n: ns[n] for n in methods}
)
pos = execute(
    [
        extract(
            SUPPORT / "kv_cache_interface.py", "get_mamba_prefill_checkpoint_position"
        )
    ],
    {},
)["get_mamba_prefill_checkpoint_position"]


class FakeDB:
    def __init__(self, group, block_size):
        self.group = group
        self.block_size = block_size
        self.store_layout = NS(local_shard_ids=(0,))
        self.metadata = NS(group_id=group)

    def key_for(self, h):
        return f"g{self.group}@{h.hex()}"

    def prepare_value_for_block(self, bid):
        return [bid * 4096], [1024]


class FakeStore:
    def __init__(self):
        self.keys = set()
        self.calls = []
        self.exists_override = None
        self.put_override = None
        self.remove_override = None
        self.refuse = False

    def batch_is_exist(self, keys):
        self.calls.append(("exists", tuple(keys)))
        if isinstance(self.exists_override, Exception):
            raise self.exists_override
        if self.exists_override is not None:
            return self.exists_override
        return [int(k in self.keys) for k in keys]

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, config):
        self.calls.append(("put", tuple(keys)))
        if isinstance(self.put_override, Exception):
            raise self.put_override
        result = self.put_override if self.put_override is not None else [0] * len(keys)
        for k, code in zip(keys, result):
            if code >= 0:
                self.keys.add(k)
        return result

    def batch_remove(self, keys, force):
        self.calls.append(("remove", tuple(keys), force))
        assert force is False
        if isinstance(self.remove_override, Exception):
            raise self.remove_override
        if self.remove_override is not None:
            return self.remove_override
        if self.refuse:
            return [-706] * len(keys)
        self.keys.difference_update(keys)
        return [0] * len(keys)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.clock = 100.0
        retire.time = NS(monotonic=lambda: self.clock)
        self.s = s = Sender()
        s.store = FakeStore()
        s.coord = NS(
            hash_block_size=4,
            lcm_block_size=16,
            use_eagle=False,
            mamba_group_ids={0, 1},
            eagle_proof_margin_by_group={},
        )
        s.block_size = 16
        s.tp_rank = 0
        s.enable_group_semantics = False
        s.supports_group_ids = False
        s.replicate_config = NS()
        s.token_databases = [FakeDB(g, b) for g, b in enumerate([16, 16, 4, 8])]
        s.group_put_steps = [1] * 4
        s.group_participates = [True] * 4
        s.done_task_lock = threading.Lock()
        s.stored_requests = {"req": {7}}
        s.enable_kv_event = False
        s._record_operation = lambda *a, **kw: None
        s._clear_store_pressure = lambda: False
        s._mark_request_skipped_for_pressure = lambda meta: None
        s._should_skip_request = lambda req_id: False
        s.is_live_store_job = lambda meta: (
            meta.store_job_id in s.stored_requests.get(meta.req_id, ())
        )
        self.finished = []
        s.finish_store_job = lambda meta: self.finished.append(meta.store_job_id)
        s.request_queue = NS(task_done=lambda: self.finished.append("task_done"))
        s._checkpoint_retirer = retire.CheckpointRetirer(s)
        self.r = s._checkpoint_retirer
        self.m = NS(
            req_id="req",
            store_job_id=7,
            superseded_boundary=4,
            successor_boundary=12,
            block_hashes=[bytes([i]) for i in range(1, 9)],
            completed_token_len=12,
            current_event=None,
            token_len_chunk=0,
            token_ids=None,
            token_ids_start=0,
            block_ids=(),
            num_prompt_tokens=13,
            prefill_end_tokens=13,
            can_save=True,
            boundary_puts=[
                (0, 10, 12),
                (1, 11, 12),
                (2, 12, 4),
                (2, 13, 8),
                (2, 14, 12),
                (3, 15, 8),
                (3, 16, 12),
            ],
        )
        self.old = {
            s.token_databases[g].key_for(self.m.block_hashes[0]) for g in (0, 1)
        }
        s.store.keys.update(self.old)

    def arm(self):
        p = self.r._plan(self.m)
        self.assertIsNotNone(p)
        self.s.store.keys.update(p.new_keys)
        self.r.after_persist(self.m)
        return p

    def remove_calls(self):
        return [c for c in self.s.store.calls if c[0] == "remove"]

    def test_default_off_and_fail_policy_unchanged(self):
        self.assertFalse(retire.validate_retirement_scope({}, 1, 1, 1, 1, "fail"))
        self.assertTrue(
            retire.validate_retirement_scope(
                {"remove_superseded_checkpoints": True}, 1, 1, 1, 1, "fail"
            )
        )

    def test_bad_config_and_nonlocal_topology_block(self):
        for bad in ("true", 1, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                retire.retirement_enabled({"remove_superseded_checkpoints": bad})
        for field, value in (
            ("enable_group_semantics", True),
            ("enable_group_semantics", " TRUE "),
            ("store_tp_size", 1),
            ("enable_store_tp_lcm", True),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                retire.validate_retirement_scope(
                    {"remove_superseded_checkpoints": True, field: value}, 1, 1, 1, 1
                )
        for topology in ((2, 1, 1, 1), (1, 2, 1, 1), (1, 1, 2, 1), (1, 1, 1, 2)):
            with self.subTest(topology=topology), self.assertRaises(ValueError):
                retire.validate_retirement_scope(
                    {"remove_superseded_checkpoints": True}, *topology
                )

    def test_missing_native_remove_blocks(self):
        with self.assertRaises(ValueError):
            retire.CheckpointRetirer(NS(store=NS()))

    def test_marks_only_admitted_exact_prompt_and_all_recurrent_groups(self):
        scheduler = NS(
            remove_superseded_checkpoints=True,
            kv_role="kv_both",
            _store_coord=self.s.coord,
            _block_size=16,
        )
        tracker = NS(prefill_end_tokens=13, admitted_tokens=4)
        self.m.superseded_boundary = self.m.successor_boundary = None
        retire.mark_prompt_retirement(scheduler, tracker, self.m, pos)
        self.assertEqual(
            (self.m.superseded_boundary, self.m.successor_boundary), (4, 12)
        )
        for change in ("resume", "consumer", "missing_group", "cold", "incomplete"):
            with self.subTest(change=change):
                scheduler.kv_role = "kv_consumer" if change == "consumer" else "kv_both"
                tracker.prefill_end_tokens = 14 if change == "resume" else 13
                tracker.admitted_tokens = 0 if change == "cold" else 4
                self.m.completed_token_len = 8 if change == "incomplete" else 12
                saved = self.m.boundary_puts
                if change == "missing_group":
                    self.m.boundary_puts = [p for p in saved if p[0] != 1]
                self.m.superseded_boundary = self.m.successor_boundary = None
                retire.mark_prompt_retirement(scheduler, tracker, self.m, pos)
                self.assertIsNone(self.m.superseded_boundary)
                self.m.boundary_puts = saved

    def test_real_metadata_arms_only_for_a_true_prefix_extension(self):
        namespace = execute(
            [
                extract(CAND / "data.py", n)
                for n in ("BoundaryPut", "RequestTracker", "ReqMeta")
            ],
            {"NamedTuple": NamedTuple, "dataclass": dataclass},
        )
        boundary, tracker_class, meta_class = (
            namespace[n] for n in ("BoundaryPut", "RequestTracker", "ReqMeta")
        )
        coord = NS(hash_block_size=256, use_eagle=True, mamba_group_ids={0, 1})
        scheduler = NS(
            remove_superseded_checkpoints=True,
            kv_role="kv_producer",
            _store_coord=coord,
            _block_size=8960,
        )
        for admitted, expected in [(1792, (None, None)), (1024, (1024, 1792))]:
            with self.subTest(admitted=admitted):
                tracker = tracker_class(
                    req_id="test",
                    token_len=2049,
                    allocated_block_ids=([101], [102]),
                    prefill_end_tokens=2049,
                    admitted_tokens=admitted,
                )
                meta = meta_class(
                    req_id="test",
                    token_len_chunk=0,
                    block_ids=tracker.allocated_block_ids,
                    block_hashes=[bytes([i]) for i in range(1, 9)],
                    can_save=True,
                    num_prompt_tokens=2049,
                    prefill_end_tokens=2049,
                    completed_token_len=2049,
                    boundary_puts=[boundary(0, 101, 1792), boundary(1, 102, 1792)],
                )
                retire.mark_prompt_retirement(scheduler, tracker, meta, pos)
                self.assertEqual(
                    (meta.superseded_boundary, meta.successor_boundary), expected
                )

    def test_full_kda_mla_indexer_cohort_delete_only_old_kda(self):
        p = self.arm()
        self.assertEqual(set(p.old_keys), self.old)
        self.assertEqual(len(p.new_keys), 7)
        self.clock += 11
        self.r.collect_due()
        self.assertFalse(self.old & self.s.store.keys)
        self.assertTrue(set(p.new_keys) <= self.s.store.keys)
        self.assertEqual(self.remove_calls(), [("remove", p.old_keys, False)])

    def test_not_a_timer_no_collection_before_delay(self):
        self.arm()
        self.r.collect_due()
        self.assertEqual(len(self.r.pending), 1)
        self.assertFalse(self.remove_calls())

    def test_missing_or_conflicting_source_never_plans(self):
        saved = self.m.boundary_puts
        for group in range(4):
            with self.subTest(group=group):
                self.m.boundary_puts = [p for p in saved if p[0] != group]
                self.assertIsNone(self.r._plan(self.m))
                sample = next(p for p in saved if p[0] == group)
                self.m.boundary_puts = saved + [(group, 999, sample[2])]
                self.assertIsNone(self.r._plan(self.m))
        self.m.boundary_puts = saved

    def test_missing_attention_or_indexer_proof_blocks(self):
        self.s.coord.eagle_proof_margin_by_group = {2: 4, 3: 4}
        self.assertIsNone(self.r._plan(self.m))
        self.m.completed_token_len = 16
        self.assertIsNone(self.r._plan(self.m))
        self.m.boundary_puts += [(2, 20, 16), (3, 21, 16)]
        # Indexer proof end 16 replaces its clipped key 12, not both.
        self.assertEqual(len(self.r._plan(self.m).new_keys), 8)

    def test_nonlocal_shards_striped_puts_and_groups_block(self):
        for field in ("participates", "stripe", "shards", "native_group"):
            with self.subTest(field=field):
                self.s.group_participates[2] = field != "participates"
                self.s.group_put_steps[2] = 2 if field == "stripe" else 1
                self.s.token_databases[2].store_layout.local_shard_ids = (
                    (0, 1) if field == "shards" else (0,)
                )
                self.s.enable_group_semantics = field == "native_group"
                self.assertIsNone(self.r._plan(self.m))

    def test_aligned_or_invalid_boundaries_never_delete(self):
        for old, new in ((0, 12), (4, 4), (16, 28), (4, 16), (3, 12), (4, 11), (4, 36)):
            with self.subTest(old=old, new=new):
                self.m.superseded_boundary, self.m.successor_boundary = old, new
                self.assertIsNone(self.r._plan(self.m))

    def test_stale_request_generation_never_enqueues(self):
        p = self.r._plan(self.m)
        self.s.store.keys.update(p.new_keys)
        self.s.stored_requests["req"] = {8}
        self.r.after_persist(self.m)
        self.assertFalse(self.r.pending)

    def test_gc_rechecks_only_new_keys_never_renews_old_lease(self):
        self.arm()
        self.clock += 11
        self.r.collect_due()
        queried = {k for c in self.s.store.calls if c[0] == "exists" for k in c[1]}
        self.assertFalse(queried & self.old)

    def test_committed_full_block_state_is_retained(self):
        self.s.token_databases[1].block_size = 4
        p = self.arm()
        self.assertEqual(len(p.old_keys), 1)
        self.clock += 11
        self.r.collect_due()
        full_key = self.s.token_databases[1].key_for(self.m.block_hashes[0])
        self.assertIn(full_key, self.s.store.keys)

    def test_partial_malformed_or_exception_exists_never_enqueues(self):
        for result in ([], [1], [1] * 6 + [0], [1] * 6 + [-1], RuntimeError("offline")):
            with self.subTest(result=result):
                self.s.store.exists_override = result
                self.r.after_persist(self.m)
                self.assertFalse(self.r.pending)

    def test_evicted_successor_on_due_is_not_removed(self):
        p = self.arm()
        self.s.store.keys.discard(p.new_keys[-1])
        self.clock += 11
        self.r.collect_due()
        self.assertFalse(self.remove_calls())
        self.assertTrue(self.old <= self.s.store.keys)

    def test_lease_refusal_exception_and_malformed_remove_are_not_retried(self):
        for result in ([-706, -706], [], RuntimeError("offline")):
            with self.subTest(result=result):
                self.arm()
                self.s.store.remove_override = result
                before = len(self.remove_calls())
                self.clock += 11
                self.r.collect_due()
                self.r.collect_due()
                self.assertEqual(len(self.remove_calls()), before + 1)
                self.assertFalse(self.r.pending)

    def test_real_sender_persists_then_enqueues_and_always_finishes(self):
        self.s._handle_request(self.m)
        self.assertEqual(len(self.r.pending), 1)
        self.assertEqual(self.finished, [7, "task_done"])

    def test_real_sender_failed_short_or_exception_put_never_arms(self):
        for result in ([-1] * 7, [], RuntimeError("offline")):
            with self.subTest(result=result):
                self.s.store.keys = set(self.old)
                self.s.store.put_override = result
                self.s._handle_request(self.m)
                self.assertFalse(self.r.pending)
                self.assertTrue(self.old <= self.s.store.keys)
        self.assertEqual(self.finished, [7, "task_done"] * 3)

    def test_real_sender_short_exists_not_success_even_if_no_missing(self):
        for result in ([], [1]):
            with self.subTest(result=result):
                self.s.store.exists_override = result
                self.s._handle_request(self.m)
                self.assertFalse(self.r.pending)
                self.assertFalse([c for c in self.s.store.calls if c[0] == "put"])

    def test_real_sender_partial_put_and_later_retry_require_all_groups(self):
        self.s.store.put_override = [0] * 6 + [-1]
        self.s._handle_request(self.m)
        self.assertFalse(self.r.pending)
        self.s.store.put_override = None
        self.s._handle_request(self.m)
        self.assertEqual(len(self.r.pending), 1)
        put_calls = [c for c in self.s.store.calls if c[0] == "put"]
        self.assertEqual([len(c[1]) for c in put_calls], [7, 1])

    def test_real_sender_event_precedes_dma_put(self):
        self.m.current_event = NS(
            synchronize=lambda: self.s.store.calls.append(("event",))
        )
        self.s._handle_request(self.m)
        names = [c[0] for c in self.s.store.calls]
        self.assertLess(names.index("event"), names.index("put"))

    def test_feature_off_real_sender_has_no_gc_calls(self):
        self.s._checkpoint_retirer = None
        self.s._handle_request(self.m)
        self.clock += 11
        self.s._handle_request(self.m)
        self.assertFalse(self.remove_calls())
        self.assertTrue(self.old <= self.s.store.keys)

    def test_real_sender_stale_job_runs_finally_without_gc(self):
        self.s.stored_requests = {}
        self.s._handle_request(self.m)
        self.assertFalse(self.s.store.calls)
        self.assertEqual(self.finished, [7, "task_done"])
