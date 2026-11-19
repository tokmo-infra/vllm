"""CPU contracts: execute complete overlay methods without loading CUDA/vLLM."""

import ast
import importlib.util
import json
import time
import unittest
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from typing import cast
from unittest.mock import MagicMock, Mock

ROOT = Path(__file__).resolve().parents[1]
SCHED = ROOT / "vllm/v1/core/sched/scheduler.py"
CORE = ROOT / "vllm/v1/engine/core.py"
spec = importlib.util.spec_from_file_location(
    "ready_wait", SCHED.with_name("ready_wait.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ReadyWait = module.ReadyWait
PauseState = NS(UNPAUSED=0, PAUSED_ALL=1)
RequestStatus = NS(
    WAITING_FOR_REMOTE_KVS=1, FINISHED_ABORTED=2, is_finished=lambda s: s == 2
)


def method(path, cls, name):
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if getattr(n, "name", None) == cls)
    fn = next(n for n in owner.body if getattr(n, "name", None) == name)
    env = dict(
        globals(),
        SchedulerOutput=NS,
        record_function_or_nullcontext=nullcontext,
        KV_FETCH_COMPLETED_WAITING="completed",
        logger=Mock(),
        ModelRunnerOutput=object,
    )
    exec(
        compile(
            ast.Module(body=[fn], type_ignores=[]),
            str(path),
            "exec",
            flags=__import__("__future__").annotations.compiler_flag,
        ),
        env,
    )
    return env[name]


class Contracts(unittest.TestCase):
    def setUp(self):
        self.t = 0
        self.log = Mock()
        self.p = ReadyWait(3, self.log, lambda: self.t)

    def ready(self, rid):
        self.p.admitted(rid)
        self.p.received(rid)

    def test_roles(self):
        t = NS(
            kv_connector="MultiConnector",
            kv_role="kv_both",
            kv_connector_module_path=None,
            kv_connector_extra_config={
                "connectors": [
                    dict(kv_connector="MooncakeConnector", kv_role="kv_producer"),
                    dict(kv_connector="MooncakeStoreConnector", kv_role="kv_both"),
                ]
            },
        )
        self.assertTrue(ReadyWait.producer(t))
        t.kv_connector_extra_config["connectors"][0]["kv_role"] = "kv_consumer"
        self.assertFalse(ReadyWait.producer(t))
        self.assertFalse(ReadyWait.producer(None))

    def test_recv_duplicate_fail_abort_late_and_reuse(self):
        recv = method(SCHED, "Scheduler", "_update_from_kv_xfer_finished")
        fail = method(SCHED, "Scheduler", "_handle_failed_recving")
        free = method(SCHED, "Scheduler", "_free_request")
        s = MagicMock(
            ready_wait_probe=self.p,
            connector=None,
            requests={},
            failed_recving_kv_req_ids=set(),
            recompute_kv_load_failures=True,
        )
        rid = "PRIVATE-RID"
        r = MagicMock(request_id=rid, status=1)
        s.requests[rid] = r
        msg = NS(finished_recving={rid}, finished_sending=None)
        self.p.admitted(rid)
        self.t = 20
        recv(s, msg)
        self.t = 80
        recv(s, msg)
        self.assertEqual(self.p.ready, {rid: 20})
        self.assertEqual(fail(s, {rid}), set())
        recv(s, msg)
        self.assertFalse(self.p.ready)
        self.p.admitted(rid)
        fail(s, {rid})
        recv(s, msg)
        self.assertEqual(self.p.events["failed_recv_loading"], 1)
        self.p.admitted(rid)
        r.status = NS(name="FINISHED_ABORTED")
        r.is_finished.return_value = True
        s._connector_finished.return_value = False, None
        s.finished_req_ids_dict = None
        s.ec_connector = None
        free(s, r, delay_free_blocks=True)
        r.status = 2
        recv(s, msg)
        self.assertFalse(self.p.ready)
        s._free_blocks.assert_called_once_with(r)
        self.ready(rid)
        self.p.submitted({rid: 0, "unrelated": 2})
        self.t += 1_000_000
        self.p.submitted({rid: 1})
        self.p.submitted({rid: 1})
        self.assertEqual(self.p.events["dispatched"], 1)
        self.assertEqual(self.p.hist[0], 1)
        self.ready(rid)
        r.status = NS(name="FINISHED_ABORTED")
        free(s, r)
        self.assertEqual(self.p.events["finished_aborted_ready"], 1)
        self.log.info.assert_not_called()
        self.ready(rid)
        self.p.admitted("loading-secret")
        self.p.close()
        self.assertFalse(self.p.ready or self.p.loading)
        payload = self.log.info.call_args.args[1]
        self.assertNotIn(rid, payload)
        self.assertNotIn("loading-secret", payload)
        self.assertEqual(json.loads(payload)["dp"], 3)

    def test_real_schedule_reasons(self):
        schedule = method(SCHED, "Scheduler", "schedule")
        for case, expected in [
            ("token_budget", {"token_budget": 2}),
            ("token_input", {"token_input": 2}),
            ("seq", {"seq": 2}),
            ("pause", {"other_pause": 2}),
            ("kv", {"kv_alloc": 1, "other_queue": 1}),
        ]:
            with self.subTest(case=case):
                self.setUp()
                self.ready("head")
                self.ready("tail")
                q = MagicMock()
                q.peek_request.return_value = NS(
                    request_id="head",
                    num_computed_tokens=2,
                    num_tokens=3,
                    num_stale_output_tokens=0,
                    has_encoder_inputs=False,
                )
                s = MagicMock(
                    ready_wait_probe=self.p,
                    current_step=0,
                    running=[],
                    kv_holding_waiting=q,
                    _pause_state=int(case == "pause"),
                    max_num_scheduled_tokens=0 if case == "token_budget" else 4,
                    max_num_active_reqs=2,
                    num_waiting_for_streaming_input=2 if case == "seq" else 0,
                    max_num_running_reqs=2,
                    lora_config=None,
                    num_spec_tokens=0,
                    num_lookahead_tokens=0,
                    connector=None,
                    _hot_tail_budget=None,
                    log_stats=False,
                    is_encoder_decoder=False,
                    need_mamba_block_aligned_split=False,
                    dynamic_sd_lookup=None,
                    scheduler_config=NS(
                        max_num_batched_tokens=0 if case == "token_input" else 4,
                        long_prefill_token_threshold=0,
                        enable_chunked_prefill=True,
                    ),
                    vllm_config=NS(speculative_config=None),
                )
                s.get_request_counts.return_value = 0, 2
                s.kv_cache_manager.take_kv_cache_block_copies.return_value = [], []
                s._ec_transfer_pending.return_value = False
                s._reserve_prefill_lookahead.side_effect = lambda r, c, n: n
                s._allocate_slots_with_hot_tail.return_value = None
                out = schedule(s)
                self.assertEqual(out.total_num_scheduled_tokens, 0)
                self.assertEqual(self.p.blocked, expected)
                s.ready_wait_probe = None
                self.assertEqual(
                    schedule(s).num_scheduled_tokens, out.num_scheduled_tokens
                )
                self.assertEqual(self.p.blocked, expected)

    def test_both_submission_paths(self):
        for name in ("step", "step_with_batch_queue"):
            with self.subTest(name=name):
                self.setUp()
                self.ready("a")
                self.ready("b")
                step = method(CORE, "EngineCore", name)
                c = MagicMock(
                    batch_queue=deque(),
                    batch_queue_size=8,
                    is_mm_encoder_only=False,
                    is_pooling_model=True,
                )
                c.scheduler.ready_wait_probe = self.p
                out = NS(
                    num_scheduled_tokens={"a": 1, "b": 0}, total_num_scheduled_tokens=1
                )
                c.scheduler.schedule.return_value = out
                c.model_executor.execute_model.side_effect = RuntimeError("no submit")
                with self.assertRaisesRegex(RuntimeError, "no submit"):
                    step(c)
                self.assertEqual(set(self.p.ready), {"a", "b"})

                def submit(*args, **kwargs):
                    self.assertEqual(self.p.events["dispatched"], 0)
                    self.t = 6_000_000
                    return MagicMock()

                c.model_executor.execute_model.side_effect = submit
                step(c)
                self.assertEqual(self.p.sum_ns, 6_000_000)
                self.assertEqual(self.p.hist[2], 1)
                self.assertEqual(set(self.p.ready), {"b"})

    def test_rate_limit(self):
        for self.t in (0, 1, 10_000_000_000, 10_000_000_001, 21_000_000_000):
            self.p.submitted({})
        self.assertEqual(self.log.info.call_count, 2)


if __name__ == "__main__":
    unittest.main()
