# SPDX-License-Identifier: Apache-2.0
"""Per-EngineCore observer. No workers, cross-rank clocks or scheduler actions."""

import json
import time
from collections import Counter


class ReadyWait:
    # Non-cumulative upper bounds, final bucket is +Inf.
    MS = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000)

    @staticmethod
    def producer(transfer):
        if (
            not transfer
            or transfer.kv_connector != "MultiConnector"
            or transfer.kv_role != "kv_both"
            or transfer.kv_connector_module_path
        ):
            return False
        children = transfer.kv_connector_extra_config.get("connectors", [])
        # With this exact pair, only the Store child can return an async hit
        # on P (the peer connector rejects do_remote_prefill on a producer).
        return [(c.get("kv_connector"), c.get("kv_role")) for c in children] == [
            ("MooncakeConnector", "kv_producer"),
            ("MooncakeStoreConnector", "kv_both"),
        ] and not any(c.get("kv_connector_module_path") for c in children)

    def __init__(self, dp, logger, clock=time.monotonic_ns):
        self.dp = dp
        self.logger = logger
        self.clock = clock
        self.loading = set()
        self.ready = {}
        self.events = Counter()
        self.blocked = Counter()
        self.hist = [0] * (len(self.MS) + 1)
        self.sum_ns = self.max_ns = 0
        self.next_report = clock() + 10_000_000_000

    def admitted(self, rid):
        self.loading.add(rid)
        self.events["async_load"] += 1

    def received(self, rid):
        if rid in self.loading:
            self.loading.remove(rid)
            self.ready[rid] = self.clock()
            self.events["notified"] += 1

    def discard(self, rid, why):
        if rid in self.loading:
            self.loading.remove(rid)
            self.events[why + "_loading"] += 1
        if self.ready.pop(rid, None) is not None:
            self.events[why + "_ready"] += 1

    def missed(self, tokens, direct, fallback):
        # One observation per ready, unscheduled request per scheduling pass.
        # Never assign a head request's allocation failure to its followers.
        for rid in self.ready:
            if not tokens.get(rid, 0):
                self.blocked[direct.get(rid, fallback)] += 1

    def submitted(self, tokens):
        now = self.clock()
        if self.ready:
            for rid, count in tokens.items():
                if count <= 0:
                    continue
                start = self.ready.pop(rid, None)
                if start is None:
                    continue
                dt = now - start
                self.events["dispatched"] += 1
                self.sum_ns += dt
                self.max_ns = max(self.max_ns, dt)
                bucket = next(
                    (i for i, ms in enumerate(self.MS) if dt <= ms * 1_000_000),
                    len(self.MS),
                )
                self.hist[bucket] += 1
        if now >= self.next_report:
            self.report()
            self.next_report = now + 10_000_000_000

    def report(self):
        self.logger.info(
            "P_READY_WAIT %s",
            json.dumps(
                dict(
                    dp=self.dp,
                    events=self.events,
                    loading=len(self.loading),
                    ready=len(self.ready),
                    blocked_passes=self.blocked,
                    wait_sum_ns=self.sum_ns,
                    wait_max_ns=self.max_ns,
                    bucket_upper_ms=(*self.MS, "+Inf"),
                    wait_bins=self.hist,
                ),
                separators=(",", ":"),
            ),
        )

    def close(self):
        self.events["shutdown_loading"] += len(self.loading)
        self.events["shutdown_ready"] += len(self.ready)
        self.loading.clear()
        self.ready.clear()
        self.report()
