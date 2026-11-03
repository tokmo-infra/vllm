# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Retain compact native request metrics while preserving summary-only storage."""

import json
import os
import time
from pathlib import Path

from diagnostic_client_run import install
from perf_benchmark.pipeline.contracts.results import (
    AttemptResultRecord,
    TurnResultRecord,
)
from perf_benchmark.plugins.saver.sqlite.summary_retention import SummaryRetention
from perf_benchmark.runtime.http.rust_executor import RustHttpExecutorRuntime

root = Path(os.environ["BENCH_RUN_DIR"])
root.mkdir(parents=True, exist_ok=True)

_diagnostic_fd, _ = install(
    os.environ["AI_INFRA_ATTEMPT_DIAGNOSTICS"], slow_seconds=480
)
clock_offset = time.time_ns() - time.perf_counter_ns()

if os.environ.get("BENCH_METRICS_CHECK") == "1":
    print(
        json.dumps(
            {
                "status": "passed",
                "summary_filter": callable(SummaryRetention.filter),
                "native_executor": callable(RustHttpExecutorRuntime.execute_attempt),
            }
        )
    )
    raise SystemExit(0)

stream = (root / "request-metrics.jsonl").open("x", buffering=1024 * 1024)
original_filter = SummaryRetention.filter
last_flush = time.monotonic()


def observed_filter(self, records):
    global last_flush
    records = tuple(records)
    for record in records:
        if isinstance(record, TurnResultRecord):
            request = record.measurements.get("request", {})
            end = request.get("response_completed_ns")
            row = {
                "kind": "turn",
                "id": record.id,
                "session_id": record.session_id,
                "ordinal": record.ordinal,
                "status": record.status,
                "error": record.error,
                "http_status": request.get("http_status"),
                "usage": dict(record.usage),
                "request_ns": request.get("started_ns"),
                "response_completed_ns": end,
                "finished_wall_ns": end + clock_offset
                if end is not None
                else time.time_ns(),
                "attempts": request.get("attempts"),
                "metrics": {},
            }
            for name in ("ttft", "tpot", "decode_throughput", "e2e"):
                metric = record.metrics.get(name)
                if metric is not None:
                    row["metrics"][name] = {
                        "value": metric.statistics.get("value"),
                        "unit": metric.unit,
                        "availability": metric.availability,
                        "unavailable_reason": metric.unavailable_reason,
                    }
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        elif isinstance(record, AttemptResultRecord) and record.status == "failed":
            stream.write(
                json.dumps(
                    {
                        "kind": "failed_attempt",
                        "turn_id": record.turn_id,
                        "status": record.status,
                        "error": record.error,
                        "error_origin": record.error_origin,
                        "http_status": record.measurements.get("http_status"),
                        "wall_ns": time.time_ns(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    if time.monotonic() - last_flush >= 5:
        stream.flush()
        last_flush = time.monotonic()
    return original_filter(self, records)


SummaryRetention.filter = observed_filter
original_execute = RustHttpExecutorRuntime.execute_attempt
first_submitted = False


async def observed_execute(self, prepared, signal):
    global first_submitted
    if not first_submitted:
        first_submitted = True
        (root / "measurement-start.json").write_text(
            json.dumps(
                {
                    "first_submit_wall_ns": time.time_ns(),
                    "first_submit_monotonic_ns": time.perf_counter_ns(),
                    "clock_offset_ns": clock_offset,
                    "definition": (
                        "first native execute_attempt submission; "
                        "excludes client preparation"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
    return await original_execute(self, prepared, signal)


RustHttpExecutorRuntime.execute_attempt = observed_execute
try:
    from perf_benchmark.app.main import console_main

    raise SystemExit(console_main())
finally:
    stream.flush()
    stream.close()
