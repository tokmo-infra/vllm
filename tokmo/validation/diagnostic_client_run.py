# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Observe failed/near-deadline attempts without changing the native executor.

No request body or generated text is retained. Native status, clocks, timeout,
cancellation, pooling and retry behavior remain in the original implementation.
"""

import datetime
import json
import logging
import os
from pathlib import Path
from time import perf_counter_ns


def attempt_record(runtime, prepared, result, submitted_ns, returned_ns):
    value = prepared.input
    request = result.request
    started, ended = request.started_ns, request.response_completed_ns
    first, last = request.first_output_ns, request.last_output_ns
    samples = result.trace.samples if result.trace is not None else []
    gaps = [(b[1] - a[1]) / 1e9 for a, b in zip(samples, samples[1:])]
    return {
        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "session_id": value.session.id,
        "session_ordinal": value.session.ordinal,
        "turn_id": value.turn.id,
        "turn_ordinal": value.turn.ordinal,
        "attempt": result.attempt,
        "status": result.status,
        "error": result.error,
        "error_origin": result.error_origin,
        "retryable": result.retryable,
        "timeout_seconds": runtime.settings.get("timeout_seconds"),
        "requested_output_tokens": value.turn.requested_output_tokens,
        "executor_max_tokens": runtime.settings.get("max_tokens"),
        "http_status": request.http_status,
        "finish_reason": request.finish_reason,
        "output_chars": request.output_chars,
        "output_event_count": len(samples),
        "started_ns": started,
        "first_output_ns": first,
        "last_output_ns": last,
        "response_completed_ns": ended,
        "submitted_ns": submitted_ns,
        "returned_ns": returned_ns,
        "total_wall_seconds": (returned_ns - submitted_ns) / 1e9,
        "before_send_seconds": (started - submitted_ns) / 1e9
        if started is not None
        else None,
        "e2e_seconds": (ended - started) / 1e9
        if started is not None and ended is not None
        else None,
        "first_output_seconds": (first - started) / 1e9
        if first is not None and started is not None
        else None,
        "last_output_age_seconds": (ended - last) / 1e9
        if ended is not None and last is not None
        else None,
        "max_inter_output_gap_seconds": max(gaps, default=None),
        "usage": result.usage,
        "first_events": samples[:3],
        "last_events": samples[-3:],
        "scope": "native attempt facts only; no body/text and no behavior override",
    }


def install(path, *, slow_seconds=480, focus_pairs=()):
    from perf_benchmark.runtime.http.rust_executor import RustHttpExecutorRuntime

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Never mix two measured runs or silently append to an existing diagnostic.
    descriptor = os.open(
        destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600
    )
    original = RustHttpExecutorRuntime.execute_attempt
    focus_pairs = frozenset(focus_pairs)

    async def observed(runtime, prepared, signal):
        submitted = perf_counter_ns()
        result = await original(runtime, prepared, signal)
        returned = perf_counter_ns()
        slow = (returned - submitted) / 1e9 >= slow_seconds
        value = prepared.input
        focused = (
            bool(focus_pairs)
            and (value.session.ordinal, value.turn.ordinal) in focus_pairs
        )
        if result.status == "failed" or (
            result.status == "succeeded" and (slow or focused)
        ):
            try:
                record = attempt_record(runtime, prepared, result, submitted, returned)
                record["focus_match"] = focused
                encoded = (
                    json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
                ).encode()
                written = os.write(descriptor, encoded)
                if written != len(encoded):
                    raise OSError("Partial diagnostic write")
            except Exception:
                logging.exception("Failed to persist native attempt diagnostics")
        return result

    RustHttpExecutorRuntime.execute_attempt = observed
    return descriptor, original


if __name__ == "__main__":
    descriptor, _ = install(
        os.environ["AI_INFRA_ATTEMPT_DIAGNOSTICS"],
        slow_seconds=float(os.getenv("AI_INFRA_SLOW_ATTEMPT_SECONDS", "480")),
        focus_pairs=[
            tuple(map(int, pair.split(":")))
            for pair in os.getenv("AI_INFRA_ATTEMPT_FOCUS", "").split(",")
            if pair
        ],
    )
    try:
        from perf_benchmark.app.main import console_main

        raise SystemExit(console_main())
    finally:
        os.close(descriptor)
