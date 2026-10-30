# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Read-only readiness: eight healthy owners, 1600GiB L2 and 2TiB L3."""

import argparse
import datetime
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("WORKDIR", Path(__file__).resolve().parent))
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch(port, path):
    with OPENER.open(f"http://127.0.0.1:{port}{path}", timeout=3) as response:
        return response.read().decode()


def gauge(metrics, name):
    prefix = name + " "
    values = [
        line[len(prefix) :] for line in metrics.splitlines() if line.startswith(prefix)
    ]
    assert len(values) == 1, (name, values)
    return int(float(values[0]))


def inspect():
    for port in range(9320, 9328):
        fetch(port, "/health")
    metrics = fetch(9017, "/metrics")
    l2 = gauge(metrics, "master_total_capacity_bytes")
    l3 = gauge(metrics, "master_total_file_capacity_bytes")
    assert l2 == 8 * 200 * 2**30, ("L2 capacity", l2)
    assert l3 == 2 * 2**40, ("L3 capacity", l3)
    segments = {}
    prefix = 'segment_total_capacity_bytes{segment="'
    for line in metrics.splitlines():
        if line.startswith(prefix):
            segment, separator, value = line[len(prefix) :].partition('"} ')
            assert separator, line
            segments[segment] = value
    expected = {f"11.5.0.55:{50140 + i}" for i in range(8)}
    assert set(segments) == expected, ("owner segments", segments)
    assert all(int(float(value)) == 200 * 2**30 for value in segments.values()), (
        segments
    )
    return metrics, dict(
        status="passed",
        owners=8,
        l2_bytes=l2,
        l3_quota_bytes=l3,
        l2_allocated_bytes=gauge(metrics, "master_allocated_bytes"),
        l3_registered_bytes=gauge(metrics, "master_allocated_file_size_bytes"),
        scope=(
            "health and registered quotas only; no generation, cache clear or benchmark"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-seconds", type=float, default=300)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            metrics, result = inspect()
            break
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Mooncake readiness failed: {exc}") from exc
            time.sleep(2)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    evidence = ROOT / "validation" / ("production-health-" + stamp)
    evidence.mkdir(parents=True, exist_ok=False)
    (evidence / "master.prom").write_text(metrics)
    (evidence / "health.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "evidence": str(evidence)}))


if __name__ == "__main__":
    main()
