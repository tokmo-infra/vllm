# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Low-frequency external counters; does not enable the benchmark live timeline."""

import datetime
import gzip
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("WORKDIR", Path(__file__).resolve().parent))
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PORTS = (30020, 30022, 30024, 30026, 9017, 18087, *range(9320, 9328))
MODEL_PREFIXES = (
    "vllm:prompt_tokens",
    "vllm:generation_tokens",
    "vllm:prefix_cache",
    "vllm:external_prefix",
    "vllm:num_requests",
    "vllm:time_to_first",
    "vllm:request_time_per",
    "vllm:e2e_request",
    "vllm:kv_connector",
)


def scrape(port):
    started = time.time()
    try:
        with OPENER.open(f"http://127.0.0.1:{port}/metrics", timeout=4) as r:
            text = r.read().decode()
        lines = []
        for line in text.splitlines():
            if line.startswith("#") or "_bucket{" in line:
                continue
            if port in (30020, 30022, 30024, 30026) and not line.startswith(
                MODEL_PREFIXES
            ):
                continue
            lines.append(line)
        return dict(
            port=port,
            time=time.time(),
            duration_s=time.time() - started,
            text="\n".join(lines),
        )
    except Exception as e:
        return dict(port=port, time=time.time(), error=str(e))


def main():
    (ROOT / "validation").mkdir(parents=True, exist_ok=True)
    path = ROOT / "validation/production-external-metrics.jsonl.gz"
    with (
        gzip.open(path, "at", compresslevel=3) as f,
        ThreadPoolExecutor(max_workers=14) as pool,
    ):
        while not (ROOT / "validation/production-monitor.stop").exists():
            started = time.monotonic()
            samples = list(pool.map(scrape, PORTS))
            disk = [
                line
                for line in Path("/proc/diskstats").read_text().splitlines()
                if line.split()[2] in ("nvme0n1", "nvme1n1")
            ]
            f.write(
                json.dumps(
                    dict(
                        utc=datetime.datetime.now(datetime.UTC).isoformat(),
                        samples=samples,
                        diskstats=disk,
                    )
                )
                + "\n"
            )
            f.flush()
            time.sleep(max(0, 10 - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
