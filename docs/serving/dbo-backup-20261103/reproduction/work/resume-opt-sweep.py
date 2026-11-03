"""Finish the in-flight sample, then cover all concurrency levels per repeat."""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--frozen-parent", type=int, required=True)
parser.add_argument("--active-benchmark", type=int, required=True)
args = parser.parse_args()
root = Path("/work/performance/optimized-dep16-p5d5-v7-001")
waiter = threading.Event()
deadline = time.monotonic() + 7200
print("Current C1-r1 benchmark continues unchanged; waiting for its completion", flush=True)
while True:
    try:
        os.kill(args.active_benchmark, 0)
    except ProcessLookupError:
        break
    # The frozen parent cannot reap its finished child until it exits.
    stat = Path(f"/proc/{args.active_benchmark}/stat").read_text()
    if stat.split(") ", 1)[1].startswith("Z "):
        break
    if time.monotonic() > deadline:
        raise TimeoutError("In-flight benchmark did not finish")
    waiter.wait(2)
assert (root / "ep16-c1-r1/status.json").exists()
assert (root / "ep16-c1-r1/result.json").exists()
os.kill(args.frozen_parent, signal.SIGKILL)
for repeat in range(1, 4):
    for concurrency in (1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        sample = root / f"ep16-c{concurrency}-r{repeat}"
        if sample.exists():
            status = json.loads((sample / "status.json").read_text())
            assert status["exit_code"] == 0, status
        else:
            subprocess.run(
                [sys.executable, "/work/run-opt-bench.py", "--ep", "16",
                 "--concurrency", str(concurrency), f"--run-tag=-r{repeat}"],
                check=True,
            )
        result = json.loads((sample / "result.json").read_text())
        expected = max(16, 4 * concurrency)
        acceptance = {key: result[key] for key in ("completed", "failed", "output_throughput")}
        acceptance.update(expected=expected, zero_error_pass=(
            result["completed"] == expected and result["failed"] == 0))
        (sample / "acceptance.json").write_text(json.dumps(acceptance, indent=2))
        print(sample.name, acceptance, flush=True)
print("All three repetitions of C1..4096 complete", flush=True)
