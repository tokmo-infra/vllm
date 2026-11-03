"""Pause only the benchmark scheduler between samples for model profiling."""
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
parser.add_argument("--scheduler-pid", required=True, type=int)
args = parser.parse_args()
pid = args.scheduler_pid
root = Path("/work/performance/optimized-dep16-p5d5-v7-001")
target = root / "ep16-c4096-r1"
record = Path("/work/diagnostics/v7/profile-scheduler-handoff.json")
waiter = threading.Event()
deadline = time.monotonic() + 14400


def verify_parent():
    command = Path(f"/proc/{pid}/cmdline").read_bytes()
    assert b"/work/resume-opt-sweep.py\x00" in command, command


print("Waiting for first C4096 sample; no profiling during timed benchmarks", flush=True)
while not target.exists():
    verify_parent()
    if time.monotonic() > deadline:
        raise TimeoutError("First curve did not reach C4096")
    waiter.wait(3)
assert not (target / "status.json").exists(), "First curve already passed; leave scheduler alone"
verify_parent()
children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
benchmarks = []
for child in children:
    command = Path(f"/proc/{child}/cmdline").read_bytes()
    if b"/work/run-opt-bench.py\x00" in command and b"\x004096\x00" in command:
        benchmarks.append(int(child))
assert len(benchmarks) == 1, benchmarks
child = benchmarks[0]
os.kill(pid, signal.SIGSTOP)
state = dict(scheduler_pid=pid, in_flight_benchmark_pid=child,
             paused=time.time(), reason="Finish C4096-r1 unchanged, profile before starting r2")
record.parent.mkdir(parents=True, exist_ok=True)
record.write_text(json.dumps(state, indent=2))
print("Scheduler paused; in-flight C4096 benchmark continues unchanged", flush=True)
try:
    while True:
        stat = Path(f"/proc/{child}/stat").read_text()
        if stat.split(") ", 1)[1].startswith("Z "):
            break
        if time.monotonic() > deadline:
            raise TimeoutError("C4096-r1 benchmark did not complete")
        waiter.wait(3)
    status = json.loads((target / "status.json").read_text())
    assert status["exit_code"] == 0, status
    assert (target / "result.json").exists()
    state["timed_sample_finished"] = time.time()
    record.write_text(json.dumps(state, indent=2))
    for concurrency, delay in ((32, 16), (512, 64)):
        print(f"Capturing separate full-model trace at C{concurrency}", flush=True)
        subprocess.run(
            [sys.executable, "/work/profile-e2e.py", "--concurrency", str(concurrency),
             "--delay", str(delay), "--prefix", f"optimized-v7-c{concurrency}",
             "--output", f"/work/diagnostics/v7/e2e-profile-c{concurrency}"],
            check=True,
        )
    state["profiles_passed"] = True
finally:
    verify_parent()
    os.kill(pid, signal.SIGCONT)
    state["resumed"] = time.time()
    record.write_text(json.dumps(state, indent=2))
    print("Benchmark scheduler resumed for repetitions2and3", flush=True)
