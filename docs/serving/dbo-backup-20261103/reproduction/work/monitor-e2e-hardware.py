"""Collect low-frequency, read-only telemetry while a serving sweep runs."""
import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--pid", required=True, type=int)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
stop = threading.Event()


def sample(node):
    command = ["docker", "exec", f"xinhao-dbo-dev{node}",
               "/work/.venv/bin/python", "/work/sample-hardware.py"]
    started = time.time()
    try:
        result = subprocess.run(
            ["ssh", "-F", "work/ssh-config", f"b300-{node}", shlex.join(command)],
            text=True, capture_output=True, timeout=12,
        )
        data = json.loads(result.stdout)
        return dict(node=node, client_time=started, received=time.time(), **data)
    except Exception as exc:
        return dict(node=node, client_time=started, error=str(exc))


args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as out, concurrent.futures.ThreadPoolExecutor(2) as pool:
    print(f"Read-only hardware sampling started: {args.output}", flush=True)
    while True:
        try:
            os.kill(args.pid, 0)
        except ProcessLookupError:
            break
        for row in pool.map(sample, (89, 91)):
            out.write(json.dumps(row) + "\n")
        out.flush()
        stop.wait(5)
print("Serving orchestrator exited; hardware sampling stopped", flush=True)
