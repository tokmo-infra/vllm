import json
import subprocess
import sys
import threading
import time
from pathlib import Path

summary = Path('/work/validation/baseline-dep8-p5d5-001/full.summary.json')
deadline = time.monotonic() + 4 * 3600
waiter = threading.Event()
while not summary.exists():
    if time.monotonic() > deadline:
        raise TimeoutError('Baseline correctness evaluation did not finish')
    waiter.wait(10)
result = json.loads(summary.read_text())
assert result['total'] == 1319 and result['errors'] == 0, result
for c in (1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096):
    for repeat in range(1, 4):
        subprocess.run([sys.executable, '/work/run-pd-bench.py', '--ep', '8', '--concurrency', str(c), f'--run-tag=-r{repeat}'], check=True)
        root = Path('/work/performance/baseline-dep8-p5d5-001') / f'ep8-c{c}-r{repeat}'
        measured = json.loads((root / 'result.json').read_text())
        assert measured['completed'] == max(16, c * 4), measured
