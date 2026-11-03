import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
root = Path('/work/validation/preopt-dep16-p5d5-001')
root.mkdir(exist_ok=False)
deadline = time.monotonic() + 7200
waiter = threading.Event()
while True:
    try:
        for url in ('http://11.11.0.90:30010/health', 'http://11.11.0.89:30011/health', 'http://11.11.0.89:28080/health'):
            with opener.open(url, timeout=5) as response:
                assert response.status == 200
        with opener.open('http://11.11.0.89:28080/workers', timeout=5) as response:
            workers = json.load(response)
        counts = {kind: sum(w['worker_type'] == kind and w['is_healthy'] for w in workers['workers']) for kind in ('prefill', 'decode')}
        assert counts == {'prefill': 2, 'decode': 16}, counts
        (root / 'workers.json').write_text(json.dumps(workers, indent=2))
        break
    except Exception as error:
        if time.monotonic() >= deadline:
            raise TimeoutError('P5/D5 services did not become ready') from error
        waiter.wait(10)
print('P5/D5 ready; checking four GSM8K requests', flush=True)
command = [sys.executable, '/work/validation/gsm8k_eval.py', 'run', '--url', 'http://11.11.0.89:28080', '--data', '/work/validation/gsm8k', '--max-tokens', '4096']
subprocess.run(command + ['--limit', '4', '--concurrency', '4', '--output', str(root / 'smoke.jsonl')], check=True)
smoke = json.loads((root / 'smoke.summary.json').read_text())
assert smoke['errors'] == 0, smoke
print('Starting complete 1319-question evaluation', flush=True)
subprocess.run(command + ['--concurrency', '32', '--output', str(root / 'full.jsonl')], check=True)
summary = json.loads((root / 'full.summary.json').read_text())
assert summary['errors'] == 0 and summary['total'] == 1319, summary
