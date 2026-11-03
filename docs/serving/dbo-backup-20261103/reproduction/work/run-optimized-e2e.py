"""Run the user's prioritized overlap service, then the DEP8-matched sweep."""
import argparse
import json
import shlex
import subprocess
import threading
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--version', required=True)
args = parser.parse_args()
assert args.version.startswith('v') and args.version[1:].isdigit()

def ssh(node, args, capture=False):
    return subprocess.run(
        ['ssh', '-F', 'work/ssh-config', f'b300-{node}', shlex.join(args)],
        check=True, text=True, stdout=subprocess.PIPE if capture else None,
    ).stdout

for node in (91, 89):
    pids = ssh(node, ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture=True).strip()
    assert not pids, f'Previous service still owns GPUs on {node}: {pids}'
    command = json.loads(Path(f'work/launch-opt-d16-{node}-{args.version}.argv.json').read_text())
    assert 'exec /tmp/epcmp/.venv/bin/python -m vllm.entrypoints.cli.main serve' in command[-1]
    serving = shlex.split(command[-1])
    assert serving[serving.index('--served-model-name') + 1] == 'deepseek-v4.1-flash-1017', 'Version label must not change the served model name'
    ssh(node, command)
print('Current overlap implementation launched on 89+91; P90 unchanged', flush=True)
waiter = threading.Event()
deadline = time.monotonic() + 7200
health = 'import urllib.request; o=urllib.request.build_opener(urllib.request.ProxyHandler({})); r=o.open("http://11.11.0.89:30011/health",timeout=5); assert r.status==200'
while True:
    result = subprocess.run(
        ['ssh','-F','work/ssh-config','b300-89',shlex.join(['docker','exec','xinhao-dbo-dev89','/work/.venv/bin/python','-c',health])],
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        break
    for node in (89, 91):
        name = f'xinhao-dbo-opt-d16-{node}-{args.version}'
        state = json.loads(ssh(node, ['docker', 'inspect', '--format', '{{json .State}}', name], capture=True))
        if not state['Running']:
            logs = ssh(node, ['docker', 'logs', '--tail', '300', name], capture=True)
            Path(f'work/optimized-{args.version}-failed-{node}.log').write_text(logs)
            raise RuntimeError(f'{name} stopped before health: {state}')
    if time.monotonic() > deadline:
        raise TimeoutError('Optimized DEP16 did not become healthy')
    waiter.wait(15)
router = json.loads(Path('work/launch-router.argv.json').read_text())
router[router.index('--name')+1] = f'xinhao-dbo-router-opt-{args.version}'
ssh(89, router)
ssh(89, ['docker','exec','xinhao-dbo-dev89','/work/.venv/bin/python','/work/run-opt-smoke.py'])
print('Optimized end-to-end smoke passed; starting C1..4096 performance first', flush=True)
ssh(89, ['docker','exec','xinhao-dbo-dev89','/work/.venv/bin/python','/work/run-opt-sweep.py'])
print('Optimized performance sweep complete; preserve results for DEP8 comparison', flush=True)
