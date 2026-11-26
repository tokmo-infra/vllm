#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check the exclusive GLM 85/86 service, smoke-test and clear its caches."""
import argparse
import json
from pathlib import Path
import time
import uuid
from urllib.request import Request, build_opener, ProxyHandler

ROOT = Path('/data/services/glm53flash-1p1d-8586')
P = ('http://11.1.0.85:28483', 'http://11.1.0.85:28493')
D = 'http://11.1.0.86:28483'
M = 'http://11.1.0.86:61743/metrics'
ROUTER = 'http://11.1.0.85:28300'
STORES = ('http://11.1.0.85:61745/health', 'http://11.1.0.86:61745/health')
ZERO = ('master_key_count','master_allocated_bytes','master_allocated_file_size_bytes')
HTTP = build_opener(ProxyHandler({}))
KEY = (ROOT / 'engine.key').read_text().strip()


def get(url, *, auth=False, reset_key=None, method='GET', body=None, timeout=8):
    headers = {'Content-Type': 'application/json'}
    if auth:
        headers['Authorization'] = f'Bearer {KEY}'
    if reset_key is not None:
        headers['x-c32-reset'] = reset_key
    with HTTP.open(Request(url, headers=headers, method=method, data=body), timeout=timeout) as r:
        return r.read().decode()


def metric(url, names, auth=False):
    text = get(url, auth=auth)
    out = {name: [] for name in names}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        metric_name = line.split(None, 1)[0].partition('{')[0]
        if metric_name in out:
            out[metric_name].append(float(line.rsplit(maxsplit=1)[-1]))
    if any(not el for el in out.values()):
        raise RuntimeError(f'missing counters at {url}: {[k for k,v in out.items() if not v]}')
    return {k:sum(v) for k,v in out.items()}


def ready():
    for u in (*P, D):
        get(u+'/health', auth=True)
    for u in STORES:
        get(u)
    registered = json.loads(get(ROUTER+'/workers'))['workers']
    if len(registered)!=16 or not all(w['is_healthy'] for w in registered):
        raise RuntimeError('Router should have 16 healthy registered workers')
    return len(registered)


def smoke():
    workers = ready()
    for i in range(6):
        suffix=uuid.uuid4().hex
        prompt=f'Request {suffix}. ' + 'The quick brown fox jumps over the lazy dog. ' * 130
        prompt+=' Answer the final question in one short sentence: what animal is the subject?'
        req=dict(model='glm-5.3-flash', messages=[dict(role='user',content=prompt)], max_tokens=64, temperature=0, reasoning_effort='low')
        obj=json.loads(get(ROUTER+'/v1/chat/completions',method='POST',body=json.dumps(req).encode(),timeout=120))
        usage=obj['usage']
        if not obj['choices'] or usage.get('completion_tokens',0) < 1:
            raise RuntimeError('empty canary response')
        print(json.dumps(dict(workers=workers,case=i,usage=usage,finish_reason=obj['choices'][0]['finish_reason'])),flush=True)


def clear():
    workers = ready()
    reset_key = (ROOT / 'cache-reset.key').read_text().strip()
    gauges=('vllm:num_requests_running','vllm:num_requests_waiting')
    for u in (*P,D):
        if json.loads(get(u+'/load',auth=True))['server_load'] or any(metric(u+'/metrics',gauges,auth=True).values()):
            raise RuntimeError(f'cache reset requires a drained engine: {u}')
    before=metric(M,ZERO)
    for u in (*P,D):
        for _ in range(8):
            r=json.loads(get(u+'/_c32/reset',reset_key=reset_key,method='POST',timeout=120))
            if r.get('success') is True:
                break
            time.sleep(2)
        else:
            raise RuntimeError(f'failed model cache reset: {u}')
    for _ in range(40):
        after=metric(M,ZERO)
        if not any(after.values()):
            print(json.dumps(dict(workers=workers, cache_before=before, cache_after=after)),flush=True)
            return
        time.sleep(2)
    raise RuntimeError(f'Mooncake key/L2/L3 counters did not reach zero: {after}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('smoke','clear','ready'))
    phase=parser.parse_args().phase
    print(ready()) if phase=='ready' else {'smoke':smoke,'clear':clear}[phase]()
