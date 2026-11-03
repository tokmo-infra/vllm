#!/usr/bin/env bash
# Image: xinhao-vllm-router:e9b0421-tokenids-20261103
# Alias of vllm-router-glm-release-xinhao:e9b0421-tokenids, matching ../b300-8p2-2d8-main/router.sh.
# 89：指定发布分支 Router；8 个 P 入口＋1 个全局 EP16 D 入口。
# 2026-11-03: 64GiB OOM; raise to128GiB after NUMA0/cache headroom readback.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" 89
# All endpoints must be healthy before DP discovery; an early Router can retain
# one fallback worker per endpoint and silently miss the second P rank.
python3 - <<'PYREADY'
import concurrent.futures,urllib.request
urls=[f"http://11.{rail}.0.{n}:{30010+2*i}/health" for n in [89,91] for i,rail in enumerate([11,9,8,6])]+["http://11.11.0.92:30011/health"]
def check(url):
 with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url,timeout=10) as r:assert r.status==200,url
with concurrent.futures.ThreadPoolExecutor(9) as p:list(p.map(check,urls))
PYREADY

docker run -d \
  --name xinhao-roce4c711-router \
  --pull=never --network host --memory 128g --memory-swap 256g --cpuset-mems 0 \
  --ulimit nofile=1048576:1048576 \
  --log-opt max-size=128m --log-opt max-file=5 \
  -e NO_PROXY='*' -e no_proxy='*' -e NVIDIA_VISIBLE_DEVICES=void \
  -e LD_PRELOAD=/usr/local/lib/router-listener.so \
  -e AI_INFRA_LISTEN_BACKLOG=4096 \
  -e VLLM_ROUTER_PD_REUSE_PROMPT_IDS=1 \
  --entrypoint /usr/local/bin/vllm-router \
  xinhao-vllm-router:e9b0421-tokenids-20261103 \
  --host 0.0.0.0 --port 28080 \
  --prometheus-host 0.0.0.0 --prometheus-port 28081 \
  --vllm-pd-disaggregation \
  --prefill http://11.11.0.89:30010 \
  --prefill http://11.9.0.89:30012 \
  --prefill http://11.8.0.89:30014 \
  --prefill http://11.6.0.89:30016 \
  --prefill http://11.11.0.91:30010 \
  --prefill http://11.9.0.91:30012 \
  --prefill http://11.8.0.91:30014 \
  --prefill http://11.6.0.91:30016 \
  --decode "http://$D0_IP:30011" \
  --prefill-policy cache_aware --decode-policy round_robin \
  --max-tree-size 2147483648 \
  --max-concurrent-requests 12000 --disable-retries \
  --api-key EMPTY --log-level warn \
  --request-timeout-secs 600 --worker-startup-timeout-secs 7200

# Verify actual discovery, rather than treating /health as proof of all ranks.
python3 - <<'PYWORKERS'
import collections,json,time,urllib.request
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
last=None
for i in range(30):
 try:
  with op.open('http://127.0.0.1:28080/workers',timeout=2) as r:d=json.load(r)['workers']
  last=dict(collections.Counter(w['worker_type'] for w in d))
  if last=={'prefill':16,'decode':16} and all(w['is_healthy'] for w in d):
   print(json.dumps({'registered':last,'healthy':32}),flush=True);break
 except Exception as e:last=str(e)
 time.sleep(1)
else:raise RuntimeError(f'Expected16P+16D healthy workers; got {last}')
PYWORKERS
