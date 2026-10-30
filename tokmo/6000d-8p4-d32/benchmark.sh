#!/usr/bin/env bash
# 在B300上执行；只运行压测客户端，模型与Router须已独立部署。
set -euo pipefail
MODE=${1:-run}
[[ "$MODE" == run || "$MODE" == --check ]] || { printf 'Usage: bash benchmark.sh [--check]\n' >&2; exit 2; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TOOLBOX=${TOOLBOX:-perf-benchmark-toolbox-glm53flash-8p4-d32}
CONFIG="$SCRIPT_DIR/c1024-a1536-swe80k.yaml"
CONFIG_IN_CONTAINER=/tmp/glm53flash-8p4-d32-c1024-a1536-swe80k.yaml
CLIENT_ROOT=/work/client-source
docker container inspect "$TOOLBOX" >/dev/null
docker cp "$CONFIG" "$TOOLBOX:$CONFIG_IN_CONTAINER" >/dev/null
docker exec -i -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH="$CLIENT_ROOT" \
  "$TOOLBOX" /work/client-venv/bin/python -B - "$CONFIG_IN_CONTAINER" <<'PY'
import concurrent.futures, json, pathlib, sys, urllib.request, yaml
from perf_benchmark.plugins.metrics.standard import parse_settings
config=yaml.safe_load(pathlib.Path(sys.argv[1]).read_text());p=config['pipeline']
assert p['scheduler']['settings']=={'max_active_sessions':1536,'max_concurrency':1024}
assert p['loader']['settings']['split']=='train[:1536]'
source=pathlib.Path(p['loader']['settings']['data_files']);assert source.is_file(), source
with source.open('rb') as dataset:
    selected=sum(1 for _,line in zip(range(1536),dataset) if line.strip())
assert selected==1536, f'Only {selected} JSONL sessions; expected 1536'
settings=parse_settings(p['metrics']['settings'])
prices=p['metrics']['settings']['pricing']
assert prices['currency']=='CNY' and prices['per_tokens']==1000000
assert [prices[k] for k in ['input_price','cached_input_price','output_price']]==[0.16,0.046,0.56]
output=pathlib.Path(p['saver']['settings']['path']);assert not output.exists(), f'Result already exists: {output}'
base=p['executor']['settings']['base_url'];assert base=='http://11.5.0.81:38080'
urls=[f'http://11.5.0.{n}:{port}/health' for n in range(81,85) for port in (31010,31014)]
urls += ['http://11.5.0.85:31011/health',base+'/health']
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def ready(url):
    with op.open(url,timeout=10) as response:
        body=response.read().decode();assert response.status==200,url
        if url==base+'/health':assert body.strip()=='All servers healthy',(url,body)
    return url
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
    checked=list(pool.map(ready,urls))
print(json.dumps({'ready_urls':checked,'dataset':str(source),'pricing':prices,'result':str(output)},ensure_ascii=False))
PY
[[ "$MODE" == --check ]] && exit 0
LOG_DIR="$SCRIPT_DIR/benchmark-logs/swe80k-c1024-a1536-run-003"
mkdir -p "$LOG_DIR"
# 不自动重试；结果DB存在时前面的校验会拒绝覆盖。
set +e
docker exec -e NO_PROXY='*' -e no_proxy='*' -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH="$CLIENT_ROOT" -e PERF_TARGET_API_KEY=EMPTY "$TOOLBOX" \
  /work/client-venv/bin/python -m perf_benchmark run "$CONFIG_IN_CONTAINER" \
  2>&1 | tee "$LOG_DIR/benchmark.log"
BENCHMARK_EXIT=${PIPESTATUS[0]}
printf '%s\n' "$BENCHMARK_EXIT" > "$LOG_DIR/exitcode"
exit "$BENCHMARK_EXIT"
