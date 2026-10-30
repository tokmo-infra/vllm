#!/usr/bin/env bash
# P/D Router：仅在 6000d-81 启动，连接 8 个 P 实例和 1 个全局 EP32 D。
# 本文件包含本服务的完整参数；直接执行只启动本服务。
set -euo pipefail

MODE=${1:-run}
if (( $# > 1 )); then
  printf 'Usage: bash %s [--dry-run|--print-argv|--print-config]\n' "${0##*/}" >&2
  exit 2
fi
case "$MODE" in
  run|--dry-run|--print-argv|--print-config) ;;
  --help|-h)
    printf 'Usage: bash %s [--dry-run|--print-argv|--print-config]\n' "${0##*/}"
    exit 0 ;;
  *) printf 'Unknown option: %s\n' "$MODE" >&2; exit 2 ;;
esac

NAME=glm53flash-8p4-d32-xinhao-router-81
IMAGE=sha256:21c9cfa19e348f84a99102ab02922d5cd3a1ef6d659881cac09e9d6479c4d07b
WORKDIR="${GLM_WORKDIR:-/data/bench/eng-xinhao/vllm-pd/glm53-flash/6000d-8p4-d32}"
MODEL_DIR="${GLM_MODEL_DIR:-/data/models/GLM-5.3-Flash-NVFP4}"
TOKEN_CONFIG="$WORKDIR/config/router-token-balance.json"

command=(
  docker run -d --init
  --name "$NAME"
  --pull=never
  --network host
  --log-opt max-size=128m
  --log-opt max-file=5
  --label ai-infra.project=glm53flash-8p4-d32-xinhao
  --memory 64g

  # 进程环境
  -e NO_PROXY='*'
  -e no_proxy='*'
  -e HF_HUB_OFFLINE=1
  -e VLLM_NO_USAGE_STATS=1
  -e NVIDIA_VISIBLE_DEVICES=void

  -e OPENAI_API_KEY=EMPTY
  -e VLLM_ROUTER_PD_TOKEN_BALANCE_CONFIG=/run/token-balance.json
  -v "$MODEL_DIR:/model:ro"
  -v "$TOKEN_CONFIG:/run/token-balance.json:ro"

  --entrypoint /usr/local/bin/vllm-router
  "$IMAGE"
  --host 0.0.0.0
  --port 38080
  --prometheus-host 0.0.0.0
  --prometheus-port 38081
  --vllm-pd-disaggregation
  --kv-connector mooncake
  --prefill http://11.5.0.81:31010 27600
  --prefill http://11.5.0.81:31014 27700
  --prefill http://11.5.0.82:31010 27600
  --prefill http://11.5.0.82:31014 27700
  --prefill http://11.5.0.83:31010 27600
  --prefill http://11.5.0.83:31014 27700
  --prefill http://11.5.0.84:31010 27600
  --prefill http://11.5.0.84:31014 27700
  --decode http://11.5.0.85:31011
  # 参考分支：前缀感知 + P 计算量 / D 请求数与实际 KV 容量负载路由。
  --prefill-policy cache_aware
  --decode-policy cache_aware
  --balance-abs-threshold 4
  --balance-rel-threshold 1.5
  --intra-node-data-parallel-size 1
  --disable-retries
  --worker-startup-check-interval 5
  --api-key EMPTY
  --request-timeout-secs 1800
  --worker-startup-timeout-secs 7200
)

case "$MODE" in
  --print-argv)
    printf '%s\0' "${command[@]}"
    exit 0 ;;
  --print-config)
    printf '{}\n'
    exit 0 ;;
  --dry-run)
    printf '%q' "${command[0]}"
    for ((i = 1; i < ${#command[@]}; i++)); do
      printf ' \\\n  %q' "${command[i]}"
    done
    printf '\n'
    exit 0 ;;
esac

# 只检查本服务的容器名，避免覆盖现有实例的挂载配置。
command -v docker >/dev/null
if docker container inspect "$NAME" >/dev/null 2>&1; then
  printf 'Container already exists: %s; inspect it before starting another instance.\n' "$NAME" >&2
  exit 1
fi

# 模型就绪后读取每路实际 KV 容量，避免把 max_model_len 当作 KV 容量。
# 此步骤只生成本 Router 的配置，不启动或修改任何模型服务。
python3 - "$TOKEN_CONFIG" <<'PYCONFIG'
import json, os, pathlib, sys, urllib.request
workers = {}
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
urls = [(f"http://11.5.0.{n}:{p}", 64) for n in range(81, 85) for p in (31010, 31014)]
urls += [("http://11.5.0.85:31011", 128)]
for url, seqs in urls:
    request = urllib.request.Request(url + "/server_info?config_format=json",
                                     headers={"Authorization": "Bearer EMPTY"})
    with opener.open(request, timeout=15) as response:
        info = json.load(response)
    tokens = info["vllm_config"]["cache_config"]["kv_cache_size_tokens"]
    assert isinstance(tokens, int) and tokens > 0, (url, tokens)
    workers[url] = {"kv_tokens": tokens, "max_seqs": seqs}
config = {"tokenizer_path": "/model/tokenizer.json", "kv_fraction": 0.85, "workers": workers}
path = pathlib.Path(sys.argv[1]); path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_name(path.name + ".new")
tmp.write_text(json.dumps(config, indent=2) + "\n"); os.replace(tmp, path)
print(json.dumps(config), file=sys.stderr)
PYCONFIG

exec "${command[@]}"
