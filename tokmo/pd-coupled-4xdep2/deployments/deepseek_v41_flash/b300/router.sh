#!/usr/bin/env bash
# 普通耦合模式：四个 HTTP 后端，每个展开为两个 DP rank，共八个 worker。
set -euo pipefail

NAME=vllm-dsv41-b300-pd-coupled-4xdep2-router
IMAGE=${ROUTER_IMAGE:-local/vllm-router:0.1.16-xinhao}
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "同名容器已存在：$NAME；请先确认实例状态。" >&2
  exit 1
fi
for port in 30020 30022 30024 30026; do
  curl --noproxy '*' --fail --silent --show-error "http://127.0.0.1:$port/health"
done

docker run -d \
  --name "$NAME" \
  --pull=never --network host \
  --memory 64g \
  --log-opt max-size=128m --log-opt max-file=5 \
  -e NO_PROXY='*' -e no_proxy='*' \
  -e NVIDIA_VISIBLE_DEVICES=void \
  "$IMAGE" \
  --host 0.0.0.0 --port 18086 \
  --prometheus-host 0.0.0.0 --prometheus-port 18087 \
  --worker-urls \
    http://11.5.0.55:30020 \
    http://11.5.0.55:30022 \
    http://11.5.0.55:30024 \
    http://11.5.0.55:30026 \
  --intra-node-data-parallel-size 2 \
  --api-key EMPTY \
  --policy cache_aware \
  --request-timeout-secs 600 \
  --worker-startup-timeout-secs 7200
