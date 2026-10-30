#!/usr/bin/env bash
# Shared metadata: eight external owners provide 200GiB L2 + 256GiB L3 each.
set -euo pipefail

IMAGE=${MASTER_IMAGE:-vllm/vllm-openai:v0.32.0}
NAME=vllm-dsv41-b300-pd-coupled-4xdep2-mooncake-master
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "同名容器已存在：$NAME；请先确认实例状态。" >&2
  exit 1
fi
docker run -d \
  --name "$NAME" \
  --pull=never --network host --restart unless-stopped \
  --memory 64g \
  --log-opt max-size=128m --log-opt max-file=5 \
  -e NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint mooncake_master \
  "$IMAGE" \
  --rpc_address=127.0.0.1 --rpc_port=50066 --metrics_port=9017 \
  --enable_offload=true --offload_on_evict=true --offload_force_evict=false \
  --promotion_on_hit=true --promotion_admission_threshold=2
