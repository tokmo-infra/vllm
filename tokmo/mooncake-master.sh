#!/usr/bin/env bash
# Image: xinhao-vllm-roce-megamoe:4c711ac-20261103 (sha256:19177b08b707...)
# 89：只启动官方 Mooncake master；使用同一份 main 模型镜像。
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" 89
docker run -d \
  --name xinhao-roce4c711-mooncake-master \
  --pull=never --network host --memory 128g --cpuset-mems 0 \
  --log-opt max-size=128m --log-opt max-file=5 \
  -e NVIDIA_VISIBLE_DEVICES=void \
  -e MC_ENABLE_DEST_DEVICE_AFFINITY=1 \
  --entrypoint mooncake_master \
  xinhao-vllm-roce-megamoe:4c711ac-20261103 \
  --rpc_address=0.0.0.0 --rpc_port="$MOONCAKE_MASTER_PORT" --metrics_port="$MOONCAKE_METRICS_PORT" \
  --enable_offload=true --offload_on_evict=true --offload_force_evict=false \
  --promotion_on_hit=true --promotion_admission_threshold=2
