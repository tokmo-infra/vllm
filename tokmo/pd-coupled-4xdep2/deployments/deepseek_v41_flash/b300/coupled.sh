#!/usr/bin/env bash
# bash coupled.sh 0|1|2|3；四个独立 TP1/DP2/EP2 耦合实例，各使用一对 GPU。
# L2 stays 200GiB/rank equivalent: eight external owners hold the 1600GiB pool.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKDIR=${WORKDIR:-$SCRIPT_DIR}
IMAGE=${IMAGE:-local/vllm-openai:0.32.0-dsv41-native-token-cache-v1}
MODEL_DIR=${MODEL_DIR:-/data2/models/DeepSeek-V4.1-Flash-1017}
INSTANCE=${1:?用法: bash coupled.sh 0|1|2|3}
[[ "$INSTANCE" =~ ^[0-3]$ ]] || { echo "实例编号只能为0到3" >&2; exit 1; }
NAME="vllm-dsv41-b300-pd-coupled-4xdep2-$INSTANCE"
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "同名容器已存在：$NAME；请先确认实例状态。" >&2
  exit 1
fi
GPU_FIRST=$((INSTANCE * 2))
GPU_IDS="$GPU_FIRST,$((GPU_FIRST + 1))"
HTTP_PORT=$((30020 + INSTANCE * 2))
DP_PORT=$((16360 + INSTANCE * 100))
CACHE="$WORKDIR/cache/coupled-$INSTANCE"
STATE="$WORKDIR/native-tokenizer-state/coupled-$INSTANCE"
mkdir -p "$CACHE/"{cuda,tilelang} "$STATE"
printf '%s\n' "$(date -u +%Y%m%dT%H%M%SZ)" > "$STATE/epoch"

docker run -d \
  --name "$NAME" \
  --pull=never --gpus all \
  --device /dev/infiniband \
  --network host --ipc host \
  --cap-add IPC_LOCK --cap-add SYS_NICE \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --log-opt max-size=128m --log-opt max-file=5 \
  -e NO_PROXY='*' -e no_proxy='*' \
  -v "$MODEL_DIR:/model:ro" \
  -v "$SCRIPT_DIR/mooncake.json:/etc/vllm/mooncake.json:ro" \
  -v "$STATE:/tokenizer-state:ro" \
  -v "$CACHE:/root/.cache" \
  -v "$CACHE/cuda:/root/.nv/ComputeCache" \
  -v "$CACHE/tilelang:/root/.tilelang" \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e VLLM_HOST_IP=11.5.0.55 \
  -e MOONCAKE_CONFIG_PATH=/etc/vllm/mooncake.json \
  -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  -e VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1 \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e VLLM_DSV41_TOKENIZER_CACHE_GIB=4 \
  -e VLLM_DSV41_TOKENIZER_CACHE_EPOCH_FILE=/tokenizer-state/epoch \
  "$IMAGE" /model \
  --served-model-name deepseek-v4.1-flash-1017 \
  --tokenizer-mode deepseek_v41 \
  --api-server-count 8 --renderer-num-workers 1 \
  --device-ids "$GPU_IDS" \
  --tensor-parallel-size 1 --data-parallel-size 2 \
  --data-parallel-backend mp --data-parallel-rpc-port "$DP_PORT" \
  --enable-expert-parallel --enable-ep-weight-filter \
  --numa-bind --numa-bind-nodes "$INSTANCE" "$INSTANCE" \
  --enable-eplb \
  --max-model-len -1 \
  --max-num-seqs 512 --max-num-batched-tokens 16384 \
  --kv-cache-dtype fp8 \
  --attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --kernel-config '{"moe_backend":"deep_gemm_mega_moe","sparse_indexer_topk_backend":"deep_select"}' \
  --engram-config '{"cpu_offload":true,"embedding_across_dp":true,"dp_shared_memory":true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":false}' \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"cache_prefix":"dsv41-flash-1017-b300-pd-coupled-4xdep2-l2-200g-l3-2t-v1","lookup_async":true,"load_async":true}}' \
  --enable-prompt-tokens-details \
  --reasoning-parser deepseek_v41 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 \
  --host 0.0.0.0 --port "$HTTP_PORT"
