#!/usr/bin/env bash
# Image: xinhao-vllm-roce-megamoe:4c711ac-20261103 (sha256:19177b08b707...)
# 一个命令只启动一个 DEP2 P。用法：bash prefill.sh 89 0（或 91，实例 0..3）。
# P_IPS and P0_IP come from sourced cluster/node configuration.
# shellcheck disable=SC2153
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "${1:?节点号 89|91}"
[[ "$NODE" == 89 || "$NODE" == 91 ]] || { echo 'P 只在 89/91' >&2; exit 2; }
INSTANCE=${2:?实例号 0|1|2|3}
[[ "$INSTANCE" =~ ^[0-3]$ && $# == 2 ]] || exit 2

GPU_FIRST=$((INSTANCE * 2))
GPU_SECOND=$((GPU_FIRST + 1))
NUMA=$((INSTANCE / 2))
P_IP=${P_IPS[$INSTANCE]}
P_HCAS=${P_RDMA_DEVICES[$INSTANCE]}
HTTP_PORT=$((30010 + INSTANCE * 2))
DP_PORT=$((16360 + INSTANCE * 100))
NIXL_PORT=$((17600 + INSTANCE * 100))
CACHE="$WORKDIR/runtime/cache-roce4c711/p$NODE-$INSTANCE"
CONFIG="$WORKDIR/runtime/config/mooncake-$NODE-$INSTANCE.json"
mkdir -p "$CACHE/"{cuda,tilelang} "$(dirname -- "$CONFIG")"
mkdir -p "$WORKDIR/runtime/tokenizer-state"
if [[ ! -f "$WORKDIR/runtime/tokenizer-state/epoch" ]]; then
  date -u +'%Y%m%dT%H%M%SZ' > "$WORKDIR/runtime/tokenizer-state/epoch"
fi

# P 不重复贡献 L2 池；16 个独立 owner 提供共享 L2/L3。
python3 - "$CONFIG" "$P0_IP" "$MOONCAKE_MASTER_PORT" "${P_STORE_DEVICES[$INSTANCE]}" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "metadata_server": "P2PHANDSHAKE",
    "mode": "standalone-store",
    "master_server_address": sys.argv[2] + ":" + sys.argv[3],
    "protocol": "rdma",
    "global_segment_size": 0,
    "local_buffer_size": 4 * 1024**3,
    "device_name": sys.argv[4],
    "enable_offload": True,
}, indent=2) + "\n")
PY

KV_TRANSFER_CONFIG=$(cat <<'JSON'
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_both",
  "kv_buffer_device": "cuda",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_producer",
        "kv_buffer_device": "cuda",
        "kv_connector_extra_config": {
          "backends": ["UCX"],
          "num_threads": 1,
          "kv_lease_duration": 600
        }
      },
      {
        "kv_connector": "MooncakeStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
          "cache_prefix": "xinhao-roce4c711-20261103",
          "lookup_async": true
        }
      }
    ]
  }
}
JSON
)

docker run -d \
  --ulimit nofile=1048576:1048576 \
  --name "xinhao-roce4c711-p$NODE-$INSTANCE" \
  --pull=never --gpus all \
  --device /dev/infiniband \
  --network host --ipc host --pid host \
  --cpuset-mems "$NUMA" \
  --cap-add IPC_LOCK --cap-add SYS_NICE \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --log-opt max-size=128m --log-opt max-file=5 \
  -e NO_PROXY='*' -e no_proxy='*' \
  -v "$MODEL_DIR:/model:ro" \
  -v "$CONFIG:/etc/vllm/mooncake.json:ro" \
  -v "$WORKDIR/runtime/tokenizer-state:/tokenizer-state:ro" \
  -v "$CACHE:/root/.cache" \
  -v "$CACHE/cuda:/root/.nv/ComputeCache" \
  -v "$CACHE/tilelang:/root/.tilelang" \
  -e CUDA_VISIBLE_DEVICES="$GPU_FIRST,$GPU_SECOND" \
  -e VLLM_HOST_IP="$P_IP" \
  -e MOONCAKE_CONFIG_PATH=/etc/vllm/mooncake.json \
  -e MC_GID_INDEX="$IB_GID_INDEX" \
  -e MC_ENABLE_DEST_DEVICE_AFFINITY=1 \
  -e WITH_NVIDIA_PEERMEM=0 -e MC_MAX_MR_SIZE=34359738368 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_DSV41_TOKENIZER_CACHE_GIB="${TOKENIZER_CACHE_GIB:-4}" \
  -e VLLM_DSV41_TOKENIZER_CACHE_EPOCH_FILE=/tokenizer-state/epoch \
  -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  -e VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1 \
  -e VLLM_NIXL_SIDE_CHANNEL_HOST="$P_IP" \
  -e VLLM_NIXL_SIDE_CHANNEL_PORT="$NIXL_PORT" \
  -e NCCL_SOCKET_IFNAME="=$SOCKET_IFNAME" -e GLOO_SOCKET_IFNAME="$SOCKET_IFNAME" \
  -e NCCL_IB_DISABLE=0 -e NCCL_NET=IB -e NCCL_IB_HCA="=$P_HCAS" \
  -e NCCL_IB_GID_INDEX="$IB_GID_INDEX" \
  -e UCX_TLS=rc,sm,self,cuda_copy,cuda_ipc -e UCX_NET_DEVICES="$P_HCAS" \
  -e UCX_IB_GID_INDEX="$IB_GID_INDEX" \
  -e UCX_MAX_RNDV_RAILS=1 -e UCX_IB_ROCE_REACHABILITY_MODE=all \
  -e UCX_IB_GPU_DIRECT_RDMA=yes \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 -e VLLM_SERVER_DEV_MODE=1 \
  xinhao-vllm-roce-megamoe:4c711ac-20261103 /model \
  --served-model-name "$MODEL_NAME" \
  --tokenizer-mode deepseek_v41 \
  --api-server-count 8 --renderer-num-workers 1 --enable-scale-out \
  --tensor-parallel-size 1 --data-parallel-size 2 \
  --data-parallel-backend mp --data-parallel-rpc-port "$DP_PORT" \
  --enable-expert-parallel --enable-ep-weight-filter --enable-eplb \
  --numa-bind --numa-bind-nodes "$NUMA" "$NUMA" \
  --max-model-len -1 \
  --max-num-seqs 512 --max-num-batched-tokens 16384 \
  --kv-cache-dtype fp8 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --kernel-config '{"moe_backend":"deep_gemm_mega_moe","sparse_indexer_topk_backend":"deep_select"}' \
  --engram-config '{"cpu_offload":true,"embedding_across_dp":true,"dp_shared_memory":true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":false}' \
  --enable-prefix-caching \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  --enable-prompt-tokens-details \
  --reasoning-parser deepseek_v41 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 \
  --host 0.0.0.0 --port "$HTTP_PORT"
