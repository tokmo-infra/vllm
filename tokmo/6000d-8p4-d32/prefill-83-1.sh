#!/usr/bin/env bash
# P：6000d-83，实例 1，GPU 4,5,6,7，TP1 / DP4 / EP4。
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

NAME=glm53flash-8p4-d32-xinhao-prefill-83-1
IMAGE=sha256:540155b18ee763049423a616628cc1fee5e22176b3b1397775465f9067569bcf
WORKDIR="${GLM_WORKDIR:-/data/bench/eng-xinhao/vllm-pd/glm53-flash/6000d-8p4-d32}"
MODEL_DIR="${GLM_MODEL_DIR:-/data/models/GLM-5.3-Flash-NVFP4}"
CACHE="$WORKDIR/cache/prefill-83-1"
MOONCAKE_CONFIG_FILE="$WORKDIR/config/mooncake-prefill-83-1.json"

# 本实例的 Mooncake 客户端配置；启动时仅写入自己的配置文件。
MOONCAKE_CONFIG=$(cat <<'JSON'
{
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "11.5.0.81:54365",
  "protocol": "rdma",
  "device_name": "mlx5_4,mlx5_5,mlx5_6,mlx5_7",
  "mode": "standalone-store",
  "global_segment_size": 0,
  "local_buffer_size": 2147483648,
  "enable_offload": false,
  "tenant_id": "default"
}
JSON
)

# P/D 与 CPU 缓存连接器；仅配置连接，不启动其他服务。
KV_TRANSFER_CONFIG=$(cat <<'JSON'
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_both",
  "engine_id": "glm53flash-8p4-d32-xinhao-prefill-83-1",
  "kv_buffer_device": "cuda",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
          "mooncake_protocol": "rdma",
          "device_name": "mlx5_4,mlx5_5,mlx5_6,mlx5_7"
        }
      },
      {
        "kv_connector": "MooncakeStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
          "cache_prefix": "glm53flash-mc-native-mtp3-h256-8p4-d32-540155-v4",
          "load_async": true,
          "lookup_async": true,
          "store_compact_group_payloads": true,
          "preferred_segment": "11.5.0.83:15466"
        }
      }
    ]
  }
}
JSON
)

command=(
  docker run -d --init
  --name "$NAME"
  --pull=never
  --network host
  --log-opt max-size=128m
  --log-opt max-file=5
  --label ai-infra.project=glm53flash-8p4-d32-xinhao
  --device /dev/infiniband
  --ipc host
  --cap-add IPC_LOCK
  --ulimit memlock=-1
  --ulimit stack=67108864
  --gpus all
  --pid host
  --cap-add SYS_NICE
  --security-opt seccomp=unconfined

  # 进程环境
  -e NO_PROXY='*'
  -e no_proxy='*'
  -e HF_HUB_OFFLINE=1
  -e VLLM_NO_USAGE_STATS=1
  -e MC_TCP_BIND_ADDRESS=11.5.0.83
  -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME=11.5.0.83
  -e WITH_NVIDIA_PEERMEM=0
  -e "MC_TE_FILTERS=mlx5_4,mlx5_5,mlx5_6,mlx5_7"
  -e "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
  -e VLLM_HOST_IP=11.5.0.83
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS
  -e VLLM_SERVER_DEV_MODE=1
  -e VLLM_KV_CACHE_LAYOUT=BLHNC
  -e MOONCAKE_CONFIG_PATH=/etc/vllm/mooncake.json
  -e VLLM_MOONCAKE_BOOTSTRAP_PORT=27700
  -e VLLM_USE_V2_MODEL_RUNNER=1
  -e VLLM_MOONCAKE_LOAD_RECV_THREADS=4
  -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120
  -e NCCL_IB_GID_INDEX=3
  -e MC_GID_INDEX=3
  -e NCCL_SOCKET_IFNAME=enp3s0f0np0
  -e GLOO_SOCKET_IFNAME=enp3s0f0np0
  -e NCCL_IB_DISABLE=0
  -e NCCL_NET=IB
  -e NCCL_IB_HCA="=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1"
  # NCCL P2P 距离上限；DeepEP 的实际 PIX P2P 另做运行验收。
  -e NCCL_P2P_DISABLE=0
  -e NCCL_P2P_LEVEL=PXB
  -e NCCL_DEBUG=INFO
  -e "NCCL_DEBUG_SUBSYS=INIT,NET"
  -e VLLM_ALLREDUCE_USE_SYMM_MEM=0
  -e PYTHONHASHSEED=0
  # 每个 API 进程独立的原生 CPU 分词缓存；headless D 无 API。
  -e VLLM_GLM53_TOKENIZER_CACHE_GIB=4
  -e VLLM_GLM53_TOKENIZER_CACHE_EPOCH_FILE=/root/.cache/tokenizer-epoch

  # 本服务的挂载
  -v "$WORKDIR/runtime/glm53_segment_cache_core.py:/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/glm53_segment_cache_core.py:ro"
  -v "$WORKDIR/runtime/glm53_native_cache.py:/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/glm53_native_cache.py:ro"
  -v "$WORKDIR/runtime/vllm_renderer_base.py:/usr/local/lib/python3.12/dist-packages/vllm/renderers/base.py:ro"
  -v "$WORKDIR/runtime/vllm_custom_ops.py:/usr/local/lib/python3.12/dist-packages/vllm/_custom_ops.py:ro"
  -v "$MODEL_DIR:/model:ro"
  -v "$CACHE:/root/.cache:rw"
  -v "$CACHE/cuda:/root/.nv/ComputeCache:rw"
  -v "$CACHE/triton:/root/.triton/cache:rw"
  -v "$CACHE/tilelang:/root/.tilelang:rw"
  -v "$CACHE/deep_ep:/root/.deep_ep:rw"
  -v "$MOONCAKE_CONFIG_FILE:/etc/vllm/mooncake.json:ro"

  --entrypoint vllm
  "$IMAGE"
  serve /model
  --served-model-name glm-5.3-flash-nvfp4
  --api-key EMPTY
  --language-model-only
  --device-ids "4,5,6,7"
  --tensor-parallel-size 1
  --data-parallel-size 4
  --data-parallel-size-local 4
  --data-parallel-backend mp
  --data-parallel-address 11.5.0.83
  --data-parallel-rpc-port 18460
  --enable-expert-parallel
  --enable-ep-weight-filter
  --all2all-backend allgather_reducescatter
  --numa-bind
  --numa-bind-nodes 1 1 1 1
  --max-model-len -1
  --max-num-seqs 64
  --max-num-batched-tokens 4096
  --gpu-memory-utilization 0.90
  --kv-cache-dtype fp8
  --block-size 64
  --attention-backend FLASHINFER_MLA_SPARSE_SM120
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  --mamba-cache-mode align
  --mamba-ssm-cache-dtype float32
  --no-disable-hybrid-kv-cache-manager
  --enable-chunked-prefill
  --enable-prefix-caching
  --prefix-match-unit 256
  --kv-transfer-config "$KV_TRANSFER_CONFIG"
  --reasoning-parser glm47
  --enable-auto-tool-choice
  --tool-call-parser glm47
  --enable-prompt-tokens-details
  --enable-force-include-usage
  --safetensors-load-strategy lazy
  --enable-cumem-allocator
  --scheduler-reserve-full-isl
  --seed 42
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":256}'
  --api-server-count 4
  --renderer-num-workers 1
  --host 0.0.0.0
  --port 31014
)

case "$MODE" in
  --print-argv)
    printf '%s\0' "${command[@]}"
    exit 0 ;;
  --print-config)
    printf '%s\n' "$MOONCAKE_CONFIG"
    exit 0 ;;
  --dry-run)
    printf '# Mooncake client config: %s\n%s\n\n' "$MOONCAKE_CONFIG_FILE" "$MOONCAKE_CONFIG"
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

# 只准备本实例自己的缓存与客户端配置。
mkdir -p "$CACHE/cuda" "$CACHE/triton" "$CACHE/tilelang" "$CACHE/deep_ep" "$WORKDIR/config"
printf '%s\n' "$MOONCAKE_CONFIG" > "$MOONCAKE_CONFIG_FILE"
printf '%s\n' "$(date -u +%Y%m%dT%H%M%S)-$$" > "$CACHE/tokenizer-epoch"

exec "${command[@]}"
