#!/usr/bin/env bash
# Verified Docker command snapshot: B300 HiSparse PD, 2026-11-14.
set -euo pipefail

command=(
  docker run -d
  --name kimi31-pd-hisparse-93-decode-c256-r17-gpu095
  --pull=never
  --runtime=runc
  --network host
  --restart no
  --label ai-infra.plan=kimi31-b300-92-93-p8-d8-tp8-dcp8-ep8
  --label ai-infra.node=93
  --label ai-infra.role=decode
  --log-driver local
  --log-opt max-size=20m
  --log-opt max-file=3
  --gpus all
  --ipc host
  --device /dev/infiniband
  --ulimit memlock=-1:-1
  --ulimit nofile=1048576:1048576
  --cap-add IPC_LOCK
  --cap-add SYS_NICE
  --security-opt label=disable
  --mount 'type=bind,src=/data/models/Kimi-K3.1-Pro,dst=/model,readonly'
  --mount 'type=bind,src=/data2/bench/eng-yangyufei/kimi31-slo-b30093-20261110/draft,dst=/draft,readonly'
  --mount 'type=bind,src=/data2/bench/eng-xinhao/hisparse-pd-correctness-20261113-ced31f9/full,dst=/work'
  --mount 'type=bind,src=/data2/bench/eng-xinhao/hisparse-pd-correctness-20261113-ced31f9/full/runtime/decode-r16/cache,dst=/root/.cache'
  --mount 'type=bind,src=/data2/bench/eng-xinhao/hisparse-pd-correctness-20261113-ced31f9/full/runtime/decode-r16/triton,dst=/root/.triton'
  -e VLLM_SERVER_DEV_MODE=1
  -e VLLM_USE_RUST_FRONTEND=1
  -e PYTHONPATH=/work
  -e KIMI31_TOKENIZER_CACHE_GIB=12
  -e KIMI31_TOKENIZER_CACHE_EPOCH_FILE=/work/runtime/tokenizer-cache.epoch
  -e KIMI31_API_NUMA_BIND=1
  -e KIMI31_PD_ROLE=decode
  -e VLLM_USE_V2_MODEL_RUNNER=1
  -e VLLM_HOST_IP=11.5.0.93
  -e GLOO_SOCKET_IFNAME=enp232s0f0np0
  -e NCCL_SOCKET_IFNAME==enp232s0f0np0
  -e 'NCCL_IB_HCA==mlx5_0:1,mlx5_6:1,mlx5_8:1,mlx5_10:1,mlx5_12:1,mlx5_14:1,mlx5_18:1,mlx5_20:1'
  -e NCCL_IB_GID_INDEX=3
  -e NCCL_IB_DISABLE=0
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200
  -e VLLM_FASTSAFETENSORS_QUEUE_SIZE=-1
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
  -e HF_HOME=/work/runtime/decode-r16/hf-cache
  -e VLLM_CACHE_ROOT=/work/runtime/decode-r16/vllm-cache
  -e CUDA_CACHE_PATH=/work/runtime/decode-r16/cuda-cache
  -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/work/runtime/decode-r16/flashinfer-cache
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS
  -e OMP_NUM_THREADS=8
  -e PYTHONUNBUFFERED=1
  -e 'NO_PROXY=*'
  -e 'no_proxy=*'
  -e MC_ENABLE_DEST_DEVICE_AFFINITY=1
  -e MC_GID_INDEX=3
  -e VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=600
  -e VLLM_MOONCAKE_STORE_TIER_LOG=1
  -e VLLM_USE_DIRECT_DCP_A2A=1
  -e VLLM_USE_DIRECT_DCP_Q_GATHER=1
  -e VLLM_USE_DIRECT_DCP_KV_GATHER=1
  -e VLLM_ALLREDUCE_USE_FLASHINFER=1
  -e VLLM_USE_NCCL_SYMM_MEM=0
  -e VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT=1
  -e VLLM_ENABLE_GEMM_RS=0
  -e TOKENSPEED_MLA_FP16_PARTIALS=0
  -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
  -e VLLM_MOONCAKE_LOAD_RECV_THREADS=4
  -e VLLM_MOONCAKE_STORE_SELECTIVE_LEASE=1
  -e MC_SLICE_SIZE=1048576
  -e MC_MAX_MR_SIZE=4294967296
  -e VLLM_MOONCAKE_BOOTSTRAP_PORT=29700
  --entrypoint vllm
  sha256:19bcda671d9ccc5d6c8775627bc3a88eed4cbb25032ac21af5075fae0c8e8b71
  serve /model
  --served-model-name kimi-k3.1-pro
  --trust-remote-code
  --host 11.5.0.93
  --port 19581
  --tensor-parallel-size 8
  --pipeline-parallel-size 1
  --data-parallel-size 1
  --data-parallel-backend mp
  --decode-context-parallel-size 8
  --api-server-count 1
  --renderer-num-workers 1
  --worker-extension-cls pd_admin.CapacityObserver
  --enable-expert-parallel
  --moe-backend deep_gemm_mega_moe
  --max-model-len 1048576
  --load-format fastsafetensors
  --enable-auto-tool-choice
  --tool-call-parser kimi_k3
  --reasoning-parser kimi_k3
  --enable-prompt-tokens-details
  --enable-per-request-metrics
  --no-enable-prefix-caching
  --mamba-cache-mode none
  --use-replayssm
  --seed 42
  --disable-uvicorn-access-log
  --dcp-comm-backend a2a
  --cp-kv-cache-interleave-size 1
  --kv-cache-dtype nvfp4_ds_mla
  --attention-config '{"indexer_kv_dtype": "mxfp4", "tokenspeed_mla_min_split_kv": 16, "hisparse_config": {"device_buffer_size": 6144}}'
  --mamba-cache-dtype auto
  --mamba-ssm-cache-dtype float32
  --no-enable-flashinfer-autotune
  --language-model-only
  --stream-interval 10
  --max-num-seqs 256
  --max-num-batched-tokens 2048
  --gpu-memory-utilization 0.95
  --speculative-config '{"model":"/draft","method":"dspark","num_speculative_tokens":4,"draft_sample_method":"probabilistic","rejection_sample_method":"block","attention_backend":"TOKENSPEED_MLA","kv_cache_dtype":"fp8"}'
  --kv-transfer-config '{"kv_connector": "MultiConnector", "kv_role": "kv_both", "kv_load_failure_policy": "fail", "kv_connector_extra_config": {"connectors": [{"kv_connector": "HiSparseConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"host_pool_gib": 1024}}, {"kv_connector": "MooncakeConnector", "kv_role": "kv_consumer", "kv_load_failure_policy": "fail", "kv_connector_extra_config": {"mooncake_protocol": "rdma", "device_name": "mlx5_0,mlx5_6,mlx5_8,mlx5_10,mlx5_12,mlx5_14,mlx5_18,mlx5_20"}}]}}'
  --prefill-schedule-interval 8
  --additional-config '{"decode_backend":"latency_specialized"}'
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "max_cudagraph_capture_size": 1280, "cudagraph_capture_sizes": [5, 10, 20, 40, 60, 80, 100, 120, 160, 200, 240, 280, 320, 400, 480, 560, 640, 800, 960, 1120, 1280]}'
  --watermark 0.05
)

case "${1:---dry-run}" in
  --dry-run)
    printf '%q ' "${command[@]}"
    printf '\n'
    ;;
  --execute)
    ip -o -4 addr show | grep -F '11.5.0.93/' >/dev/null || { echo 'Run on b300-93' >&2; exit 1; }
    if docker inspect kimi31-pd-hisparse-93-decode-c256-r17-gpu095 >/dev/null 2>&1; then
      echo 'Container already exists; inspect it before starting another instance.' >&2
      exit 1
    fi
    exec "${command[@]}"
    ;;
  *)
    echo 'Usage: [--dry-run|--execute]' >&2
    exit 2
    ;;
esac
