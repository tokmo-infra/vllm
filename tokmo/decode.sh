#!/usr/bin/env bash
# Image: xinhao-vllm-roce-megamoe:4c711ac-20261103 (sha256:19177b08b707...)
# 92/93：一个跨机 DP16/EP16 D；93 为 rank8..15 的 headless 节点。
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "${1:?节点号 92|93}"
[[ $# == 1 && ( "$NODE" == 92 || "$NODE" == 93 ) ]] || exit 2
D_SPECULATIVE_TOKENS=${D_SPECULATIVE_TOKENS:-5}
[[ "$D_SPECULATIVE_TOKENS" == 1 || "$D_SPECULATIVE_TOKENS" == 5 ]] || exit 2
SPEC_CONFIG="{\"method\":\"dspark\",\"num_speculative_tokens\":$D_SPECULATIVE_TOKENS,\"enable_adaptive_verification\":false}"
PROFILE_FLAGS=()
if [[ -n "${D_PROFILE_DIR:-}" ]]; then
  [[ "$D_PROFILE_DIR" == /evidence/* ]]
  PROFILE_FLAGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$D_PROFILE_DIR\",\"torch_profiler_with_stack\":false,\"torch_profiler_activities\":[\"CPU\",\"CUDA\"],\"ignore_frontend\":true,\"delay_iterations\":4,\"max_iterations\":16}")
  PROFILE_FLAGS+=(--cudagraph-metrics)
fi
PROFILE_MOUNTS=()
if [[ -n "${D_PROFILE_SOURCE:-}" ]]; then
  [[ -f "$D_PROFILE_SOURCE" && -n "${D_PROFILE_DIR:-}" ]]
  PROFILE_MOUNTS+=(-v "$D_PROFILE_SOURCE:/usr/local/lib/python3.12/dist-packages/vllm/distributed/hierarchical_mega_moe.py:ro")
fi
CACHE="$WORKDIR/runtime/cache-roce4c711/d$NODE"
mkdir -p "$CACHE/"{cuda,tilelang} "$WORKDIR/validation"
D_FLAGS=(--data-parallel-size-local 8 --data-parallel-address "$D0_IP")
if [[ "$NODE" == 92 ]]; then
  # An explicit start-rank on an API node infers hybrid LB in this main.
  # Omit it so the head owns all 16 engines, including headless node 93.
  # Native internal-LB default is the global DP size. Match two DEP8 heads'
  # aggregate 16 APIs; eight APIs overflowed the HTTP listener in run004.
  D_FLAGS+=(--api-server-count 16 --renderer-num-workers 1)
else
  D_FLAGS+=(--data-parallel-start-rank 8 --headless)
fi
NET_EXTRA=()
case "${MEGA_MOE_GDR_READ:-auto}" in
  auto) ;;
  0|1) NET_EXTRA=(-e "NCCL_NET_GDR_READ=$MEGA_MOE_GDR_READ") ;;
  *) exit 2 ;;
esac

docker run -d \
  --ulimit nofile=1048576:1048576 \
  --name "xinhao-roce4c711-d$NODE" \
  --pull=never --gpus all \
  --device /dev/infiniband \
  --network host --ipc host --pid host \
  --cap-add IPC_LOCK --cap-add SYS_NICE \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --log-opt max-size=128m --log-opt max-file=5 \
  -e NO_PROXY='*' -e no_proxy='*' \
  -e MC_ENABLE_DEST_DEVICE_AFFINITY=1 \
  -v "$MODEL_DIR:/model:ro" \
  -v "$CACHE:/root/.cache" \
  -v "$CACHE/cuda:/root/.nv/ComputeCache" \
  -v "$CACHE/tilelang:/root/.tilelang" \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e VLLM_HOST_IP="$NODE_IP" \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -v "$WORKDIR/validation:/evidence" \
  "${PROFILE_MOUNTS[@]}" \
  -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  -e VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1 \
  -e VLLM_NIXL_SIDE_CHANNEL_HOST="$NODE_IP" \
  -e VLLM_NIXL_SIDE_CHANNEL_PORT=18600 \
  -e NCCL_SOCKET_IFNAME="=$SOCKET_IFNAME" -e GLOO_SOCKET_IFNAME="$SOCKET_IFNAME" \
  -e NCCL_IB_DISABLE=0 -e NCCL_NET=IB -e NCCL_IB_HCA="=$EP_RDMA_DEVICES" \
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET \
  -e NCCL_DEBUG_FILE='/evidence/nccl-%h-%p.log' \
  -e NCCL_IB_GID_INDEX="$IB_GID_INDEX" \
  ${NET_EXTRA[@]+"${NET_EXTRA[@]}"} \
  -e UCX_TLS=rc,sm,self,cuda_copy,cuda_ipc -e UCX_NET_DEVICES="$RDMA_DEVICES" \
  -e UCX_IB_GID_INDEX="$IB_GID_INDEX" \
  -e UCX_MAX_RNDV_RAILS=1 -e UCX_IB_ROCE_REACHABILITY_MODE=all \
  -e UCX_IB_GPU_DIRECT_RDMA=yes \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 -e VLLM_SERVER_DEV_MODE=1 \
  xinhao-vllm-roce-megamoe:4c711ac-20261103 /model \
  --served-model-name "$MODEL_NAME" \
  --tokenizer-mode deepseek_v41 \
  "${D_FLAGS[@]}" \
  --tensor-parallel-size 1 --data-parallel-size 16 \
  --data-parallel-backend mp --data-parallel-rpc-port 26360 \
  --enable-expert-parallel --enable-ep-weight-filter --enable-eplb \
  --eplb-config '{"communicator":"nixl","use_async":true,"window_size":1000,"step_interval":3000}' \
  --numa-bind \
  --max-model-len -1 \
  --gpu-memory-utilization "${D_GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-num-seqs 512 --max-num-batched-tokens 4096 \
  --kv-cache-dtype fp8 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --kernel-config '{"moe_backend":"deep_gemm_hierarchical_mega_moe","sparse_indexer_topk_backend":"deep_select","hierarchical_mega_moe_transport":"verbs","hierarchical_mega_moe_overlap":true,"hierarchical_mega_moe_chunk_size":2048,"hierarchical_mega_moe_wire_dtype":"mxfp8","hierarchical_mega_moe_direct_staging":true,"hierarchical_mega_moe_rdma_hcas":["mlx5_0:1","mlx5_6:1","mlx5_8:1","mlx5_10:1","mlx5_12:1","mlx5_14:1","mlx5_18:1","mlx5_20:1"],"hierarchical_mega_moe_rdma_gid_index":3,"hierarchical_mega_moe_rdma_qps":4,"hierarchical_mega_moe_rdma_traffic_class":106,"hierarchical_mega_moe_rdma_host_staging":false}' \
  --engram-config '{"cpu_offload":true,"embedding_across_dp":true,"dp_shared_memory":true}' \
  --speculative-config "$SPEC_CONFIG" \
  "${PROFILE_FLAGS[@]}" \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda","kv_load_failure_policy":"fail","kv_connector_extra_config":{"backends":["UCX"],"num_threads":1}}' \
  --enable-prompt-tokens-details \
  --reasoning-parser deepseek_v41 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 \
  --host 0.0.0.0 --port 30011
