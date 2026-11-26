#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
source ./deploy.env
: "${NODE_ROLE:?请先设置 NODE_ROLE=p或 d1}"
# shellcheck disable=SC2153
case "$NODE_ROLE" in
  p)  NODE_IP="$P_IP"; NET_IF="$P_NET_IF"; GID_INDEX="$P_GID_INDEX"; NODE_HCAS="$P0_HCAS,$P1_HCAS" ;;
  d1) NODE_IP="$D1_IP"; NET_IF="$D1_NET_IF"; GID_INDEX="$D1_GID_INDEX"; NODE_HCAS="$D1_HCAS" ;;
  *) echo "未知角色：$NODE_ROLE" >&2; exit 2 ;;
esac
ip -o -4 addr show dev "$NET_IF" |
  awk -v ip="$NODE_IP" '{split($4,a,"/"); if (a[1]==ip) ok=1} END {exit !ok}' || {
  echo "请核对 $NODE_ROLE 的 $NET_IF 是否配置了 $NODE_IP。" >&2; exit 1;
}

MASTER_IP="$D1_IP"
MASTER_RPC="$MASTER_IP:$MASTER_RPC_PORT"
MASTER_METADATA_URL="http://$MASTER_IP:$MASTER_METADATA_PORT/metadata"
MASTER_METRICS_URL="http://$MASTER_IP:$MASTER_METRICS_PORT/metrics"
P0_URL="http://$P_IP:$P0_HTTP_PORT"; P1_URL="http://$P_IP:$P1_HTTP_PORT"
D1_URL="http://$D1_IP:$D1_HTTP_PORT"
BASE_WORK="$SERVICE_ROOT/$SERVICE_PREFIX"
CACHE_NAMESPACE="$SERVICE_PREFIX-nvfp4-prefix256"
TOKENIZER_DIR="/dev/shm/$SERVICE_PREFIX/tokenizer"
test -f "$INTERNAL_API_KEY_FILE"; test ! -L "$INTERNAL_API_KEY_FILE"
test "$(stat -c '%u:%g:%a' "$INTERNAL_API_KEY_FILE")" = '0:0:600'
INTERNAL_API_KEY=$(cat -- "$INTERNAL_API_KEY_FILE")
test -n "$INTERNAL_API_KEY"
MODEL_NAME=glm-5.3-flash
here="$PWD"

P_IMAGE="${ENGINE_IMAGE:-local/glm53flash-dflash2:r15b-rust-pd-85-86}"
D_IMAGE="$P_IMAGE"
MASTER_IMAGE="${MOONCAKE_IMAGE:-local/glm53flash-mooncake:source-71973589}"
STORE_IMAGE="$MASTER_IMAGE"
ROUTER_IMAGE="${ROUTER_IMAGE:-router:glm53flash-api-v7-20261124}"

start_master() {
test "$NODE_ROLE" = d1
LIBCUDA_PATH=$(readlink -f /lib/x86_64-linux-gnu/libcuda.so.1)
test -f "$LIBCUDA_PATH"
docker run -d --pull=never --restart unless-stopped --name "$SERVICE_PREFIX-master" \
  --network host --runtime=runc --memory 4g \
  -e NVIDIA_VISIBLE_DEVICES=void \
  -e MC_TCP_BIND_ADDRESS="$MASTER_IP" \
  -v "$LIBCUDA_PATH:/opt/host-driver/libcuda.so.1:ro" \
  -e LD_LIBRARY_PATH=/opt/host-driver:/usr/local/cuda/lib64 \
  -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME="$MASTER_IP" \
  -e MC_EVICTION_FIT_REQUEST_BYTES=60318720 \
  -e MC_EVICTION_FIT_MIN_SLOTS=64 -e MC_EVICTION_FIT_CHECK_MS=1000 \
  --entrypoint mooncake_master "$MASTER_IMAGE" \
  --rpc_port="$MASTER_RPC_PORT" --enable_http_metadata_server=true \
  --http_metadata_server_host="$MASTER_METADATA_BIND_HOST" --http_metadata_server_port="$MASTER_METADATA_PORT" \
  --metrics_port="$MASTER_METRICS_PORT" --eviction_high_watermark_ratio=0.95 \
  --enable_offload=true --offload_on_evict=true \
  --allocation_strategy=local_first --logtostderr=true

curl -fsS --retry 20 --retry-connrefused --retry-delay 2 "$MASTER_METRICS_URL" >/dev/null
}
prepare_model_files() {
test -f "$MODEL_DIR/tokenizer.json"
# NUMA 包装脚本保存在本服务目录，由各实例只读挂载。
install -d -m 0700 "$BASE_WORK"
if ! cmp -s "$here/numa_wrapper.sh" "$BASE_WORK/numa_wrapper.sh"; then
  install -m 0755 "$here/numa_wrapper.sh" "$BASE_WORK/numa_wrapper.sh"
fi
install -d -m 0700 "$TOKENIZER_DIR"
if [[ ! -e "$TOKENIZER_DIR/epoch" ]]; then
  (set -o noclobber; cat /proc/sys/kernel/random/uuid > "$TOKENIZER_DIR/epoch")
fi
test ! -L "$TOKENIZER_DIR/epoch"; test -s "$TOKENIZER_DIR/epoch"
chown 0:0 "$TOKENIZER_DIR" "$TOKENIZER_DIR/epoch"
chmod 0700 "$TOKENIZER_DIR"
chmod 0600 "$TOKENIZER_DIR/epoch"


}
write_mc_config() {
  cat > "$1" <<EOF
{
  "mode": "standalone-store",
  "metadata_server": "$MASTER_METADATA_URL",
  "master_server_address": "$MASTER_RPC",
  "protocol": "rdma",
  "device_name": "$2",
  "global_segment_size": 0,
  "local_buffer_size": 2147483648,
  "enable_offload": true,
  "tenant_id": "default"
}
EOF
}

start_model() {
  local slot="$1" image gpu_request hcas port bootstrap rpc dp apis
  local seq batch shared_gib cache_gib pd_role store_role role_work engine_id kv ib_hca cache_dir
  local retire_checkpoints=false gpu_util=0.95
  local -a role_env=() graph_args=() reset_mount=()
  if [[ -e "$BASE_WORK/cache-reset.key" ]]; then
    test -f "$BASE_WORK/cache-reset.key"; test ! -L "$BASE_WORK/cache-reset.key"
    test "$(stat -c '%u:%g:%a' "$BASE_WORK/cache-reset.key")" = '0:0:600'
    grep -Eq '^[[:xdigit:]]{64}$' "$BASE_WORK/cache-reset.key"
    reset_mount=(-v "$BASE_WORK/cache-reset.key:/run/bench84-reset-key:ro")
  fi
  case "$slot" in
    p0|p1)
      image="$P_IMAGE"
      dp=4; apis=1; seq=24; batch=2048; shared_gib=16; cache_gib=2
      gpu_util="${P_GPU_UTIL:-0.95}"; pd_role=kv_producer; store_role=kv_both; retire_checkpoints=true
      role_env=(-e VLLM_GLM53_P_MTP_CONTEXT_ONLY=0 -e VLLM_GLM53_P_READY_WAIT=1
                -e VLLM_GLM53_NVFP4_EXTENDED_PREFETCH=1
                -e VLLM_MOONCAKE_LOAD_RECV_THREADS=4
                -e VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION=1
                -e VLLM_HOT_TAIL_ADMISSION_DIAGNOSTICS=0)
      if [ "$slot" = p0 ]; then
        gpu_request='"device=0,1,2,3"'; hcas="$P0_HCAS"
        port="$P0_HTTP_PORT"; bootstrap="$P0_BOOTSTRAP_PORT"; rpc="$P0_DP_RPC_PORT"
      else
        gpu_request='"device=4,5,6,7"'; hcas="$P1_HCAS"
        port="$P1_HTTP_PORT"; bootstrap="$P1_BOOTSTRAP_PORT"; rpc="$P1_DP_RPC_PORT"
      fi
      ;;
    d0)
      image="$D_IMAGE"
      dp=8; apis=1; seq=32; batch=1024; shared_gib=16; cache_gib=16
      role_env=(-e VLLM_GLM53_P_MTP_CONTEXT_ONLY=0
                -e VLLM_GLM53_NVFP4_EXTENDED_PREFETCH=0)
      pd_role=kv_consumer; store_role=kv_consumer
      gpu_request='"device=0,1,2,3,4,5,6,7"'; hcas="$NODE_HCAS"
      case "$NODE_ROLE" in
        d1) port="$D1_HTTP_PORT"; bootstrap="$D1_BOOTSTRAP_PORT"; rpc="$D1_DP_RPC_PORT" ;;
        *) return 1 ;;
      esac
      graph_args=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}')
      ;;
    *) return 1 ;;
  esac
  ib_hca="=$(printf '%s' "$hcas" | sed 's/,/:1,/g'):1"
  role_work="$BASE_WORK/$slot"
  install -d -m 0700 "$role_work"
  cache_dir="$COMPILE_CACHE_ROOT/$slot"
  install -d -m 0700 "$cache_dir" "$cache_dir/tilelang"
  local -a capture_sizes=(1 2 4 6 8 12 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144)
  [[ $slot != d0 ]] || capture_sizes+=(152 160 168 176 184 192)
  write_mc_config "$role_work/mooncake.json" "$hcas"
  engine_id="$SERVICE_PREFIX-$slot-$NODE_ROLE-$(cat /proc/sys/kernel/random/uuid)"
  kv=$(cat <<EOF
{
  "kv_connector":"MultiConnector","kv_role":"kv_both","engine_id":"$engine_id",
  "kv_load_failure_policy":"fail",
  "kv_connector_extra_config":{"connectors":[
    {"kv_connector":"MooncakeConnector","kv_role":"$pd_role",
     "kv_connector_extra_config":{"mooncake_protocol":"rdma","device_name":"$hcas"}},
    {"kv_connector":"MooncakeStoreConnector","kv_role":"$store_role",
     "kv_connector_extra_config":{"cache_prefix":"$CACHE_NAMESPACE",
       "load_async":true,"lookup_async":true,"store_compact_group_payloads":true,
       "remove_superseded_checkpoints":$retire_checkpoints}}
  ]}
}
EOF
)
  docker run -d --pull=never --restart unless-stopped --name "$SERVICE_PREFIX-$slot-$NODE_ROLE" \
    --init --network host --shm-size 64g --memory 64g --memory-swap 64g \
    --gpus "$gpu_request" --device /dev/infiniband \
    --cap-add IPC_LOCK --cap-add SYS_NICE --ulimit memlock=-1:-1 \
    --ulimit stack=67108864:67108864 \
    -v "$MODEL_DIR:/model:ro" -v "$role_work:/work" \
    -v "$DRAFT_DIR:/draft:ro" -v "$cache_dir/tilelang:/root/.tilelang" \
    -e VLLM_USE_RUST_FRONTEND=1 -e VLLM_RUST_FRONTEND_PATH=/usr/local/bin/vllm-rs \
    -v "$cache_dir:/root/.cache" \
    -v "$BASE_WORK/numa_wrapper.sh:/usr/local/lib/python3.12/dist-packages/vllm/utils/numa_wrapper.sh:ro" \
    -v "$TOKENIZER_DIR:/run/glm-tokenizer:rw" \
    -e VLLM_GLM53_TOKENIZER_SHARED_PATH=/run/glm-tokenizer/segments.db \
    -e VLLM_GLM53_TOKENIZER_SHARED_GIB="$shared_gib" \
    -e VLLM_GLM53_TOKENIZER_CACHE_GIB="$cache_gib" \
    -e VLLM_GLM53_TOKENIZER_LOCAL_L1_MIB=128 \
    -e VLLM_GLM53_TOKENIZER_CACHE_EPOCH_FILE=/run/glm-tokenizer/epoch \
    -e MC_TCP_BIND_ADDRESS="$NODE_IP" -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME="$NODE_IP" \
    -e MC_ENABLE_DEST_DEVICE_AFFINITY=1 \
    -e WITH_NVIDIA_PEERMEM=0 \
    -e VLLM_HOST_IP="$NODE_IP" -e NCCL_SOCKET_IFNAME="$NET_IF" \
    -e GLOO_SOCKET_IFNAME="$NET_IF" -e NCCL_IB_HCA="$ib_hca" \
    -e NCCL_IB_GID_INDEX="$GID_INDEX" -e MC_GID_INDEX="$GID_INDEX" -e NCCL_DEBUG=INFO \
    -e MAX_JOBS=4 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_ENGINE_READY_TIMEOUT_S=2400 \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    ${reset_mount[@]+"${reset_mount[@]}"} \
    -e VLLM_GLM53_NVFP4_FUSED_MOE=1 \
    -e VLLM_GLM53_NVFP4_STAGE_FC2_INPUT=1 \
    -e VLLM_GLM53_NVFP4_VECTOR_WEIGHTS=1 \
    -e MOONCAKE_CONFIG_PATH=/work/mooncake.json \
    -e VLLM_MOONCAKE_BOOTSTRAP_PORT="$bootstrap" -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 \
    ${role_env[@]+"${role_env[@]}"} \
    --entrypoint vllm "$image" serve /model \
    --served-model-name "$MODEL_NAME" --host "$NODE_IP" --port "$port" \
    --api-key "$INTERNAL_API_KEY" --api-server-count "$apis" \
    --tensor-parallel-size 1 --data-parallel-size "$dp" --data-parallel-size-local "$dp" \
    --data-parallel-backend mp --data-parallel-rpc-port "$rpc" \
    --enable-expert-parallel --enable-ep-weight-filter \
    --max-model-len auto --max-num-seqs "$seq" --max-num-batched-tokens "$batch" \
    --gpu-memory-utilization "$gpu_util" --kv-cache-dtype nvfp4 \
    --block-size 64 --prefix-match-unit 256 --mamba-cache-mode align \
    --speculative-config '{"method":"dflash","num_speculative_tokens":5,"model":"/draft","kv_cache_dtype":"fp8","attention_backend":"TRITON_ATTN"}' \
    --max-cudagraph-capture-size "$((seq * 6))" --cudagraph-capture-sizes "${capture_sizes[@]}" --cudagraph-metrics \
    --kv-transfer-config "$kv" --scheduler-reserve-full-isl \
    --enable-prefix-caching --reasoning-parser glm47 \
    --enable-auto-tool-choice --tool-call-parser glm47 \
    --safetensors-load-strategy lazy --enable-cumem-allocator --seed 42 \
    --enable-prompt-tokens-details --enable-force-include-usage --numa-bind \
    ${graph_args[@]+"${graph_args[@]}"}
}
start_store() {
case "$NODE_ROLE" in
  p)  STORE_GIB="$L2_P_GIB" ;;
  d1) STORE_GIB="$L2_D1_GIB" ;;
esac
if [[ ! "$STORE_GIB" =~ ^[1-9][0-9]{0,2}$ ]] || (( STORE_GIB > 750 )); then
  echo 'L2 容量必须为 1–750 GiB 的整数' >&2; exit 2;
fi
STORE_BYTES=$((STORE_GIB * 1024 * 1024 * 1024))
# 每台 NVMe 按需使用 2.5 TB（十进制）。
L3_MAX_BYTES=2500000000000
L3_DIR="$L3_MOUNT/l3/$SERVICE_PREFIX"
test "$(findmnt -n -T "$L3_MOUNT" -o TARGET)" = "$L3_MOUNT"
test ! -L "$L3_MOUNT/l3"
mkdir -p "$L3_MOUNT/l3"
test ! -L "$L3_DIR"
install -d -m 0700 "$L3_DIR"
test "$(realpath "$L3_DIR")" = "$L3_DIR"
test "$(findmnt -n -T "$L3_DIR" -o TARGET)" = "$L3_MOUNT"
L3_EXISTING_BYTES=$(du -sxB1 -- "$L3_DIR" | awk '{print $1}')
L3_FREE_BYTES=$(df -B1 --output=avail "$L3_DIR" | tail -1 | tr -d '[:space:]')
(( L3_FREE_BYTES + L3_EXISTING_BYTES >= L3_MAX_BYTES + 200*1024*1024*1024 )) || {
  echo 'Store 需要 2.5 TB L3 容量。' >&2; exit 1;
}
LIBCUDA_PATH=$(readlink -f /lib/x86_64-linux-gnu/libcuda.so.1)
test -f "$LIBCUDA_PATH"
docker run -d --pull=never --restart unless-stopped --name "$SERVICE_PREFIX-store-$NODE_ROLE" \
  --init --network host --ipc host --runtime=runc \
  --memory 832g --memory-swap 832g --ulimit memlock=-1:-1 \
  --device /dev/infiniband --cap-add IPC_LOCK --cap-add SYS_NICE \
  -e NVIDIA_VISIBLE_DEVICES=void -e MC_GID_INDEX="$GID_INDEX" \
  -e MC_TCP_BIND_ADDRESS="$NODE_IP" -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME="$NODE_IP" \
  -e NCCL_SOCKET_IFNAME="$NET_IF" -e GLOO_SOCKET_IFNAME="$NET_IF" \
  -v "$LIBCUDA_PATH:/opt/host-driver/libcuda.so.1:ro" \
  -e LD_LIBRARY_PATH=/opt/host-driver:/usr/local/cuda/lib64 \
  --mount "type=bind,src=$L3_DIR,dst=/mooncake-ssd" \
  -e MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/mooncake-ssd \
  -e MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend \
  -e MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru \
  -e MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE="$L3_MAX_BYTES" \
  -e MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES="$L3_MAX_BYTES" \
  -e MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES="$L3_MAX_BYTES" \
  -e MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1342177280 \
  --entrypoint mooncake_client "$STORE_IMAGE" \
  --master_server_address="$MASTER_RPC" \
  --metadata_server="$MASTER_METADATA_URL" \
  --protocol=rdma --host="$NODE_IP" --port="$STORE_RPC_PORT" \
  --device_names="$NODE_HCAS" --global_segment_size="$STORE_BYTES" \
  --local_buffer_size=0 --enable_http_server=true --http_port="$STORE_HEALTH_PORT" \
  --enable_offload=true --start_offload_rpc_server=true --logtostderr=true
curl -fsS --retry 20 --retry-connrefused --retry-delay 3 "http://$NODE_IP:$STORE_HEALTH_PORT/health"
}
start_router() {
docker run -d --pull=never --name "$SERVICE_PREFIX-router" \
  --runtime=runc --network host --restart unless-stopped \
  --memory 32g --memory-swap 32g --cpuset-mems 0 \
  --ipc private --shm-size 64m --ulimit nofile=1048576:1048576 \
  --log-driver local --log-opt max-size=128m --log-opt max-file=5 \
  -e OPENAI_API_KEY="$INTERNAL_API_KEY" -e NVIDIA_VISIBLE_DEVICES=void \
  -e NO_PROXY='*' -e no_proxy='*' \
  -e API_KEY_VALIDATION_URLS="${ROUTER_CLIENT_VALIDATION_URLS:-}" -e VLLM_ROUTER_GLM_INLINE_IMAGE_MAX_BYTES \
  -e VLLM_ROUTER_PD_REUSE_PROMPT_IDS=1 \
  -e VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP=1 \
  --entrypoint /usr/local/bin/vllm-router "$ROUTER_IMAGE" \
  --host "$ROUTER_BIND_HOST" --port "$ROUTER_PORT" \
  --prometheus-host "$ROUTER_METRICS_BIND_HOST" --prometheus-port "$ROUTER_METRICS_PORT" \
  --vllm-pd-disaggregation --kv-connector mooncake \
  --prefill "$P0_URL" "$P0_BOOTSTRAP_PORT" --prefill "$P1_URL" "$P1_BOOTSTRAP_PORT" \
  --decode "$D1_URL" \
  --prefill-policy cache_aware --decode-policy round_robin \
  --max-tree-size 2147483648 --eviction-interval 120 --max-concurrent-requests 4096 \
  --disable-retries --log-level warn \
  --request-timeout-secs "$ROUTER_REQUEST_TIMEOUT_SECS" --worker-startup-timeout-secs 7200
}

wait_models() {
  local endpoint i
  for endpoint in "$P0_URL" "$P1_URL" "$D1_URL"; do
    for ((i=0;i<360;i++)); do
      if curl -fSs --max-time 2 "$endpoint/health" >/dev/null 2>&1; then break; fi
      sleep 5
    done
    (( i < 360 )) || { echo "模型未就绪：$endpoint" >&2; return 1; }
  done
}
case "${1:-}" in
  master) start_master ;;
  models)
    [[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" == 0 ]] || { echo 'GPU 已占用' >&2; exit 1; }
    prepare_model_files
    if [[ $NODE_ROLE == p ]]; then start_model p0; start_model p1; else start_model d0; fi
    ;;
  wait) wait_models ;;
  store) wait_models; start_store ;;
  router) [[ $NODE_ROLE == p ]]; wait_models; start_router ;;
  *) echo 'launch.sh master|models|wait|store|router' >&2; exit 2 ;;
esac
