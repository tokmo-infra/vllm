#!/usr/bin/env bash
set -euo pipefail
umask 077
phase=${1:?master|model|store|router|health}
NODE_ID=${2:?82|83|84}
SERVICE_PREFIX=${SERVICE_PREFIX:-glm53flash-1p2d-c220-r1}
KEY_FILE=${API_KEY_FILE:-/data/services/$SERVICE_PREFIX/api.key}
case "$NODE_ID" in 82|83|84) ;; *) echo '节点编号为 82、83 或 84' >&2; exit 2;; esac
case "$phase" in master|store|model|router|health) ;; *) exit 2;; esac
[[ $(id -u) == 0 ]]
if [[ "$phase" == model || "$phase" == router || "$phase" == health ]]; then
  test ! -L "$KEY_FILE"; test "$(stat -c '%u:%g:%a' "$KEY_FILE")" = '0:0:600'
  API_KEY=$(cat -- "$KEY_FILE")
  test -n "$API_KEY"
fi
export P_IMAGE="${ENGINE_IMAGE:-local/glm53flash-nvfp4kv:compat-78159a469b9a}" D_IMAGE="${ENGINE_IMAGE:-local/glm53flash-nvfp4kv:compat-78159a469b9a}"
export MASTER_IMAGE="${MOONCAKE_IMAGE:-local/glm53flash-mooncake:source-71973589}"
export STORE_IMAGE="$MASTER_IMAGE"
export ROUTER_IMAGE="${ROUTER_IMAGE:-router:glm53flash-responses-tools-ecee97b77d8e}"
export NODE_ID
export NODE_IP="${NODE_IP:-11.8.0.$NODE_ID}"
export MASTER_IP="${MASTER_IP:-11.8.0.83}"
export ROUTER_IP="${ROUTER_IP:-11.8.0.82}"
export P_IP="${P_IP:-11.8.0.82}"
export D1_IP="${D1_IP:-11.8.0.83}" D2_IP="${D2_IP:-11.8.0.84}"
export P_HOSTS="$P_IP"
export D_HOSTS="$D1_IP $D2_IP"
case "$NODE_ID" in
  82) expected_ip=$P_IP ;;
  83) expected_ip=$D1_IP ;;
  84) expected_ip=$D2_IP ;;
esac
export MODEL_DIR="${MODEL_DIR:-/data/models/GLM-5.3-Flash-NVFP4}"
export MODEL_NAME="${MODEL_NAME:-glm-5.3-flash}"
export SERVICE_PREFIX
export BASE_WORK="${BASE_WORK:-/data/services/$SERVICE_PREFIX}"
export CACHE_NAMESPACE="${CACHE_NAMESPACE:-$SERVICE_PREFIX-nvfp4-prefix256}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-/dev/shm/$SERVICE_PREFIX/tokenizer}"
export GID_INDEX="${GID_INDEX:-3}"
export NET_IF="${NET_IF:-enp115s0f1np1}"
if [[ "$NODE_IP" != "$expected_ip" ]] || ! ip -o -4 addr show dev "$NET_IF" |
  awk -v ip="$NODE_IP" '{split($4,a,"/"); if (a[1]==ip) ok=1} END {exit !ok}'; then
  echo "节点 $NODE_ID 的角色地址为 ${expected_ip}；请核对 NODE_IP=$NODE_IP 和本机网口 ${NET_IF}。" >&2
  exit 1
fi
export HCAS_ALL="${HCAS_ALL:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export HCAS_P0="${HCAS_P0:-mlx5_0,mlx5_1,mlx5_2,mlx5_3}"
export HCAS_P1="${HCAS_P1:-mlx5_4,mlx5_5,mlx5_6,mlx5_7}"

if [[ "$phase" == model ]]; then
  test -f "$MODEL_DIR/tokenizer.json"
  # epoch 改变时，现有 API 进程清理对应的同机分词池和私有 L1。
  install -d -m 0700 "$TOKENIZER_DIR"
  if [[ ! -e "$TOKENIZER_DIR/epoch" ]]; then
    (set -o noclobber; cat /proc/sys/kernel/random/uuid > "$TOKENIZER_DIR/epoch")
  fi
  test ! -L "$TOKENIZER_DIR/epoch"; test -s "$TOKENIZER_DIR/epoch"
  chown 0:0 "$TOKENIZER_DIR" "$TOKENIZER_DIR/epoch"
  chmod 0700 "$TOKENIZER_DIR"
  chmod 0600 "$TOKENIZER_DIR/epoch"
fi

write_mc_config() {
  cat > "$1" <<EOF
{
  "mode": "standalone-store",
  "metadata_server": "http://$MASTER_IP:61242/metadata",
  "master_server_address": "$MASTER_IP:61241",
  "protocol": "rdma",
  "device_name": "$2",
  "global_segment_size": 0,
  "local_buffer_size": 2147483648,
  "enable_offload": true,
  "tenant_id": "default"
}
EOF
}

l2_gib_for() {
  local val
  case "$1" in
    82) val="${L2_82_GIB:-700}" ;;
    83) val="${L2_83_GIB:-750}" ;;
    84) val="${L2_84_GIB:-750}" ;;
    *) return 2 ;;
  esac
  if [[ ! "$val" =~ ^[1-9][0-9]{0,2}$ ]] || (( val > 750 )); then
    echo 'L2 容量必须为 1–750 GiB 的整数' >&2; return 2;
  fi
  printf '%s\n' "$val"
}

case "$phase" in
master)
if [ "$NODE_IP" = "$MASTER_IP" ]; then
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
    --rpc_port=61241 --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 --http_metadata_server_port=61242 \
    --metrics_port=61243 --eviction_high_watermark_ratio=0.98 \
    --enable_offload=true --offload_on_evict=true \
    --allocation_strategy=local_first --logtostderr=true
fi

;;
store)
STORE_GIB=$(l2_gib_for "$NODE_ID")
STORE_BYTES=$((STORE_GIB * 1024 * 1024 * 1024))
# 每台 NVMe 按需使用 2.5 TB（十进制）。
L3_MAX_BYTES=2500000000000
L3_DIR="/data/l3/$SERVICE_PREFIX"
test "$(findmnt -n -T /data -o TARGET)" = /data
test ! -L /data/l3
mkdir -p /data/l3
test ! -L "$L3_DIR"
install -d -m 0700 "$L3_DIR"
test "$(realpath "$L3_DIR")" = "$L3_DIR"
test "$(findmnt -n -T "$L3_DIR" -o TARGET)" = /data
L3_EXISTING_BYTES=$(du -sxB1 -- "$L3_DIR" | awk '{print $1}')
L3_FREE_BYTES=$(df -B1 --output=avail "$L3_DIR" | tail -1 | tr -d '[:space:]')
(( L3_FREE_BYTES + L3_EXISTING_BYTES >= L3_MAX_BYTES )) || {
  echo 'Store 需要 2.5 TB L3 容量。' >&2; exit 1;
}
LIBCUDA_PATH=$(readlink -f /lib/x86_64-linux-gnu/libcuda.so.1)
test -f "$LIBCUDA_PATH"
docker run -d --pull=never --restart unless-stopped --name "$SERVICE_PREFIX-store$NODE_ID" \
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
  --master_server_address="$MASTER_IP:61241" \
  --metadata_server="http://$MASTER_IP:61242/metadata" \
  --protocol=rdma --host="$NODE_IP" --port=61244 \
  --device_names="$HCAS_ALL" --global_segment_size="$STORE_BYTES" \
  --local_buffer_size=0 --enable_http_server=true --http_port=61245 \
  --enable_offload=true --start_offload_rpc_server=true --logtostderr=true
curl -fsS --retry 20 --retry-connrefused --retry-delay 3 "http://$NODE_IP:61245/health"

;;
model)
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
start_model() {
  local slot="$1" image gpu_request hcas port bootstrap rpc dp apis
  local seq batch shared_gib cache_gib pd_role store_role role_work engine_id kv ib_hca cache_dir
  local retire_checkpoints=false gpu_util=0.95
  local -a role_env=() graph_args=()
  case "$slot" in
    p0|p1)
      image="$P_IMAGE"
      dp=4; apis=4; seq=24; batch=2048; shared_gib=16; cache_gib=2
      gpu_util="${P_GPU_UTIL:-0.95}"; pd_role=kv_producer; store_role=kv_both; retire_checkpoints=true
      role_env=(-e VLLM_GLM53_P_MTP_CONTEXT_ONLY=1 -e VLLM_GLM53_P_READY_WAIT=1
                -e VLLM_GLM53_NVFP4_EXTENDED_PREFETCH=1
                -e VLLM_MOONCAKE_LOAD_RECV_THREADS=4
                -e VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION=1
                -e VLLM_HOT_TAIL_ADMISSION_DIAGNOSTICS=0)
      if [ "$slot" = p0 ]; then
        gpu_request='"device=0,1,2,3"'; hcas="$HCAS_P0"
        port=18483; bootstrap=19083; rpc=19550
      else
        gpu_request='"device=4,5,6,7"'; hcas="$HCAS_P1"
        port=18493; bootstrap=19093; rpc=19650
      fi
      ;;
    d0)
      image="$D_IMAGE"
      dp=8; apis=1; seq=32; batch=1024; shared_gib=16; cache_gib=16
      role_env=(-e VLLM_GLM53_P_MTP_CONTEXT_ONLY=0
                -e VLLM_GLM53_NVFP4_EXTENDED_PREFETCH=0)
      pd_role=kv_consumer; store_role=kv_consumer
      gpu_request='"device=0,1,2,3,4,5,6,7"'; hcas="$HCAS_ALL"
      port=18483; bootstrap=19083; rpc=19550
      graph_args=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":128}')
      ;;
    *) return 1 ;;
  esac
  ib_hca="=$(printf '%s' "$hcas" | sed 's/,/:1,/g'):1"
  role_work="$BASE_WORK/$slot"
  install -d -m 0700 "$role_work"
  if ! cmp -s "$here/numa_wrapper.sh" "$BASE_WORK/numa_wrapper.sh"; then
    install -m 0755 "$here/numa_wrapper.sh" "$BASE_WORK/numa_wrapper.sh"
  fi
  cache_dir="/var/cache/glm53flash-1p2d/d50c0828c8dd62/$slot"
  install -d -m 0700 "$cache_dir"
  write_mc_config "$role_work/mooncake.json" "$hcas"
  engine_id="$SERVICE_PREFIX-$slot-$NODE_ID-$(cat /proc/sys/kernel/random/uuid)"
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
  docker run -d --pull=never --restart unless-stopped --name "$SERVICE_PREFIX-$slot-$NODE_ID" \
    --init --network host --shm-size 64g --memory 64g --memory-swap 64g \
    --gpus "$gpu_request" --device /dev/infiniband \
    --cap-add IPC_LOCK --cap-add SYS_NICE --ulimit memlock=-1:-1 \
    --ulimit stack=67108864:67108864 \
    -v "$MODEL_DIR:/model:ro" -v "$role_work:/work" \
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
    -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_SERVER_DEV_MODE=1 \
    -e VLLM_GLM53_NVFP4_FUSED_MOE=1 \
    -e VLLM_GLM53_NVFP4_STAGE_FC2_INPUT=1 \
    -e VLLM_GLM53_NVFP4_VECTOR_WEIGHTS=1 \
    -e MOONCAKE_CONFIG_PATH=/work/mooncake.json \
    -e VLLM_MOONCAKE_BOOTSTRAP_PORT="$bootstrap" -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 \
    ${role_env[@]+"${role_env[@]}"} \
    --entrypoint vllm "$image" serve /model \
    --served-model-name "$MODEL_NAME" --host "$NODE_IP" --port "$port" \
    --api-key "$API_KEY" --api-server-count "$apis" \
    --tensor-parallel-size 1 --data-parallel-size "$dp" --data-parallel-size-local "$dp" \
    --data-parallel-backend mp --data-parallel-rpc-port "$rpc" \
    --enable-expert-parallel --enable-ep-weight-filter \
    --max-model-len auto --max-num-seqs "$seq" --max-num-batched-tokens "$batch" \
    --gpu-memory-utilization "$gpu_util" --kv-cache-dtype nvfp4 \
    --block-size 64 --prefix-match-unit 256 --mamba-cache-mode align \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --kv-transfer-config "$kv" --scheduler-reserve-full-isl \
    --enable-prefix-caching --reasoning-parser glm47 \
    --enable-auto-tool-choice --tool-call-parser glm47 \
    --safetensors-load-strategy lazy --enable-cumem-allocator --seed 42 \
    --enable-prompt-tokens-details --enable-force-include-usage --numa-bind \
    ${graph_args[@]+"${graph_args[@]}"}
}

if [ "$NODE_IP" = "$P_IP" ]; then
  case "${P_WARM_SLOT:-both}" in
    p0) start_model p0;;
    p1) start_model p1;;
    both) start_model p0; start_model p1;;
    *) exit 2;;
  esac
else
  start_model d0
fi

;;
router)
if [ "$NODE_IP" = "$ROUTER_IP" ]; then
  read -r -a p_hosts <<< "$P_HOSTS"
  read -r -a d_hosts <<< "$D_HOSTS"
  test "${#p_hosts[@]}" -eq 1
  test "${#d_hosts[@]}" -eq 2
  routes=()
  for ip in "${p_hosts[@]}"; do
    routes+=(--prefill "http://$ip:18483" 19083 --prefill "http://$ip:18493" 19093)
  done
  for ip in "${d_hosts[@]}"; do
    routes+=(--decode "http://$ip:18483")
  done

  docker run -d --pull=never --name "$SERVICE_PREFIX-router" \
    --runtime=runc --network host --restart unless-stopped \
    --memory 32g --memory-swap 32g --cpuset-mems 0 \
    --ipc private --shm-size 64m --ulimit nofile=1048576:1048576 \
    --log-driver local --log-opt max-size=128m --log-opt max-file=5 \
    -e OPENAI_API_KEY="$API_KEY" -e NVIDIA_VISIBLE_DEVICES=void \
    -e NO_PROXY='*' -e no_proxy='*' \
    -e VLLM_ROUTER_PD_REUSE_PROMPT_IDS=1 \
    -e VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP=1 \
    --entrypoint /usr/local/bin/vllm-router "$ROUTER_IMAGE" \
    --host 0.0.0.0 --port 18300 \
    --prometheus-host 0.0.0.0 --prometheus-port 18301 \
    --vllm-pd-disaggregation --kv-connector mooncake \
    "${routes[@]}" \
    --prefill-policy cache_aware --decode-policy round_robin \
    --max-tree-size 2147483648 --eviction-interval 120 --max-concurrent-requests 4096 \
    --disable-retries --log-level warn \
    --request-timeout-secs 600 --worker-startup-timeout-secs 7200
fi

;;
health)
for ip in $P_HOSTS $D_HOSTS; do
  curl -fsS "http://$ip:61245/health"
done
l2_82=$(l2_gib_for 82)
l2_83=$(l2_gib_for 83)
l2_84=$(l2_gib_for 84)
expected_l2_bytes=$(( (l2_82 + l2_83 + l2_84) * 1024 * 1024 * 1024 ))
curl -fsS "http://$MASTER_IP:61243/metrics" |
  awk -v expected="$expected_l2_bytes" '$1 == "master_total_capacity_bytes" {print; ok=($2 == expected)} END {exit !ok}'

check_model() {
  local url="$1" dp="$2" seq="$3" batch="$4" graph="$5" graph_mode="${6:-}"
  curl -fsS -H "Authorization: Bearer $API_KEY" \
    "$url/server_info?config_format=json" |
    python3 -c '
import json, sys
url, dp, seq, batch, graph, mode = sys.argv[1:]
c = json.load(sys.stdin)["vllm_config"]
s, p, k, m, g = [c[x] for x in ("scheduler_config", "parallel_config", "cache_config", "speculative_config", "compilation_config")]
actual = (p["data_parallel_size"], s["max_num_seqs"], s["max_num_batched_tokens"], g["max_cudagraph_capture_size"])
expected = tuple(map(int, (dp, seq, batch, graph)))
serialized_mode = {"FULL_DECODE_ONLY": [2, 0]}.get(mode)
if not (actual == expected and (not mode or g["cudagraph_mode"] in (mode, serialized_mode))
        and s["async_scheduling"] and s["scheduler_reserve_full_isl"] and s["watermark"] == 0
        and p["tensor_parallel_size"] == 1 and p["all2all_backend"] == "allgather_reducescatter"
        and k["cache_dtype"] == "nvfp4" and k["prefix_match_unit"] == 256
        and m["num_speculative_tokens"] == 3):
    sys.exit(f"{url}: 参数不符，DP/seq/batch/graph={actual}，预期 {expected}")
print(f"{url}: NVFP4，DP/seq/batch/graph={actual}")' "$url" "$dp" "$seq" "$batch" "$graph" "$graph_mode"
}
for ip in $P_HOSTS; do
  check_model "http://$ip:18483" 4 24 2048 192
  check_model "http://$ip:18493" 4 24 2048 192
done
for ip in $D_HOSTS; do
  check_model "http://$ip:18483" 8 32 1024 128 FULL_DECODE_ONLY
done
curl -fsS "http://$ROUTER_IP:18300/workers" |
  python3 -c '
import json, sys
w = json.load(sys.stdin)["workers"]
healthy = sum(x["is_healthy"] is True for x in w)
print(f"Router: {healthy}/{len(w)} 健康")
sys.exit(0 if healthy == len(w) == 24 else 1)'

;;
esac
