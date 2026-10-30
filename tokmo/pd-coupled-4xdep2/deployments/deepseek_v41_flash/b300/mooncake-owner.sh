#!/usr/bin/env bash
# Eight independent owners: 200GiB L2 + 256GiB L3 each; 1600GiB + 2TiB total.
set -euo pipefail

OWNER=${1:?usage: bash mooncake-owner.sh 0..7}
[[ "$OWNER" =~ ^[0-7]$ ]] || { echo "Owner must be 0..7" >&2; exit 1; }
IMAGE=${IMAGE:-local/vllm-openai:0.32.0-dsv41-native-token-cache-v1}
NAME="vllm-dsv41-b300-pd-coupled-4xdep2-mooncake-owner-$OWNER"
NUMA=$((OWNER / 2))
L3_ROOT=${MOONCAKE_L3_ROOT:-/data1/lab/eng-xinhao/mooncake-l3/b300-pd-coupled-4xdep2-l2-200g-l3-2t-v1}
DIRECTORY="$L3_ROOT/owner-$OWNER"
L2_BYTES=214748364800
L3_BYTES=274877906944
STAGING_BYTES=1342177280

if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "Container exists: $NAME; inspect it before starting another owner." >&2
  exit 1
fi
mkdir -p "$DIRECTORY"
docker run -d --name "$NAME" --pull=never --network host \
  --restart unless-stopped \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=none \
  --device /dev/infiniband --cap-add IPC_LOCK --cap-add SYS_NICE \
  --security-opt seccomp=unconfined --ulimit memlock=-1 \
  --cpuset-cpus "$(cat /sys/devices/system/node/node$NUMA/cpulist)" --cpuset-mems "$NUMA" \
  --memory 224g --log-opt max-size=128m --log-opt max-file=5 \
  -v "$DIRECTORY:/ssd" \
  -e MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/ssd \
  -e MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend \
  -e MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru \
  -e MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE="$L3_BYTES" \
  -e MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES="$L3_BYTES" \
  -e MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES="$L3_BYTES" \
  -e MOONCAKE_OFFLOAD_BUCKET_DISK_SCAN_CACHE_MS=500 \
  -e MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES="$STAGING_BYTES" \
  -e MOONCAKE_OFFLOAD_ENABLE_DISK_WATERMARK_EVICTION=true \
  -e MOONCAKE_OFFLOAD_DISK_EVICTION_HIGH_WATERMARK_RATIO=0.9 \
  -e MOONCAKE_OFFLOAD_DISK_EVICTION_LOW_WATERMARK_RATIO=0.8 \
  -e MOONCAKE_OFFLOAD_USE_URING=true \
  --entrypoint mooncake_client "$IMAGE" \
  --host="11.5.0.55:$((50140 + OWNER))" --port="$((50080 + OWNER))" \
  --master_server_address=127.0.0.1:50066 --metadata_server=P2PHANDSHAKE \
  --protocol=rdma --device_names=mlx5_6 --global_segment_size="$L2_BYTES" \
  --local_buffer_size=0 --enable_offload=true --start_offload_rpc_server=true \
  --enable_http_server=true --http_port="$((9320 + OWNER))"
