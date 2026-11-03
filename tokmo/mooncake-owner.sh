#!/usr/bin/env bash
# Image: xinhao-vllm-roce-megamoe:4c711ac-20261103 (sha256:19177b08b707...)
# 一条命令启动一个 L2 DRAM＋L3 SSD owner。每台分别运行实例 0/1/2/3。
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "${1:?节点号}"
INSTANCE=${2:?owner 实例号 0|1|2|3}
[[ $# == 2 && "$INSTANCE" =~ ^[0-3]$ ]] || exit 2
NUMA=$((INSTANCE / 2))
L2_BYTES=$((${STORE_L2_GIB[$INSTANCE]} * 1024 * 1024 * 1024))
L3_BYTES=$((${STORE_L3_GIB[$INSTANCE]} * 1024 * 1024 * 1024))
MEMORY_GIB=$((${STORE_L2_GIB[$INSTANCE]} + 20))
SSD="${STORE_FILESYSTEMS[$INSTANCE]}/bench/eng-xinhao/mooncake-l3/xinhao-roce4c711-20261103/owner-$INSTANCE"

docker run -d \
  --name "xinhao-roce4c711-store-$NODE-$INSTANCE" \
  --pull=never --network host --restart unless-stopped \
  --memory "${MEMORY_GIB}g" --cpuset-mems "$NUMA" \
  --device /dev/infiniband \
  --cap-add IPC_LOCK --cap-add SYS_NICE \
  --security-opt seccomp=unconfined --ulimit memlock=-1 \
  --log-opt max-size=128m --log-opt max-file=5 \
  --mount "type=bind,source=$SSD,target=/ssd" \
  -v /usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro \
  -v /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -e MC_GID_INDEX="$IB_GID_INDEX" -e MC_TE_FILTERS="${STORE_HCAS[$INSTANCE]}" \
  -e MC_MAX_MR_SIZE=34359738368 \
  -e MC_ENABLE_DEST_DEVICE_AFFINITY=1 \
  -e MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/ssd \
  -e MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend \
  -e MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru \
  -e MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE="$L3_BYTES" \
  -e MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES="$L3_BYTES" \
  -e MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES="$L3_BYTES" \
  -e MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=8589934592 \
  -e MOONCAKE_OFFLOAD_ENABLE_DISK_WATERMARK_EVICTION=true \
  -e MOONCAKE_OFFLOAD_DISK_EVICTION_HIGH_WATERMARK_RATIO=0.95 \
  -e MOONCAKE_OFFLOAD_DISK_EVICTION_LOW_WATERMARK_RATIO=0.90 \
  -e MOONCAKE_OFFLOAD_USE_URING=true \
  --entrypoint /usr/bin/numactl \
  xinhao-vllm-roce-megamoe:4c711ac-20261103 \
  --cpunodebind="$NUMA" --membind="$NUMA" -- mooncake_client \
  --host="${STORE_IPS[$INSTANCE]}:$((50140 + INSTANCE))" --port="$((50080 + INSTANCE))" \
  --master_server_address="$P0_IP:$MOONCAKE_MASTER_PORT" \
  --metadata_server=P2PHANDSHAKE --protocol=rdma --device_names="${STORE_HCAS[$INSTANCE]}" \
  --global_segment_size="$L2_BYTES" --local_buffer_size=0 \
  --enable_offload=true --start_offload_rpc_server=true \
  --enable_http_server=true --http_port="$((9320 + INSTANCE))"
