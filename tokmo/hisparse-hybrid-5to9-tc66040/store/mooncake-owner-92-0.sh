#!/usr/bin/env bash
# Verified Docker command snapshot: B300 HiSparse PD, 2026-11-14.
set -euo pipefail

command=(
  docker run -d
  --name kimi31-hisparse-store-r15-owner-92-0
  --pull=never
  --runtime=runc
  --network host
  --restart no
  --memory 528g
  --memory-swap 528g
  --cpuset-mems 0
  --cpuset-cpus '32-39,160-167'
  --device /dev/infiniband
  --cap-add IPC_LOCK
  --cap-add SYS_NICE
  --security-opt seccomp=unconfined
  --ulimit memlock=-1:-1
  --ulimit nofile=1048576:1048576
  --log-driver local
  --log-opt max-size=20m
  --log-opt max-file=3
  --mount 'type=bind,src=/data/bench/eng-xinhao/kimi31-hisparse-store-r15/owner-0,dst=/ssd'
  --mount 'type=bind,src=/usr/lib/x86_64-linux-gnu/libcuda.so.1,dst=/usr/lib/x86_64-linux-gnu/libcuda.so.1,readonly'
  --mount 'type=bind,src=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1,dst=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1,readonly'
  -e NVIDIA_VISIBLE_DEVICES=void
  -e 'NO_PROXY=*'
  -e 'no_proxy=*'
  -e MC_GID_INDEX=3
  -e MC_ENABLE_DEST_DEVICE_AFFINITY=1
  -e 'MC_TE_FILTERS=mlx5_0,mlx5_6'
  -e MC_MAX_MR_SIZE=4294967296
  -e MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/ssd
  -e MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend
  -e MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
  -e MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=1425929142272
  -e MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES=1425929142272
  -e MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=1425929142272
  -e MOONCAKE_OFFLOAD_BUCKET_DISK_SCAN_CACHE_MS=500
  -e MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824
  -e MOONCAKE_OFFLOAD_ENABLE_DISK_WATERMARK_EVICTION=true
  -e MOONCAKE_OFFLOAD_DISK_EVICTION_HIGH_WATERMARK_RATIO=0.95
  -e MOONCAKE_OFFLOAD_DISK_EVICTION_LOW_WATERMARK_RATIO=0.90
  -e MOONCAKE_OFFLOAD_USE_URING=true
  --entrypoint mooncake_client
  sha256:a2071779007d3e9d17c59afc4555eb0afa471f437858cd849026122e81add9fa
  --host=11.14.0.92:62040
  --port=62140
  --master_server_address=11.5.0.92:51561
  --metadata_server=P2PHANDSHAKE
  --protocol=rdma
  '--device_names=mlx5_0,mlx5_6'
  --global_segment_size=532575944704
  --local_buffer_size=0
  --enable_offload=true
  --start_offload_rpc_server=true
  --enable_http_server=true
  --http_port=62240
)

case "${1:---dry-run}" in
  --dry-run)
    printf '%q ' "${command[@]}"
    printf '\n'
    ;;
  --execute)
    ip -o -4 addr show | grep -F '11.5.0.92/' >/dev/null || { echo 'Run on b300-92' >&2; exit 1; }
    if docker inspect kimi31-hisparse-store-r15-owner-92-0 >/dev/null 2>&1; then
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
