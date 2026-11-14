#!/usr/bin/env bash
# Verified Docker command snapshot: B300 HiSparse PD, 2026-11-14.
set -euo pipefail

command=(
  docker run -d
  --name kimi31-hisparse-store-r15-owner-93-2
  --pull=never
  --runtime=runc
  --network host
  --restart no
  --memory 336g
  --memory-swap 336g
  --cpuset-mems 1
  --cpuset-cpus '96-103,224-231'
  --device /dev/infiniband
  --cap-add IPC_LOCK
  --cap-add SYS_NICE
  --security-opt seccomp=unconfined
  --ulimit memlock=-1:-1
  --ulimit nofile=1048576:1048576
  --log-driver local
  --log-opt max-size=20m
  --log-opt max-file=3
  --mount 'type=bind,src=/data3/bench/eng-xinhao/kimi31-hisparse-store-r15/owner-2,dst=/ssd'
  --mount 'type=bind,src=/usr/lib/x86_64-linux-gnu/libcuda.so.1,dst=/usr/lib/x86_64-linux-gnu/libcuda.so.1,readonly'
  --mount 'type=bind,src=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1,dst=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1,readonly'
  -e NVIDIA_VISIBLE_DEVICES=void
  -e 'NO_PROXY=*'
  -e 'no_proxy=*'
  -e MC_GID_INDEX=3
  -e MC_ENABLE_DEST_DEVICE_AFFINITY=1
  -e 'MC_TE_FILTERS=mlx5_12,mlx5_14'
  -e MC_MAX_MR_SIZE=4294967296
  -e MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/ssd
  -e MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend
  -e MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
  -e MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=3418793967616
  -e MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES=3418793967616
  -e MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=3418793967616
  -e MOONCAKE_OFFLOAD_BUCKET_DISK_SCAN_CACHE_MS=500
  -e MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824
  -e MOONCAKE_OFFLOAD_ENABLE_DISK_WATERMARK_EVICTION=true
  -e MOONCAKE_OFFLOAD_DISK_EVICTION_HIGH_WATERMARK_RATIO=0.95
  -e MOONCAKE_OFFLOAD_DISK_EVICTION_LOW_WATERMARK_RATIO=0.90
  -e MOONCAKE_OFFLOAD_USE_URING=true
  --entrypoint mooncake_client
  sha256:27ed3f34dbe3f231482eff23b1cdc6e4e2b624bfa4b10d1c5205194d8dd589b1
  --host=11.1.0.93:62042
  --port=62142
  --master_server_address=11.5.0.92:51561
  --metadata_server=P2PHANDSHAKE
  --protocol=rdma
  '--device_names=mlx5_12,mlx5_14'
  --global_segment_size=326417514496
  --local_buffer_size=0
  --enable_offload=true
  --start_offload_rpc_server=true
  --enable_http_server=true
  --http_port=62242
)

case "${1:---dry-run}" in
  --dry-run)
    printf '%q ' "${command[@]}"
    printf '\n'
    ;;
  --execute)
    ip -o -4 addr show | grep -F '11.5.0.93/' >/dev/null || { echo 'Run on b300-93' >&2; exit 1; }
    if docker inspect kimi31-hisparse-store-r15-owner-93-2 >/dev/null 2>&1; then
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
