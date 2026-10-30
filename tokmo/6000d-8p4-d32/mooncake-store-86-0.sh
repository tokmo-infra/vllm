#!/usr/bin/env bash
# Mooncake CPU Store：6000d-86，NUMA0 本地 256 GiB CPU 池。
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

NAME=glm53flash-8p4-d32-xinhao-store-86-0
IMAGE=sha256:bd8a5d7a8774800c92d714c0c226b32c1863f929150f36efd610aebcd12a335d

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
  --memory 280g
  --cpuset-cpus "0-31,64-95"
  --cpuset-mems 0
  --cap-add SYS_NICE
  --security-opt seccomp=unconfined

  # 进程环境
  -e NO_PROXY='*'
  -e no_proxy='*'
  -e HF_HUB_OFFLINE=1
  -e VLLM_NO_USAGE_STATS=1
  -e NVIDIA_VISIBLE_DEVICES=void
  -e MC_TCP_BIND_ADDRESS=11.5.0.86
  -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME=11.5.0.86
  -e WITH_NVIDIA_PEERMEM=0
  -e "MC_TE_FILTERS=mlx5_1,mlx5_3"
  -e LD_LIBRARY_PATH=/opt/host-driver:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib
  # CPU Store 仅挂载驱动库，不暴露 GPU 设备。

  # 本服务的挂载
  -v /usr/lib/x86_64-linux-gnu/libcuda.so.1:/opt/host-driver/libcuda.so.1:ro

  --entrypoint numactl
  "$IMAGE"
  --cpunodebind=0 --membind=0 mooncake_client
  # 固定 Transfer Engine / segment 地址；下方 --port 是独立的客户端 RPC。
  --host=11.5.0.86:15465
  --port=54465
  --metadata_server=P2PHANDSHAKE
  --master_server_address=11.5.0.81:54365
  --protocol=rdma
  "--device_names=mlx5_1,mlx5_3"
  --global_segment_size=274877906944
  --local_buffer_size=4294967296
  --enable_offload=false
  --enable_http_server=true
  --http_port=29466
  --logtostderr=true
)

case "$MODE" in
  --print-argv)
    printf '%s\0' "${command[@]}"
    exit 0 ;;
  --print-config)
    printf '{}\n'
    exit 0 ;;
  --dry-run)
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

exec "${command[@]}"
