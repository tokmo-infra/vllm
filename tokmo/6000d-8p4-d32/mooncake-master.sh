#!/usr/bin/env bash
# Mooncake 元数据 Master：仅在 6000d-81 启动。
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

NAME=glm53flash-8p4-d32-xinhao-master-81
IMAGE=sha256:bd8a5d7a8774800c92d714c0c226b32c1863f929150f36efd610aebcd12a335d

command=(
  docker run -d --init
  --name "$NAME"
  --pull=never
  --network host
  --log-opt max-size=128m
  --log-opt max-file=5
  --label ai-infra.project=glm53flash-8p4-d32-xinhao
  --memory 8g

  # 进程环境
  -e NO_PROXY='*'
  -e no_proxy='*'
  -e HF_HUB_OFFLINE=1
  -e VLLM_NO_USAGE_STATS=1
  -e NVIDIA_VISIBLE_DEVICES=void

  --entrypoint mooncake_master
  "$IMAGE"
  --rpc_address=11.5.0.81
  --port=54365
  --metrics_port=29465
  # 显式 preferred_segment 优先，然后本主机其他段，再全局回退。
  --allocation_strategy=local_first
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
