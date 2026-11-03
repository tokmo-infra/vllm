#!/usr/bin/env bash
# 只读配置；各服务的 docker run 保留在各自脚本中。
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/images.env"
source "$SCRIPT_DIR/cluster.env"
NODE=${1:?需要节点号: 89|91|92|93}
case "$NODE" in 89|91|92|93) ;; *) echo "未知节点: $NODE" >&2; exit 2 ;; esac
# shellcheck source=/dev/null
source "$SCRIPT_DIR/nodes/$NODE.env"
# Arrays below are consumed by scripts sourcing common.sh.
# shellcheck disable=SC2034
STORE_HCAS=(mlx5_1 mlx5_9 mlx5_13 mlx5_19)
# shellcheck disable=SC2034
STORE_IPS=("11.13.0.$NODE" "11.15.0.$NODE" "11.2.0.$NODE" "11.4.0.$NODE")
