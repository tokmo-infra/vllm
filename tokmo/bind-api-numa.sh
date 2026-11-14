#!/usr/bin/env bash
set -euo pipefail
CONTAINER=${1:?container name required}
HEALTH_URL=${2:?health URL required}
DEADLINE=$((SECONDS + 7200))
until curl -fsS --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; do
  (( SECONDS < DEADLINE )) || { echo 'API readiness timeout' >&2; exit 1; }
  [[ $(docker inspect --format '{{.State.Running}}' "$CONTAINER") == true ]] || exit 1
  sleep 1
done
RUST_PID=$(docker top "$CONTAINER" -eo pid,comm | awk '$2 == "vllm-rs" {print $1}')
[[ "$RUST_PID" =~ ^[0-9]+$ ]] || { echo 'Expected exactly one Rust frontend' >&2; exit 1; }
BUS=$(nvidia-smi -i 0 --query-gpu=pci.bus_id --format=csv,noheader | tr '[:upper:]' '[:lower:]')
BUS=${BUS/00000000:/0000:}
NUMA_NODE=$(cat "/sys/bus/pci/devices/$BUS/numa_node")
[[ "$NUMA_NODE" =~ ^[0-9]+$ ]] || exit 1
CPUS=$(cat "/sys/devices/system/node/node$NUMA_NODE/cpulist")
# Change only the frontend's threads. Future threads inherit this affinity.
sudo -n taskset -apc "$CPUS" "$RUST_PID"
for STATUS in /proc/"$RUST_PID"/task/*/status; do
  ACTUAL=$(awk '/^Cpus_allowed_list:/ {print $2}' "$STATUS")
  [[ "$ACTUAL" == "$CPUS" ]] || { echo "Affinity mismatch: $STATUS $ACTUAL" >&2; exit 1; }
done
printf 'container=%s rust_host_pid=%s gpu0_bdf=%s numa_node=%s cpus=%s verified=true\n' "$CONTAINER" "$RUST_PID" "$BUS" "$NUMA_NODE" "$CPUS"
