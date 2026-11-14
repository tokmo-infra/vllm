#!/usr/bin/env bash
# Verified Docker command snapshot: B300 HiSparse PD, 2026-11-14.
set -euo pipefail

command=(
  docker run -d
  --name kimi31-hisparse-store-r15-master-92
  --pull=never
  --runtime=runc
  --network host
  --restart no
  --label ai-infra.plan=kimi31-b300-92-93-p8-d8-tp8-dcp8-ep8
  --label ai-infra.node=92
  --label ai-infra.role=master
  --log-driver local
  --log-opt max-size=20m
  --log-opt max-file=3
  --memory 32g
  --cpuset-mems 0
  --cpuset-cpus '48-55,176-183'
  --memory-swap 32g
  --entrypoint mooncake_master
  sha256:5a259407d16d003765361f98e9178c83915dc615287f37d525efdb4012642f21
  --rpc_address=11.5.0.92
  --rpc_port=51561
  --metrics_host=127.0.0.1
  --metrics_port=20503
  --enable_http_metadata_server=true
  --http_metadata_server_host=11.5.0.92
  --http_metadata_server_port=19590
  --default_kv_lease_ttl=60000
  --eviction_high_watermark_ratio=0.95
  --eviction_ratio=0.10
  --enable_offload=true
  --offload_on_evict=true
  --offload_force_evict=false
  --promotion_on_hit=true
  --promotion_admission_threshold=2
  --allocation_strategy=free_ratio_first
)

case "${1:---dry-run}" in
  --dry-run)
    printf '%q ' "${command[@]}"
    printf '\n'
    ;;
  --execute)
    ip -o -4 addr show | grep -F '11.5.0.92/' >/dev/null || { echo 'Run on b300-92' >&2; exit 1; }
    if docker inspect kimi31-hisparse-store-r15-master-92 >/dev/null 2>&1; then
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
