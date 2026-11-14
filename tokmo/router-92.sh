#!/usr/bin/env bash
# Verified Docker command snapshot: B300 HiSparse PD, 2026-11-14.
set -euo pipefail

command=(
  docker run -d
  --name kimi31-pd-hisparse-92-router-full-c256-r14
  --pull=never
  --runtime=runc
  --network host
  --restart no
  --label ai-infra.plan=kimi31-b300-92-93-p8-d8-tp8-dcp8-ep8
  --label ai-infra.node=92
  --label ai-infra.role=router
  --log-driver local
  --log-opt max-size=20m
  --log-opt max-file=3
  --memory 384g
  --cpus 8
  --ulimit nofile=1048576:1048576
  -e NVIDIA_VISIBLE_DEVICES=void
  -e 'NO_PROXY=*'
  -e 'no_proxy=*'
  -e VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP=1
  -e VLLM_ROUTER_PD_REUSE_PROMPT_IDS=1
  --entrypoint /usr/local/bin/vllm-router
  sha256:55b0495c0784ed15003156818c7a911dde678d3352ab791418df12de025b7981
  --host 0.0.0.0
  --port 19500
  --vllm-pd-disaggregation
  --kv-connector mooncake
  --prefill http://11.5.0.92:19580
  29600
  --decode http://11.5.0.93:19581
  --prefill-policy cache_aware
  --decode-policy round_robin
  --max-tree-size 67108864
  --max-concurrent-requests 1024
  --disable-retries
  --request-timeout-secs 7200
  --worker-startup-timeout-secs 7200
  --prometheus-host 127.0.0.1
  --prometheus-port 20505
  --log-level warn
)

case "${1:---dry-run}" in
  --dry-run)
    printf '%q ' "${command[@]}"
    printf '\n'
    ;;
  --execute)
    ip -o -4 addr show | grep -F '11.5.0.92/' >/dev/null || { echo 'Run on b300-92' >&2; exit 1; }
    if docker inspect kimi31-pd-hisparse-92-router-full-c256-r14 >/dev/null 2>&1; then
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
