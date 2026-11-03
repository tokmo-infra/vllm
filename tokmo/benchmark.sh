#!/usr/bin/env bash
# Client image: local/dsv41-pd-bench:0.8.0-sharded-summary-20261101
# 89：只启动 CPU 压测客户端。C7480 / A12000，复用原来的封存缓存。
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" 89
[[ $(docker image inspect "$CLIENT_IMAGE" --format '{{.Id}}') == "$CLIENT_IMAGE_ID" ]]
SEALED="$CLIENT_HOME/runtime/prepared-cache/$PREPARED_CACHE_KEY"
READER="$WORKDIR/runtime/prepared-reader"

python3 - "$SEALED" "$DATASET_SHA256" "$WORKDIR/benchmark.yaml" <<'PY'
import json, sys
from pathlib import Path
sealed = Path(sys.argv[1])
m = json.loads((sealed / 'manifest.json').read_text())
c = json.loads(Path(sys.argv[3]).read_text())
assert m['counts']['selected'] == 12000 and m['counts']['selected_turns'] == 1196074
assert m['identity']['source_sha256'] == sys.argv[2]
assert (sealed / 'payloads.bin').stat().st_size == m['payload_stat']['bytes']
assert (sealed / 'selected.arrow').is_file()
assert c['pipeline']['loader']['settings']['data_files'] == m['identity']['source']
assert c['seed'] == m['identity']['seed']
assert not Path(c['pipeline']['saver']['settings']['path']).exists(), '换一个新的结果路径，避免覆盖'
print('复用 CPU prepared cache: 12000 sessions / 1196074 turns')
PY
curl --noproxy '*' -fsS http://127.0.0.1:28080/health

# 只允许已存在的缓存 key。缓存身份变化会失败，不会重建约 556 GiB 正文。
mkdir -p "$READER" "$WORKDIR/results"
if [[ ! -L "$READER/$PREPARED_CACHE_KEY" ]]; then
  ln -s "$SEALED" "$READER/$PREPARED_CACHE_KEY"
fi
[[ $(readlink "$READER/$PREPARED_CACHE_KEY") == "$SEALED" ]]
touch "$READER/$PREPARED_CACHE_KEY.lock"
chmod 664 "$READER/$PREPARED_CACHE_KEY.lock"
chmod 555 "$READER"

docker run \
  --name xinhao-roce4c711-benchmark-dspark5-c7480-a12000-001 \
  --pull=never --network host --memory 64g --cpuset-mems 1 \
  --ulimit nofile=1048576:1048576 \
  --user "$(id -u):$(id -g)" --group-add 2004 \
  -e NVIDIA_VISIBLE_DEVICES=void -e NO_PROXY='*' -e no_proxy='*' \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH="$CLIENT_HOME/client-source:$CLIENT_HOME/runtime/client-observer-libs" \
  -e BENCH_RUN_DIR="$WORKDIR/results/dspark5-c7480-a12000-20261103-001" \
  -e AI_INFRA_ATTEMPT_DIAGNOSTICS="$WORKDIR/results/dspark5-c7480-a12000-20261103-001/attempt-diagnostics.jsonl" \
  -e AI_INFRA_SLOW_ATTEMPT_SECONDS=480 \
  -e PERF_BENCHMARK_PREPARED_CACHE="$READER" \
  -e HF_DATASETS_CACHE="$WORKDIR/runtime/datasets-cache" \
  -e XDG_CACHE_HOME="$WORKDIR/runtime/client-cache" \
  -v "$CLIENT_HOME:$CLIENT_HOME:ro" \
  -v "$WORKDIR:$WORKDIR" -w "$WORKDIR" \
  --entrypoint /opt/client-venv/bin/python local/dsv41-pd-bench:0.8.0-sharded-summary-20261101 \
  "$WORKDIR/validation/run_benchmark_with_metrics.py" run "$WORKDIR/benchmark.yaml"
