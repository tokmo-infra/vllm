#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
base_image=${BASE_IMAGE:-local/glm53flash-nvfp4kv:compat-api-v7-20261124}
engine_image=${ENGINE_IMAGE:-local/glm53flash-dflash2:r15b-rust-pd-85-86}

die() { printf 'build-dflash: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: build-dflash.sh [--source-root DIR]
By default, use the checkout or source archive containing integrations/glm53flash/1p1d.
BASE_IMAGE overrides the local NVFP4 KV base tag. ENGINE_IMAGE sets the output tag.
EOF
}

find_source_root() {
  local dir=$here
  while [[ $dir != / ]]; do
    case $dir in
      */integrations/glm53flash/1p1d)
        # This path is identical in a Git checkout and a git archive.
        printf '%s\n' "${dir%/integrations/glm53flash/1p1d}"
        return
        ;;
    esac
    dir=${dir%/*}
    dir=${dir:-/}
  done
  return 1
}

case $# in
  0) root=$(find_source_root) || die 'use the script under integrations/glm53flash/1p1d or pass --source-root DIR' ;;
  1) case $1 in -h|--help) usage; exit 0 ;; *) usage >&2; exit 2 ;; esac ;;
  2) [[ $1 == --source-root ]] || { usage >&2; exit 2; }
     [[ -n $2 && -d $2 ]] || die "source root is not a directory: $2"
     root=$(cd -- "$2" && pwd -P) ;;
  *) usage >&2; exit 2 ;;
esac

for file in rust/Cargo.toml rust/Cargo.lock rust/src/cmd/Cargo.toml rust/vendor/llm-multimodal/Cargo.toml; do
  [[ -f $root/$file ]] || die "missing $root/$file"
done

# 84's r15b engine files, plus the 85/86 default model state override.
runtime_files=(
  vllm/config/model.py
  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py
  vllm/entrypoints/cli/serve.py
  vllm/model_executor/models/qwen3_dflash.py
  vllm/utils/numa_wrapper.sh
  vllm/v1/core/kv_cache_coordinator.py
  vllm/v1/core/kv_cache_utils.py
  vllm/v1/core/sched/scheduler.py
  vllm/v1/kv_cache_interface.py
  vllm/v1/utils.py
  vllm/v1/worker/gpu/model_states/default.py
  vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
  vllm/v1/worker/gpu/spec_decode/dflash/utils.py
)
for file in "${runtime_files[@]}"; do
  [[ -f $root/$file ]] || die "missing runtime file: $root/$file"
done

[[ $engine_image != "$base_image" ]] || die 'the output tag must differ from the base tag'
command -v docker >/dev/null || die 'docker is required'
base_platform=$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$base_image") ||
  die "local base $base_image is missing; build it with the archived build-source/build-all.sh source"
[[ $base_platform == linux/amd64 ]] || die "base $base_image is $base_platform; expected linux/amd64"

context=$(mktemp -d "${TMPDIR:-/tmp}/glm53-dflash.XXXXXXXX")
trap 'rm -rf -- "$context"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

COPYFILE_DISABLE=1 tar --exclude=target --exclude=.git --exclude=.DS_Store -C "$root" -cf - rust |
  COPYFILE_DISABLE=1 tar -C "$context" -xf -
for file in "${runtime_files[@]}"; do
  dest=$context/runtime/$file
  mkdir -p -- "${dest%/*}"
  mode=0644
  [[ $file != *.sh ]] || mode=0755
  install -m "$mode" "$root/$file" "$dest"
done

docker build --platform linux/amd64 --pull=false -f "$here/Dockerfile" \
  --build-arg "BASE_IMAGE=$base_image" -t "$engine_image" "$context"
