#!/usr/bin/env bash
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
SOURCE_COMMIT=$(git rev-parse "${SOURCE_COMMIT:-HEAD}^{commit}")
IMAGE_TAG=${IMAGE_TAG:-vllm/vllm-openai:kimi-k31-yyf}
BASE=vllm/vllm-openai:kimi-k31
EXPECTED=sha256:1a8a702f2bfeded3c90259ae60fefbd9e4ca014241946e7d5f3396badce99e76
test "$(docker image inspect "$BASE" --format '{{.Id}}')" = "$EXPECTED"
CONTEXT=$(mktemp -d)
trap 'rm -rf "$CONTEXT"' EXIT
mkdir -p "$CONTEXT/vllm/renderers" "$CONTEXT/vllm/tokenizers"
for file in vllm/renderers/kimi_k3.py vllm/tokenizers/kimi_k3_cache.py \
  vllm/tokenizers/kimi_k3_cache_shared.py; do
  git -C "$ROOT" show "$SOURCE_COMMIT:$file" > "$CONTEXT/$file"
done
git -C "$ROOT" show "$SOURCE_COMMIT:docker/kimi31-b300-xinhao/Dockerfile.runtime" \
  > "$CONTEXT/Dockerfile"
docker build --pull=false --build-arg VLLM_BUILD_COMMIT="$SOURCE_COMMIT" \
  -t "$IMAGE_TAG" "$CONTEXT"
