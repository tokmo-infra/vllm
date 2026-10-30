#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
IMAGE=${IMAGE:-local/vllm-openai:0.32.0-dsv41-native-token-cache-v1}
BASE_IMAGE=${BASE_IMAGE:-vllm/vllm-openai:v0.32.0}
EXPECTED_BASE_IMAGE_ID=${EXPECTED_BASE_IMAGE_ID:-sha256:bd8a5d7a8774800c92d714c0c226b32c1863f929150f36efd610aebcd12a335d}

ACTUAL_BASE_IMAGE_ID=$(docker image inspect "$BASE_IMAGE" --format '{{.Id}}')
if [[ "$ACTUAL_BASE_IMAGE_ID" != "$EXPECTED_BASE_IMAGE_ID" ]]; then
    echo "Base image ID mismatch: $ACTUAL_BASE_IMAGE_ID" >&2
    exit 1
fi

docker build --pull=false --network none \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    --build-arg "SOURCE_REVISION=$(git -C "$REPO_ROOT" rev-parse HEAD)" \
    -t "$IMAGE" -f "$REPO_ROOT/tokmo/images/deepseek_v41_flash_b300/Dockerfile" \
    "$REPO_ROOT"
