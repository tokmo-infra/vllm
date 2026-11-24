#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

here=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
revision=0966e4b11b9006e2c5e48fd41d865032ced1e1b2
subtree=integrations/glm53flash/1p2d

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 IMAGE [docker buildx options...] | --prepare DIRECTORY" >&2
    exit 2
fi
if ! git -C "$repo" cat-file -e "$revision^{commit}" 2>/dev/null; then
    git -C "$repo" fetch --no-tags --depth=1 \
        https://github.com/tokmo-infra/vllm.git "$revision"
fi

prepare() {
    local dest=$1
    git -C "$repo" archive "$revision:$subtree/router" | tar -xf - -C "$dest"
    git -C "$repo" show "$revision:$subtree/build/Dockerfile.router" > "$dest/Dockerfile.api"
    git -C "$repo" show "$revision:$subtree/build/router-production.sha256" > "$dest/router.sha256"
    if command -v sha256sum >/dev/null; then
        (cd "$dest" && sha256sum -c router.sha256)
    else
        (cd "$dest" && shasum -a 256 -c router.sha256)
    fi
}

if [[ $1 == --prepare ]]; then
    [[ $# == 2 ]] || exit 2
    mkdir -p -- "$2"
    [[ -z $(ls -A -- "$2") ]] || { echo 'Context must be empty' >&2; exit 1; }
    prepare "$2"
    exit 0
fi

image=$1
shift
context=$(mktemp -d)
trap 'rm -rf -- "$context"' EXIT
prepare "$context"
docker buildx build --platform linux/amd64 --load --tag "$image" \
    --label "org.opencontainers.image.revision=$revision" \
    --build-context "router_src=$context" --file "$context/Dockerfile.api" "$@" "$context"
