#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

usage() {
    echo "Usage: $0 FLASHINFER_WHEEL IMAGE [docker build options...]" >&2
    echo "       $0 --base-image FLASHINFER_WHEEL IMAGE [docker build options...]" >&2
    echo "       $0 --prepare FLASHINFER_WHEEL EMPTY_CONTEXT_DIR" >&2
    exit 2
}
prepare_only=0
from_base=0
if [[ ${1:-} == --prepare ]]; then
    prepare_only=1
    shift
elif [[ ${1:-} == --base-image ]]; then
    from_base=1
    shift
fi
[[ $# -ge 2 ]] || usage
patch_dir=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$patch_dir" rev-parse --show-toplevel)
base=e0d00e4b8eb0c89ec30921d0695cfdd409e9c211
nightly=vllm/vllm-openai@sha256:aeb580d55ee3dcccd7b5c78d7871326f367bee5cfd8ca8e411c253d7983a0046
if [[ $from_base == 1 && -n ${QWEN38_NIGHTLY_IMAGE:-} ]]; then
    expected=sha256:749210a8fa8cac5b13c9d7c8b5f04bc85ad91ffa5fac7381a10399e264c3495e
    [[ $(docker image inspect --format '{{.Id}}' "$QWEN38_NIGHTLY_IMAGE") == "$expected" ]] || {
        echo 'The cached base does not match the tested nightly image ID' >&2
        exit 1
    }
    nightly=$QWEN38_NIGHTLY_IMAGE
fi
wheel_name=flashinfer_python-0.7.5+1658f0bf-py3-none-any.whl
wheel_sha=2472cb319b28ebc3d81e946f25db761f3013d0138e2d74b32879a514eef3cdab
wheel=$1
output=$2
shift 2
if [[ ! -f $wheel || ${wheel##*/} != "$wheel_name" ]]; then
    echo "Expected the tested FlashInfer artifact: $wheel_name" >&2
    exit 1
fi
if command -v sha256sum >/dev/null; then
    checksum=(sha256sum)
else
    checksum=(shasum -a 256)
fi
hash=$("${checksum[@]}" < "$wheel")
[[ ${hash%% *} == "$wheel_sha" ]] || { echo 'FlashInfer wheel checksum mismatch' >&2; exit 1; }

if [[ $prepare_only == 1 ]]; then
    [[ $# == 0 ]] || usage
    mkdir -p -- "$output"
    ctx=$(cd -- "$output" && pwd)
    [[ -z $(ls -A -- "$ctx") ]] || { echo 'Context must be empty' >&2; exit 1; }
else
    ctx=$(mktemp -d)
fi

if ! git -C "$repo" cat-file -e "$base^{commit}" 2>/dev/null; then
    git -C "$repo" fetch --no-tags --depth=1 https://github.com/vllm-project/vllm.git "$base"
fi
git -C "$repo" archive "$base" | tar -xf - -C "$ctx"
git init -q "$ctx" # The upstream Dockerfile bind-mounts .git during the build.
mkdir -p "$ctx/qwen38-build/wheels" "$ctx/qwen38-build/prefill"
cp -- "$wheel" "$ctx/qwen38-build/wheels/$wheel_name"

(
    cd -- "$ctx"
    manifest() {
        sed -n 's|^+++ b/||p' "$patch_dir/common.patch" \
            "$patch_dir/api-protocol.patch" "$patch_dir/$1.patch" |
            LC_ALL=C sort -u | while IFS= read -r file; do
                "${checksum[@]}" "$file"
            done > "qwen38-build/$1.sha256"
    }
    git apply "$patch_dir/common.patch"
    git apply "$patch_dir/api-protocol.patch"
    git apply "$patch_dir/prefill.patch"
    manifest prefill
    sed -n 's|^+++ b/||p' "$patch_dir/api-protocol.patch" "$patch_dir/prefill.patch" |
        LC_ALL=C sort -u | while IFS= read -r file; do
        mkdir -p "qwen38-build/prefill/$(dirname -- "$file")"
        cp -- "$file" "qwen38-build/prefill/$file"
    done
    git apply -R "$patch_dir/prefill.patch"
    git apply "$patch_dir/decode.patch"
    manifest decode

    if [[ $from_base == 1 ]]; then
        while read -r _ file; do
            mkdir -p "qwen38-build/decode/$(dirname -- "$file")"
            cp -- "$file" "qwen38-build/decode/$file"
        done < qwen38-build/decode.sha256
        cat > qwen38-build/Dockerfile <<EOF
FROM $nightly AS vllm-openai
COPY qwen38-build/decode/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY qwen38-build/wheels/ /opt/qwen38-wheels/
RUN uv pip install --system --offline --no-deps /opt/qwen38-wheels/*.whl
EOF
        cat "$patch_dir/Dockerfile.runtime" >> qwen38-build/Dockerfile
        cp -- "$patch_dir/entrypoint.sh" qwen38-build/
        exit 0
    fi

    sed 's/^flashinfer-python==0[.]7[.]2[.]post1$/flashinfer-python==0.7.5+1658f0bf/' \
        requirements/cuda.txt > qwen38-build/cuda.txt
    grep -Fxq 'flashinfer-python==0.7.5+1658f0bf' qwen38-build/cuda.txt
    mv qwen38-build/cuda.txt requirements/cuda.txt
    awk '
        /^FROM (base AS build|rust-build-cache AS rust-build)$/ {
            print; print "ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.33.0rc1.dev447+qwen38"; next
        }
        /^COPY requirements\/cuda[.]txt / {
            print "COPY qwen38-build/wheels/ /opt/qwen38-wheels/"
            print "ENV UV_FIND_LINKS=/opt/qwen38-wheels"
        }
        /^RUN --mount=type=cache,target=\/opt\/pip\/cache \\$/ {
            line=$0; getline
            if ($0 ~ /flashinfer download-kernels$/) next
            print line
        }
        { print }
    ' docker/Dockerfile > qwen38-build/Dockerfile
    cat "$patch_dir/Dockerfile.runtime" >> qwen38-build/Dockerfile
    cp -- "$patch_dir/entrypoint.sh" qwen38-build/
)

echo "Source build context: $ctx"
if [[ $prepare_only == 0 ]]; then
    if [[ $from_base == 1 ]]; then
        docker build --platform linux/amd64 --target qwen38-openai --tag "$output" \
            "$@" -f "$ctx/qwen38-build/Dockerfile" "$ctx"
        exit 0
    fi
    docker buildx build --load --target qwen38-openai --build-arg INSTALL_KV_CONNECTORS=true \
        --build-arg "VLLM_BUILD_COMMIT=$base" --build-arg torch_cuda_arch_list=12.0 \
        --tag "$output" "$@" -f "$ctx/qwen38-build/Dockerfile" "$ctx"
fi
