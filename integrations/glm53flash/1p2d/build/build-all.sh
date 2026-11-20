#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../../../.." && pwd)
mode=${1:-source}
jobs=${BUILD_JOBS:-8}
work=$(mktemp -d "${BUILD_PARENT:-/tmp}/glm53-1p2d.XXXXXXXX")
base=${ENGINE_BASE_IMAGE:-local/glm53flash-1103:source-67ca8f23}
engine=${ENGINE_IMAGE:-local/glm53flash-nvfp4kv:compat-78159a469b9a}
mc=${MOONCAKE_IMAGE:-local/glm53flash-mooncake:source-71973589}
router=${ROUTER_IMAGE:-router:glm53flash-responses-tools-ecee97b77d8e}
fi_repo=https://github.com/flashinfer-ai/flashinfer.git
pin_dir="$root/tools/glm_nvfp4kv"
patchdir="$root/integrations/glm53flash/patches"
case "$mode" in source|base|overlay) ;; *) echo '用法：build-all.sh source|base|overlay' >&2; exit 2;; esac
printf '构建文件：%s\n' "$work"
build=(docker buildx build --platform linux/amd64 --progress=plain)
if [[ -n ${GLM_BUILDER:-} ]]; then build+=(--builder "$GLM_BUILDER"); fi
if [[ -n ${GLM_BUILD_PROXY:-} ]]; then
  build+=(--build-arg "HTTP_PROXY=$GLM_BUILD_PROXY" --build-arg "HTTPS_PROXY=$GLM_BUILD_PROXY")
fi
checkout() {
  local repo="$1" rev="$2" dir="$3"
  git init -q "$dir"
  git -C "$dir" fetch --depth=1 "$repo" "$rev"
  git -C "$dir" checkout -q --detach FETCH_HEAD
  test "$(git -C "$dir" rev-parse HEAD)" = "$rev"
}
if [[ "$mode" != overlay ]]; then
  checkout https://github.com/kvcache-ai/Mooncake.git 719735896c86b56fabec6cf3e825fb2ea640597a "$work/mooncake"
  git -C "$work/mooncake" submodule update --init --recursive
  for patch in mooncake-occupied-accounting mooncake-fit-pressure; do
    git -C "$work/mooncake" apply "$patchdir/$patch.patch"
  done
  "${build[@]}" -f "$here/Dockerfile.mooncake" --target wheels \
    --build-context "mooncake_src=$work/mooncake" --build-arg "BUILD_JOBS=$jobs" \
    --output "type=local,dest=$work/mooncake-wheels" "$here"
  "${build[@]}" -f "$here/Dockerfile.mooncake" --target service --load \
    --build-context "mooncake_wheels=$work/mooncake-wheels" -t "$mc" "$here"
  cp -a "$here/../router" "$work/router"
  (
    cd "$work/router"
    patch --batch --fuzz=0 -p1 < "$here/router-production.patch"
    sha256sum -c "$here/router-production.sha256" >/dev/null
  )
  "${build[@]}" -f "$here/Dockerfile.router" --build-context "router_src=$work/router" \
    --build-arg "BUILD_JOBS=$jobs" --load -t "$router" "$here"
fi
if [[ "$mode" == source ]]; then
  checkout "$fi_repo" 0b46115d238d29ea2e92c6b9b2cb077d40fbc352 "$work/flashinfer071"
  git -C "$work/flashinfer071" submodule update --init --recursive
  git -C "$work/flashinfer071" apply "$patchdir/flashinfer-autotune.patch"
  "${build[@]}" -f "$here/Dockerfile.flashinfer" --target wheels \
    --build-context "flashinfer_src=$work/flashinfer071" \
    --output "type=local,dest=$work/flashinfer-wheels" "$here"
  # 先从源代码重建现场所用的 67ca8f23 基底，再应用同一份 NVFP4 KV 覆盖层。
  base_rev=67ca8f23ae3192992af6b099b3cc8041316fe787
  if ! git -C "$root" cat-file -e "$base_rev^{commit}" 2>/dev/null; then
    git -C "$root" fetch --depth=1 https://github.com/tokmo-infra/vllm.git "$base_rev"
  fi
  mkdir "$work/vllm"
  git -C "$root" archive "$base_rev" | tar -xf - -C "$work/vllm"
  python3 "$here/prepare-vllm.py" "$work/vllm" "$work/vllm/docker/Dockerfile.glm53-source"
  sed -i 's/^flashinfer-python==0\.7\.0\.post1$/flashinfer-python==0.7.1/' "$work/vllm/requirements/cuda.txt"
  grep -qx 'flashinfer-python==0.7.1' "$work/vllm/requirements/cuda.txt"
  args=()
  if [[ -n ${GLM_BUILD_BASE_OCI:-} ]]; then
    args+=(--build-context "glm_pytorch_buildbase=oci-layout://$GLM_BUILD_BASE_OCI" --build-arg BUILD_BASE_IMAGE=glm_pytorch_buildbase)
  fi
  "${build[@]}" "${args[@]}" -f "$work/vllm/docker/Dockerfile.glm53-source" --target glm53flash-source \
    --build-context "build_helpers=$here" --build-context "flashinfer_wheels=$work/flashinfer-wheels" \
    --build-context "mooncake_wheels=$work/mooncake-wheels" \
    --build-arg 'torch_cuda_arch_list=12.0' --build-arg "max_jobs=$jobs" --build-arg nvcc_threads=2 \
    --build-arg 'VLLM_USE_PRECOMPILED=' --build-arg INSTALL_KV_CONNECTORS=false \
    --build-arg RUN_WHEEL_CHECK=false --load -t "$base" "$work/vllm"
else
  # 和现场基底一致；已有不同 tag 时可用 ENGINE_BASE_IMAGE 引用同一镜像。
  test "$(docker image inspect --format '{{.Id}}' "$base")" = \
    sha256:60de6d0cd0715ebead3f7005fa36d668874ff813ae8124ec5e9741cd198353bd
fi

# 0.7.1 负责原来的 FlashInfer/MoE；只取较新版 B12X 源目录与私有锁实现。
checkout "$fi_repo" 1658f0bf02b65fe9a71862286f6da546d0fce890 "$work/flashinfer-b12x"
ctx="$work/overlay"
mkdir -p "$ctx/flashinfer/experimental" "$ctx/b12x"
cp -a "$work/flashinfer-b12x/flashinfer/experimental/b12x" "$ctx/flashinfer/experimental/"
cp "$work/flashinfer-b12x/_flashinfer_filelock.py" "$ctx/"
cp "$here/b12x/__init__.py" "$ctx/b12x/"
(
  cd "$root"; sha256sum -c "$pin_dir/runtime.sha256" >/dev/null
  awk '{print $2}' "$pin_dir/runtime.sha256" | tar -cf - -T -
) | tar -xf - -C "$ctx"
(
  cd "$ctx"
  # Keep the repository sources lintable while reproducing the deployed files.
  patch --batch --fuzz=0 -p1 < "$here/runtime-production.patch"
  sha256sum -c "$here/runtime-production.sha256" >/dev/null
  sha256sum -c tools/glm_nvfp4kv/b12x-before.sha256 >/dev/null
  patch --batch --fuzz=0 -p1 < tools/glm_nvfp4kv/b12x-masked-record.patch
  sha256sum -c "$pin_dir/b12x-runtime.sha256" >/dev/null
)
docker build --network none --pull=false -f "$here/Dockerfile.nvfp4-kv" \
  --build-arg "BASE_IMAGE=$base" -t "$engine" "$ctx"
docker image inspect --format '{{.Id}} {{join .RepoTags " "}}' "$engine" "$mc" "$router"
