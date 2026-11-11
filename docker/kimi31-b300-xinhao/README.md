# Kimi31 B300 YYF development runtime

Canonical development branch: kimi-3.1-1110, rebased on kimi31-b300-xinhao at
fd39348c2ba44c0882757526c6cb774276daf6c0. Xinhao already descends from the previous
Kimi branch; this rebase fast-forwards the history without rewriting its commits.

The runtime is a **Python-only derivative** of the pinned Kimi image:

    vllm/vllm-openai:kimi-k31
    sha256:1a8a702f2bfeded3c90259ae60fefbd9e4ca014241946e7d5f3396badce99e76

C++/CUDA/Rust binaries and serving dependencies are retained, not rebuilt by this
runtime target. Full native source builds remain available through
docker/Dockerfile --target kimi-k31 for future native changes.

## Build and verify

From a committed checkout on a host with that exact base image:

    bash docker/kimi31-b300-xinhao/build-runtime.sh
    # Defaults: SOURCE_COMMIT=HEAD, IMAGE_TAG=vllm/vllm-openai:kimi-k31-yyf

    docker image inspect vllm/vllm-openai:kimi-k31-yyf \
      --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
    docker run --rm --network=none --entrypoint cat \
      vllm/vllm-openai:kimi-k31-yyf /opt/kimi31-yyf-source.sha256

The context contains three Python payload files and the Dockerfile from one
selected commit, not arbitrary uncommitted files. SOURCE_COMMIT selects a
committed revision; IMAGE_TAG changes the output tag. The original image is not
overwritten. Revision labels and installed hashes identify actual code. A
successful Docker build alone does not prove model correctness or performance.

## Baseline and cache-fix comparison

Node92 only: TP8 / DP1 / DCP8, expert parallel disabled, DSpark 5, nvfp4_ds_mla,
max model length -1, seq128 and Graph768. GPU utilization starts at0.98, reduced
only for actual OOM. Mooncake's private RDMA owner is sized after model readiness
from80% of then-available host memory.

The image defaults to one16GiB shared CPU tokenizer L2 and64MiB L1 per API. See
[configuration](../../docs/serving/kimi31_tokenizer_cache.md). Use
`--prefix-match-unit 64` for **KV prefix matching**; this does not force the
physical block size or make every 64-token boundary a saved Mamba checkpoint.
Record the effective hash size and hit alignment separately.

Each version runs one C68 point,102 samples and102 users, original SWE13w,
reasoning max and streamed usage, with an explicit55% statistics start. Stop at
the existing continuous60-second underfill condition. Do not silently use the53%
default for1.5x samples, or start other concurrency points. Results, raw data,
deployment evidence and credentials remain outside Git. Preserve the immutable
baseline image and result before porting applicable cache fixes from
glm5.3-flash-1103. Then run the optimized image with the same configuration,
dataset and statistics window; do not mix fixes into the running baseline.
Checks cover large-batch autotune coverage, lookup/get/put versus real reused
tokens, pressure recovery, leases, checkpoint alignment and load-error handling.
A lookup hit alone proves neither successful retrieval nor token reuse.
