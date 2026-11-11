# Kimi31 B300 Xinhao

Branch: `kimi31-b300-xinhao`, based on `kimi-3.1-1110` / `3c1cbff4e`.
The installed runtime baseline is `266e88c051ca46607fd9de20216bd58426f80052`.

The deployed image is a Python-only derivative of the pinned Kimi runtime.
It adds exact native text-piece tokenization caching through the Kimi renderer;
the original C++/CUDA/Rust libraries and serving dependencies are retained.
The full source build remains available through `docker/Dockerfile --target kimi-k31`.

```bash
test "$(docker image inspect vllm/vllm-openai:kimi-k31 --format '{{.Id}}')" = \
  sha256:1a8a702f2bfeded3c90259ae60fefbd9e4ca014241946e7d5f3396badce99e76
docker build --pull=false -f docker/kimi31-b300-xinhao/Dockerfile.runtime \
  -t tokmo-vllm:kimi31-b300-xinhao-u095 .
```

Current deployment: node 93, TP8 / DP1 / DCP8, expert parallel disabled,
DSpark 5, `nvfp4_ds_mla`, GPU memory utilization 0.95, 12 GiB tokenizer cache per
API. Mooncake L2 is sized after model initialization, retaining 256 GiB for the
system, API/Router growth and a 64 GiB CPU benchmark client. L3 is disabled.

The tokenizer cache keeps Kimi's structural and ordinary text policies separate.
See [configuration and metrics](../../docs/serving/kimi31_tokenizer_cache.md).
Native renderer checks used the actual Kimi tokenizer: 3 chat cases, 10
text-piece/mode cases, 48 concurrent renders, mutable return isolation and bounded
LRU eviction all passed. No benchmark corpus was preloaded for those checks.

Deployment scripts and compact C64/A128 and C96/A192 configurations are retained
with the local deployment notes. C64 is authorized after service readiness;
C96 requires a separate instruction. Draft KV observation reads per-rank backing
storage and active group block ownership, deduplicating aliases and keeping
physical TP replicas distinct. No draft SWA behavior has been enabled.

## Deployed source identity

[The source record](runtime-source.json) identifies image
`sha256:c0c11bb4c2b0c5e8dcc4c8542c1152d8e2ac535c2613ad25a03589b02c651601`.
Its exact two-file Python payload is archived in commit
`a79a8889f8d750243361bdc93849bb61711ab848` on this branch. Subsequent source cleanup
normalizes formatting and narrows the cache-hit Optional type; the native encoding
and cache policy are unchanged. To reproduce that deployed payload, use:

```bash
bash docker/kimi31-b300-xinhao/build-runtime.sh
```

`SOURCE_COMMIT=HEAD` instead builds the current source. Image IDs can differ across
builds; compare the installed Python file hashes with the selected source commit.
