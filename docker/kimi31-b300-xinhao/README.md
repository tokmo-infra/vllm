# Kimi31 B300: official image with tokenizer caching

Source baseline: upstream `78e9a0c28093d71d2805f9acc74d3c08c88a4314`, as
recorded by the vLLM package in the pinned image. The native Kimi renderer was
compared byte for byte with that commit before adding the cache hook.

Base image:
`vllm/vllm-openai:kimi-k31-mc@sha256:f81998764e8594d71b40788c5285ae112acf9230129d5d0d238b5df83da5acf6`

Its linux/amd64 Docker image ID is
`sha256:5a259407d16d003765361f98e9178c83915dc615287f37d525efdb4012642f21`.
The two Python files in `Dockerfile.runtime` are the only private runtime
changes. C++/CUDA/Rust binaries come from the official image.

To package the same files, from the repository root:

```bash
docker build -f docker/kimi31-b300-xinhao/Dockerfile.runtime -t kimi31-b300:tokenizer-cache .
```

Set `KIMI31_TOKENIZER_CACHE_GIB=12` and, for the eight-GPU deployment,
`KIMI31_API_NUMA_BIND=1`. Set the cache epoch file to an instance-specific
shared writable path. See the [cache documentation](../../docs/serving/kimi31_tokenizer_cache.md).

The initial 93 deployment uses the pinned base image with the cache files
mounted read-only, allowing startup without an image build. This does not mean
the base image itself contains this cache. Deployment receipts record the
base image ID, mount paths, and patch SHA256 values. The initial deployed
cache module differs from the final source only in typing comments; its AST
is identical, and both hashes are recorded in `runtime-source.json`.
The optional Dockerfile
packages that same payload; a packaged image requires its own image-ID receipt.
