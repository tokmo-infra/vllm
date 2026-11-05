# B300 disaggregated PD runtime source snapshot

This repository contains the complete upstream source at `e0d00e4b8eb0c89ec30921d0695cfdd409e9c211`, overlaid with all5943 source/resource files exported from the pinned vLLM runtime image. Installed DeepGEMM/CUTLASS and FlashAttention sources are included. The three native tokenizer-cache overlays are integrated into `vllm/renderers/base.py` and `vllm/tokenizers/`.

`image-source-manifest.json` records the installed package files and SHA256 values for14 compiled binaries. Compiled objects remain in the pinned base image; the corresponding upstream vLLM C++/CUDA/Rust build tree is present in this repository. `SOURCE.json` records image identity and the three tokenizer overlay hashes. Their executable bodies match the deployed cache after formatting and type annotations.

`vllm/v1/worker/gpu/input_batch.py` is the original pinned-image source, without the experimental adaptive-profile KV group isolation patch. The deployment baseline uses fixed five-token DSpark verification; the upstream optional adaptive-verification implementation is otherwise unchanged.

Topology: four DEP2 prefill instances on55 and independent DEP8 decode instances on89/91, TP1 and EP,24 GPU ranks, FP8 KV, MegaAttention and MegaMoE. The native tokenizer segment cache is opt-in via `VLLM_DSV41_TOKENIZER_CACHE_GIB`; the deployment uses32 private API caches of12GiB each. Set `VLLM_DSV41_TOKENIZER_CACHE_EPOCH_FILE` to a shared epoch file to invalidate each cache before the next encoding. Unsupported tokenization options follow the native tokenizer path.

Build from the repository root, retaining the base image's compiled extensions:

```bash
docker build -f tokmo/pd-disaggregated-4p2-2d8/Dockerfile.runtime -t tokmo-vllm:b300-current .
```

The lint configuration excludes installed vendored FlashAttention sources and the generated version file, as it already excludes third-party sources. The deployed cache retains its exact `re` literal matcher with an explicit checker exemption.
