# Hierarchical MegaMoE archive validation

This is a development archive, not a production acceptance report. The base
is vLLM main `62c9b6263cea4a31444e597ebde1b1fe64dbe208` (2026-11-02).
The old RDMA MegaMoE implementation was not cherry-picked. The tokenizer
cache provenance and deployment carryovers are documented in the
[serving guide](hierarchical_mega_moe.md).

## Completed checks

- CPU/Gloo: 10 cases cover one to four logical islands, noncontiguous ranks,
  unequal and empty batches, source/receive chunking, masks and physical
  expert moves. Command:
  `.venv/bin/python -m pytest --noconftest tests/distributed/test_hierarchical_mega_moe.py -q`.
- Native tokenizer cache: three real BPE tokenizer tests passed, covering
  concurrent reuse, salts, epoch reset, fallback and eviction.
- B300: 92 kernel/module tests passed with `--noconftest`, including eight
  exact native MXFP8 staging cases (H=256/5120; T=7/65/936/1025).
  The remaining 31 tests required `dist_init`, absent with that invocation;
  they were not counted as passes. Retrying from an isolated tests copy
  collected all 123 but still did not load the top-level fixture; a complete
  repository test environment is needed for those 31 cases.
- Two B300 hosts, eight GPUs each: synthetic Wide EP16 ran at 768, 936,
  1024, 1536 and 4096 source tokens/GPU. Every case checked an independent
  dense oracle, changed graph inputs, masks, and cross-host physical expert
  relocation. Each timing followed 1000 CUDA graph replays, then 100 timed
  iterations; reported latency is the maximum across ranks.
- The standalone GPUNetIO C++ probe and pinned MSCCL++ dependency compiled
  with CUDA 13.0.88. Device-QP setup failed; no transport performance claim
  follows from that build.

The GPU environment used current-main Python sources over the existing
`e0d00e4b8eb0` image's native libraries, PyTorch 2.14.0+cu130 and NCCL
2.31.2. The relevant MegaMoE native API matched; this is not a clean build
of every native extension from the new main revision.

## Single-layer timing

Milliseconds, synthetic MXFP4 weights, E=384, H=5120, intermediate=2304,
top-k=6, fast math disabled. The local reference is two independent EP8
groups, each holding the full expert set. Whole-model behavior is not
measured here.

| Source tokens/GPU | Reconstruct BF16 | Direct MXFP8 | Tile 256 | Tile 512 | Tile 1024 | Independent EP8 |
| --- | --- | --- | --- | --- | --- | --- |
| 768 | 1.102 | 1.089 | 1.449 | 1.244 | 1.041 | 0.225 |
| 936 | 1.262 | 1.267 | 1.818 | 1.438 | 1.258 | 0.250 |
| 1024 | 1.350 | 1.346 | 1.891 | 1.501 | 1.342 | 0.258 |
| 1536 | 1.890 | 1.882 | 2.782 | 2.199 | 2.068 | 0.323 |
| 4096 | 4.850 | 4.798 | 7.240 | 5.656 | 5.138 | 0.741 |

Direct staging removes redundant conversions but gives only a small change
in total latency, including variation in either direction. Small ordered
tiles are slower here, so the default sends the full padded batch.
These data do not establish performance parity with NVL72 or an improvement
over the previous implementation under a controlled paired comparison.

## Failed overlap experiment and pending work

A two-stream, double-buffer prototype completed a 16-token source-tile
smoke case but timed out at 32 tokens, including with four NCCL CTAs.
DeepGEMM rejects reducing the SM count for MegaMoE because its locality
scheduler requires the full GPU. The unsafe prototype and unsupported SM
reservation knob were removed from the serving backend.

The next task is to profile actual Wide EP16 compute and transport at larger
per-step token counts. Compute time must be remeasured as batch size grows;
the earlier 0.78 ms measurement is not a constant for high concurrency.
Distinguish concurrent requests from scheduler-limited tokens per step.

Pending acceptance includes: complete native-extension build, real model
startup and accuracy, source EPLB load-accounting integration under serving,
draft warmup regression, sustained high-concurrency serving, and the prior
HTTP/long-generation timeout regressions. GPUNetIO setup and safe
communication/compute overlap are still research work. This branch must not
be treated as having passed those checks.
