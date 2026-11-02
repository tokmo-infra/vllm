# Hierarchical MegaMoE over RoCE

`deep_gemm_hierarchical_mega_moe` is an experimental MoE backend for Wide EP
across equal-sized NVLink hosts connected by RoCE. It keeps experts sharded
across the entire EP group, transports activations between hosts with NCCL,
and runs native DeepGEMM MegaMoE inside each host. It is opt-in; the native
`deep_gemm_mega_moe` backend is unchanged.

## Requirements and configuration

Use SM100-family GPUs supported by native MegaMoE, MXFP4 expert weights,
TP1, and at least two equal-sized hosts. Every GPU within a host must support
the peer memory access required by DeepGEMM. Host membership is discovered
through vLLM's shared-memory check; membership alone does not prove NVLink
connectivity. Verify GPU/NIC affinity and RoCE connectivity before launching.
DBO/ubatching and fused shared experts are unsupported. Shared experts run
once on the source rank after combining the routed output.

For two hosts with eight GPUs each, use global DP16/EP16 and local DP8:

```bash
# Run on both hosts. Replace MODEL, HEAD_IP and NODE_FLAGS for your deployment.
# Head: NODE_FLAGS=(--api-server-count 16)
# Worker: NODE_FLAGS=(--data-parallel-start-rank 8 --headless)
vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    --data-parallel-size 16 --data-parallel-size-local 8 \
    --data-parallel-address "$HEAD_IP" --data-parallel-backend mp \
    --enable-expert-parallel \
    --kernel-config '{"moe_backend":"deep_gemm_hierarchical_mega_moe"}' \
    "${NODE_FLAGS[@]}"
```

Retain model-specific attention, tokenizer, quantization and speculative
decoding settings. For DeepSeek-V4.1, the backend disables shared-expert
fusion automatically. Do not set the former task-local
`VLLM_DSV41_MEGA_MOE_RDMA` environment variable: selection now uses
`--kernel-config`. This implementation does not require an MSCCL++ fork.

Kernel configuration fields:

| Field | Default | Effect |
| --- | --- | --- |
| `hierarchical_mega_moe_wire_dtype` | `mxfp8` | Native 1x32 E4M3 activations/E8M0 scales on dispatch; `bfloat16` provides a reference path. |
| `hierarchical_mega_moe_direct_staging` | `true` | Copy received MXFP8 bytes/scales into native buffers. Set `false` to reconstruct BF16 and requantize for comparison. Ignored for BF16 wire. |
| `hierarchical_mega_moe_chunk_size` | `0` | Source tokens per network tile; zero sends the complete padded batch. Smaller tiles bound temporary wire memory but add launches. Native receive chunks also respect the configured symmetric-buffer capacity. |

Set `NCCL_NET=IB`, `NCCL_IB_DISABLE=0`, and the correct `NCCL_IB_HCA`,
`NCCL_IB_GID_INDEX`, `NCCL_SOCKET_IFNAME`, and `GLOO_SOCKET_IFNAME` for the
cluster. Inspect NCCL logs to confirm the intended RDMA transport and NICs.
Benchmark `NCCL_NET_GDR_READ` on the actual topology; its best value is not
portable across clusters. See the [deployment carryovers](#deployment-carryovers)
for the previous experiment's CPU-side fixes.

## Data path and correctness

Each source GPU has a lane connecting the same local GPU position on other
hosts. Each tile contains one activation row and its routes for every
destination host. Physical expert IDs are mapped into that host's local EP
numbering; other routes become `-1` with zero gate weight. Native MegaMoE
dispatches over local NVLink, computes both expert projections and the
activation, and returns a gate-weighted BF16 partial. The lane exchanges
partials back and the source sums them in FP32 before converting to BF16.

The MXFP8 packer uses the same quantization primitive as native MegaMoE.
Direct staging preserves activation and scale bytes, including the native
packed/strided scale layout. It removes the receive-side BF16 reconstruction
and requantization. Return partials remain BF16; return compression is not
implemented. EPLB mapping and load recording happen once at the source,
before routing physical slots to hosts. Logical routing capture stays at
the source as well.

Uniform, padded DP batches use fixed capacities and support CUDA graph
capture/replay. Eager batches with unequal token counts first agree on the
global maximum and pad to it. Empty or masked ranks still participate in
the collective schedule. Capture without uniform CPU-side DP metadata is
rejected instead of reading device counts during capture. Warm up before
capturing so topology tensors, communicators and native buffers exist.

## Performance limits and concurrency

Native MegaMoE is fast because dispatch, both GEMMs, activation and combine
share a persistent kernel and directly address peer GPU memory within the
NVLink domain. RoCE does not extend CUDA peer pointers into an equivalent
NVLink memory domain. This backend adds an explicit network boundary around
the native kernel; it does not turn two RoCE hosts into an NVL72 supernode.

For hidden size 5120 and top-k 6, dispatch uses 5328 bytes per source token
per destination host and return uses 10240 bytes. On two eight-GPU hosts,
936 tokens/GPU require about 116.6 MB sent per host over the complete round
trip. Eight 400-Gbit/s links imply a theoretical 0.291 ms serialization
floor, before protocol overhead, packing and compute. At 4096 tokens that
floor is 1.275 ms. These are byte-count bounds, not measured performance.

Network transfers and native compute are deliberately stream-ordered. A
two-stream double-buffer prototype timed out at a 32-token source tile in
the two-host test, including with four NCCL CTAs. The pinned DeepGEMM
implementation requires all device SMs for its locality-domain scheduler
and rejects lowering `num_sms`. The unsafe overlap prototype is not exposed
as a serving option. Efficient overlap needs a compatible native scheduling
change or a transport whose progress does not compete with that scheduler.

## Reproduce the measurements

Use exclusive GPUs and the same model dimensions, routing, token counts,
NCCL settings and CUDA graph mode for every case. On each host:

```bash
.venv/bin/python -m torch.distributed.run \
    --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
    --master-addr="$HEAD_IP" --master-port=29719 \
    benchmarks/kernels/benchmark_hierarchical_mega_moe.py \
    --tokens 768,936,1024,1536,4096 --source-chunks 256,512,1024 \
    --cuda-graph --stress-replays 1000 --iterations 100 \
    --output /tmp/wide-ep-results.json
```

This measures synthetic FP4 experts with an independent dense oracle,
including changed inputs, padding and physical expert relocation across
hosts on graph replay. It compares reconstruction, direct staging, bounded
network tiles and independent local EP groups. It does not measure router
selection, shared MLPs, end-to-end serving throughput or model accuracy.
See [validation](hierarchical_mega_moe_validation.md) for the recorded results
and remaining acceptance work.

The [matched-payload probes](../../benchmarks/kernels/roce_megamoe/README.md)
compare NCCL with MSCCL++'s GPUNetIO device PortChannel separately from GEMMs.
MSCCL++ [PR 935](https://github.com/microsoft/mscclpp/pull/935) and
[PR 943](https://github.com/microsoft/mscclpp/pull/943) are useful communication
design references; they are not drop-in native MXFP8 MegaMoE integrations.

## Deployment carryovers

This branch cherry-picks the exact DSV4.1 native tokenizer cache from
`9170113763ceabe68e01156c1d4fb18c7a69ace8`, then adapts formatting and its
backend constructor. It does not cherry-pick the old RDMA MegaMoE code.

- `VLLM_DSV41_TOKENIZER_CACHE_GIB` defaults to `0` (disabled). A positive
  value enables a private exact segment LRU per API process, only with
  `--tokenizer-mode deepseek_v41`. For example, 4 GiB across 16 processes
  budgets up to 64 GiB; it is not a shared 4-GiB pool.
- `VLLM_DSV41_TOKENIZER_CACHE_EPOCH_FILE` optionally names a readable text
  file. Changing its contents clears each process's cache before its next
  encode. Unsupported normalization or added-token behavior is rejected.
  Offsets, multimodal input, special-token insertion, and result-changing
  truncation use the original tokenizer. `cache_salt` isolates reuse.
- Metrics include `vllm:tokenizer_cache_tokens`,
  `vllm:tokenizer_cache_requests`, `vllm:tokenizer_cache_bytes`,
  `vllm:tokenizer_cache_max_bytes`, `vllm:tokenizer_cache_evictions`,
  `vllm:frontend_stage_seconds`, and `vllm:frontend_active_renders`.
- Main already warms the target DSV4.1 indexer projection in FlashInfer
  autotuning (upstream PR 60543). This branch also visits the speculative
  model when present.
- For internal load balancing, keep the Wide EP16 head's API process count
  at 16, matching two EP8 heads' combined count. The prior explicit value
  of eight caused listener pressure. Use NUMA CPU and memory binding and
  reserve CPU memory for model loading, API caches and the operating system.
- Previous decode comparisons used one speculative token and GPU memory
  utilization 0.85. Keep these and all request limits equal in A/B tests;
  they are deployment settings, not new library defaults.

The separate router's cache-aware preprocessing patch and process-scoped
TCP backlog adapter are preserved under
[`examples/online_serving/roce_wide_ep`](../../examples/online_serving/roce_wide_ep/README.md).
They are not changes to vLLM's Rust frontend. The historical 600-second
long-generation timeout remains unresolved; no retry or deadline extension
is presented as a fix. Validate long requests and connection pressure again
with the new serving stack.
