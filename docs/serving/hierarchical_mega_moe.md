# Hierarchical MegaMoE over RoCE

`deep_gemm_hierarchical_mega_moe` is an experimental MoE backend for Wide EP
across equal-sized NVLink hosts connected by RoCE. It keeps experts sharded
across the entire EP group and runs native DeepGEMM MegaMoE inside each host.
The default cross-host transport is NCCL. An experimental verbs implementation
adds a CPU-driven RDMA pipeline around bounded native compute chunks. It is
opt-in; the native
`deep_gemm_mega_moe` backend is unchanged.

## Requirements and configuration

Use SM100-family GPUs supported by native MegaMoE, MXFP4 expert weights,
TP1, and at least two equal-sized hosts. Every GPU within a host must support
the peer memory access required by DeepGEMM. Host membership is discovered
through vLLM's shared-memory check; membership alone does not prove NVLink
connectivity. Verify GPU/NIC affinity and RoCE connectivity before launching.
DBO/ubatching and fused shared experts are unsupported. The verbs backend also
rejects sleep-mode memory remapping of registered GPU storage. Shared experts run
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
| `hierarchical_mega_moe_chunk_size` | `0` | Source tokens per network tile. NCCL zero sends the complete padded batch. Verbs zero selects at most 2048 source rows fitting one native call. Smaller tiles bound temporary memory but add launches. |
| `hierarchical_mega_moe_transport` | `nccl` | `verbs` selects the experimental two-host GPUDirect pipeline described below. |
| `hierarchical_mega_moe_overlap` | `true` | Pipeline verbs transfers and native compute. Disable for a matched serial verbs reference. NCCL remains ordered. |
| `hierarchical_mega_moe_rdma_hcas` | `[]` | Required for verbs: one `HCA[:port]` per local EP lane, in lane order. |
| `hierarchical_mega_moe_rdma_gid_index` | `0` | RoCE GID index for verbs. |
| `hierarchical_mega_moe_rdma_qps` | `4` | RC QPs per lane, from 1 to 8. |
| `hierarchical_mega_moe_rdma_traffic_class` | `0` | Per-QP traffic class for verbs; does not change NIC or switch configuration. |
| `hierarchical_mega_moe_rdma_timeout_seconds` | `120` | Per-transfer proxy watchdog. Transport failure terminates the worker rather than releasing incomplete results. |

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

With the NCCL transport, transfers and native compute are stream-ordered. A
two-stream double-buffer prototype timed out at a 32-token source tile in
the two-host test, including with four NCCL CTAs. The pinned DeepGEMM
implementation requires all device SMs for its locality-domain scheduler
and rejects lowering `num_sms`. The unsafe overlap prototype is not exposed
as a serving option. Efficient overlap needs a compatible native scheduling
change or a transport whose progress does not compete with that scheduler.

## Experimental verbs pipeline

**Implementation status: GPU build, correctness, graph replay, and performance
validation are pending.** The source and acceptance harness are prepared; no
speedup is established for this backend. The previously recorded results refer
to the NCCL baseline. Hardware testing was deferred at the user's request.

This path uses standard RC verbs and GPU DMA-BUF registration, without the
GPUNetIO device-QP requirement. A CPU progress thread submits NIC operations;
payloads remain in GPU memory. It requires Linux, CUDA driver development
headers, a C++17 compiler, Ninja, libibverbs development headers/libraries and
working GPU DMA-BUF export. The small C++ extension is compiled on first use
with PyTorch's extension cache. Only two equal-sized NVLink hosts are supported.

The compute stream packs two input tiles, publishes their requests, and waits
for the current tile's receive. It runs native MegaMoE, publishes that tile's
BF16 return, and reuses its input slot to prefetch a later tile. Before reusing
an output slot it waits for the older return and combines that result. The
last two returns are drained before the forward finishes. Input and output
slots have separate lifetimes, so return can proceed during later compute.

This is **network-tile pipelining across native kernel invocations**. The
native GEMM/SwiGLU/NVLink fusion inside each tile is preserved. This change does
not implement per-GEMM-tile RDMA inside one persistent kernel spanning the
entire batch, and it does not remove DeepGEMM's internal grid barriers. Smaller
source tiles may reduce GEMM efficiency; the acceptance sweep must measure
that tradeoff, as well as shared GPU-memory and NIC contention.

The synchronization protocol has these invariants:

1. GPU pack/output production precedes a CUDA stream memory write publishing
   the request. Default memory-write fencing makes send bytes visible before
   the proxy posts NIC reads. These operations use no CUDA communication CTA.
2. Both peers advertise the same job's next 64-bit epoch before either writes
   data. Advertising grants a receive credit only after the prior receive was
   consumed and the new send was produced. Each QP transfers one disjoint
   stripe and delivers a write-with-immediate completion.
3. The proxy waits for every send completion and every receive stripe. GPU
   receive visibility uses the device's advertised owner-ordering guarantee,
   or the supported host GPUDirect flush API. It then clears the request and
   publishes completion. A stream memory wait releases the consumer, followed
   by a GPU acknowledgement before the next request can reuse that job.
4. All local CUDA dependencies are on one stream. The proxy independently
   progresses network traffic while native MegaMoE owns all SMs. No NCCL
   communication kernel runs concurrently with the native grid on this path.
5. One endpoint and shape-keyed plans are shared by sequential MoE layers.
   Concurrent forwards or overlapping graph replays using the same endpoint
   are unsupported. Warm up every tile shape before capture. Finish work and
   retire graphs before closing the groups and releasing registered storage.

For the two-host deployment, replace the original `--kernel-config` with:

```json
{
  "moe_backend": "deep_gemm_hierarchical_mega_moe",
  "hierarchical_mega_moe_transport": "verbs",
  "hierarchical_mega_moe_overlap": true,
  "hierarchical_mega_moe_chunk_size": 2048,
  "hierarchical_mega_moe_rdma_hcas": [
    "mlx5_0:1", "mlx5_6:1", "mlx5_8:1", "mlx5_10:1",
    "mlx5_12:1", "mlx5_14:1", "mlx5_18:1", "mlx5_20:1"
  ],
  "hierarchical_mega_moe_rdma_gid_index": 3,
  "hierarchical_mega_moe_rdma_traffic_class": 106,
  "hierarchical_mega_moe_rdma_qps": 4
}
```

Those NIC names, GID and traffic class are deployment-specific. The setting
does not repair inconsistent DSCP/PFC configuration or establish attainable
GPU-to-NIC bandwidth. The host proxy also needs CPU scheduling capacity; do not
constrain an entire worker and its progress thread to a single busy CPU core.
NCCL/Gloo settings still apply to bootstrap, the baseline and local group setup.

### Pending single-layer acceptance

Run the following on both hosts only after GPUs are available. Start with the
small correctness command, then run the sweep. `tokens` means source tokens
**per GPU**, not global batch tokens or serving request concurrency.

```bash
.venv/bin/python -m torch.distributed.run \
    --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
    --master-addr="$HEAD_IP" --master-port=29723 \
    benchmarks/kernels/benchmark_roce_mega_moe_overlap.py \
    --hcas mlx5_0:1,mlx5_6:1,mlx5_8:1,mlx5_10:1,mlx5_12:1,mlx5_14:1,mlx5_18:1,mlx5_20:1 \
    --gid 3 --traffic-class 106 --qps 4 \
    --tokens 33,257 --tiles 16,128 --stress-replays 10 \
    --samples 2 --iterations 3 --output-directory /tmp/roce-overlap-smoke

.venv/bin/python -m torch.distributed.run \
    --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
    --master-addr="$HEAD_IP" --master-port=29724 \
    benchmarks/kernels/benchmark_roce_mega_moe_overlap.py \
    --hcas mlx5_0:1,mlx5_6:1,mlx5_8:1,mlx5_10:1,mlx5_12:1,mlx5_14:1,mlx5_18:1,mlx5_20:1 \
    --gid 3 --traffic-class 106 --qps 4 \
    --tokens 768,1536,4096,8192,16384 --tiles 256,512,1024,2048 \
    --stress-replays 1000 --samples 5 --iterations 20 --trace \
    --output-directory /tmp/roce-overlap-sweep
```

The harness checks changing transport payloads exactly, compares pipeline
output bit-for-bit with matched-tiling serial verbs, and checks a separate
synthetic-expert oracle. It exercises unequal eager counts (including an empty
rank), tail padding, all-masked inputs, and physical expert relocation across
hosts during graph replay. Oracle tolerances retain the existing quantized
single-layer benchmark's limits; serial/pipeline parity has zero tolerance.
Failures must be investigated before reporting performance.

Outputs include all ranks' sample latencies, rank 0/8 CUDA timelines, and each
rank's proxy request/post/completion timestamps. Proxy intervals bound the
transfer's NIC service and polling time; they are not direct NIC hardware
timestamps. Compare the maximum rank for each sample, then summarize samples.
Also rerun the unchanged NCCL baseline in the same environment. A valid result
must show actual overlap and lower whole-layer latency, not just shorter
isolated communication spans. Whole-model numerical and serving acceptance
remain separate work after this single-layer step.

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
