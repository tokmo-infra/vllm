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
| `hierarchical_mega_moe_sparse_dispatch` | `true` | Verbs only: compact each destination's selected token rows on the GPU. Registered capacities and native call shapes stay fixed. `false` is a dense reference; tail padding is still excluded from the network. |
| `hierarchical_mega_moe_return_dtype` | `bfloat16` | Verbs only: `mxfp8` compresses remote island partials with 1x32 E8M0 scales. The source uses its own BF16 partial directly. Validate model quality before enabling this lossy transport. |
| `hierarchical_mega_moe_chunk_size` | `0` | Source tokens per network tile. NCCL zero sends the complete padded batch. Verbs zero selects at most 2048 source rows fitting one native call. Smaller tiles bound temporary memory but add launches. |
| `hierarchical_mega_moe_transport` | `nccl` | `verbs` selects the experimental multi-host GPUDirect pipeline described below. |
| `hierarchical_mega_moe_rdma_host_staging` | `false` | Use an independent CUDA copy stream and pinned CPU send buffers; receives remain GPUDirect and the configured return precision is preserved. Benchmark on the target topology. |
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

Check PCIe routing before tuning the transport. On the measured B300 pair,
ACS request/completion redirection limited GPU bidirectional RDMA to about
153 Gb/s per direction while host memory reached 390 Gb/s. A controlled
redirect-only change on the relevant switch ports restored approximately
390 Gb/s for GPU memory too. Merely seeing a 400G link and a nearby GPU/NIC
in `nvidia-smi topo -m` does not establish a fast peer path. Have the platform
owner validate ACS and device isolation for the deployment; this backend
does not change PCI registers. See the
[controlled routing experiment](hierarchical_mega_moe_validation.md#pcie-routing-diagnosis).

With corrected routing, direct verbs sends, four QPs and 2048 source rows per
tile achieved 1.54x/1.59x over the faster tested NCCL configuration at
8192/16384 tokens per GPU. That comparison used `NCCL_NET_GDR_READ=1`.
Pinned-host staging is an optional workaround when the GPU send path remains
constrained; it is not the recommended path for that corrected topology.

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
and requantization. Return partials use the configured BF16 or MXFP8 format.
EPLB mapping and load recording happen once at the source,
before routing physical slots to hosts. Logical routing capture stays at
the source as well.

Uniform, padded DP batches use fixed capacities and support CUDA graph
capture/replay. Eager batches with unequal token counts first agree on the
global maximum and pad to it. Empty or masked ranks still participate in
the collective schedule. Capture without uniform CPU-side DP metadata is
rejected instead of reading device counts during capture. Warm up before
capturing so topology tensors, communicators and native buffers exist.

### Variable-length verbs payloads

The verbs path separates a registered peer slot's capacity from its current
row count. A GPU prefix scan builds one stable source-to-compact-row map per
destination. Each token is sent at most once per selected host. Tail and
empty-rank rows send no payload. A zero-row peer still exchanges an epoch
credit, retaining one common collective schedule without a CPU count sync.

Credits advertise current byte counts. Peers transfer only those bytes and
copy received row counts to the GPU after completion. Unused receive routes
are masked before the fixed-shape native calls. Return traffic uses dispatch
receive counts; the source restores token order using its saved map. Input
maps and pending return maps have separate lifetimes, allowing input-slot
reuse while an older return is in flight. The protocol supports equal-sized
multi-host groups; two-host tests do not establish larger-EP hardware acceptance.
All hosts must upgrade their Python transport and JIT extension together:
protocol 2 credits include payload lengths and reject protocol 1 peers.

For MXFP8 returns, a GPU packing kernel encodes native BF16 partials. Source
reduction decodes remote MXFP8 directly into an FP32 sum and adds the original
local BF16 partial. Final output is BF16. Expert weights and the native kernel
are unchanged. Compare quality with BF16 returns before enabling it for a model.

## Performance limits and concurrency

Native MegaMoE is fast because dispatch, both GEMMs, activation and combine
share a persistent kernel and directly address peer GPU memory within the
NVLink domain. RoCE does not extend CUDA peer pointers into an equivalent
NVLink memory domain. This backend adds an explicit network boundary around
the native kernel; it does not turn two RoCE hosts into an NVL72 supernode.

For hidden size 5120 and top-k 6, dispatch uses 5328 bytes per selected source
token per destination host. BF16 return uses 10240 bytes; MXFP8 return uses
5280 bytes, a 48.4% reduction. Compaction also excludes unselected hosts and
tail rows. For the dense BF16 reference on two eight-GPU hosts,
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

**Implementation status: single-layer hardware validation is available; full
model serving acceptance remains pending.** See the
[validation report](hierarchical_mega_moe_validation.md) for the measured
baseline, matched-tiling comparisons, network diagnostics and limitations.

This path uses standard RC verbs and GPU DMA-BUF registration, without the
GPUNetIO device-QP requirement. A CPU progress thread submits NIC operations;
payloads remain in GPU memory by default. It requires Linux, CUDA driver development
headers, a C++17 compiler, Ninja, libibverbs development headers/libraries and
working GPU DMA-BUF export. The small C++ extension is compiled on first use
with PyTorch's extension cache. Hosts must have equal-sized NVLink islands.
Each GPU lane opens an independent RC endpoint to every remote host. Their
credits and completion flags are independent; all remote contributions must
complete before compute or reduction consumes the shared peer-major slot.
The local contribution is copied once, and remote peers write disjoint rows.
Two-host calls retain one endpoint and the original synchronization protocol.

The compute stream packs two input tiles, publishes their requests, and waits
for the current tile's receive. It runs native MegaMoE, publishes that tile's
return in the configured format, and reuses its input slot to prefetch a later tile. Before reusing
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
   nonempty stripe and delivers a write-with-immediate completion. Zero-byte
   stripes complete through the credit protocol. Two length slots preserve
   current-epoch metadata if a zero-length transfer allows the next credit
   to arrive before the local proxy posts the current epoch.
3. The proxy waits for every send completion and every receive stripe. GPU
   receive visibility uses the device's advertised owner-ordering guarantee,
   or the supported host GPUDirect flush API. It then clears the request and
   publishes completion. A stream memory wait releases the consumer, followed
   by a GPU acknowledgement before the next request can reuse that job.
   Memory registration requests relaxed ordering and falls back to ordinary
   ordering if the provider rejects that option. Completion visibility is
   unchanged; the credit protocol prevents overlapping writes to reused slots.
4. Direct-send CUDA dependencies are on one stream. The proxy independently
   progresses network traffic while native MegaMoE owns all SMs. No NCCL
   communication kernel runs concurrently with the native grid on this path.
   CUDA enqueue calls run outside the proxy mutex: an enqueue may block under
   stream backpressure, so holding that mutex would prevent the network work
   needed to unblock the GPU and would also stop the watchdog.
5. The lane's peer endpoints and shape-keyed plans are shared by sequential MoE layers.
   Concurrent forwards or overlapping graph replays using the same endpoint
   are unsupported. Warm up every tile shape before capture. Finish work and
   retire graphs before closing the groups and releasing registered storage.

With `hierarchical_mega_moe_rdma_host_staging=true`, an independent CUDA copy
stream waits for each producer, copies the remote-bound bytes to a pinned
CPU buffer, and publishes the request. The NIC then reads CPU memory and
writes directly to the peer GPU. A consumer joins that copy stream with a
CUDA event before its usual completion wait, including during graph capture.
This lets copies and RDMA progress during later native compute without using
communication CTAs. The completed return/consume protocol protects both GPU
and pinned buffers against premature reuse. Only the send path is staged;
the configured wire format is unchanged. Staging currently copies the full
registered send capacity; NIC transfers still use active byte counts. Each cached two-slot shape adds
`2 * (hosts - 1) * source_tile_rows * (dispatch_row_bytes + return_row_bytes)` pinned bytes
per rank, approximately 60.8 MiB at 2048 rows for the measured model shape.
Pinned-memory and PCIe costs can outweigh the benefit on other topologies.

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

Sparse destination dispatch is enabled by default for verbs. To additionally
enable lossy MXFP8 remote returns, add
`"hierarchical_mega_moe_return_dtype": "mxfp8"` to that kernel configuration.
Use `"hierarchical_mega_moe_sparse_dispatch": false` and BF16 returns for the
dense reference. Tail rows are excluded in both modes.

For four eight-GPU hosts, use global DP32/EP32, local DP8 and TP1, with each
host's DP start rank set to `8 * node_rank`. The same kernel configuration
discovers four islands automatically. A 2048-row source tile gathers 8192 rows
per local lane; the native workspace must fit that tile or compute splits it
into smaller calls. Each lane needs three progress threads and three sets of
QP stripes. Dense dispatch and BF16 returns send three remote copies per
source row, so EP32 does not imply better latency than local EP8. Validate
bandwidth, correctness and matched workload performance on all four hosts.

Batch size must distinguish requests, effective tokens and graph padding.
For this DeepSeek DSpark5 checkpoint's default anchor mode, 1024 requests per
DP rank produce up to 6144 target verification tokens and 5120 parallel draft
query tokens. The three draft layers run once per draft pass. CUDA Graph
selection uses padded token counts and the required batch descriptor; a
request count alone does not identify the selected target or draft graph.

### Reproduce single-layer acceptance

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
    --include-nccl-baseline \
    --output-directory /tmp/roce-overlap-sweep
```

The harness checks changing transport payloads exactly, compares pipeline
output bit-for-bit with matched-tiling serial verbs, and checks a separate
synthetic-expert oracle. It exercises unequal eager counts (including an empty
rank), tail padding, all-masked inputs, and physical expert relocation across
hosts during graph replay. Oracle tolerances retain the existing quantized
single-layer benchmark's limits for BF16 returns; serial/pipeline parity has
zero tolerance. MXFP8 runs also execute a matched BF16-return reference:
the native oracle keeps `rtol=0.08, atol=0.01`, while additional return rounding
uses `rtol=0.065, atol=0.01`. This extra relative bound applies to these
nonnegative synthetic partials; it is not a model-quality acceptance threshold.
Failures must be investigated before reporting performance.

Use `--sparse-dispatch --return-dtype mxfp8` to exercise all three payload
optimizations. `--transport-only --virtual-island-size 2` with four local
processes per host checks four logical islands on two physical hosts. This
checks multi-peer synchronization, not four-host bandwidth or native EP32.

`--include-nccl-baseline` measures the full-batch NCCL path using the same
warmup and sample timing. `--nccl-only` runs just that baseline, while
`--transport-only` isolates full-batch dispatch, BF16 return and concurrent
dispatch/return without native compute. Partial rank files preserve completed
measurements if a later case fails; do not present a failed run as acceptance.

Outputs include all ranks' sample latencies, rank 0/8 CUDA timelines, and each
rank's proxy request/post/completion timestamps. Proxy intervals bound the
transfer's NIC service and polling time; they are not direct NIC hardware
timestamps. Compare the maximum rank for each sample, then summarize samples.
The `scheduled_rows` field reports fixed compute capacity, not transmitted
rows; proxy trace `bytes` records actual NIC payload lengths.
Also rerun the unchanged NCCL baseline in the same environment. A valid result
must show actual overlap and lower whole-layer latency, not just shorter
isolated communication spans. Whole-model numerical and serving acceptance
remain separate work after this single-layer step.

### Replay checkpoint weights and captured inputs against DEP8

`benchmark_roce_mega_moe_replay.py` compares independent local DEP8 islands
with Wide EP using the same captured hidden states, gates, expert IDs and
checkpoint weights. It times the complete routed MoE operation, including
packing, network transfers and reduction. Attention and shared experts are
outside this boundary. Each physical GPU receives the same source rows in
both topologies. Keep source tokens per GPU and the total physical GPU count
fixed; the DEP8 arm replicates all experts on each host.

The input `.pt` pool contains `hidden[N,5120]`, `gates[N,6]`, `ids[N,6]`,
`reference[N,5120]`, `fast_math` and
`kernel_kwargs={"activation_clamp": 10.0}`. Use actual captured target-layer
rows with their matching reference output. The harness selects rows by rank
and cycles through the pool when it is smaller than the requested global
shape. Report that reuse; it does not reconstruct a simultaneous production
batch or a sequential multi-layer execution.

```bash
.venv/bin/python -m torch.distributed.run \
    --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
    --master-addr="$HEAD_IP" --master-port=29725 \
    benchmarks/kernels/benchmark_roce_mega_moe_replay.py \
    --model "$MODEL" --inputs /path/to/layer20-pool.pt \
    --hcas mlx5_0:1,mlx5_6:1,mlx5_8:1,mlx5_10:1,mlx5_12:1,mlx5_14:1,mlx5_18:1,mlx5_20:1 \
    --gid 3 --traffic-class 106 --qps 4 \
    --tokens 768,1536,3072,4096 --tile 2048 --chunk 4096 \
    --samples 7 --iterations 20 --stress-replays 100 \
    --output-directory /tmp/roce-checkpoint-replay
```

BF16 Wide EP uses the existing cross-topology numerical bounds. The MXFP8
wire/reduction result must exactly match an independent reference that rounds
remote partials and transmits them as BF16. Its error against DEP8 is recorded
separately, without claiming model-quality acceptance. Changed-input graph
checks use fresh executions in the same topology, and all-masked rows must
return zero. `--fixed-input-only` omits those changed-input checks explicitly;
it is useful for a narrowly scoped historical timing reference, not acceptance.
Use `--reference-version` only with the old transport implementation installed.

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
