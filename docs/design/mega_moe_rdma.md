# Experimental MegaMoE NVSHMEM transport

`VLLM_MEGAMOE_TRANSPORT=nvshmem` selects the experimental native RDMA transport
for DeepGEMM MegaMoE. The default is `nvlink`. This first implementation is
limited to SM100-family GPUs, FP8 activations, MXFP4 routed weights and SwiGLU,
including the existing FP8 fused shared-expert path. Both DeepGEMM build modes
and single-node B300 kernel correctness have been validated with CUDA 13.0,
including forced remote staging with P2P disabled. Cross-node EP8 correctness
and synthetic MoE layer timing have also been exercised with four GPUs per
node. Model evaluation and end-to-end serving performance remain pending.

Build vLLM against the matching DeepGEMM development tree:

```sh
export DEEPGEMM_SRC_DIR=/path/to/deepgemm-megamoe-rdma
export DG_ENABLE_NVSHMEM=1
export NVSHMEM_HOME=/path/to/nvshmem
```

Use the normal vLLM source-build workflow with those variables. NVSHMEM must
provide `nvshmemx_collective_launch_attr` for MegaMoE's 2-SM clusters and
`nvshmemx_quiet_warp` for collective RMA completion. Consult
the paired DeepGEMM tree's `docs/mega_moe_rdma.md` for the inspected API revision,
linking requirements, protocol and validation commands. CMake records the
transport build options so switching them invalidates an earlier native build.
The upstream DeepGEMM pin is unchanged; enabling the build option without the
matching local source produces an error.

Load the NVSHMEM runtime that matches the build before importing PyTorch.
The B300 validation image required explicit selection because its bundled
NVSHMEM was too old for cluster launch attributes:

```sh
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$NVSHMEM_HOME/lib/libnvshmem_host.so${LD_PRELOAD:+:$LD_PRELOAD}"
```

Set `VLLM_MEGAMOE_TRANSPORT=nvshmem` on every worker when using the native
MegaMoE backend. vLLM passes its EP device and Gloo groups to DeepGEMM, includes
the transport in its buffer-cache key, and requires RDMA ABI version 1 from the
package actually imported. An external `deep_gemm` installation has precedence
over the vendored package; a stale package produces an error with its path.

The first version uses process-lifetime symmetric allocations and reuses cached
buffers across layers. Extra staging can be substantial: it reserves full input
and per-peer/top-k return slabs. DeepGEMM limits each staging allocation to 4 GiB
by default; `DG_MEGAMOE_RDMA_MAX_STAGING_BYTES` raises that budget explicitly.
Size `NVSHMEM_SYMMETRIC_SIZE` for the sum of all native and staging allocations,
including distinct main/draft configurations. Restart the worker to release
these process-lifetime allocations.

DBO and overlapping microbatches are rejected. One EP ordering and GPU per
process are supported, and the process must not already host another NVSHMEM
world. Prepare and warm up before CUDA Graph capture. Kernel launches use the
NVSHMEM collective launch with cluster attributes and PDL disabled.
Use the paired DeepGEMM revision with the per-buffer GPU epoch fix: CUDA Graph
replays can reuse `%gridid`, which cannot safely tag a fresh RDMA return.

Single-node validation on 2026-11-04 also exercised the real vLLM expert wrapper
at EP1: MXFP4 weight loading, default unclamped activation, distinct transport
buffers, eager output and 20 changing-route CUDA Graph replays. NVSHMEM output
matched NVLink exactly. This used branch Python sources with the validation
image's existing vLLM binary extensions and newly built DeepGEMM; a full vLLM
source build and model serving were not exercised.

Cross-node validation uses the paired DeepGEMM harness with exact native EP1
reference checks, including shared experts, an empty rank and masked routes.
Its optional `--benchmark-iterations` mode restores identical seeded inputs
before timing and checks results after each batch. See the paired DeepGEMM
`docs/mega_moe_rdma.md` for commands and layer timing results. These results do
not establish a serving TPOT or throughput improvement.

IBGDA configuration also needs runtime verification. In the tested environment,
`NVSHMEM_IB_ENABLE_IBGDA=1` enabled the transport, but a requested GPU NIC handler
initially fell back to CPU host memory after an I/O-memory mapping error.
Adding `CAP_SYS_ADMIN` to the isolated validation containers resolved the mapping
failure on both nodes; adding `SYS_RAWIO` alone had not. The tested NVIDIA driver
uses `CAP_SYS_ADMIN` for its administrator check. No host driver reload was needed.
Use `NVSHMEM_IB_ENABLE_IBGDA=1`, `NVSHMEM_IBGDA_NIC_HANDLER=gpu` and
`NVSHMEM_DEBUG=INFO`, then require `NIC handler will be GPU` in every worker's
log. The handler request alone does not prevent this NVSHMEM build from falling
back. Follow the deployment's approved driver/container configuration for GPU
peer mappings; this experiment does not require a fully privileged container.

Select a GPU-local HCA for each worker with `NVSHMEM_HCA_PE_MAPPING`,
`NVSHMEM_ENABLE_NIC_PE_MAPPING=1` and `NVSHMEM_MAX_NICS_PER_PE=1`; unset
`NVSHMEM_HCA_LIST` when using an explicit mapping. Set
`NVSHMEM_IB_TRAFFIC_CLASS` to the fabric's configured class (106 in the tested
environment) and choose its RoCE GID (index 3 here). HCA names and traffic class
are deployment-specific. The paired DeepGEMM document records both hosts'
tested mappings. This validation did not change host NIC or driver settings.

With live-token input transfers, parallel return rows and collective completion
in the paired DeepGEMM branch, plus the runtime corrections above, cross-node
EP8 layer latency fell from 11.719 to 1.072 ms at 128 tokens/rank, and from
16.398 to 2.800 ms at 512 tokens/rank. Both cases passed 100 changing-route
replays and exact checks after each timed batch, using the baseline input
hashes. These improvements include both kernel and runtime changes; they are
synthetic layer results and do not measure full-model TPOT.

The matching DeepGEMM branch now shares each fetched remote input within its
NVLink-accessible NVSHMEM team and sends a result row as soon as its final GEMM2
tile is stored. It preserves native GEMM/TMA scheduling, ring credits, reduction
order and native NVLink synchronization. Return completion uses one elected
warp after the existing grid rendezvous; cleanup can overlap receive waits and
combine. Counters reuse the completed dispatch slab, with a fallback when the
slab is too small, so the staging budget and RDMA ABI remain unchanged.

Under the same test configuration, cross-node medians are now **0.846 / 1.986 ms**
at 128 / 512 tokens/rank, reducing latency by **21% / 29%** versus the preceding
1.072 / 2.800 ms. Each passed 100 changing-route replays plus exact checks after
seven timed batches, with matching baseline input hashes. These are DeepGEMM
layer results; this update does not establish full-model serving performance.
Use both matching feature branches when building; no new vLLM option is needed.
The paired native NVLink controls passed 20 changing-route replays and every
timed-batch check, measuring 0.191 / 0.225 ms versus the previous 0.192 / 0.225 ms.
The cross-node path remains about 4.4 / 8.8 times slower in these layer tests.

DeepGEMM also provides an **opt-in queued-return experiment**. The measured
results above apply to the preceding streaming baseline (`67f08c5`); newer
results are recorded below. After rebuilding the matching DeepGEMM extension,
set `DG_MEGAMOE_RDMA_RETURN_SENDERS=16` on every worker before buffer allocation
and graph warmup to enable it. `0` is the default and retains direct streaming;
`1..32` selects the number of sender warps, capped by the kernel's CTA count.
Restart workers when changing the setting. Native NVLink ignores this option.

Selected dispatch warps submit queued rows only after completing their original
input pulls. Epilogue warps publish row readiness and continue computing; they
do not wait for network submission. This retains the original GEMM/TMA
scheduling, ring credits, shared-expert math, reduction order and compute CTA
count. Full-launch queue capacity prevents queue-full backpressure on producers.
The queues reuse the completed route slab, preserving allocation sizes and ABI 1;
insufficient storage falls back to the preceding paths. Sender completion is
joined before the existing quiet/signals, with cleanup/combine overlap retained.

Use the paired DeepGEMM document's correctness and performance matrix before
enabling this for a deployment. CPU protocol tests and offline CUDA compilation
do not establish GPU correctness, an overlap improvement or a serving benefit.
No vLLM execution-path change or new vLLM option is needed for this experiment.

On 2026-11-05, the paired DeepGEMM `8e5b4be` passed the main EP8/EP24 matrix
with exact checks and 100 changing-route replays. EP8 on nodes 55/89 measured
direct/queue8 medians of 0.658/0.619 ms at 128 tokens and 0.979/0.956 ms at 512.
The latter was noisy; queue16/32 regressed the larger case. EP24 on 55/89/91
measured 2.065/2.044 ms and 8.353/9.859 ms, also noisy. Native EP8 on node 91
measured 0.195/0.227 ms with matching EP8 input hashes. These different-host
layer controls are not serving results; queue8 still took about 3.17/4.22x as
long as native. `DG_MEGAMOE_RDMA_RETURN_SENDERS=0` remains the default.

The next candidate is `DG_MEGAMOE_RDMA_OVERLAP=1` in
[DeepGEMM `2fbc8f1`](https://github.com/tokmo-infra/DeepGEMM/commit/2fbc8f13298633cd73bf8288dc4d8c9c46a25e6a).
Its EP8 CUDA/GPU validation is recorded below; full-model evaluation remains
pending. The default `0` preserves the preceding protocol.
Rebuild DeepGEMM, set the same value on all workers before allocation and
warmup, and restart workers when changing it. Native NVLink ignores this
option; the vLLM transport and ABI requirements above stay unchanged.

The candidate publishes expert counts and releases shared epilogues before
bulk input prefetch. Dispatch still waits for completed input caches before
producing ring rows. Local output readiness is signaled before remote quiet;
combine waits only for peers selected by the current token. A GPU-wide cache
shares one successful NVSHMEM target-consistency probe per required remote
peer per launch, then per-warp register caches avoid repeated checks.
The cache uses reserved staging-header space, with explicit per-launch reset.

Scheduler, GEMM/TMA computation, ring credits, shared-expert math and reduction
order are retained. Bulk input gating of routed ring pulls, per-row return
atomics, PE-wide remote quiet and the cleanup world barrier remain; preserving
the compute code alone does not preserve all of its overlap. This candidate
adds a dispatch-only rendezvous. No model evaluation has run for it.

On 2026-11-06, the same kernel source was rebuilt and tested on B300 nodes
55/89/91. Cross-node EP8 (55/89, four GPUs each) passed all eight combinations
of 128/512 tokens, overlap0/1 and sender0/8, each with 100 changing-route
replays and exact output/count checks after every timed batch. Native controls
on 55 and 91, local NVSHMEM controls, and boundary cases also passed; see the
paired DeepGEMM document for the complete scope and remaining validation.

| Single-layer mode | 128 tokens/rank, us | 512 tokens/rank, us |
| --- | ---: | ---: |
| Native NVLink on node 55 | 190.432 | 224.557 |
| Cross-node default (overlap0, sender0) | 657.683 | 977.282 |
| Cross-node overlap1, sender0 | 652.493 | 976.245 |
| Cross-node overlap0, sender8 | 619.618 | 959.754 |
| Cross-node overlap1, sender8 | 607.706 | 924.978 |

Inputs matched by hash. Overlap alone gave no material layer improvement
(0.79%/0.11%). The 512-token sender8 pair repeated at 943.160/923.198 us for
overlap0/1; it remains noisy. The best combination still takes about 3.19/4.12x
the node-55 NVLink time. Both experiment defaults stay disabled.

Device timestamps confirmed that the first shared epilogue tile became ready
185–194 us earlier, during input prefetch. Whole-prefix input preparation,
the routed pipeline and the cleanup tail still determine total latency.
Selected-peer waiting moves into combine, so removal of a dedicated receive
barrier must not be reported as removal of all network waiting. These results
validate the tested layer paths, not serving TPOT or full-model correctness.

DeepGEMM's RDC build also emits ptxas C7504 (`setmaxnreg` ignored across
compilation units). Source-level preservation of register budgets is therefore
insufficient to claim preservation of native register redistribution. Its
runtime contribution remains unmeasured; see the paired DeepGEMM document.

This transport extends the MoE EP path only. Keep TP/GEMM-RS and Engram sharing
within their supported NVLink domains. Both main and draft physical expert
counts must be divisible by their actual EP sizes; 256 experts cannot run EP24
without a separate ownership/replication change. No serving configuration or
throughput result is implied by the transport implementation.
