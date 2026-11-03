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
16.398 to 2.800 ms at 512 tokens/rank. Both final cases passed 100 changing-route
replays and exact checks after each timed batch, using the baseline input
hashes. These improvements include both kernel and runtime changes; they are
synthetic layer results and do not measure full-model TPOT.

This transport extends the MoE EP path only. Keep TP/GEMM-RS and Engram sharing
within their supported NVLink domains. Both main and draft physical expert
counts must be divisible by their actual EP sizes; 256 experts cannot run EP24
without a separate ownership/replication change. No serving configuration or
throughput result is implied by the transport implementation.
