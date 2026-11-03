# Experimental MegaMoE NVSHMEM transport

`VLLM_MEGAMOE_TRANSPORT=nvshmem` selects the experimental native RDMA transport
for DeepGEMM MegaMoE. The default is `nvlink`. This first implementation is
limited to SM100-family GPUs, FP8 activations, MXFP4 routed weights and SwiGLU,
including the existing FP8 fused shared-expert path. CUDA compilation, GPU
correctness, model evaluation and performance validation are still pending.

Build vLLM against the matching DeepGEMM development tree:

```sh
export DEEPGEMM_SRC_DIR=/path/to/deepgemm-megamoe-rdma
export DG_ENABLE_NVSHMEM=1
export NVSHMEM_HOME=/path/to/nvshmem
```

Use the normal vLLM source-build workflow with those variables. NVSHMEM must
provide `nvshmemx_collective_launch_attr` for MegaMoE's 2-SM clusters. Consult
the paired DeepGEMM tree's `docs/mega_moe_rdma.md` for the inspected API revision,
linking requirements, protocol and validation commands. CMake records the
transport build options so switching them invalidates an earlier native build.
The upstream DeepGEMM pin is unchanged; enabling the build option without the
matching local source produces an error.

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

This transport extends the MoE EP path only. Keep TP/GEMM-RS and Engram sharing
within their supported NVLink domains. Both main and draft physical expert
counts must be divisible by their actual EP sizes; 256 experts cannot run EP24
without a separate ownership/replication change. No serving configuration or
throughput result is implied by the transport implementation.
