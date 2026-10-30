# RTX6000D Eight-Node GLM5.3 Flash PD Deployment

This snapshot contains 30 independent launch commands for nodes 81–88.
81–84 host eight NUMA-local TP1/DP4/EP4 prefill instances; 85–88 form one
global DP32/EP32 decode group. Each file starts only its named container.
Inspect with `bash FILE.sh --dry-run`, `--print-argv`, or `--print-config`.
No command starts another service or sources a common launch script.

| Setting | Prefill | Decode |
| --- | --- | --- |
| API processes | 4 per instance, 32 total | 8 on node 85; others headless |
| Sequences / rank | 64 | 128 |
| Batched tokens | 4096 | 2048 |
| GPU memory fraction | 0.90 | 0.95 |
| MTP tokens | 3 | 3 |
| CUDA Graph maximum | 256 | 512 |
| max-model-len | -1 | -1 |
| Native tokenizer cache | 4 GiB per API | 4 GiB per API; headless 0 |

All 16 Mooncake CPU Stores allocate 256 GiB each, totaling 4 TiB, with explicit
NUMA CPU/memory binding. Prefill uses local preferred segments and global fallback.
PD uses native MooncakeConnector RDMA; CPU KV uses MooncakeStoreConnector.
Decode EP HCA 0/2/5/6 and KV HCA 1/3/4/7 are disjoint. PIX GPU pairs retain P2P.
Inspect `cluster.json` and the complete commands for actual ports, HCA names,
NUMA nodes, and image IDs; this is a specific PCIe/RoCE deployment.

## Prepare Runtime Mounts

The base model image is `local/glm53flash-vllm:0.32.0`, ID `540155b18ee7`.
The Router is `local/glm53flash-router:0.1.16`, ID `21c9cfa19e34`.
The existing Master/Stores use the separately pinned image in `cluster.json`.
Model scripts read-only mount five Python files. Prepare these from the pinned
model image, using this checkout mounted at `/source` and an output directory
mounted at `/output`:

```bash
uv venv --system-site-packages /tmp/cache-prep/.venv
/tmp/cache-prep/.venv/bin/python \
  /source/tokmo/6000d-8p4-d32/prepare_runtime.py \
  --output /output/runtime
```

The preparer verifies original and output SHA256s. It generates the exact
validated renderer cache hooks, the default-false native MLA operator adapter,
and the explicit DeepEP SM32 compatibility patch. Native kernels remain unchanged.
It copies the two cache modules from this branch and refuses changed existing
outputs. `runtime/*.json` records both the edits and their hashes; full third-party
source copies are unnecessary. Copy those manifests alongside the five generated
files on every deployment node.

Copy the independent launch scripts, `cluster.json`, and `preflight.py` to
`/data/bench/eng-xinhao/vllm-pd/glm53-flash/6000d-8p4-d32` on each node.
`GLM_WORKDIR` and `GLM_MODEL_DIR` can override script paths; update the inventory
consistently for preflight checks. The recorded model mount is
`/data/models/GLM-5.3-Flash-NVFP4`.

## Start and Check

Start the Master, then each node's two Stores, then all four decode commands
and eight prefill commands. The distributed decode commands must run concurrently.
Wait for all nine model base APIs to report healthy before executing `router.sh`.
It reads actual engine KV capacities before starting the Router.
Keep existing CUDA/Triton compilation caches during model restarts.

Before a selected service starts, run its local preflight in a prepared Python
environment. This checks identity, image, GPUs, ports, HCA state, source/tokenizer
hashes, and host RAM including the API tokenizer-cache budget:

```bash
.venv/bin/python preflight.py 81 --script prefill-81-0.sh
bash prefill-81-0.sh
```

Run the [public tokenizer acceptance check](validate_tokenizer_cache.py)
and confirm cache reuse counters before formal benchmarking. The
[cache guide](TOKENIZER_CACHE.md) explains
native fallbacks, per-process limits, and epoch invalidation.

For a cold test, stop this deployment's benchmark, wait until all engines are
idle, pause all nine model base APIs with native maintenance endpoints, reset
GPU and external KV, and verify this dedicated Master's key count and allocated
bytes are zero. Resume the APIs and restart the deployment's Router.
Change all tokenizer epochs before the next trial; invalidation occurs lazily
on each API's next tokenization call. Use a new trial cache salt as well.
Never clear other deployments' pools.

The recorded SWE80K benchmark uses C1024/A1536, streaming, temperature 0,
max_tokens 4096 with natural EOS, and native client billing at CNY 0.16/0.046/0.56
per million uncached-input/cached-input/output tokens. Cache correctness tests
do not establish a TTFT improvement; use completed native benchmark results.

## SWE80K Benchmark

`c1024-a1536-swe80k.yaml` and `benchmark.sh` retain the exact run003 configuration
and prices. Execute them on the recorded B300 client after all services are ready.
They require the existing `perf_benchmark` client/toolbox, not `vllm bench`.
The script refuses to overwrite a result DB; choose a new run-specific DB and log
path for each subsequent trial. Dataset files and client dependencies are external.
