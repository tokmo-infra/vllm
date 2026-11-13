# Kimi K3.1 CPU tokenizer cache

The renderer reuses native XTML text-piece encoding without changing chunk
boundaries, ordinary/control-token policy or token IDs. This is a **CPU cache**,
separate from GPU prefix KV matching and Mooncake's KV store. An 8-GPU deployment
can budget 2 GiB per GPU as one **16 GiB shared L2**, not 16 GiB per API and not GPU
memory.

## Configuration

The YYF runtime image enables:

    KIMI31_TOKENIZER_CACHE_GIB=16
    KIMI31_TOKENIZER_CACHE_L1_MIB=64
    KIMI31_TOKENIZER_CACHE_SHARED_PATH=/tmp/kimi31-tokenizer-cache-yyf

Each API has a bounded 64 MiB private L1. One private local service owns the
shared 16 GiB L2, growing on demand; APIs in the serving container connect to the
same service. Container termination also terminates the cache service. Use a
deployment-private shared path. L1 memory is additional: eight API processes can
retain up to 512 MiB of aggregate L1.

The 16 GiB L2 and 64 MiB L1 are recipe defaults, not topology limits. Set
KIMI31_TOKENIZER_CACHE_GIB to another capacity, such as 32 for one 32 GiB shared
pool, and KIMI31_TOKENIZER_CACHE_L1_MIB to the desired private budget per API.
All clients sharing a path must agree on the L2 budget. The pool grows on
demand rather than preallocating that capacity; budget for available host RAM
and the additional private L1s. A local service does not share memory across
hosts: each serving container needs its own private path and budget.

Set KIMI31_TOKENIZER_CACHE_GIB=0 to disable caching. Without a shared path,
KIMI31_TOKENIZER_CACHE_GIB retains the original process-private behavior and
KIMI31_TOKENIZER_CACHE_L1_MIB does not apply. Budgets must be finite and
nonnegative. Startup and communication errors are explicit, not reported as hits.

L2 identity fingerprints the native encoder source, BPE ranks, regex and special
token map. Keys retain exact text and ordinary/control-token policy, not a
process-local object address. Returned token lists are independent copies.
Text is not rechunked or approximately matched.

## Metrics and reset

Existing metrics have the api_process label:

- vllm:tokenizer_cache_bytes and vllm:tokenizer_cache_max_bytes describe private
  L1 in shared mode, or the original private cache otherwise.
- vllm:tokenizer_cache_tokens_total, source=reused_tokens/encoded_tokens/special_tokens
- vllm:tokenizer_cache_requests_total, path=hit/miss/fallback
- vllm:tokenizer_cache_evictions_total

A hit can come from L1 or L2. Bytes conservatively charge retained strings, ID
arrays and LRU entries; RSS also includes Python and tokenizer memory. Shared
occupancy and limits are available through the cache service's stats(). Do not
sum the shared capacity once per API. CPU token reuse does not measure GPU KV
hits or prove that Mooncake get/put completed.

The single-API HTTP /metrics endpoint also exports:

- vllm:tokenizer_l2_cache_bytes
- vllm:tokenizer_l2_cache_entries
- vllm:tokenizer_l2_cache_max_bytes
- vllm:tokenizer_l2_cache_evictions_total

An independent collector preserves the L1 counters and gauges when engine logger
initialization unregisters its metric wrappers. With no
PROMETHEUS_MULTIPROC_DIR, the L2 daemon enables mmap metrics in the private
SHARED_PATH/metrics directory before importing prometheus_client; the API keeps
its normal single-process registry. Scrapes only read these files: they make no
cache RPC, acquire no cache lock, and neither encode text nor change LRU order.
Cache keys, byte charging, capacity and native token IDs are unchanged.
The existing per-put L2 gauge updates now write to mmap files; this has
observability overhead to account for in CPU replay. Gauges are sampled
independently, not as one atomic stats() snapshot.

When PROMETHEUS_MULTIPROC_DIR is already configured, the daemon keeps using that
directory and the normal multiprocess exporter collects its budget once. Set
this environment before starting processes/importing prometheus_client; do not
set or change it midway through a run. A replacement daemon clears its old live
gauges before publishing the new budget. The files can retain the last snapshot
after an abnormal exit, and multiprocess eviction counters can include earlier
daemon lifetimes: these metrics are not a daemon-health probe or an L2 hit-rate
measurement. Reuse of a terminated deployment's metric directory is unsupported;
use a fresh private cache path. Do not modify or delete metric files during a run.

KIMI31_TOKENIZER_CACHE_EPOCH_FILE optionally names a shared epoch file. Change
it only after ingress stops and requests drain. Each API checks it at most once
per second and clears L1 on change. L2 identity includes the epoch, so an old
in-flight computation cannot populate a new epoch. Old epoch entries remain
bounded by the LRU until eviction; an epoch change is not proof of zero resident
bytes. Use a fresh deployment-private shared path for a genuinely fresh cache.
