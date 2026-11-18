# Kimi K3 text-piece tokenization cache

This branch starts from upstream commit
`dbbc9e310d9b1bf6f9499bc27136686245acc7c7`, used as the source baseline for
`vllm/vllm-openai:kimi-k31-mc-dbbc9e3@sha256:0507dd0ba274bdb5bb8223f3248c0772db2a9517cfd1c06f132ec2989713635e`.
The previous deployment and its build scripts remain in Git history.

Set `KIMI31_TOKENIZER_CACHE_GIB=12` on each model API process to enable a
process-private, bounded LRU cache of Kimi's native text-piece tokenization.
Unset or zero disables the cache. Eight API processes have a combined charged
cache limit of 96 GiB; actual process RSS requires additional headroom.
Coupled, prefill, and decode servers can all enable this cache.

For the eight-GPU B300 deployments, `KIMI31_API_NUMA_BIND=1` binds each API
process's existing threads to the CPUs local to GPU `api_process % gpu_count`
through NVML, then verifies the applied masks. Later threads inherit the mask.
This does not change the CUDA container's cgroup or set a memory policy.
Check actual thread masks and page placement after startup.

The cache preserves the original tokenizer on misses. Its key separates
tokenizer instances and the ordinary/control-token policy. Returned lists are
independent of cached storage. It does not replace GPU prefix caching, Mooncake
KV caching, or P-to-D prompt-token-ID forwarding.

For cold-run resets, set `KIMI31_TOKENIZER_CACHE_EPOCH_FILE` to a writable shared
file visible to every API process of the instance. After requests drain,
atomically replace that file with a fresh epoch. Each API acknowledges the epoch
and clears its cache within its next background poll, including idle APIs.
Wait for every API's active epoch to match and charged bytes to reach zero.
Resetting during active requests does not provide a cold-run boundary.

Metrics labeled by `api_process`:

- `vllm:tokenizer_cache_tokens_total`: encoded, reused, and special tokens.
- `vllm:tokenizer_cache_requests_total`: hit, miss, and fallback calls.
- `vllm:tokenizer_cache_bytes` and `vllm:tokenizer_cache_max_bytes`.
- `vllm:tokenizer_cache_evictions_total`.
- `vllm:tokenizer_cache_epoch`: the first 12 hexadecimal digits of the active
  epoch's SHA256 interpreted as an integer.

The existing Kimi renderer tests cover exact control-token behavior, mutation
isolation, bounded eviction, and an idle API's epoch reset. Deployment also
requires checking the real tokenizer, API metrics, and end-to-end requests.
