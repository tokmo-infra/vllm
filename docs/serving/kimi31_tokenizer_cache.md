# Kimi K3.1 CPU tokenizer cache

The Kimi renderer can reuse native XTML text-piece tokenization with a bounded,
process-private cache. Set `KIMI31_TOKENIZER_CACHE_GIB=12` to allow up to 12 GiB
per API process. The default is `0` (disabled). This budget grows on demand;
it is separate from GPU prefix KV caching and Mooncake L2.

The cache calls the original Kimi `_encode_text_piece` on misses and keeps its
chunking and ordinary/control-token policy. The key includes that policy and
the tokenizer backend identity. Returned token lists are independent copies.
One shared LRU budget covers tokenizer instances in an API process.

Prometheus metrics use the `api_process` label:

- `vllm:tokenizer_cache_bytes`, `vllm:tokenizer_cache_max_bytes`
- `vllm:tokenizer_cache_tokens_total{source="reused_tokens"|"encoded_tokens"|"special_tokens"}`
- `vllm:tokenizer_cache_requests_total{path="hit"|"miss"|"fallback"}`
- `vllm:tokenizer_cache_evictions_total`

Bytes are a conservative charge for retained strings, ID arrays and LRU entries;
monitor API RSS as well. Token reuse is `reused / (reused + encoded + special)`
over counter deltas. These counters cover native text-piece encoding, including
structural tokens. They measure CPU tokenization reuse, independently of KV hits.

`KIMI31_TOKENIZER_CACHE_EPOCH_FILE` optionally names a shared epoch file. Change
it only after ingress is stopped and requests drain. Each process checks the
file at most once per second while encoding; the next encoding clears entries
when the epoch changes. Clearing is lazy, so a changed file alone does not prove
that the byte gauge has already reached zero.

A Kimi tokenizer that lacks `_encode_text_piece` fails explicitly when the cache
is requested. The cache does not change renderer concurrency or implement P/D
token-ID forwarding. The current TP8/DP1/DCP8 deployment is coupled.
