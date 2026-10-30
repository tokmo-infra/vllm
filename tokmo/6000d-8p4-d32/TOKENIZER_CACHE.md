# GLM5.3 Native Tokenizer Segment Cache

The native Python API renderer can reuse exact token IDs for repeated text
segments between special added tokens. It remains disabled by default. A cache
miss calls the original Hugging Face `tokenizers` backend with
`add_special_tokens=False`; the vocabulary, BPE algorithm, chat template,
reasoning parser, and model computation remain unchanged.

## Enable

Set these variables in each `vllm serve` API process environment:

```bash
export VLLM_GLM53_TOKENIZER_CACHE_GIB=4
export VLLM_GLM53_TOKENIZER_CACHE_EPOCH_FILE=/state/tokenizer-epoch
printf 'initial\n' > /state/tokenizer-epoch
```

`VLLM_GLM53_TOKENIZER_CACHE_GIB` is a finite, nonnegative GiB limit **per API
process**; `0` disables caching. The optional epoch file must remain readable.
Changing its contents discards that process's retained segments on its next
tokenization call. Restarting the process also discards them. Existing native
`--api-server-count` and `--renderer-num-workers` options still control API and
renderer concurrency. There is no additional tokenizer-cache CLI switch.

For example, eight prefill instances with four API processes each and one
decode head with eight API processes have 40 independent caches. At 4 GiB per
API, their retained-entry budget totals 160 GiB, populated on demand. Headless
decode nodes have no API cache. The budget accounts for retained text, compact
token arrays, keys, and estimated mapping overhead; total process RSS also
includes tokenization temporaries, native tokenizer state, and allocator overhead.
Reserve that additional host RAM alongside NUMA-local Mooncake Stores.

## Supported Tokenizer and Requests

Activation checks `model_type=glm5_next` and the exact `tokenizer.json` SHA256:

```text
19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d
```

A changed tokenizer requires new complete token-ID validation before enabling
this implementation. The cache clones a private, deterministic native BPE backend.
It rejects unsupported normalization and added-token whitespace rules. All added
tokens participate in boundary matching, including ordinary added tokens that
contain a special-token substring. Only proven special-token boundaries split
text; arbitrary character or byte chunking is not used.

Requests needing offsets, multimodal data, BOS/EOS insertion, unsupported encode
options, or effective truncation continue through the original native path.
Chat requests propagate `cache_salt` into tokenization so independent salts do
not share retained segment entries. Each cache is a byte-bounded LRU; repeated
conversation turns can reuse unchanged earlier segments.

## Metrics and KV Cache Maintenance

The `/metrics` endpoint exposes `vllm:tokenizer_cache_enabled`, retained
`vllm:tokenizer_cache_bytes`, `vllm:tokenizer_cache_max_bytes`, evictions, request
paths, and `vllm:tokenizer_cache_tokens_total` by `reused_tokens`, `encoded_tokens`,
and `special_tokens`. Labels include the API process rank.
`vllm:frontend_stage_seconds` records `tokenize_queue`, `cache_encode`, and
`render_chat_total`; these stages overlap and must not be added together.

This CPU token-ID cache is separate from GPU prefix KV, Mooncake CPU KV, and the
Router's token-count estimate. Clearing KV does not clear token IDs. For a cold
benchmark, stop requests, clear GPU and external KV with the deployment's native
maintenance procedure, then change all API epoch files or restart the APIs.
Epoch invalidation is lazy: old entries and byte gauges remain until that API's
next encoding call clears them. Restart APIs when immediate empty-byte verification
is required. A fresh trial salt additionally prevents reuse across trials.
Keep CUDA Graph and compilation caches.

## Validation

CPU regression coverage lives in `tests/renderers/test_glm53_native_cache.py`
and is selected by the existing CPU Reasoning + Renderers CI step. It checks
full native token-ID equality, added-token overlap, Unicode, concurrent encoding,
byte limits, eviction, salt isolation, appended history, fallbacks, and epoch
invalidation. Deployment acceptance additionally compares complete `/tokenize`
responses against the unchanged pinned tokenizer, and checks repeated requests
increase the native cache reuse counters before starting a cold SWE benchmark.

Token-ID equality is a correctness result. It does not establish a serving TTFT
improvement; scheduler, GPU KV, external KV preparation, and model execution can
still dominate latency.

Run the public endpoint acceptance check from an environment containing the pinned
model tokenizer and this branch:

```bash
.venv/bin/python tokmo/6000d-8p4-d32/validate_tokenizer_cache.py \
  --model-path /model --model-name glm-5.3-flash-nvfp4 \
  --url http://prefill:31010 --url http://decode:31011
```

The check sends completion, chat, and BOS fallback tokenization requests twice
over persistent HTTP connections and compares every returned token ID. It sends
no generation requests and does not clear or populate GPU/CPU KV.
