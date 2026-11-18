# MTP (Multi-Token Prediction)

MTP is a speculative decoding method where the target model includes native
multi-token prediction capability. Unlike draft-model-based methods, you do not
need to provide a separate draft model.

MTP is useful when:

- Your model natively supports MTP.
- You want model-based speculative decoding with minimal extra configuration.

## Gemma 4 Assistant Models

Gemma 4 assistant checkpoints use vLLM's Gemma 4 MTP path. They are not generic
draft models, even though they are passed through the `model` field in
`--speculative-config`.

Use `"method": "mtp"` when serving Gemma 4 with an assistant checkpoint:

```bash
vllm serve google/gemma-4-E2B-it \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --speculative-config '{"method":"mtp","model":"gg-hf-am/gemma-4-E2B-it-assistant","num_speculative_tokens":1}'
```

The E2B, E4B, 12B, 26B-A4B, and 31B Gemma 4 IT assistant checkpoints are supported.
Tower-based variants use `model_type: gemma4_assistant` and the encoder-free
Gemma 4 Unified variant (12B) uses `model_type: gemma4_unified_assistant`.
vLLM maps both to `Gemma4MTPModel` internally and wires the assistant layers
to share KV cache with the target model.

If an older vLLM release logs `SpeculativeConfig(method='draft_model', ...)`
for a Gemma 4 assistant checkpoint, that release is treating the assistant as a
generic draft model and may fail during initialization for multimodal Gemma 4
targets. Upgrade to a version with Gemma 4 MTP support instead.

## Offline Example

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="XiaomiMiMo/MiMo-7B-Base",
    tensor_parallel_size=1,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## Online Example

```bash
vllm serve XiaomiMiMo/MiMo-7B-Base \
    --tensor-parallel-size 1 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## Qwen3.8 Flash-Next: dedicated prefill producer

On a dedicated NVIDIA Model Runner V2 Mooncake P2P producer, set
`"qwen38_p_mtp_context_only": true` in the producer's `--speculative-config` JSON,
alongside `"method": "mtp"` and your normal `num_speculative_tokens`. Leave the
flag off on decode and coupled instances. P retains the full target and the
draft input fusion, attention, and cache updates without the draft MoE, final
mixer, head, or autoregressive proposals. The router must issue one-output-token
requests with `do_remote_decode` and a KV transfer ID. Streaming input and prompt
embeddings are not supported on a context-only producer.

Keep lookahead block hashing and EAGLE-style block drop enabled. Mooncake's
hybrid handoff still computes the target through the penultimate prompt token.
P forwards the removed final token only to MTP, so the last draft KV row
conditions on the original prompt, not P's throwaway sampled token.

**Start cold in a separate Store namespace.** With Mooncake Store in a
MultiConnector, set the Store child's `cache_prefix` to
`qwen38-mtp-context-v1/<deployment>` with a fresh, unused deployment suffix. Do
not use the previous Store prefix, migrate old entries, or route the paired
decoder to legacy producers. A native decoder may read the new Store prefix
with the same checkpoint, lookahead hashing, and cache-layout settings. Keep
the optimization off on the decoder.

The optimization is scoped to CUDA, single-module, PP=1 native MTP with BF16
or FP8 checkpoints through the regular weight loader. Other quantization or
prepacked loaders, non-CUDA backends, multi-module MTP and configurations that
need a draft head or adaptive verification keep the full native path. Do not
use the reserved Store prefix on fallback producers. This mode does not change
configured MTP depth, FP8 KV settings, chunked prefill, asynchronous scheduling,
or target/context graph selection.

### With packed NVFP4 target QSA KV

For native Qwen3.8 MTP on a dedicated Mooncake P/D pair, selecting
`--kv-cache-dtype nvfp4` on both endpoints automatically requires context-only
MTP on P. Keep the existing speculative configuration, including its proposal
depth. D still runs full native MTP; the draft main QSA cache defaults to
FP8 E4M3 on both endpoints. This does not change GDN or the raw/compressed QSA
index caches. Context-only P accepts the native ModelOpt mixed checkpoint
configuration through the regular loader, with weight coverage checks on the
retained modules. It fails on incompatible configurations rather than silently
generating drafts.

If the pair also uses Mooncake Store, the runtime derives a separate reserved
namespace from the existing prefix and draft KV dtype. Both endpoints must
start cold in that namespace; do not populate it with old FP8 Store entries
or mix endpoints from before and after changing the dtype. This namespace
does not replace packed transfer ABI validation. Lookahead hashing is still
required, and the physical page size and graph behavior are not overridden by
MTP activation.

## Notes

- MTP only works for model families that support MTP in vLLM.
- `num_speculative_tokens` controls speculative depth. A small value like `1`
  is a good default to start with.
- If your model does not support MTP, use another method such as EAGLE or draft
  model speculation.
- When the MTP head shares the target's unquantized lm_head (most MTP models),
  `"draft_lm_head_quantization": "fp8"` or `"nvfp4"` in `--speculative-config`
  gives the drafter a quantized copy of that head (Model Runner V2). The target
  still verifies with its own head, so outputs are unchanged; only the
  acceptance length can move. It helps most on bandwidth-bound GPUs with a
  large vocabulary. The copy costs about one byte (FP8) or 0.56 bytes (NVFP4)
  per lm_head weight, e.g. 0.6 or 0.35 GiB for a 248k x 2560 head; it is
  counted in the model weights, so with `--kv-cache-memory-bytes` leave room
  for it.
