# Adaptive Verification

Speculative decoding buys fewer decode steps with more compute. At batch size 1 that is a good trade: the GPU is memory-bound with spare compute, so the extra draft tokens are close to free. At batch size 256 it is a much more delicate one. Draft tokens now compete with real tokens for the same compute, and every rejected token is compute wasted; with enough of them, throughput drops.

That matters because per-position acceptance decays fast. While the GPU is memory-bound that slot is effectively free and worth the gamble; once it saturates the gamble has a real throughput cost. The crossover moves with load and with workload-dependent acceptance rates, so no static `num_speculative_tokens` is right across concurrencies.

Adaptive verification decides per step how much of the draft to verify instead. Every (request, position) draft slot is scored by its *survival probability*, the running product of that request's per-position acceptance estimates, and the highest-scoring slots are admitted until a global budget is spent. Slots compete across requests: position 5 of a confident request can outrank position 1 of a doubtful one, so one request keeps its full block while another could be trimmed after a token or two.

The budget itself comes from a cost model profiled at startup. vLLM measures what a step costs at each shape, then picks the token count that maximizes expected accepted tokens per second.

The practical effect is that one configuration holds up across the whole load range, which removes most of the need to tune `num_speculative_tokens` per deployment.

## Support

Adaptive verification is supported for model-based drafters (EAGLE, MTP, DFlash, DSpark and standalone draft models). The per-position acceptance estimates come from one of two sources:

- **Confidence head**: DSpark checkpoints that ship a trained confidence head use its predictions directly.
- **Online acceptance estimator**: every other drafter predicts acceptance from its draft logits with a small logistic model, recalibrated during serving against the target's verification results.

## Usage

It is off by default. Enable it in the speculative config with `enable_adaptive_verification`. For example, with DSpark:

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --tokenizer-mode deepseek_v4 --trust-remote-code \
  --speculative-config '{
    "method": "dspark",
    "model": "deepseek-ai/DeepSeek-V4-Flash-DSpark",
    "num_speculative_tokens": 7,
    "draft_sample_method": "probabilistic",
    "enable_adaptive_verification": true
  }'
```

Or with DFlash on MiMo-V2.5-Pro, whose drafter ships in the checkpoint's `dflash/` subdirectory:

```bash
hf download XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash --local-dir MiMo-V2.5-Pro-FP4-DFlash

vllm serve MiMo-V2.5-Pro-FP4-DFlash \
  --tensor-parallel-size 4 --trust-remote-code --max-model-len auto \
  --speculative-config '{
    "method": "dflash",
    "model": "MiMo-V2.5-Pro-FP4-DFlash/dflash",
    "num_speculative_tokens": 8,
    "enable_adaptive_verification": true
  }'
```

Set `enable_adaptive_verification: false` to verify the full block for every request.

### Discrete verification lengths

To choose a verification prefix from a small set while keeping the draft width
fixed, set `adaptive_verification_candidate_steps`:

```json
{
  "method": "dspark",
  "num_speculative_tokens": 7,
  "enable_adaptive_verification": true,
  "adaptive_verification_candidate_steps": [1, 3, 5, 7]
}
```

Each worker batch selects one prefix length that maximizes expected output tokens
divided by the profiled draft-plus-verification cost. It verifies that prefix for
every request with sufficient scheduled drafts; shorter drafts are not extended.
The drafter still computes all seven candidates. Graph capture and KV capacity
remain sized for the maximum, and DP continues to synchronize padded token counts.
This does not enable the DP-incompatible dynamic drafting schedule.

The list must be nonempty, strictly increasing, and contain lengths from 1 through
`num_speculative_tokens`. If even the smallest prefix exceeds the rejection
sampler's single-chunk capacity, verification falls back to zero drafts for that
batch. Omitting the list retains the unrestricted per-request budget policy.
The cost model is an estimate; measure end-to-end performance against a fixed
length on the deployment workload before treating the policy as an optimization.

## Requirements and limitations

- The attention backend must tolerate device-decided query lengths, since the CPU lengths only bound them from above. Backends that plan off the CPU lengths are excluded by the attention selector, and rejected at startup for models that hard-wire their backend.
- Full cudagraphs are required: step costs are profiled from captured graphs, so `--enforce-eager` is rejected at startup.
- Not supported with LoRA (the per-token LoRA mapping is built from CPU-side boundaries) or pipeline parallelism (cost curves and confidences exist only on the last rank).
- Without a confidence head, `use_local_argmax_reduction` is rejected. The estimator needs the full draft logits, which local argmax reduction never materializes.

## Tuning the cost profile

Step costs are profiled against a synthetic KV context, 8192 tokens by default. Deployments serving much longer contexts may want to raise it so the profiled step reads a more realistic amount of cache (this matters a bit less for sparse attention models like DeepSeek-v4 since the cheap indexer is the main cost that scales with context length).

```bash
export VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=131072
```
