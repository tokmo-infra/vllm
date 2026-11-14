# Kimi K3.1 tensor-core RecoverSSM verification

The NVIDIA specialized decode can opt into the `KdaFusedTC` kernel adapted from
[vLLM PR #66040](https://github.com/vllm-project/vllm/pull/66040), pinned to
`3b73e7acef8327103f8b19198e58ce047240902b`. It reuses the specialized decode's
existing deferred replay records, state slots, helpers and CUDA-graph launches.

Enable it alongside the existing decode backend:

```bash
--additional-config '{"decode_backend":"latency_specialized","kda_fused_tc":true}'
```

The option defaults to `false`. Set it to `false` to restore the existing KDA
kernel selection. It does not enable the stock-path changes from the upstream
PR or change the speculative draft depth, cache precision, HiSparse capacity,
tensor parallelism or decode context parallelism.

The candidate targets B300 TP8/DCP8/EP8 with DSpark4 (five verification tokens),
BF16 convolution state and FP32 recurrent state. Selection still requires a
deferred RecoverSSM step and the existing fused-kernel batch threshold. The
tensor-core kernel additionally requires a streamable convolution window of
`query_tokens + 2`, a contiguous inner FP32 state matrix, a BF16 F_B matrix and
a shared-memory footprint within its limit. Other shapes retain the existing
kernel selection. For example, six-token verification with a contiguous
`[slot, channel, window]` convolution state exceeds this kernel's shared-memory
limit.

The state remains FP32. State dot products use the upstream three-term TF32
split, which changes floating-point accumulation order; identical generated
tokens are not implied. Validate long-context model accuracy and real acceptance
length before enabling it for serving.

The existing B200 Kimi test step in `.buildkite/test_areas/models_basic.yaml`
selects the specialized KDA suite. Its added cases cover TP8/Q5 batches of 16,
64, 224 and 256 requests, both gate modes and convolution layouts, padded state-page strides,
null slots, CUDA-graph replay with updated metadata and unsupported-shape
fallbacks. Run on a Blackwell GPU:

```bash
.venv/bin/python -m pytest -q \
  tests/models/kimi_k3/nvidia/specialized_decode/test_specialized_decode_kda_fused.py
```

The host-only policy suite uses meta tensors and needs no CUDA or CuTe runtime:

```bash
.venv/bin/python -m pytest --noconftest -q \
  tests/models/kimi_k3/nvidia/specialized_decode/test_kda_tensorcore_policy.py
```

Compare with the option disabled at the same concurrency, request corpus,
context lengths, cache budgets and real speculative acceptance. Report output
tokens/s, TPOT, GPU memory and errors separately from kernel timings.
