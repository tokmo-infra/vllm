# Previous Wide EP deployment fixes

These are carryovers from the earlier two-host B300 experiment. They are
independent of the rewritten RDMA MegaMoE data path. See the
[serving guide](../../../docs/serving/hierarchical_mega_moe.md) for backend
selection, API counts, NUMA budgeting and the native tokenizer cache.

## Separate router patch

`router-cache-aware.patch` is preserved byte-for-byte from the old deployment:

- Target: the separate `vllm-project/router` repository at
  `c42ca7d3e65db0c69cbef7cca90bcae8f0e372c4`.
- SHA256: `2899481f0193451a7d60fa5a328ea9bdbcbc9f335e3fcd166103847f640fb588`.
- Apply with `git apply --check` first, then `git apply`, in that checkout.
  Do not apply this patch to vLLM's `rust/` directory.
- `AI_INFRA_HTTP_RENDER_HOMOGENEOUS=1` shares render work only when workers
  have identical tokenizer/template behavior and tracks salted render affinity.
- `AI_INFRA_PD_REUSE_PROMPT_IDS=1` reuses native prefill prompt IDs at decode;
  invalid or missing metadata is an error.
- `AI_INFRA_CACHE_AWARE_REQUIRE_TOKENS=1` rejects cache-aware requests that
  lack token IDs rather than silently degrading the routing policy.
- `AI_INFRA_HTTP_PREPARATION_BUDGET_SECS` optionally overrides preprocessing
  time, within 1–600 seconds. Leave unset to retain the router default. It
  does not fix the separate total-generation deadline.

Prefill must expose its native `/render` path (`--enable-scale-out` in the
previous deployment). Check that cache-aware requests have actual token IDs
and that `missing_tokens` does not grow. This archive preserves the patch;
it does not claim to revalidate or rebuild that separate router revision.

## Process-scoped TCP backlog

`router-listener.c` is also preserved byte-for-byte:
SHA256 `e917f95fe48f793ac8569294bd35edf59c48852734e41f55ee0266970ebb3825`.
The pinned router used a listen backlog of 128 without a CLI override.

```bash
cc -O2 -shared -fPIC -o /tmp/router-listener.so router-listener.c -ldl -pthread
AI_INFRA_LISTEN_BACKLOG=4096 LD_PRELOAD=/tmp/router-listener.so \
    vllm-router "$@"
```

This affects only TCP listeners in the process loading the adapter. Linux
still caps the requested backlog at `net.core.somaxconn`. Verify effective
backlog and listen drops; a larger backlog is not proof that connection or
long-generation timeouts are fixed. Do not change host-wide sysctls as part
of applying this artifact.
