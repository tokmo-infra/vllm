# Experimental producer hot-tail admission

This GLM branch retains the previously tested P-side Mamba tail copy-on-write
(CoW) admission candidate as an **opt-in experiment**, not a default deployment
or a generally validated preemption fix. Existing image tags and running services
are not changed by this source commit. An image containing this source is required.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION` | `0` | Exact string `1` enables eligibility checking and the additional admission budget on P. |
| `VLLM_HOT_TAIL_ADMISSION_DIAGNOSTICS` | `0` | Exact string `1` permits the first enrolled, deferred and spent event per scheduler, at most three lines. Does not enable the experiment. |

For a newly built experimental P container, pass:

```bash
docker run \
  -e VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION=1 \
  -e VLLM_HOT_TAIL_ADMISSION_DIAGNOSTICS=0 \
  <image-containing-this-source> <existing-P-arguments>
```

Omit both variables or set them to `0` to use native scheduling. They are read at
scheduler creation, not hot-reloaded. Keep them unset on D. The diagnostics flag
alone cannot activate budgeting. The baseline deployment commands remain unchanged.

## Scope and asynchronous behavior

The experiment forecasts a future destination block for each eligible Mamba group
before a virgin, purely external asynchronous load is admitted. After publication,
the original cached partial tail needs CoW even when logical sequence growth is
zero. Reserving its future budget can avoid admitting other work that consumes
the last free blocks. Group counts are derived from the actual managers, not
hard-coded to three.

Only the verified producer configuration is eligible: asynchronous scheduling,
prefix caching, align-mode Mamba, no replay/shared checkpoint/reprefill, single
DCP/PCP rank, and the exact supported MultiConnector children:
Mooncake PD producer plus asynchronous compact Mooncake Store. Consumers and
unsupported/custom layouts retain native scheduling. Only same-physical-tail-block
initial loads are enrolled; local hits, resumed/preempted requests and cross-block
growth are not generalized into this proof.

The ledger stores scalar metadata. Native allocation, connector, reference-count,
lease, Store and GPU-fence operations are unchanged; no GPU synchronization is
added and async/FP8/MTP/CUDA graphs are not disabled. A reservation-only rejection
skips the current request so the owner can progress, rather than triggering the
normal physical-shortage victim selection. Native physical or legacy-budget
shortages still follow their original paths. Current/native/legacy-covered demand
is deduplicated. Finished, aborted, preempted and failed-load owners lose only
their experiment metadata.

With the switch off, no helper is created or loaded and no additional predictor
or connector call runs. With it on, this is deliberately **not** a global
no-preemption guarantee or a proof of general scheduling liveness.

## Faults and monitoring

Initialization logs `P_HOT_TAIL_BUDGET init ... enabled=True/False`. Explicit
eligibility rejection leaves the native path active. Metadata-contract failures
latch the experiment off and emit bounded
`P_HOT_TAIL_BUDGET ... stop_required=true` logs; original native exceptions are not
swallowed. The external test controller must stop and invalidate such a trial.
Failing open is a research safeguard, not a recovered budget proof.

Do not infer safety from absent progress logs. Do not remove in-use shared
Tokenizer pools or epoch files while testing. Return to the native path by
removing the switch and restarting the affected P services through the normal
drain/new-engine-ID procedure, not by changing environment variables in place.

## Previously measured result

Same hosts, original SWE8w SHA-256
`f6fb8de6deee44d727affbc6014b61e7064b0719facfb13fab172305b0475129`,
C230/U345/N345, 25,512 planned turns, HCA affinity enabled; 11 complete minutes
after 55% in each score window:

| Metric | Native baseline | Candidate first trial | Candidate repeat |
| --- | ---: | ---: | ---: |
| Total TPM / 10,000 | 5429.736 | 5352.572 | 5372.659 |
| Relative total TPM | — | -1.42% | -1.05% |
| TTFT mean / s | 6.372 | 6.327 | 6.394 |
| TTFT P99 / s | 13.613 | 14.045 | 14.107 |
| TPOT P99 / ms | 25.578 | 25.620 | 25.427 |
| Whole-run P/D preemptions | 18/0 | 0/0 | 0/0 |
| Whole-run HTTP failures | 0 | 0 | 0 |

Runs: baseline `run-61d41ad9e1ee`, first `run-f1f42e81641c`, repeat
`run-9f069bf546dc`. Fewer preemptions did **not** deliver an overall performance
benefit. Both candidate deployments were rolled back. Retaining source behind a
default-off switch does not change that decision or claim stable gains. These
model runs validated the frozen candidate against its historical runtime; the
source integration into the later branch additionally preserves existing draft
and priority-scheduler changes and requires its own CPU regressions. No new model
benchmark, image rebuild or service deployment is implied.
