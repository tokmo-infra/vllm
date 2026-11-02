# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stream-ordered transport contract for hierarchical MegaMoE payloads."""

from typing import Protocol

import torch
import torch.distributed as dist


class MegaMoETransport(Protocol):
    """Exchange peer-major rows, completing on the caller's CUDA stream.

    Implementations must retain send storage until the exchange completes and
    support the same collective order on every lane member. CUDA graph users
    must initialize connections and register memory before capture.
    """

    def exchange(
        self,
        receive: torch.Tensor,
        send: torch.Tensor,
        receive_counts: list[int] | None = None,
        send_counts: list[int] | None = None,
    ) -> None: ...


class ProcessGroupTransport:
    """NCCL on CUDA, Gloo for CPU correctness tests; network policy is external."""

    def __init__(self, group: dist.ProcessGroup):
        self.group = group

    def exchange(self, receive, send, receive_counts=None, send_counts=None):
        work = dist.all_to_all_single(
            receive,
            send,
            output_split_sizes=receive_counts,
            input_split_sizes=send_counts,
            group=self.group,
            async_op=True,
        )
        # NCCL wait adds a dependency to the current CUDA stream. Gloo waits
        # for host completion. Do not synchronize the device in the hot path.
        work.wait()
