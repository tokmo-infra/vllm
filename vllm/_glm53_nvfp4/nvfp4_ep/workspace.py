# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workspace of the SM120 NVFP4 W4A4 nosync MegaMoE backend.

One symmetric buffer per rank holds the staged inputs, the expert pools, the
intermediate activations, the combine partials, the flags and the counters; the
BF16 output is a separate local tensor. With the ``"auto"`` / ``"nic"``
transports the buffer is an NCCL window and the workspace also owns the NCCL
device communicator the kernel's GIN puts use.
"""

from __future__ import annotations

from typing import Any

import torch

from .contracts import (
    COUNTER_INTS,
    CTR_SLOT,
    ERROR_INDEX,
    buffer_layout,
    pool_row_bytes,
    validate_shape,
)
from .jit import gin_session_module as _gin_session_module
from .jit import kernel_module

GIN_CONTEXTS = 4


class _DeviceBytes:
    """CUDA array interface for the workspace-owned NCCL window."""

    def __init__(self, pointer: int, num_bytes: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (num_bytes,),
            "typestr": "|i1",
            "data": (pointer, False),
            "strides": None,
            "version": 3,
        }


class Nvfp4Workspace:
    """EP collective allocation and rendezvous before CUDA graph capture."""

    def __init__(
        self,
        *,
        group: Any,
        rank: int,
        world_size: int,
        num_experts: int,
        capacity: int,
        top_k: int,
        hidden: int,
        intermediate: int,
        transport: str,
        fast_math: bool = False,
        diagnostic: bool = False,
        stage_fc2_input: bool = False,
        vector_weights: bool = False,
        extended_prefetch: bool = False,
    ) -> None:
        validate_shape(
            world_size, num_experts, capacity, top_k, hidden, intermediate, diagnostic
        )
        if type(extended_prefetch) is not bool:
            raise ValueError("extended_prefetch must be an initialization-time bool")
        self.extended_prefetch = extended_prefetch
        if type(vector_weights) is not bool:
            raise ValueError("vector_weights must be an initialization-time bool")
        self.vector_weights = vector_weights
        if type(stage_fc2_input) is not bool:
            raise ValueError("stage_fc2_input must be a fixed initialization-time bool")
        if stage_fc2_input and (intermediate != 2048 or transport != "p2p"):
            raise ValueError("FC2 input staging requires audited I2048/P2P")
        if torch.cuda.get_device_capability() != (12, 0):
            raise ValueError("NVFP4 candidate requires actual SM120")
        if capacity > 128 and transport != "p2p":
            raise ValueError("capacity256 is P2P-only")
        self.stage_fc2_input = stage_fc2_input
        self.group = group
        self.rank = rank
        self.world_size = world_size
        self.num_experts = num_experts
        self.capacity = capacity
        self.top_k = top_k
        self.hidden = hidden
        self.intermediate = intermediate
        self.device = torch.device("cuda", torch.accelerator.current_device_index())
        self.staged_rows = 0
        self.compile_tokens_per_rank: int | None = None
        self.gin: list[
            int
        ] = []  # [device ncclDevComm*, ncclWindow_t, GIN rank mask, contexts]
        self._gin_type: int | None = None
        self._gin_session: tuple[Any, int] | None = None
        self._symm_handle: Any = None
        self.offsets = buffer_layout(
            world_size,
            num_experts,
            capacity,
            top_k,
            hidden,
            intermediate,
            gin=transport != "p2p",
        )
        num_bytes = self.offsets[0]
        if world_size == 1:
            self.buffer = torch.zeros(num_bytes, dtype=torch.int8, device=self.device)
            self.peer_ptrs = [self.buffer.data_ptr()]
        elif transport == "p2p":
            import torch.distributed._symmetric_memory as symm_mem

            self.buffer = symm_mem.empty(
                num_bytes, dtype=torch.int8, device=self.device
            )
            self._symm_handle = symm_mem.rendezvous(self.buffer, group=group)
            self.buffer.zero_()
            self.peer_ptrs = [int(p) for p in self._symm_handle.buffer_ptrs]
        else:
            self._init_nccl_window(num_bytes, transport)
        # Separate JIT modules have separate cooperative-occupancy grid caches.
        self.module = kernel_module(
            fast_math,
            self._gin_type if self.gin else None,
            stage_fc2_input=stage_fc2_input,
            vector_weights=vector_weights,
            extended_prefetch=extended_prefetch,
        )
        if world_size > 1:
            torch.distributed.barrier(group=group)
        torch.accelerator.synchronize(self.device)
        self.x = self._view(1, (capacity, hidden // 2), torch.uint8)
        self.x_sf = self._view(2, (capacity, hidden // 16), torch.uint8)
        self.topk_idx = self._view(3, (capacity, top_k), torch.int64)
        self.topk_weights = self._view(4, (capacity, top_k), torch.float32)
        self.output = torch.zeros(
            capacity, hidden, dtype=torch.bfloat16, device=self.device
        )
        self.destroyed = False

    # INIT ONLY: collective discovery precedes graph capture.
    def _init_nccl_window(self, num_bytes: int, transport: str) -> None:
        """Use mapped peer pointers, or GIN for unmapped (auto) / all (nic) peers."""
        group, rank, ranks = self.group, self.rank, self.world_size
        torch.distributed.barrier(
            group=group
        )  # ProcessGroupNCCL creates its communicator lazily
        backend = group._get_backend(torch.device("cuda"))
        if not hasattr(backend, "_comm_ptr"):
            raise RuntimeError(
                "the NIC transport needs a ProcessGroupNCCL exposing _comm_ptr"
            )
        session = _gin_session_module()
        handle = int(
            session.megamoe_sm120_gin_session_create(
                int(backend._comm_ptr()), rank, ranks, num_bytes, GIN_CONTEXTS
            )
        )
        self._gin_session = (session, handle)
        base, win, dev_comm, gin_type, _, contexts = (
            int(v) for v in session.megamoe_sm120_gin_session_info(handle)
        )
        peers = [int(p) for p in session.megamoe_sm120_gin_session_peers(handle)]
        mask = 0
        for w in range(ranks):
            if w != rank and (transport == "nic" or peers[w] == 0):
                mask |= 1 << w
        # Pairwise send/receive transports must agree.
        masks: list[Any] = [None] * ranks
        torch.distributed.all_gather_object(masks, mask, group=group)
        for w in range(ranks):
            if (mask >> w & 1) != (masks[w] >> rank & 1):
                raise RuntimeError(
                    f"ranks {rank} and {w} disagree on the NIC transport between them "
                    "(asymmetric NCCL load/store mapping)"
                )
        if mask and dev_comm == 0:
            raise RuntimeError(
                f"ranks {[w for w in range(ranks) if mask >> w & 1]} need NIC, "
                "but the NCCL communicator offers no GIN"
            )
        self.peer_ptrs = [0 if mask >> w & 1 else peers[w] for w in range(ranks)]
        self.gin = [dev_comm, win, mask, contexts] if mask else []
        self._gin_type = gin_type
        self.buffer = torch.as_tensor(_DeviceBytes(base, num_bytes), device="cuda")

    def _view(
        self, slot: int, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        itemsize = torch.empty((), dtype=dtype).element_size()
        numel = 1
        for s in shape:
            numel *= s
        offset = self.offsets[slot]
        return self.buffer[offset : offset + numel * itemsize].view(dtype).view(*shape)

    def launch(
        self,
        y: torch.Tensor,
        weights: Any,
        num_tokens: int,
        swiglu_limit: float,
        tokens_per_item: int | None,
    ) -> None:
        self.module.megamoe_sm120_nvfp4_fused(
            y[:num_tokens],
            self.buffer,
            self.peer_ptrs,
            self.rank,
            weights.w13,
            weights.w2,
            weights.w13_scale,
            weights.w2_scale,
            self.offsets,
            [
                self.num_experts,
                self.top_k,
                self.hidden,
                self.intermediate,
                self.capacity,
            ],
            num_tokens,
            float(swiglu_limit),
            None,
            weights.g1_alphas,
            weights.g2_alphas,
            self.gin,
            tokens_per_item or 8,
            weights.a2_gscale,
            weights.w13_layout,
            weights.scale_layout,
        )

    def stage(self, hidden_states, topk_ids, topk_weights, weights, *, uniform_bound):
        """Current-stream quantize/copy; uniform_bound is outer rank-uniform metadata.

        Never use rank-local rows to select fused versus fallback. This bound
        must be the same metadata-derived padded bound on all participating ranks.
        """
        rows = hidden_states.shape[0]
        if not (0 <= rows <= uniform_bound <= self.capacity):
            raise ValueError("need 0<=local rows<=rank-uniform bound<=capacity")
        if (
            hidden_states.dtype != torch.bfloat16
            or not hidden_states.is_contiguous()
            or tuple(hidden_states.shape) != (rows, self.hidden)
        ):
            raise ValueError("require contiguous BF16 [rows,H]")
        if topk_ids.dtype != torch.int64 or topk_weights.dtype != torch.float32:
            raise ValueError("ids int64, normalized/routed-scaled weights FP32")
        if (
            tuple(topk_ids.shape) != (rows, self.top_k)
            or topk_weights.shape != topk_ids.shape
        ):
            raise ValueError("routing shape mismatch")
        if not topk_ids.is_contiguous() or not topk_weights.is_contiguous():
            raise ValueError("routing tensors must be contiguous")
        for t in (hidden_states, topk_ids, topk_weights, weights.a1_gscale):
            if t.device != self.device:
                raise ValueError("all inputs must share the workspace CUDA device")
        self.module.megamoe_sm120_nvfp4_stage(
            hidden_states, weights.a1_gscale, self.x[:rows], self.x_sf[:rows]
        )
        self.topk_idx[:rows].copy_(topk_ids)
        self.topk_weights[:rows].copy_(topk_weights)
        if rows == 0:
            self.topk_idx[0].fill_(-1)
            self.x[0].zero_()
            self.x_sf[0].zero_()
        self.staged_rows = rows

    def compute(self, weights, *, output=None, swiglu_limit=10.0, tokens_per_item=8):
        """Current-stream launch; output and protocol errors remain on device."""
        n = self.staged_rows
        y = self.output if output is None else output
        if (
            y.dtype != torch.bfloat16
            or not y.is_contiguous()
            or y.shape[1] != self.hidden
            or y.shape[0] < max(n, 1)
            or y.device != self.device
        ):
            raise ValueError(
                "output must be contiguous CUDA BF16 [>=max(local_rows,1),H]"
            )
        self.launch(y, weights, max(n, 1), swiglu_limit, tokens_per_item)
        return y[:n]

    def protocol_error_tensor(self):
        """Device view; inspect only after a finite validation run, outside hot path."""
        return self._view(CTR_SLOT, (COUNTER_INTS,), torch.int32)[
            ERROR_INDEX : ERROR_INDEX + 1
        ]

    # TEARDOWN ONLY: must run after all graph/execution streams finish.
    def destroy(self) -> None:
        """Release the buffer; NIC teardown collectively deregisters the window."""
        if self.destroyed:
            return
        self.destroyed = True
        torch.accelerator.synchronize(self.device)
        self.x = self.x_sf = self.topk_idx = self.topk_weights = None
        self.buffer = None
        self._symm_handle = None
        session, self._gin_session = self._gin_session, None
        if session is not None:
            session[0].megamoe_sm120_gin_session_close(session[1])


__all__ = ["Nvfp4Workspace", "buffer_layout", "kernel_module", "pool_row_bytes"]
