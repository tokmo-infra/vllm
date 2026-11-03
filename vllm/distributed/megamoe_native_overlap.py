# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-stable RDMA storage for the native streaming MegaMoE launch."""

import math

import torch
import torch.distributed as dist

from vllm.distributed.megamoe_rdma_ops import combine_compact, pack_fixed
from vllm.distributed.roce.transport import PeerBuffer


def streaming_shape(tokens, nodes, experts, target_chunk):
    """Bound virtual experts while keeping small batches unpadded."""
    limit = min(32 // nodes, 2048 // experts)
    if limit < 1:
        raise ValueError("Native streaming exceeds the virtual expert capacity")
    chunks = 1
    while chunks * 2 <= limit and tokens > chunks * target_chunk:
        chunks *= 2
    capacity = tokens if tokens <= 64 else math.ceil(tokens / chunks / 64) * 64
    return chunks, capacity


def streaming_row_bytes(width, topk):
    # Native TMA pulls require every token's base address to be 16-byte aligned.
    return math.ceil((width + width // 32 + topk * 8) / 16) * 16


class NativeInputStorage:
    def __init__(self, groups, transport, chunks, capacity, width, topk, slot):
        self.nodes = len(groups.topology.nodes)
        self.row_bytes = streaming_row_bytes(width, topk)
        self.inputs = []
        self.input_descriptors = torch.zeros(
            (groups.topology.local_size, self.nodes * chunks, 3),
            device="cuda",
            dtype=torch.int64,
        )
        for index in range(chunks):
            size = self.nodes * capacity * self.row_bytes
            send, receive = (PeerBuffer(groups.local_group, size) for _ in range(2))
            wire_send, wire_receive = (
                b.region[:size].view(self.nodes * capacity, self.row_bytes)
                for b in (send, receive)
            )
            transfer = transport.register_transfer(
                wire_send,
                wire_receive,
                send.region,
                receive.region,
                dict(phase="stream-dispatch", chunk=index, slot=slot),
                acquire_before_send=True,
            )
            flags = [0] * self.nodes
            for peer, signals in transfer.peer_signals.items():
                flags[peer] = signals.receive_done
            peer_flags = [None] * groups.topology.local_size
            dist.all_gather_object(peer_flags, flags, group=groups.local_group)
            for lane in range(groups.topology.local_size):
                for peer in range(self.nodes):
                    self.input_descriptors[lane, peer * chunks + index] = torch.tensor(
                        [
                            peer_flags[lane][peer],
                            receive.buffer_ptrs[lane]
                            + peer * capacity * self.row_bytes,
                            self.row_bytes,
                        ],
                        device="cuda",
                    )
            self.inputs.append((transfer, send, receive))


class NativeReturnStorage:
    def __init__(self, groups, transport, tokens, width, slot):
        self.nodes, self.tokens = len(groups.topology.nodes), tokens
        quantum = max(64, 65536 // math.gcd(width * 2, 65536))
        self.return_chunk = min(quantum, tokens)
        if tokens % self.return_chunk:
            self.return_chunk = tokens
        output_bytes = self.nodes * self.tokens * width * 2
        region_bytes = math.ceil(output_bytes / 65536) * 65536
        tail = (self.tokens - self.return_chunk) * width * 2
        self.send = PeerBuffer(groups.local_group, region_bytes + tail)
        self.receive = PeerBuffer(groups.local_group, region_bytes + tail)
        self.partial, self.returned = (
            b.region[:output_bytes]
            .view(torch.bfloat16)
            .view(self.nodes * self.tokens, width)
            for b in (self.send, self.receive)
        )
        self.return_descriptors = torch.zeros(
            (self.nodes * self.tokens // self.return_chunk, 4),
            device="cuda",
            dtype=torch.int64,
        )
        self.returns = []
        for index in range(self.tokens // self.return_chunk):
            offset = index * self.return_chunk * width * 2
            a, b = (
                v.region[offset : offset + region_bytes]
                for v in (self.send, self.receive)
            )
            transfer = transport.register_transfer(
                a[:output_bytes].view(torch.bfloat16).view_as(self.partial),
                b[:output_bytes].view(torch.bfloat16).view_as(self.partial),
                a,
                b,
                dict(phase="stream-return", chunk=index, slot=slot),
                copy_self=False,
                acquire_before_send=True,
            )
            for peer, signals in transfer.peer_signals.items():
                self.return_descriptors[
                    peer * (self.tokens // self.return_chunk) + index, :2
                ] = torch.tensor([signals.request, signals.send_rows], device="cuda")
            self.returns.append(transfer)
        self.output = torch.empty(
            (self.tokens, width), dtype=torch.bfloat16, device="cuda"
        )
        self.row_map = (
            torch.arange(self.tokens, device="cuda", dtype=torch.int32)
            .expand(self.nodes, -1)
            .contiguous()
        )
        self.shared_output = torch.empty_like(self.output)


class NativeOverlapPlan:
    """One shape shared by sequential layers; all readers join before reuse.

    Input groups have independent receive credits. Native FC2 reductions write
    directly to the registered return buffer and publish each finished chunk.
    These buffers remain owned by the transport until it deregisters the MRs.
    """

    def __init__(
        self,
        groups,
        transport,
        experts,
        chunks,
        capacity,
        width,
        topk,
        dispatch_slot=0,
        return_slot=0,
    ):
        self.groups, self.transport = groups, transport
        self.experts, self.chunks, self.capacity = experts, chunks, capacity
        self.width, self.topk = width, topk
        self.nodes = len(groups.topology.nodes)
        self.tokens = chunks * capacity
        input_key = (experts, chunks, capacity, width, topk, dispatch_slot)
        inputs = transport.native_inputs.get(input_key)
        if inputs is None:
            inputs = NativeInputStorage(
                groups, transport, chunks, capacity, width, topk, dispatch_slot
            )
            transport.native_inputs[input_key] = inputs
        self.inputs = inputs.inputs
        self.input_descriptors = inputs.input_descriptors
        output_key = (self.tokens, width, return_slot)
        outputs = transport.native_outputs.get(output_key)
        if outputs is None:
            outputs = NativeReturnStorage(
                groups, transport, self.tokens, width, return_slot
            )
            transport.native_outputs[output_key] = outputs
        self.partial, self.returned = outputs.partial, outputs.returned
        self.returns = outputs.returns
        self.return_descriptors = outputs.return_descriptors
        self.return_chunk = outputs.return_chunk
        self.output, self.shared_output = outputs.output, outputs.shared_output
        self.row_map = outputs.row_map
        self.scratch = None

    def scheduler_workspace(self, buffer):
        if self.scratch is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up native scheduling before graph capture")
            local_experts = buffer.num_experts // self.groups.topology.local_size
            blocks = (
                math.ceil(
                    buffer.num_max_tokens_per_rank
                    * self.groups.topology.local_size
                    * min(self.topk, local_experts)
                    / 8
                )
                + local_experts
            )
            self.scratch = torch.empty(
                (4, blocks + 1), device="cuda", dtype=torch.int64
            )
            # A fast lane must not enter the resident NVLink kernel while a
            # peer is still allocating a new plan. CUDA allocation/IPC setup
            # may wait for peer device work, creating a cycle with that kernel.
            # This rendezvous runs only during eager shape warmup, never replay.
            torch.cuda.synchronize()
            dist.barrier(
                group=self.groups.cpu_group
                if self.groups.cpu_group is not None
                else self.groups.local_group
            )
        return self.scratch

    def dispatch(self, hidden, gates, ids, tables):
        for transfer, _, _ in self.inputs:
            transfer.wait_send()
            transfer.reserve()
        for index, (transfer, _, _) in enumerate(self.inputs):
            rows = slice(index * self.capacity, (index + 1) * self.capacity)
            pack_fixed(
                hidden[rows],
                gates[rows],
                ids[rows],
                tables,
                self.experts,
                self.groups.topology.world_size,
                "mxfp8",
                out=transfer.send,
                row_alignment=16,
            )
            transfer.publish_fixed()
        for transfer in self.returns:
            transfer.wait_send()
            transfer.reserve()

    def combine(self, shared):
        for transfer in self.returns:
            transfer.wait_receive(fixed=True)
        output = torch.empty_like(self.output)
        combine_compact(
            self.returned,
            self.partial,
            self.row_map,
            output,
            self.transport.rank,
            "bfloat16",
            shared=shared,
        )
        for transfer, _, _ in self.inputs:
            transfer.wait_receive(fixed=True)
        return output
