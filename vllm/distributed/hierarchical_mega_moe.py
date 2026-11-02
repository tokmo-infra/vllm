# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, stream-ordered network tiles around native NVLink MegaMoE.

The network owns peer-major wire slots; the compute stream alone owns the
symmetric MegaMoE workspace. Every source row contributes one partial per
island. Only the source adds shared experts and records logical routing loads.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

from vllm.distributed.megamoe_transport import MegaMoETransport, ProcessGroupTransport


def has_uniform_token_counts(tokens, counts, world_size):
    """Use host DP metadata, never a device read, to select graph-safe routing."""
    return (
        counts is not None
        and counts.device.type == "cpu"
        and counts.numel() == world_size
        and counts.tolist() == [tokens] * world_size
    )


@dataclass(frozen=True)
class IslandTopology:
    nodes: tuple[tuple[int, ...], ...]
    rank: int

    @classmethod
    def from_hosts(cls, hosts: Sequence[str], rank: int):
        if not 0 <= rank < len(hosts):
            raise ValueError("Invalid EP rank")
        labels = dict.fromkeys(hosts)
        nodes = tuple(
            tuple(i for i, h in enumerate(hosts) if h == label) for label in labels
        )
        if len({len(node) for node in nodes}) != 1:
            raise ValueError("Expected equal-sized NVLink islands")
        return cls(nodes, rank)

    @property
    def local_size(self):
        return len(self.nodes[0])

    @property
    def world_size(self):
        return sum(map(len, self.nodes))

    @property
    def node(self):
        return next(i for i, node in enumerate(self.nodes) if self.rank in node)

    @property
    def lane(self):
        return self.nodes[self.node].index(self.rank)

    @property
    def lane_ranks(self):
        return tuple(sorted(node[self.lane] for node in self.nodes))

    @property
    def lane_nodes(self):
        membership = {rank: i for i, node in enumerate(self.nodes) for rank in node}
        return tuple(membership[rank] for rank in self.lane_ranks)

    def tables(self, device):
        owner = [0] * self.world_size
        local = [0] * self.world_size
        for node, ranks in enumerate(self.nodes):
            for lane, rank in enumerate(ranks):
                owner[rank], local[rank] = node, lane
        return tuple(
            torch.tensor(values, device=device, dtype=torch.int64)
            for values in (owner, local, self.lane_nodes)
        )


@dataclass
class HierarchicalGroups:
    topology: IslandTopology
    local_group: dist.ProcessGroup
    lane_group: dist.ProcessGroup
    world_group: dist.ProcessGroup

    @classmethod
    def create(cls, global_ranks, hosts, rank, backend, world_group=None):
        if list(global_ranks) != sorted(set(global_ranks)):
            raise ValueError("EP global ranks must be ordered and unique")
        topology = IslandTopology.from_hosts(hosts, rank)
        local = None
        lane = None
        # All members create overlapping groups in the same order.
        peer_sets = list(topology.nodes) + [
            tuple(sorted(node[i] for node in topology.nodes))
            for i in range(topology.local_size)
        ]
        for index, peers in enumerate(peer_sets):
            if rank in peers:
                group = dist.new_group(
                    ranks=[global_ranks[p] for p in peers],
                    backend=backend,
                    timeout=timedelta(seconds=120),
                    use_local_synchronization=True,
                )
                if index < len(topology.nodes):
                    local = group
                else:
                    lane = group
        assert local is not None and lane is not None
        return cls(topology, local, lane, world_group or dist.group.WORLD)

    def close(self):
        dist.destroy_process_group(self.lane_group)
        dist.destroy_process_group(self.local_group)


_GROUPS: dict[tuple, HierarchicalGroups] = {}


def get_hierarchical_groups(device_group, cpu_group):
    key = (device_group, cpu_group)
    if key in _GROUPS:
        return _GROUPS[key]
    from vllm.distributed.parallel_state import in_the_same_node_as

    size = dist.get_world_size(device_group)
    labels = [""] * size
    for leader in range(size):
        if labels[leader]:
            continue
        peers = in_the_same_node_as(cpu_group, leader)
        if not peers[leader]:
            raise RuntimeError("Cannot discover host IPC membership")
        for peer, same in enumerate(peers):
            if same:
                labels[peer] = str(leader)
    result = HierarchicalGroups.create(
        dist.get_process_group_ranks(device_group),
        labels,
        dist.get_rank(device_group),
        dist.get_backend(device_group),
        device_group,
    )
    _GROUPS[key] = result
    return result


def destroy_hierarchical_groups():
    for groups in _GROUPS.values():
        groups.close()
    _GROUPS.clear()


@dataclass
class WireSlot:
    send: torch.Tensor
    receive: torch.Tensor
    partial: torch.Tensor
    returned: torch.Tensor


class HierarchicalMegaMoE:
    """Route globally owned experts through equal, host-local NVLink islands.

    Uniform batches use fixed peer-major rows, including masked rows. Uneven
    eager batches pad to the global maximum before entering the same protocol.
    A native callback returns BF16, already gate-weighted island partials.
    """

    def __init__(
        self,
        groups,
        num_experts,
        max_tokens_per_chunk,
        wire_dtype="bfloat16",
        fused_pack=False,
        source_chunk_size=0,
        transport: MegaMoETransport | None = None,
    ):
        if not 0 < num_experts < 2**31 or num_experts % groups.topology.world_size:
            raise ValueError("Expert count must fit int32 and divide EP size")
        if max_tokens_per_chunk <= 0 or source_chunk_size < 0:
            raise ValueError("Invalid chunk capacity")
        if wire_dtype not in ("bfloat16", "mxfp8"):
            raise ValueError("Wire dtype must be bfloat16 or mxfp8")
        self.groups = groups
        self.num_experts = num_experts
        self.chunk = max_tokens_per_chunk
        self.wire_dtype = wire_dtype
        self.fused_pack = fused_pack
        self.source_chunk_size = source_chunk_size
        self.transport = transport or ProcessGroupTransport(groups.lane_group)
        self.last_dispatch_tokens = 0
        self._tables: dict[torch.device, tuple[torch.Tensor, ...]] = {}

    def forward(
        self,
        hidden,
        weights,
        ids,
        local_moe: Callable,
        *,
        fixed_capacity=False,
        local_moe_mxfp8: Callable | None = None,
    ):
        if (
            hidden.ndim != 2
            or ids.ndim != 2
            or weights.shape != ids.shape
            or hidden.shape[0] != ids.shape[0]
            or ids.shape[1] == 0
        ):
            raise ValueError("Expected hidden[T,H] and routing[T,K]")
        capturing = hidden.is_cuda and torch.cuda.is_current_stream_capturing()
        if capturing and not fixed_capacity:
            raise RuntimeError("CUDA graphs require uniform padded token counts")
        if hidden.device not in self._tables:
            if capturing:
                raise RuntimeError("Warm up topology tables before graph capture")
            self._tables[hidden.device] = self.groups.topology.tables(hidden.device)
        source_tokens = hidden.shape[0]
        if not fixed_capacity:
            maximum = torch.tensor(
                source_tokens, device=hidden.device, dtype=torch.int64
            )
            dist.all_reduce(
                maximum, op=dist.ReduceOp.MAX, group=self.groups.world_group
            )
            capacity = int(maximum.item())
            padding = capacity - source_tokens
            hidden = torch.nn.functional.pad(hidden, (0, 0, 0, padding))
            weights = torch.nn.functional.pad(weights, (0, 0, 0, padding))
            ids = torch.nn.functional.pad(ids, (0, 0, 0, padding), value=-1)
        tokens, width = hidden.shape
        nodes = len(self.groups.topology.nodes)
        self.last_dispatch_tokens = nodes * tokens
        if tokens == 0:
            return torch.empty_like(hidden)
        capacity = min(tokens, self.source_chunk_size or tokens)
        if self.wire_dtype == "mxfp8":
            x_bytes = width + width // 32
        else:
            x_bytes = width * hidden.element_size()
        gate_offset = (x_bytes + 3) // 4 * 4
        route_offset = gate_offset + 4 * ids.shape[1]
        row_bytes = route_offset + 4 * ids.shape[1]
        output = torch.empty_like(hidden)
        storage = self._slot(hidden, nodes * capacity, row_bytes)
        # The upstream MegaMoE kernel occupies the full GPU and requires all
        # local peers to progress. Overlapping NCCL kernels can deadlock it.
        # Keep network and native compute ordered, including under graph replay.
        for start in range(0, tokens, capacity):
            end = min(tokens, start + capacity)
            rows = nodes * (end - start)
            slot = WireSlot(
                *(
                    value[:rows]
                    for value in (
                        storage.send,
                        storage.receive,
                        storage.partial,
                        storage.returned,
                    )
                )
            )
            self._dispatch(hidden[start:end], weights[start:end], ids[start:end], slot)
            self._compute(
                slot,
                width,
                x_bytes,
                gate_offset,
                route_offset,
                local_moe,
                local_moe_mxfp8,
            )
            self._return(slot, output[start:end])
        return output[:source_tokens]

    @staticmethod
    def _slot(hidden, rows, row_bytes):
        send = torch.empty((rows, row_bytes), dtype=torch.uint8, device=hidden.device)
        partial = hidden.new_empty((rows, hidden.shape[1]))
        return WireSlot(
            send, torch.empty_like(send), partial, torch.empty_like(partial)
        )

    def _dispatch(self, hidden, weights, ids, slot):
        if hidden.is_cuda and self.fused_pack:
            from vllm.distributed.megamoe_rdma_ops import pack_fixed

            pack_fixed(
                hidden,
                weights,
                ids,
                self._tables[hidden.device],
                self.num_experts,
                self.groups.topology.world_size,
                self.wire_dtype,
                out=slot.send,
            )
        else:
            slot.send.copy_(self._pack_torch(hidden, weights, ids))
        self.transport.exchange(slot.receive, slot.send)

    def _compute(self, slot, width, x_bytes, gate_offset, route_offset, local, direct):
        payload = slot.receive[:, :x_bytes]
        if self.wire_dtype == "mxfp8" and direct is not None:
            callback = direct
        else:
            callback = local
            if self.wire_dtype == "mxfp8":
                from vllm.distributed.megamoe_rdma_ops import unpack_mxfp8

                payload = unpack_mxfp8(payload, width)
            else:
                payload = payload.view(slot.partial.dtype)
        gates = slot.receive[:, gate_offset:route_offset].view(torch.float32)
        routes = slot.receive[:, route_offset:].view(torch.int32)
        for start in range(0, payload.shape[0], self.chunk):
            end = min(payload.shape[0], start + self.chunk)
            slot.partial[start:end] = callback(
                payload[start:end], gates[start:end], routes[start:end]
            )

    def _return(self, slot, output):
        self.transport.exchange(slot.returned, slot.partial)
        if output.is_cuda and self.fused_pack:
            from vllm.distributed.megamoe_rdma_ops import combine_fixed

            combine_fixed(
                slot.returned,
                output.shape[0],
                output.shape[1],
                len(self.groups.topology.nodes),
                out=output,
            )
        else:
            output.copy_(
                slot.returned.view(-1, *output.shape).sum(0, dtype=torch.float32)
            )

    def _pack_torch(self, hidden, weights, ids):
        topology = self.groups.topology
        owner, local, destinations = self._tables[hidden.device]
        per_rank = self.num_experts // topology.world_size
        valid = (ids >= 0) & (ids < self.num_experts)
        ranks = ids.clamp(0, self.num_experts - 1).long() // per_rank
        keep = valid[None] & (owner[ranks][None] == destinations[:, None, None])
        local_ids = local[ranks] * per_rank + ids.remainder(per_rank)
        routes = torch.where(keep, local_ids[None], -1).to(torch.int32)
        gates = torch.where(keep, weights[None], 0).float()
        if self.wire_dtype == "mxfp8":
            from vllm.distributed.megamoe_rdma_ops import pack_mxfp8

            activations = pack_mxfp8(hidden)
        else:
            activations = hidden.contiguous().view(torch.uint8)
        activations = torch.nn.functional.pad(
            activations, (0, -activations.shape[1] % 4)
        )
        return torch.cat(
            (
                activations.repeat(len(topology.nodes), 1),
                gates.flatten(0, 1).view(torch.uint8),
                routes.flatten(0, 1).view(torch.uint8),
            ),
            dim=1,
        )
