# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-host, graph-replayable GPUDirect RDMA with host progress threads.

Compute uses the caller's stream. Optional pinned-host send staging uses a
copy stream joined by CUDA events. Stream memory writes publish requests and
stream memory waits guard consumers without a communication kernel. The proxy
owns RC QPs, exchanges per-epoch credits, and drains send and receive CQs.
"""

import functools
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.distributed as dist

_ALIGNMENT = 65536


class _Endpoint(Protocol):
    def info(self) -> dict[str, Any]: ...
    def connect(self, remote: dict[str, Any]) -> None: ...
    def register_job(
        self,
        send: int,
        receive: int,
        allocation_bytes: int,
        peer_bytes: int,
        row_bytes: int,
    ) -> dict[str, Any]: ...
    def connect_job(self, id: int, remote: dict[str, Any]) -> None: ...
    def submit(self, id: int, stream: int, rows: int) -> None: ...
    def wait(self, id: int, stream: int, rows: int) -> None: ...
    def set_tracing(self, enabled: bool) -> None: ...
    def trace_snapshot(self) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


class _PeerEndpoints:
    """Independent RC credits per peer, sharing disjoint peer-major storage."""

    def __init__(self, rank):
        self.rank = rank
        self.peers: dict[int, _Endpoint] = {}

    def info(self):
        return {peer: endpoint.info() for peer, endpoint in self.peers.items()}

    def connect(self, remote):
        for peer, endpoint in self.peers.items():
            endpoint.connect(remote[peer][self.rank])

    def register_job(self, send, receive, allocation_bytes, peer_bytes, row_bytes):
        metadata = {
            peer: endpoint.register_job(
                send, receive, allocation_bytes, peer_bytes, row_bytes
            )
            for peer, endpoint in self.peers.items()
        }
        ids = {entry["id"] for entry in metadata.values()}
        if len(ids) != 1:
            raise RuntimeError("Peer transfer registration order differs")
        return dict(
            id=ids.pop(),
            peers=metadata,
            relaxed_ordering=all(m["relaxed_ordering"] for m in metadata.values()),
            host_staging=any(m["host_staging"] for m in metadata.values()),
        )

    def connect_job(self, id, remote):
        for peer, endpoint in self.peers.items():
            endpoint.connect_job(id, remote[peer]["peers"][self.rank])

    def submit(self, id, stream, rows):
        for peer, endpoint in self.peers.items():
            endpoint.submit(id, stream, rows + peer * 4)

    def wait(self, id, stream, rows):
        for peer, endpoint in self.peers.items():
            endpoint.wait(id, stream, rows + peer * 4)

    def set_tracing(self, enabled):
        for endpoint in self.peers.values():
            endpoint.set_tracing(enabled)

    def trace_snapshot(self):
        return [
            record | {"peer": peer}
            for peer, endpoint in self.peers.items()
            for record in endpoint.trace_snapshot()
        ]

    def close(self):
        for endpoint in self.peers.values():
            endpoint.close()


@functools.cache
def _extension():
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("The verbs MegaMoE backend requires CUDA driver headers")
    return load(
        name="vllm_megamoe_verbs",
        sources=[str(Path(__file__).with_name("verbs.cpp"))],
        extra_include_paths=[str(Path(CUDA_HOME) / "include")],
        extra_cflags=["-O3", "-std=c++17", "-pthread"],
        extra_ldflags=[
            f"-L{CUDA_HOME}/lib64/stubs",
            "-lcuda",
            "-libverbs",
            "-pthread",
        ],
    )


def _aligned_tensor(shape, dtype, device):
    size = torch.empty((), dtype=dtype).element_size()
    elements = 1
    for dimension in shape:
        elements *= dimension
    padded = (elements * size + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
    backing = torch.empty(padded + _ALIGNMENT, dtype=torch.uint8, device=device)
    offset = -backing.data_ptr() % _ALIGNMENT
    # Keep the padded view alive as well as its typed, logical subview. DMA-BUF
    # registration covers only storage owned by this allocation.
    region = backing[offset : offset + padded]
    return region[: elements * size].view(dtype).view(shape), region


@dataclass
class Transfer:
    endpoint: _Endpoint
    id: int
    send: torch.Tensor
    receive: torch.Tensor
    regions: tuple[torch.Tensor, torch.Tensor]
    send_counts: torch.Tensor
    receive_counts: torch.Tensor
    capacity: int

    def submit(self, counts=None):
        if counts is None:
            self.send_counts.fill_(self.capacity)
        else:
            if (
                counts.dtype != torch.int32
                or counts.shape != self.send_counts.shape
                or counts.device != self.send.device
            ):
                raise ValueError("Expected one CUDA int32 row count per host")
            self.send_counts.copy_(counts)
        self.receive_counts.copy_(self.send_counts)
        self.endpoint.submit(
            self.id,
            torch.cuda.current_stream().cuda_stream,
            self.send_counts.data_ptr(),
        )

    def wait(self):
        self.endpoint.wait(
            self.id,
            torch.cuda.current_stream().cuda_stream,
            self.receive_counts.data_ptr(),
        )


@dataclass
class PipelineSlot:
    dispatch: Transfer
    returning: Transfer
    partial: torch.Tensor
    row_map: torch.Tensor
    return_map: torch.Tensor

    def wire(self):
        from vllm.distributed.hierarchical_mega_moe import WireSlot

        return WireSlot(
            self.dispatch.send,
            self.dispatch.receive,
            self.partial,
            self.returning.receive,
        )


class VerbsTransport:
    """One endpoint per remote host and GPU lane, shared by sequential layers.

    Plans own two sets of dispatch/return storage. Shape-compatible layers and
    graphs reuse them. Concurrent graph replays and overlapping forwards are
    unsupported: the caller must serialize use of this native MoE workspace.
    """

    def __init__(
        self,
        groups,
        hcas: tuple[str, ...],
        gid_index: int = 0,
        qps: int = 4,
        traffic_class: int = 0,
        timeout_seconds: int = 120,
        host_staging: bool = False,
    ):
        topology = groups.topology
        if len(topology.nodes) < 2:
            raise ValueError("The verbs pipeline requires at least two NVLink hosts")
        if len(hcas) != topology.local_size:
            raise ValueError("Specify one RDMA HCA[:port] per local EP lane")
        if (
            not 1 <= qps <= 8
            or not 0 <= gid_index <= 255
            or not 0 <= traffic_class <= 255
        ):
            raise ValueError("Invalid RDMA QP, GID or traffic-class configuration")
        if timeout_seconds <= 0:
            raise ValueError("RDMA timeout must be positive")
        self.device = torch.device("cuda", torch.accelerator.current_device_index())
        self.rank = dist.get_rank(groups.lane_group)
        self.size = len(topology.nodes)
        self.plans: dict[tuple, tuple[PipelineSlot, ...]] = {}
        self.labels: dict[int, dict] = {}
        self.closed = False
        self.endpoint = _PeerEndpoints(self.rank)
        self.bootstrap = None
        name, separator, port = hcas[topology.lane].partition(":")
        if not name or (
            separator and (not port.isdigit() or not 1 <= int(port) <= 255)
        ):
            raise ValueError("Expected RDMA HCA name with an optional port")
        module = _extension()
        try:
            self.bootstrap = dist.new_group(
                ranks=dist.get_process_group_ranks(groups.lane_group),
                backend="gloo",
                timeout=timedelta(seconds=timeout_seconds),
                use_local_synchronization=True,
            )
            peers = [peer for peer in range(self.size) if peer != self.rank]
            for peer in peers:
                self.endpoint.peers[peer] = module.Endpoint(
                    name,
                    int(port) if separator else 1,
                    gid_index,
                    qps,
                    traffic_class,
                    timeout_seconds,
                    self.rank,
                    host_staging,
                    peer,
                    self.size,
                    peer == peers[0],
                )
            remote = self._peers(self.endpoint.info())
            self.endpoint.connect(remote)
        except BaseException:
            if self.endpoint is not None:
                self.endpoint.close()
            if self.bootstrap is not None:
                dist.destroy_process_group(self.bootstrap)
            raise

    def _peers(self, value):
        peers = [None] * self.size
        dist.all_gather_object(peers, value, group=self.bootstrap)
        return peers

    def _transfer(self, shape, dtype, label):
        assert self.endpoint is not None
        send, send_region = _aligned_tensor(shape, dtype, self.device)
        receive, receive_region = _aligned_tensor(shape, dtype, self.device)
        try:
            metadata = self.endpoint.register_job(
                send.data_ptr(),
                receive.data_ptr(),
                send_region.numel(),
                send.numel() * send.element_size() // self.size,
                send[0].numel() * send.element_size(),
            )
            self.endpoint.connect_job(metadata["id"], self._peers(metadata))
        except BaseException:
            # Deregister before local tensors can be recycled by the allocator.
            self.close()
            raise
        self.labels[metadata["id"]] = label | {
            "relaxed_ordering": metadata["relaxed_ordering"],
            "host_staging": metadata["host_staging"],
        }
        return Transfer(
            self.endpoint,
            metadata["id"],
            send,
            receive,
            (send_region, receive_region),
            torch.empty(self.size, dtype=torch.int32, device=self.device),
            torch.empty(self.size, dtype=torch.int32, device=self.device),
            shape[0] // self.size,
        )

    def plan(self, capacity, width, row_bytes, dtype, return_dtype="bfloat16"):
        if self.closed:
            raise RuntimeError("RDMA transport is closed")
        if torch.accelerator.current_device_index() != self.device.index:
            raise RuntimeError("RDMA transport used on a different CUDA device")
        if return_dtype not in ("bfloat16", "mxfp8"):
            raise ValueError("Return dtype must be bfloat16 or mxfp8")
        key = (capacity, width, row_bytes, dtype, return_dtype)
        if key not in self.plans:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up this RDMA tile shape before CUDA capture")
            slots = []
            for index in range(2):
                dispatch = self._transfer(
                    (self.size * capacity, row_bytes),
                    torch.uint8,
                    dict(phase="dispatch", slot=index, capacity=capacity),
                )
                return_width = width + width // 32 if return_dtype == "mxfp8" else width
                returning = self._transfer(
                    (self.size * capacity, return_width),
                    torch.uint8 if return_dtype == "mxfp8" else dtype,
                    dict(phase="return", slot=index, capacity=capacity),
                )
                partial = (
                    torch.empty(
                        (self.size * capacity, width), device=self.device, dtype=dtype
                    )
                    if return_dtype == "mxfp8"
                    else returning.send
                )
                row_map = torch.empty(
                    (self.size, capacity), dtype=torch.int32, device=self.device
                )
                slots.append(
                    PipelineSlot(
                        dispatch, returning, partial, row_map, torch.empty_like(row_map)
                    )
                )
            self.plans[key] = tuple(slots)
        return self.plans[key]

    def set_tracing(self, enabled):
        """Record bounded CPU request/post/CQ intervals for a diagnostic replay."""
        assert self.endpoint is not None
        self.endpoint.set_tracing(enabled)

    def trace_snapshot(self):
        assert self.endpoint is not None
        return [
            record | self.labels[record["job"]]
            for record in self.endpoint.trace_snapshot()
        ]

    def close(self):
        if self.closed:
            return
        assert self.endpoint is not None
        torch.accelerator.synchronize(self.device.index)
        self.endpoint.close()
        self.plans.clear()
        self.labels.clear()
        dist.destroy_process_group(self.bootstrap)
        self.closed = True


def get_verbs_transport(groups, **kwargs):
    # The same endpoint and bounded wire storage serve all sequential layers.
    signature = tuple(sorted(kwargs.items()))
    existing = groups.verbs_transport
    if existing is None:
        existing = VerbsTransport(groups, **kwargs)
        groups.verbs_transport = existing
        groups.verbs_signature = signature
    elif groups.verbs_signature != signature:
        raise ValueError(
            "All layers sharing an EP lane must use the same RDMA settings"
        )
    return existing
