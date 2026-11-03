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
        copy_self: bool = True,
        acquire_before_send: bool = False,
    ) -> dict[str, Any]: ...
    def connect_job(self, id: int, remote: dict[str, Any]) -> None: ...
    def submit(self, id: int, stream: int, rows: int) -> None: ...
    def reserve(self, id: int, stream: int) -> None: ...
    def publish(self, id: int, stream: int, rows: int) -> None: ...
    def wait(self, id: int, stream: int, rows: int) -> None: ...
    def wait_receive(self, id: int, stream: int, rows: int) -> None: ...
    def wait_send(self, id: int, stream: int) -> None: ...
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

    def register_job(
        self,
        send,
        receive,
        allocation_bytes,
        peer_bytes,
        row_bytes,
        copy_self=True,
        acquire_before_send=False,
    ):
        metadata = {
            peer: endpoint.register_job(
                send,
                receive,
                allocation_bytes,
                peer_bytes,
                row_bytes,
                copy_self,
                acquire_before_send,
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
            endpoint.submit(id, stream, rows + peer * 4 if rows else 0)

    def reserve(self, id, stream):
        for endpoint in self.peers.values():
            endpoint.reserve(id, stream)

    def publish(self, id, stream, rows):
        for peer, endpoint in self.peers.items():
            endpoint.publish(id, stream, rows + peer * 4 if rows else 0)

    def wait(self, id, stream, rows):
        for peer, endpoint in self.peers.items():
            endpoint.wait(id, stream, rows + peer * 4 if rows else 0)

    def wait_receive(self, id, stream, rows):
        for peer, endpoint in self.peers.items():
            endpoint.wait_receive(id, stream, rows + peer * 4 if rows else 0)

    def wait_send(self, id, stream):
        for endpoint in self.peers.values():
            endpoint.wait_send(id, stream)

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
class PeerSignals:
    """Mapped device addresses for one peer's stream/device handshake.

    A producer publishes send_rows before request with a system release fence.
    Consumers acquire receive_done before reading, then clear it before granting
    receive_request again. Send storage stays immutable until send_done is
    acquired and cleared. Each flag has one GPU writer and one CPU proxy writer.
    """

    request: int
    send_done: int
    receive_request: int
    receive_done: int
    send_rows: int
    receive_rows: int


class PeerBuffer:
    """Aligned RDMA storage with CUDA IPC views on every local EP rank.

    Allocate collectively before graph capture. Users must join all local
    readers before reusing or releasing the storage. Tensor views retain the
    underlying allocation and peer mappings.
    """

    def __init__(self, group, size_bytes):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Allocate peer buffers before CUDA capture")
        size_bytes = (size_bytes + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
        allocation = _extension().PeerAllocation(size_bytes)
        peers = [None] * dist.get_world_size(group)
        dist.all_gather_object(peers, allocation.info(), group=group)
        allocation.connect(peers, dist.get_rank(group))
        self.region = torch.from_dlpack(allocation.view())
        self.peers = tuple(
            torch.from_dlpack(allocation.view(peer)) for peer in range(len(peers))
        )
        self.buffer_ptrs = tuple(allocation.addresses())


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
    peer_signals: dict[int, PeerSignals]

    def reserve(self):
        """Grant a receive credit after the preceding consumer on this stream."""
        self.endpoint.reserve(self.id, torch.cuda.current_stream().cuda_stream)

    def publish_fixed(self):
        """Publish produced bytes after an earlier reserve, without counts."""
        self.endpoint.publish(self.id, torch.cuda.current_stream().cuda_stream, 0)

    def wait_receive(self, *, fixed=False):
        """Make received bytes visible without waiting for outgoing sends."""
        self.endpoint.wait_receive(
            self.id,
            torch.cuda.current_stream().cuda_stream,
            0 if fixed else self.receive_counts.data_ptr(),
        )

    def wait_send(self):
        """Release the send allocation only after the NIC has stopped reading."""
        self.endpoint.wait_send(self.id, torch.cuda.current_stream().cuda_stream)

    def submit_fixed(self):
        """Publish the registered capacity without a device-to-host count copy."""
        self.endpoint.submit(self.id, torch.cuda.current_stream().cuda_stream, 0)

    def wait_fixed(self):
        """Join a fixed-length exchange without copying its known row count."""
        self.endpoint.wait(self.id, torch.cuda.current_stream().cuda_stream, 0)

    def _prepare_counts(self, counts):
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

    def publish(self, counts=None):
        """Publish produced data for a separately reserved receive epoch."""
        self._prepare_counts(counts)
        self.endpoint.publish(
            self.id,
            torch.cuda.current_stream().cuda_stream,
            self.send_counts.data_ptr(),
        )

    def submit(self, counts=None):
        self._prepare_counts(counts)
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
        peer_reads: bool = False,
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
        self.native_plans: dict[tuple, object] = {}
        self.native_inputs: dict[tuple, object] = {}
        self.native_outputs: dict[tuple, object] = {}
        self.registered_transfers: list[Transfer] = []
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
                    peer_reads,
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
        send, send_region = _aligned_tensor(shape, dtype, self.device)
        receive, receive_region = _aligned_tensor(shape, dtype, self.device)
        return self.register_transfer(send, receive, send_region, receive_region, label)

    def register_transfer(
        self,
        send,
        receive,
        send_region,
        receive_region,
        label,
        *,
        copy_self=True,
        acquire_before_send=False,
    ):
        """Register existing aligned storage, including native symmetric buffers.

        Regions include owned alignment padding. Logical tensors must start at
        their respective region's base and have equal, contiguous peer-major
        layouts. The returned transfer retains both regions until deregistration.
        Set copy_self=False when the caller consumes the local partial directly,
        including chunk views whose peer stride is larger than the payload.
        With acquire_before_send, call wait_send before every producer, including
        its first use. This permits deferring TX completion to slot reuse.
        """
        if self.closed or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Register live RDMA storage before CUDA capture")
        for tensor, region in ((send, send_region), (receive, receive_region)):
            if (
                tensor.ndim != 2
                or tensor.shape[0] == 0
                or tensor.shape[1] == 0
                or tensor.shape[0] % self.size
                or not tensor.is_contiguous()
                or tensor.device != self.device
                or region.dtype != torch.uint8
                or region.device != self.device
                or not region.is_contiguous()
                or region.data_ptr() % _ALIGNMENT
                or region.numel() % _ALIGNMENT
                or tensor.data_ptr() != region.data_ptr()
                or tensor.nbytes > region.nbytes
            ):
                raise ValueError("Invalid aligned peer-major RDMA storage")
        if (
            send.shape != receive.shape
            or send.dtype != receive.dtype
            or send_region.nbytes != receive_region.nbytes
        ):
            raise ValueError("RDMA send and receive layouts must match")
        if (
            send_region.data_ptr() < receive_region.data_ptr() + receive_region.nbytes
            and receive_region.data_ptr() < send_region.data_ptr() + send_region.nbytes
        ):
            raise ValueError("RDMA send and receive regions must not overlap")
        assert self.endpoint is not None
        try:
            metadata = self.endpoint.register_job(
                send.data_ptr(),
                receive.data_ptr(),
                send_region.numel(),
                send.numel() * send.element_size() // self.size,
                send[0].numel() * send.element_size(),
                copy_self,
                acquire_before_send,
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
        transfer = Transfer(
            self.endpoint,
            metadata["id"],
            send,
            receive,
            (send_region, receive_region),
            torch.empty(self.size, dtype=torch.int32, device=self.device),
            torch.empty(self.size, dtype=torch.int32, device=self.device),
            send.shape[0] // self.size,
            {
                peer: PeerSignals(
                    **{
                        name: entry[f"{name}_address"]
                        for name in PeerSignals.__dataclass_fields__
                    }
                )
                for peer, entry in metadata["peers"].items()
            },
        )
        # The MR outlives a caller dropping its transfer or graph. Deregister
        # before releasing storage, including externally allocated buffers.
        self.registered_transfers.append(transfer)
        return transfer

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
        self.native_plans.clear()
        self.native_inputs.clear()
        self.native_outputs.clear()
        self.registered_transfers.clear()
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
