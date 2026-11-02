# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-host, graph-replayable GPUDirect RDMA with a host progress thread.

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
        self, send: int, receive: int, allocation_bytes: int, peer_bytes: int
    ) -> dict[str, Any]: ...
    def connect_job(self, id: int, remote: dict[str, Any]) -> None: ...
    def submit(self, id: int, stream: int) -> None: ...
    def wait(self, id: int, stream: int) -> None: ...
    def set_tracing(self, enabled: bool) -> None: ...
    def trace_snapshot(self) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


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

    def submit(self):
        self.endpoint.submit(self.id, torch.cuda.current_stream().cuda_stream)

    def wait(self):
        self.endpoint.wait(self.id, torch.cuda.current_stream().cuda_stream)


@dataclass
class PipelineSlot:
    dispatch: Transfer
    returning: Transfer

    def wire(self):
        from vllm.distributed.hierarchical_mega_moe import WireSlot

        return WireSlot(
            self.dispatch.send,
            self.dispatch.receive,
            self.returning.send,
            self.returning.receive,
        )


class VerbsTransport:
    """One endpoint per GPU lane, shared across sequential MoE layers.

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
        if len(topology.nodes) != 2:
            raise ValueError("The verbs pipeline currently requires two NVLink hosts")
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
        self.plans: dict[tuple, tuple[PipelineSlot, ...]] = {}
        self.labels: dict[int, dict] = {}
        self.closed = False
        self.endpoint: _Endpoint | None = None
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
            self.endpoint = module.Endpoint(
                name,
                int(port) if separator else 1,
                gid_index,
                qps,
                traffic_class,
                timeout_seconds,
                self.rank,
                host_staging,
            )
            remote = self._peer(self.endpoint.info())
            self.endpoint.connect(remote)
        except BaseException:
            if self.endpoint is not None:
                self.endpoint.close()
            if self.bootstrap is not None:
                dist.destroy_process_group(self.bootstrap)
            raise

    def _peer(self, value):
        peers = [None, None]
        dist.all_gather_object(peers, value, group=self.bootstrap)
        return peers[1 - self.rank]

    def _transfer(self, shape, dtype, label):
        assert self.endpoint is not None
        send, send_region = _aligned_tensor(shape, dtype, self.device)
        receive, receive_region = _aligned_tensor(shape, dtype, self.device)
        try:
            metadata = self.endpoint.register_job(
                send.data_ptr(),
                receive.data_ptr(),
                send_region.numel(),
                send.numel() * send.element_size() // 2,
            )
            self.endpoint.connect_job(metadata["id"], self._peer(metadata))
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
        )

    def plan(self, capacity, width, row_bytes, dtype):
        if self.closed:
            raise RuntimeError("RDMA transport is closed")
        if torch.accelerator.current_device_index() != self.device.index:
            raise RuntimeError("RDMA transport used on a different CUDA device")
        key = (capacity, width, row_bytes, dtype)
        if key not in self.plans:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up this RDMA tile shape before CUDA capture")
            slots = []
            for index in range(2):
                dispatch = self._transfer(
                    (2 * capacity, row_bytes),
                    torch.uint8,
                    dict(phase="dispatch", slot=index, capacity=capacity),
                )
                returning = self._transfer(
                    (2 * capacity, width),
                    dtype,
                    dict(phase="return", slot=index, capacity=capacity),
                )
                slots.append(PipelineSlot(dispatch, returning))
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
