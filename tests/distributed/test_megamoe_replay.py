# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Changing graph inputs must not reuse an earlier peer's combine completion."""

import time
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def test_streaming_draft_padding_preserves_packed_payload():
    """Odd top-k rows must be TMA aligned without moving gates or expert IDs."""
    from vllm.distributed.megamoe_native_overlap import streaming_row_bytes
    from vllm.distributed.megamoe_rdma_ops import pack_fixed

    torch.manual_seed(71)
    tokens, width, topk = 33, 5120, 3
    x = torch.randn(tokens, width, device="cuda").bfloat16()
    ids = torch.tensor([0, 5, -1], device="cuda").expand(tokens, -1)
    gates = torch.rand(tokens, topk, device="cuda")
    tables = tuple(
        torch.tensor(values, device="cuda") for values in ([0, 1], [0, 0], [0, 1])
    )
    packed, *_ = pack_fixed(x, gates, ids, tables, 8, 2, "mxfp8")
    aligned, *_ = pack_fixed(x, gates, ids, tables, 8, 2, "mxfp8", row_alignment=16)
    assert aligned.stride(0) == streaming_row_bytes(width, topk)
    assert aligned.stride(0) % 16 == 0
    torch.testing.assert_close(aligned[:, : packed.shape[1]], packed, atol=0, rtol=0)


def _check_replay(rank, rendezvous, mode, topk):
    from vllm.distributed.megamoe_native_overlap import streaming_row_bytes
    from vllm.distributed.megamoe_rdma_ops import pack_mxfp8
    from vllm.model_executor.layers.fused_moe.prepare_megamoe import (
        prepare_megamoe_inputs,
    )
    from vllm.utils.deep_gemm import _import_deep_gemm

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        device_id=torch.device("cuda", rank),
    )
    dg = _import_deep_gemm()
    torch.manual_seed(71 + rank)
    tokens, hidden, intermediate, experts = 64, 1024, 512, 8
    buffers = [
        dg.get_symm_buffer_for_mega_moe(
            dist.group.WORLD,
            experts,
            tokens,
            topk,
            hidden,
            intermediate,
            mma_type="fp8xfp8",
            routing_groups=4 if index == 1 and mode == "streaming" else 1,
        )
        for index in range(2)
    ]

    def weight(n, k):
        value = (torch.randn(experts // 2, n, k, device="cuda") * 0.03).to(
            torch.float8_e4m3fn
        )
        sf = torch.full(
            (experts // 2, k // 128, n),
            0x7F7F7F7F,
            device="cuda",
            dtype=torch.int32,
        )
        return value, sf.transpose(-1, -2)

    l1, l2 = dg.transform_weights_for_mega_moe(
        weight(intermediate * 2, hidden), weight(hidden, intermediate)
    )
    original = torch.randn(tokens, hidden, device="cuda").bfloat16()
    x = original.clone()
    # Rank 1 has no expert work. Its combine must still wait for rank 0's
    # current output, including when CUDA reuses %gridid on graph replay.
    ids = torch.arange(topk, device="cuda").expand(tokens, topk).contiguous()
    gates = torch.full((tokens, topk), 1 / topk, device="cuda")
    outputs = [torch.empty_like(x) for _ in buffers]
    wire = wire_handle = descriptors = scratch = None
    if mode == "streaming":
        import torch.distributed._symmetric_memory as symm_mem

        row_bytes, chunk = streaming_row_bytes(hidden, topk), 17
        wire = symm_mem.empty((tokens, row_bytes), dtype=torch.uint8, device="cuda")
        wire_handle = symm_mem.rendezvous(wire, group=dist.group.WORLD)
        descriptors = torch.tensor(
            [
                [[0, ptr + group * chunk * row_bytes, row_bytes] for group in range(4)]
                for ptr in wire_handle.buffer_ptrs
            ],
            dtype=torch.int64,
            device="cuda",
        )
        capacity = buffers[1].num_max_tokens_per_rank
        scratch = torch.empty(
            (4, (capacity * 2 * topk + 7) // 8 + experts // 2 * 4 + 1),
            dtype=torch.int64,
            device="cuda",
        )

    def forward(index):
        b = buffers[index]
        if index == 1 and mode == "streaming":
            gate_offset = hidden + hidden // 32
            wire[:, :gate_offset].copy_(pack_mxfp8(x))
            wire[:, gate_offset : gate_offset + 4 * topk].view(torch.float32).copy_(
                gates
            )
            wire[:, gate_offset + 4 * topk : gate_offset + 8 * topk].view(
                torch.int32
            ).copy_(ids)
        else:
            prepare_megamoe_inputs(
                x,
                gates,
                ids,
                b.x[:tokens],
                b.x_sf[:tokens],
                b.topk_idx[:tokens],
                b.topk_weights[:tokens],
            )
        dg.fp8_fp4_mega_moe(
            outputs[index],
            l1,
            l2,
            b,
            activation_clamp=10.0,
            fast_math=False,
            early_combine=index == 1 and mode != "ordered",
            **(
                dict(
                    input_descriptors=descriptors,
                    input_chunk_size=17,
                    scheduler_workspace=scratch,
                )
                if index == 1 and mode == "streaming"
                else {}
            ),
        )

    try:
        for _ in range(3):
            forward(0)
            forward(1)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward(1)
        graph.replay()
        torch.cuda.synchronize()
        for generation in range(8):
            x.copy_(original * (generation + 1) / 8)
            ids.copy_(torch.arange(topk, device="cuda").expand_as(ids))
            if generation == 3:
                ids[-13:] = -1
            elif generation == 7:
                ids.fill_(-1)
            forward(0)
            torch.cuda.synchronize()
            dist.barrier()
            for _ in range(3):
                graph.replay()
                torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
        del graph
    except BaseException:
        traceback.print_exc()
        raise
    else:
        torch.cuda.synchronize()
        for buffer in buffers:
            buffer.destroy()
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("mode", "topk"), [("ordered", 2), ("early", 2), ("streaming", 2), ("streaming", 3)]
)
def test_megamoe_graph_replay_consumes_current_peer_partials(tmp_path, mode, topk):
    """Two SM100 GPUs, independent workspaces, changed inputs and an empty rank."""
    assert torch.cuda.device_count() >= 2, "This distributed test requires two GPUs"
    workers = mp.spawn(
        _check_replay,
        args=((tmp_path / "rendezvous").as_uri(), mode, topk),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 180
    try:
        while time.monotonic() < deadline:
            if workers.join(timeout=max(0, deadline - time.monotonic())):
                return
        raise AssertionError("MegaMoE replay workers did not finish within 180 seconds")
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.kill()
            process.join()
