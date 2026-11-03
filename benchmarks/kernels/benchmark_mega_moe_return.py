# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise native in-kernel BF16 return publication through real RoCE QPs.

Use two physical hosts or --virtual-island-size for an intra-host acceptance
probe. Virtual islands validate the protocol, not inter-host performance.
"""

import argparse
import json
import os
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist
from benchmark_hierarchical_mega_moe import NativeExperts
from benchmark_roce_mega_moe_overlap import capture, measure

from vllm.distributed.hierarchical_mega_moe import HierarchicalGroups
from vllm.distributed.megamoe_rdma_ops import pack_mxfp8, stage_mxfp8
from vllm.distributed.roce.transport import PeerBuffer, VerbsTransport
from vllm.model_executor.layers.fused_moe.prepare_megamoe import prepare_megamoe_inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--return-chunk", type=int, default=64)
    p.add_argument("--hidden", type=int, default=5120)
    p.add_argument("--intermediate", type=int, default=2304)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--experts-per-rank", type=int, default=24)
    p.add_argument("--routing-groups", type=int, default=1)
    p.add_argument("--ready-scheduling", action="store_true")
    p.add_argument("--stream-input", action="store_true")
    p.add_argument("--input-chunk", type=int, default=0)
    p.add_argument("--trace-input", action="store_true")
    p.add_argument("--gate-last-input", action="store_true")
    p.add_argument("--reverse-arrival", action="store_true")
    p.add_argument("--virtual-island-size", type=int, default=0)
    p.add_argument("--hcas", required=True)
    p.add_argument("--gid", type=int, default=3)
    p.add_argument("--traffic-class", type=int, default=106)
    p.add_argument("--qps", type=int, default=4)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--stress-replays", type=int, default=100)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()
    cpu = dist.new_group(backend="gloo")
    island = args.virtual_island_size or int(os.environ["LOCAL_WORLD_SIZE"])
    assert world == island * 2, "This probe requires exactly two equal islands"
    hosts = [str(i // island) for i in range(world)]
    groups = HierarchicalGroups.create(list(range(world)), hosts, rank, "nccl")
    topo = groups.topology
    hcas = tuple(args.hcas.split(","))
    if args.virtual_island_size:
        hcas = hcas[topo.node * island : (topo.node + 1) * island]
    transport = VerbsTransport(
        groups,
        hcas,
        args.gid,
        args.qps,
        args.traffic_class,
        peer_reads=args.stream_input,
    )
    tokens, width, chunk = args.tokens, args.hidden, args.return_chunk
    assert tokens % chunk == 0 and chunk * width * 2 % 65536 == 0
    args.chunk, args.fast_math = tokens * 2, False
    input_chunk = args.input_chunk or chunk
    if args.stream_input:
        assert tokens % input_chunk == 0
        assert args.routing_groups == 2 * tokens // input_chunk
        assert args.ready_scheduling
    reference = NativeExperts(
        groups.local_group,
        range(
            topo.lane * args.experts_per_rank, (topo.lane + 1) * args.experts_per_rank
        ),
        args,
    )
    dg = reference.dg
    native = dg.get_symm_buffer_for_mega_moe(
        groups.local_group,
        island * args.experts_per_rank,
        tokens * 2,
        args.top_k,
        width,
        args.intermediate,
        mma_type="fp8xfp4",
        routing_groups=args.routing_groups,
    )
    blocks = (
        native.num_max_tokens_per_rank
        * island
        * min(args.top_k, args.experts_per_rank * args.routing_groups)
        + 7
    ) // 8 + args.experts_per_rank * args.routing_groups
    scratch = torch.empty((4, blocks + 1), device="cuda", dtype=torch.int64)
    group_ids = (
        torch.arange(tokens * 2, device="cuda") * args.routing_groups // (tokens * 2)
    )[:, None]
    torch.manual_seed(713 + rank)
    reference.l1[0].random_(-128, 128)
    reference.l2[0].random_(-128, 128)
    x = (torch.randn(tokens * 2, width, device="cuda") * 0.01).bfloat16()
    original = x.clone()
    ids = (
        torch.rand(tokens * 2, island * args.experts_per_rank, device="cuda")
        .topk(args.top_k, 1)
        .indices
    )
    original_ids = ids.clone()
    gates = torch.rand(tokens * 2, args.top_k, device="cuda").softmax(1)
    # One unused peer span lets each chunk registration retain the common
    # peer stride without moving the native output into temporary wire slots.
    send = PeerBuffer(groups.local_group, 3 * tokens * width * 2)
    receive = PeerBuffer(groups.local_group, 3 * tokens * width * 2)
    y = (
        send.region[: 2 * tokens * width * 2]
        .view(torch.bfloat16)
        .view(2 * tokens, width)
    )
    returned = receive.region[: 2 * tokens * width * 2].view(torch.bfloat16).view_as(y)
    transfers = []
    descriptors = torch.zeros(
        (2 * tokens // chunk, 4), device="cuda", dtype=torch.int64
    )
    counts = torch.full((2,), chunk, device="cuda", dtype=torch.int32)
    for j in range(tokens // chunk):
        offset = j * chunk * width * 2
        size = 2 * tokens * width * 2
        a, b = (
            send.region[offset : offset + size],
            receive.region[offset : offset + size],
        )
        transfer = transport.register_transfer(
            a.view(torch.bfloat16).view(2 * tokens, width),
            b.view(torch.bfloat16).view(2 * tokens, width),
            a,
            b,
            dict(phase="native-return", chunk=j),
            copy_self=False,
        )
        transfers.append(transfer)
        peer = 1 - topo.node
        signal = transfer.peer_signals[peer]
        descriptors[peer * (tokens // chunk) + j, :2] = torch.tensor(
            (signal.request, signal.send_rows), device="cuda"
        )

    incoming_x, incoming_gates, incoming_ids = x.clone(), gates.clone(), ids.clone()
    diagnostics = torch.zeros(
        (args.routing_groups, 4), device="cuda", dtype=torch.int64
    )
    input_descriptors = None
    dispatch = []
    if args.stream_input:
        row_bytes = width + width // 32 + args.top_k * 8
        input_descriptors = torch.zeros(
            (island, args.routing_groups, 3), device="cuda", dtype=torch.int64
        )
        for j in range(tokens // input_chunk):
            a, b = (
                PeerBuffer(groups.local_group, 2 * input_chunk * row_bytes)
                for _ in range(2)
            )
            aw, bw = (
                v.region[: 2 * input_chunk * row_bytes].view(2 * input_chunk, row_bytes)
                for v in (a, b)
            )
            transfer = transport.register_transfer(
                aw,
                bw,
                a.region,
                b.region,
                dict(phase="native-input", chunk=j),
                copy_self=False,
            )
            ready = [0, 0]
            ready[1 - topo.node] = transfer.peer_signals[1 - topo.node].receive_done
            all_ready = [None] * island
            dist.all_gather_object(all_ready, ready, group=groups.local_group)
            for lane in range(island):
                for origin in range(2):
                    group = origin * (tokens // input_chunk) + j
                    input_descriptors[lane, group] = torch.tensor(
                        (
                            all_ready[lane][origin],
                            b.buffer_ptrs[lane] + origin * input_chunk * row_bytes,
                            row_bytes,
                        ),
                        device="cuda",
                    )
            dispatch.append((transfer, a, b))

    gate_threads, gate_results = [], []
    producer_stream = torch.cuda.Stream()

    def forward(early, gate_last=False):
        buffer = native if early else reference.buffer
        if args.stream_input:
            for transfer, _, _ in dispatch:
                transfer.reserve()
            order = list(range(len(dispatch)))
            if args.reverse_arrival:
                order.reverse()
            for j in order:
                transfer = dispatch[j][0]
                for peer in range(2):
                    rows = slice(
                        peer * tokens + j * input_chunk,
                        peer * tokens + (j + 1) * input_chunk,
                    )
                    output = transfer.send[
                        peer * input_chunk : (peer + 1) * input_chunk
                    ]
                    output[:, : width + width // 32].copy_(pack_mxfp8(x[rows]))
                    output[
                        :, width + width // 32 : width + width // 32 + 4 * args.top_k
                    ].view(torch.float32).copy_(gates[rows])
                    output[:, width + width // 32 + 4 * args.top_k :].view(
                        torch.int32
                    ).copy_(ids[rows])
                local_rows = slice(
                    topo.node * input_chunk, (topo.node + 1) * input_chunk
                )
                transfer.receive[local_rows].copy_(transfer.send[local_rows])
                if not gate_last or j != len(dispatch) - 1:
                    transfer.publish_fixed()
            if gate_last:
                assert early and len(dispatch) > 1
                packed = torch.cuda.Event()
                packed.record()

                def release_last_input():
                    torch.cuda.set_device(local)
                    deadline = time.monotonic() + 30
                    waiter = threading.Event()
                    observed = None
                    while time.monotonic() < deadline:
                        observed = next(
                            (
                                r
                                for r in transport.trace_snapshot()
                                if r["phase"] == "native-return"
                                and r["chunk"] == 0
                                and r["direction"] == "send"
                                and r["post_ns"]
                            ),
                            None,
                        )
                        if observed is not None:
                            break
                        waiter.wait(0.001)
                    # On timeout, release the deliberately held request so the
                    # assertion fails cleanly instead of leaving a resident grid.
                    with torch.cuda.stream(producer_stream):
                        producer_stream.wait_event(packed)
                        dispatch[-1][0].publish_fixed()
                    gate_results.append(
                        dict(first_return=observed, released_at_ns=time.monotonic_ns())
                    )

                thread = threading.Thread(target=release_last_input)
                thread.start()
                gate_threads.append(thread)
            if not early:
                for j, (transfer, _, _) in enumerate(dispatch):
                    transfer.wait_receive(fixed=True)
                    for origin in range(2):
                        rows = slice(
                            origin * tokens + j * input_chunk,
                            origin * tokens + (j + 1) * input_chunk,
                        )
                        wire = transfer.receive[
                            origin * input_chunk : (origin + 1) * input_chunk
                        ]
                        gate_offset = width + width // 32
                        route_offset = gate_offset + 4 * args.top_k
                        stage_mxfp8(
                            wire[:, :gate_offset],
                            wire[:, gate_offset:route_offset].view(torch.float32),
                            wire[:, route_offset:].view(torch.int32),
                            buffer.x[rows],
                            buffer.x_sf[rows],
                            buffer.topk_idx[rows],
                            buffer.topk_weights[rows],
                        )
        else:
            routes = incoming_ids if args.stream_input else ids
            if early and args.routing_groups != 1:
                e = args.experts_per_rank
                routes = torch.where(
                    routes >= 0,
                    (routes // e) * (e * args.routing_groups)
                    + group_ids * e
                    + routes % e,
                    -1,
                )
            prepare_megamoe_inputs(
                incoming_x if args.stream_input else x,
                incoming_gates if args.stream_input else gates,
                routes,
                buffer.x[: 2 * tokens],
                buffer.x_sf[: 2 * tokens],
                buffer.topk_idx[: 2 * tokens],
                buffer.topk_weights[: 2 * tokens],
            )
        for transfer in transfers:
            transfer.reserve()
        dg.fp8_fp4_mega_moe(
            y,
            reference.l1,
            reference.l2,
            buffer,
            activation_clamp=10.0,
            fast_math=False,
            early_combine=early,
            scheduler_workspace=scratch if early and args.ready_scheduling else None,
            return_descriptors=descriptors if early else None,
            return_chunk_size=chunk if early else 0,
            input_descriptors=input_descriptors if early else None,
            input_chunk_size=input_chunk if early and args.stream_input else 0,
            streaming_diagnostics=diagnostics if early and args.trace_input else None,
        )
        if not early:
            for transfer in transfers:
                transfer.publish(counts)
        for transfer in transfers:
            transfer.wait_receive()
            transfer.wait_send()
        if args.stream_input:
            for transfer, _, _ in dispatch:
                if early:
                    transfer.wait_receive(fixed=True)
                transfer.wait_send()
        return y

    def refresh_incoming():
        if args.stream_input:
            for dst, src in (
                (incoming_x, x),
                (incoming_gates, gates),
                (incoming_ids, ids),
            ):
                dist.all_to_all_single(dst, src, group=groups.lane_group)

    refresh_incoming()
    serial, _ = capture(lambda: forward(False))
    early, _ = capture(lambda: forward(True))
    peer = 1 - topo.node
    remote_slice = slice(peer * tokens, (peer + 1) * tokens)
    expected_peer = torch.empty_like(y)
    for generation in range(4):
        x.copy_(original * (generation + 1) / 4)
        ids.copy_(original_ids)
        if generation == 1:
            ids[::3] = -1
        elif generation == 2:
            ids.copy_(torch.arange(args.top_k, device="cuda").expand_as(ids))
        elif generation == 3:
            ids.fill_(-1)
        refresh_incoming()
        expected = (
            reference(incoming_x, incoming_gates, incoming_ids)
            if args.stream_input
            else reference(x, gates, ids)
        )
        dist.all_to_all_single(expected_peer, expected, group=groups.lane_group)
        for graph in (serial, early):
            for _ in range(8):
                graph.replay()
                torch.testing.assert_close(y, expected, atol=0, rtol=0)
                torch.testing.assert_close(
                    returned[remote_slice], expected_peer[remote_slice], atol=0, rtol=0
                )
        if rank == 0:
            print(f"verified native return generation={generation}", flush=True)
    x.copy_(original)
    ids.copy_(original_ids)
    refresh_incoming()
    samples = {
        name: measure(graph, args)
        for name, graph in (("serial", serial), ("early", early))
    }
    expected = (
        reference(incoming_x, incoming_gates, incoming_ids)
        if args.stream_input
        else reference(x, gates, ids)
    )
    dist.all_to_all_single(expected_peer, expected, group=groups.lane_group)
    torch.testing.assert_close(y, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        returned[remote_slice], expected_peer[remote_slice], atol=0, rtol=0
    )
    if args.gate_last_input:
        assert args.stream_input
        transport.set_tracing(True)
        forward(True, gate_last=True)
        torch.cuda.synchronize()
        for thread in gate_threads:
            thread.join(timeout=35)
            assert not thread.is_alive(), "Delayed input publisher did not finish"
        assert gate_results and all(
            r["first_return"] is not None for r in gate_results
        ), (
            "Native return stalled before the deliberately withheld last input"
        )
        torch.testing.assert_close(y, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            returned[remote_slice], expected_peer[remote_slice], atol=0, rtol=0
        )
        transport.set_tracing(False)
        if rank == 0:
            print(
                "verified first native return before last input publication", flush=True
            )
    args.output.mkdir(parents=True, exist_ok=True)
    transport.set_tracing(True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        dist.barrier()
        clock_pair = dict(wall_ns=time.time_ns(), monotonic_ns=time.monotonic_ns())
        early.replay()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(args.output / f"rank{rank}.cuda.json"))
    (args.output / f"rank{rank}.json").write_text(
        json.dumps(
            dict(
                config=vars(args) | {"output": str(args.output)},
                samples_ms=samples,
                gate_last_input=gate_results,
                clock_pair=clock_pair,
                transfers=transport.trace_snapshot(),
                gpu_timestamps_ns=descriptors[:, 2:].cpu().tolist(),
                input_timestamps_ns=diagnostics.cpu().tolist()
                if args.trace_input
                else None,
            ),
            indent=2,
        )
        + "\n"
    )
    transport.set_tracing(False)
    del serial, early
    torch.cuda.synchronize()
    dist.barrier()
    transport.close()
    native.destroy()
    reference.buffer.destroy()
    groups.close()
    dist.destroy_process_group(cpu)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
