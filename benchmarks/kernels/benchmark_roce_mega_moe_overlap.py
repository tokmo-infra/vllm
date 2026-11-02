# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-host single-layer acceptance for the experimental verbs pipeline.

Run on exclusive GPUs. Checks transport bytes, changed-input graph replays,
tails, masked routes, empty ranks and exact matched-tiling serial parity.
Timing and traces are emitted only after correctness checks pass.
"""

import argparse
import json
import os
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from benchmark_hierarchical_mega_moe import NativeExperts, oracle

from vllm.distributed.hierarchical_mega_moe import (
    HierarchicalMegaMoE,
    destroy_hierarchical_groups,
    get_hierarchical_groups,
)
from vllm.distributed.roce.transport import get_verbs_transport


def capture(fn):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    graph.replay()
    torch.accelerator.synchronize()
    return graph, output


def check_payload(transport, args):
    """Changing the generation exposes stale ready flags and reversed peers."""
    row_bytes = args.hidden + args.hidden // 32 + 8 * args.top_k
    slots = transport.plan(257, args.hidden, row_bytes, torch.bfloat16)
    transfers = [
        transfer for slot in slots for transfer in (slot.dispatch, slot.returning)
    ]

    def exchange():
        for transfer in transfers:
            transfer.submit()
        for transfer in transfers:
            transfer.wait()

    for transfer in transfers:
        transfer.send.zero_()
    graph, _ = capture(exchange)
    for generation in range(8):
        for index, transfer in enumerate(transfers):
            for peer in range(2):
                value = (
                    transport.rank * 67 + peer * 29 + generation * 11 + index
                ) % 251
                transfer.send.chunk(2)[peer].fill_(value)
        for _ in range(3):
            graph.replay()
        torch.accelerator.synchronize()
        for index, transfer in enumerate(transfers):
            for peer in range(2):
                expected = (
                    peer * 67 + transport.rank * 29 + generation * 11 + index
                ) % 251
                assert torch.all(transfer.receive.chunk(2)[peer] == expected).item()
    del graph


def check_graph_pair(serial, pipeline, model, x, gates, ids, args, shift):
    original = tuple(value.clone() for value in (x, gates, ids))
    if dist.get_rank() == 0:
        print("capturing serial", flush=True)
    serial_graph, reference = capture(serial)
    if dist.get_rank() == 0:
        print("capturing pipeline", flush=True)
    pipeline_graph, actual = capture(pipeline)
    for generation in range(4):
        if dist.get_rank() == 0:
            print(f"checking generation={generation}", flush=True)
        x.copy_(original[0] * (1 - generation / 8))
        gates.copy_(original[1])
        if generation:
            gates[::3] = 0
        physical_shift = shift if generation == 2 else 0
        ids.copy_((original[2] - physical_shift) % args.experts)
        if generation:
            ids[::3] = -1
        if generation == 3:
            gates.zero_()
            ids.fill_(-1)
        for slot, physical in enumerate(model.expert_ids):
            logical = (physical + physical_shift) % args.experts
            model.l1[1][slot].fill_((124 - logical % 7) * 0x01010101)
            model.l2[1][slot].fill_((124 - logical % 5) * 0x01010101)
        serial_graph.replay()
        pipeline_graph.replay()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.float(),
            oracle(x, gates, original[2], args.intermediate),
            rtol=0.08,
            atol=0.01,
        )
        if generation:
            assert torch.count_nonzero(actual[::3]).item() == 0
    for target, source in zip((x, gates, ids), original):
        target.copy_(source)
    for slot, expert in enumerate(model.expert_ids):
        model.l1[1][slot].fill_((124 - expert % 7) * 0x01010101)
        model.l2[1][slot].fill_((124 - expert % 5) * 0x01010101)
    serial_graph.replay()
    pipeline_graph.replay()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    return serial_graph, pipeline_graph, reference, actual


def measure(graph, args):
    for _ in range(args.stress_replays):
        graph.replay()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(args.samples):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / args.iterations)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hcas", required=True, help="GPU-lane-ordered HCA[:port] list"
    )
    parser.add_argument("--gid", type=int, default=0)
    parser.add_argument("--qps", type=int, default=4)
    parser.add_argument("--host-staging", action="store_true")
    parser.add_argument("--traffic-class", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=2304)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--tokens", default="768,1536,4096,8192,16384")
    parser.add_argument("--tiles", default="256,512,1024,2048")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--stress-replays", type=int, default=1000)
    parser.add_argument("--fast-math", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--include-nccl-baseline",
        action="store_true",
        help="Measure the full-batch NCCL path with the same replay timing",
    )
    parser.add_argument(
        "--nccl-only",
        action="store_true",
        help="Measure NCCL in a separate process run",
    )
    parser.add_argument(
        "--transport-only",
        action="store_true",
        help="Measure full-batch dispatch, BF16 return and concurrent transport",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    local = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    cpu = dist.new_group(backend="gloo")
    groups = get_hierarchical_groups(dist.group.WORLD, cpu)
    rank = dist.get_rank()
    transport = get_verbs_transport(
        groups,
        hcas=tuple(args.hcas.split(",")),
        gid_index=args.gid,
        qps=args.qps,
        traffic_class=args.traffic_class,
        timeout_seconds=args.timeout,
        host_staging=args.host_staging,
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    per_rank = args.experts // dist.get_world_size()
    model = (
        None
        if args.transport_only
        else NativeExperts(
            groups.local_group, range(rank * per_rank, (rank + 1) * per_rank), args
        )
    )
    records = []

    def checkpoint():
        (args.output_directory / f"partial-rank{rank}.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )

    check_payload(transport, args)
    for tokens in map(int, args.tokens.split(",")):
        if args.transport_only:
            row_bytes = args.hidden + args.hidden // 32 + 8 * args.top_k
            slots = transport.plan(tokens, args.hidden, row_bytes, torch.bfloat16)
            dispatch, returning = slots[0].dispatch, slots[0].returning
            for transfer in (dispatch, returning):
                transfer.send.fill_(transport.rank + 1)
            for name, transfers in (
                ("dispatch_only", (dispatch,)),
                ("return_only", (returning,)),
                ("dispatch_return_concurrent", (dispatch, returning)),
            ):

                def exchange(transfers=transfers):
                    for transfer in transfers:
                        transfer.submit()
                    for transfer in transfers:
                        transfer.wait()

                graph, _ = capture(exchange)
                for transfer in transfers:
                    for peer, received in enumerate(transfer.receive.chunk(2)):
                        assert torch.all(received == peer + 1).item()
                records.append(
                    dict(
                        tokens=tokens,
                        tile=tokens,
                        effective_tile=tokens,
                        wire_tokens=tokens,
                        case=name,
                        samples_ms=measure(graph, args),
                    )
                )
                checkpoint()
                del graph
            continue
        assert model is not None
        torch.manual_seed(1234 + rank)
        x = (torch.rand(tokens, args.hidden, device="cuda") * 0.01).bfloat16()
        ids = (
            torch.rand(tokens, args.experts, device="cuda").topk(args.top_k, 1).indices
        )
        gates = torch.rand(tokens, args.top_k, device="cuda").softmax(1)
        if args.include_nccl_baseline or args.nccl_only:
            if rank == 0:
                print(f"measuring NCCL tokens={tokens}", flush=True)
            baseline = HierarchicalMegaMoE(
                groups,
                args.experts,
                args.chunk,
                wire_dtype="mxfp8",
                fused_pack=True,
            )
            graph, output = capture(
                partial(
                    baseline.forward,
                    x,
                    gates,
                    ids,
                    model,
                    fixed_capacity=True,
                    local_moe_mxfp8=model.direct,
                )
            )
            torch.testing.assert_close(
                output.float(),
                oracle(x, gates, ids, args.intermediate),
                rtol=0.08,
                atol=0.01,
            )
            if rank == 0:
                print(f"timing NCCL tokens={tokens}", flush=True)
            records.append(
                dict(
                    tokens=tokens,
                    tile=0,
                    effective_tile=tokens,
                    wire_tokens=tokens,
                    case="baseline_nccl",
                    samples_ms=measure(graph, args),
                )
            )
            checkpoint()
            del graph, output, baseline
        if args.nccl_only:
            continue
        for tile in map(int, args.tiles.split(",")):
            if rank == 0:
                print(f"validating tokens={tokens} tile={tile}", flush=True)
            adapters = [
                HierarchicalMegaMoE(
                    groups,
                    args.experts,
                    args.chunk,
                    wire_dtype="mxfp8",
                    fused_pack=True,
                    source_chunk_size=tile,
                    pipeline_transport=transport,
                    overlap=overlap,
                )
                for overlap in (False, True)
            ]
            callbacks = [
                partial(
                    adapter.forward,
                    x,
                    gates,
                    ids,
                    model,
                    fixed_capacity=True,
                    local_moe_mxfp8=model.direct,
                )
                for adapter in adapters
            ]
            # Uneven eager source counts include an empty rank. The remaining
            # ranks force padding and, for most shapes, a final partial tile.
            count = 0 if rank == 0 else max(1, tokens - rank - 1)
            eager = []
            for adapter in adapters:
                if rank == 0:
                    print(f"eager overlap={adapter.overlap}", flush=True)
                eager.append(
                    adapter.forward(
                        x[:count],
                        gates[:count],
                        ids[:count],
                        model,
                        local_moe_mxfp8=model.direct,
                    )
                )
            torch.testing.assert_close(eager[1], eager[0], rtol=0, atol=0)
            torch.testing.assert_close(
                eager[1].float(),
                oracle(x[:count], gates[:count], ids[:count], args.intermediate),
                rtol=0.08,
                atol=0.01,
            )
            if rank == 0:
                print("checking graph pair", flush=True)
            serial, pipeline, reference, actual = check_graph_pair(
                *callbacks,
                model,
                x,
                gates,
                ids,
                args,
                shift=per_rank * groups.topology.local_size,
            )
            for name, graph in (("serial_verbs", serial), ("pipeline_verbs", pipeline)):
                if rank == 0:
                    print(f"timing {name} tokens={tokens} tile={tile}", flush=True)
                samples = measure(graph, args)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                records.append(
                    dict(
                        tokens=tokens,
                        tile=tile,
                        effective_tile=min(tokens, tile),
                        wire_tokens=(tokens + min(tokens, tile) - 1)
                        // min(tokens, tile)
                        * min(tokens, tile),
                        case=name,
                        samples_ms=samples,
                    )
                )
                checkpoint()
            if args.trace:
                transport.set_tracing(True)
                context = (
                    torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    )
                    if local == 0
                    else nullcontext()
                )
                with context as profiler:
                    dist.barrier()
                    clock_pair = dict(
                        wall_ns=time.time_ns(), monotonic_ns=time.monotonic_ns()
                    )
                    pipeline.replay()
                    torch.accelerator.synchronize()
                stem = args.output_directory / f"r{rank}-t{tokens}-tile{tile}"
                stem.with_suffix(".rdma.json").write_text(
                    json.dumps(
                        dict(
                            clock_pair=clock_pair, transfers=transport.trace_snapshot()
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                transport.set_tracing(False)
                if local == 0:
                    profiler.export_chrome_trace(str(stem.with_suffix(".cuda.json")))
            # Graphs reference the flags and registered buffers. Destroy them
            # before transport teardown, after all replays have completed.
            del serial, pipeline, graph
            if rank == 0:
                print(f"validated tokens={tokens} tile={tile}", flush=True)
    all_records = [None] * dist.get_world_size()
    dist.all_gather_object(all_records, records, group=cpu)
    if rank == 0:
        (args.output_directory / "results.json").write_text(
            json.dumps(
                dict(
                    config=vars(args)
                    | {"output_directory": str(args.output_directory)},
                    per_rank=all_records,
                ),
                indent=2,
            )
            + "\n"
        )
    torch.accelerator.synchronize()
    dist.barrier()
    destroy_hierarchical_groups()
    dist.destroy_process_group(cpu)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
