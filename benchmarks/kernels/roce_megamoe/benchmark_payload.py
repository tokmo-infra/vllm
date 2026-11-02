# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL peer-only payload baseline for the standalone GPUNetIO probe."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.distributed.hierarchical_mega_moe import (
    destroy_hierarchical_groups,
    get_hierarchical_groups,
)
from vllm.distributed.megamoe_transport import ProcessGroupTransport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="768,936,1024,1536,4096")
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.accelerator.set_device_index(device)
    dist.init_process_group("nccl", device_id=device)
    cpu = dist.new_group(backend="gloo")
    groups = get_hierarchical_groups(dist.group.WORLD, cpu)
    topo = groups.topology
    if len(topo.nodes) != 2:
        raise ValueError("Matched GPUNetIO comparison requires exactly two hosts")
    transport = ProcessGroupTransport(groups.lane_group)
    remote = topo.lane_nodes.index(1 - topo.node)
    results = []
    try:
        for tokens in map(int, args.tokens.split(",")):
            sizes = [
                tokens * (args.hidden + args.hidden // 32 + 8 * args.top_k),
                tokens * 2 * args.hidden,
            ]
            for one_way in (True, False):
                sends = [0, 0]
                receives = [0, 0]
                sends[remote] = int(not one_way or topo.node == 0)
                receives[remote] = int(not one_way or topo.node == 1)
                buffers = [
                    (
                        torch.full(
                            (sum(sends), size),
                            topo.node + 17,
                            device=device,
                            dtype=torch.uint8,
                        ),
                        torch.empty(
                            (sum(receives), size), device=device, dtype=torch.uint8
                        ),
                    )
                    for size in sizes
                ]
                for name, phases in (
                    ("dispatch", (0,)),
                    ("return", (1,)),
                    ("roundtrip", (0, 1)),
                ):

                    def exchange(
                        phases=phases, buffers=buffers, receives=receives, sends=sends
                    ):
                        for phase in phases:
                            send, receive = buffers[phase]
                            transport.exchange(receive, send, receives, sends)

                    exchange()
                    torch.accelerator.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        exchange()
                    dist.barrier()
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(args.iterations):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    elapsed = torch.tensor(
                        begin.elapsed_time(end) / args.iterations, device=device
                    )
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                    for phase in phases:
                        if buffers[phase][1].numel():
                            assert torch.all(buffers[phase][1] == 18 - topo.node).item()
                    results.append(
                        dict(
                            tokens=tokens,
                            phase=name,
                            direction="one_way" if one_way else "duplex",
                            sent_bytes_per_active_lane=sum(sizes[p] for p in phases),
                            milliseconds=elapsed.item(),
                        )
                    )
                    if dist.get_rank() == 0:
                        print(json.dumps(results[-1]), flush=True)
        if dist.get_rank() == 0:
            args.output.write_text(json.dumps(results, indent=2) + "\n")
    finally:
        destroy_hierarchical_groups()
        dist.destroy_process_group(cpu)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
