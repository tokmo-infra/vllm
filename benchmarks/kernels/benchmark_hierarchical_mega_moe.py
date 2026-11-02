# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""torchrun probe: matched synthetic Wide EP and independent local EP islands.

Run only on exclusive SM100 GPUs. Uses actual native MegaMoE kernels and
synthetic FP4 weights with an independent dense oracle, not model accuracy.
"""

import argparse
import json
import os
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.distributed.hierarchical_mega_moe import (
    HierarchicalMegaMoE,
    destroy_hierarchical_groups,
    get_hierarchical_groups,
)
from vllm.distributed.megamoe_rdma_ops import stage_mxfp8
from vllm.model_executor.layers.fused_moe.prepare_megamoe import prepare_megamoe_inputs
from vllm.utils.deep_gemm import _import_deep_gemm


class NativeExperts:
    def __init__(self, group, expert_ids, args):
        self.dg = _import_deep_gemm()
        self.expert_ids = list(expert_ids)
        self.width = args.hidden
        e, h, inter = len(expert_ids), args.hidden, args.intermediate
        self.buffer = self.dg.get_symm_buffer_for_mega_moe(
            group,
            e * dist.get_world_size(group),
            args.chunk,
            args.top_k,
            h,
            inter,
            mma_type="fp8xfp4",
            activation="swiglu",
        )
        self.l1 = self.weights(expert_ids, 2 * inter, h, 7)
        self.l2 = self.weights(expert_ids, h, inter, 5)
        self.fast_math = args.fast_math

    @staticmethod
    def weights(experts, n, k, modulus):
        w = torch.full((len(experts), n, k // 2), 0x22, device="cuda", dtype=torch.int8)
        sf = torch.empty((len(experts), k // 128, n), device="cuda", dtype=torch.int32)
        for index, expert in enumerate(experts):
            sf[index].fill_((124 - expert % modulus) * 0x01010101)
        return w, sf.transpose(-1, -2)

    def __call__(self, hidden, gates, ids, direct=False):
        count = hidden.shape[0]
        b = self.buffer
        stage = stage_mxfp8 if direct else prepare_megamoe_inputs
        stage(
            hidden,
            gates,
            ids,
            b.x[:count],
            b.x_sf[:count],
            b.topk_idx[:count],
            b.topk_weights[:count],
        )
        output = torch.empty((count, self.width), device="cuda", dtype=torch.bfloat16)
        self.dg.fp8_fp4_mega_moe(
            output,
            self.l1,
            self.l2,
            b,
            activation_clamp=10.0,
            fast_math=self.fast_math,
        )
        return output

    def direct(self, hidden, gates, ids):
        return self(hidden, gates, ids, direct=True)


def oracle(x, gates, ids, intermediate):
    experts = ids.clamp_min(0)
    a = x.float().sum(1, keepdim=True) * torch.exp2(-3.0 - experts % 7)
    values = (
        torch.nn.functional.silu(a.clamp(max=10))
        * a.clamp(-10, 10)
        * intermediate
        * torch.exp2(-3.0 - experts % 5)
    )
    return (torch.where(ids >= 0, values, 0) * gates).sum(1, keepdim=True).expand_as(x)


def check_replay(fn, model, x, gates, ids, args, shift):
    """Reuse one graph across new inputs, masks and cross-island physical moves."""
    originals = [tensor.clone() for tensor in (x, gates, ids)]
    graph = torch.cuda.CUDAGraph()
    fn()
    torch.accelerator.synchronize()
    with torch.cuda.graph(graph):
        output = fn()
    x.mul_(0.75)
    gates[::3].zero_()
    ids.copy_((originals[2] - shift) % args.experts)
    ids[::3] = -1
    for slot, physical in enumerate(model.expert_ids):
        logical = (physical + shift) % args.experts
        model.l1[1][slot].fill_((124 - logical % 7) * 0x01010101)
        model.l2[1][slot].fill_((124 - logical % 5) * 0x01010101)
    graph.replay()
    expected = oracle(x, gates, originals[2], args.intermediate)
    torch.testing.assert_close(output.float(), expected, rtol=0.08, atol=0.01)
    assert torch.count_nonzero(output[::3]).item() == 0
    for slot, logical in enumerate(model.expert_ids):
        model.l1[1][slot].fill_((124 - logical % 7) * 0x01010101)
        model.l2[1][slot].fill_((124 - logical % 5) * 0x01010101)
    for target, original in zip((x, gates, ids), originals):
        target.copy_(original)
    graph.replay()
    torch.testing.assert_close(
        output.float(), oracle(x, gates, ids, args.intermediate), rtol=0.08, atol=0.01
    )


def measure(fn, args):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    if args.cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = fn()
        run = graph.replay
    else:
        result = fn()
        run = fn
    for _ in range(args.stress_replays):
        run()
    torch.accelerator.synchronize()
    dist.barrier()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(args.iterations):
        run()
    end.record()
    end.synchronize()
    elapsed = torch.tensor(start.elapsed_time(end) / args.iterations, device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return elapsed.item(), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=2304)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--tokens", default="768,936,1024,1536,4096")
    parser.add_argument("--source-chunks", default="256,512,1024")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--stress-replays", type=int, default=1000)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--fast-math", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    )
    cpu = dist.new_group(backend="gloo")
    groups = get_hierarchical_groups(dist.group.WORLD, cpu)
    topo = groups.topology
    rank = dist.get_rank()
    if args.experts % topo.world_size:
        raise ValueError("Experts must divide global EP size")
    wide_per_rank = args.experts // topo.world_size
    local_per_rank = args.experts // topo.local_size
    wide = NativeExperts(
        groups.local_group,
        range(rank * wide_per_rank, (rank + 1) * wide_per_rank),
        args,
    )
    local = NativeExperts(
        groups.local_group,
        range(topo.lane * local_per_rank, (topo.lane + 1) * local_per_rank),
        args,
    )
    records = []
    try:
        for tokens in map(int, args.tokens.split(",")):
            torch.manual_seed(1234 + rank)
            x = (torch.rand(tokens, args.hidden, device="cuda") * 0.01).bfloat16()
            ids = (
                torch.rand(tokens, args.experts, device="cuda")
                .topk(args.top_k, dim=1)
                .indices
            )
            gates = torch.rand(tokens, args.top_k, device="cuda").softmax(1)
            expected = oracle(x, gates, ids, args.intermediate)
            cases = [
                ("serial_mxfp8_reconstruct", False, 0),
                ("serial_mxfp8_direct", True, 0),
            ]
            cases += [
                (f"chunked_{chunk}", True, chunk)
                for chunk in map(int, filter(None, args.source_chunks.split(",")))
            ]
            for name, direct, chunk in cases:
                adapter = HierarchicalMegaMoE(
                    groups,
                    args.experts,
                    args.chunk,
                    wire_dtype="mxfp8",
                    fused_pack=True,
                    source_chunk_size=chunk,
                )

                forward = partial(
                    adapter.forward,
                    x,
                    gates,
                    ids,
                    wide,
                    fixed_capacity=True,
                    local_moe_mxfp8=wide.direct if direct else None,
                )
                if rank == 0:
                    print(f"checking {name} tokens={tokens}", flush=True)
                actual = forward()
                torch.testing.assert_close(
                    actual.float(), expected, rtol=0.08, atol=0.01
                )
                if args.cuda_graph:
                    check_replay(
                        forward,
                        wide,
                        x,
                        gates,
                        ids,
                        args,
                        wide_per_rank * topo.local_size,
                    )
                elapsed, actual = measure(forward, args)
                torch.testing.assert_close(
                    actual.float(), expected, rtol=0.08, atol=0.01
                )
                records.append(dict(tokens=tokens, case=name, milliseconds=elapsed))
                if rank == 0:
                    print(json.dumps(records[-1]), flush=True)
            elapsed, actual = measure(partial(local, x, gates, ids), args)
            torch.testing.assert_close(actual.float(), expected, rtol=0.08, atol=0.01)
            records.append(
                dict(tokens=tokens, case="independent_local_ep", milliseconds=elapsed)
            )
            if rank == 0:
                print(json.dumps(records[-1]), flush=True)
        if rank == 0:
            args.output.write_text(
                json.dumps(
                    dict(
                        config=vars(args) | {"output": str(args.output)},
                        results=records,
                    ),
                    indent=2,
                )
                + "\n"
            )
    finally:
        destroy_hierarchical_groups()
        dist.destroy_process_group(cpu)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
