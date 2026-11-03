# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare native issue-order and readiness scheduling on exclusive SM100 GPUs."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from benchmark_hierarchical_mega_moe import NativeExperts
from benchmark_roce_mega_moe_overlap import capture, measure

from vllm.model_executor.layers.fused_moe.prepare_megamoe import prepare_megamoe_inputs
from vllm.utils.deep_gemm import _import_deep_gemm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="1,31,256,1536,3072,4097")
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=2304)
    parser.add_argument("--experts", type=int, default=192)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--shared", type=int, choices=(0, 1), default=0)
    parser.add_argument("--shared-rows", type=int, default=None)
    parser.add_argument("--routing-groups", type=int, default=1)
    parser.add_argument("--ordered-target", action="store_true")
    parser.add_argument("--early-combine", action="store_true")
    parser.add_argument("--no-return-trace", action="store_true")
    parser.add_argument("--return-chunk-size", type=int, default=64)
    parser.add_argument("--check-stability", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--stress-replays", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shared_rows is not None and (not args.shared or args.shared_rows < 0):
        parser.error("--shared-rows requires --shared=1 and a nonnegative count")
    local = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    cpu = dist.new_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    dg = _import_deep_gemm()
    per_rank = args.experts // world
    sizes = list(map(int, args.tokens.split(",")))
    buffer = dg.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        args.experts,
        max(sizes),
        args.top_k,
        args.hidden,
        args.intermediate,
        num_shared_experts=args.shared,
        mma_type="fp8xfp4",
        routing_groups=args.routing_groups,
    )
    # Independent workspaces expose stale replay flags. Alternating graph IDs
    # on a single workspace can accidentally hide an invalid epoch scheme.
    reference_buffer = dg.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        args.experts,
        max(sizes),
        args.top_k,
        args.hidden,
        args.intermediate,
        num_shared_experts=args.shared if args.shared_rows is None else 0,
        mma_type="fp8xfp4",
    )
    shared_reference_buffer = (
        dg.get_symm_buffer_for_mega_moe(
            dist.group.WORLD,
            args.experts,
            max(sizes),
            args.top_k,
            args.hidden,
            args.intermediate,
            num_shared_experts=args.shared,
            mma_type="fp8xfp4",
        )
        if args.shared_rows is not None
        else None
    )
    torch.manual_seed(123 + rank)
    experts = range(rank * per_rank, (rank + 1) * per_rank)
    l1 = NativeExperts.weights(experts, 2 * args.intermediate, args.hidden, 7)
    l2 = NativeExperts.weights(experts, args.hidden, args.intermediate, 5)
    # Signed FP4 values exercise cancellation as well as positive reductions.
    l1[0].random_(-128, 128)
    l2[0].random_(-128, 128)
    kwargs = {}
    if args.shared:
        for name, n, k in (
            ("shared_l1_weights", 2 * args.intermediate, args.hidden),
            ("shared_l2_weights", args.hidden, args.intermediate),
        ):
            weight = (torch.randn(n, k, device="cuda") * 0.03).to(torch.float8_e4m3fn)
            scale = torch.full(
                (k // 128, n), 0x7F7F7F7F, dtype=torch.int32, device="cuda"
            ).t()
            kwargs[name] = (weight, scale)
    blocks = (
        buffer.num_max_tokens_per_rank
        * world
        * min(args.top_k, per_rank * args.routing_groups)
        + 7
    ) // 8 + per_rank * args.routing_groups
    scratch = torch.empty((4, blocks + 1), dtype=torch.int64, device="cuda")
    records = []
    for tokens in sizes:
        x = (torch.randn(tokens, args.hidden, device="cuda") * 0.01).bfloat16()
        ids = (
            torch.rand(tokens, args.experts, device="cuda").topk(args.top_k, 1).indices
        )
        gates = torch.rand(tokens, args.top_k, device="cuda").softmax(1)
        saved_x, saved_ids = x.clone(), ids.clone()
        shared_count = (
            min(tokens, args.shared_rows) if args.shared_rows is not None else 0
        )
        shared_actual = (
            torch.empty(
                (shared_count, args.hidden), dtype=torch.bfloat16, device="cuda"
            )
            if args.shared_rows is not None
            else None
        )
        block_m = dg.get_block_m_for_mega_moe(
            world,
            buffer.num_experts,
            buffer.num_max_tokens_per_rank,
            tokens,
            args.top_k,
            "fp8xfp4",
        )
        reference_block_m = dg.get_block_m_for_mega_moe(
            world,
            args.experts,
            reference_buffer.num_max_tokens_per_rank,
            tokens,
            args.top_k,
            "fp8xfp4",
        )
        group_ids = (
            torch.arange(tokens, device="cuda")
            // ((tokens + args.routing_groups - 1) // args.routing_groups)
        )[:, None]

        descriptors = (
            torch.zeros(
                ((tokens + args.return_chunk_size - 1) // args.return_chunk_size, 4),
                dtype=torch.int64,
                device="cuda",
            )
            if args.early_combine and not args.no_return_trace
            else None
        )

        def forward(
            ready,
            x=x,
            gates=gates,
            ids=ids,
            tokens=tokens,
            block_m=block_m,
            reference_block_m=reference_block_m,
            group_ids=group_ids,
            shared_actual=shared_actual,
        ):
            b = buffer if ready else reference_buffer
            routes = ids
            if ready and args.routing_groups != 1:
                routes = torch.where(
                    ids >= 0,
                    (ids // per_rank) * (per_rank * args.routing_groups)
                    + group_ids * per_rank
                    + ids % per_rank,
                    -1,
                )
            prepare_megamoe_inputs(
                x,
                gates,
                routes,
                b.x[:tokens],
                b.x_sf[:tokens],
                b.topk_idx[:tokens],
                b.topk_weights[:tokens],
                shared_x_sf=b.shared_l1_acts_sf if b.num_shared_experts else None,
                shared_block_m=(block_m if ready else reference_block_m)
                if b.num_shared_experts
                else None,
            )
            y = torch.empty_like(x)
            dg.fp8_fp4_mega_moe(
                y,
                l1,
                l2,
                b,
                activation_clamp=10.0,
                fast_math=False,
                scheduler_workspace=scratch
                if ready and not args.ordered_target
                else None,
                shared_output=shared_actual if ready else None,
                early_combine=args.early_combine and ready,
                return_descriptors=descriptors if ready else None,
                return_chunk_size=args.return_chunk_size if ready else 0,
                **(kwargs if ready or args.shared_rows is None else {}),
            )
            return y

        def shared_reference(x=x, gates=gates, ids=ids, shared_count=shared_count):
            y = torch.empty_like(x[:shared_count])
            if shared_count:
                b = shared_reference_buffer
                prepare_megamoe_inputs(
                    x[:shared_count],
                    torch.zeros_like(gates[:shared_count]),
                    torch.full_like(ids[:shared_count], -1),
                    b.x[:shared_count],
                    b.x_sf[:shared_count],
                    b.topk_idx[:shared_count],
                    b.topk_weights[:shared_count],
                    shared_x_sf=b.shared_l1_acts_sf,
                    shared_block_m=dg.get_block_m_for_mega_moe(
                        world,
                        args.experts,
                        b.num_max_tokens_per_rank,
                        shared_count,
                        args.top_k,
                        "fp8xfp4",
                    ),
                )
                dg.fp8_fp4_mega_moe(
                    y, l1, l2, b, activation_clamp=10.0, fast_math=False, **kwargs
                )
            return y

        if rank == 0:
            print(
                f"capture tokens={tokens} block_m={block_m} "
                f"reference_block_m={reference_block_m} shared={args.shared}",
                flush=True,
            )
        if args.shared_rows is None:
            baseline, expected = capture(lambda: forward(False))
        else:
            baseline, (expected, shared_expected) = capture(
                lambda: (forward(False), shared_reference())
            )
        ready, actual = capture(lambda: forward(True))
        for generation in range(4):
            if rank == 0:
                print(f"checking tokens={tokens} generation={generation}", flush=True)
            x.copy_(saved_x * (1 - generation / 8))
            ids.copy_(saved_ids)
            if generation == 1:
                ids[::3] = -1
            elif generation == 2:
                ids.copy_(torch.arange(args.top_k, device="cuda").expand_as(ids))
            elif generation == 3:
                ids.fill_(-1)
            baseline.replay()
            ready.replay()
            if args.check_stability:
                for name, graph, output in (
                    ("baseline", baseline, expected),
                    ("target", ready, actual),
                ):
                    snapshot = output.clone()
                    for repeat in range(8):
                        graph.replay()
                        label = f"{name} generation={generation} repeat={repeat}"
                        torch.testing.assert_close(
                            output,
                            snapshot,
                            rtol=0,
                            atol=0,
                            msg=lambda message, label=label: f"{label}: {message}",
                        )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if shared_actual is not None:
                torch.testing.assert_close(
                    shared_actual, shared_expected, rtol=0, atol=0
                )
        x.copy_(saved_x)
        ids.copy_(saved_ids)
        target_name = "ordered_target" if args.ordered_target else "ready"
        for name, graph in (("baseline", baseline), (target_name, ready)):
            samples = measure(graph, args)
            records.append(
                dict(tokens=tokens, block_m=block_m, case=name, samples_ms=samples)
            )
        baseline.replay()
        ready.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if shared_actual is not None:
            torch.testing.assert_close(shared_actual, shared_expected, rtol=0, atol=0)
        if descriptors is not None:
            timestamps = descriptors[:, 2:].cpu().tolist()
            assert all(
                published > 0 and finished > 0 for published, finished in timestamps
            )
            records.append(
                dict(tokens=tokens, case="return_publication", timestamps_ns=timestamps)
            )
        if rank == 0:
            print(f"validated tokens={tokens} shared={args.shared}", flush=True)
        del baseline, ready
    gathered = [None] * world
    dist.all_gather_object(gathered, records, group=cpu)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                dict(
                    config=vars(args) | {"output": str(args.output)}, per_rank=gathered
                ),
                indent=2,
            )
            + "\n"
        )
    torch.accelerator.synchronize()
    dist.barrier()
    buffer.destroy()
    if reference_buffer is not buffer:
        reference_buffer.destroy()
    dist.destroy_process_group(cpu)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
