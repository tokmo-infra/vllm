# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare independent DEP8 and Wide EP using captured routed-layer inputs.

Checkpoint loading, input sampling, and checks are outside CUDA graph timing.
Large shapes cycle through the captured pool; they are shape replays rather
than a simultaneous production batch. Shared experts and attention are excluded.
"""

import argparse
import json
import os
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from benchmark_hierarchical_mega_moe import NativeExperts
from benchmark_roce_mega_moe_overlap import capture, measure
from safetensors import safe_open

from vllm.distributed.hierarchical_mega_moe import (
    HierarchicalMegaMoE,
    destroy_hierarchical_groups,
    get_hierarchical_groups,
)
from vllm.distributed.megamoe_rdma_ops import pack_mxfp8, unpack_mxfp8
from vllm.distributed.roce.transport import get_verbs_transport
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    DeepGemmMegaMoEExperts,
)


class CheckpointExperts(NativeExperts):
    _pack_scale = staticmethod(DeepGemmMegaMoEExperts._pack_scale)
    _kernel_layout_shard = staticmethod(DeepGemmMegaMoEExperts._kernel_layout_shard)

    def __init__(self, group, expert_ids, args):
        super().__init__(group, expert_ids, args)
        weights = json.loads((args.model / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        needed = {}
        for slot, expert in enumerate(self.expert_ids):
            for name in ("w1", "w2", "w3"):
                for suffix in ("weight", "scale"):
                    key = f"layers.{args.layer}.ffn.experts.{expert}.{name}.{suffix}"
                    needed.setdefault(weights[key], []).append(
                        (key, slot, name, suffix)
                    )
        for filename, entries in needed.items():
            with safe_open(
                args.model / filename, framework="pt", device="cpu"
            ) as source:
                for key, slot, name, suffix in entries:
                    value = source.get_tensor(key).to("cuda")
                    pair = self.l2 if name == "w2" else self.l1
                    is_scale = suffix == "scale"
                    DeepGemmMegaMoEExperts._load_weight(
                        self, pair[int(is_scale)][slot], value, name, is_scale
                    )


class QuantizedReturnReference(HierarchicalMegaMoE):
    """Round remote partials, then send BF16 to check MXFP8 wire and reduction."""

    def _compute(self, slot, *args):
        super()._compute(slot, *args)
        for peer, part in enumerate(
            slot.partial.chunk(len(self.groups.topology.nodes))
        ):
            if peer != self.pipeline_transport.rank:
                part.copy_(unpack_mxfp8(pack_mxfp8(part), part.shape[1]))


def error_metrics(actual, expected):
    a, b = actual.float(), expected.float()
    error = a - b
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=1)
    cosine = torch.where((a == b).all(1), 1.0, cosine)
    return dict(
        max_abs=error.abs().max().item(),
        relative_l2=(error.norm() / b.norm().clamp_min(1e-12)).item(),
        max_token_relative_l2=(error.norm(dim=1) / b.norm(dim=1).clamp_min(1e-12))
        .max()
        .item(),
        min_token_cosine=cosine.min().item(),
        normalized_squared_error=(
            error.square().sum()
            / (a.square().sum() + b.square().sum()).clamp_min(1e-12)
        ).item(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="Captured pool .pt")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--tokens", default="768,1536,4096,6144")
    parser.add_argument("--tile", type=int, default=2048)
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--reference-version", action="store_true")
    parser.add_argument(
        "--fixed-input-only",
        action="store_true",
        help="Time original captured inputs; omit changed-input replay checks",
    )
    parser.add_argument("--hcas", required=True)
    parser.add_argument("--gid", type=int, default=3)
    parser.add_argument("--traffic-class", type=int, default=106)
    parser.add_argument("--qps", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--stress-replays", type=int, default=100)
    args = parser.parse_args()
    args.experts, args.hidden, args.intermediate, args.top_k = 384, 5120, 2304, 6
    shapes = list(map(int, args.tokens.split(",")))
    if min(shapes) <= 0 or max(shapes) > args.chunk:
        parser.error("Positive token shapes must fit --chunk for the DEP8 call")
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", torch.accelerator.current_device_index())
    )
    cpu = dist.new_group(backend="gloo")
    groups = get_hierarchical_groups(dist.group.WORLD, cpu)
    topology = groups.topology
    if topology.local_size != 8:
        raise ValueError("DEP8 comparison requires eight GPUs per host")
    rank, size = dist.get_rank(), dist.get_world_size()
    transport = get_verbs_transport(
        groups,
        hcas=tuple(args.hcas.split(",")),
        gid_index=args.gid,
        qps=args.qps,
        traffic_class=args.traffic_class,
        timeout_seconds=args.timeout,
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records = []

    def check(actual, expected, label, exact=False):
        metrics = error_metrics(actual, expected)
        okay = (
            torch.equal(actual, expected)
            if exact
            else metrics["normalized_squared_error"] < 1e-5
            and metrics["max_token_relative_l2"] < 0.005
            and metrics["min_token_cosine"] > 0.99998
        )
        statuses = [None] * size
        dist.all_gather_object(
            statuses, dict(rank=rank, label=label, okay=okay, **metrics), group=cpu
        )
        if rank == 0:
            with (args.output_directory / "checks.jsonl").open("a") as stream:
                stream.write(json.dumps(statuses) + "\n")
        if not all(entry["okay"] for entry in statuses):
            raise AssertionError(f"Numerical check failed: {label}; see checks.jsonl")
        return metrics

    pool = torch.load(args.inputs, weights_only=True, map_location="cpu")
    args.fast_math = pool["fast_math"]
    assert pool["kernel_kwargs"] == {"activation_clamp": 10.0}
    if rank == 0:
        (args.output_directory / "config.json").write_text(
            json.dumps(
                vars(args) | {"pool_rows": pool["hidden"].shape[0]},
                default=str,
                indent=2,
            )
            + "\n"
        )
        print("loading checkpoint experts for DEP8 and Wide EP", flush=True)
    per_rank = args.experts // size
    wide = CheckpointExperts(
        groups.local_group, range(rank * per_rank, (rank + 1) * per_rank), args
    )
    local = CheckpointExperts(
        groups.local_group,
        range(topology.lane * 48, (topology.lane + 1) * 48),
        args,
    )
    for tokens in shapes:
        if rank == 0:
            print(f"replaying layer={args.layer} tokens={tokens}", flush=True)
        rows = (torch.arange(tokens) + rank * tokens) % pool["hidden"].shape[0]
        x, gates, ids, recorded = (
            pool[key][rows].to("cuda").contiguous()
            for key in ("hidden", "gates", "ids", "reference")
        )
        base_graph, base = capture(partial(local, x, gates, ids))
        check(base, recorded, "DEP8_vs_captured")
        cases = [("dep8", base_graph, base)]
        callbacks = []
        modes = ["bfloat16"] if args.reference_version else ["bfloat16", "mxfp8"]
        for mode in modes:
            options = (
                {}
                if args.reference_version
                else dict(sparse_dispatch=True, return_dtype=mode)
            )
            adapter = HierarchicalMegaMoE(
                groups,
                args.experts,
                args.chunk,
                wire_dtype="mxfp8",
                fused_pack=True,
                source_chunk_size=args.tile,
                pipeline_transport=transport,
                **options,
            )
            callback = partial(
                adapter.forward,
                x,
                gates,
                ids,
                wide,
                fixed_capacity=True,
                local_moe_mxfp8=wide.direct,
            )
            callbacks.append(callback)
            graph, output = capture(callback)
            if mode == "bfloat16":
                check(output, base, "BF16_Wide_EP_vs_DEP8")
            else:
                quantized = QuantizedReturnReference(
                    groups,
                    args.experts,
                    args.chunk,
                    wire_dtype="mxfp8",
                    fused_pack=True,
                    source_chunk_size=args.tile,
                    pipeline_transport=transport,
                    sparse_dispatch=True,
                )
                expected = quantized.forward(
                    x,
                    gates,
                    ids,
                    wide,
                    fixed_capacity=True,
                    local_moe_mxfp8=wide.direct,
                )
                check(output, expected, "MXFP8_wire_vs_quantized_BF16", exact=True)
                del expected, quantized
            cases.append((f"wide_{mode}", graph, output))
        # Compare changed graph inputs against fresh execution in the same
        # topology. Cross-topology cancellation is reported on original data.
        if tokens == shapes[0] and not args.fixed_input_only:
            original = [value.clone() for value in (x, gates, ids)]
            for masked in (False, True):
                x.copy_(original[0] * 0.75)
                ids.copy_(original[2])
                gates.copy_(original[1])
                ids[::3] = -1
                gates[::3] = 0
                if masked:
                    ids.fill_(-1)
                    gates.zero_()
                for (_, graph, output), callback in zip(cases[1:], callbacks):
                    graph.replay()
                    torch.accelerator.synchronize()
                    dist.barrier(group=cpu)
                    fresh = callback()
                    check(output, fresh, "changed_graph_vs_fresh", exact=True)
                    if masked:
                        check(
                            output, torch.zeros_like(output), "all_masked", exact=True
                        )
                    del fresh
            for value, source in zip((x, gates, ids), original):
                value.copy_(source)
            del original
        # Share warmup and clock bias between cases across shape passes.
        if tokens // 256 % 2:
            cases.reverse()
        for name, graph, output in cases:
            samples = measure(graph, args)
            if name != "wide_mxfp8":
                check(output, base, "timed_" + name)
            records.append(
                dict(
                    layer=args.layer,
                    tokens=tokens,
                    tile=args.tile,
                    case=name,
                    samples_ms=samples,
                    error_vs_dep8=error_metrics(output, base),
                )
            )
            (args.output_directory / f"rank{rank}.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
        del cases, callbacks, base_graph, base, graph, output, callback, adapter
    all_records = [None] * size
    dist.all_gather_object(all_records, records, group=cpu)
    if rank == 0:
        (args.output_directory / "results.json").write_text(
            json.dumps(dict(per_rank=all_records), indent=2) + "\n"
        )
    torch.accelerator.synchronize()
    dist.barrier()
    destroy_hierarchical_groups()
    dist.destroy_process_group(cpu)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
