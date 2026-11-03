# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check fixed dense RDMA against the previous path with real checkpoint rows.

The DEP8 arm processes twice the source rows per rank, matching the global
batch of EP16. Its second island is an identical replica for numerical checks.
Attention and shared experts are excluded. Run on two exclusive eight-GPU hosts.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from benchmark_roce_mega_moe_overlap import capture, measure
from benchmark_roce_mega_moe_replay import CheckpointExperts, error_metrics

from vllm.distributed.hierarchical_mega_moe import (
    HierarchicalMegaMoE,
    destroy_hierarchical_groups,
    get_hierarchical_groups,
)
from vllm.distributed.roce.transport import get_verbs_transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--reference-module", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--tokens", default="24,48,96,192,384,768,1536")
    parser.add_argument("--tile", type=int, default=2048)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--hcas", required=True)
    parser.add_argument("--qps", type=int, default=4)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--stress-replays", type=int, default=100)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    hca = args.hcas.split(",")[local_rank].split(":")[0]
    numa = int((Path("/sys/class/infiniband") / hca / "device/numa_node").read_text())
    cpus = set()
    if numa >= 0:
        for interval in (
            Path(f"/sys/devices/system/node/node{numa}/cpulist")
            .read_text()
            .strip()
            .split(",")
        ):
            bounds = list(map(int, interval.split("-")))
            cpus.update(range(bounds[0], bounds[-1] + 1))
        os.sched_setaffinity(0, cpus & os.sched_getaffinity(0))
    torch.set_num_threads(1)
    shapes = list(map(int, args.tokens.split(",")))
    if min(shapes) <= 0 or 2 * max(shapes) > args.chunk:
        parser.error("Twice each positive source shape must fit --chunk")
    args.experts, args.hidden, args.intermediate, args.top_k = 384, 5120, 2304, 6
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    )
    cpu = dist.new_group(backend="gloo")
    groups = get_hierarchical_groups(dist.group.WORLD, cpu)
    topology = groups.topology
    assert topology.world_size == 16 and topology.local_size == 8
    rank = dist.get_rank()
    spec = importlib.util.spec_from_file_location(
        "dense_reference", args.reference_module
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    transport = get_verbs_transport(
        groups,
        hcas=tuple(args.hcas.split(",")),
        gid_index=3,
        qps=args.qps,
        traffic_class=106,
        timeout_seconds=60,
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    bundles = [
        torch.load(args.bundles / f"layer{args.layer}-rank{r}.pt", weights_only=True)
        for r in range(16)
    ]
    pool = {
        k: torch.cat([b[k] for b in bundles])
        for k in ("hidden", "gates", "ids", "reference")
    }
    args.fast_math = bundles[0]["fast_math"]
    if rank == 0:
        (args.output_directory / "config.json").write_text(
            json.dumps(vars(args), default=str, indent=2)
        )
        print("Loading real checkpoint experts", flush=True)
    wide = CheckpointExperts(
        groups.local_group, range(rank * 24, (rank + 1) * 24), args
    )
    local = CheckpointExperts(
        groups.local_group, range(topology.lane * 48, (topology.lane + 1) * 48), args
    )
    records = []

    def check(actual, expected, label, exact=False):
        metrics = error_metrics(actual, expected)
        okay = (
            torch.equal(actual, expected)
            if exact
            else (
                metrics["normalized_squared_error"] < 1e-5
                and metrics["max_token_relative_l2"] < 0.005
                and metrics["min_token_cosine"] > 0.99998
            )
        )
        checks = [None] * 16
        dist.all_gather_object(
            checks, dict(rank=rank, label=label, okay=okay, **metrics), group=cpu
        )
        if rank == 0:
            with (args.output_directory / "checks.jsonl").open("a") as f:
                f.write(json.dumps(checks) + "\n")
        assert all(c["okay"] for c in checks), label

    for tokens in shapes:
        rows = (torch.arange(tokens) + rank * tokens) % len(pool["hidden"])
        x, gates, ids = [
            pool[k][rows].cuda().contiguous() for k in ("hidden", "gates", "ids")
        ]
        local_rows = torch.cat(
            [
                torch.arange(tokens) + r * tokens
                for r in (topology.lane, topology.lane + 8)
            ]
        ) % len(pool["hidden"])
        lx, lg, li = [
            pool[k][local_rows].cuda().contiguous() for k in ("hidden", "gates", "ids")
        ]
        base_graph, base = capture(partial(local, lx, lg, li))
        expected = base[topology.node * tokens : (topology.node + 1) * tokens]
        cases = [("dep8_equal_global_batch", base_graph, base)]
        callbacks = []
        for name, cls, sparse in [
            ("old_sparse", module.HierarchicalMegaMoE, True),
            ("old_dense", module.HierarchicalMegaMoE, False),
            ("fixed_dense", HierarchicalMegaMoE, False),
        ]:
            adapter = cls(
                groups,
                384,
                args.chunk,
                wire_dtype="mxfp8",
                fused_pack=True,
                source_chunk_size=args.tile,
                pipeline_transport=transport,
                sparse_dispatch=sparse,
                return_dtype="bfloat16",
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
            graph, output = capture(callback)
            check(output, expected, f"{name}_vs_DEP8_t{tokens}")
            cases.append((name, graph, output))
            callbacks.append(callback)
        check(cases[-1][2], cases[-2][2], f"fixed_vs_old_dense_t{tokens}", exact=True)
        original = [v.clone() for v in (x, gates, ids)]
        for mask_all in (False, True):
            x.copy_(original[0] * 0.75)
            gates.copy_(original[1])
            ids.copy_(original[2])
            ids[::3] = -1
            gates[::3] = 0
            if mask_all:
                ids.fill_(-1)
                gates.zero_()
            outputs = []
            for (_, graph, output), callback in zip(cases[1:], callbacks):
                graph.replay()
                torch.accelerator.synchronize()
                dist.barrier(group=cpu)
                fresh = callback()
                check(output, fresh, "changed_graph_vs_fresh", exact=True)
                outputs.append(output)
                if mask_all:
                    check(output, torch.zeros_like(output), "all_masked", exact=True)
            check(outputs[-1], outputs[-2], "changed_fixed_vs_old_dense", exact=True)
        for value, saved in zip((x, gates, ids), original):
            value.copy_(saved)
        for index, (name, graph, output) in enumerate(cases):
            samples = measure(graph, args)
            if index:
                check(output, expected, "timed_" + name)
            records.append(
                dict(
                    tokens_per_ep16_rank=tokens,
                    tokens_per_ep8_rank=2 * tokens,
                    case=name,
                    samples_ms=samples,
                )
            )
            (args.output_directory / f"rank{rank}.json").write_text(
                json.dumps(records, indent=2)
            )
            if args.trace and index:
                transport.set_tracing(True)
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profiler:
                    clock_pair = dict(
                        monotonic_ns=time.monotonic_ns(), wall_ns=time.time_ns()
                    )
                    for _ in range(3):
                        graph.replay()
                    torch.accelerator.synchronize()
                (
                    args.output_directory / f"r{rank}-t{tokens}-{name}.rdma.json"
                ).write_text(
                    json.dumps(
                        dict(
                            clock_pair=clock_pair, transfers=transport.trace_snapshot()
                        )
                    )
                )
                transport.set_tracing(False)
                profiler.export_chrome_trace(
                    str(args.output_directory / f"r{rank}-t{tokens}-{name}.cuda.json")
                )
            if rank == 0:
                print(f"tokens={tokens} {name} rank0_ms={samples}", flush=True)
        del cases, callbacks, base_graph, base, adapter, callback, graph, output
    gathered = [None] * 16
    dist.all_gather_object(gathered, records, group=cpu)
    if rank == 0:
        (args.output_directory / "results.json").write_text(
            json.dumps(gathered, indent=2)
        )
    torch.accelerator.synchronize()
    dist.barrier()
    destroy_hierarchical_groups()
    dist.destroy_process_group(cpu)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
