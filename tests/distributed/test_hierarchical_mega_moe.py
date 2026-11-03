# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU/Gloo proves network routing and reduction independently of GPU kernels."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Keep this correctness test usable without a CUDA vLLM installation.
_root = Path(__file__).parents[2] / "vllm/distributed"
_transport_spec = importlib.util.spec_from_file_location(
    "vllm.distributed.megamoe_transport", _root / "megamoe_transport.py"
)
assert _transport_spec is not None and _transport_spec.loader is not None
_transport = importlib.util.module_from_spec(_transport_spec)
sys.modules[_transport_spec.name] = _transport
_transport_spec.loader.exec_module(_transport)
_path = _root / "hierarchical_mega_moe.py"
_spec = importlib.util.spec_from_file_location("_hierarchical_moe_test", _path)
assert _spec is not None and _spec.loader is not None
hier = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = hier
_spec.loader.exec_module(hier)


def test_graph_warmup_uses_uniform_cpu_counts_without_a_graph_mode():
    # V2 uses mode NONE inside full capture. Counts, not the mode, establish
    # the equal-size communication contract for warmup and capture alike.
    assert hier.has_uniform_token_counts(6, torch.tensor([6, 6, 6, 6]), 4)
    assert hier.has_uniform_token_counts(0, torch.zeros(4, dtype=torch.int32), 4)
    assert not hier.has_uniform_token_counts(6, torch.tensor([6, 0, 6, 6]), 4)
    assert not hier.has_uniform_token_counts(6, torch.tensor([6, 6]), 4)
    assert not hier.has_uniform_token_counts(6, None, 4)
    assert not hier.has_uniform_token_counts(6, torch.empty(4, device="meta"), 4)


def test_topology_rejects_unequal_islands():
    with pytest.raises(ValueError, match="equal-sized"):
        hier.IslandTopology.from_hosts(("a", "a", "b"), 0)


def _distributed_case(rank, hosts, rendezvous, source_chunk_size):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=len(hosts)
    )
    groups = hier.HierarchicalGroups.create(
        list(range(len(hosts))), hosts, rank, "gloo"
    )
    topo = groups.topology
    per_rank = 4
    experts = per_rank * len(hosts)
    adapter = hier.HierarchicalMegaMoE(
        groups, experts, max_tokens_per_chunk=2, source_chunk_size=source_chunk_size
    )
    try:
        # Zero source rows, uneven receives and several chunk rounds occur together.
        count = (0, 1, 5, 2, 3, 0, 2, 4)[rank]
        hidden = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / 10
        hidden += rank
        ids = torch.tensor(
            [
                [0, 1, (i * 3 + rank) % experts, (i * 7 + rank + 3) % experts]
                for i in range(count)
            ],
            dtype=torch.int64,
        ).reshape(count, 4)
        weights = torch.full((count, 4), 0.25)
        calls = 0

        def local_experts(x, gates, routes):
            nonlocal calls
            calls += 1
            assert x.shape[0] <= 2
            # Cooperative kernels must see one common per-round tile shape,
            # including ranks whose real receive count is zero.
            sizes = [None] * topo.local_size
            dist.all_gather_object(sizes, x.shape[0], group=groups.local_group)
            assert len(set(sizes)) == 1
            assert (
                (routes == -1) | ((routes >= 0) & (routes < per_rank * topo.local_size))
            ).all()
            # A deterministic expert y=(expert_id+1)*x is an independent oracle.
            owners = torch.tensor(topo.nodes[topo.node])
            global_ids = owners[routes.clamp_min(0) // per_rank] * per_rank
            global_ids += routes.remainder(per_rank)
            gains = torch.where(routes >= 0, global_ids + 1, 0)
            return x * (gains * gates).sum(dim=1, keepdim=True)

        actual = adapter.forward(hidden, weights, ids, local_experts)
        expected = hidden * ((ids + 1) * weights).sum(dim=1, keepdim=True)
        torch.testing.assert_close(actual, expected)
        # The source, not every destination island, owns the shared expert.
        torch.testing.assert_close(actual + 2 * hidden, expected + 2 * hidden)
        assert calls > 0
        # Even islands with no selected experts must participate in the same
        # cooperative kernel calls and inverse exchanges.
        one_host_ids = torch.arange(4).expand(count, 4)
        actual = adapter.forward(hidden, weights, one_host_ids, local_experts)
        torch.testing.assert_close(actual, hidden * 2.5)
        # A globally empty step must not enter a cooperative local kernel.
        before = calls
        empty = adapter.forward(hidden[:0], weights[:0], ids[:0], local_experts)
        assert empty.shape == (0, 3) and calls == before
        _check_fixed_capacity_dispatch(adapter, topo, per_rank, experts, rank)
    finally:
        groups.close()
        dist.destroy_process_group()


def _check_fixed_capacity_dispatch(adapter, topo, per_rank, experts, rank):
    """Padded routing stays correct when physical slots move across hosts."""
    count = 5
    x = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / 10 + rank
    logical = (torch.arange(count * 4).reshape(count, 4) + rank) % experts
    gates = torch.tensor([0.1, 0.2, 0.3, 0.4]).expand(count, 4).clone()
    padded = (torch.arange(count) + rank) % 3 == 0
    gates[padded] = 0
    expected = x * ((logical + 1) * gates).sum(dim=1, keepdim=True)

    for shift in (0, per_rank):
        physical = torch.where(padded[:, None], -1, (logical - shift) % experts)

        def resident_experts(hidden, weights, routes, shift=shift):
            assert hidden.shape[0] <= adapter.chunk
            owners = torch.tensor(topo.nodes[topo.node])
            slots = owners[routes.clamp_min(0) // per_rank] * per_rank
            slots += routes.remainder(per_rank)
            values = (slots + shift) % experts + 1
            values = torch.where(routes >= 0, values, 0)
            return hidden * (values * weights).sum(dim=1, keepdim=True)

        actual = adapter.forward(
            x, gates, physical, resident_experts, fixed_capacity=True
        )
        torch.testing.assert_close(actual, expected)
        assert adapter.last_dispatch_tokens == len(topo.nodes) * count
    empty = adapter.forward(
        x[:0], gates[:0], physical[:0], resident_experts, fixed_capacity=True
    )
    assert empty.shape == (0, 3)


@pytest.mark.parametrize(
    "hosts",
    [
        ("a", "b", "c", "b", "c", "a"),
        ("a", "b", "c", "d"),
        ("a", "b", "c", "d", "d", "c", "b", "a"),
        ("a", "b", "b", "a"),
        ("a", "a"),
    ],
)
@pytest.mark.parametrize("source_chunk_size", [0, 2])
def test_distributed_inverse_combine_on_one_to_four_islands(
    tmp_path, hosts, source_chunk_size
):
    mp.spawn(
        _distributed_case,
        args=(hosts, (tmp_path / "rendezvous").as_uri(), source_chunk_size),
        nprocs=len(hosts),
        join=True,
    )
