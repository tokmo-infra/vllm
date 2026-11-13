# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for cross-rank ownership and differently packed PD imports."""

import pytest
import torch

from vllm.v1.hisparse.dcp import HiSparseDCPShard, import_page_offset, pair_import_pages
from vllm.v1.hisparse.types import SparseKVRowMirror

pytestmark = pytest.mark.cpu_test


def test_kimi_layer_builders_share_one_workspace_without_cross_model_retention():
    """Kimi creates per-layer builders around a single model-wide top-k tensor."""
    import gc
    import weakref

    from vllm.v1.attention.backends.mla.index_group import SparseMLAIndexGroupBuilder

    topk = torch.empty(10, 2048, dtype=torch.int32)
    builders = [SparseMLAIndexGroupBuilder(topk, 10) for _ in range(24)]
    workspace = builders[0].hisparse_decode_buffers
    assert all(b.hisparse_decode_buffers is workspace for b in builders)
    independent = SparseMLAIndexGroupBuilder(torch.empty_like(topk), 10)
    assert independent.hisparse_decode_buffers is not workspace
    topk_ref = weakref.ref(topk)
    del builders, topk
    gc.collect()
    assert topk_ref() is None


@pytest.mark.parametrize("interleave", [1, 4, 64])
def test_localized_topk_is_a_contiguous_prefix_for_sparse_decode(interleave):
    positions = torch.tensor([[7, 0, 65, 9, 1, 520, -1, 64, 511, 3]])
    original = positions.clone()
    for rank in range(8):
        shard = HiSparseDCPShard(8, rank, interleave)
        localized = shard.localize(positions)
        expected = [
            p // (8 * interleave) * interleave + p % interleave
            for p in positions[0].tolist()
            if p >= 0 and p // interleave % 8 == rank
        ]
        # Sparse MLA reads only valid_count entries; holes lose owned KV rows.
        assert localized[0, : len(expected)].tolist() == expected
        assert torch.all(localized[0, len(expected) :] == -1)
    assert torch.equal(positions, original)


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("interleave", [1, 4, 64])
def test_owned_rows_partition_history_and_verification_topk(size, interleave):
    positions = torch.tensor([[-1, *range(size * interleave * 3 + 7)]])
    owners = torch.zeros_like(positions)
    for rank in range(size):
        shard = HiSparseDCPShard(size, rank, interleave)
        local = shard.localize(positions)
        valid = local[local >= 0]
        global_positions = (
            valid // interleave * size * interleave
            + rank * interleave
            + valid % interleave
        )
        owners[0, global_positions + 1] += 1
        expected = [
            p
            for p in positions[0].tolist()
            if p >= 0 and p // interleave % size == rank
        ]
        assert local[local >= 0].tolist() == list(range(len(expected)))
        for position in range(positions.shape[1]):
            reference = sum(p // interleave % size == rank for p in range(position))
            assert shard.count_before(position) == reference
    assert owners[0, 0] == 0
    assert torch.all(owners[0, 1:] == 1)


@pytest.mark.parametrize("interleave", [1, 4, 64])
def test_mirrors_preserve_physical_block_ids_and_partial_interleaves(interleave):
    block_size, size = 64, 8
    virtual = block_size * size
    for offset, count in [(0, 1), (3, 137), (virtual - 9, 9), (0, virtual)]:
        mirror = SparseKVRowMirror(
            (7 * virtual + offset, 9 * virtual + offset), 11 * virtual + offset, count
        )
        total = 0
        for rank in range(size):
            shard = HiSparseDCPShard(size, rank, interleave)
            expected_offsets = [
                p
                for p in range(offset, offset + count)
                if p // interleave % size == rank
            ]
            result = shard.mirror(mirror, block_size)
            if not expected_offsets:
                assert result is None
                continue
            local_offset = sum(
                p // interleave % size == rank for p in range(expected_offsets[0])
            )
            assert result == SparseKVRowMirror(
                (7 * block_size + local_offset, 9 * block_size + local_offset),
                11 * block_size + local_offset,
                len(expected_offsets),
            )
            total += result.num_rows
        assert total == count


@pytest.mark.parametrize(
    "source_ratio,destination_ratio", [(32, 1), (1, 32), (32, 32), (32, 64)]
)
@pytest.mark.parametrize("rows", [1, 63, 64, 65, 2047, 2048, 2049])
def test_import_copies_only_computed_pages_and_preserves_cached_prefix(
    source_ratio, destination_ratio, rows
):
    pages = (rows + 63) // 64
    source_blocks = list(range(2, 2 + (pages + source_ratio - 1) // source_ratio))
    destination_blocks = list(
        range(5, 5 + (pages + destination_ratio - 1) // destination_ratio)
    )
    for cached_blocks in range(len(destination_blocks) + 1):
        suffix = destination_blocks[cached_blocks:]
        pairs = pair_import_pages(
            source_blocks,
            suffix,
            source_ratio=source_ratio,
            destination_ratio=destination_ratio,
            num_rows=rows,
            kernel_block_size=64,
        )
        start = cached_blocks * destination_ratio
        source = torch.arange((source_blocks[-1] + 1) * source_ratio * 64).reshape(
            -1, 64
        )
        destination = torch.full(
            ((destination_blocks[-1] + 1) * destination_ratio, 64), -1
        )
        for src, dst in pairs:
            destination[dst] = source[src]
        for page in range(pages):
            dst = destination_blocks[page // destination_ratio] * destination_ratio
            dst += page % destination_ratio
            if page < start:
                assert torch.all(destination[dst] == -1)
            else:
                src = source_blocks[page // source_ratio] * source_ratio
                src += page % source_ratio
                assert torch.equal(destination[dst], source[src])
        copied = {dst for _, dst in pairs}
        assert all(
            torch.all(row == -1) for i, row in enumerate(destination) if i not in copied
        )


def test_import_rejects_missing_source_and_null_pages():
    with pytest.raises(ValueError, match="does not cover"):
        pair_import_pages(
            [2],
            [3, 4],
            source_ratio=1,
            destination_ratio=1,
            num_rows=65,
            kernel_block_size=64,
        )
    with pytest.raises(ValueError, match="null attention"):
        pair_import_pages(
            [0],
            [3],
            source_ratio=1,
            destination_ratio=1,
            num_rows=1,
            kernel_block_size=64,
        )


@pytest.mark.parametrize("registered_rows", [64, 2048])
@pytest.mark.parametrize("manager_ratio", [32, 64])
def test_import_addresses_manager_pages_with_padding(registered_rows, manager_ratio):
    row_bytes, kernel = 352, 64
    stride = registered_rows * row_bytes + 4096
    for manager in (1, 7):
        for within in range(manager_ratio):
            page = manager * manager_ratio + within
            address = import_page_offset(
                page,
                manager_ratio=manager_ratio,
                registered_rows=registered_rows,
                kernel_rows=kernel,
                row_bytes=row_bytes,
                registered_stride=stride,
            )
            # Enumerate the registered blocks in one manager page.
            rows = manager * manager_ratio * kernel + within * kernel
            registered, offset = divmod(rows, registered_rows)
            assert address == registered * stride + offset * row_bytes


@pytest.mark.parametrize("hot_rows", [6144, 12288])
def test_hybrid_hma_hot_rows_do_not_consume_one_ssm_lease_per_64_rows(hot_rows):
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.config.attention import HiSparseConfig
    from vllm.v1.core.kv_cache_planning import (
        _get_kv_cache_bytes_per_block,
        _get_packed_kv_cache_groups,
    )
    from vllm.v1.hisparse.layout import (
        _lay_out_dcp_groups,
        get_hisparse_kv_cache_groups,
        is_hisparse_mla_spec,
    )
    from vllm.v1.hisparse.runtime import ResolvedHiSparseConfig
    from vllm.v1.kv_cache_interface import (
        HiSparseHotSpec,
        KVCacheGroupSpec,
        MambaSpec,
        MLAAttentionSpec,
        SparseCacheRole,
        UniformTypeKVCacheSpecs,
    )
    from vllm.v1.kv_cache_layout import KVCacheLayout

    sources = {
        f"target.{i}": MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.uint8,
            state_content_bytes=352,
            topk_buffer_width=2048,
            block_stride_alignment=64 * 352,
        )
        for i in range(24)
    }
    indexers = {
        f"indexer.{i}": MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=68,
            dtype=torch.uint8,
            state_content_bytes=68,
            cache_role=SparseCacheRole.INDEXER,
            block_stride_alignment=64 * 68,
        )
        for i in range(24)
    }
    draft = {
        f"draft.{i}": MLAAttentionSpec(
            block_size=2048,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.float8_e4m3fn,
            is_draft=True,
            block_stride_alignment=64 * 576,
        )
        for i in range(5)
    }
    assert all(not is_hisparse_mla_spec(s) for s in draft.values())
    draft_spec = UniformTypeKVCacheSpecs.from_specs(draft)
    assert draft_spec is not None
    regular = [KVCacheGroupSpec(list(draft), draft_spec)]
    regular_specs = dict(draft)
    for group_id, count in enumerate([6] * 9 + [5] * 3):
        states = {
            f"kda.{group_id}.{i}": MambaSpec(
                block_size=2048,
                shapes=((212736,),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
                num_prefill_checkpoint_blocks=1,
            )
            for i in range(count)
        }
        spec = UniformTypeKVCacheSpecs.from_specs(states)
        assert spec is not None
        regular.append(KVCacheGroupSpec(list(states), spec))
        regular_specs.update(states)
    normal = VllmConfig(device_config=DeviceConfig(device="cpu"))
    normal.cache_config.kv_cache_layout = "BLHNC"
    planned_regular = _get_packed_kv_cache_groups(normal, regular_specs)
    assert planned_regular is not None
    assert sorted(len(g.layer_names) for g in planned_regular) == sorted(
        len(g.layer_names) for g in regular
    )
    regular = planned_regular
    config = ResolvedHiSparseConfig(
        top_k=2048, device_buffer_size=hot_rows, max_union_rows=min(hot_rows, 10240)
    )
    groups = _lay_out_dcp_groups(sources, indexers, regular, config)
    state_groups = [
        g
        for g in groups
        if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
        and isinstance(g.kv_cache_spec.first_spec, MambaSpec)
    ]
    assert len(state_groups) == 9
    assert sorted(len(g.layer_names) for g in state_groups) == [5] + [8] * 8
    assert groups[0].host_resident and groups[0].kv_cache_spec.block_size == 64
    assert groups[1].kv_cache_spec.block_size == 4096
    resident, hot = groups[2:4]
    assert resident.kv_cache_spec.block_size == 64
    assert isinstance(hot.kv_cache_spec, HiSparseHotSpec)
    assert hot.kv_cache_spec.block_size == 768
    assert hot.kv_cache_spec.blocks_per_request == hot_rows // 768
    quantum = _get_kv_cache_bytes_per_block(groups[1:], KVCacheLayout.BLHNC)
    assert quantum == 6893568
    assert hot.kv_cache_spec.blocks_per_request * quantum < 110 * 2**20
    # Replanning already-resolved indexer specs must keep their coarse pages.
    second = _lay_out_dcp_groups(
        sources,
        groups[1].kv_cache_spec.kv_cache_specs,
        regular,
        config,
    )
    assert second[1].kv_cache_spec.block_size == 4096
    assert second[3].kv_cache_spec.block_size == 768
    # Worker spec discovery precedes the engine's physical-layout resolution.
    early = VllmConfig(device_config=DeviceConfig(device="cpu"))
    early.attention_config.hisparse_config = HiSparseConfig(device_buffer_size=hot_rows)
    early.parallel_config.decode_context_parallel_size = 8
    early.scheduler_config.disable_hybrid_kv_cache_manager = False
    early_groups = get_hisparse_kv_cache_groups(
        early, sources | indexers | regular_specs
    )
    assert early.cache_config.kv_cache_layout is None
    assert early_groups is not None
    assert early_groups[1].kv_cache_spec.block_size == 4096
    assert early_groups[3].kv_cache_spec.blocks_per_request == hot_rows // 768

    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        resolve_kv_cache_block_sizes,
    )
    from vllm.v1.hisparse.coordinator import get_hisparse_coordinator
    from vllm.v1.kv_cache_interface import KVCacheConfig

    cache = generate_scheduler_kv_cache_config(
        [
            KVCacheConfig(
                num_blocks=256,
                kv_cache_tensors=[],
                kv_cache_groups=groups,
                hisparse_host_num_blocks=4096,
            )
        ]
    )
    early.cache_config.block_size = 64
    early.cache_config.prefix_match_unit = 128
    early.cache_config.enable_prefix_caching = True
    scheduler_block, hash_block = resolve_kv_cache_block_sizes(cache, early)
    manager = KVCacheManager(
        cache,
        max_model_len=1048576,
        max_in_flight_tokens=32768,
        scheduler_block_size=scheduler_block,
        hash_block_size=hash_block,
        dcp_world_size=8,
    )
    coordinator = get_hisparse_coordinator(manager)
    assert coordinator.host_manager.block_size == 512
    assert coordinator.resident_managers[0].block_size == 512
    assert coordinator.resident_managers[0].max_admission_blocks_per_request == 64
    assert coordinator.hot_managers[0].block_size == 768

    from vllm import SamplingParams
    from vllm.v1.request import Request

    request = Request(
        "pd-dcp",
        prompt_token_ids=list(range(513)),
        sampling_params=SamplingParams(max_tokens=32),
        pooling_params=None,
    )
    blocks = manager.allocate_slots(
        request,
        num_new_tokens=1,
        num_external_computed_tokens=512,
        num_lookahead_tokens=4,
        delay_cache_blocks=True,
    )
    assert blocks is not None
    held = manager.get_blocks(request.request_id)
    assert len(held.blocks[0]) == 2
    assert len(held.blocks[1]) == 1
    manager.free(request)
    assert coordinator.host_manager.block_pool.get_num_free_blocks() == 4095
    assert manager.block_pool.get_num_free_blocks() == 255


@pytest.mark.parametrize("dcp", [1, 8])
def test_small_hot_q5_requires_dcp_query_chunking(dcp):
    from vllm.config import DeviceConfig, SpeculativeConfig, VllmConfig
    from vllm.config.attention import HiSparseConfig
    from vllm.v1.hisparse.runtime import ResolvedHiSparseConfig

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=4,
        prompt_lookup_min=1,
        prompt_lookup_max=5,
    )
    config.parallel_config.decode_context_parallel_size = dcp
    config.attention_config.hisparse_config = HiSparseConfig(device_buffer_size=6144)
    if dcp == 1:
        with pytest.raises(ValueError, match="expected at least 10240"):
            ResolvedHiSparseConfig.from_vllm_config(config, 2048)
    else:
        resolved = ResolvedHiSparseConfig.from_vllm_config(config, 2048)
        assert resolved is not None
        assert resolved.device_buffer_size == resolved.max_union_rows == 6144


@pytest.mark.parametrize("world", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("interleave", [1, 4, 64])
@pytest.mark.parametrize("length", [0, 1, 63, 64, 65, 130001, 1048576, 1048577])
def test_decode_logits_width_covers_every_local_shard(length, world, interleave):
    from vllm.v1.attention.ops.cp_common import max_dcp_local_seq_len

    expected = max(
        HiSparseDCPShard(world, rank, interleave).count_before(length)
        for rank in range(world)
    )
    assert max_dcp_local_seq_len(length, world, interleave) == expected
    if world == 8 and interleave == 1 and length == 1048576:
        assert expected == 131072


@pytest.mark.parametrize("lengths", [(5, 5), (1, 5, 2, 4, 3), (5, 3), (1,), (7, 2)])
@pytest.mark.parametrize("limit", [1, 2, 3])
def test_query_chunks_preserve_all_tokens_and_bound_each_request(lengths, limit):
    from vllm.v1.hisparse.dcp import query_chunk_rows

    requests = [r for r, count in enumerate(lengths) for _ in range(count)]
    chunks = query_chunk_rows(len(requests), max(lengths), limit)
    assert sorted(row for chunk in chunks for row in chunk) == list(
        range(len(requests))
    )
    for chunk in chunks:
        for r in range(len(lengths)):
            assert sum(requests[row] == r for row in chunk) <= limit
    # Five disjoint top-2048 selections concentrated on one rank exceed6144.
    # Each chunk still retains every selected row, with at most3queries per request.
    selected = [
        set(range(row * 2048, (row + 1) * 2048)) for row in range(len(requests))
    ]
    for chunk in chunks:
        for r in range(len(lengths)):
            union = set().union(*(selected[row] for row in chunk if requests[row] == r))
            assert len(union) <= limit * 2048


def test_hot_view_preserves_row_stride_with_partial_lease_padding():
    from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
    from vllm.v1.hisparse.runtime import PagedCacheView

    row_bytes, stride, hot_block = 352, 6893568, 768
    offset = 13 * hot_block * row_bytes
    raw = torch.zeros(4 * stride, dtype=torch.uint8)
    view = PagedCacheView.bind(
        raw,
        dtype=torch.uint8,
        row_width=row_bytes,
        byte_offset=offset,
        block_stride=stride,
        num_blocks=4,
        block_size=hot_block,
    )
    rows, physical_stride = flat_kv_row_view(view.attention_cache, hot_block)
    assert physical_stride == view.attention_block_stride == stride // row_bytes
    payload = torch.arange(row_bytes, dtype=torch.int64).to(torch.uint8)
    view.cache[2, 65].copy_(payload)
    assert torch.equal(rows[2 * physical_stride + 65], payload)
    assert torch.equal(
        raw[
            offset + 2 * stride + 65 * row_bytes : offset + 2 * stride + 66 * row_bytes
        ],
        payload,
    )
