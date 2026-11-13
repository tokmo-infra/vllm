# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Mooncake Store payload layouts."""

import ctypes
import random

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    DCPPartialStoreLayout,
    KeyMetadata,
    LBHNCStoreLayout,
    LBNHCStoreLayout,
    RankLocalStoreLayout,
)
from vllm.utils.math_utils import cdiv

BLOCK_SIZE = 128


def _rank_local_layout(num_regions: int, num_block_lens: int) -> RankLocalStoreLayout:
    metadata = KeyMetadata("test-model", 1, 0, 0, 0)
    layout = RankLocalStoreLayout(metadata, BLOCK_SIZE, BLOCK_SIZE)
    layout.set_kv_caches_base_addr(
        [0x7F00_0000_0000 + i * (1 << 30) for i in range(num_regions)]
    )
    layout.set_block_len([30_208 + 512 * i for i in range(num_block_lens)])
    return layout


def _rank_local_reference(
    layout: RankLocalStoreLayout, start: int, end: int, block_ids: list[int]
) -> tuple[list[int], list[int], int]:
    block_id = block_ids[start // layout.block_size]
    length = len(layout.block_stride)
    addrs = [
        base_addr + block_id * layout.block_stride[index % length]
        for index, base_addr in enumerate(layout.kv_caches_base_addr)
    ]
    sizes = [
        layout.block_len[index % length] * cdiv(end - start, layout.block_size)
        for index in range(len(layout.kv_caches_base_addr))
    ]
    return addrs, sizes, block_id


@pytest.mark.parametrize("num_regions,num_block_lens", [(96, 96), (96, 2), (1, 1)])
def test_rank_local_descriptors_match_reference(num_regions: int, num_block_lens: int):
    layout = _rank_local_layout(num_regions, num_block_lens)
    rng = random.Random(0)
    block_ids = [rng.randrange(0, 1 << 20) for _ in range(300)]
    chunks = []
    block = 0
    while block < len(block_ids) - 4:
        span = rng.choice([1, 1, 1, 2, 4])
        chunks.append((block * BLOCK_SIZE, (block + span) * BLOCK_SIZE))
        block += span + rng.choice([0, 1])

    addrs, sizes, selected_blocks = layout.prepare_values(
        chunks, block_ids, [0] * len(chunks)
    )

    for chunk, chunk_addrs, chunk_sizes, block_id in zip(
        chunks, addrs, sizes, selected_blocks, strict=True
    ):
        assert (chunk_addrs, chunk_sizes, block_id) == _rank_local_reference(
            layout, *chunk, block_ids
        )
        assert all(type(addr) is int for addr in chunk_addrs)
        assert type(block_id) is int


def test_rank_local_descriptors_handle_empty_and_invalid_chunks():
    layout = _rank_local_layout(4, 4)
    assert layout.prepare_values([], [1, 2, 3], []) == ([], [], [])
    with pytest.raises(AssertionError):
        layout.prepare_values([(0, BLOCK_SIZE + 1)], [0, 1], [0])


def test_rank_local_registers_padded_kernel_blocks_as_one_region():
    # Merged DSA cache (Kimi K3.1): 64-token kernel blocks of 64 x 788 B, each
    # padded to 50512 B, 17 per manager block, grouped to manager blocks.
    num_blocks, ratio, unit, padded = 4, 17, 64 * 788, 50_512
    raw = torch.zeros(num_blocks * ratio * padded, dtype=torch.uint8)
    cache = raw.as_strided((num_blocks * ratio, 1, 64, 788), (padded, unit, 788, 1))
    layout = RankLocalStoreLayout(KeyMetadata("test-model", 0, 0, 0, 0), 1088, 1088)

    layout.register_kv_caches([cache], num_blocks)

    # One region of whole manager blocks, so a block's put never spills into
    # the next block.
    assert layout.kv_caches_base_addr == [raw.data_ptr()]
    assert layout.block_stride == [ratio * padded]
    # Stored up to the last kernel block's data, skipping its tail padding.
    assert layout.block_len == [(ratio - 1) * padded + unit]
    addrs, sizes = layout.prepare_value_for_block(num_blocks - 1)
    assert addrs[0] + sizes[0] == raw.data_ptr() + raw.numel() - (padded - unit)


def test_rank_local_splits_interleaved_blocks_per_head():
    # K and V of all blocks in separate planes: a block is not stride-compact.
    num_blocks = 4
    kv = torch.zeros(2, num_blocks, BLOCK_SIZE, 64, dtype=torch.uint8)
    layout = RankLocalStoreLayout(
        KeyMetadata("test-model", 0, 0, 0, 0), BLOCK_SIZE, BLOCK_SIZE
    )

    layout.register_kv_caches([kv.permute(1, 0, 2, 3)], num_blocks)

    assert layout.kv_caches_base_addr == [kv[0].data_ptr(), kv[1].data_ptr()]
    assert layout.block_len == [BLOCK_SIZE * 64] * 2


def test_rank_local_layout_skips_block_padding():
    """A state padded up to a larger page, like a Mamba state padded to the
    attention page size, is stored without the padding."""
    num_layers, num_blocks, page, state = 2, 3, 64, 40
    # Layer-outermost buffer; each layer views its state at the start of
    # every page.
    raw = torch.zeros(num_layers * num_blocks * page, dtype=torch.int8)
    views = [
        torch.as_strided(
            raw,
            (num_blocks, 1, 1, state),
            (page, state, state, 1),
            layer * num_blocks * page,
        )
        for layer in range(num_layers)
    ]
    layout = RankLocalStoreLayout(
        KeyMetadata("test-model", 0, 0, 0, 0), BLOCK_SIZE, BLOCK_SIZE
    )
    layout.register_kv_caches(views, num_blocks)

    assert layout.kv_caches_base_addr == [view.data_ptr() for view in views]
    assert layout.block_stride == [page] * num_layers
    assert layout.block_len == [state] * num_layers
    addrs, sizes, _ = layout.prepare_values(
        [(0, BLOCK_SIZE), (BLOCK_SIZE, 3 * BLOCK_SIZE)], [2, 0, 1], [0, 0]
    )
    assert addrs == [
        [view.data_ptr() + 2 * page for view in views],
        [view.data_ptr() for view in views],
    ]
    # A two-block chunk copies the padding between its blocks.
    assert sizes == [[state] * num_layers, [page + state] * num_layers]
    assert layout.prepare_value_for_block(1) == (
        [view.data_ptr() + page for view in views],
        [state] * num_layers,
    )


def _dcp_partial_layout(
    raw,
    *,
    rank=0,
    tp_rank=None,
    interleave=1,
    widths=(4, 8, 12),
    dtypes=None,
    dcp_size=8,
    block_size=128,
    hash_block_size=32,
):
    rows = block_size // dcp_size
    stride = rows * sum(widths) + 16
    views = []
    offset = 0
    for width in widths:
        views.append(
            raw.as_strided(
                (3, 1, rows, width), (stride, rows * width, width, 1), offset
            )
        )
        offset += rows * width
    if dtypes is not None:
        views = [view.view(dtype) for view, dtype in zip(views, dtypes, strict=True)]
    layout = DCPPartialStoreLayout(
        KeyMetadata("test-model", rank if tp_rank is None else tp_rank, 0, rank, 0),
        block_size,
        hash_block_size,
        dcp_size=dcp_size,
        dcp_rank=rank,
        interleave=interleave,
    )
    layout.register_kv_caches(views, 3)
    return layout, views


@pytest.mark.cpu_test
@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("interleave", [1, 2])
@pytest.mark.parametrize("boundary", [32, 64, 96, 128])
@pytest.mark.parametrize("widths", [(4, 8, 12), (352, 4, 8)])
def test_partial_dcp_values_roundtrip_every_valid_row(
    rank, interleave, boundary, widths
):
    """All layer/scales bytes return; uncached rows and other slots do not."""
    stride = 16 * sum(widths) + 16
    raw = torch.arange(3 * stride, dtype=torch.int64).to(torch.uint8)
    destination = torch.full_like(raw, 197)
    writer, source_views = _dcp_partial_layout(
        raw, rank=rank, interleave=interleave, widths=widths
    )
    reader, target_views = _dcp_partial_layout(
        destination, rank=rank, interleave=interleave, widths=widths
    )
    addrs, sizes = writer.prepare_value_for_block(1, num_tokens=boundary)
    targets, target_sizes, ids = reader.prepare_values(
        [(128, 128 + boundary)], [0, 1, 2], [0]
    )
    assert sizes == target_sizes[0] and ids == [1]
    assert sum(sizes) == boundary // 8 * sum(widths)
    saved = [ctypes.string_at(addr, size) for addr, size in zip(addrs, sizes)]
    for addr, data in zip(targets[0], saved):
        ctypes.memmove(addr, data, len(data))
    rows = boundary // 8
    for source, target in zip(source_views, target_views):
        assert torch.equal(target[1, :, :rows], source[1, :, :rows])
        assert torch.all(target[1, :, rows:] == 197)
        assert torch.all(target[0] == 197) and torch.all(target[2] == 197)
    assert torch.all(destination[2 * stride - 16 : 2 * stride] == 197)
    assert writer.key_for(0, b"hash") == reader.key_for(0, b"hash")


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "block_size,hash_block_size,boundary",
    [(128, 32, 32), (128, 32, 96), (128, 32, 128), (32, 32, 32)],
    ids=["quarter-tail", "three-quarter-tail", "full-page", "block-equals-hash"],
)
@pytest.mark.parametrize(
    "widths,dtypes",
    [
        ((4, 8, 12), (torch.uint8,) * 3),
        ((352, 4, 8), (torch.uint8, torch.float32, torch.float32)),
    ],
    ids=["dense-u8", "nvfp4-scale-f32"],
)
def test_partial_dcp1_roundtrip_preserves_rows_scales_and_replica_namespace(
    block_size, hash_block_size, boundary, widths, dtypes
):
    """Independent replicas keep every valid row byte, not block padding."""
    stride = block_size * sum(widths) + 16
    namespaces, keys = set(), set()
    for tp_rank in range(4):
        raw = torch.arange(3 * stride, dtype=torch.int64).to(torch.uint8)
        destination = torch.full_like(raw, 197)
        kwargs = dict(
            tp_rank=tp_rank,
            widths=widths,
            dtypes=dtypes,
            dcp_size=1,
            block_size=block_size,
            hash_block_size=hash_block_size,
        )
        writer, source_views = _dcp_partial_layout(raw, **kwargs)
        reader, target_views = _dcp_partial_layout(destination, **kwargs)
        key = writer.key_for(0, b"hash")
        assert key == reader.key_for(0, b"hash")
        assert writer.metadata.store_namespace == reader.metadata.store_namespace
        namespaces.add(writer.metadata.store_namespace)
        keys.add(key)
        writer.register_kv_caches(source_views, 3)
        assert writer.key_for(0, b"hash") == key

        legacy = RankLocalStoreLayout(
            KeyMetadata("test-model", tp_rank, 0, 0, 0),
            block_size,
            hash_block_size,
        )
        legacy.register_kv_caches(source_views, 3)
        assert legacy.key_for(0, b"hash") != key
        assert writer.prepare_value_for_block(1) == legacy.prepare_value_for_block(1)
        full_bytes = sum(legacy.prepare_value_for_block(1)[1])
        assert full_bytes == block_size * sum(widths)

        addrs, sizes = writer.prepare_value_for_block(
            1, num_tokens=block_size + boundary
        )
        targets, target_sizes, ids = reader.prepare_values(
            [(block_size, block_size + boundary)], [0, 1, 2], [0]
        )
        assert ids == [1] and sizes == target_sizes[0]
        saved = [
            ctypes.string_at(addr, size)
            for addr, size in zip(addrs, sizes, strict=True)
        ]
        assert b"".join(saved) == b"".join(
            view[1, :, :boundary].contiguous().numpy().tobytes()
            for view in source_views
        )
        stored_bytes = sum(map(len, saved))
        assert stored_bytes == boundary * sum(widths)
        if boundary < block_size:
            assert stored_bytes < full_bytes
        else:
            assert stored_bytes == full_bytes
        for addr, payload in zip(targets[0], saved, strict=True):
            ctypes.memmove(addr, payload, len(payload))
        for source, target in zip(source_views, target_views, strict=True):
            source_bytes, target_bytes = (
                source.view(torch.uint8),
                target.view(torch.uint8),
            )
            assert torch.equal(
                target_bytes[1, :, :boundary], source_bytes[1, :, :boundary]
            )
            assert torch.all(target_bytes[1, :, boundary:] == 197)
            assert torch.all(target_bytes[0] == 197)
            assert torch.all(target_bytes[2] == 197)
        for slot in range(3):
            assert torch.all(
                destination[(slot + 1) * stride - 16 : (slot + 1) * stride] == 197
            )
    assert len(namespaces) == 1
    assert len(keys) == 4


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "boundary", [4, 8, None], ids=["partial", "full", "default-full"]
)
@pytest.mark.parametrize("dcp_size", [1, 4])
@pytest.mark.parametrize("packed_source", [False, True])
def test_partial_dcp_wire_preserves_logical_order_across_allocations(
    boundary, dcp_size, packed_source
):
    """Store bytes follow layers, not sender/receiver address order."""
    num_blocks, block_size, width = 3, 8, 4
    rows = block_size // dcp_size
    shape = (num_blocks, 1, rows, width)
    if packed_source:
        raw = torch.full((num_blocks, 2, 1, rows, width), 197, dtype=torch.uint8)
        source = [raw[:, 1], raw[:, 0]]
    else:
        source = sorted(
            [torch.empty(shape, dtype=torch.uint8) for _ in range(2)],
            key=lambda cache: cache.data_ptr(),
        )
    target = sorted(
        [torch.full(shape, 197, dtype=torch.uint8) for _ in range(2)],
        key=lambda cache: cache.data_ptr(),
        reverse=True,
    )
    for index, cache in enumerate(source):
        cache.fill_(11 * (index + 1))

    layouts = []
    for caches in (source, target):
        layout = DCPPartialStoreLayout(
            KeyMetadata("test-model", 0, 0, 0, 0),
            block_size,
            4,
            dcp_size=dcp_size,
            dcp_rank=0,
            interleave=1,
        )
        layout.register_kv_caches(caches, num_blocks)
        layouts.append(layout)
    writer, reader = layouts
    assert writer.key_for(0, b"hash") == reader.key_for(0, b"hash")
    addrs, sizes = writer.prepare_value_for_block(1, num_tokens=boundary)
    end = block_size if boundary is None else boundary
    targets, target_sizes, ids = reader.prepare_values(
        [(block_size, block_size + end)], [0, 1, 2], [0]
    )
    assert ids == [1]
    saved = b"".join(
        ctypes.string_at(addr, size) for addr, size in zip(addrs, sizes, strict=True)
    )
    valid_rows = end // dcp_size
    expected = b"".join(
        cache[1, :, :valid_rows].contiguous().numpy().tobytes() for cache in source
    )
    assert saved == expected
    offset = 0
    for addr, size in zip(targets[0], target_sizes[0], strict=True):
        ctypes.memmove(addr, saved[offset : offset + size], size)
        offset += size
    assert offset == len(saved)
    for src, dst in zip(source, target, strict=True):
        assert torch.equal(src[1, :, :valid_rows], dst[1, :, :valid_rows])
        assert torch.all(dst[1, :, valid_rows:] == 197)
        assert torch.all(dst[0] == 197) and torch.all(dst[2] == 197)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "boundary", [4, 8, None], ids=["partial", "full", "default-full"]
)
def test_partial_dcp_namespace_ignores_page_count_and_allocation_padding(boundary):
    """The same logical wire format survives GPU pool size/layout changes."""
    keys, payloads = set(), []
    for num_blocks, padding in ((2, 0), (5, 16)):
        rows, width = 8, 4
        stride = rows * width + padding
        raw = torch.full((2 * num_blocks * stride,), 197, dtype=torch.uint8)
        caches = [
            raw.as_strided(
                (num_blocks, 1, rows, width),
                (stride, rows * width, width, 1),
                layer * num_blocks * stride,
            )
            for layer in range(2)
        ]
        for index, cache in enumerate(caches):
            cache.fill_(11 * (index + 1))
        layout = DCPPartialStoreLayout(
            KeyMetadata("test-model", 0, 0, 0, 0),
            8,
            4,
            dcp_size=1,
            dcp_rank=0,
            interleave=1,
        )
        layout.register_kv_caches(caches, num_blocks)
        keys.add(layout.key_for(0, b"hash"))
        addrs, sizes = layout.prepare_value_for_block(1, num_tokens=boundary)
        payloads.append(
            b"".join(
                ctypes.string_at(addr, size)
                for addr, size in zip(addrs, sizes, strict=True)
            )
        )
    assert len(keys) == 1
    assert payloads[0] == payloads[1]


@pytest.mark.cpu_test
def test_partial_dcp_namespace_is_stable_but_isolates_legacy_values():
    raw = torch.zeros(1200, dtype=torch.uint8)
    layout, views = _dcp_partial_layout(raw)
    assert layout.metadata.store_namespace.startswith("@partial_layout_v2:")
    key = layout.key_for(0, b"hash")
    layout.register_kv_caches(views, 3)
    assert layout.key_for(0, b"hash") == key
    legacy = RankLocalStoreLayout(KeyMetadata("test-model", 0, 0, 0, 0), 128, 32)
    legacy.register_kv_caches(views, 3)
    assert legacy.key_for(0, b"hash") != key
    old_partial = RankLocalStoreLayout(
        KeyMetadata(
            "test-model",
            0,
            0,
            0,
            0,
            store_namespace="@partial_layout:fb966b02c66c38927243e74b",
        ),
        128,
        32,
    )
    assert old_partial.key_for(0, b"hash") != key
    assert sum(legacy.prepare_value_for_block(1)[1]) == 384
    assert sum(layout.prepare_value_for_block(1, num_tokens=32)[1]) == 96
    assert layout.prepare_value_for_block(1, num_tokens=128) == (
        legacy.prepare_value_for_block(1)
    )


@pytest.mark.cpu_test
def test_partial_dcp_rejects_incomplete_rank_boundaries_and_nonrow_layouts():
    with pytest.raises(ValueError, match="rank-complete"):
        DCPPartialStoreLayout(
            KeyMetadata("model", 0, 0, 0, 0),
            128,
            8,
            dcp_size=8,
            dcp_rank=0,
            interleave=2,
        )
    layout, _ = _dcp_partial_layout(torch.zeros(1200, dtype=torch.uint8))
    with pytest.raises(ValueError, match="dense manager-block"):
        layout.register_kv_caches([torch.zeros(3, 16, 4)], 3)


@pytest.mark.cpu_test
@pytest.mark.parametrize("kind", ["token_padding", "overlap", "multi_head"])
def test_partial_dcp_rejects_unsupported_row_strides(kind):
    layout, _ = _dcp_partial_layout(torch.zeros(1200, dtype=torch.uint8))
    if kind == "token_padding":
        cache = torch.zeros(3, 1, 16, 8, dtype=torch.uint8)[..., :4]
    elif kind == "overlap":
        cache = torch.zeros(1, 1, 16, 4, dtype=torch.uint8).expand(3, -1, -1, -1)
    else:
        cache = torch.zeros(3, 2, 16, 4, dtype=torch.uint8)
    with pytest.raises(ValueError, match="dense manager-block"):
        layout.register_kv_caches([cache], 3)


@pytest.mark.cpu_test
def test_partial_dcp_keys_isolate_interleave_and_hash_units():
    raw = torch.zeros(1200, dtype=torch.uint8)
    layout, views = _dcp_partial_layout(raw)
    keys = {layout.key_for(0, b"hash")}
    for hash_size, interleave in [(64, 1), (32, 2)]:
        other = DCPPartialStoreLayout(
            KeyMetadata("test-model", 0, 0, 0, 0),
            128,
            hash_size,
            dcp_size=8,
            dcp_rank=0,
            interleave=interleave,
        )
        other.register_kv_caches(views, 3)
        keys.add(other.key_for(0, b"hash"))
    assert len(keys) == 3


@pytest.mark.parametrize(
    ("layout_cls", "store_format"),
    [
        (LBHNCStoreLayout, "tp_shared_lbhnc"),
        (LBNHCStoreLayout, "tp_shared_lbnhc"),
    ],
)
def test_tp_shared_layout_owns_store_namespace(layout_cls, store_format):
    assert (
        layout_cls.shared_namespace(4, 2)
        == f"@store_tp:4@store_pp:2@store_format:{store_format}"
    )


def test_rank_local_database_api_rejects_tp_shared_layout():
    metadata = KeyMetadata("test-model", 0, 0, 0, 0)
    layout = LBHNCStoreLayout(
        metadata,
        block_size=16,
        hash_block_size=16,
        local_tp_size=2,
        store_tp_size=4,
        tp_rank=0,
        num_kv_heads=8,
    )
    database = ChunkedTokenDatabase(metadata, 16, store_layout=layout)

    with pytest.raises(RuntimeError, match="rank-local"):
        database.prepare_value_for_block(0)


@pytest.mark.parametrize(
    ("layout_cls", "producer_strides", "consumer_strides"),
    [
        (LBHNCStoreLayout, (128, 64, 4, 1), (256, 64, 4, 1)),
        (LBNHCStoreLayout, (128, 4, 8, 1), (256, 4, 16, 1)),
    ],
)
def test_tp_shared_layout_round_trip_across_tp_sizes(
    layout_cls, producer_strides, consumer_strides
):
    block_size = 16
    shape = (1, 2, block_size, 4)
    stored: dict[int, bytes] = {}
    producer_tensors = []

    for tp_rank in range(4):
        tensor = torch.empty_strided(shape, producer_strides, dtype=torch.float16)
        tensor.copy_(
            torch.arange(128, dtype=torch.float16).view(shape) + tp_rank * 1000
        )
        producer_tensors.append(tensor)
        metadata = KeyMetadata("test-model", tp_rank, 0, 0, 0)
        layout = layout_cls(
            metadata,
            block_size,
            block_size,
            local_tp_size=4,
            store_tp_size=4,
            tp_rank=tp_rank,
            num_kv_heads=8,
        )
        layout.register_kv_caches([tensor], 1)
        addrs, sizes, _ = layout.prepare_values([(0, block_size)], [0], [tp_rank])
        stored[tp_rank] = b"".join(
            ctypes.string_at(addr, size)
            for addr, size in zip(addrs[0], sizes[0], strict=True)
        )

    for tp_rank in range(2):
        tensor = torch.empty_strided(
            (1, 4, block_size, 4), consumer_strides, dtype=torch.float16
        )
        tensor.zero_()
        metadata = KeyMetadata("test-model", tp_rank, 0, 0, 0)
        layout = layout_cls(
            metadata,
            block_size,
            block_size,
            local_tp_size=2,
            store_tp_size=4,
            tp_rank=tp_rank,
            num_kv_heads=8,
        )
        layout.register_kv_caches([tensor], 1)
        shard_ids = layout.local_shard_ids
        addrs, sizes, _ = layout.prepare_values(
            [(0, block_size)] * len(shard_ids), [0], shard_ids
        )
        for shard_id, shard_addrs, shard_sizes in zip(
            shard_ids, addrs, sizes, strict=True
        ):
            offset = 0
            for addr, size in zip(shard_addrs, shard_sizes, strict=True):
                ctypes.memmove(addr, stored[shard_id][offset : offset + size], size)
                offset += size

        expected = torch.cat(producer_tensors[tp_rank * 2 : tp_rank * 2 + 2], dim=1)
        torch.testing.assert_close(tensor, expected)
