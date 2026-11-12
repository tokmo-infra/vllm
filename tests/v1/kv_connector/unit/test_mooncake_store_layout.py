# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Mooncake Store payload layouts."""

import ctypes
import random

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
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
    length = len(layout.block_len)
    addrs = [
        base_addr + block_id * layout.block_len[index % length]
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


@pytest.mark.cpu_test
@pytest.mark.parametrize("producer_strides", [(48, 24, 6, 2), (3, 72, 18, 1)])
def test_compact_group_payload_round_trip_omits_padding_and_other_groups(
    producer_strides,
):
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
        CompactGroupStoreLayout,
    )

    shape = (3, 2, 4, 3)
    source_storage = torch.full((512,), 0xEE, dtype=torch.uint8)
    source = source_storage.as_strided(shape, producer_strides, storage_offset=5)
    source.copy_(torch.arange(source.numel(), dtype=torch.uint8).reshape(shape))
    target_storage = torch.full((512,), 0xDD, dtype=torch.uint8)
    target = target_storage.as_strided(shape, (80, 32, 6, 1), storage_offset=7)
    metadata = KeyMetadata("test-model", 0, 0, 0, 0)
    producer = CompactGroupStoreLayout(metadata, 16, 4)
    consumer = CompactGroupStoreLayout(metadata, 16, 4)
    producer.register_kv_caches([source], 3)
    consumer.register_kv_caches([target], 3)

    # The partial block deliberately retains every logical state element.
    chunks = [(0, 16), (16, 20)]
    source_addrs, source_sizes, selected = producer.prepare_values(
        chunks, [2, 0], [0, 0]
    )
    target_addrs, target_sizes, _ = consumer.prepare_values(chunks, [1, 2], [0, 0])
    assert selected == [2, 0]
    assert producer.compact_block_bytes == source[0].numel()
    assert [sum(sizes) for sizes in source_sizes] == [24, 24]
    touched: set[int] = set()
    for source_block, addrs, sizes, dst_addrs, dst_sizes in zip(
        [2, 0], source_addrs, source_sizes, target_addrs, target_sizes, strict=True
    ):
        payload = b"".join(
            ctypes.string_at(addr, size)
            for addr, size in zip(addrs, sizes, strict=True)
        )
        assert payload == bytes(source[source_block].flatten().tolist())
        offset = 0
        for addr, size in zip(dst_addrs, dst_sizes, strict=True):
            ctypes.memmove(addr, payload[offset : offset + size], size)
            touched.update(
                range(
                    addr - target_storage.data_ptr(),
                    addr - target_storage.data_ptr() + size,
                )
            )
            offset += size
        assert offset == len(payload)
    assert torch.equal(target[1], source[2])
    assert torch.equal(target[2], source[0])
    # Neither padding, adjacent group storage, nor the unused page is written.
    assert all(
        value == 0xDD
        for i, value in enumerate(target_storage.tolist())
        if i not in touched
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("invalid_stride", [(0, 1), (1, 1)])
def test_compact_group_layout_rejects_aliasing_views(invalid_stride):
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
        CompactGroupStoreLayout,
    )

    tensor = torch.empty(16, dtype=torch.uint8).as_strided((3, 4), invalid_stride)
    layout = CompactGroupStoreLayout(KeyMetadata("test-model", 0, 0, 0, 0), 16, 4)
    with pytest.raises(ValueError, match="strides|non-overlapping"):
        layout.register_kv_caches([tensor], 3)


@pytest.mark.cpu_test
def test_compact_group_layout_keeps_all_state_views_in_payload_order():
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
        CompactGroupStoreLayout,
    )

    first = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    second = torch.arange(18, dtype=torch.int16).reshape(3, 6)
    layout = CompactGroupStoreLayout(KeyMetadata("test-model", 0, 0, 0, 0), 16, 4)
    layout.register_kv_caches([first, second], 3)
    addrs, sizes = layout.prepare_value_for_block(1)
    assert addrs == [first[1].data_ptr(), second[1].data_ptr()]
    assert sizes == [16, 12]
    assert layout.compact_block_bytes == 28
    with pytest.raises(IndexError, match="capacity"):
        layout.prepare_value_for_block(3)
    with pytest.raises(ValueError, match="boundary"):
        layout.prepare_values([(4, 8)], [1], [0])
    with pytest.raises(ValueError, match="shard"):
        layout.prepare_values([(0, 16)], [1], [1])
