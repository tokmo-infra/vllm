# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Position geometry for rank-private, context-sharded HiSparse caches."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.hisparse.types import SparseKVRowMirror

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class HiSparseDCPShard:
    size: int = 1
    rank: int = 0
    interleave: int = 1

    def __post_init__(self) -> None:
        if self.size < 1 or not 0 <= self.rank < self.size or self.interleave < 1:
            raise ValueError("Invalid HiSparse DCP shard geometry")

    def count_before(self, position: int) -> int:
        """Number of this rank's tokens in the global interval [0, position)."""
        if position < 0:
            raise ValueError("Token positions must be nonnegative")
        cycles, remainder = divmod(position, self.size * self.interleave)
        return cycles * self.interleave + min(
            max(remainder - self.rank * self.interleave, 0), self.interleave
        )

    def localize(self, positions: "torch.Tensor") -> "torch.Tensor":
        """Pack owned positions into a stable prefix, then pad with -1.

        Sparse decode reads only valid_count entries, so filtering other
        ranks must not leave holes among those entries.
        """
        import torch

        owned = (positions >= 0) & (
            positions // self.interleave % self.size == self.rank
        )
        local = (
            positions // (self.interleave * self.size) * self.interleave
            + positions % self.interleave
        )
        local = torch.where(owned, local, -1)
        order = torch.argsort((~owned).to(torch.int32), dim=-1, stable=True)
        return local.gather(-1, order)

    def mirror(
        self, mirror: SparseKVRowMirror, block_size: int
    ) -> SparseKVRowMirror | None:
        """Translate scheduler virtual rows to this worker's physical rows.

        The scheduler describes one global-page span. Block identities agree
        across ranks; each rank writes only its own interleaved payload.
        """
        virtual_block = block_size * self.size
        if block_size <= 0 or block_size % self.interleave:
            raise ValueError("HiSparse blocks must contain whole DCP interleaves")

        def local_start(start: int) -> int:
            block, offset = divmod(start, virtual_block)
            return block * block_size + self.count_before(offset)

        offset = mirror.destination_start % virtual_block
        if offset + mirror.num_rows > virtual_block:
            raise ValueError("HiSparse mirrors must not cross a virtual page")
        count = self.count_before(offset + mirror.num_rows) - self.count_before(offset)
        if not count:
            return None
        return SparseKVRowMirror(
            tuple(local_start(start) for start in mirror.source_starts),
            local_start(mirror.destination_start),
            count,
        )


def pair_import_pages(
    source: list[int],
    destination: list[int],
    *,
    source_ratio: int,
    destination_ratio: int,
    num_rows: int,
    kernel_block_size: int,
) -> list[tuple[int, int]]:
    """Pair a missing suffix by logical position, excluding source padding.

    Source IDs cover the whole prefix. Destination IDs cover its missing
    contiguous suffix. Each endpoint can pack a different number of kernel
    pages in one scheduler block; taking the last N source pages would copy
    uncomputed padding when the final source block is only partly filled.
    """
    if min(source_ratio, destination_ratio, kernel_block_size) <= 0 or num_rows < 0:
        raise ValueError("Invalid HiSparse import geometry")
    pages = (num_rows + kernel_block_size - 1) // kernel_block_size
    destination_blocks = (pages + destination_ratio - 1) // destination_ratio
    start_block = destination_blocks - len(destination)
    if start_block < 0:
        raise ValueError("HiSparse import requests pages beyond the prefix")
    result = []
    for page in range(start_block * destination_ratio, pages):
        source_index, source_offset = divmod(page, source_ratio)
        destination_index, destination_offset = divmod(page, destination_ratio)
        if source_index >= len(source):
            raise ValueError("HiSparse import source does not cover the prefix")
        src = source[source_index]
        dst = destination[destination_index - start_block]
        if src <= 0 or dst <= 0:
            raise ValueError("HiSparse imports cannot address null attention pages")
        result.append(
            (
                src * source_ratio + source_offset,
                dst * destination_ratio + destination_offset,
            )
        )
    return result


def import_page_offset(
    page: int,
    *,
    manager_ratio: int,
    registered_rows: int,
    kernel_rows: int,
    row_bytes: int,
    registered_stride: int,
) -> int:
    """Byte offset of a kernel page in a packed or split registered cache."""
    if (
        min(manager_ratio, registered_rows, kernel_rows, row_bytes) <= 0
        or registered_rows % kernel_rows
        or manager_ratio * kernel_rows % registered_rows
    ):
        raise ValueError("Incompatible registered HiSparse page geometry")
    manager, within = divmod(page, manager_ratio)
    rows = within * kernel_rows
    registered_per_manager = manager_ratio * kernel_rows // registered_rows
    block = manager * registered_per_manager + rows // registered_rows
    return block * registered_stride + (rows % registered_rows) * row_bytes
