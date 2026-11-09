# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only ABI/layout calculations. Keep byte-identical to the CUDA header."""

from dataclasses import dataclass

COUNTER_INTS = 2616
CTR_SLOT = 14
ERROR_INDEX = 3
ABI_VERSION = "nvfp4_e4m3_g16_alias_native_v3_p_imbalance256"


def align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def pool_row_bytes(hidden):
    return hidden // 2 + align(hidden // 16 + 4, 16)


def weight_row(row, rows, layout):
    if layout in (-1, 2):
        return row
    if layout not in (0, 1):
        raise ValueError(
            "weight layout must be checkpoint_gate_up=0, native_up_gate=1 "
            "or interleaved8=2"
        )
    logical = row // 16 * 8 + row % 8
    up = row % 16 >= 8
    return logical + (rows // 2 if (not up if layout == 1 else up) else 0)


def scale_word(row, k64, K, layout):
    if layout == 0:
        return row * (K // 64) + k64
    if layout != 1:
        raise ValueError("scale layout must be raw=0 or native_swizzle128=1")
    return (row // 128 * (K // 64) + k64) * 128 + row % 32 * 4 + row % 128 // 32


def buffer_layout(ranks, experts, capacity, topk, hidden, intermediate, gin=False):
    validate_shape(
        ranks, experts, capacity, topk, hidden, intermediate, diagnostic=True
    )
    rows = experts * capacity  # EL * ranks * capacity, every expert/source/token
    epr = experts // ranks
    rb = pool_row_bytes(hidden)
    sizes = (
        capacity * (hidden // 2),
        capacity * (hidden // 16),
        capacity * topk * 8,
        capacity * topk * 4,
        rows * rb,
        ranks * epr * 4,
        ranks * 4,
        capacity * topk * hidden * 2,
        ranks * 4,
        ranks * (1 + capacity * topk) * 4,
        ranks * 4,
        rows * (intermediate // 2),
        rows * (intermediate // 16),
        COUNTER_INTS * 4,
        capacity * rb if gin else 0,
        ranks * capacity * rb if gin else 0,
        ranks * capacity * topk * hidden * 2 if gin else 0,
        ranks * capacity * topk * hidden * 2 if gin else 0,
        ranks * capacity * topk * 4 if gin else 0,
        ranks * capacity * topk * 4 if gin else 0,
        capacity * topk * 4 if gin else 0,
    )
    offsets, cursor = [], 0
    for size in sizes:
        offsets.append(cursor)
        cursor = align(cursor + size, 256)
    return [cursor, *offsets]


def validate_shape(
    world, experts, capacity, topk, hidden, intermediate, diagnostic=False
):
    if not (
        1 <= world <= 8
        and experts > 0
        and experts % world == 0
        and experts // world <= 256
    ):
        raise ValueError(
            "one-host world<=8, divisible positive experts, local experts<=256"
        )
    if not (1 <= capacity <= 256 and 1 <= topk <= min(8, experts)):
        raise ValueError(
            "workspace capacity <=256, topk 1..8 bounded by routed experts"
        )
    if not (
        0 < hidden <= 4096
        and hidden % 256 == 0
        and intermediate > 0
        and intermediate % 256 == 0
    ):
        raise ValueError("hidden<=4096, H and I positive multiples of256")
    if not diagnostic and (world, experts, topk, hidden, intermediate) not in (
        (4, 288, 8, 4096, 2048),
        (8, 288, 8, 4096, 2048),
    ):
        raise ValueError(
            "production candidate is gated to GLM H4096/I2048/E288/top8 EP4/EP8"
        )


@dataclass(frozen=True)
class StaticConfig:
    world: int = 8
    experts: int = 288
    topk: int = 8
    hidden: int = 4096
    intermediate: int = 2048
    capacity: int = 128
    swiglu_limit: float = 10.0
    tokens_per_item: int = 8
    transport: str = "p2p"
    fast_math: bool = False
    diagnostic: bool = False

    def __post_init__(self):
        validate_shape(
            self.world,
            self.experts,
            self.capacity,
            self.topk,
            self.hidden,
            self.intermediate,
            self.diagnostic,
        )
        if self.tokens_per_item not in (8, 16, 24, 32):
            raise ValueError("tokens_per_item must be8/16/24/32")
        if self.transport not in ("p2p", "auto", "nic"):
            raise ValueError("transport must be p2p/auto/nic")
        if self.swiglu_limit != 10.0 and not self.diagnostic:
            raise ValueError("actual GLM checkpoint requires swiglu_limit10")
