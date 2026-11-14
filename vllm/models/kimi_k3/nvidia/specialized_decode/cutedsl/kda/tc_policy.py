# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only layout and shared-memory policy for tensor-core KDA verification."""

from typing import Any

import torch

K, TILE, PROBES, META = 128, 2, 16, 8
QUARTER_BYTES = K * 32 * 4
FB_BYTES = K * K * 2
XG = 4 * 2 * PROBES + 4
X_FLOATS = K // 4 * XG
FA_BYTES = PROBES * K * 2
SMEM_LIMIT = 227 * 1024
TensorLayout = tuple[tuple[int, ...], tuple[int, ...], torch.dtype]


def tensorcore_enabled(additional_config: Any) -> bool:
    """Read the opt-in without treating strings or integers as booleans."""
    value = (
        additional_config.get("kda_fused_tc", False)
        if isinstance(additional_config, dict)
        else False
    )
    if not isinstance(value, bool):
        raise ValueError("additional_config.kda_fused_tc must be a boolean")
    return value


def stream_layout(specs, tokens: int) -> tuple[bool, int] | None:
    """The aligned convolution layout and window, or None if it cannot stream."""
    (_, x_stride, _), conv = specs[0], specs[3]
    shape, stride, dtype = conv
    size = torch.empty((), dtype=dtype, device="meta").element_size()
    window = shape[-1]
    if stride[-1] == 1 and stride[-2] == window:
        dim_first = True
    elif stride[-2] == 1:
        dim_first = False
    else:
        return None
    aligned = (stride[0] * size) % 16 == 0 and (x_stride[0] * 2) % 16 == 0
    if not dim_first:
        aligned = aligned and (stride[-1] * size) % 16 == 0
    return (dim_first, window) if aligned and window >= tokens + 2 else None


def operand_bytes(tokens: int, rq: int) -> int:
    p2 = 1 << (tokens * tokens - 1).bit_length()
    floats = X_FLOATS + (2 * rq + 1 + tokens) * K + 8 + p2
    return floats * 4 + META * 4


def staging_bytes(
    tokens: int, rq: int, conv_dim_first: bool, window: int, conv_size: int
) -> int:
    rows = 3 * tokens * K * 2
    conv = 3 * K * window * conv_size if conv_dim_first else 9 * K * conv_size
    records = rq * 2 * K * 2 + rq * K * 4
    return rows + FA_BYTES + conv + records


def cta_items(heads: int, requests: int, ctas: int) -> int:
    """Worst CTA's padded item count, including an odd batch's dead tile row."""
    tiles = -(-requests // TILE) * heads
    return -(-tiles // ctas) * TILE


def shared_bytes(
    tokens: int, rq: int, conv_dim_first: bool, window: int, conv_size: int, items: int
) -> int:
    staging = staging_bytes(tokens, rq, conv_dim_first, window, conv_size)
    sums = 2 * 64 * 4 * 4 + 2 * 32 * 4 * 4 + 8 * 4 * 4 + 2 * tokens * K * 2
    quarters_fb = 4 * QUARTER_BYTES + 2 * FB_BYTES
    fixed = quarters_fb + 2 * operand_bytes(tokens, rq) + 2 * staging + sums + 2048
    return fixed + items * 3 * 4


def tensorcore_layout(
    specs: tuple[TensorLayout, ...], heads: int, requests: int, tokens: int, sms: int
) -> tuple[bool, int] | None:
    """Return a supported layout; unsupported operands keep the existing kernel."""
    if min(heads, requests, sms) <= 0 or not 3 <= tokens <= PROBES // 2:
        return None
    streamed = stream_layout(specs, tokens)
    if streamed is None:
        return None
    dim_first, window = streamed
    state, fb = specs[9], specs[13]
    if (
        window != tokens + 2
        or specs[0][2] != torch.bfloat16
        or state[2] != torch.float32
        or state[0][-2:] != (K, K)
        or state[1][-2:] != (K, 1)
        or state[1][0] * 4 % 16 != 0
        or state[1][1] * 4 % 16 != 0
        or fb[2] != torch.bfloat16
        or fb[1][-1] != 1
        or fb[1][0] * 2 % 16 != 0
    ):
        return None
    conv_size = torch.empty((), dtype=specs[3][2], device="meta").element_size()
    size = shared_bytes(
        tokens,
        specs[15][0][2],
        dim_first,
        window,
        conv_size,
        cta_items(heads, requests, sms),
    )
    return streamed if size <= SMEM_LIMIT else None
