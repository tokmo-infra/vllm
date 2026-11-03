# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peer-major wire packing, native MXFP8 staging and source-side reduction."""

import torch

from vllm.model_executor.layers.fused_moe.prepare_megamoe import _quantize_fp8_groups
from vllm.triton_utils import tl, triton


@triton.jit
def _encode_wire(
    hidden,
    weights,
    ids,
    owner_nodes,
    owner_lanes,
    destination_nodes,
    output,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    EP: tl.constexpr,
    XS: tl.constexpr,
    WS: tl.constexpr,
    WKS: tl.constexpr,
    IS: tl.constexpr,
    IKS: tl.constexpr,
    ROW: tl.constexpr,
    GATE: tl.constexpr,
    ROUTE: tl.constexpr,
    FP8: tl.constexpr,
    BM: tl.constexpr,
    KT: tl.constexpr,
    row_map,
    COMPACT: tl.constexpr,
    SKIP_START: tl.constexpr = 0,
    SKIP_ROWS: tl.constexpr = 0,
):
    flat = tl.program_id(0) * BM + tl.arange(0, BM)
    active_tokens = T - SKIP_ROWS
    source = flat % active_tokens
    if SKIP_ROWS:
        source += tl.where(source >= SKIP_START, SKIP_ROWS, 0)
    destination = flat // active_tokens
    target_row = (
        tl.load(row_map + flat, flat < N * active_tokens, -1) if COMPACT else source
    )
    output_row = destination * T + target_row
    columns = tl.program_id(1) * 128 + tl.arange(0, 128)
    valid = (
        (flat[:, None] < N * active_tokens)
        & (target_row[:, None] >= 0)
        & (columns[None, :] < H)
    )
    x = tl.load(hidden + source[:, None] * XS + columns[None, :], valid, 0)
    if FP8:
        values, _, scale = _quantize_fp8_groups(x.to(tl.float32), BM, 128, 32, True)
        tl.store(
            output + output_row[:, None] * ROW + columns[None, :],
            values.to(tl.uint8, bitcast=True),
            valid,
        )
        scale_columns = tl.program_id(1) * 4 + tl.arange(0, 4)
        tl.store(
            output + output_row[:, None] * ROW + H + scale_columns[None, :],
            scale,
            (flat[:, None] < N * active_tokens) & (target_row[:, None] >= 0),
        )
    else:
        bf16 = output.to(tl.pointer_type(tl.bfloat16))
        tl.store(bf16 + output_row[:, None] * (ROW // 2) + columns[None, :], x, valid)
    if K > 0 and tl.program_id(1) == 0:
        experts = tl.arange(0, KT)
        route_valid = (
            (flat[:, None] < N * active_tokens)
            & (target_row[:, None] >= 0)
            & (experts[None, :] < K)
        )
        physical = tl.load(
            ids + source[:, None] * IS + experts[None, :] * IKS, route_valid, -1
        ).to(tl.int64)
        rank = tl.minimum(tl.maximum(physical, 0), E - 1) // (E // EP)
        target = tl.load(destination_nodes + destination, flat < N * T, 0)
        keep = (
            (physical >= 0)
            & (physical < E)
            & (tl.load(owner_nodes + rank) == target[:, None])
        )
        local = tl.load(owner_lanes + rank) * (E // EP) + physical % (E // EP)
        gate = tl.load(
            weights + source[:, None] * WS + experts[None, :] * WKS, route_valid, 0
        )
        gout = output.to(tl.pointer_type(tl.float32))
        rout = output.to(tl.pointer_type(tl.int32))
        offsets = output_row[:, None] * (ROW // 4) + experts[None, :]
        tl.store(gout + offsets + GATE // 4, tl.where(keep, gate, 0), route_valid)
        tl.store(rout + offsets + ROUTE // 4, tl.where(keep, local, -1), route_valid)


def _validate_activations(hidden):
    if hidden.dtype != torch.bfloat16 or hidden.shape[1] % 128 or hidden.stride(1) != 1:
        raise ValueError("Wire packing requires BF16 rows and H divisible by 128")


def pack_fixed(
    hidden,
    gates,
    ids,
    tables,
    num_experts,
    world_size,
    wire_dtype,
    *,
    out=None,
    row_map=None,
):
    _validate_activations(hidden)
    tokens, width = hidden.shape
    nodes, top_k = tables[2].numel(), ids.shape[1]
    fp8 = wire_dtype == "mxfp8"
    activation_bytes = width + width // 32 if fp8 else 2 * width
    route_offset = activation_bytes + 4 * top_k
    row_bytes = route_offset + 4 * top_k
    if out is None:
        out = torch.empty(
            (nodes * tokens, row_bytes), device=hidden.device, dtype=torch.uint8
        )
    if (
        out.shape != (nodes * tokens, row_bytes)
        or out.dtype != torch.uint8
        or not out.is_contiguous()
        or out.device != hidden.device
    ):
        raise ValueError("Incorrect dispatch buffer layout")
    if tokens:
        block = 32 if tokens >= 1024 else 8 if tokens >= 64 else 1
        _encode_wire[(triton.cdiv(nodes * tokens, block), width // 128)](
            hidden,
            gates,
            ids,
            *tables,
            out,
            tokens,
            width,
            top_k,
            nodes,
            num_experts,
            world_size,
            hidden.stride(0),
            *gates.stride(),
            *ids.stride(),
            row_bytes,
            activation_bytes,
            route_offset,
            fp8,
            block,
            triton.next_power_of_2(top_k),
            row_map,
            row_map is not None,
        )
    return out, activation_bytes, activation_bytes, route_offset


def pack_mxfp8(hidden, *, out=None, exclude_rows=None):
    """Encode native 1x32 MXFP8; excluded rows leave output storage untouched."""
    _validate_activations(hidden)
    tokens, width = hidden.shape
    start, skip = exclude_rows if exclude_rows is not None else (0, 0)
    if start < 0 or skip < 0 or start + skip > tokens:
        raise ValueError("Excluded MXFP8 rows exceed the input")
    active = tokens - skip
    row_bytes = width + width // 32
    if out is None:
        out = torch.empty((tokens, row_bytes), device=hidden.device, dtype=torch.uint8)
    if (
        out.shape != (tokens, row_bytes)
        or out.dtype != torch.uint8
        or not out.is_contiguous()
    ):
        raise ValueError("Incorrect MXFP8 output layout")
    if active:
        block = 32 if active >= 1024 else 8 if active >= 64 else 1
        _encode_wire[(triton.cdiv(active, block), width // 128)](
            hidden,
            None,
            None,
            None,
            None,
            None,
            out,
            tokens,
            width,
            0,
            1,
            1,
            1,
            hidden.stride(0),
            0,
            0,
            0,
            0,
            row_bytes,
            0,
            0,
            True,
            block,
            1,
            None,
            False,
            start,
            skip,
        )
    return out


@triton.jit
def _route_map(
    ids,
    owner_nodes,
    destinations,
    row_map,
    counts,
    T: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    EP: tl.constexpr,
    IS: tl.constexpr,
    IKS: tl.constexpr,
    VALID: tl.constexpr,
    SPARSE: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    peer = tl.program_id(0)
    rows = tl.arange(0, BT)
    active = rows < VALID
    if SPARSE:
        ks = tl.arange(0, BK)
        physical = tl.load(
            ids + rows[:, None] * IS + ks[None, :] * IKS,
            (rows[:, None] < T) & (ks[None, :] < K),
            -1,
        ).to(tl.int64)
        rank = tl.minimum(tl.maximum(physical, 0), E - 1) // (E // EP)
        target = tl.load(destinations + peer)
        keep = (
            (physical >= 0)
            & (physical < E)
            & (ks[None, :] < K)
            & (tl.load(owner_nodes + rank) == target)
        )
        active = active & (tl.sum(keep.to(tl.int32), 1) > 0)
    offsets = tl.cumsum(active.to(tl.int32)) - 1
    tl.store(row_map + peer * T + rows, tl.where(active, offsets, -1), rows < T)
    tl.store(counts + peer, tl.sum(active.to(tl.int32)))


def pack_compact(
    hidden,
    gates,
    ids,
    tables,
    num_experts,
    world_size,
    wire_dtype,
    out,
    row_map,
    counts,
    valid_tokens,
    sparse,
):
    tokens = hidden.shape[0]
    _route_map[(tables[2].numel(),)](
        ids,
        tables[0],
        tables[2],
        row_map,
        counts,
        tokens,
        ids.shape[1],
        num_experts,
        world_size,
        *ids.stride(),
        valid_tokens,
        sparse,
        triton.next_power_of_2(tokens),
        triton.next_power_of_2(ids.shape[1]),
        num_warps=8 if tokens >= 512 else 4,
    )
    pack_fixed(
        hidden,
        gates,
        ids,
        tables,
        num_experts,
        world_size,
        wire_dtype,
        out=out,
        row_map=row_map,
    )


@triton.jit
def _mask_received_routes(
    wire,
    counts,
    ROW: tl.constexpr,
    CAP: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GATE: tl.constexpr,
    ROUTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, k = offset // K, offset % K
    count = tl.load(counts + row // CAP, row < N * CAP, 0)
    invalid = (row < N * CAP) & (row % CAP >= count)
    tl.store(
        wire.to(tl.pointer_type(tl.int32)) + row * (ROW // 4) + ROUTE // 4 + k,
        -1,
        invalid,
    )
    tl.store(
        wire.to(tl.pointer_type(tl.float32)) + row * (ROW // 4) + GATE // 4 + k,
        0,
        invalid,
    )


def mask_received_routes(wire, counts, capacity, top_k, gate_offset, route_offset):
    _mask_received_routes[(triton.cdiv(wire.shape[0] * top_k, 256),)](
        wire,
        counts,
        wire.shape[1],
        capacity,
        counts.numel(),
        top_k,
        gate_offset,
        route_offset,
        256,
    )


@triton.jit
def _combine_compact(
    returned,
    local,
    row_map,
    output,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    SELF: tl.constexpr,
    FP8: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // H, index % H
    total = tl.full((BLOCK,), 0, tl.float32)
    for peer in range(N):
        source = tl.load(row_map + peer * T + row, index < T * H, -1)
        valid = (index < T * H) & (source >= 0)
        if peer == SELF:
            value = tl.load(local + (peer * T + source) * H + col, valid, 0).to(
                tl.float32
            )
        elif FP8:
            base = (peer * T + source) * (H + H // 32)
            value = (
                tl.load(returned + base + col, valid, 0)
                .to(tl.float8e4nv, bitcast=True)
                .to(tl.float32)
            )
            exponent = tl.load(returned + base + H + col // 32, valid, 127).to(
                tl.uint32
            )
            value *= (exponent << 23).to(tl.float32, bitcast=True)
        else:
            value = tl.load(returned + (peer * T + source) * H + col, valid, 0).to(
                tl.float32
            )
        total += value
    tl.store(output + index, total, index < T * H)


def combine_compact(returned, local, row_map, out, self_rank, return_dtype):
    _combine_compact[(triton.cdiv(out.numel(), 1024),)](
        returned,
        local,
        row_map,
        out,
        out.shape[0],
        out.shape[1],
        row_map.shape[0],
        self_rank,
        return_dtype == "mxfp8",
        1024,
    )


@triton.jit
def _decode_wire(
    wire,
    output,
    T: tl.constexpr,
    H: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // H, offset % H
    valid = offset < T * H
    fp8 = tl.load(wire + row * STRIDE + col, valid, 0).to(tl.float8e4nv, bitcast=True)
    exponent = tl.load(wire + row * STRIDE + H + col // 32, valid, 127).to(tl.uint32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    tl.store(output + offset, fp8.to(tl.float32) * scale, valid)


def unpack_mxfp8(wire, width):
    output = torch.empty(
        (wire.shape[0], width), dtype=torch.bfloat16, device=wire.device
    )
    if output.numel():
        _decode_wire[(triton.cdiv(output.numel(), 1024),)](
            wire,
            output,
            wire.shape[0],
            width,
            wire.stride(0),
            1024,
        )
    return output


@triton.jit
def _reduce_partials(
    partials, output, ELEMENTS: tl.constexpr, NODES: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0, tl.float32)
    for node in range(NODES):
        total += tl.load(partials + node * ELEMENTS + index, index < ELEMENTS, 0).to(
            tl.float32
        )
    tl.store(output + index, total, index < ELEMENTS)


def combine_fixed(returned, tokens, width, nodes, *, out=None):
    if out is None:
        out = torch.empty((tokens, width), device=returned.device, dtype=returned.dtype)
    if out.numel():
        _reduce_partials[(triton.cdiv(out.numel(), 1024),)](
            returned,
            out,
            tokens * width,
            nodes,
            1024,
        )
    return out


@triton.jit
def _stage_mxfp8(
    wire,
    gates,
    ids,
    x,
    sf,
    idx_out,
    gates_out,
    tokens,
    H: tl.constexpr,
    K: tl.constexpr,
    WS: tl.constexpr,
    GS: tl.constexpr,
    GKS: tl.constexpr,
    IS: tl.constexpr,
    IKS: tl.constexpr,
    XS: tl.constexpr,
    XKS: tl.constexpr,
    SS: tl.constexpr,
    SKS: tl.constexpr,
    OS: tl.constexpr,
    OKS: tl.constexpr,
    OGS: tl.constexpr,
    OGKS: tl.constexpr,
    BM: tl.constexpr,
    KT: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    block = tl.program_id(1)
    cols = block * 128 + tl.arange(0, 128)
    mask = (rows[:, None] < tokens) & (cols[None, :] < H)
    values = tl.load(wire + rows[:, None] * WS + cols[None, :], mask, 0)
    tl.store(
        x + rows[:, None] * XS + cols[None, :] * XKS,
        values.to(tl.float8e4nv, bitcast=True),
        mask,
    )
    byte = tl.arange(0, 4)
    exponents = tl.load(
        wire + rows[:, None] * WS + H + block * 4 + byte[None, :],
        rows[:, None] < tokens,
        0,
    ).to(tl.uint32)
    packed = tl.sum(exponents << (8 * byte[None, :]), axis=1).to(tl.int32)
    tl.store(sf + rows * SS + block * SKS, packed, rows < tokens)
    if block == 0:
        ks = tl.arange(0, KT)
        valid = (rows[:, None] < tokens) & (ks[None, :] < K)
        routes = tl.load(ids + rows[:, None] * IS + ks[None, :] * IKS, valid, -1)
        weights = tl.load(gates + rows[:, None] * GS + ks[None, :] * GKS, valid, 0)
        tl.store(idx_out + rows[:, None] * OS + ks[None, :] * OKS, routes, valid)
        tl.store(gates_out + rows[:, None] * OGS + ks[None, :] * OGKS, weights, valid)


def stage_mxfp8(packed, gates, ids, x, x_sf, topk_idx, topk_weights):
    """Copy native MXFP8 bytes and pack four E8M0 scales per int32 word.

    Routing is already mapped to physical, island-local expert IDs. In
    particular this stage must not remap EPLB or count destination copies.
    """
    tokens, width = x.shape
    if (
        width % 128
        or packed.dtype != torch.uint8
        or packed.shape != (tokens, width + width // 32)
        or x.dtype != torch.float8_e4m3fn
        or x_sf.dtype != torch.int32
        or x_sf.shape != (tokens, width // 128)
    ):
        raise ValueError("Expected native MXFP8 values and packed int32 E8M0 scales")
    if gates.shape != ids.shape or ids.shape[0] != tokens:
        raise ValueError("Routing shape does not match MXFP8 rows")
    if topk_idx.shape != ids.shape or topk_weights.shape != ids.shape:
        raise ValueError("Output routing shape does not match input routing")
    if tokens == 0:
        return
    block_m = 32 if tokens >= 1024 else 8 if tokens >= 64 else 1
    _stage_mxfp8[(triton.cdiv(tokens, block_m), width // 128)](
        packed,
        gates,
        ids,
        x,
        x_sf,
        topk_idx,
        topk_weights,
        tokens,
        width,
        ids.shape[1],
        packed.stride(0),
        *gates.stride(),
        *ids.stride(),
        *x.stride(),
        *x_sf.stride(),
        *topk_idx.stride(),
        *topk_weights.stride(),
        block_m,
        triton.next_power_of_2(ids.shape[1]),
    )
