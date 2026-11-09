# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic-only MoE combine rounding references; never imported by the kernel.

Inputs are per-token, original-top-k-slot FC2 values [T][K][H], global
expert IDs [T][K], and FP32 routing weights [T][K]. No token regrouping or
hidden all-expert FP32 sum is performed. CPU functions intentionally model
*specified* operation orders, not an assumed NCCL ring/tree.

Native separate-finalizer path: BF16 FC2 expert values -> per-owner FP32
top-k subtotal -> BF16 local partial -> actual NCCL BF16 ReduceScatter.
Candidate v2: BF16 FC2 expert values -> flat FP32 top-k sum -> BF16.

Use the parent's actual communicator with reduce_scatter_actual for the
last step. bf16_reduce_tree only explores explicitly provided trees.
"""

import ctypes
import math
import operator
from functools import lru_cache

from .reference_cpu import bf16_round, f32


@lru_cache(maxsize=1)
def _fmaf():
    # libSystem on macOS and libm on Linux expose fmaf. Fail rather than
    # silently substituting a double-precision or separate mul/add model.
    fn = getattr(ctypes.CDLL(None), "fmaf", None)
    if fn is None:
        from ctypes import util

        fn = ctypes.CDLL(util.find_library("m")).fmaf
    fn.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    fn.restype = ctypes.c_float
    return fn


def fma32(a, b, c):
    """C fmaf: FP32 operands and a single FP32 result rounding. CPU only."""
    return float(_fmaf()(f32(a), f32(b), f32(c)))


def _accumulate(acc, weight, value, arithmetic):
    if arithmetic == "fma32":
        return fma32(weight, value, acc)
    if arithmetic == "separate32":
        return f32(acc + f32(f32(weight) * f32(value)))
    raise ValueError("arithmetic must be fma32 or separate32")


def _validate(values, ids, weights, experts, world):
    experts, world = operator.index(experts), operator.index(world)
    if experts <= 0 or world <= 0 or experts % world:
        raise ValueError("positive equal contiguous EP expert partitions required")
    if not values or len(values) != len(ids) or len(values) != len(weights):
        raise ValueError("matching nonempty token dimensions required")
    k = len(ids[0])
    if not k or not values[0] or not values[0][0]:
        raise ValueError("nonempty top-k and hidden dimensions required")
    h = len(values[0][0])
    for rows, token_ids, token_weights in zip(values, ids, weights):
        if len(rows) != k or len(token_ids) != k or len(token_weights) != k:
            raise ValueError("ragged top-k dimensions")
        for row, eid, weight in zip(rows, token_ids, token_weights):
            if len(row) != h:
                raise ValueError("ragged hidden dimensions")
            eid = operator.index(eid)
            if eid < -1 or eid >= experts:
                raise ValueError("expert ID out of range (-1 is the only mask)")
            if not math.isfinite(weight) or any(not math.isfinite(x) for x in row):
                raise ValueError("finite diagnostic inputs required")
    return len(values), k, h, experts // world


def candidate_flat_combine(
    fc2_bf16, ids, weights, *, experts, world, arithmetic="fma32"
):
    """Frozen v2 combine: k-slot order, one final BF16 store."""
    t, k, h, _ = _validate(fc2_bf16, ids, weights, experts, world)
    out = [[0.0] * h for _ in range(t)]
    for token in range(t):
        for slot in range(k):
            if ids[token][slot] == -1:
                continue
            for col in range(h):
                out[token][col] = _accumulate(
                    out[token][col],
                    weights[token][slot],
                    bf16_round(fc2_bf16[token][slot][col]),
                    arithmetic,
                )
        out[token] = [bf16_round(x) for x in out[token]]
    return out


def native_local_partials(
    fc2_bf16, ids, weights, *, experts, world, arithmetic="fma32"
):
    """Separate-finalizer local partials [owner_rank][T][H], already BF16.

    Each owner visits original top-k slots in order, skips other owners,
    accumulates in FP32, and rounds its complete subtotal exactly once.
    This function does not perform the subsequent NCCL reduction.
    """
    t, k, h, local_e = _validate(fc2_bf16, ids, weights, experts, world)
    partials = [[[0.0] * h for _ in range(t)] for _ in range(world)]
    for owner in range(world):
        for token in range(t):
            for slot in range(k):
                eid = ids[token][slot]
                if eid == -1 or eid // local_e != owner:
                    continue
                for col in range(h):
                    partials[owner][token][col] = _accumulate(
                        partials[owner][token][col],
                        weights[token][slot],
                        bf16_round(fc2_bf16[token][slot][col]),
                        arithmetic,
                    )
            partials[owner][token] = [bf16_round(x) for x in partials[owner][token]]
    return partials


def bf16_reduce_tree(partials, tree):
    """Sum BF16 partials using an explicit binary tree of rank indices.

    Example ((0, 1), (2, 3)); every internal addition rounds BF16. All
    ranks must occur exactly once. This is an order-sensitivity model,
    NOT a claim about the algorithm used by the parent's NCCL library.
    """
    if not partials or not partials[0] or not partials[0][0]:
        raise ValueError("nonempty partials required")
    t, h = len(partials[0]), len(partials[0][0])
    if any(len(p) != t or any(len(row) != h for row in p) for p in partials):
        raise ValueError("ragged partials")
    leaves = []

    def visit(node):
        if isinstance(node, int) and not isinstance(node, bool):
            if not 0 <= node < len(partials):
                raise ValueError("invalid tree leaf")
            leaves.append(node)
            return [[bf16_round(x) for x in row] for row in partials[node]]
        if not isinstance(node, (tuple, list)) or len(node) != 2:
            raise ValueError("binary rank tree required")
        left, right = visit(node[0]), visit(node[1])
        return [
            [bf16_round(f32(a + b)) for a, b in zip(lrow, rrow)]
            for lrow, rrow in zip(left, right)
        ]

    out = visit(tree)
    if sorted(leaves) != list(range(len(partials))):
        raise ValueError("each rank must occur exactly once")
    return out


def fused_finalize_partials(fc2_fp32, ids, weights, *, experts, world, slot_order):
    """Alternative FINALIZE-tactic contract, not separate finalization.

    fc2_fp32 is the FP32 GEMM epilogue value BEFORE any BF16 expert
    storage. FP32 router weight multiplication happens before BF16 cast.
    Each weighted expert is BF16 atomically added into BF16 partials.
    slot_order [T][K] explicitly supplies an arrival order; production
    CTA/atomic scheduling does not guarantee this particular order.
    The no-bias GLM checkpoint is assumed, alpha already applied.
    """
    t, k, h, local_e = _validate(fc2_fp32, ids, weights, experts, world)
    if len(slot_order) != t or any(
        sorted(order) != list(range(k)) for order in slot_order
    ):
        raise ValueError("slot_order must be a top-k permutation for each token")
    partials = [[[0.0] * h for _ in range(t)] for _ in range(world)]
    for token in range(t):
        for slot in slot_order[token]:
            eid = ids[token][slot]
            if eid == -1:
                continue
            owner = eid // local_e
            for col in range(h):
                term = bf16_round(
                    f32(f32(weights[token][slot]) * f32(fc2_fp32[token][slot][col]))
                )
                partials[owner][token][col] = bf16_round(
                    f32(partials[owner][token][col] + term)
                )
    return partials


def reduce_scatter_actual(comm, output, local_partial, *, stream=None):
    """Enqueue the parent's SAME actual BF16 SUM communicator and stream.

    No tensor value reads, allocations, synchronization, or CPU collective.
    Parent supplies native-order [world*local_T, H] BF16 local partial
    and [local_T,H] BF16 output. Queue work on the same stream as the
    partial producer (or use existing event dependencies). The caller
    owns communicator health, lifetime, graph capture and measurement.
    Metadata checks below neither transfer values nor synchronize CUDA.
    Diagnostic adapter only; NOT wired into the production fused path.
    """
    if getattr(comm, "disabled", False):
        raise ValueError("disabled PyNccl communicator would silently skip RS")
    if (
        str(local_partial.dtype) != "torch.bfloat16"
        or output.dtype != local_partial.dtype
    ):
        raise ValueError("both tensors must be BF16")
    if output.device != local_partial.device or output.device != comm.device:
        raise ValueError("tensors must be on the communicator device")
    if not local_partial.is_contiguous() or not output.is_contiguous():
        raise ValueError("contiguous native layout required")
    if (
        local_partial.ndim != 2
        or output.ndim != 2
        or local_partial.shape[1] != output.shape[1]
    ):
        raise ValueError("expected [world*local_T,H] and [local_T,H]")
    if local_partial.shape[0] != comm.world_size * output.shape[0]:
        raise ValueError("ReduceScatter shape/world mismatch")
    # PyNccl default op is SUM; stream=None uses its current_stream().
    comm.reduce_scatter(output, local_partial, stream=stream)
    return output
