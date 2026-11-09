# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initialization-only aliases of unmodified ModelOpt NVFP4 weight/scale tensors.

No global-scale baking into E4M3, no NVFP4->MXFP4 conversion, no W13 copy.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Nvfp4Weights:
    w13: Any
    w2: Any
    w13_scale: Any
    w2_scale: Any
    g1_alphas: Any
    g2_alphas: Any
    a1_gscale: Any
    a2_gscale: Any
    w13_layout: int = 1
    scale_layout: int = 1


def alias_native(
    *,
    w13,
    w2,
    w13_scale,
    w2_scale,
    g1_alphas,
    g2_alphas,
    a1_gscale,
    a2_gscale,
    local_experts,
    hidden=4096,
    intermediate=2048,
    w13_layout=1,
    scale_layout=1,
    a1_quant_scalar=None,
):
    """Bind native FI/CUTLASS post-process tensors; ALPHAS ALREADY INCLUDE input_scale.

    Caller must establish finite positive calibration at initialization and pass
    the same rank-uniform input quant calibration as the native baseline.
    `a1_gscale` may be scalar or local_E. For an array, the caller must provide
    a1_quant_scalar from the EXACT native input-quantizer reduction/normalization;
    no min/max policy is guessed here. The hot path consumes only that device scalar.
    `a2_gscale` is scalar or local_E. Do not derive different per-rank calibration.
    """
    import torch

    E, H, intermediate_dim = local_experts, hidden, intermediate
    expected = (
        (w13, (E, 2 * intermediate_dim, H // 2)),
        (w2, (E, H, intermediate_dim // 2)),
        (w13_scale, (E, 2 * intermediate_dim, H // 16)),
        (w2_scale, (E, H, intermediate_dim // 16)),
    )
    device = w13.device
    if device.type != "cuda":
        raise ValueError("native NVFP4 binding requires CUDA-resident tensors")
    for tensor, shape in expected:
        if (
            tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
            or tensor.device != device
        ):
            raise ValueError(
                f"expected contiguous {shape} on {device}; got {tensor.shape}"
            )
    if w13.dtype not in (torch.uint8, torch.int8) or w2.dtype not in (
        torch.uint8,
        torch.int8,
    ):
        raise ValueError("FP4 weights must be packed8-bit storage")
    if w13_scale.dtype != torch.float8_e4m3fn or w2_scale.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "require actual E4M3 scale dtype; uint8/MXFP4 scale tags rejected"
        )
    for name, tensor, sizes in (
        ("g1_alphas", g1_alphas, (E,)),
        ("g2_alphas", g2_alphas, (E,)),
        ("a1_gscale", a1_gscale, (1, E)),
        ("a2_gscale", a2_gscale, (1, E)),
    ):
        if (
            tensor.dtype != torch.float32
            or tensor.numel() not in sizes
            or tensor.device != device
        ):
            raise ValueError(f"{name}: expected CUDA FP32 sizes{sizes}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if w13_layout not in (0, 1, 2) or scale_layout not in (0, 1):
        raise ValueError("unrecognized weight or blockscale layout")
    a1 = a1_gscale
    if a1_quant_scalar is not None:
        if (
            a1_quant_scalar.numel() != 1
            or a1_quant_scalar.dtype != torch.float32
            or a1_quant_scalar.device != device
            or not a1_quant_scalar.is_contiguous()
        ):
            raise ValueError(
                "native-selected a1_quant_scalar must be contiguous CUDA FP32 scalar"
            )
        a1 = a1_quant_scalar
    elif a1_gscale.numel() != 1:
        raise ValueError(
            "array a1_gscale needs explicit native-selected a1_quant_scalar; "
            "no guessed min/max"
        )
    # Only tiny per-expert activation calibration expansion may allocate here.
    a2 = a2_gscale.reshape(-1)
    if a2.numel() == 1:
        a2 = a2.expand(E).contiguous()
    return Nvfp4Weights(
        w13.view(torch.uint8),
        w2.view(torch.uint8),
        w13_scale.view(torch.uint8),
        w2_scale.view(torch.uint8),
        g1_alphas.reshape(E),
        g2_alphas.reshape(E),
        a1.reshape(1),
        a2,
        w13_layout,
        scale_layout,
    )


def from_checkpoint(
    *,
    w13,
    w2,
    w13_scale,
    w2_scale,
    w13_global,
    w2_global,
    a1_gscale,
    a2_gscale,
    local_experts,
    hidden=4096,
    intermediate=2048,
    a1_quant_scalar=None,
):
    """Raw [gate;up] checkpoint view; form FP32 alphas only, preserve FP4/E4M3 bytes.

    Must use exactly the same normalized global input calibration as native.
    Call before native process_weights_after_loading mutates weight globals.
    """
    return alias_native(
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        g1_alphas=w13_global / a1_gscale,
        g2_alphas=w2_global / a2_gscale,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        local_experts=local_experts,
        hidden=hidden,
        intermediate=intermediate,
        w13_layout=0,
        scale_layout=0,
        a1_quant_scalar=a1_quant_scalar,
    )
