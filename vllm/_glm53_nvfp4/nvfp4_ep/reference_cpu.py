# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small independent CPU model of the NVFP4 format/register contracts.

This module is diagnostic only; it is never imported by runtime workspace/JIT.
"""

import math
import struct

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def bf16_round(x):
    bits = struct.unpack("I", struct.pack("f", f32(x)))[0]
    if bits & 0x7F800000 != 0x7F800000:
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("f", struct.pack("I", bits))[0]


def native_glu_rounding_reference(gate_acc, up_acc, alpha, limit=10.0):
    """Rounding-point reference, not a model of CUDA fast-exp/fast-div last bits."""
    gate = min(bf16_round(f32(gate_acc * alpha)), limit)
    up = min(max(bf16_round(f32(up_acc * alpha)), -limit), limit)
    ex = f32(math.exp(-gate)) if gate > -88.0 else float("inf")
    sigmoid = f32(1.0 / f32(1.0 + ex))
    activated = f32(f32(gate * sigmoid) * up)
    return bf16_round(activated)


def decode_fp4(code):
    return -E2M1[code & 7] if code & 8 else E2M1[code & 7]


def encode_fp4(x):
    value = min(6.0, abs(x))
    c = min(range(8), key=lambda i: (abs(value - E2M1[i]), i & 1))
    return c | (8 if math.copysign(1.0, x) < 0 else 0)


def decode_e4m3(code):
    if not 0 <= code <= 126:
        raise ValueError("positive finite E4M3 scales only, not UE8M0 exponents")
    e, m = code >> 3, code & 7
    return m * 2.0**-9 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)


def encode_e4m3(x):
    if not math.isfinite(x) or x < 0:
        raise ValueError("finite nonnegative NVFP4 block scale required")
    x = min(448.0, x)
    return min(range(127), key=lambda c: (abs(x - decode_e4m3(c)), c & 1))


def pack_fp4(values):
    if len(values) % 2:
        raise ValueError("packed E2M1 requires an even value count")
    return bytes(
        encode_fp4(values[i]) | (encode_fp4(values[i + 1]) << 4)
        for i in range(0, len(values), 2)
    )


def unpack_fp4(values):
    return [decode_fp4((b >> shift) & 15) for b in values for shift in (0, 4)]


def quant16(values, a_gscale):
    if len(values) != 16 or not math.isfinite(a_gscale) or a_gscale <= 0:
        raise ValueError(
            "NVFP4 quant requires16 values and a positive quant multiplier"
        )
    s = encode_e4m3(f32(a_gscale * f32(max(map(abs, values)) / 6.0)))
    scale = decode_e4m3(s)
    factor = a_gscale / scale if scale else 0.0
    return pack_fp4([f32(v * factor) for v in values]), s


def quant_row(values, a_gscale):
    if len(values) % 16:
        raise ValueError("NVFP4 group16 required")
    data, scales = bytearray(), bytearray()
    for i in range(0, len(values), 16):
        q, s = quant16(values[i : i + 16], a_gscale)
        data.extend(q)
        scales.append(s)
    return bytes(data), bytes(scales)


def dequant_row(packed, scales, global_dequant=1.0):
    vals = unpack_fp4(packed)
    if len(vals) != len(scales) * 16:
        raise ValueError("data/E4M3 group16 scale shape mismatch")
    return [
        v * decode_e4m3(scales[i // 16]) * global_dequant for i, v in enumerate(vals)
    ]


def packed_word(data):
    return int.from_bytes(data, "little")


def mma_fragments(weight_rows, weight_scales, activation_rows, activation_scales):
    """Build one actual PTX16x8x64 fragment set as the CUDA helper does.

    Return A[32][4] B[32][2] SFA[32] SFB[32]; use integer-byte-coded test
    scales to separately prove layout without assuming all scales identical.
    """
    a, b, sa, sb = [], [], [], []
    for lane in range(32):
        g, c = lane // 4, lane % 4
        off = c * 4
        a.append(
            [
                packed_word(weight_rows[r][off + k : off + k + 4])
                for k in (0, 16)
                for r in (g, g + 8)
            ]
        )
        b.append(
            [packed_word(activation_rows[g][off + k : off + k + 4]) for k in (0, 16)]
        )
        sa.append(packed_word(weight_scales[g + 8 * (c & 1)]))
        sb.append(packed_word(activation_scales[g]))
    return a, b, sa, sb


def reference_mma_tile(weight_rows, weight_scales, activation_rows, activation_scales):
    wa = [dequant_row(w, s) for w, s in zip(weight_rows, weight_scales)]
    xb = [dequant_row(x, s) for x, s in zip(activation_rows, activation_scales)]
    return [[sum(w[k] * x[k] for k in range(64)) for x in xb] for w in wa]
