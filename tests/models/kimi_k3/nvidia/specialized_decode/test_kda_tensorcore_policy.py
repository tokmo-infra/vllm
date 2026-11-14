# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only TC selection using real meta tensors with the HiSparse page strides."""

import importlib.util
from pathlib import Path

import pytest
import torch

# Load the host-only module without importing the model's CUDA implementation.
_ROOT = Path(__file__).resolve().parents[5]
_SOURCE = _ROOT / (
    "vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/kda/tc_policy.py"
)
_SPEC = importlib.util.spec_from_file_location("kda_tensorcore_policy", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
policy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(policy)

HEADS, DIM, PAGE_BYTES = 12, 12 * 128, 6893568


def layouts(requests=256, tokens=5, dim_first=True, window=None, record_tokens=None):
    window = window if window is not None else tokens + 2
    record_tokens = record_tokens if record_tokens is not None else tokens
    meta = {"device": "meta"}
    pitch = -(-(4 * DIM + 128 + HEADS) // 16) * 16
    projected = torch.empty(requests * tokens, pitch, dtype=torch.bfloat16, **meta)
    conv_stride = (PAGE_BYTES // 2, window, 1)
    if not dim_first:
        conv_stride = (PAGE_BYTES // 2, 1, 3 * DIM)
    operands = (
        projected[:, :3 * DIM],
        torch.empty(3, 4, DIM, **meta),
        torch.empty(DIM, **meta),
        torch.empty_strided((6993, 3 * DIM, window), conv_stride,
                            dtype=torch.bfloat16, **meta),
        projected[:, 4 * DIM:4 * DIM + 128],
        projected[:, 4 * DIM + 128:4 * DIM + 128 + HEADS],
        torch.empty(HEADS, **meta), torch.empty(DIM, **meta),
        torch.empty(requests, 1, dtype=torch.int32, **meta),
        torch.empty_strided((6993, HEADS, 128, 128),
                            (PAGE_BYTES // 4, 16384, 128, 1), **meta),
        torch.empty(requests * tokens, DIM, dtype=torch.bfloat16, **meta),
        projected[:, 3 * DIM:4 * DIM], torch.empty(128, **meta),
        torch.empty(DIM, 128, dtype=torch.bfloat16, **meta),
        torch.empty(requests, dtype=torch.int32, **meta),
        torch.empty(256, HEADS, record_tokens, 128, **meta),
        torch.empty(256, HEADS, record_tokens, 256, dtype=torch.bfloat16, **meta),
        torch.empty(requests, dtype=torch.int32, **meta),
        torch.empty(256, dtype=torch.int32, **meta),
    )  # fmt: skip
    return tuple((tuple(t.shape), t.stride(), t.dtype) for t in operands)


def replace(specs, index, *, stride=None, dtype=None):
    shape, old_stride, old_dtype = specs[index]
    tensor = torch.empty_strided(
        shape, stride or old_stride, dtype=dtype or old_dtype, device="meta"
    )
    return (*specs[:index], (tuple(tensor.shape), tensor.stride(), tensor.dtype),
            *specs[index + 1:])  # fmt: skip


@pytest.mark.parametrize("requests", [16, 64, 224, 256])
@pytest.mark.parametrize("dim_first", [False, True])
def test_q5_accepts_hisparse_physical_pages(requests, dim_first):
    specs = layouts(requests, dim_first=dim_first)
    assert policy.tensorcore_layout(specs, HEADS, requests, 5, 148) == (dim_first, 7)


@pytest.mark.parametrize("tokens,dim_first", [(2, True), (6, True), (8, True)])
def test_query_or_shared_memory_limits_keep_the_fallback(tokens, dim_first):
    assert (
        policy.tensorcore_layout(
            layouts(tokens=tokens, dim_first=dim_first), HEADS, 256, tokens, 148
        )
        is None
    )


@pytest.mark.parametrize("window", [6, 8])
def test_convolution_window_must_match_the_verify_shape(window):
    assert policy.tensorcore_layout(layouts(window=window), HEADS, 256, 5, 148) is None


@pytest.mark.parametrize(
    "index,stride,dtype",
    [
        (0, None, torch.float32),
        (0, (6289, 1), None),
        (3, (PAGE_BYTES // 2 + 1, 7, 1), None),
        (9, None, torch.bfloat16),
        (9, (PAGE_BYTES // 4, 16384, 1, 128), None),
        (9, (PAGE_BYTES // 4 + 1, 16384, 128, 1), None),
        (9, (PAGE_BYTES // 4, 16385, 128, 1), None),
        (13, None, torch.float32),
        (13, (129, 1), None),
        (13, (256, 2), None),
    ],
)
def test_tma_and_streaming_constraints_reject_incompatible_storage(
    index, stride, dtype
):
    specs = replace(layouts(), index, stride=stride, dtype=dtype)
    assert policy.tensorcore_layout(specs, HEADS, 256, 5, 148) is None


def test_large_replay_records_keep_the_fallback():
    assert (
        policy.tensorcore_layout(layouts(record_tokens=8), HEADS, 256, 5, 148) is None
    )


def test_per_cta_metadata_is_included_in_the_shared_memory_budget():
    # One CTA: 42 requests fit; rounding the 43rd request to a two-request tile
    # crosses the limit. This exercises the metadata term, not only fixed buffers.
    assert policy.tensorcore_layout(layouts(42), HEADS, 42, 5, 1) == (True, 7)
    assert policy.tensorcore_layout(layouts(43), HEADS, 43, 5, 1) is None


@pytest.mark.parametrize(
    "heads,requests,sms", [(0, 16, 148), (12, 0, 148), (12, 16, 0)]
)
def test_empty_or_invalid_launch_dimensions_keep_the_fallback(heads, requests, sms):
    assert policy.tensorcore_layout(layouts(requests), heads, requests, 5, sms) is None


@pytest.mark.parametrize("requests", [1, 3, 16, 63, 224, 255, 256])
@pytest.mark.parametrize("sms", [1, 16, 148])
def test_item_table_covers_every_cta_including_odd_batch_padding(requests, sms):
    tiles = list(range(0, requests, 2))
    assignment = [(request, head) for request in tiles for head in range(HEADS)]
    counts = [len(assignment[cta::sms]) * 2 for cta in range(sms)]
    assert policy.cta_items(HEADS, requests, sms) == max(counts)


@pytest.mark.parametrize(
    "config", [None, {}, {"decode_backend": "latency_specialized"}]
)
def test_tensorcore_requires_an_explicit_opt_in(config):
    assert policy.tensorcore_enabled(config) is False


@pytest.mark.parametrize("enabled", [False, True])
def test_tensorcore_accepts_boolean_configuration(enabled):
    assert policy.tensorcore_enabled({"kda_fused_tc": enabled}) is enabled


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_tensorcore_rejects_ambiguous_configuration(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        policy.tensorcore_enabled({"kda_fused_tc": value})
