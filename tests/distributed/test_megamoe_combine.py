# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused Shared add must preserve the routed BF16 rounding boundary."""

import pytest
import torch

from vllm.distributed.megamoe_rdma_ops import combine_compact, combine_fixed


@pytest.mark.parametrize("nodes", [2, 3])
@pytest.mark.parametrize("tokens", [1, 33, 1025])
def test_fused_combine_preserves_intermediate_rounding(nodes, tokens):
    torch.manual_seed(53)
    width = 256
    partials = torch.randn(nodes, tokens, width, device="cuda").bfloat16()
    shared = torch.randn(tokens, width, device="cuda").bfloat16()
    # Removing the BF16 round between reduction and Shared changes 0 to 1/256.
    partials[:, :, 0] = 0
    partials[0, :, 0] = 1
    partials[1, :, 0] = 1 / 256
    shared[:, 0] = -1
    total = torch.zeros_like(shared, dtype=torch.float32)
    for partial in partials:
        total.add_(partial.float())
    expected = total.bfloat16() + shared
    actual = combine_fixed(partials, tokens, width, nodes, shared=shared)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.count_nonzero(actual[:, 0]).item() == 0

    # Compact inverse routing must select the local slot and omit masked rows.
    local = partials.clone()
    returned = partials.clone()
    returned[1].fill_(float("nan"))
    row_map = torch.stack(
        [torch.arange(tokens - 1, -1, -1, device="cuda") for _ in range(nodes)]
    ).int()
    row_map[0, ::3] = -1
    total.zero_()
    for node in range(nodes):
        positions = row_map[node].long()
        source = local[node] if node == 1 else returned[node]
        total.add_(
            torch.where(
                positions[:, None] >= 0, source[positions.clamp_min(0)], 0
            ).float()
        )
    expected = total.bfloat16() + shared
    out = torch.empty_like(shared)
    combine_compact(returned, local, row_map, out, 1, "bfloat16", shared=shared)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_fused_combine_rejects_a_different_shared_precision():
    partials = torch.zeros(2, 1, 256, device="cuda", dtype=torch.bfloat16)
    shared = torch.zeros(1, 256, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="BF16"):
        combine_fixed(partials, 1, 256, 2, shared=shared)
