# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-uniform dispatch with native fallback for large dense batches."""

CAPACITY = 256
SMALL_CAPACITY = 128
DENSE_CAPACITY = 192


def tokens_per_item(sizes, workspace_capacity):
    """Use validated, rank-uniform CPU sizes; leave P scheduling unchanged."""
    if workspace_capacity == SMALL_CAPACITY and sum(sizes) >= 64 * len(sizes):
        return 32
    return 24


def uniform_bound(sizes, rows, ep_size, allow_extended=False):
    if ep_size not in (4, 8) or not isinstance(sizes, (list, tuple)):
        return None
    if len(sizes) != ep_size or type(rows) is not int or rows < 0:
        return None
    if any(type(value) is not int or value < 0 for value in sizes):
        return None
    bound = max(sizes)
    if rows > bound or bound > CAPACITY:
        return None
    if bound > SMALL_CAPACITY and (
        not allow_extended
        or (bound > DENSE_CAPACITY and sum(sizes) > SMALL_CAPACITY * ep_size)
    ):
        return None
    return max(bound, 1)


def supported_geometry(hidden, intermediate, experts, topk, local_experts, ep_size):
    return ep_size in (4, 8) and (
        hidden,
        intermediate,
        experts,
        topk,
        local_experts,
    ) == (4096, 2048, 288, 8, 288 // ep_size)
