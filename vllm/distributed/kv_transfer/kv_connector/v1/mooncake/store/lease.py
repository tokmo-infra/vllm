# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded fresh-replica reads without bypassing native lease validation."""

from collections.abc import Sequence
from typing import Any

LEASE_EXPIRED = -707
REFRESHABLE_GET_ERRORS = frozenset((LEASE_EXPIRED, -800))


def validate_batch_results(results: Sequence[int], count: int) -> None:
    """Fail closed on truncated or malformed native batch responses."""
    if len(results) != count or any(type(value) is not int for value in results):
        raise ValueError("Mooncake batch returned malformed per-key statuses")


def retry_expired_get_once(
    store: Any,
    keys: list[str],
    buffers: list[list[int]],
    sizes: list[list[int]],
    results: list[int],
) -> list[int]:
    """Re-query replicas and re-read only -707/-800 keys, at most once.

    Use the full GET, not a transfer against cached replica descriptors. The
    native Store still enforces its lease on both attempts. The caller keeps
    destination blocks pinned until the fresh read completes, and must treat
    any remaining failure as a cache miss rather than accepting stale bytes.
    """
    if not len(keys) == len(buffers) == len(sizes):
        raise ValueError("Mooncake GET vectors must have matching lengths")
    validate_batch_results(results, len(keys))
    expired = [i for i, value in enumerate(results) if value in REFRESHABLE_GET_ERRORS]
    merged = list(results)
    if not expired:
        return merged
    retried = store.batch_get_into_multi_buffers(
        [keys[i] for i in expired],
        [buffers[i] for i in expired],
        [sizes[i] for i in expired],
    )
    validate_batch_results(retried, len(expired))
    for position, i in enumerate(expired):
        merged[i] = retried[position]
    return merged
