# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-query expired read leases without accepting an expired transfer."""

from typing import Any

LEASE_EXPIRED = -707


def retry_expired_get_once(
    store: Any,
    keys: list[str],
    buffers: list[list[int]],
    sizes: list[list[int]],
    results: list[int],
) -> list[int]:
    """Issue one fresh full GET only for keys whose native lease expired.

    The full GET API queries current replicas again. A second failure remains
    a failure; neither lease validation nor the caller's KV-load policy changes.
    The caller retains the destination buffers until this function returns.
    """
    if not len(keys) == len(buffers) == len(sizes) == len(results):
        raise ValueError("Mooncake GET vectors must have matching lengths")
    expired = [i for i, value in enumerate(results) if value == LEASE_EXPIRED]
    merged = list(results)
    if not expired:
        return merged
    retried = store.batch_get_into_multi_buffers(
        [keys[i] for i in expired],
        [buffers[i] for i in expired],
        [sizes[i] for i in expired],
    )
    if len(retried) != len(expired):
        raise ValueError("Mooncake lease refresh returned a different key count")
    for position, i in enumerate(expired):
        merged[i] = retried[position]
    return merged
