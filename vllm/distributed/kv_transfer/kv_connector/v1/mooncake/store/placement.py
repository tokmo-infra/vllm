# SPDX-License-Identifier: Apache-2.0
"""Place immutable prefix groups in proportion to owner SSD capacity.

Placement is advisory: the native master still handles allocation and fallback.
All shards/groups for a prefix choose the same owner. Object keys, lifecycle
group IDs, transfer addresses and per-key result order remain unchanged.
"""

import hashlib


def normalize_placement_segments(value):
    if not isinstance(value, dict):
        raise ValueError("placement_segments must map owner endpoints to weights")
    for endpoint, weight in value.items():
        if not isinstance(endpoint, str) or not endpoint or (
            type(weight) is not int or weight <= 0
        ):
            raise ValueError(
                "Placement endpoints and positive integer weights are required"
            )
    return tuple(sorted(value.items()))


def preferred_owner(key, segments):
    total = sum(weight for _, weight in segments)
    if not total:
        return None
    # Shard dimensions precede the final @chunk_hash in native object keys.
    prefix = key.rsplit("@", 1)[-1]
    position = (
        int.from_bytes(hashlib.sha256(prefix.encode()).digest()[:8], "big") % total
    )
    for owner, weight in segments:
        if position < weight:
            return owner
        position -= weight
    raise AssertionError("Invalid weighted placement")


def put_with_placement(store, keys, buffers, sizes, config, segments, group_ids=None):
    """Partition one sender's PUT while retaining native error semantics."""
    if not len(keys) == len(buffers) == len(sizes):
        raise ValueError("PUT vectors must have matching lengths")
    if group_ids is not None and len(group_ids) != len(keys):
        raise ValueError("PUT group IDs must match the keys")
    if not segments:
        return store.batch_put_from_multi_buffers(keys, buffers, sizes, config)
    batches = {}
    for i, key in enumerate(keys):
        batches.setdefault(preferred_owner(key, segments), []).append(i)
    original_owner = config.preferred_segment
    original_groups = config.group_ids if group_ids is not None else None
    results = [None] * len(keys)
    try:
        for owner, indexes in batches.items():
            config.preferred_segment = owner
            if group_ids is not None:
                config.group_ids = [group_ids[i] for i in indexes]
            values = store.batch_put_from_multi_buffers(
                [keys[i] for i in indexes],
                [buffers[i] for i in indexes],
                [sizes[i] for i in indexes],
                config,
            )
            if len(values) != len(indexes):
                raise ValueError("Native PUT returned a different key count")
            for i, value in zip(indexes, values, strict=True):
                results[i] = value
    finally:
        config.preferred_segment = original_owner
        if group_ids is not None:
            config.group_ids = original_groups
    return results
