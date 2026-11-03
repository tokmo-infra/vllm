"""CPU-only lookup ordering for staged Mamba boundary existence checks."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LookupCandidate:
    """One logical (group, hash) candidate and all namespaces needed to load it."""

    group_id: int
    block_hash: bytes
    namespace_keys: tuple[str, ...]
    boundary_tokens: int | None = None


def staged_lookup_exists(
    candidates: Sequence[LookupCandidate],
    *,
    token_len: int,
    mamba_group_ids: Collection[int] | None,
    batch_is_exist: Callable[[Sequence[str]], Sequence[int]],
    find_hit_length: Callable[[set[tuple[int, bytes]], int], int],
    boundaries_per_batch: int = 8,
) -> set[tuple[int, bytes]]:
    """Return loadable candidates, lazily probing older Mamba boundaries.

    Non-Mamba keys are checked together first. Mamba keys are checked in
    descending token-boundary batches; all groups and TP namespaces for a
    boundary stay together. After each batch the coordinator callback evaluates
    the accumulated exists set at the original token length. Once its hit
    reaches this batch's oldest boundary, still older checkpoints cannot
    extend the hit and are left untouched.

    Unknown Mamba groups/boundaries fall back to one all-candidate lookup.
    Namespace keys are deduplicated for Store calls, but every candidate still
    requires all its namespace keys to exist.
    """
    if boundaries_per_batch <= 0:
        raise ValueError("boundaries_per_batch must be positive")
    if not candidates:
        return set()

    mamba_ids = set(mamba_group_ids or ())
    mamba = [candidate for candidate in candidates if candidate.group_id in mamba_ids]
    staged = bool(mamba_ids and mamba) and all(
        candidate.boundary_tokens is not None
        and 0 < candidate.boundary_tokens <= token_len
        for candidate in mamba
    )

    exists: set[tuple[int, bytes]] = set()
    checked_keys: dict[str, bool] = {}

    def check(batch: Sequence[LookupCandidate]) -> None:
        keys = list(
            dict.fromkeys(
                key
                for candidate in batch
                for key in candidate.namespace_keys
                if key not in checked_keys
            )
        )
        if keys:
            values = batch_is_exist(keys)
            if len(values) != len(keys):
                raise ValueError("batch_is_exist returned the wrong number of results")
            checked_keys.update(
                (key, value == 1) for key, value in zip(keys, values, strict=True)
            )

        for candidate in batch:
            if candidate.namespace_keys and all(checked_keys[key] for key in candidate.namespace_keys):
                exists.add((candidate.group_id, bytes(candidate.block_hash)))

    if not staged:
        check(candidates)
        return exists

    non_mamba = [candidate for candidate in candidates if candidate.group_id not in mamba_ids]
    check(non_mamba)

    by_boundary: dict[int, list[LookupCandidate]] = {}
    for candidate in mamba:
        # The staged predicate above guarantees a concrete boundary.
        boundary = candidate.boundary_tokens
        assert boundary is not None
        by_boundary.setdefault(boundary, []).append(candidate)
    boundaries = sorted(by_boundary, reverse=True)
    for offset in range(0, len(boundaries), boundaries_per_batch):
        batch_boundaries = boundaries[offset : offset + boundaries_per_batch]
        batch = [candidate for boundary in batch_boundaries for candidate in by_boundary[boundary]]
        check(batch)
        hit_length = find_hit_length(exists, token_len)
        if hit_length >= min(batch_boundaries):
            break

    return exists
