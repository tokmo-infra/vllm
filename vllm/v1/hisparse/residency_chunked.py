"""Bound HiSparse resolver CTA shared memory without reducing the top-k union.

Keep the complete hot buffer. Resolve the verification rows in per-request
chunks, using the existing CUDA operator, then gather all compacted misses once.
Prior chunks remain MRU; hot_size >= the complete union prevents their eviction.
All row remapping is device-side and uses fixed sizes for CUDA graph replay.
"""
from __future__ import annotations

import torch


def required_shared_bytes(hot_size: int, union_rows: int) -> int:
    return 4 * (2 * (union_rows + 1) + 3 * ((hot_size + 31) // 32) + 7) + 2 * hot_size


def query_chunk_size(hot_size: int, top_k: int, max_union_rows: int, limit: int) -> int:
    max_queries = (max_union_rows + top_k - 1) // top_k
    for q in range(max_queries, 0, -1):
        if required_shared_bytes(hot_size, q * top_k) <= limit:
            return q
    raise ValueError("Even one HiSparse query exceeds the shared-memory limit")


def resolve_residency(*args, max_queries_per_chunk: int, original=None):
    if original is None:
        original = torch.ops._C_cache_ops.hisparse_resolve_residency
    if len(args) != 25:
        raise ValueError("Unexpected HiSparse resolver ABI")
    values = list(args)
    indices = values[3]
    request_states = values[7]
    max_union_rows = values[9]
    request_ids = values[14]
    if request_ids is None or request_states is None:
        raise ValueError("Chunked HiSparse requires explicit request IDs and states")
    n, top_k = indices.shape
    queries = (max_union_rows + top_k - 1) // top_k
    if queries <= max_queries_per_chunk:
        return original(*values)
    if n == 0:
        return
    if n <= max_queries_per_chunk:
        # Small draft batches fit even if every row belongs to one request.
        values[9] = n * top_k
        return original(*values)
    if values[8] < max_union_rows:
        raise ValueError("Hot capacity must cover the complete verification union")
    requests = request_states.numel()
    ids = request_ids.to(torch.long)
    valid = (ids >= 0) & (ids < requests)
    safe_ids = torch.where(valid, ids, requests)
    # Prefix occurrence count also handles non-contiguous request IDs safely.
    onehot = torch.zeros((n, requests + 1), dtype=torch.int32, device=ids.device)
    onehot.scatter_(1, safe_ids[:, None], 1)
    ordinal = onehot.cumsum(0, dtype=torch.int32).gather(1, safe_ids[:, None])[:, 0] - 1
    valid &= ordinal < queries
    table_pad = requests * queries
    keys = torch.where(valid, safe_ids * queries + ordinal, table_pad)
    table = torch.full((table_pad + 1,), n, dtype=torch.long, device=ids.device)
    table.scatter_(0, keys, torch.arange(n, device=ids.device))
    table = table[:-1].view(requests, queries)
    inputs = torch.cat((indices, indices.new_full((1, top_k), -1)), 0)
    # Outputs 19/20/21 are compact plans: scatter their *whole query rows*,
    # not a mask over the original union, to preserve every compacted miss.
    output_positions = (4, 10, 12, 17, 18, 19, 20, 21)
    zeros = {10, 18, 21}
    merged = {}
    for pos in output_positions:
        out = values[pos]
        if out is not None:
            if out.shape[0] != n:
                raise ValueError(f"Unexpected resolver output rows at argument {pos}")
            merged[pos] = out.new_full((n + 1, *out.shape[1:]), 0 if pos in zeros else -1)
    for start in range(0, queries, max_queries_per_chunk):
        count = min(max_queries_per_chunk, queries - start)
        rowmap = table[:, start:start + count].reshape(-1)
        call = list(values)
        call[3] = inputs.index_select(0, rowmap).contiguous()
        call[9] = count * top_k
        call[14] = torch.arange(requests, device=ids.device, dtype=request_ids.dtype).repeat_interleave(count)
        for pos in merged:
            out = values[pos]
            call[pos] = out.new_full((requests * count, *out.shape[1:]), 0 if pos in zeros else -1)
        original(*call)
        for pos, target in merged.items():
            # The only duplicate index is the disposable padding row n.
            target.index_copy_(0, rowmap, call[pos])
    for pos, target in merged.items():
        values[pos].copy_(target[:-1])
