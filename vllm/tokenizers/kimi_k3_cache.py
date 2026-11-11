# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact CPU reuse of Kimi's native XTML text-piece tokenization."""

import hashlib
import inspect
import math
import os
import sys
import threading
import time
from array import array
from collections import OrderedDict
from pathlib import Path
from types import MethodType

from prometheus_client import Counter, Gauge

_LABELS = ["api_process"]
_TOKENS = Counter(
    "vllm:tokenizer_cache_tokens",
    "Kimi text-piece tokens by source",
    _LABELS + ["source"],
)
_REQUESTS = Counter(
    "vllm:tokenizer_cache_requests", "Kimi text-piece calls by path", _LABELS + ["path"]
)
_BYTES = Gauge(
    "vllm:tokenizer_cache_bytes",
    "Charged CPU cache bytes",
    _LABELS,
    multiprocess_mode="livesum",
)
_LIMIT = Gauge(
    "vllm:tokenizer_cache_max_bytes",
    "CPU cache byte budget",
    _LABELS,
    multiprocess_mode="livesum",
)
_EVICTIONS = Counter("vllm:tokenizer_cache_evictions", "Kimi cache evictions", _LABELS)
_CACHE = None
_INIT_LOCK = threading.Lock()


class TextPieceCache:
    """Private L1 with optional node-shared L2; misses use native encoding."""

    def __init__(self, max_bytes, api_process, epoch_path="", shared=None):
        self.max_bytes = max_bytes
        self.rank = str(api_process)
        self.epoch_path = Path(epoch_path) if epoch_path else None
        self.epoch = self._epoch()
        self.next_epoch_check = 0.0
        self.entries = OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()
        self.shared = shared
        _LIMIT.labels(self.rank).set(max_bytes)
        _BYTES.labels(self.rank).set(0)

    def _epoch(self):
        return self.epoch_path.read_text().strip() if self.epoch_path else ""

    def encode(self, namespace, original, text, allow_special, special_ids):
        key = (namespace, allow_special, text)
        hash(key)
        with self.lock:
            now = time.monotonic()
            if now >= self.next_epoch_check:
                epoch = self._epoch()
                if epoch != self.epoch:
                    self.entries.clear()
                    self.bytes = 0
                    self.epoch = epoch
                    _BYTES.labels(self.rank).set(0)
                self.next_epoch_check = now + 1.0
            epoch = self.epoch
            item = self.entries.get(key)
            if item is not None:
                self.entries.move_to_end(key)
        hit = item is not None
        if item is not None:
            ids, special_count, _ = item
        else:
            shared_item = (
                self.shared.get(key, epoch) if self.shared is not None else None
            )
            if shared_item is not None:
                packed, special_count = shared_item
                ids = array("I")
                ids.frombytes(packed)
                hit = True
            else:
                # Keep native chunking and ordinary/control-token BPE unchanged.
                ids = array("I", original(text, allow_special_tokens=allow_special))
                special_count = sum(token in special_ids for token in ids)
                if self.shared is not None:
                    self.shared.put(key, ids.tobytes(), special_count, epoch)
            size = sys.getsizeof(text) + sys.getsizeof(ids) + sys.getsizeof(key) + 256
            if size <= self.max_bytes:
                with self.lock:
                    if epoch == self.epoch and key not in self.entries:
                        while self.entries and self.bytes + size > self.max_bytes:
                            _, (_, _, old_size) = self.entries.popitem(last=False)
                            self.bytes -= old_size
                            _EVICTIONS.labels(self.rank).inc()
                        self.entries[key] = (ids, special_count, size)
                        self.bytes += size
                        _BYTES.labels(self.rank).set(self.bytes)
        ordinary = len(ids) - special_count
        source = "reused_tokens" if hit else "encoded_tokens"
        _TOKENS.labels(self.rank, source).inc(ordinary)
        _TOKENS.labels(self.rank, "special_tokens").inc(special_count)
        _REQUESTS.labels(self.rank, "hit" if hit else "miss").inc()
        return list(ids)


def _shared_namespace(tokenizer, original):
    """Identify encoder data and code, not a process-local object address."""
    model = tokenizer.model
    try:
        ranks = model._mergeable_ranks
        specials = model._special_tokens
        pattern = model._pat_str
        source = inspect.getsourcefile(original)
        if source is None:
            raise ValueError("native tokenizer source is unavailable")
        digest = hashlib.sha256(Path(source).read_bytes())
        digest.update(pattern.encode("utf-8"))
        for text, token_id in sorted(ranks.items(), key=lambda item: item[1]):
            digest.update(token_id.to_bytes(8, "little"))
            digest.update(len(text).to_bytes(8, "little"))
            digest.update(text)
        for text, token_id in sorted(specials.items()):
            encoded = text.encode("utf-8")
            digest.update(token_id.to_bytes(8, "little"))
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    except (AttributeError, TypeError, ValueError, OSError) as exc:
        raise RuntimeError(
            "Shared Kimi cache requires a fingerprintable native tiktoken backend"
        ) from exc
    return digest.hexdigest()


def attach_cache(config, tokenizer):
    """Cache the tokenizer's native pieces, retaining ordinary/control separation."""
    global _CACHE
    gib = float(os.environ.get("KIMI31_TOKENIZER_CACHE_GIB", "0"))
    if not math.isfinite(gib) or gib < 0:
        raise ValueError("KIMI31_TOKENIZER_CACHE_GIB must be finite and nonnegative")
    if gib <= 0 or tokenizer is None:
        return
    if getattr(tokenizer, "_kimi31_piece_cache_attached", False):
        return
    original = getattr(tokenizer, "_encode_text_piece", None)
    if not callable(original):
        raise RuntimeError("Kimi text-piece cache requires _encode_text_piece")
    rank = config.parallel_config._api_process_rank
    shared_path = os.environ.get("KIMI31_TOKENIZER_CACHE_SHARED_PATH", "")
    l1_mib = float(os.environ.get("KIMI31_TOKENIZER_CACHE_L1_MIB", "64"))
    if shared_path and (not math.isfinite(l1_mib) or l1_mib < 0):
        raise ValueError("KIMI31_TOKENIZER_CACHE_L1_MIB must be finite and nonnegative")
    limit = int(l1_mib * 1024**2) if shared_path else int(gib * 1024**3)
    with _INIT_LOCK:
        if _CACHE is None:
            epoch_path = os.environ.get("KIMI31_TOKENIZER_CACHE_EPOCH_FILE", "")
            shared = None
            if shared_path:
                from .kimi_k3_cache_shared import SharedTextPieceCache

                shared = SharedTextPieceCache(shared_path, int(gib * 1024**3))
            _CACHE = TextPieceCache(limit, rank, epoch_path, shared)
        elif _CACHE.max_bytes != limit or bool(_CACHE.shared) != bool(shared_path):
            raise RuntimeError("Inconsistent Kimi cache configuration")
    cache = _CACHE
    namespace = (
        _shared_namespace(tokenizer, original)
        if shared_path
        else (type(tokenizer).__module__, id(tokenizer.model))
    )
    special_ids = frozenset(tokenizer.all_special_ids)

    def encode_piece(self, text, allow_special_tokens=True):
        if type(text) is not str or type(allow_special_tokens) is not bool:
            _REQUESTS.labels(cache.rank, "fallback").inc()
            return original(text, allow_special_tokens=allow_special_tokens)
        return cache.encode(
            namespace, original, text, allow_special_tokens, special_ids
        )

    tokenizer._encode_text_piece = MethodType(encode_piece, tokenizer)
    tokenizer._kimi31_piece_cache_attached = True
    from vllm.logger import init_logger

    if shared_path:
        init_logger(__name__).info(
            "Kimi text-piece cache: %.1f GiB shared L2, %.1f MiB L1/API %s",
            gib,
            l1_mib,
            rank,
        )
    else:
        init_logger(__name__).info(
            "Kimi text-piece cache enabled: %.1f GiB/API process %s", gib, rank
        )
