# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact CPU reuse of Kimi's native XTML text-piece tokenization."""

import hashlib
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
_EPOCH = Gauge(
    "vllm:tokenizer_cache_epoch",
    "48-bit SHA256 of the active cache epoch",
    _LABELS,
    multiprocess_mode="livemostrecent",
)
_CACHE = None
_INIT_LOCK = threading.Lock()


def _bind_api_threads(api_process):
    if os.environ.get("KIMI31_API_NUMA_BIND", "0") != "1":
        return
    import pynvml

    pynvml.nvmlInit()
    try:
        device = int(api_process) % pynvml.nvmlDeviceGetCount()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        # This opt-in path runs on Linux with a validated NVML CPU topology.
        masks = pynvml.nvmlDeviceGetCpuAffinity(
            handle,
            (os.cpu_count() + 63) // 64,  # type: ignore[operator]
        )
        cpus = {
            word * 64 + bit
            for word, mask in enumerate(masks)
            for bit in range(64)
            if mask & (1 << bit)
        } & os.sched_getaffinity(0)  # type: ignore[attr-defined]
        if not cpus:
            raise RuntimeError("No GPU-local CPUs available for Kimi API process")
        for task in Path("/proc/self/task").iterdir():
            try:
                os.sched_setaffinity(int(task.name), cpus)  # type: ignore[attr-defined]
                if os.sched_getaffinity(int(task.name)) != cpus:  # type: ignore[attr-defined]
                    raise RuntimeError("Kimi API thread affinity verification failed")
            except ProcessLookupError:
                continue
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "Kimi API %s GPU-local CPU affinity applied: device=%d cpus=%s",
            api_process,
            device,
            sorted(cpus),
        )
    finally:
        pynvml.nvmlShutdown()


class TextPieceCache:
    """Bounded process-private LRU; original encoding runs on misses."""

    def __init__(self, max_bytes, api_process, epoch_path=""):
        self.max_bytes = max_bytes
        self.rank = str(api_process)
        self.epoch_path = Path(epoch_path) if epoch_path else None
        self.epoch = self._epoch()
        self.next_epoch_check = 0.0
        self.entries = OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        _LIMIT.labels(self.rank).set(max_bytes)
        _BYTES.labels(self.rank).set(0)
        self._record_epoch()
        if self.epoch_path:
            threading.Thread(
                target=self._watch_epoch, name="kimi-token-cache-epoch", daemon=True
            ).start()

    def _epoch(self):
        return self.epoch_path.read_text().strip() if self.epoch_path else None

    def _record_epoch(self):
        value = int(hashlib.sha256((self.epoch or "").encode()).hexdigest()[:12], 16)
        _EPOCH.labels(self.rank).set(value)

    def _refresh_epoch(self):
        epoch = self._epoch()
        if epoch != self.epoch:
            self.entries.clear()
            self.bytes = 0
            self.epoch = epoch
            _BYTES.labels(self.rank).set(0)
            self._record_epoch()

    def _watch_epoch(self):
        # Idle APIs must also clear and acknowledge the epoch before a cold run.
        while not self.stop.wait(1):
            try:
                with self.lock:
                    self._refresh_epoch()
            except OSError:
                from vllm.logger import init_logger

                init_logger(__name__).exception("Cannot read tokenizer cache epoch")

    def close(self):
        self.stop.set()

    def encode(self, namespace, original, text, allow_special, special_ids):
        key = (namespace, allow_special, text)
        hash(key)
        with self.lock:
            now = time.monotonic()
            if now >= self.next_epoch_check:
                self._refresh_epoch()
                self.next_epoch_check = now + 1.0
            epoch = self.epoch
            item = self.entries.get(key)
            if item is not None:
                self.entries.move_to_end(key)
        hit = item is not None
        if item is not None:
            ids, special_count, _ = item
        else:
            # Preserve Kimi's own chunking, special-token policy and BPE exactly.
            ids = array("I", original(text, allow_special_tokens=allow_special))
            special_count = sum(token in special_ids for token in ids)
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


def attach_cache(config, tokenizer):
    """Cache the tokenizer's native pieces, retaining ordinary/control separation."""
    global _CACHE
    gib = float(os.environ.get("KIMI31_TOKENIZER_CACHE_GIB", "0"))
    if gib <= 0 or tokenizer is None:
        return
    if getattr(tokenizer, "_kimi31_piece_cache_attached", False):
        return
    original = getattr(tokenizer, "_encode_text_piece", None)
    if not callable(original):
        raise RuntimeError("Kimi text-piece cache requires _encode_text_piece")
    rank = config.parallel_config._api_process_rank
    with _INIT_LOCK:
        if _CACHE is None:
            _bind_api_threads(rank)
            epoch_path = os.environ.get("KIMI31_TOKENIZER_CACHE_EPOCH_FILE", "")
            _CACHE = TextPieceCache(int(gib * 1024**3), rank, epoch_path)
        elif _CACHE.max_bytes != int(gib * 1024**3):
            raise RuntimeError("Inconsistent process-private Kimi cache budget")
    cache = _CACHE
    # Isolate independently constructed tokenizer backends within one RAM budget.
    namespace = (type(tokenizer).__module__, id(tokenizer.model))
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

    init_logger(__name__).info(
        "Kimi text-piece cache enabled: %.1f GiB/API process %s", gib, rank
    )
