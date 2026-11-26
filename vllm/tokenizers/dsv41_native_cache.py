# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in exact DSV4.1 CPU cache in the native renderer, one budget per API process."""

import asyncio
import functools
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from prometheus_client import Counter, Gauge, Histogram
from tokenizers import Tokenizer
from tokenizers.processors import ByteLevel

from vllm import envs
from vllm.tokenizers.dsv41_segment_cache_core import SegmentCache

_metrics_lock = threading.Lock()


@functools.cache
def _metrics():
    # The loggers unregister vLLM collectors during startup.
    labels = ["api_process"]
    tokens = Counter(
        "vllm:tokenizer_cache_tokens",
        "Exact segment tokens by source",
        labels + ["source"],
    )
    requests = Counter(
        "vllm:tokenizer_cache_requests",
        "Native tokenizer cache requests",
        labels + ["path"],
    )
    size = Gauge(
        "vllm:tokenizer_cache_bytes",
        "Retained CPU segment cache bytes",
        labels,
        multiprocess_mode="livesum",
    )
    limit = Gauge(
        "vllm:tokenizer_cache_max_bytes",
        "CPU segment cache limit",
        labels,
        multiprocess_mode="livesum",
    )
    evictions = Counter(
        "vllm:tokenizer_cache_evictions", "Native segment cache evictions", labels
    )
    timing = Histogram(
        "vllm:frontend_stage_seconds",
        "Native frontend stage wall time; stages overlap",
        labels + ["stage"],
        buckets=(
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1,
            2.5,
            5,
            10,
            20,
            60,
        ),
    )
    active = Gauge(
        "vllm:frontend_active_renders",
        "Active native render_chat_async calls",
        labels,
        multiprocess_mode="livesum",
    )
    return SimpleNamespace(
        active=active,
        tokens=tokens,
        requests=requests,
        size=size,
        limit=limit,
        evictions=evictions,
        timing=timing,
    )


class NativeCache:
    def __init__(self, tokenizer, gib, rank):
        # The original tokenizer pool can mutate padding/truncation. Keep a private
        # immutable backend for exact segment encoding rather than sharing it.
        backend = Tokenizer.from_str(tokenizer.backend_tokenizer.to_str())
        # ByteLevel post-processing only adjusts offsets. Its single-input IDs
        # are identical with or without add_special_tokens (as with no processor).
        processor = backend.post_processor
        self._special_tokens_preserve_ids = processor is None or isinstance(
            processor, ByteLevel
        )
        self.cache = SegmentCache(
            SimpleNamespace(backend_tokenizer=backend), None, int(gib * 1024**3)
        )
        self.rank = str(rank)
        self.epoch_path = envs.VLLM_DSV41_TOKENIZER_CACHE_EPOCH_FILE
        self.epoch = self._epoch()
        self.last_evictions = 0
        self._metrics_lock = threading.Lock()
        self.metrics = None

    def _get_metrics(self):
        if self.metrics is None:
            # Multiple renderer threads may enter concurrently on the first request.
            with _metrics_lock:
                if self.metrics is None:
                    self.metrics = _metrics()
                    self.metrics.limit.labels(self.rank).set(self.cache.max_bytes)
        return self.metrics

    def track_render(self, render):
        @functools.wraps(render)
        async def wrapped(*args, **kwargs):
            metrics = self._get_metrics()
            started = time.monotonic()
            active = metrics.active.labels(self.rank)
            active.inc()
            try:
                return await render(*args, **kwargs)
            finally:
                metrics.timing.labels(self.rank, "render_chat_total").observe(
                    time.monotonic() - started
                )
                active.dec()

        return wrapped

    def _epoch(self):
        if not self.epoch_path:
            return None
        return Path(self.epoch_path).read_text().strip()

    def try_encode(self, prompt, kwargs, want_offsets=False):
        metrics = self._get_metrics()
        started = time.monotonic()
        epoch = self._epoch()
        with self._metrics_lock:
            if epoch != self.epoch:
                self.cache.reset()
                self.last_evictions = 0
                self.epoch = epoch
                metrics.size.labels(self.rank).set(0)
        add_special_tokens = kwargs.get("add_special_tokens")
        safe = (
            not want_offsets
            and not prompt.get("multi_modal_data")
            and not prompt.get("multi_modal_uuids")
            and (
                add_special_tokens is False
                or (add_special_tokens is True and self._special_tokens_preserve_ids)
            )
            and set(kwargs) <= {"add_special_tokens", "truncation", "max_length"}
        )
        if not safe:
            metrics.requests.labels(self.rank, "fallback").inc()
            return None
        ids, counts = self.cache.encode(
            prompt["prompt"], prompt.get("cache_salt") or ""
        )
        metrics.timing.labels(self.rank, "cache_encode").observe(
            time.monotonic() - started
        )
        for source in ("reused_tokens", "encoded_tokens", "special_tokens"):
            metrics.tokens.labels(self.rank, source).inc(counts[source])
        with self._metrics_lock:
            state = self.cache.snapshot()
            metrics.size.labels(self.rank).set(state["cache_bytes"])
            evicted = state.get("evictions", 0)
            metrics.evictions.labels(self.rank).inc(evicted - self.last_evictions)
            self.last_evictions = evicted
        # If truncation could change the result, use the original implementation
        # (including its truncation_side semantics), not a hand-written slice.
        if kwargs.get("truncation") and len(ids) > kwargs["max_length"]:
            metrics.requests.labels(self.rank, "fallback").inc()
            return None
        metrics.requests.labels(self.rank, "cached").inc()
        return ids


def timed_tokenize_async(renderer, executor):
    def run(queued_at, prompt, params):
        cache = renderer._dsv41_native_cache
        cache._get_metrics().timing.labels(cache.rank, "tokenize_queue").observe(
            time.monotonic() - queued_at
        )
        return renderer._tokenize_prompt(prompt, params)

    async def tokenize(prompt, params):
        return await asyncio.get_running_loop().run_in_executor(
            executor, run, time.monotonic(), prompt, params
        )

    return tokenize
