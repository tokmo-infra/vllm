# SPDX-License-Identifier: Apache-2.0
"""Opt-in exact DSV4.1 CPU cache in the native renderer, one budget per API process."""
import asyncio
import functools
import os
from pathlib import Path
import time
from types import SimpleNamespace

from prometheus_client import Counter, Gauge, Histogram
from tokenizers import Tokenizer
from vllm.tokenizers.dsv41_segment_cache_core import SegmentCache

LABELS = ['api_process']
TOKENS = Counter('vllm:tokenizer_cache_tokens', 'Exact segment tokens by source', LABELS + ['source'])
REQUESTS = Counter('vllm:tokenizer_cache_requests', 'Native tokenizer cache requests', LABELS + ['path'])
SIZE = Gauge('vllm:tokenizer_cache_bytes', 'Retained CPU segment cache bytes', LABELS, multiprocess_mode='livesum')
LIMIT = Gauge('vllm:tokenizer_cache_max_bytes', 'CPU segment cache limit', LABELS, multiprocess_mode='livesum')
EVICTIONS = Counter('vllm:tokenizer_cache_evictions', 'Native segment cache evictions', LABELS)
TIMING = Histogram('vllm:frontend_stage_seconds', 'Native frontend stage wall time; stages overlap', LABELS + ['stage'], buckets=(.001,.005,.01,.025,.05,.1,.25,.5,1,2.5,5,10,20,60))
ACTIVE = Gauge('vllm:frontend_active_renders', 'Active native render_chat_async calls', LABELS, multiprocess_mode='livesum')


class NativeCache:
    def __init__(self, tokenizer, gib, rank):
        # The original tokenizer pool can mutate padding/truncation. Keep a private
        # immutable backend for exact segment encoding rather than sharing it.
        backend = Tokenizer.from_str(tokenizer.backend_tokenizer.to_str())
        self.cache = SegmentCache(SimpleNamespace(backend_tokenizer=backend), None, int(gib * 1024**3))
        self.rank = str(rank)
        self.epoch_path = os.environ.get('VLLM_DSV41_TOKENIZER_CACHE_EPOCH_FILE', '')
        self.epoch = self._epoch()
        self.last_evictions = 0
        LIMIT.labels(self.rank).set(self.cache.max_bytes)
        SIZE.labels(self.rank).set(0)

    def _epoch(self):
        if not self.epoch_path:
            return None
        return Path(self.epoch_path).read_text().strip()

    def try_encode(self, prompt, kwargs, want_offsets=False, queued_at=None):
        started = time.monotonic()
        if queued_at is not None:
            TIMING.labels(self.rank, 'tokenize_queue').observe(started - queued_at)
        epoch = self._epoch()
        if epoch != self.epoch:
            self.cache.reset()
            self.last_evictions = 0
            self.epoch = epoch
            SIZE.labels(self.rank).set(0)
        # Keep offsets, multimodal, BOS/EOS and unsupported encoding options on
        # the original tokenizer path. Post-tokenization remains fully native.
        safe = (not want_offsets and not prompt.get('multi_modal_data')
                and not prompt.get('multi_modal_uuids')
                and kwargs.get('add_special_tokens') is False
                and set(kwargs) <= {'add_special_tokens', 'truncation', 'max_length'})
        if not safe:
            REQUESTS.labels(self.rank, 'fallback').inc()
            return None
        ids, counts = self.cache.encode(prompt['prompt'], prompt.get('cache_salt') or '')
        TIMING.labels(self.rank, 'cache_encode').observe(time.monotonic() - started)
        for source in ('reused_tokens', 'encoded_tokens', 'special_tokens'):
            TOKENS.labels(self.rank, source).inc(counts[source])
        state = self.cache.snapshot()
        SIZE.labels(self.rank).set(state['cache_bytes'])
        evicted = state.get('evictions', 0)
        EVICTIONS.labels(self.rank).inc(evicted - self.last_evictions)
        self.last_evictions = evicted
        # If truncation could change the result, use the original implementation
        # (including its truncation_side semantics), not a hand-written slice.
        if kwargs.get('truncation') and len(ids) > kwargs['max_length']:
            REQUESTS.labels(self.rank, 'fallback').inc()
            return None
        REQUESTS.labels(self.rank, 'cached').inc()
        return ids


def make_cache(config, tokenizer):
    gib = float(os.environ.get('VLLM_DSV41_TOKENIZER_CACHE_GIB', '0'))
    if gib <= 0 or tokenizer is None or config.model_config.tokenizer_mode != 'deepseek_v41':
        return None
    return NativeCache(tokenizer, gib, config.parallel_config._api_process_rank)


def timed_tokenize_async(renderer, executor):
    async def tokenize(prompt, params):
        queued_at = time.monotonic()
        call = functools.partial(renderer._tokenize_prompt, prompt, params, _queued_at=queued_at)
        return await asyncio.get_running_loop().run_in_executor(executor, call)
    return tokenize


def track_render(fn):
    @functools.wraps(fn)
    async def wrapped(self, *args, **kwargs):
        cache = self._dsv41_native_cache
        if cache is None:
            return await fn(self, *args, **kwargs)
        started = time.monotonic()
        ACTIVE.labels(cache.rank).inc()
        try:
            return await fn(self, *args, **kwargs)
        finally:
            TIMING.labels(cache.rank, 'render_chat_total').observe(time.monotonic() - started)
            ACTIVE.labels(cache.rank).dec()
    return wrapped
