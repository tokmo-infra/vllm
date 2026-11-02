# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact token IDs, namespace isolation and fallback for the opt-in CPU cache."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, trainers
from transformers import TokenizersBackend

from vllm.tokenizers.dsv41_native_cache import NativeCache
from vllm.tokenizers.dsv41_segment_cache_core import SegmentCache


@pytest.fixture
def tokenizer():
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        ["hello world 中文🙂\n\n repeated prefix", "<tool>value</tool>"],
        trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]", "<s>"]),
    )
    backend.add_tokens([AddedToken("prefix<s>suffix", normalized=False)])
    return TokenizersBackend(tokenizer_object=backend, unk_token="[UNK]")


def test_segment_reuse_preserves_ids_and_salt_with_concurrent_requests(tokenizer):
    cache = SegmentCache(tokenizer.backend_tokenizer, max_bytes=8192)
    texts = [
        "hello world<s>中文🙂\n\n repeated prefix",
        "prefix<s>suffix<s>hello world",
        "<s><s>",
        "",
    ]
    expected = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(executor.map(lambda text: cache.encode(text, "a")[0], texts * 8))
    assert actual == expected * 8
    _, counts = cache.encode(texts[0], "a")
    assert counts["reused_tokens"] > 0 and counts["encoded_tokens"] == 0
    _, counts = cache.encode(texts[0], "b")
    assert counts["reused_tokens"] == 0 and counts["encoded_tokens"] > 0
    assert cache.snapshot()["cache_bytes"] <= 8192


def test_native_cache_falls_back_and_resets_on_epoch(tokenizer, tmp_path, monkeypatch):
    epoch = tmp_path / "epoch"
    epoch.write_text("1")
    monkeypatch.setenv("VLLM_DSV41_TOKENIZER_CACHE_EPOCH_FILE", str(epoch))
    cache = NativeCache(tokenizer, 8192 / 1024**3, "test")
    prompt = {"prompt": "hello world<s>中文🙂", "cache_salt": "tenant"}
    kwargs = {"add_special_tokens": False, "truncation": False}
    expected = tokenizer.encode(prompt["prompt"], add_special_tokens=False)
    assert cache.try_encode(prompt, kwargs) == expected
    assert cache.try_encode(prompt, kwargs) == expected
    assert cache.cache.snapshot()["reused_tokens"] > 0
    epoch.write_text("2")
    assert cache.try_encode(prompt, kwargs) == expected
    assert cache.cache.snapshot()["reused_tokens"] == 0
    assert cache.try_encode(prompt, kwargs, want_offsets=True) is None
    assert cache.try_encode(prompt, {**kwargs, "add_special_tokens": True}) is None
    assert (
        cache.try_encode(prompt, {**kwargs, "truncation": True, "max_length": 1})
        is None
    )
    assert (
        cache.try_encode({**prompt, "multi_modal_data": {"image": [1]}}, kwargs) is None
    )


def test_segment_eviction_is_bounded_and_reencoding_is_exact(tokenizer):
    cache = SegmentCache(tokenizer.backend_tokenizer, max_bytes=600)
    for text in ["hello", "world", "hello", "中文🙂", "world"]:
        actual, _ = cache.encode(text)
        assert actual == tokenizer.encode(text, add_special_tokens=False)
        assert cache.snapshot()["cache_bytes"] <= 600
    assert cache.snapshot()["evictions"] > 0
