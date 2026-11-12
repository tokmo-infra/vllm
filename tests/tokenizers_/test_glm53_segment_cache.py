# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, trainers
from transformers import TokenizersBackend

from vllm.tokenizers.glm53_native_cache import NativeCache
from vllm.tokenizers.glm53_segment_cache_core import SegmentCache

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def tokenizer():
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        ["中文工具结果 alpha beta gamma\n" * 8],
        trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]"]),
    )
    backend.add_special_tokens(
        [
            AddedToken(text, special=True, normalized=False)
            for text in ("<|user|>", "<|assistant|>")
        ]
    )
    return TokenizersBackend(tokenizer_object=backend)


def test_cold_and_cached_token_ids_equal_native_for_unicode_and_boundaries(tokenizer):
    cache = SegmentCache(tokenizer, None, max_bytes=1 << 20)
    for text in (
        "<|user|>中文工具结果\n<|assistant|>alpha beta",
        "<|user|><|assistant|>",
        "alpha <|user|>beta<|assistant|>gamma",
        "",
    ):
        expected = tokenizer.encode(text, add_special_tokens=False)
        assert cache.encode(text, "one")[0] == expected
        assert cache.encode(text, "one")[0] == expected
        assert cache.encode(text, "two")[0] == expected


def test_cache_salt_and_epoch_do_not_reuse_other_namespaces(tokenizer):
    cache = SegmentCache(tokenizer, None, max_bytes=1 << 20)
    text = "<|user|>alpha beta<|assistant|>gamma"
    assert cache.encode(text, "one")[1]["reused_tokens"] == 0
    assert cache.encode(text, "one")[1]["reused_tokens"] > 0
    assert cache.encode(text, "two")[1]["reused_tokens"] == 0
    cache.reset(generation="next")
    assert cache.encode(text, "one")[1]["reused_tokens"] == 0


def test_lru_never_exceeds_its_byte_budget(tokenizer):
    cache = SegmentCache(tokenizer, None, max_bytes=1024)
    for index in range(16):
        cache.encode(f"<|user|>alpha beta {index}<|assistant|>")
        assert cache.snapshot()["cache_bytes"] <= 1024
    assert cache.snapshot()["evictions"] > 0


def test_api_l1_and_shared_store_reuse_exact_ids_then_reset(
    tokenizer, tmp_path, monkeypatch
):
    epoch = tmp_path / "epoch"
    epoch.write_text("first")
    monkeypatch.setenv("VLLM_GLM53_TOKENIZER_SHARED_PATH", str(tmp_path / "shared.db"))
    monkeypatch.setenv("VLLM_GLM53_TOKENIZER_SHARED_GIB", "0.01")
    monkeypatch.setenv("VLLM_GLM53_TOKENIZER_LOCAL_L1_MIB", "1")
    monkeypatch.setenv("VLLM_GLM53_TOKENIZER_CACHE_EPOCH_FILE", str(epoch))
    first = NativeCache(tokenizer, 0.01, "first")
    second = NativeCache(tokenizer, 0.01, "second")
    prompt = {"prompt": "<|user|>中文工具结果<|assistant|>alpha beta"}
    kwargs = {"add_special_tokens": False}
    expected = tokenizer.encode(prompt["prompt"], **kwargs)
    assert first.try_encode(prompt, kwargs) == expected
    assert second.try_encode(prompt, kwargs) == expected
    assert second.cache.snapshot()["shared_hits"] > 0
    assert second.cache.max_bytes == 1 << 20
    epoch.write_text("next")
    assert second.try_encode(prompt, kwargs) == expected
    assert second.epoch == "next"
    assert second.cache.snapshot()["encoded_tokens"] > 0


@pytest.mark.parametrize(
    "kwargs,offsets",
    [
        ({"add_special_tokens": True}, False),
        ({"add_special_tokens": False}, True),
        ({"add_special_tokens": False, "truncation": True, "max_length": 1}, False),
    ],
)
def test_unsupported_encoding_keeps_original_tokenizer_path(tokenizer, kwargs, offsets):
    cache = NativeCache(tokenizer, 0.01, "fallback")
    assert (
        cache.try_encode(
            {"prompt": "<|user|>中文 alpha beta<|assistant|>"}, kwargs, offsets
        )
        is None
    )
