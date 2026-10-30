# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact native token IDs, isolation, fallbacks and bounded CPU segment reuse."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import TokenizersBackend

from vllm.tokenizers.glm53_native_cache import NativeCache
from vllm.tokenizers.glm53_segment_cache_core import SegmentCache


@pytest.fixture
def tokenizer():
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["hello world", "代码\nprint('你好')", " spaces  \t end", "quoted literal"],
        trainers.BpeTrainer(
            vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<unk>", "<u>", "<a>", "<e>"],
        ),
    )
    backend.add_tokens([AddedToken("quoted<u>literal", normalized=False)])
    return TokenizersBackend(tokenizer_object=backend)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello world",
        "<u>hello<a>world<e>",
        "<u><a><e>",
        "<u> 代码\nprint('你好')<e>",
        "quoted<u>literal<a>end",
        "<u> spaces  \t end<e>",
        "<u>hello<e><u>world<a>",
    ],
)
def test_cached_segments_preserve_full_native_ids(tokenizer, text):
    cache = SegmentCache(tokenizer, None, 64 * 1024)
    expected = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert cache.encode(text)[0] == expected
    assert cache.encode(text)[0] == expected


def test_namespaces_and_appended_turns_reuse_only_their_history(tokenizer):
    cache = SegmentCache(tokenizer, None, 64 * 1024)
    history = "<u>hello world<e>"
    cache.encode(history, "session-a")
    extended = history + "<a>new answer<e>"
    ids, warm = cache.encode(extended, "session-a")
    assert ids == tokenizer(extended, add_special_tokens=False)["input_ids"]
    assert warm["reused_tokens"] > 0
    assert cache.encode(history, "session-b")[1]["reused_tokens"] == 0


def test_concurrent_encoding_and_eviction_remain_exact_and_bounded(tokenizer):
    cache = SegmentCache(tokenizer, None, 2048)
    texts = [f"<u>hello {i} world<e>" for i in range(40)]
    expected = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda text: cache.encode(text)[0], texts))
    assert actual == expected
    state = cache.snapshot()
    assert state["cache_bytes"] <= state["cache_max_bytes"]
    assert state["evictions"] > 0


@pytest.mark.parametrize(
    "kwargs,offsets",
    [
        ({"add_special_tokens": True}, False),
        ({"add_special_tokens": False, "padding": True}, False),
        ({"add_special_tokens": False, "truncation": True, "max_length": 1}, False),
        ({"add_special_tokens": False}, True),
    ],
)
def test_unsupported_options_return_to_native_encoding(tokenizer, kwargs, offsets):
    cache = NativeCache(tokenizer, 0.001, "fallback-test")
    assert cache.try_encode({"prompt": "<u>hello world<e>"}, kwargs, offsets) is None


def test_epoch_discards_retained_history_before_next_encode(
    tokenizer, monkeypatch, tmp_path
):
    epoch = tmp_path / "epoch"
    epoch.write_text("first")
    monkeypatch.setenv("VLLM_GLM53_TOKENIZER_CACHE_EPOCH_FILE", str(epoch))
    cache = NativeCache(tokenizer, 0.001, "epoch-test")
    prompt = {"prompt": "<u>hello world<e>", "cache_salt": "trial-a"}
    params = {"add_special_tokens": False}
    expected = tokenizer(prompt["prompt"], **params)["input_ids"]
    assert cache.try_encode(prompt, params) == expected
    assert cache.try_encode(prompt, params) == expected
    assert cache.cache.snapshot()["reused_tokens"] > 0
    epoch.write_text("second")
    assert cache.try_encode(prompt, params) == expected
    state = cache.cache.snapshot()
    assert state["reused_tokens"] == 0
    assert state["encoded_requests"] == 1
