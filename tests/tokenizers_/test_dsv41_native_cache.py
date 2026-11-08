# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor

import pytest
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import TokenizersBackend

from vllm.tokenizers.dsv41_native_cache import EVICTIONS, SIZE, NativeCache


@pytest.fixture
def cache_tokenizer():
    vocab = {c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = TokenizersBackend(
        tokenizer_object=backend,
        additional_special_tokens=["<sp>", "<sp-long>"],
    )
    tokenizer.backend_tokenizer.add_tokens(
        [AddedToken("ordinary<sp>shield", special=False, normalized=False)]
    )
    return tokenizer


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain text \u4e2d\u6587 \U0001f642",
        "first<sp>second<sp-long>last",
        "ordinary<sp>shield<sp>unshielded",
        "\x00\n\t<sp><sp>\u00e9",
    ],
)
def test_native_cache_misses_and_hits_keep_exact_token_ids(cache_tokenizer, text):
    cache = NativeCache(cache_tokenizer, gib=1 / 1024, rank="exact")
    expected = cache_tokenizer.backend_tokenizer.encode(
        text, add_special_tokens=False
    ).ids
    for namespace in ("first", "second"):
        prompt = {"prompt": text, "cache_salt": namespace}
        for _ in range(2):
            actual = cache.try_encode(prompt, {"add_special_tokens": False})
            assert actual == expected


def _metric_value(metric, name, rank):
    return next(
        sample.value
        for family in metric.collect()
        for sample in family.samples
        if sample.name == name and sample.labels["api_process"] == rank
    )


def test_native_cache_parallel_evictions_and_quiescent_epoch_reset(
    cache_tokenizer, tmp_path, monkeypatch
):
    epoch = tmp_path / "epoch"
    epoch.write_text("first")
    monkeypatch.setenv("VLLM_DSV41_TOKENIZER_CACHE_EPOCH_FILE", str(epoch))
    rank = "parallel"
    cache = NativeCache(cache_tokenizer, gib=8192 / 1024**3, rank=rank)
    assert cache.try_encode({"prompt": ""}, {"add_special_tokens": False}) == []
    texts = [
        f"request-{i}:" + "unicode \u4e2d\u6587 \U0001f642 " * 16 for i in range(64)
    ]
    expected = [
        cache_tokenizer.backend_tokenizer.encode(text, add_special_tokens=False).ids
        for text in texts
    ]

    def encode(index):
        index %= len(texts)
        return cache.try_encode({"prompt": texts[index]}, {"add_special_tokens": False})

    for generation in ("first", "second"):
        # Each previous pool has drained before changing the cold-cache epoch.
        epoch.write_text(generation)
        before = _metric_value(EVICTIONS, "vllm:tokenizer_cache_evictions_total", rank)
        with ThreadPoolExecutor(4) as pool:
            outputs = list(pool.map(encode, range(256)))
        assert outputs == [expected[i % len(texts)] for i in range(256)]
        state = cache.cache.snapshot()
        assert state["evictions"] > 0
        assert state["cache_bytes"] <= state["cache_max_bytes"] == 8192
        after = _metric_value(EVICTIONS, "vllm:tokenizer_cache_evictions_total", rank)
        assert after - before == state["evictions"]
        assert (
            _metric_value(SIZE, "vllm:tokenizer_cache_bytes", rank)
            == state["cache_bytes"]
        )
