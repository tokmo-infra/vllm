# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for the native DSV4.1 renderer using local model assets."""

import asyncio
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.renderers.hf import HfRenderer
from vllm.renderers.params import TokenizeParams
from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer
from vllm.tokenizers.dsv41_native_cache import NativeCache, timed_tokenize_async
from vllm.tokenizers.dsv41_segment_cache_core import SegmentCache


@pytest.fixture(scope="module")
def tokenizers():
    model = Path(os.environ.get("TOKMO_DSV41_MODEL_DIR", "/model"))
    if not (model / "tokenizer.json").is_file():
        pytest.skip("Set TOKMO_DSV41_MODEL_DIR to local DSV4.1 tokenizer assets")
    return (
        DeepseekV41Tokenizer.from_pretrained(model, local_files_only=True),
        DeepseekV41Tokenizer.from_pretrained(model, local_files_only=True),
        model,
    )


@pytest.fixture
def renderer(tokenizers):
    tokenizer, _, _ = tokenizers
    result = object.__new__(HfRenderer)
    result.tokenizer = tokenizer
    result.model_config = SimpleNamespace(is_encoder_decoder=False)
    result._dsv41_native_cache = NativeCache(tokenizer, 0.01, "regression")
    return result


def test_added_boundaries_and_unicode_keep_native_ids(renderer, tokenizers):
    """Added-token boundaries and repeated segments must stay token exact."""
    _, golden, model = tokenizers
    added = json.loads((model / "tokenizer.json").read_text())["added_tokens"]
    params = TokenizeParams(add_special_tokens=False, max_total_tokens=1048576)
    texts = ["before 中文 " + t["content"] + "<｜User｜> after é\u0301" for t in added]
    rng = random.Random(42)
    alphabet = [" aaa ", "\n\t", "中文é\u0301"] + [t["content"] for t in added]
    texts.extend("".join(rng.choices(alphabet, k=15)) for _ in range(160))
    for text in texts:
        expected = golden(text, **params.get_encode_kwargs())["input_ids"]
        for _ in range(2):
            actual = renderer._tokenize_prompt(
                {"prompt": text, "cache_salt": "tenant-a"}, params
            )
            assert actual["prompt_token_ids"] == expected
    assert renderer._dsv41_native_cache.cache.snapshot()["reused_tokens"] > 0


@pytest.mark.parametrize(
    "options",
    [
        {"add_special_tokens": True},
        {"return_token_offsets": True},
        {"truncate_prompt_tokens": 30},
        {"truncate_prompt_tokens": 30, "truncation_side": "left"},
    ],
)
def test_native_fallback_preserves_ids_and_offsets(renderer, tokenizers, options):
    _, golden, _ = tokenizers
    text = "中文 hello " * 200 + "<｜User｜> tail"
    params = TokenizeParams(max_total_tokens=1048576, **options)
    kwargs = params.get_encode_kwargs()
    if options.get("return_token_offsets"):
        kwargs["return_offsets_mapping"] = True
    expected = golden(text, **kwargs)
    actual = renderer._tokenize_prompt({"prompt": text}, params)
    assert actual["prompt_token_ids"] == expected["input_ids"]
    if options.get("return_token_offsets"):
        assert actual["prompt_token_offsets"] == expected["offset_mapping"]


def test_bounded_eviction_and_namespace_isolation(tokenizers):
    tokenizer, golden, model = tokenizers
    cache = SegmentCache(tokenizer, model / "tokenizer.json", 4096)
    for i in range(30):
        text = (str(i) + " hello ") * 100
        assert cache.encode(text)[0] == golden.encode(text, add_special_tokens=False)
        assert cache.snapshot()["cache_bytes"] <= 4096
    assert cache.snapshot().get("evictions", 0) > 0
    cache.reset()
    text = "hello<｜User｜>world"
    assert cache.encode(text, "tenant-a")[1]["reused_tokens"] == 0
    assert cache.encode(text, "tenant-b")[1]["reused_tokens"] == 0
    assert cache.encode(text, "tenant-a")[1]["reused_tokens"] > 0


def test_epoch_invalidates_cache_before_next_request(tokenizers, tmp_path, monkeypatch):
    tokenizer, _, _ = tokenizers
    epoch = tmp_path / "epoch"
    epoch.write_text("one")
    monkeypatch.setenv("VLLM_DSV41_TOKENIZER_CACHE_EPOCH_FILE", str(epoch))
    cache = NativeCache(tokenizer, 0.01, "epoch-regression")
    prompt = {"prompt": "hello<｜User｜>world", "cache_salt": "tenant-a"}
    kwargs = {"add_special_tokens": False}
    cache.try_encode(prompt, kwargs)
    cache.try_encode(prompt, kwargs)
    assert cache.cache.snapshot()["reused_tokens"] > 0
    epoch.write_text("two")
    cache.try_encode(prompt, kwargs)
    assert cache.cache.snapshot()["reused_tokens"] == 0


def test_threaded_and_async_native_tokenization(renderer, tokenizers):
    _, golden, _ = tokenizers
    text = "中文 hello " * 200 + "<｜User｜> tail"
    expected = golden.encode(text, add_special_tokens=False)
    params = TokenizeParams(add_special_tokens=False, max_total_tokens=1048576)
    prompt = {"prompt": text, "cache_salt": "parallel"}
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = pool.map(
            lambda _: renderer._tokenize_prompt(prompt, params), range(32)
        )
        assert all(output["prompt_token_ids"] == expected for output in outputs)

    async def check():
        with ThreadPoolExecutor(max_workers=1) as pool:
            tokenize = timed_tokenize_async(renderer, pool)
            outputs = await asyncio.gather(
                *(tokenize(prompt, params) for _ in range(16))
            )
            assert all(output["prompt_token_ids"] == expected for output in outputs)

    asyncio.run(check())


def test_chat_passes_namespace_before_sync_and_async_encoding(renderer):
    text = "hello<｜User｜>world"
    renderer.render_messages = lambda conv, cp: ([], {"prompt": text})

    async def render_messages_async(conv, cp):
        return [], {"prompt": text}

    async def process_async(prompt, arrival_time, **kwargs):
        return prompt

    renderer.render_messages_async = render_messages_async
    renderer.process_for_engine = lambda prompt, arrival_time, **kwargs: prompt
    renderer.process_for_engine_async = process_async
    params = TokenizeParams(add_special_tokens=False, max_total_tokens=1048576)
    cp = SimpleNamespace(media_io_kwargs={})
    _, prompts = renderer.render_chat(
        [[]], cp, params, prompt_extras={"cache_salt": "sync-salt"}
    )
    assert prompts[0]["cache_salt"] == "sync-salt"

    async def check():
        with ThreadPoolExecutor(max_workers=1) as pool:
            renderer._tokenize_prompt_async = timed_tokenize_async(renderer, pool)
            _, prompts = await renderer.render_chat_async(
                [[]], cp, params, prompt_extras={"cache_salt": "async-salt"}
            )
            assert prompts[0]["cache_salt"] == "async-salt"

    asyncio.run(check())
    namespaces = {key[0] for key in renderer._dsv41_native_cache.cache.entries}
    assert {"sync-salt", "async-salt"} <= namespaces
