# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer


@pytest.fixture(scope="module")
def chat_server_with_force_include_usage(request):
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "128",
        "--enforce-eager",
        "--max-num-seqs",
        "4",
        "--enable-force-include-usage",
        "--port",
        "55857",
        "--gpu-memory-utilization",
        "0.2",
    ]

    with RemoteOpenAIServer("Qwen/Qwen3-0.6B", args, auto_port=False) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def chat_client_with_force_include_usage(chat_server_with_force_include_usage):
    async with chat_server_with_force_include_usage.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_chat_with_enable_force_include_usage(
    chat_client_with_force_include_usage: openai.AsyncOpenAI,
):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    stream = await chat_client_with_force_include_usage.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=messages,
        max_completion_tokens=10,
        extra_body=dict(min_tokens=10),
        temperature=0.0,
        stream=True,
    )
    usage = []
    async for chunk in stream:
        if chunk.usage is not None:
            assert not chunk.choices
            usage.append(chunk.usage)
    assert len(usage) == 1
    assert usage[0].completion_tokens == 10
    assert usage[0].total_tokens == usage[0].prompt_tokens + usage[0].completion_tokens
