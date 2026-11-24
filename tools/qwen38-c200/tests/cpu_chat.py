# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise actual Chat response serialization without GPUs or a model."""

import asyncio
import json
from unittest.mock import MagicMock

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.outputs import CompletionOutput, RequestOutput

MESSAGES = [{"role": "user", "content": "Reply or call the demo tool."}]
TOOL = {"type": "function", "function": {"name": "demo", "parameters": {}}}
NAMED = {"type": "function", "function": {"name": "demo"}}


def serving(force):
    chat = OpenAIServingChat.__new__(OpenAIServingChat)
    chat.response_role = "assistant"
    chat.parser_cls = MagicMock()
    chat.enable_auto_tools = True
    chat._include_reasoning_tokens_details = False
    chat.enable_prompt_tokens_details = True
    chat.enable_log_outputs = False
    chat.enable_log_deltas = False
    chat.enable_force_include_usage = force
    chat.request_logger = None
    chat.system_fingerprint = None
    chat.enable_per_request_metrics = False
    return chat


def output(reason, ids):
    return RequestOutput(
        request_id="chatcmpl-test",
        prompt="test",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        num_cached_tokens=1,
        outputs=[CompletionOutput(0, "text", ids, None, None, finish_reason=reason)],
        finished=reason is not None,
    )


async def generate(*rows):
    for row in rows:
        yield row


async def stream(chat, request, *rows):
    events = [
        line.removeprefix("data: ").strip()
        async for line in chat.chat_completion_stream_generator(
            request,
            generate(*rows),
            "chatcmpl-test",
            "test",
            MESSAGES,
            MagicMock(),
            RequestResponseMetadata(request_id="chatcmpl-test"),
        )
    ]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(line) for line in events[:-1]]
    assert chunks and not any("error" in c for c in chunks), chunks
    return chunks


async def main():
    for choice, tools, reason, want in (
        (NAMED, True, "stop", "tool_calls"),
        ("required", True, "stop", "tool_calls"),
        ("auto", True, "stop", "tool_calls"),
        ("auto", True, "length", "length"),
        (NAMED, True, "length", "length"),
        ("auto", False, "stop", "stop"),
        (NAMED, False, "stop", "stop"),
    ):
        chat, parser = serving(False), MagicMock()
        chat.parser_cls.return_value = parser
        chat._make_parser = MagicMock(return_value=parser)
        request = ChatCompletionRequest(
            model="test", messages=MESSAGES, tools=[TOOL], tool_choice=choice
        )
        parser.parse.return_value = (
            None,
            "text",
            [FunctionCall(id="call_1", name="demo", arguments="{}")] if tools else [],
        )
        parser.parse_delta.side_effect = [
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0,
                        id="call_1",
                        type="function",
                        function=DeltaFunctionCall(name="demo", arguments="{}"),
                    )
                ]
            )
            if tools
            else DeltaMessage(content="text"),
            DeltaMessage(),
        ]
        chunks = await stream(chat, request, output(None, [10]), output(reason, []))
        ends = [
            c["finish_reason"]
            for x in chunks
            for c in x["choices"]
            if c.get("finish_reason")
        ]
        assert ends == [want], (choice, ends)
        full = await chat.chat_completion_full_generator(
            request,
            generate(output(reason, [10])),
            "chatcmpl-test",
            "test",
            MESSAGES,
            MagicMock(),
            RequestResponseMetadata(request_id="chatcmpl-test"),
            parser=parser,
        )
        assert full.choices[0].finish_reason == want, (choice, full)
        calls = full.choices[0].message.tool_calls
        assert bool(calls) == tools
        if tools:
            assert calls[0].id == "call_1" and calls[0].function.arguments == "{}"

    for force, options, expected in (
        (True, None, 1),
        (True, {"include_usage": True}, 1),
        (True, {"include_usage": False}, 1),
        (False, {"include_usage": True}, 1),
        (False, None, 0),
        (True, {"include_usage": True, "continuous_usage_stats": True}, 4),
    ):
        chat = serving(force)
        chat.parser_cls = None
        request = ChatCompletionRequest(
            model="test", messages=MESSAGES, stream=True, stream_options=options
        )
        chunks = await stream(chat, request, output(None, [10]), output("stop", [11]))
        usage = [chunk for chunk in chunks if chunk.get("usage") is not None]
        assert len(usage) == expected, (force, options, usage)
        if expected:
            assert usage[-1] is chunks[-1] and usage[-1]["choices"] == []
            final = usage[-1]["usage"]
            assert final["prompt_tokens"] == 3
            assert final["completion_tokens"] == 2 and final["total_tokens"] == 5
            assert final["prompt_tokens_details"]["cached_tokens"] == 1
    print("PASS: 7 streaming/non-streaming finish cases and 6 stream usage cases")


if __name__ == "__main__":
    asyncio.run(main())
