# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import deepcopy

import pytest
from pydantic import ValidationError

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError


@pytest.mark.parametrize(
    "ref,extra,detail",
    [
        ("img", {}, "auto"),
        ({"url": "img"}, {}, "auto"),
        ({"url": "img", "detail": "low"}, {}, "low"),
        ({"url": "img"}, {"detail": "high"}, "high"),
        ({"url": "img", "detail": "low"}, {"detail": "high"}, None),
        ({"url": "img", "unknown_option": True}, {}, None),
    ],
)
def test_responses_image_options(ref, extra, detail):
    part = {"type": "input_image", "image_url": ref, **extra}
    body = {"model": "test", "input": [{"role": "user", "content": [part]}]}
    original_input = deepcopy(body["input"])
    if detail is None:
        with pytest.raises(ValidationError):
            ResponsesRequest.model_validate(body)
    else:
        r = ResponsesRequest.model_validate(body).model_dump()["input"][0]["content"][0]
        assert (r["image_url"], r["detail"]) == ("img", detail)
    assert body["input"] == original_input


@pytest.mark.parametrize("producer", [True, False])
def test_messages_private_prefill_stop(producer):
    rows = [{"role": "user", "content": "hello"}]
    kv = {"do_remote_decode": producer, "transfer_id": "test"}
    msg = AnthropicMessagesRequest(
        model="test",
        max_tokens=1,
        messages=rows,
        stop_sequences=["STOP"],
        kv_transfer_params=kv,
    )
    req = AnthropicServingMessages._build_base_request(msg, rows)
    assert req.ignore_eos == producer
    assert req.stop == ([] if producer else ["STOP"])
    assert req.kv_transfer_params == kv


@pytest.mark.parametrize("producer", [True, False])
def test_responses_private_prefill_stop(producer):
    request = ResponsesRequest(
        input="hello",
        max_output_tokens=1,
        stop=["STOP"],
        stop_token_ids=[7],
        kv_transfer_params={"do_remote_decode": producer, "transfer_id": "test"},
    )
    params = request.to_sampling_params(1, {})
    assert params.ignore_eos == producer
    assert params.stop == ([] if producer else ["STOP"])
    assert params.stop_token_ids == ([] if producer else [7])


def test_responses_wrapped_function_matches_flat_function():
    function = {"name": "demo", "parameters": {"type": "object"}, "strict": False}
    named = {"type": "function", "name": "demo"}
    wrapped = {"type": "function", "function": function}
    req = ResponsesRequest(input="test", tools=[wrapped], tool_choice=named)
    assert (
        req.model_dump()["tools"]
        == ResponsesRequest(
            input="test", tools=[{"type": "function", **function}], tool_choice=named
        ).model_dump()["tools"]
    )
    assert wrapped == {"type": "function", "function": function}
    with pytest.raises(VLLMValidationError, match="Conflicting function tool field"):
        ResponsesRequest(input="test", tools=[{**wrapped, "name": "other"}])
