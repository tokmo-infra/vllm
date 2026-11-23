# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU video bridge tests. Set GLM_VIDEO_MODEL_DIR to use offline metadata."""

import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import cv2
import numpy as np
import pybase64 as base64
import pytest
import torch
from pydantic import ValidationError

from vllm.config import ModelConfig
from vllm.entrypoints.anthropic.protocol import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
)
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.harmony import response_input_to_harmony
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import construct_input_messages
from vllm.exceptions import VLLMValidationError
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.renderers.chat_utils import parse_chat_messages_async
from vllm.transformers_utils.repo_utils import hf_api


@pytest.fixture(scope="module")
def glm():
    model = os.environ.get("GLM_VIDEO_MODEL_DIR") or hf_api().snapshot_download(
        "zai-org/GLM-5.3-Flash",
        revision="46a050bec13002ad29ee795c16e61cf4fa3c4a9c",
        allow_patterns=[
            "config.json",
            "processor_config.json",
            "tokenizer*.json",
            "chat_template.jinja",
        ],
    )
    config = ModelConfig(
        model=model, seed=0, enforce_eager=True, mm_processor_cache_gb=0
    )
    return config, MULTIMODAL_REGISTRY.create_processor(config)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (56, 56))
    assert writer.isOpened()
    try:
        for i in range(8):
            writer.write(np.full((56, 56, 3), i * 30, dtype=np.uint8))
    finally:
        writer.release()
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    handler = partial(SimpleHTTPRequestHandler, directory=str(path.parent))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as httpd:
        thread = Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{httpd.server_port}/{path.name}", data
        finally:
            httpd.shutdown()
            thread.join()


def messages(protocol, part):
    text_type = "input_text" if protocol == "responses" else "text"
    rows = [
        {
            "role": "user",
            "content": [
                {"type": text_type, "text": "Before"},
                part,
                {"type": text_type, "text": "After"},
            ],
        }
    ]
    if protocol == "responses":
        return construct_input_messages(
            request_input=ResponsesRequest(input=rows).input
        )
    if protocol == "chat":
        return ChatCompletionRequest(model="glm", messages=rows).messages
    req = (
        AnthropicCountTokensRequest(model="glm", messages=rows)
        if protocol == "count"
        else AnthropicMessagesRequest(model="glm", max_tokens=4, messages=rows)
    )
    return AnthropicServingMessages.to_chat_completion_request(req).messages


def block(protocol, url):
    if protocol == "chat":
        return {"type": "video_url", "video_url": {"url": url}}
    if protocol == "responses":
        return {"type": "input_video", "video_url": url}
    if url.startswith("data:"):
        source = {
            "type": "base64",
            "media_type": "video/mp4",
            "data": url.split(",", 1)[1],
        }
    else:
        source = {"type": "url", "url": url}
    return {"type": "video", "source": source}


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["inline", "http"])
async def test_video_reaches_glm_from_each_protocol(glm, video, transport):
    config, processor = glm
    http, data = video
    url = http if transport == "http" else f"data:video/mp4;base64,{data}"
    baseline = None
    for protocol in ("chat", "responses", "messages", "count"):
        rows = messages(protocol, block(protocol, url))
        assert rows[0]["content"][1] == {"type": "video_url", "video_url": {"url": url}}
        conversation, media, _ = await parse_chat_messages_async(rows, config, "openai")
        assert conversation[0]["content"] == [
            {"type": "text", "text": "Before"},
            {"type": "video"},
            {"type": "text", "text": "After"},
        ]
        assert media is not None
        clips = media["video"]
        assert isinstance(clips, list) and len(clips) == 1
        (clip,) = clips
        assert clip is not None
        frames, timing = clip.media
        assert timing["total_num_frames"] == 8 and timing["fps"] == 4
        assert len(frames) >= 2 and frames[-1].mean() > frames[0].mean() + 50
        prompt = processor.info.get_tokenizer().apply_chat_template(
            conversation, tokenize=False
        )
        inputs = processor(prompt, mm_items=processor.info.parse_mm_data(media))
        features = inputs["mm_kwargs"].get_data()
        pixels, grid = features["pixel_values_videos"], features["video_grid_thw"]
        assert grid[0, 0] > 0 and pixels.shape[0] == grid.prod()
        assert len(inputs["mm_placeholders"]["video"]) == 1
        if baseline is None:
            baseline = pixels
        else:
            torch.testing.assert_close(pixels, baseline)


@pytest.mark.asyncio
async def test_model_rejects_second_video_across_turns(glm, video):
    config, _ = glm
    rows = messages("responses", block("responses", video[0]))
    with pytest.raises(VLLMValidationError, match="At most 1 video"):
        await parse_chat_messages_async(rows * 2, config, "openai")


@pytest.mark.parametrize(
    "url",
    [
        None,
        {},
        "",
        "data:image/png;base64,YQ==",
        "data:video/mp4;base64,",
        {"url": "https://vid.test/v", "other": 1},
    ],
)
def test_responses_invalid_video_is_rejected(url):
    with pytest.raises(ValidationError):
        messages("responses", {"type": "input_video", "video_url": url})


@pytest.mark.parametrize(
    "source",
    [
        {},
        {"type": "base64", "media_type": "image/png", "data": "AA=="},
        {"type": "base64", "media_type": "video/mp4", "data": ""},
        {"type": "url", "url": "", "data": "AA=="},
    ],
)
def test_messages_invalid_video_is_rejected(source):
    with pytest.raises(ValidationError):
        messages("messages", {"type": "video", "source": source})


def test_responses_wrapper_history_and_harmony():
    url = "https://video.test/clip.mp4"
    req = ResponsesRequest(
        input=[
            {"role": "assistant", "content": "previous"},
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "https://img.test/a"},
                    {"type": "input_video", "video_url": {"url": url}},
                ],
            },
        ]
    )
    history, current = construct_input_messages(request_input=req.input)
    assert history["content"] == "previous"
    assert current["content"][0]["detail"] == "auto"
    assert current["content"][1] == block("chat", url)
    with pytest.raises(VLLMValidationError, match="video"):
        response_input_to_harmony(req.input[1], [])


def test_responses_rejects_video_in_unsupported_positions():
    part = block("responses", "https://video.test/clip.mp4")
    with pytest.raises(ValidationError):
        ResponsesRequest(input=[{"role": "assistant", "id": "old", "content": [part]}])
    with pytest.raises(VLLMValidationError, match="user message"):
        ResponsesRequest(
            input=[{"type": "function_call_output", "call_id": "t", "output": [part]}]
        )
    with pytest.raises(VLLMValidationError, match="Replay videos"):
        ResponsesRequest(
            input="next", previous_input_messages=[{"role": "user", "content": [part]}]
        )


@pytest.mark.asyncio
async def test_invalid_video_bytes_do_not_degrade_to_text(glm):
    config, _ = glm
    url = "data:video/mp4;base64,bm90IGEgdmlkZW8="
    for protocol in ("responses", "messages"):
        rows = messages(protocol, block(protocol, url))
        with pytest.raises(ValueError):
            await parse_chat_messages_async(rows, config, "openai")


def test_messages_tool_result_preserves_visual_order():
    url = "https://video.test/clip.mp4"
    item = {
        "type": "tool_result",
        "tool_use_id": "t",
        "content": [
            block("messages", url),
            {"type": "image", "source": {"type": "url", "url": "https://img.test/a"}},
        ],
    }
    req = AnthropicMessagesRequest(
        model="glm", max_tokens=8, messages=[{"role": "user", "content": [item]}]
    )
    tool, media = AnthropicServingMessages.to_chat_completion_request(req).messages
    assert tool["tool_call_id"] == "t"
    assert media["content"] == [
        block("chat", url),
        {"type": "image_url", "image_url": {"url": "https://img.test/a"}},
    ]
    for role in ("assistant", "system"):
        with pytest.raises(ValidationError, match="only supported in user"):
            AnthropicCountTokensRequest(
                model="glm", messages=[{"role": role, "content": [item]}]
            )
