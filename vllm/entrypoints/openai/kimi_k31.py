# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 API parameters shared by the OpenAI-compatible endpoints."""

from typing import TYPE_CHECKING

from vllm.exceptions import VLLMValidationError

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

K31_SAMPLING = {"temperature": 1.0, "top_p": 0.95}
K31_DEFAULT_OUTPUT_TOKENS = 131072


def validate_k31_request(
    request: "ChatCompletionRequest | ResponsesRequest",
    effort: str | None,
) -> None:
    for name, expected in {
        **K31_SAMPLING,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "n": 1,
    }.items():
        value = getattr(request, name, None)
        if value is not None and value != expected:
            raise VLLMValidationError(
                f"Kimi K3.1 requires {name}={expected}", parameter=name, value=value
            )
    kwargs = request.chat_template_kwargs or {}
    for name, value in (
        ("reasoning_effort", effort),
        ("reasoning_effort", kwargs.get("reasoning_effort")),
        ("thinking_effort", kwargs.get("thinking_effort")),
    ):
        if value is not None and value not in ("low", "high", "max"):
            raise VLLMValidationError(
                "Kimi K3.1 supports low, high and max reasoning effort",
                parameter=name,
                value=value,
            )
    if (request.model_extra or {}).get("thinking") is not None or any(
        name in kwargs and kwargs[name] is not True
        for name in ("thinking", "enable_thinking")
    ):
        raise VLLMValidationError("Kimi K3.1 always reasons", parameter="thinking")
