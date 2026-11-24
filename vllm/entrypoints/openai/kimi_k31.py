# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 API parameters shared by the OpenAI-compatible endpoints."""

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ValidationError

from vllm.exceptions import VLLMValidationError

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

K31_SAMPLING = {"temperature": 1.0, "top_p": 0.95}
K31_DEFAULT_OUTPUT_TOKENS = 131072


def is_k31_model_name(name: str | None) -> bool:
    return bool(
        name
        and name.rsplit("/", 1)[-1].lower()
        in (
            "kimi-k3.1-pro",
            "kimi-k3.1-pro-highspeed",
        )
    )


K31CacheTTL = Literal["5m", "1h"]


class K31PromptCacheOptions(BaseModel):
    mode: Literal["implicit"] = "implicit"
    ttl: K31CacheTTL = "5m"

    def to_response(self) -> "K31PromptCacheOptions":
        return self


class K31MessageCacheControl(BaseModel):
    type: Literal["ephemeral"]
    ttl: K31CacheTTL = "5m"


def validate_k31_request(
    request: "ChatCompletionRequest | ResponsesRequest",
    effort: str | None,
) -> None:
    # Chat allows arbitrary extension fields. Validate this extension before
    # scheduling; Responses already parses it as part of the request schema.
    options = (request.model_extra or {}).get("prompt_cache_options")
    if options is not None:
        try:
            K31PromptCacheOptions.model_validate(options)
        except ValidationError as exc:
            raise VLLMValidationError(
                "Expected implicit prompt caching with a 5m or 1h TTL",
                parameter="prompt_cache_options",
            ) from exc
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
