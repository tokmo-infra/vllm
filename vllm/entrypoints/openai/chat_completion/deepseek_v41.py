# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 Flash wire contract for the public Chat Completions endpoint."""

from collections.abc import AsyncIterator
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse


def is_flash(model: str) -> bool:
    return model.lower() in {
        "deepseek-v4.1-flash-1017",
        "deepseek-v4.1-flash",
        "deepseek-4.1-flash",
    }


def prepare_request(request: ChatCompletionRequest) -> None:
    if "max_tokens" in request.model_fields_set:
        limit = request.max_tokens
        if limit is not None and not 1 <= limit <= 393216:
            raise ValueError("max_tokens must be between 1 and 393216")
    # Explicit native thinking takes precedence over effort and template defaults.
    kwargs = dict(request.chat_template_kwargs or {})
    effort = request.reasoning_effort
    if request.thinking is not None:
        thinking = request.thinking.type == "enabled"
    elif effort is None:
        thinking = kwargs.get("thinking", kwargs.get("enable_thinking", True))
    else:
        thinking = effort != "none"
    if effort in ("medium", "xhigh") or (effort == "none" and thinking):
        request.reasoning_effort = "high"
    elif effort == "minimal":
        request.reasoning_effort = "low"
    kwargs.update(thinking=thinking, enable_thinking=thinking)
    request.chat_template_kwargs = kwargs
    if request.tool_choice == "none":
        # Leaving tool definitions in the prompt invites hidden calls that the
        # parser then discards, sometimes leaving a whitespace-only answer.
        request.tools = None
    request._deepseek_chat = True


def wire_fields(payload: dict[str, Any]) -> dict[str, Any]:
    for choice in payload["choices"]:
        part = choice["message"] if "message" in choice else choice["delta"]
        if "reasoning" in part:
            part.setdefault("reasoning_content", part["reasoning"])
    usage = payload.get("usage")
    if usage:
        details = usage.get("prompt_tokens_details") or {}
        hit = details.get("cached_tokens")
        if hit is not None:
            usage["prompt_cache_hit_tokens"] = hit
            usage["prompt_cache_miss_tokens"] = usage["prompt_tokens"] - hit
    return payload


async def stream_chunks(
    chunks: AsyncIterator[ChatCompletionStreamResponse | ErrorResponse],
    request: ChatCompletionRequest,
) -> AsyncIterator[dict[str, Any]]:
    """Attach measured final usage to the final choice without delaying text."""
    remaining = request.n or 1
    terminal = None
    null_usage = bool(request.stream_options and request.stream_options.include_usage)
    async for chunk in chunks:
        if isinstance(chunk, ErrorResponse):
            yield chunk.model_dump()
            return
        payload = wire_fields(
            chunk.model_dump(exclude_unset=True, exclude_none=not chunk.choices)
        )
        if not chunk.choices:
            if terminal is None or not payload.get("usage"):
                raise ValueError(
                    "DeepSeek stream ended without a final choice or usage"
                )
            terminal.update({k: v for k, v in payload.items() if k != "choices"})
            yield terminal
            terminal = None
            continue

        choice = payload["choices"][0]
        if choice.get("finish_reason") is not None:
            remaining -= 1
            if remaining == 0:
                terminal_choice = dict(choice, delta={})
                terminal = dict(payload, choices=[terminal_choice])
                # A backend step may carry both the last token and the stop.
                # Keep logprobs/token IDs on the token rather than replaying them.
                delta = choice.get("delta", {})
                if not (
                    any(delta.values())
                    or choice.get("logprobs")
                    or choice.get("token_ids")
                ):
                    continue
                terminal_choice.pop("logprobs", None)
                terminal_choice.pop("token_ids", None)
                choice["finish_reason"] = None
                choice.pop("stop_reason", None)
                choice.pop("weight_versions", None)
                payload.pop("kv_transfer_params", None)
        if null_usage:
            payload["usage"] = None
        yield payload
    if remaining or terminal is not None:
        raise ValueError("DeepSeek stream ended before final choice and usage")
