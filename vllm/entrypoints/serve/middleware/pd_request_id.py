# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Awaitable

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from vllm.logger import init_logger

logger = init_logger(__name__)


class PDRequestIdMiddleware:
    """Log the router's handoff ID once when an inference request reaches a worker."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] == "http" and scope["path"] in (
            "/v1/chat/completions",
            "/v1/messages",
            "/v1/responses",
        ):
            request_id = Headers(scope=scope).get("x-request-id")
            if request_id:
                logger.info(
                    "P/D received path=%s worker_request_id=%r",
                    scope["path"],
                    request_id,
                )
        return self.app(scope, receive, send)
