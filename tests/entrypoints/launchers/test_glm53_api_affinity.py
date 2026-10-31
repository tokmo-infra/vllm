# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import socket
import tempfile
from pathlib import Path

import httpx
import pytest

from integrations.glm53flash.glm_api_cache_router import RouterConfig, create_app
from vllm.entrypoints.launchers.api_affinity_listener import create_private_listener


def test_private_socket_refuses_collision(monkeypatch):
    with tempfile.TemporaryDirectory(dir="/tmp") as temp:
        monkeypatch.setenv("VLLM_API_PRIVATE_SOCKET_DIR", temp)
        sock = create_private_listener(0, 1)
        try:
            assert sock.family == socket.AF_UNIX
            with pytest.raises(OSError):
                create_private_listener(0, 1)
            assert (Path(temp) / "api-0.sock").exists()
        finally:
            sock.close()


def test_proxy_preserves_body_stream_and_rank_header():
    async def run():
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"data: [DONE]\n\n"

        async def upstream(request):
            assert request.content == b'{"prompt":"hello"}'
            assert request.headers["x-data-parallel-rank"] == "7"
            return httpx.Response(200, stream=Stream())

        app = create_app(
            RouterConfig("http://control", ("http://api",)),
            httpx.MockTransport(upstream),
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://router"
            ) as client:
                response = await client.post(
                    "/v1/completions",
                    content=b'{"prompt":"hello"}',
                    headers={"x-data-parallel-rank": "7"},
                )
                assert response.content == b"data: [DONE]\n\n"
                assert app.state.pool.active == [0]
        finally:
            await app.state.client.aclose()

    asyncio.run(run())
