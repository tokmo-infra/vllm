# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.common.cli_args import FrontendArgs
from vllm.entrypoints.serve.middleware.register import init_entrypoints_middleware


def test_pd_request_id_logs_inference_arrival_even_on_validation_error():
    module = "vllm.entrypoints.serve.middleware.pd_request_id"
    app = FastAPI()

    @app.post("/v1/messages")
    def messages(body: dict):
        return body

    args = FrontendArgs(middleware=[f"{module}.PDRequestIdMiddleware"])
    init_entrypoints_middleware(args, app, ())
    client = TestClient(app)
    with patch(f"{module}.logger.info") as log:
        assert client.post("/v1/messages", json={}).is_success
        client.get("/health", headers={"X-Request-Id": "probe"})
        assert (
            client.post("/v1/messages", headers={"X-Request-Id": "handoff"}).status_code
            == 422
        )
        assert log.call_count == 1
        assert log.call_args.args[1:] == ("/v1/messages", "handoff")
