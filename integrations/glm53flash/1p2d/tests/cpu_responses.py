# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only validation with deterministic generation and loopback image fetch."""

import asyncio
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import MagicMock

from PIL import Image

from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.responses.api_router import _convert_stream_to_sse_events
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.multimodal.media.connector import MediaConnector
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams


def make_serving():
    engine = MagicMock()
    engine.model_config.max_model_len = 256
    engine.model_config.hf_config.model_type = "test"
    engine.model_config.get_diff_sampling_param.return_value = {}
    return OpenAIServingResponses(
        engine_client=engine,
        models=MagicMock(),
        online_renderer=MagicMock(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )


async def main():
    serving = make_serving()
    for reason, want in [
        ("stop", "completed"),
        ("length", "incomplete"),
        ("error", "failed"),
    ]:
        request = ResponsesRequest(input="hi", tools=[], stream=True, store=False)
        params = SamplingParams(max_tokens=16)
        context = SimpleContext(response_parser=None)

        async def tokens(context, request_id, reason):
            context.append_output(
                RequestOutput(
                    request_id=request_id,
                    prompt="hi",
                    prompt_token_ids=[7, 8],
                    prompt_logprobs=None,
                    num_cached_tokens=0,
                    finished=True,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text="hello",
                            token_ids=[10],
                            cumulative_logprob=None,
                            logprobs=None,
                            finish_reason=reason,
                        )
                    ],
                )
            )
            yield context

        args = (
            request,
            params,
            tokens(context, request.request_id, reason),
            context,
            "glm-test",
            MagicMock(),
            RequestResponseMetadata(request_id=request.request_id),
        )
        stream = serving.responses_stream_generator(*args)
        sse = [item async for item in _convert_stream_to_sse_events(stream)]
        frames = [
            (line.splitlines()[0][7:], json.loads(line.splitlines()[1][6:]))
            for line in sse
        ]
        event, final = frames[-1]
        assert event == final["type"] == f"response.{want}", final
        assert final["response"]["status"] == want
        assert final["response"]["id"] == frames[0][1]["response"]["id"]
        assert [f[1]["sequence_number"] for f in frames] == list(range(len(frames)))
        if reason != "error":
            assert final["response"]["usage"]["total_tokens"] == 3
        if reason == "length":
            assert (
                final["response"]["incomplete_details"]["reason"] == "max_output_tokens"
            )
        if reason == "error":
            assert final["response"]["error"]["code"] == "server_error"
        print(f"{event}: SSE envelope, ID, sequence and status verified", flush=True)

        context = SimpleContext(response_parser=None)
        args = (
            params,
            tokens(context, request.request_id, reason),
            context,
            "glm-test",
            MagicMock(),
            args[-1],
        )
        await serving._run_background_request_stream(request, *args)
        replay = serving.responses_background_stream_generator(request.request_id)
        events = await asyncio.wait_for(collect(replay), timeout=2)
        assert events[-1].type == event
        replay_after_end = serving.responses_background_stream_generator(
            request.request_id, starting_after=events[-1].sequence_number
        )
        assert not await asyncio.wait_for(collect(replay_after_end), timeout=2)


async def collect(iterator):
    return [event async for event in iterator]


def verify_http_image():
    with io.BytesIO() as out:
        Image.new("RGB", (8, 8), (255, 0, 0)).save(out, format="PNG")
        png = out.getvalue()

    class Origin(BaseHTTPRequestHandler):
        def do_GET(self):
            body = png if self.path == "/valid.png" else b"not-an-image"
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer(("127.0.0.1", 0), Origin) as origin:
        worker = Thread(target=origin.serve_forever, daemon=True)
        worker.start()
        try:
            connector = MediaConnector(allowed_media_domains=["127.0.0.1"])
            url = f"http://127.0.0.1:{origin.server_port}"
            img = connector.fetch_image(url + "/valid.png")
            assert img.getpixel((0, 0)) == (255, 0, 0)
            try:
                connector.fetch_image(url + "/broken.png")
            except ValueError:
                pass
            else:
                raise AssertionError("corrupt image accepted")
        finally:
            origin.shutdown()
            worker.join()
    print("HTTP image: decoded red PNG and rejected invalid bytes", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
    verify_http_image()
