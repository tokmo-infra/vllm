# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare real /tokenize responses with the unchanged native GLM tokenizer."""

import argparse
import hashlib
import http.client
import json
import time
from urllib.parse import urlsplit

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--url", action="append", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    messages = [
        {"role": "system", "content": "你是代码助手。"},
        {"role": "user", "content": "def f(x):\n    return x + 1\n" * 1000},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    cases = [
        {"prompt": text, "add_special_tokens": False},
        {"messages": messages, "add_generation_prompt": True},
        {"prompt": "Unicode 你好\n \t end", "add_special_tokens": True},
    ]
    results = []
    for url in args.url:
        address = urlsplit(url)
        connection = http.client.HTTPConnection(
            address.hostname, address.port, timeout=60
        )
        for case in cases:
            if "messages" in case:
                golden_text = tokenizer.apply_chat_template(
                    case["messages"], tokenize=False, add_generation_prompt=True
                )
                expected = tokenizer(golden_text, add_special_tokens=False)["input_ids"]
            else:
                expected = tokenizer(
                    case["prompt"], add_special_tokens=case["add_special_tokens"]
                )["input_ids"]
            timings = []
            for _ in range(2):
                started = time.monotonic()
                connection.request(
                    "POST",
                    "/tokenize",
                    json.dumps(
                        {"model": args.model_name, **case}, ensure_ascii=False
                    ).encode(),
                    {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer " + args.api_key,
                    },
                )
                response = connection.getresponse()
                body = response.read()
                assert response.status == 200, (url, response.status, body[:1000])
                actual = json.loads(body)
                assert actual["tokens"] == expected, (
                    url,
                    case.keys(),
                    actual["count"],
                    len(expected),
                )
                timings.append(time.monotonic() - started)
            results.append(
                {
                    "url": url,
                    "case": "chat" if "messages" in case else "completion",
                    "add_special_tokens": case.get("add_special_tokens", False),
                    "count": len(expected),
                    "full_ids_equal": True,
                    "seconds": timings,
                    "token_ids_sha256": hashlib.sha256(
                        json.dumps(expected).encode()
                    ).hexdigest(),
                }
            )
        connection.close()
    print(json.dumps({"status": "passed", "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
