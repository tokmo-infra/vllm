"""Capture a separate full P/D request run, outside timed performance samples."""
import argparse
import concurrent.futures
import json
import random
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--concurrency", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--prefix", required=True)
parser.add_argument("--delay", type=int, default=16)
args = parser.parse_args()
assert 1 <= args.concurrency <= 512
args.output.mkdir(parents=True, exist_ok=False)
router = "http://11.11.0.89:28080"
decode = "http://11.11.0.89:30011"


def post(url, payload):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with opener.open(request, timeout=600) as response:
        body = response.read()
        return json.loads(body) if body else {"status": response.status}


def complete(index):
    rng = random.Random(42 + index)
    payload = dict(model="deepseek-v4.1-flash-1017",
                   prompt=[rng.randint(100, 10000) for _ in range(1024)],
                   max_tokens=1024, temperature=0, ignore_eos=True, stream=False)
    started = time.time()
    try:
        response = post(router + "/v1/completions", payload)
        assert response["usage"]["completion_tokens"] == 1024, response.get("usage")
        return dict(index=index, started=started, finished=time.time(), response=response)
    except Exception as exc:
        return dict(index=index, started=started, finished=time.time(), error=str(exc))


warmup = complete(-1)
(args.output / "warmup.json").write_text(json.dumps(warmup, indent=2))
assert "error" not in warmup, warmup
config = dict(profile_prefix=args.prefix, delay_iterations=args.delay, max_iterations=8)
(args.output / "config.json").write_text(json.dumps(dict(
    concurrency=args.concurrency, input_tokens=1024, output_tokens=1024,
    scope="Full P90/P5 + D89/D91/D5 requests. Synthetic integer-token prompts; diagnostic only, not a throughput comparison.",
    profile=config,
), indent=2))
post(decode + "/start_profile", config)
results = []
try:
    with (args.output / "requests.jsonl").open("w") as out:
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
            futures = [pool.submit(complete, index) for index in range(args.concurrency)]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                out.write(json.dumps(row) + "\n")
                out.flush()
                results.append(row)
finally:
    post(decode + "/stop_profile", {})
summary = dict(total=len(results), errors=sum("error" in row for row in results))
(args.output / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
assert summary["errors"] == 0, summary
