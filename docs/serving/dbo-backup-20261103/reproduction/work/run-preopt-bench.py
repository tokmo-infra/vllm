import argparse
import concurrent.futures
import datetime
import json
import os
import pathlib
import subprocess
import threading
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--ep", type=int, required=True)
p.add_argument("--concurrency", type=int, required=True)
p.add_argument("--run-tag", default="")
a = p.parse_args()
c = a.concurrency
root = (
    pathlib.Path("/work/performance/preopt-dep16-p5d5-001")
    / f"ep{a.ep}-c{c}{a.run_tag}"
)
root.mkdir(parents=True, exist_ok=False)
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
endpoints = {
    "p90": "http://11.11.0.90:30010",
    "d89": "http://11.11.0.89:30011",
    "router": "http://11.11.0.89:28080",
}
for name, url in endpoints.items():
    with op.open(url + "/health", timeout=5) as response:
        assert response.status == 200
with op.open(endpoints["router"] + "/workers", timeout=10) as response:
    workers = json.load(response)
(root / "workers.json").write_text(json.dumps(workers, indent=2))
counts = {
    t: sum(w["worker_type"] == t and w["is_healthy"] for w in workers["workers"])
    for t in ["prefill", "decode"]
}
assert counts == {"prefill": 2, "decode": a.ep}, counts

stop = threading.Event()


def monitor():
    with (root / "metrics.jsonl").open("w") as out:
        while not stop.is_set():
            for name, url in endpoints.items():
                try:
                    with op.open(url + "/metrics", timeout=3) as r:
                        text = r.read().decode()
                    out.write(
                        json.dumps(
                            dict(
                                time=time.time(),
                                monotonic=time.perf_counter(),
                                name=name,
                                metrics=text,
                            )
                        )
                        + "\n"
                    )
                except Exception as exc:
                    out.write(
                        json.dumps(dict(time=time.time(), name=name, error=str(exc)))
                        + "\n"
                    )
            out.flush()
            stop.wait(5)


t = threading.Thread(target=monitor, daemon=True)
t.start()
cmd = [
    "/work/.venv/bin/python",
    "-m",
    "vllm.entrypoints.cli.main",
    "bench",
    "serve",
    "--backend",
    "vllm",
    "--base-url",
    endpoints["router"],
    "--endpoint",
    "/v1/completions",
    "--model",
    "deepseek-v4.1-flash-1017",
    "--tokenizer",
    "/model",
    "--tokenizer-mode",
    "deepseek_v41",
    "--dataset-name",
    "random",
    "--random-input-len",
    "1024",
    "--random-output-len",
    "1024",
    "--random-range-ratio",
    "0",
    "--random-prefix-len",
    "0",
    "--seed",
    "42",
    "--temperature",
    "0",
    "--ignore-eos",
    "--num-prompts",
    str(max(16, c * 4)),
    "--max-concurrency",
    str(c),
    "--request-rate",
    "inf",
    "--num-warmups",
    "16",
    "--save-result",
    "--save-detailed",
    "--result-dir",
    str(root),
    "--result-filename",
    "result.json",
    "--metric-percentiles",
    "50,90,95,99",
    "--percentile-metrics",
    "ttft,tpot,itl,e2el",
    "--label",
    f"ep{a.ep}-c{c}",
]
(root / "benchmark-cli.json").write_text(json.dumps(cmd, indent=2))
started = time.time()
print(
    json.dumps(
        dict(
            status="starting",
            ep=a.ep,
            concurrency=c,
            num_prompts=max(16, c * 4),
            time=started,
        )
    ),
    flush=True,
)
with (root / "benchmark.log").open("w") as log:
    result = subprocess.run(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=dict(os.environ, OPENAI_API_KEY="EMPTY"),
    )
stop.set()
t.join(10)
status = dict(
    exit_code=result.returncode,
    started=started,
    finished=time.time(),
    clock_offset=time.time() - time.perf_counter(),
    ep=a.ep,
    concurrency=c,
    num_prompts=max(16, c * 4),
    run_tag=a.run_tag,
)
(root / "status.json").write_text(json.dumps(status, indent=2))
print(json.dumps(status), flush=True)
print((root / "benchmark.log").read_text()[-5000:], flush=True)
raise SystemExit(result.returncode)
