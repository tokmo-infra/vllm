# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real process/RPC tests; load by file path without GPU/vLLM dependencies.

Minimal CPU-only invocation (bypasses the repository's GPU conftest):
    .venv/bin/python -m pytest --confcutdir=tests/tokenizers \
        tests/tokenizers/test_kimi_k3_shared_cache.py

These exercise the standalone L2 API and real CPU-only Prometheus HTTP export,
not model loading or full OpenAI server integration.
"""

import contextlib
import fcntl
import importlib.util
import multiprocessing
import os
import shutil
import signal
import struct
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / "vllm/tokenizers/kimi_k3_cache_shared.py"
KEY = (("kimi", "stable-tokenizer-sha"), False, "你好 <|special|>")
PACKED = struct.pack("=III", 7, 11, 2**32 - 1)
BUDGET = 1024**2


def _client(path, budget=BUDGET):
    spec = importlib.util.spec_from_file_location("standalone_kimi_cache", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SharedTextPieceCache(str(path), budget)


def _writer(path, queue):
    cache = _client(path)
    cache.put(KEY, PACKED, 1, "e1")
    queue.put(cache.stats())


def _reader(path, queue):
    queue.put(_client(path).get(KEY, "e1"))


def _concurrent(path, index, start, queue):
    assert start.wait(15)
    cache = _client(path)

    def round_trip(item):
        key = (KEY[0], False, f"worker-{index}-item-{item}")
        cache.put(key, PACKED, 1, "e1")
        assert cache.get(key, "e1") == (PACKED, 1)

    with ThreadPoolExecutor(max_workers=4) as threads:
        list(threads.map(round_trip, range(24)))
    queue.put(cache.stats())


def _late_writer(path, start, queue):
    cache = _client(path)
    assert cache.get(KEY, "old") is None
    queue.put("ready")
    assert start.wait(15)
    cache.put(KEY, PACKED, 1, "old")
    queue.put(cache.get(KEY, "new"))


def _inherited_reader(cache, queue):
    queue.put(cache.get(KEY, "e1"))


def _single_api_scrape(path, queue):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from prometheus_client import (
        REGISTRY,
        CollectorRegistry,
        make_asgi_app,
        multiprocess,
    )

    spec = importlib.util.spec_from_file_location(
        "single_api_cache_test", MODULE.with_name("kimi_k3_cache.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shared = _client(path)
    cache = module.TextPieceCache(4096, "single-api", shared=shared)

    def native(text, allow_special_tokens=True):
        return [7, 11]

    for _ in range(2):
        assert cache.encode("backend", native, "exact text", False, set()) == [7, 11]
    # The default logger removes MetricWrapperBase collectors by this prefix.
    for collector in list(REGISTRY._collector_to_names):
        if getattr(collector, "_name", "").startswith("vllm:"):
            REGISTRY.unregister(collector)
    registry = REGISTRY
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR") is not None:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
    app = FastAPI()
    app.mount("/metrics", make_asgi_app(registry=registry))
    before = shared.stats()
    with TestClient(app) as client:
        first = client.get("/metrics/")
        second = client.get("/metrics/")
    queue.put(
        (
            first.status_code,
            first.text,
            second.text,
            before,
            shared.stats(),
            os.environ.get("PROMETHEUS_MULTIPROC_DIR"),
        )
    )


def _scrape_samples(path, monkeypatch, multiprocess_mode=False):
    from prometheus_client.parser import text_string_to_metric_families

    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    monkeypatch.delenv("prometheus_multiproc_dir", raising=False)
    if multiprocess_mode:
        (path / "metrics").mkdir()
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(path / "metrics"))
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    worker = ctx.Process(target=_single_api_scrape, args=(path, queue))
    worker.start()
    status, first, second, before, after, env = queue.get(timeout=20)
    _join(worker)
    assert status == 200
    assert before == after
    assert env == (str(path / "metrics") if multiprocess_mode else None)

    def samples(text):
        result = {}
        for metric in text_string_to_metric_families(text):
            for sample in metric.samples:
                if sample.name.startswith("vllm:tokenizer"):
                    key = sample.name, tuple(sorted(sample.labels.items()))
                    assert key not in result, "duplicate cache metric sample"
                    result[key] = sample.value
        return result

    assert samples(first) == samples(second)
    return samples(first), before


def _join(process):
    process.join(timeout=20)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
        pytest.fail("cache worker did not finish")
    assert process.exitcode == 0


def _stop_server(path):
    pid_file = path / "server.pid"
    if pid_file.exists():
        pid = int(pid_file.read_text())
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        with (path / "server.lock").open("rb") as lease:
            while True:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    assert time.monotonic() < deadline, "server did not stop"
                    time.sleep(0.01)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, 0)


@pytest.fixture
def cache_path():
    # AF_UNIX paths must stay short even on macOS's long pytest temp roots.
    path = Path(tempfile.mkdtemp(prefix="kimi-l2-", dir="/tmp"))
    yield path
    _stop_server(path)
    shutil.rmtree(path)


def test_spawn_writer_exit_does_not_lose_shared_entries(cache_path):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    writer = ctx.Process(target=_writer, args=(cache_path, queue))
    writer.start()
    assert queue.get(timeout=20)["entries"] == 1
    _join(writer)
    reader = ctx.Process(target=_reader, args=(cache_path, queue))
    reader.start()
    assert queue.get(timeout=20) == (PACKED, 1)
    _join(reader)


def test_shared_budget_is_configurable_beyond_single_node_example(cache_path):
    """The operator's shared budget is not capped by the 8-GPU 16-GiB recipe."""
    budget = 32 * 1024**3
    cache = _client(cache_path, budget=budget)
    cache.put(KEY, PACKED, 1, "e1")
    peer = _client(cache_path, budget=budget)
    assert peer.get(KEY, "e1") == (PACKED, 1)
    stats = peer.stats()
    assert stats["max_bytes"] == budget
    assert len(PACKED) < stats["bytes"] < BUDGET
    assert stats["entries"] == 1


def test_simultaneous_spawn_startup_and_threaded_round_trips(cache_path):
    ctx = multiprocessing.get_context("spawn")
    start, queue = ctx.Event(), ctx.Queue()
    workers = [
        ctx.Process(target=_concurrent, args=(cache_path, i, start, queue))
        for i in range(4)
    ]
    for worker in workers:
        worker.start()
    start.set()
    for _ in workers:
        stats = queue.get(timeout=20)
        assert 0 < stats["bytes"] <= stats["max_bytes"] == BUDGET
    for worker in workers:
        _join(worker)
    assert _client(cache_path).stats()["entries"] == 96


def test_exact_key_special_policy_and_mutable_decode_are_independent(cache_path):
    cache = _client(cache_path)
    cache.put(KEY, PACKED, 1, "e1")
    assert cache.get((KEY[0], True, KEY[2]), "e1") is None
    control_key = (KEY[0], True, KEY[2])
    cache.put(control_key, struct.pack("=I", 99), 1, "e1")
    assert cache.get(control_key, "e1") == (struct.pack("=I", 99), 1)
    assert cache.get((("kimi", "different-sha"), False, KEY[2]), "e1") is None
    assert cache.get((KEY[0], False, KEY[2] + " "), "e1") is None
    packed, count = cache.get(KEY, "e1")
    mutable = bytearray(packed)
    mutable[0] ^= 255
    decoded = list(struct.unpack("=III", packed))
    decoded[0] = 999
    snapshot = cache.stats()
    snapshot["bytes"] = 0
    assert cache.get(KEY, "e1") == (PACKED, count)
    assert _client(cache_path).stats()["bytes"] > len(PACKED)


def test_lru_charges_keys_and_evicts_without_exceeding_total_budget(cache_path):
    cache = _client(cache_path, 4096)
    keys = [(KEY[0], False, str(i)) for i in range(3)]
    cache.put(keys[0], PACKED, 1, "e1")
    cache.put(keys[1], PACKED, 1, "e1")
    assert cache.stats()["entries"] == 2
    assert cache.get(keys[0], "e1") == (PACKED, 1)
    cache.put(keys[2], PACKED, 1, "e1")
    assert cache.get(keys[1], "e1") is None
    assert cache.get(keys[0], "e1") == (PACKED, 1)
    stats = cache.stats()
    assert set(stats) == {"bytes", "max_bytes", "entries", "evictions"}
    assert stats["entries"] == 2 and stats["evictions"] == 1
    assert 0 < stats["bytes"] <= stats["max_bytes"] == 4096
    cache.put((KEY[0], False, "巨" * 5000), PACKED, 0, "e1")
    assert cache.stats() == stats
    cache.put(keys[0], b"", 0, "e1")
    assert cache.get(keys[0], "e1") == (b"", 0)
    assert cache.stats()["entries"] == 2


def test_late_old_epoch_write_cannot_overwrite_new_epoch(cache_path):
    ctx = multiprocessing.get_context("spawn")
    start, queue = ctx.Event(), ctx.Queue()
    worker = ctx.Process(target=_late_writer, args=(cache_path, start, queue))
    worker.start()
    assert queue.get(timeout=20) == "ready"
    cache = _client(cache_path)
    new_packed = struct.pack("=I", 999)
    cache.put(KEY, new_packed, 0, "new")
    start.set()
    assert queue.get(timeout=20) == (new_packed, 0)
    _join(worker)
    assert cache.get(KEY, "new") == (new_packed, 0)
    assert cache.get(KEY, "old") == (PACKED, 1)
    assert cache.stats()["entries"] == 2


def test_fork_reconnects_live_parent_connection(cache_path):
    cache = _client(cache_path)
    cache.put(KEY, PACKED, 1, "e1")
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    worker = ctx.Process(target=_inherited_reader, args=(cache, queue))
    worker.start()
    assert queue.get(timeout=20) == (PACKED, 1)
    _join(worker)
    assert cache.get(KEY, "e1") == (PACKED, 1)


def test_private_files_and_budget_disagreement_fail_explicitly(cache_path):
    _client(cache_path)
    assert cache_path.stat().st_mode & 0o777 == 0o700
    for name in ("auth", "cache.sock", "startup.lock", "server.lock", "server.pid"):
        assert (cache_path / name).stat().st_mode & 0o777 == 0o600
    assert len((cache_path / "auth").read_bytes()) == 32
    with pytest.raises(ValueError, match="same max_bytes"):
        _client(cache_path, BUDGET * 2)
    cache_path.chmod(0o755)
    try:
        with pytest.raises(PermissionError, match="0700"):
            _client(cache_path)
    finally:
        cache_path.chmod(0o700)


def test_server_only_multiprocess_metrics_count_one_budget(cache_path, monkeypatch):
    from prometheus_client import CollectorRegistry, multiprocess

    metrics_dir = cache_path / "metrics"
    metrics_dir.mkdir()
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(metrics_dir))
    clients = [_client(cache_path) for _ in range(8)]
    for client in clients:
        client.put(KEY, PACKED, 1, "e1")
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=str(metrics_dir))
    assert registry.get_sample_value("vllm:tokenizer_l2_cache_max_bytes") == BUDGET
    assert registry.get_sample_value("vllm:tokenizer_l2_cache_entries") == 1
    assert (
        registry.get_sample_value("vllm:tokenizer_l2_cache_bytes")
        == clients[0].stats()["bytes"]
    )
    server_pid = (cache_path / "server.pid").read_text()
    assert all(p.stem.endswith("_" + server_pid) for p in metrics_dir.glob("*.db"))
    for i in range(800):
        clients[0].put((KEY[0], False, str(i)), PACKED, 0, "e1")
    evictions = clients[0].stats()["evictions"]
    assert evictions > 0
    assert (
        registry.get_sample_value("vllm:tokenizer_l2_cache_evictions_total")
        == evictions
    )


@pytest.mark.parametrize("multiprocess_mode", [False, True])
def test_single_api_l1_metrics_survive_logger_cleanup(
    cache_path, monkeypatch, multiprocess_mode
):
    samples, _ = _scrape_samples(cache_path, monkeypatch, multiprocess_mode)
    label = (("api_process", "single-api"),)
    assert samples.get(("vllm:tokenizer_cache_max_bytes", label)) == 4096
    assert samples.get(("vllm:tokenizer_cache_bytes", label), 0) > 0
    assert (
        samples.get(
            (
                "vllm:tokenizer_cache_tokens_total",
                label + (("source", "encoded_tokens"),),
            )
        )
        == 2
    )
    assert (
        samples.get(
            (
                "vllm:tokenizer_cache_tokens_total",
                label + (("source", "reused_tokens"),),
            )
        )
        == 2
    )
    assert (
        samples.get(("vllm:tokenizer_cache_requests_total", label + (("path", "hit"),)))
        == 1
    )


def test_single_api_l2_metrics_scrape_without_cache_mutation(cache_path, monkeypatch):
    samples, stats = _scrape_samples(cache_path, monkeypatch)
    assert samples.get(("vllm:tokenizer_l2_cache_max_bytes", ())) == BUDGET
    assert samples.get(("vllm:tokenizer_l2_cache_entries", ())) == stats["entries"] == 1
    assert samples.get(("vllm:tokenizer_l2_cache_bytes", ())) == stats["bytes"]
    assert samples.get(("vllm:tokenizer_l2_cache_evictions_total", ())) == 0
    files = list((cache_path / "metrics").glob("*.db"))
    assert files
    pid = (cache_path / "server.pid").read_text()
    assert all(path.stem.endswith("_" + pid) for path in files)


def test_single_api_default_metrics_restart_counts_one_live_budget(
    cache_path, monkeypatch
):
    from prometheus_client import CollectorRegistry, multiprocess

    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    monkeypatch.delenv("prometheus_multiproc_dir", raising=False)
    cache = _client(cache_path)
    cache.put(KEY, PACKED, 1, "e1")
    _stop_server(cache_path)
    replacement = _client(cache_path)
    assert replacement.get(KEY, "e1") is None
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=str(cache_path / "metrics"))
    assert registry.get_sample_value("vllm:tokenizer_l2_cache_max_bytes") == BUDGET
    assert registry.get_sample_value("vllm:tokenizer_l2_cache_bytes") == 0
    assert registry.get_sample_value("vllm:tokenizer_l2_cache_entries") == 0


def test_dead_server_rpc_fails_and_new_client_restarts_without_double_budget(
    cache_path, monkeypatch
):
    from prometheus_client import CollectorRegistry, multiprocess

    metrics_dir = cache_path / "metrics"
    metrics_dir.mkdir()
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(metrics_dir))
    client = _client(cache_path)
    client.put(KEY, PACKED, 1, "e1")
    old_auth = (cache_path / "auth").read_bytes()
    _stop_server(cache_path)
    with pytest.raises(RuntimeError, match="RPC failed"):
        client.stats()
    replacement = _client(cache_path)
    assert replacement.get(KEY, "e1") is None
    assert (cache_path / "auth").read_bytes() != old_auth
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=str(metrics_dir))
    assert registry.get_sample_value("vllm:tokenizer_l2_cache_max_bytes") == BUDGET
    assert registry.get_sample_value("vllm:tokenizer_l2_cache_bytes") == 0


def test_startup_failure_is_explicit_for_overlong_socket_path(cache_path):
    with pytest.raises(RuntimeError, match="failed to start"):
        _client(cache_path / ("x" * 120))


@pytest.mark.parametrize(
    ("packed", "count"), [(b"x", 0), (PACKED, -1), (PACKED, 4), (bytearray(4), 0)]
)
def test_invalid_packed_data_is_not_stored(cache_path, packed, count):
    cache = _client(cache_path)
    with pytest.raises(ValueError):
        cache.put(KEY, packed, count, "e1")
    assert cache.stats()["entries"] == 0


def test_private_l1s_reuse_native_bpe_from_shared_l2_without_reencoding(cache_path):
    """Empty peer L1 must reuse exact native IDs, including control-token policy."""
    import tiktoken

    spec = importlib.util.spec_from_file_location(
        "native_piece_cache_test", MODULE.with_name("kimi_k3_cache.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoding = tiktoken.Encoding(
        name="shared-native-policy-test",
        pat_str=r"\s+|[^\s]+",
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={"<|control|>": 256},
    )
    calls = []

    def native(text, allow_special_tokens=True):
        calls.append(allow_special_tokens)
        if allow_special_tokens:
            return encoding.encode(text, allowed_special="all")
        return encoding.encode(text, disallowed_special=())

    first = module.TextPieceCache(4096, "first", shared=_client(cache_path))
    peer = module.TextPieceCache(4096, "peer", shared=_client(cache_path))
    text = "你好 <|control|> exact text"
    for flag in [True, False]:
        expected = first.encode("same-backend", native, text, flag, {256})
        result = peer.encode("same-backend", native, text, flag, {256})
        assert result == expected
        result.clear()
        assert peer.encode("same-backend", native, text, flag, {256}) == expected
    assert calls == [True, False]
    assert _client(cache_path).stats()["entries"] == 2
