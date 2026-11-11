# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real process/RPC tests; load by file path without GPU/vLLM dependencies.

Minimal CPU-only invocation (bypasses the repository's GPU conftest):
    .venv/bin/python -m pytest --confcutdir=tests/tokenizers \
        tests/tokenizers/test_kimi_k3_shared_cache.py

These exercise the standalone L2 API, not renderer/tokenizer integration.
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
