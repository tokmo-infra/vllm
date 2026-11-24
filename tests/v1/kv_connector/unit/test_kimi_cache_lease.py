# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import hashlib
import shutil
import subprocess
import time

import pytest

from vllm.v1.core import kimi_cache_lease as leases


@pytest.fixture
def lease_server(tmp_path, monkeypatch):
    redis = pytest.importorskip("redis")
    server = shutil.which("redis-server")
    if server is None:
        pytest.skip("redis-server is required for the shared Kimi lease tests")
    assert server is not None
    socket = tmp_path / "cache.sock"
    url = tmp_path / "url"
    url.write_text(f"unix://{socket}")
    monkeypatch.setenv("VLLM_K31_CACHE_LEASE_URL_FILE", str(url))
    monkeypatch.setenv("VLLM_K31_CACHE_LEASE_NAMESPACE", "test")
    for cached in (leases._settings, leases._reader, leases._writer):
        cached.cache_clear()
    process = subprocess.Popen(
        [
            server,
            "--port",
            "0",
            "--unixsocket",
            str(socket),
            "--save",
            "",
            "--dir",
            str(tmp_path),
        ],
        stdout=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5
        while not socket.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        client = redis.Redis(unix_socket_path=str(socket))
        assert client.ping()
        yield client
    finally:
        process.terminate()
        process.wait(timeout=5)
        for cached in (leases._settings, leases._reader, leases._writer):
            cached.cache_clear()


@pytest.mark.asyncio
async def test_first_write_locks_tier_and_only_measured_prefix_is_published(
    lease_server,
):
    keys = [hashlib.sha256(bytes([i])).digest() for i in range(4)]
    first = leases.prepare_k31_cache(keys, 2048, 3600)
    assert first.read_limit == 0
    assert await first.publish(0, 4096) == (4096, 0, 4096)
    warm = leases.prepare_k31_cache(keys, 2048, 300)
    assert warm.read_limit == 4096
    assert await warm.publish(4096, 2048) == (2048, 0, 2048)
    assert leases.prepare_k31_cache(keys, 2048, 300).read_limit == 6144
    for key in warm.keys[:3]:
        assert lease_server.ttl(key) >= 3599
        lease_server.pexpire(key, 1)
    for _ in range(100):
        if not lease_server.exists(*warm.keys):
            break
        await asyncio.sleep(0.01)
    # An old reader completing late cannot resurrect an expired generation.
    assert await warm.publish(4096, 0) == (0, 0, 0)
    cold = leases.prepare_k31_cache(keys, 2048, 300)
    assert cold.read_limit == 0
    assert await cold.publish(0, 6144) == (6144, 6144, 0)


@pytest.mark.asyncio
async def test_concurrent_prefills_charge_each_live_block_once(lease_server):
    keys = [hashlib.sha256(b"shared").digest()]
    a = leases.prepare_k31_cache(keys, 2048, 300)
    b = leases.prepare_k31_cache(keys, 2048, 3600)
    x, y = await asyncio.gather(a.publish(0, 2048), b.publish(0, 2048))
    assert x[0] + y[0] == x[1] + y[2] == 2048
    assert (x[0] == 2048) != (y[0] == 2048)
