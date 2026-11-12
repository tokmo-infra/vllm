# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests of the native Store read/lease protocol boundary."""

import importlib.util
from pathlib import Path

import pytest

# This leaf module has no vLLM runtime dependencies. Load it independently so
# the protocol contract can also be exercised on hosts without torch/CUDA.
_path = (
    Path(__file__).resolve().parents[4]
    / "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/lease.py"
)
_spec = importlib.util.spec_from_file_location("mooncake_store_lease", _path)
assert _spec is not None and _spec.loader is not None
lease = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lease)

pytestmark = pytest.mark.cpu_test


class ReplicaStore:
    """External Store fake; reads expose exactly which replicas were refreshed."""

    def __init__(self, response):
        self.response = response
        self.reads = []

    def batch_get_into_multi_buffers(self, keys, buffers, sizes):
        self.reads.append((keys, buffers, sizes))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.parametrize("code", [-707, -800])
def test_refresh_requeries_only_expired_replicas_and_preserves_other_results(code):
    store = ReplicaStore([256])
    original = [256, code, -10]
    result = lease.retry_expired_get_once(
        store,
        ["hit", "expired", "missing"],
        [[0x1000], [0x2000], [0x3000]],
        [[256], [256], [256]],
        original,
    )
    assert result == [256, 256, -10]
    assert original == [256, code, -10]
    assert store.reads == [(["expired"], [[0x2000]], [[256]])]


def test_mixed_refresh_errors_share_one_fresh_full_get():
    store = ReplicaStore([128, 256])
    assert lease.retry_expired_get_once(
        store, ["a", "b"], [[1, 2], [3]], [[64, 64], [256]], [-707, -800]
    ) == [128, 256]
    assert store.reads == [(["a", "b"], [[1, 2], [3]], [[64, 64], [256]])]


@pytest.mark.parametrize("code", [-707, -800, -10])
def test_second_failure_is_not_accepted_or_retried_indefinitely(code):
    store = ReplicaStore([code])
    assert lease.retry_expired_get_once(store, ["expired"], [[1]], [[16]], [-707]) == [
        code
    ]
    assert len(store.reads) == 1


@pytest.mark.parametrize("results", [[], [256], [-10], [-200]])
def test_success_and_unrelated_failures_do_not_refresh(results):
    store = ReplicaStore(RuntimeError("unexpected retry"))
    count = len(results)
    assert (
        lease.retry_expired_get_once(
            store, ["key"] * count, [[1]] * count, [[256]] * count, results
        )
        == results
    )
    assert store.reads == []


@pytest.mark.parametrize("results", [[], [256, 256], [None], [True], ["256"]])
def test_malformed_refresh_response_fails_closed(results):
    store = ReplicaStore(results)
    with pytest.raises(ValueError, match="malformed"):
        lease.retry_expired_get_once(store, ["key"], [[1]], [[256]], [-800])
    assert len(store.reads) == 1


def test_refresh_exception_is_not_translated_to_a_cache_hit():
    store = ReplicaStore(RuntimeError("replica unavailable"))
    with pytest.raises(RuntimeError, match="replica unavailable"):
        lease.retry_expired_get_once(store, ["key"], [[1]], [[256]], [-707])
    assert len(store.reads) == 1


@pytest.mark.parametrize("bad_vector", ["keys", "buffers", "sizes", "results"])
def test_malformed_input_vectors_never_start_a_transfer(bad_vector):
    store = ReplicaStore([256])
    vectors = dict(keys=["key"], buffers=[[1]], sizes=[[256]], results=[-707])
    vectors[bad_vector] = []
    with pytest.raises(ValueError):
        lease.retry_expired_get_once(store, **vectors)
    assert store.reads == []
