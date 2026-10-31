# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.placement import (  # noqa: E501
    normalize_placement_segments,
    preferred_owner,
    put_with_placement,
)
from vllm.v1.request import RequestStatus

from .test_mooncake_store_worker import (
    _DISK_OFFLOAD_BUDGET_FOR_SPLIT,
    _make_load_req,
    _make_store_recving_thread,
)
from .utils import (
    create_mock_connector,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test


def test_weighted_placement_preserves_group_and_result_order():
    segments = normalize_placement_segments({"a": 1, "b": 3})
    counts = Counter(preferred_owner(f"key@{i}", segments) for i in range(10000))
    assert 7200 < counts["b"] < 7800
    assert preferred_owner("shard1@hash", segments) == preferred_owner(
        "shard2@hash", segments
    )
    keys = [f"key@{i}" for i in range(20)]
    groups = [f"group-{i}" for i in range(20)]
    config = SimpleNamespace(preferred_segment="original", group_ids=groups)

    class Store:
        def batch_put_from_multi_buffers(self, batch, buffers, sizes, cfg):
            for key, buffer, group in zip(batch, buffers, cfg.group_ids, strict=True):
                i = keys.index(key)
                assert buffer == [i]
                assert group == groups[i]
            return [keys.index(key) for key in batch]

    result = put_with_placement(
        Store(), keys, [[i] for i in range(20)], [[1]] * 20, config, segments, groups
    )
    assert result == list(range(20))
    assert config.preferred_segment == "original"
    assert config.group_ids == groups


def test_placement_restores_configuration_on_store_failure():
    config = SimpleNamespace(preferred_segment="original", group_ids=["old"])

    class Store:
        def batch_put_from_multi_buffers(self, *args):
            raise RuntimeError("store failed")

    with pytest.raises(RuntimeError, match="store failed"):
        put_with_placement(
            Store(), ["key"], [[1]], [[1]], config, (("owner", 1),), ["group"]
        )
    assert config.preferred_segment == "original"
    assert config.group_ids == ["old"]


def test_failed_subbatch_invalidates_unread_blocks():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [-10, -10]
    thread = _make_store_recving_thread(
        store, disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_FOR_SPLIT
    )
    thread._handle_request(_make_load_req("req-a", [b"a0", b"a1", b"a2"], token_len=48))
    assert thread.get_and_clear_block_ids_with_load_errors() == {0, 1, 2}


def test_failed_receive_recomputes_without_querying_external_cache_again():
    scheduler = create_scheduler(create_vllm_config(kv_load_failure_policy="recompute"))
    request = create_request(num_tokens=3 * scheduler.block_size)
    scheduler.add_request(request)
    scheduler.connector = create_mock_connector()
    scheduler.connector.get_loaded_kv_cache_group_ids.return_value = (0,)
    scheduler.connector.get_num_new_matched_tokens.return_value = (
        2 * scheduler.block_size, True
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()
    scheduled = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    output = create_model_runner_output([], finished_recving={request.request_id})
    assert output.kv_connector_output is not None
    output.kv_connector_output.failed_recving = {request.request_id}
    scheduler.update_from_output(scheduled, output)
    scheduled = scheduler.schedule()
    assert request.request_id in scheduled.num_scheduled_tokens
    assert request.skip_reading_external_kv
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 1
