# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only lease-refresh contract; no vLLM engine or CUDA initialization."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import Mock

if os.environ.get("MOONCAKE_LEASE_TEST_MODULE"):
    PATH = Path(os.environ["MOONCAKE_LEASE_TEST_MODULE"])
else:
    ROOT = Path(__file__).resolve().parents[4]
    PATH = ROOT / "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/lease.py"
spec = importlib.util.spec_from_file_location("mooncake_lease_contract", PATH)
assert spec is not None and spec.loader is not None
lease = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lease)


class LeaseRefreshTests(unittest.TestCase):
    def setUp(self):
        self.store = Mock()
        self.keys = ["a", "b", "c"]
        self.buffers = [[10], [20, 21], [30]]
        self.sizes = [[8], [3, 5], [8]]

    def run_get(self, result):
        return lease.retry_expired_get_once(
            self.store, self.keys, self.buffers, self.sizes, result
        )

    def test_success_does_not_retry(self):
        self.assertEqual(self.run_get([8, 8, 8]), [8, 8, 8])
        self.store.batch_get_into_multi_buffers.assert_not_called()

    def test_only_expired_key_and_matching_buffers_are_retried(self):
        initial = [8, -707, 8]
        self.store.batch_get_into_multi_buffers.return_value = [8]
        self.assertEqual(self.run_get(initial), [8, 8, 8])
        self.assertEqual(initial, [8, -707, 8])
        self.store.batch_get_into_multi_buffers.assert_called_once_with(
            ["b"], [[20, 21]], [[3, 5]]
        )

    def test_second_expiry_is_not_accepted_or_retried_again(self):
        self.store.batch_get_into_multi_buffers.return_value = [-707]
        self.assertEqual(self.run_get([8, -707, 8]), [8, -707, 8])
        self.assertEqual(self.store.batch_get_into_multi_buffers.call_count, 1)

    def test_other_errors_remain_failures(self):
        self.assertEqual(self.run_get([-800, 8, -703]), [-800, 8, -703])
        self.store.batch_get_into_multi_buffers.assert_not_called()

    def test_mixed_errors_and_multi_expiry_preserve_order(self):
        self.store.batch_get_into_multi_buffers.return_value = [8, -703]
        self.assertEqual(self.run_get([-707, -800, -707]), [8, -800, -703])
        self.store.batch_get_into_multi_buffers.assert_called_once_with(
            ["a", "c"], [[10], [30]], [[8], [8]]
        )

    def test_malformed_retry_vector_is_rejected(self):
        self.store.batch_get_into_multi_buffers.return_value = []
        with self.assertRaises(ValueError):
            self.run_get([8, -707, 8])

    def test_native_retry_exception_is_not_hidden(self):
        self.store.batch_get_into_multi_buffers.side_effect = RuntimeError("native")
        with self.assertRaisesRegex(RuntimeError, "native"):
            self.run_get([8, -707, 8])

    def test_malformed_input_is_rejected_before_rpc(self):
        with self.assertRaises(ValueError):
            self.run_get([-707])
        self.store.batch_get_into_multi_buffers.assert_not_called()


if __name__ == "__main__":
    unittest.main()
