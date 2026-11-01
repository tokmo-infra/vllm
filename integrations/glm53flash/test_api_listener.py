# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE = (
    Path(__file__).resolve().parents[2]
    / "vllm/entrypoints/launchers/api_affinity_listener.py"
)
spec = importlib.util.spec_from_file_location("glm_api_listener", MODULE)
listener = importlib.util.module_from_spec(spec)
spec.loader.exec_module(listener)


class PrivateListeners(unittest.TestCase):
    def test_disabled_uses_existing_shared_listener(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(listener.create_private_listener(0, 4))

    def test_unique_api_sockets_and_conflicts_do_not_replace_live_socket(self):
        with (
            tempfile.TemporaryDirectory(prefix="api-", dir="/tmp") as directory,
            patch.dict(os.environ, {"VLLM_API_PRIVATE_SOCKET_DIR": directory}),
        ):
            sockets = [listener.create_private_listener(i, 4) for i in range(4)]
            try:
                paths = [s.getsockname() for s in sockets]
                self.assertEqual(len(set(paths)), 4)
                original = Path(paths[0]).stat().st_ino
                with self.assertRaises(OSError):
                    listener.create_private_listener(0, 4)
                self.assertEqual(Path(paths[0]).stat().st_ino, original)
                with self.assertRaises(ValueError):
                    listener.create_private_listener(4, 4)
            finally:
                for sock in sockets:
                    sock.close()


if __name__ == "__main__":
    unittest.main()
