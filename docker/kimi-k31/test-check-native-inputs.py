#!/usr/bin/env python3
"""Tests for detecting changes to native and build inputs."""

import hashlib
import runpy
import tempfile
import unittest
from pathlib import Path


find_changes = runpy.run_path(str(Path(__file__).with_name("check-native-inputs.py")))[
    "find_changes"
]


class CheckNativeInputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        inputs = {
            "csrc/kernel.cu": b"kernel v1\n",
            "cmake/common.cmake": b"cmake\n",
            "rust/src/lib.rs": b"rust\n",
            "requirements/cuda.txt": b"requirements\n",
            "tools/build.py": b"build tool\n",
            "vllm/native/existing.cpp": b"native\n",
            "CMakeLists.txt": b"cmake root\n",
            "setup.py": b"setup\n",
            "pyproject.toml": b"project\n",
            "MANIFEST.in": b"manifest\n",
            "rust-toolchain.toml": b"toolchain\n",
        }
        self.expected = {}
        for name, content in inputs.items():
            self.write(name, content)
            self.expected[name] = hashlib.sha256(content).hexdigest()
        self.write("vllm/models/model.py", b"original python\n")

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_unchanged_baseline(self):
        self.assertEqual(find_changes(self.root, self.expected), [])

    def test_new_cuda_source(self):
        self.write("csrc/new_kernel.cu", b"new kernel\n")
        self.assertEqual(find_changes(self.root, self.expected), ["csrc/new_kernel.cu"])

    def test_modified_cuda_source(self):
        self.write("csrc/kernel.cu", b"kernel v2\n")
        self.assertEqual(find_changes(self.root, self.expected), ["csrc/kernel.cu"])

    def test_deleted_cuda_source(self):
        (self.root / "csrc/kernel.cu").unlink()
        self.assertEqual(find_changes(self.root, self.expected), ["csrc/kernel.cu"])

    def test_new_native_file_under_vllm(self):
        self.write("vllm/models/new_kernel.cu", b"new kernel\n")
        self.assertEqual(
            find_changes(self.root, self.expected), ["vllm/models/new_kernel.cu"]
        )

    def test_ordinary_vllm_python_changes_are_ignored(self):
        self.write("vllm/models/model.py", b"modified python\n")
        self.assertEqual(find_changes(self.root, self.expected), [])
        added = self.write("vllm/models/added.py", b"new python\n")
        self.assertEqual(find_changes(self.root, self.expected), [])
        added.unlink()
        (self.root / "vllm/models/model.py").unlink()
        self.assertEqual(find_changes(self.root, self.expected), [])

    def test_other_build_inputs_are_tracked(self):
        names = (
            "CMakeLists.txt",
            "cmake/common.cmake",
            "requirements/cuda.txt",
            "rust/src/lib.rs",
            "tools/build.py",
        )
        for name in names:
            self.write(name, b"changed\n")
        self.assertEqual(find_changes(self.root, self.expected), sorted(names))

    def test_rejects_symlinks_in_source_trees(self):
        cases = (
            ("csrc/link.cu", "csrc/kernel.cu", False),
            ("vllm/linked", "csrc", True),
            ("vllm/link.py", "vllm/models/model.py", False),
        )
        for name, target, is_directory in cases:
            with self.subTest(path=name):
                link = self.root / name
                link.symlink_to(self.root / target, target_is_directory=is_directory)
                try:
                    with self.assertRaises(Exception):
                        find_changes(self.root, self.expected)
                finally:
                    link.unlink()

    def test_rejects_top_level_symlink(self):
        path = self.root / "CMakeLists.txt"
        path.unlink()
        path.symlink_to(self.root / "csrc/kernel.cu")
        with self.assertRaises(Exception):
            find_changes(self.root, self.expected)


if __name__ == "__main__":
    unittest.main()
