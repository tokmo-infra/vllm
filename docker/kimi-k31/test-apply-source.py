#!/usr/bin/env python3
"""Contract tests for applying a development source tree to an image."""

import hashlib
from pathlib import Path
import runpy
import tempfile
import unittest


sync_source = runpy.run_path(str(Path(__file__).with_name("apply-source.py")))[
    "sync_source"
]


class ApplySourceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.installed = self.root / "image" / "vllm"
        self.source.mkdir()
        self.installed.mkdir(parents=True)
        self.manifest = {"files": {}}

    @staticmethod
    def write(root, name, content):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def baseline(self, name, original, current=None, *, in_image=True):
        self.manifest["files"][name] = {
            "in_image": in_image,
            "sha256": hashlib.sha256(original).hexdigest(),
            "size": len(original),
        }
        if in_image:
            self.write(self.installed, name, original)
        if current is not None:
            self.write(self.source, name, current)

    def test_applies_source_changes_and_preserves_image_only_files(self):
        self.baseline("pkg/changed.py", b"old\n", b"new\n")
        self.baseline("pkg/deleted.py", b"deleted\n")
        self.baseline("pkg/unchanged.py", b"same\n", b"same\n")
        self.baseline("pkg/config.json", b"{}", b'{"new": true}')
        self.baseline("not_packaged.txt", b"old", b"new", in_image=False)
        self.baseline("also_not_packaged.txt", b"same", b"same", in_image=False)
        self.write(self.source, "pkg/added.py", b"added\n")
        self.write(self.source, "pkg/template.jinja", b"{{ value }}")

        image_only = {
            "pkg/generated.py": b"generated\n",
            "third_party/vendor.py": b"vendored\n",
            "_version.py": b"version\n",
            "_C.abi3.so": b"binary",
        }
        for name, content in image_only.items():
            self.write(self.installed, name, content)

        stale_caches = (
            "pkg/__pycache__/changed.cpython-312.pyc",
            "pkg/__pycache__/changed.cpython-312.opt-1.pyc",
            "pkg/__pycache__/deleted.cpython-311.pyc",
        )
        for name in stale_caches:
            self.write(self.installed, name, b"stale bytecode")

        sync_source(self.source, self.installed, self.manifest)

        expected = {
            "pkg/changed.py": b"new\n",
            "pkg/unchanged.py": b"same\n",
            "pkg/config.json": b'{"new": true}',
            "not_packaged.txt": b"new",
            "pkg/added.py": b"added\n",
            "pkg/template.jinja": b"{{ value }}",
            **image_only,
        }
        for name, content in expected.items():
            with self.subTest(path=name):
                self.assertEqual((self.installed / name).read_bytes(), content)
        self.assertFalse((self.installed / "pkg/deleted.py").exists())
        self.assertFalse((self.installed / "also_not_packaged.txt").exists())
        for name in stale_caches:
            with self.subTest(cache=name):
                self.assertFalse((self.installed / name).exists())

    def test_rejects_source_file_symlink(self):
        outside = self.write(self.root, "outside.py", b"private\n")
        (self.source / "link.py").symlink_to(outside)

        with self.assertRaises(Exception):
            sync_source(self.source, self.installed, self.manifest)

        self.assertFalse((self.installed / "link.py").exists())
        self.assertFalse((self.installed / "link.py").is_symlink())
        self.assertEqual(outside.read_bytes(), b"private\n")

    def test_rejects_source_directory_symlink(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.write(outside, "module.py", b"private\n")
        (self.source / "linked").symlink_to(outside, target_is_directory=True)

        with self.assertRaises(Exception):
            sync_source(self.source, self.installed, self.manifest)

        self.assertFalse((self.installed / "linked").exists())
        self.assertFalse((self.installed / "linked").is_symlink())
        self.assertEqual((outside / "module.py").read_bytes(), b"private\n")

    def test_rejects_installed_file_symlink(self):
        self.baseline("changed.py", b"old\n", b"new\n")
        outside = self.write(self.root, "outside.py", b"private\n")
        target = self.installed / "changed.py"
        target.unlink()
        target.symlink_to(outside)

        with self.assertRaises(Exception):
            sync_source(self.source, self.installed, self.manifest)

        self.assertEqual(outside.read_bytes(), b"private\n")

    def test_rejects_installed_directory_symlink(self):
        self.write(self.source, "pkg/new.py", b"new\n")
        outside = self.root / "outside"
        outside.mkdir()
        (self.installed / "pkg").symlink_to(outside, target_is_directory=True)

        with self.assertRaises(Exception):
            sync_source(self.source, self.installed, self.manifest)

        self.assertFalse((outside / "new.py").exists())

    def test_rejects_manifest_paths_outside_the_package(self):
        outside = self.write(self.root / "image", "outside.py", b"private\n")
        original = b"private\n"
        record = {
            "in_image": True,
            "sha256": hashlib.sha256(original).hexdigest(),
            "size": len(original),
        }
        for name in ("../outside.py", str(self.root / "absent.py")):
            with self.subTest(path=name):
                with self.assertRaises(Exception):
                    sync_source(self.source, self.installed, {"files": {name: record}})
                self.assertEqual(outside.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
