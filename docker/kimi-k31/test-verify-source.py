#!/usr/bin/env python3
"""Self-contained regression tests for the baseline source verifier."""

import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest


class VerifySourceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "source"
        self.installed = root / "installed"
        self.source.mkdir()
        self.installed.mkdir()
        (self.source / "a.py").write_bytes(b"pass")
        (self.installed / "a.py").write_bytes(b"pass")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "base_image_id": "test",
            "source_commit": "test",
            "files": {"a.py": {
                "size": 4,
                "sha256": hashlib.sha256(b"pass").hexdigest(),
                "in_image": True,
            }},
        }))
        self.manifest = manifest
        self.marker = root / "marker.json"
        module = runpy.run_path(str(Path(__file__).with_name("verify-source.py")))
        self.verify = module["main"]
        self.verify.__globals__.update(
            SOURCE=self.source,
            INSTALLED=self.installed,
            MANIFEST=manifest,
            MARKER=self.marker,
            EXPECTED_MANIFEST=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )

    def test_matching_files_are_accepted(self):
        self.verify()
        self.assertEqual(json.loads(self.marker.read_text())["image_files_verified"], 1)

    def test_changed_source_is_rejected(self):
        (self.source / "a.py").write_bytes(b"fail")
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_changed_installed_file_is_rejected(self):
        (self.installed / "a.py").write_bytes(b"fail")
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_extra_source_is_rejected(self):
        (self.source / "extra.py").write_bytes(b"pass")
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_changed_manifest_is_rejected(self):
        self.manifest.write_text("{}")
        with self.assertRaises(RuntimeError):
            self.verify()


if __name__ == "__main__":
    unittest.main()
