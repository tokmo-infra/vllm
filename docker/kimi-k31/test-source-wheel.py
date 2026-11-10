#!/usr/bin/env python3
"""Exercise replacement and fail-closed behavior without modifying a real image."""

import ast
import importlib.util
import json
import os
import re
import shutil
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "kimi_source_wheel", Path(__file__).with_name("source-wheel.py")
)
source_wheel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source_wheel)
REAL_PIP_CHECK = source_wheel.pip_check


class SourceWheelTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.site = self.root / "site-packages"
        self.package = self.site / "vllm"
        self.package.mkdir(parents=True)
        (self.package / "__init__.py").write_text("")
        (self.package / "_C.abi3.so").write_bytes(b"old native")
        (self.package / "vllm-rs").write_bytes(b"old rust")
        (self.package / "stale.py").write_text("old")
        self.wheel_dir = self.root / "dist"
        self.wheel_dir.mkdir()
        self.wheel = self.wheel_dir / "vllm-1.0-py3-none-any.whl"
        (self.wheel_dir / "build-abi.json").write_text(json.dumps({"abi": 1}))
        (self.wheel_dir / "csrc-abi.json").write_text(json.dumps({"abi": 1}))
        self.record = self.root / "record.json"
        self.calls = []
        self.patchers = [
            mock.patch.object(source_wheel, "fingerprint", return_value={"abi": 1}),
            mock.patch.object(
                source_wheel.importlib.util,
                "find_spec",
                return_value=types.SimpleNamespace(
                    origin=str(self.package / "__init__.py")
                ),
            ),
            mock.patch.object(
                source_wheel.sysconfig,
                "get_paths",
                return_value={"purelib": str(self.site), "platlib": str(self.site)},
            ),
            mock.patch.object(
                source_wheel.importlib.metadata, "requires", return_value=[]
            ),
            mock.patch.object(source_wheel, "pip_check", return_value={"exit_code": 0}),
            mock.patch.object(source_wheel.subprocess, "run", side_effect=self.run_uv),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_wheel(self, *, with_native=True, requirements=(), extras=()):
        metadata = "Metadata-Version: 2.1\nName: vllm\nVersion: 1.0\n"
        metadata += "".join(f"Requires-Dist: {req}\n" for req in requirements)
        metadata += "".join(f"Provides-Extra: {extra}\n" for extra in extras)
        with zipfile.ZipFile(self.wheel, "w") as archive:
            archive.writestr("vllm-1.0.dist-info/METADATA", metadata)
            archive.writestr("vllm/__init__.py", "")
            archive.writestr("vllm/vllm-rs", b"new rust")
            if with_native:
                archive.writestr("vllm/_C.abi3.so", b"new native")

    def run_uv(self, command, **kwargs):
        self.calls.append(command)
        if command[2] == "uninstall":
            shutil.rmtree(self.package)
        elif command[2] == "install":
            with zipfile.ZipFile(self.wheel) as archive:
                archive.extractall(self.site)
        return types.SimpleNamespace(returncode=0, stdout="")

    def test_source_wheel_replaces_old_files_and_records_hashes(self):
        self.make_wheel()
        source_wheel.install(self.wheel_dir, self.record)
        self.assertEqual((self.package / "_C.abi3.so").read_bytes(), b"new native")
        self.assertEqual((self.package / "vllm-rs").read_bytes(), b"new rust")
        self.assertFalse((self.package / "stale.py").exists())
        record = json.loads(self.record.read_text())
        self.assertEqual(
            set(record["native_sha256"]), {"vllm/_C.abi3.so", "vllm/vllm-rs"}
        )
        self.assertEqual(len(self.calls), 2)

    def test_missing_native_extension_fails_before_uninstall(self):
        self.make_wheel(with_native=False)
        with self.assertRaisesRegex(RuntimeError, "missing runtime artifacts"):
            source_wheel.install(self.wheel_dir, self.record)
        self.assertEqual((self.package / "_C.abi3.so").read_bytes(), b"old native")
        self.assertFalse(self.calls)

    def test_abi_mismatch_fails_before_uninstall(self):
        self.make_wheel()
        (self.wheel_dir / "build-abi.json").write_text(json.dumps({"abi": 2}))
        with self.assertRaisesRegex(RuntimeError, "ABI mismatch"):
            source_wheel.install(self.wheel_dir, self.record)
        self.assertFalse(self.calls)

    def test_compiler_abi_mismatch_fails_before_uninstall(self):
        self.make_wheel()
        (self.wheel_dir / "csrc-abi.json").write_text(json.dumps({"abi": 2}))
        with self.assertRaisesRegex(RuntimeError, "ABI mismatch"):
            source_wheel.install(self.wheel_dir, self.record)
        self.assertFalse(self.calls)

    def test_missing_jit_header_fails_before_uninstall(self):
        include = self.package / "third_party/deep_gemm/include"
        include.mkdir(parents=True)
        (include / "kernel.cuh").write_text("// required by JIT")
        self.make_wheel()
        with self.assertRaisesRegex(RuntimeError, "missing runtime artifacts"):
            source_wheel.install(self.wheel_dir, self.record)
        self.assertFalse(self.calls)

    def test_new_missing_dependency_fails_before_uninstall(self):
        self.make_wheel(requirements=["missing-kimi-test-package>=1"])
        with self.assertRaisesRegex(RuntimeError, "lacks new or changed dependencies"):
            source_wheel.install(self.wheel_dir, self.record)
        self.assertFalse(self.calls)

    def test_new_vllm_extra_dependency_fails_before_uninstall(self):
        self.make_wheel(
            requirements=['missing-kimi-test-package>=1; extra == "audio"'],
            extras=["audio"],
        )
        with self.assertRaisesRegex(RuntimeError, "lacks new or changed dependencies"):
            source_wheel.install(self.wheel_dir, self.record)
        self.assertFalse(self.calls)

    def test_unchanged_uninstalled_extra_does_not_block(self):
        req = 'missing-kimi-test-package>=1; extra == "audio"'
        self.assertFalse(source_wheel.new_missing_requirements([req], [req], ["audio"]))

    def test_new_dependency_extra_is_checked_transitively(self):
        def version(name):
            if name == "example-base":
                return "1.0"
            raise source_wheel.importlib.metadata.PackageNotFoundError(name)

        def requires(name):
            if name == "example-base":
                return ['example-extra>=2; extra == "cu13"']
            return []

        with (
            mock.patch.object(source_wheel.importlib.metadata, "version", version),
            mock.patch.object(source_wheel.importlib.metadata, "requires", requires),
        ):
            self.assertEqual(source_wheel.missing_requirements(["example-base"]), set())
            self.assertTrue(
                any(
                    "example-extra" in item
                    for item in source_wheel.missing_requirements(
                        ["example-base[cu13]"]
                    )
                )
            )

    def test_new_pip_conflict_is_rejected(self):
        self.make_wheel()
        with (
            mock.patch.object(
                source_wheel,
                "pip_check",
                side_effect=[
                    {"exit_code": 1, "output": "old conflict"},
                    {"exit_code": 1, "output": "old conflict\nnew conflict"},
                ],
            ),
            self.assertRaisesRegex(RuntimeError, "introduced dependency conflicts"),
        ):
            source_wheel.install(self.wheel_dir, self.record)

    def test_existing_vllm_pip_conflict_ignores_version_label(self):
        old = {"exit_code": 1, "output": "vllm 1.0 has requirement example"}
        new = {"exit_code": 1, "output": "vllm 2.0 has requirement example"}
        self.assertEqual(source_wheel.pip_problems(old), source_wheel.pip_problems(new))

    def test_new_root_cannot_hide_a_preexisting_missing_transitive_dependency(self):
        def version(name):
            if name in {"example-a", "example-c"}:
                return "1.0"
            raise source_wheel.importlib.metadata.PackageNotFoundError(name)

        def requires(name):
            return ["example-b>=1"] if name in {"example-a", "example-c"} else []

        with (
            mock.patch.object(source_wheel.importlib.metadata, "version", version),
            mock.patch.object(source_wheel.importlib.metadata, "requires", requires),
        ):
            missing = source_wheel.new_missing_requirements(
                ["example-a"], ["example-a", "example-c"]
            )
        self.assertIn("example-b>=1", missing)

    def test_direct_url_requires_matching_immutable_source(self):
        digest = "a" * 64
        requirement = source_wheel.Requirement(
            f"example @ https://example.org/example.whl#sha256={digest}"
        )
        info = {
            "url": "https://example.org/example.whl",
            "archive_info": {"hashes": {"sha256": digest}},
        }
        distribution = types.SimpleNamespace(read_text=lambda _: json.dumps(info))
        with mock.patch.object(
            source_wheel.importlib.metadata, "distribution", return_value=distribution
        ):
            self.assertTrue(source_wheel.direct_url_matches(requirement))
            info["archive_info"]["hashes"]["sha256"] = "b" * 64
            self.assertFalse(source_wheel.direct_url_matches(requirement))

    def test_direct_url_accepts_legacy_hash_but_rejects_disagreement(self):
        digest = "a" * 64
        requirement = source_wheel.Requirement(
            f"example @ https://example.org/example.whl#sha256={digest}"
        )
        info = {
            "url": "https://example.org/example.whl",
            "archive_info": {"hash": f"sha256={digest}"},
        }
        distribution = types.SimpleNamespace(read_text=lambda _: json.dumps(info))
        with mock.patch.object(
            source_wheel.importlib.metadata, "distribution", return_value=distribution
        ):
            self.assertTrue(source_wheel.direct_url_matches(requirement))
            info["archive_info"]["hashes"] = {"sha256": "b" * 64}
            self.assertFalse(source_wheel.direct_url_matches(requirement))

    def test_pip_check_rejects_execution_error_but_accepts_conflicts(self):
        result = types.SimpleNamespace(
            returncode=1, stdout="python3: No module named pip\n"
        )
        with mock.patch.object(source_wheel.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "did not complete successfully"):
                REAL_PIP_CHECK()
            result.stdout = "vllm 1.0 requires missing, which is not installed.\n"
            self.assertEqual(REAL_PIP_CHECK()["exit_code"], 1)


class WheelInputTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "vllm"
        self.package.mkdir()
        (self.package / "__init__.py").write_text("source")
        self.snapshot = self.root / "source.json"
        source_wheel.snapshot_source(self.package, self.snapshot)
        (self.package / "vllm-rs").write_bytes(b"new rust")
        self.rust_snapshot = self.root / "rust.json"
        source_wheel.snapshot_native(self.package, self.rust_snapshot)
        self.native = self.root / "native"
        self.final = self.root / "final"
        self.native.mkdir()
        self.final.mkdir()
        with zipfile.ZipFile(self.native / "vllm-1.whl", "w") as archive:
            archive.writestr("vllm/_new_C.abi3.so", b"new native")

    def final_wheel(self, *, source="source", native=True, extra_native=False):
        with zipfile.ZipFile(self.final / "vllm-1.whl", "w") as archive:
            archive.writestr("vllm/__init__.py", source)
            archive.writestr("vllm/vllm-rs", b"new rust")
            if native:
                archive.writestr("vllm/_new_C.abi3.so", b"new native")
            if extra_native:
                archive.writestr("vllm/libs/old.so", b"old binary")

    def test_verifies_new_native_extension_and_source(self):
        self.final_wheel()
        source_wheel.verify_wheel_inputs(
            self.snapshot, self.rust_snapshot, self.native, self.final
        )

    def test_rejects_missing_new_native_extension(self):
        self.final_wheel(native=False)
        with self.assertRaisesRegex(RuntimeError, "missing .*source/native"):
            source_wheel.verify_wheel_inputs(
                self.snapshot, self.rust_snapshot, self.native, self.final
            )

    def test_rejects_overwritten_python_source(self):
        self.final_wheel(source="generated")
        with self.assertRaisesRegex(RuntimeError, "differs from its source inputs"):
            source_wheel.verify_wheel_inputs(
                self.snapshot, self.rust_snapshot, self.native, self.final
            )

    def test_rejects_missing_source_configuration(self):
        (self.package / "config.json").write_text("{}")
        source_wheel.snapshot_source(self.package, self.snapshot)
        self.final_wheel()
        with self.assertRaisesRegex(RuntimeError, "missing .*source/native"):
            source_wheel.verify_wheel_inputs(
                self.snapshot, self.rust_snapshot, self.native, self.final
            )

    def test_rejects_missing_unusual_runtime_assets(self):
        for name in (
            "include/kernel.inl",
            "scripts/start.sh",
            "static/view.css",
            "new_file.unknown",
        ):
            path = self.package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("runtime source")
        source_wheel.snapshot_source(self.package, self.snapshot)
        snapshot = json.loads(self.snapshot.read_text())
        self.assertIn("vllm/include/kernel.inl", snapshot)
        self.assertIn("vllm/scripts/start.sh", snapshot)
        self.assertIn("vllm/static/view.css", snapshot)
        self.assertIn("vllm/new_file.unknown", snapshot)
        self.final_wheel()
        with self.assertRaisesRegex(RuntimeError, "missing .*source/native"):
            source_wheel.verify_wheel_inputs(
                self.snapshot, self.rust_snapshot, self.native, self.final
            )

    def test_generated_version_file_is_not_snapshotted(self):
        (self.package / "_version.py").write_text("stale")
        source_wheel.snapshot_source(self.package, self.snapshot)
        self.assertNotIn("vllm/_version.py", json.loads(self.snapshot.read_text()))

    def test_rejects_unproven_native_binary(self):
        self.final_wheel(extra_native=True)
        with self.assertRaisesRegex(RuntimeError, "differs from its source inputs"):
            source_wheel.verify_wheel_inputs(
                self.snapshot, self.rust_snapshot, self.native, self.final
            )

    def test_csrc_extraction_preserves_source_and_adds_native(self):
        setup = Path(__file__).resolve().parents[2] / "setup.py"
        module = ast.parse(setup.read_text())
        parent = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "precompiled_wheel_utils"
        )
        method = next(
            node
            for node in parent.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "extract_precompiled_and_patch_package"
        )
        parent.body = [method]
        isolated = ast.fix_missing_locations(ast.Module(body=[parent], type_ignores=[]))
        namespace = {
            "os": os,
            "re": re,
            "shutil": shutil,
            "PRECOMPILED_RUST_EXTENSION_MEMBER_REGEX": re.compile(
                r"vllm/_rust_.*\.so$"
            ),
        }
        exec(compile(isolated, str(setup), "exec"), namespace)
        extractor = namespace["precompiled_wheel_utils"]
        source = self.package / "third_party/triton_kernels/matmul.py"
        source.parent.mkdir(parents=True)
        source.write_text("my source")
        wheel = self.root / "csrc.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("vllm/third_party/triton_kernels/matmul.py", "generated")
            archive.writestr("vllm/_new_C.abi3.so", b"new native")
        old_cwd = Path.cwd()
        self.addCleanup(os.chdir, old_cwd)
        os.chdir(self.root)
        with mock.patch.dict(os.environ, {"VLLM_DOCKER_SOURCE_BUILD": "1"}):
            patch = extractor.extract_precompiled_and_patch_package(
                str(wheel), None, extract_extensions=True, extract_rust_frontend=False
            )
        self.assertEqual(source.read_text(), "my source")
        self.assertEqual((self.package / "_new_C.abi3.so").read_bytes(), b"new native")
        self.assertIn("_new_C.abi3.so", patch["vllm"])


if __name__ == "__main__":
    unittest.main()
