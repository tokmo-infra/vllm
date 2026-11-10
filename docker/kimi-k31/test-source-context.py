#!/usr/bin/env python3
"""Exercise source-context checks using temporary Git repositories."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("source-context.py").resolve()
GENERATED_FILES = (
    "vllm/vllm_flash_attn/cute/nested/kernel.py",
    "vllm/third_party/triton_kernels/nested/kernel.py",
    "vllm/third_party/deep_gemm/include/nested/kernel.cuh",
    "vllm/third_party/fmha_sm100/csrc/nested/kernel.cu",
    "vllm/third_party/tml_fa4/nested/kernel.py",
    "vllm/third_party/flashmla/flash_mla_interface.py",
)


class SourceContextTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        self.git("init", "-q")
        self.write(
            ".gitignore",
            "vllm/vllm_flash_attn/*\n"
            "!vllm/vllm_flash_attn/__init__.py\n"
            "!vllm/vllm_flash_attn/flash_attn_interface.py\n"
            "vllm/third_party/triton_kernels/*\n"
            "vllm/third_party/deep_gemm/\n"
            "vllm/third_party/fmha_sm100/\n"
            "vllm/third_party/tml_fa4/\n"
            "vllm/third_party/flashmla/flash_mla_interface.py\n",
        )
        self.existing = (
            "vllm/vllm_flash_attn/.gitkeep",
            "vllm/vllm_flash_attn/__init__.py",
            "vllm/vllm_flash_attn/flash_attn_interface.py",
        )
        for name in self.existing:
            self.write(name, "existing source\n")
        self.git("add", "-f", "--", ".gitignore", *self.existing)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def git(self, *args):
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            check=True,
        )

    def run_cli(self, *args, cwd=None):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=cwd or self.root,
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_existing_tracked_and_unrelated_files_are_allowed(self):
        self.write("vllm/third_party/flashmla/__init__.py", "unrelated\n")
        self.write("vllm/third_party/other/kernel.py", "unrelated\n")
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_untracked_generated_files_fail_in_every_scope_without_removal(self):
        for name in GENERATED_FILES:
            self.write(name, "old generated content\n")
        result = self.run_cli()
        self.assertEqual(result.returncode, 1, result.stderr)
        for name in GENERATED_FILES:
            with self.subTest(name=name):
                self.assertIn(name, result.stderr)
                self.assertEqual(
                    (self.root / name).read_text(encoding="utf-8"),
                    "old generated content\n",
                )
        self.assertIn("Remove stale generated files", result.stderr)
        self.assertIn("git add -f -- <path>", result.stderr)

    def test_force_added_generated_files_and_later_edits_are_allowed(self):
        for name in GENERATED_FILES:
            self.write(name, "explicit source\n")
        self.git("add", "-f", "--", *GENERATED_FILES)
        self.write(GENERATED_FILES[0], "unstaged source edit\n")
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_removed_index_entry_is_untracked_even_if_file_remains(self):
        name = GENERATED_FILES[1]
        self.write(name, "generated\n")
        self.git("add", "-f", "--", name)
        self.git("rm", "--cached", "-f", "--", name)
        result = self.run_cli()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(name, result.stderr)
        self.assertTrue((self.root / name).is_file())

    def test_untracked_directory_symlink_is_reported(self):
        target = self.root / "external-sources"
        target.mkdir()
        link = self.root / "vllm/third_party/deep_gemm"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
        result = self.run_cli()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("vllm/third_party/deep_gemm", result.stderr)
        self.assertTrue(link.is_symlink())

    def test_tracked_symlink_does_not_hide_untracked_context_files(self):
        tracked = "external-sources/tracked.py"
        untracked = "external-sources/nested/untracked.py"
        self.write(tracked, "tracked\n")
        self.write(untracked, "untracked\n")
        link = self.root / "vllm/third_party/deep_gemm"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(self.root / "external-sources", target_is_directory=True)
        self.git("add", "-f", "--", str(link.relative_to(self.root)), tracked)
        result = self.run_cli()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(untracked, result.stderr)
        self.git("add", "-f", "--", untracked)
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unusual_filenames_are_compared_as_nul_delimited_git_paths(self):
        name = "vllm/third_party/tml_fa4/nested/a file\nwith newline.py"
        self.write(name, "explicit source\n")
        self.git("add", "-f", "--", name)
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_git_index_fails_closed(self):
        shutil.rmtree(self.root / ".git")
        result = self.run_cli()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("Cannot read the Git index", result.stderr)
        self.assertIn(".git mounted", result.stderr)

    def test_read_only_index_can_be_inspected(self):
        index = self.root / ".git/index"
        original = index.read_bytes()
        index.chmod(0o444)
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(index.read_bytes(), original)

    def test_requires_repository_root_and_no_arguments(self):
        result = self.run_cli(cwd=self.root / "vllm")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("repository root", result.stderr)
        result = self.run_cli("unexpected")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("usage: source-context.py", result.stderr)


if __name__ == "__main__":
    unittest.main()
