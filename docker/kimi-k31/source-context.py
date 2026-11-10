#!/usr/bin/env python3
"""Reject untracked generated files before assembling a source-built wheel."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

GENERATED_PATHS = (
    "vllm/vllm_flash_attn",
    "vllm/third_party/triton_kernels",
    "vllm/third_party/deep_gemm",
    "vllm/third_party/fmha_sm100",
    "vllm/third_party/tml_fa4",
    "vllm/third_party/flashmla/flash_mla_interface.py",
)


def git_output(root: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = (
            os.fsdecode(exc.stderr).strip()
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr
            else str(exc)
        )
        raise RuntimeError(
            "Cannot read the Git index; run at the repository root with "
            ".git mounted (for example, /workspace/.git): " + detail
        ) from exc
    return result.stdout


def tracked_paths(root: Path) -> set[str]:
    top = os.fsdecode(git_output(root, "rev-parse", "--show-toplevel")).rstrip("\n")
    if Path(top).resolve() != root.resolve():
        raise RuntimeError("Run source-context.py from the repository root.")
    # A tracked symlink in a generated directory can point elsewhere in the
    # context, so its targets may be tracked outside GENERATED_PATHS.
    output = git_output(root, "ls-files", "--cached", "--full-name", "-z")
    return {os.fsdecode(path) for path in output.split(b"\0") if path}


def generated_files(root: Path):
    root = root.resolve()
    visited_dirs = set()

    def walk(path: Path):
        if path.is_symlink():
            yield path.relative_to(root).as_posix()
            target = path.resolve()
            # Only follow targets that are themselves in the build context.
            if target.is_relative_to(root):
                yield from walk(target)
        elif path.is_dir():
            if path in visited_dirs:
                return
            visited_dirs.add(path)
            with os.scandir(path) as children:
                for child in children:
                    yield from walk(Path(child.path))
        elif path.exists():
            yield path.relative_to(root).as_posix()

    for name in GENERATED_PATHS:
        yield from walk(root / name)


def main() -> int:
    if len(sys.argv) != 1:
        print("usage: source-context.py", file=sys.stderr)
        return 2

    root = Path.cwd()
    try:
        tracked = tracked_paths(root)
        untracked = sorted(set(generated_files(root)) - tracked)
    except (OSError, RuntimeError) as exc:
        print(f"source-context: {exc}", file=sys.stderr)
        return 1

    if untracked:
        print("Untracked generated files found in the build context:", file=sys.stderr)
        for path in untracked:
            print(f"  {shlex.quote(path)}", file=sys.stderr)
        print(
            "Remove stale generated files from the build context, or explicitly "
            "include intended files in the Git index with: git add -f -- <path>",
            file=sys.stderr,
        )
        return 1

    print("Source context contains no untracked generated files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
