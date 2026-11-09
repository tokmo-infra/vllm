#!/usr/bin/env python3
"""Prevent native/build changes from being silently ignored."""

import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/tmp/kimi-k31-source")
MANIFEST = ROOT / "native-input-manifest.json"
EXPECTED_MANIFEST = "2713e25ee207826934fc37d5b7ba9a94507e3cf75ed88d17e08c11f6e867ba85"
DIRECTORIES = ("csrc", "cmake", "rust", "requirements", "tools")
TOP_LEVEL = (
    "CMakeLists.txt",
    "setup.py",
    "pyproject.toml",
    "MANIFEST.in",
    "rust-toolchain.toml",
)
NATIVE_SUFFIXES = {
    ".a", ".c", ".cc", ".cpp", ".cubin", ".cu", ".cuh", ".cxx",
    ".dll", ".dylib", ".fatbin", ".h", ".hip", ".hpp", ".inl",
    ".o", ".ptx", ".rs", ".so",
}


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files_under(root: Path, directory: str, native_only: bool = False) -> dict:
    base = root / directory
    if not base.is_dir() or base.is_symlink():
        raise RuntimeError(f"构建源码目录不存在或是符号链接: {base}")
    result = {}
    for parent, dirs, files in os.walk(base, followlinks=False):
        for name in dirs + files:
            path = Path(parent) / name
            if path.is_symlink():
                raise RuntimeError(f"构建源码中不允许符号链接: {path}")
            if path.is_file() and (not native_only or path.suffix in NATIVE_SUFFIXES):
                result[path.relative_to(root).as_posix()] = checksum(path)
            elif not path.is_file() and not path.is_dir():
                raise RuntimeError(f"构建源码中不允许特殊文件: {path}")
    return result


def find_changes(root: Path, expected: dict[str, str]) -> list[str]:
    actual = {}
    for directory in DIRECTORIES:
        actual.update(files_under(root, directory))
    actual.update(files_under(root, "vllm", native_only=True))
    for name in TOP_LEVEL:
        path = root / name
        if path.is_symlink():
            raise RuntimeError(f"构建源码中不允许符号链接: {path}")
        if path.is_file():
            actual[name] = checksum(path)
    return sorted(name for name in set(expected) | set(actual)
                  if expected.get(name) != actual.get(name))


def main() -> None:
    if checksum(MANIFEST) != EXPECTED_MANIFEST:
        raise RuntimeError("原生扩展构建输入基线清单已被修改")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    changes = find_changes(ROOT, manifest["files"])
    if changes:
        raise RuntimeError(
            "检测到原生扩展或构建依赖变更；当前 Python 增量打包入口不会编译它们，"
            "需要先准备对应的原生构建流程，不能静默忽略：\n" + "\n".join(changes)
        )
    print("原生扩展及构建依赖输入与现用镜像源码基线一致", flush=True)


if __name__ == "__main__":
    main()
