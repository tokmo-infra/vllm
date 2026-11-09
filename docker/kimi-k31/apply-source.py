#!/usr/bin/env python3
"""Apply repository changes on top of the installed image package."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil


SOURCE = Path("/tmp/kimi-k31-source/vllm")
INSTALLED = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MANIFEST = Path("/tmp/kimi-k31-source/source-manifest.json")
MARKER = Path("/opt/kimi-k31-build/source-update.json")
EXPECTED_MANIFEST = "5fcea0f86e9ef27fced08c0d7dce8d3f6b7c89f89bacfd4a749bfe4c322822ff"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_child(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (
        not name
        or relative.is_absolute()
        or ".." in relative.parts
        or str(relative) != name
    ):
        raise RuntimeError(f"不安全的源码路径: {name!r}")
    current = root
    if current.is_symlink():
        raise RuntimeError(f"不允许符号链接: {current}")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"不允许符号链接: {current}")
    return current


def source_files(root: Path) -> dict[str, Path]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"找不到源码目录或目录是符号链接: {root}")
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            if path.is_symlink():
                raise RuntimeError(f"源码中不允许符号链接: {path}")
            if path.is_file():
                result[path.relative_to(root).as_posix()] = path
            elif not path.is_dir():
                raise RuntimeError(f"源码中不允许特殊文件: {path}")
    return result


def clear_bytecode(installed: Path, name: str) -> None:
    if not name.endswith(".py"):
        return
    source = checked_child(installed, name)
    cache = source.parent / "__pycache__"
    if cache.is_symlink():
        raise RuntimeError(f"不允许符号链接: {cache}")
    if cache.is_dir():
        prefix = source.stem + "."
        for path in cache.iterdir():
            if path.name.startswith(prefix) and path.name.endswith(".pyc"):
                if path.is_symlink() or not path.is_file():
                    raise RuntimeError(f"不安全的 Python 缓存: {path}")
                path.unlink()


def sync_source(source: Path, installed: Path, manifest: dict) -> dict:
    """Apply source additions, edits and removals without deleting image extras."""
    if not installed.is_dir() or installed.is_symlink():
        raise RuntimeError(f"找不到镜像内 vllm 目录: {installed}")
    baseline = manifest["files"]
    current = source_files(source)
    for name in baseline:
        checked_child(source, name)
        checked_child(installed, name)

    changes = {"added": [], "modified": [], "removed": []}
    digest = hashlib.sha256()
    for name, path in sorted(current.items()):
        file_digest = checksum(path)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(file_digest))
        original = baseline.get(name)
        if original and original["sha256"] == file_digest:
            continue
        target = checked_child(installed, name)
        if target.exists() and not target.is_file():
            raise RuntimeError(f"镜像中的目标不是普通文件: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        clear_bytecode(installed, name)
        shutil.copy2(path, target)
        changes["modified" if original else "added"].append(name)

    for name, original in sorted(baseline.items()):
        if name in current or not original["in_image"]:
            continue
        target = checked_child(installed, name)
        if target.exists():
            if not target.is_file():
                raise RuntimeError(f"镜像中的目标不是普通文件: {target}")
            clear_bytecode(installed, name)
            target.unlink()
        changes["removed"].append(name)

    return {"source_sha256": digest.hexdigest(), **changes}


def main() -> None:
    if checksum(MANIFEST) != EXPECTED_MANIFEST:
        raise RuntimeError("镜像源码基线清单已被修改，无法安全应用改动")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    changes = sync_source(SOURCE, INSTALLED, manifest)
    report = {
        "base_image_id": manifest["base_image_id"],
        "base_source_commit": manifest["source_commit"],
        **changes,
    }
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    counts = {name: len(report[name]) for name in ("added", "modified", "removed")}
    print(json.dumps({"source_sha256": report["source_sha256"], **counts}), flush=True)


if __name__ == "__main__":
    main()
