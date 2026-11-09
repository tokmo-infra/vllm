#!/usr/bin/env python3
"""Check the repository snapshot against the immutable Kimi K3.1 image."""

import hashlib
import json
from pathlib import Path


SOURCE = Path("/tmp/kimi-k31-source/vllm")
INSTALLED = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MANIFEST = Path("/tmp/kimi-k31-source/source-manifest.json")
MARKER = Path("/tmp/kimi-k31-verified.json")
EXPECTED_MANIFEST = "5fcea0f86e9ef27fced08c0d7dce8d3f6b7c89f89bacfd4a749bfe4c322822ff"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if checksum(MANIFEST) != EXPECTED_MANIFEST:
        raise RuntimeError("源码基线清单已被修改，拒绝使用现用镜像打包")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["files"]
    actual = {}
    for path in SOURCE.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"源码中不允许符号链接: {path}")
        if path.is_file():
            actual[path.relative_to(SOURCE).as_posix()] = path
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise RuntimeError(f"源码文件清单不一致: 缺少={missing}, 多出={extra}")

    image_matches = 0
    for name, record in expected.items():
        path = actual[name]
        if path.stat().st_size != record["size"] or checksum(path) != record["sha256"]:
            raise RuntimeError(f"源码与镜像基线不一致: {name}")
        if record["in_image"]:
            installed = INSTALLED / name
            if not installed.is_file() or checksum(installed) != record["sha256"]:
                raise RuntimeError(f"源码与镜像内安装文件不一致: {name}")
            image_matches += 1

    proof = {
        "base_image_id": manifest["base_image_id"],
        "source_commit": manifest["source_commit"],
        "source_files": len(expected),
        "image_files_verified": image_matches,
        "source_manifest_sha256": checksum(MANIFEST),
    }
    MARKER.write_text(json.dumps(proof, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(proof, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
