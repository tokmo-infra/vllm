# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install the reviewed Python overlay onto an exact vLLM release."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_overlay(
    source_root: Path,
    package_root: Path,
    manifest: dict,
    installed_version: str,
) -> list[str]:
    """Validate the complete source/base pair before writing any file."""
    if installed_version != manifest["base_vllm_version"]:
        raise ValueError(f"vLLM release mismatch: {installed_version}")
    pending: list[tuple[Path, Path, str]] = []
    seen: set[str] = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[:1] != ("vllm",)
            or relative.suffix != ".py"
            or str(relative) in seen
        ):
            raise ValueError(f"Invalid overlay path: {relative}")
        seen.add(str(relative))
        source = source_root / relative
        destination = package_root.joinpath(*relative.parts[1:])
        if source.is_symlink() or sha256(source) != entry["sha256"]:
            raise ValueError(f"Overlay source fingerprint mismatch: {relative}")
        installed_hash = sha256(destination) if destination.exists() else None
        if destination.is_symlink() or installed_hash not in (
            entry["base_sha256"],
            entry["sha256"],
        ):
            raise ValueError(f"Base fingerprint mismatch: {relative}")
        pending.append((source, destination, str(relative)))
    for source, destination, _ in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return [relative for _, _, relative in pending]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    release_path = args.source / "integrations/glm53nvfp4/release.json"
    manifest = json.loads(release_path.read_text())
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("The base image has no installed vLLM package")
    version = importlib.metadata.version("vllm")
    paths = install_overlay(args.source, Path(spec.origin).parent, manifest, version)
    shutil.copyfile(release_path, "/opt/glm53-nvfp4-release.json")
    print(json.dumps({"base_vllm": version, "patched_files": paths}))


if __name__ == "__main__":
    main()
