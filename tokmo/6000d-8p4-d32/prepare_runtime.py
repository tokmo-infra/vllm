# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialize SHA-guarded read-only runtime mounts from the pinned 540155 image."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--site-packages",
        type=Path,
        default=Path("/usr/local/lib/python3.12/dist-packages"),
    )
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    patches = json.loads((here / "runtime/image-patches.json").read_text())
    prepared = {}
    for patch in patches:
        raw = (args.site_packages / patch["source"]).read_bytes()
        if digest(raw) != patch["original_sha256"]:
            raise ValueError("Pinned image source differs: " + patch["source"])
        text = raw.decode()
        for edit in reversed(patch["edits"]):
            start, end = edit["start"], edit["end"]
            if text[start:end] != edit["old"]:
                raise ValueError("Source edit context differs: " + patch["source"])
            text = text[:start] + edit["new"] + text[end:]
        data = text.encode()
        if digest(data) != patch["patched_sha256"]:
            raise ValueError("Prepared runtime SHA differs: " + patch["output"])
        prepared[patch["output"]] = data
    manifest = json.loads((here / "runtime/tokenizer-cache-manifest.json").read_text())
    for name in ("glm53_native_cache.py", "glm53_segment_cache_core.py"):
        data = (repo / "vllm/tokenizers" / name).read_bytes()
        if digest(data) != manifest["files_sha256"][name]:
            raise ValueError("Validated tokenizer module differs: " + name)
        prepared[name] = data
    for name, data in prepared.items():
        destination = args.output / name
        if destination.exists() and destination.read_bytes() != data:
            raise FileExistsError(
                "Refusing to overwrite changed runtime: " + str(destination)
            )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in prepared.items():
        destination = args.output / name
        if not destination.exists():
            destination.write_bytes(data)
    print(
        json.dumps(
            {
                "status": "prepared",
                "files_sha256": {name: digest(data) for name, data in prepared.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
