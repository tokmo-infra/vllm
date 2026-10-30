# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restore the pinned image sources and apply the archived integration patches."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("vllm", "flashinfer", "router"):
        component = manifest[name]
        dest = args.output / name
        dest.mkdir()
        subprocess.run(["git", "init", "--quiet", str(dest)], check=True)
        commands = [
            ["remote", "add", "origin", component["repository"]],
            ["fetch", "--depth=1", "origin", component["commit"]],
            ["checkout", "--quiet", "--detach", "FETCH_HEAD"],
        ]
        commands.extend(
            ["apply", str(root / patch["path"])] for patch in component["patches"]
        )
        for command in commands:
            subprocess.run(["git", "-C", str(dest), *command], check=True)
    print(f"Prepared sources in {args.output.resolve()}")


if __name__ == "__main__":
    main()
