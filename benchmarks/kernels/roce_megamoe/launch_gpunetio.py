# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch with torchrun on exactly two equal hosts; each GPU uses its paired NIC."""

import argparse
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--nics", required=True, help="Comma-separated NICs in local GPU order"
    )
    parser.add_argument("--gid", type=int, default=3)
    parser.add_argument("--bootstrap-port", type=int, default=29750)
    parser.add_argument("--tokens", default="768,936,1024,1536,4096")
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--qps", default="1,4")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    nics = args.nics.split(",")
    if world != 2 * local_world or len(nics) != local_world:
        raise ValueError(
            "Exactly two equal hosts and one NIC per local GPU are required"
        )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    port = args.bootstrap_port
    for tokens in map(int, args.tokens.split(",")):
        for qps in map(int, args.qps.split(",")):
            output = args.output_directory / f"rank{rank}-t{tokens}-qp{qps}.jsonl"
            with output.open("x") as stream:
                subprocess.run(
                    [
                        str(args.binary.resolve()),
                        str(rank),
                        str(local),
                        nics[local],
                        str(args.gid),
                        f"{os.environ['MASTER_ADDR']}:{port}",
                        str(tokens),
                        str(args.hidden),
                        str(args.top_k),
                        str(qps),
                    ],
                    stdout=stream,
                    check=True,
                    timeout=180,
                )
            port += 1


if __name__ == "__main__":
    main()
