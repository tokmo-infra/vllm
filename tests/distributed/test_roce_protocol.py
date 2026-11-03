# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the proxy's actual epoch state without a GPU or an RDMA NIC."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def protocol_binary(tmp_path_factory):
    source = Path(__file__).with_name("roce_protocol_test.cpp")
    root = source.parents[2]
    binary = tmp_path_factory.mktemp("roce_protocol") / "test"
    subprocess.run(
        [
            os.environ.get("CXX", "c++"),
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        timeout=60,
    )
    return binary


@pytest.mark.parametrize(
    "case",
    [
        "receive_before_local_producer",
        "descriptor_after_data",
        "zero_length_next_epoch",
        "invalid_reuse_and_messages",
        "changing_replays",
    ],
)
def test_independent_rdma_directions(protocol_binary, case):
    subprocess.run([str(protocol_binary), case], check=True, timeout=30)
