# SPDX-License-Identifier: Apache-2.0
"""Optional per-API listeners, separate from engine/DP-rank addressing."""

import os
import socket
from pathlib import Path


def create_private_listener(index: int, count: int) -> socket.socket | None:
    directory = os.environ.get("VLLM_API_PRIVATE_SOCKET_DIR", "")
    if directory:
        if not 0 <= index < count:
            raise ValueError("Invalid API process index")
        root = Path(directory)
        if not root.is_absolute() or root.is_symlink():
            raise ValueError("API socket directory must be an absolute owned path")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = root / f"api-{index}.sock"
        if len(os.fsencode(path)) >= 108:
            raise ValueError("API Unix socket path exceeds the platform limit")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            # Existing paths are startup errors; never unlink another API's
            # live socket as an implicit recovery action.
            sock.bind(str(path))
            sock.setblocking(False)
            return sock
        except BaseException:
            sock.close()
            raise
    configured = os.environ.get("VLLM_API_PRIVATE_PORT_BASE", "")
    if not configured:
        return None
    base = int(configured)
    if not 0 <= index < count or not 1 <= base <= 65535 - count + 1:
        raise ValueError("Invalid API private listener port range or process index")
    host = os.environ.get("VLLM_API_PRIVATE_HOST", "127.0.0.1")
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, base + index))
        sock.setblocking(False)
        return sock
    except BaseException:
        sock.close()
        raise
