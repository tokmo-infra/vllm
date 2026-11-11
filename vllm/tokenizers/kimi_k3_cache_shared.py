# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One container-local, on-demand CPU L2 for Kimi text-piece tokenization.

All API processes must use the same private directory and total byte budget
(at most 16 GiB). Keys are hashable, pickleable ``(namespace, allow_special,
original_text)`` tuples; the caller supplies a stable tokenizer namespace.
Epoch strings are part of the exact cache key, not an ordered global clock.
Use a fresh epoch for invalidation, never reuse one. Old-epoch writes cannot
overwrite new-epoch entries; old entries share the budget until LRU eviction.

The budget conservatively charges retained entry objects, not interpreter RSS
or transient RPC copies. IDs stay packed bytes, never a replicated 16 GiB L2.
The detached server outlives API clients and stops with the container. No GPU
or vLLM imports are required, including when executed directly by file path.
Only the server creates Prometheus L2 metrics; livesum gauges thus count the
shared capacity once. Pass PROMETHEUS_MULTIPROC_DIR through to the server.
"""

import argparse
import contextlib
import fcntl
import os
import select
import stat
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from multiprocessing.managers import BaseManager
from pathlib import Path

_START_TIMEOUT = 10.0
_MAX_BYTES = 16 * 1024**3
_METHODS = ("get", "put", "stats")
_CACHE_FACTORY = "cache"


def _deep_size(value):
    # Deliberately double-charge aliases; include tuple-contained namespaces.
    size = sys.getsizeof(value)
    if isinstance(value, (tuple, frozenset)):
        size += sum(_deep_size(item) for item in value)
    elif not isinstance(value, (str, bytes, bool, int, float, type(None))):
        raise TypeError(
            "namespace must contain immutable scalar/tuple/frozenset values"
        )
    return size


def _validate_key(key, epoch):
    if (
        type(key) is not tuple
        or len(key) != 3
        or type(key[1]) is not bool
        or type(key[2]) is not str
    ):
        raise TypeError("key must be (stable_namespace, bool, original_text: str)")
    if type(epoch) is not str:
        raise TypeError("epoch must be a string shared by all API processes")
    hash(key)


class _LRU:
    def __init__(self, max_bytes):
        from prometheus_client import Counter, Gauge

        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.evictions = 0
        self.lock = threading.Lock()
        self.byte_metric = Gauge(
            "vllm:tokenizer_l2_cache_bytes",
            "Charged shared CPU text-piece cache bytes",
            multiprocess_mode="livesum",
        )
        self.entry_metric = Gauge(
            "vllm:tokenizer_l2_cache_entries",
            "Shared CPU text-piece cache entries",
            multiprocess_mode="livesum",
        )
        Gauge(
            "vllm:tokenizer_l2_cache_max_bytes",
            "Total shared CPU text-piece cache budget",
            multiprocess_mode="livesum",
        ).set(max_bytes)
        self.eviction_metric = Counter(
            "vllm:tokenizer_l2_cache_evictions",
            "Shared CPU text-piece LRU evictions",
        )
        self._update_metrics()

    def _update_metrics(self):
        self.byte_metric.set(self.bytes)
        self.entry_metric.set(len(self.entries))

    def get(self, key, epoch):
        _validate_key(key, epoch)
        with self.lock:
            item = self.entries.get((epoch, key))
            if item is None:
                return None
            self.entries.move_to_end((epoch, key))
            return item[0], item[1]

    def put(self, key, packed_ids, special_count, epoch):
        _validate_key(key, epoch)
        if type(packed_ids) is not bytes or len(packed_ids) % 4:
            raise ValueError("packed_ids must be bytes containing packed uint32 IDs")
        if (
            type(special_count) is not int
            or not 0 <= special_count <= len(packed_ids) // 4
        ):
            raise ValueError("special_count must be between zero and the token count")
        full_key = (epoch, key)
        # Includes key/text/namespace, bytes, integer and value tuple plus ample
        # per-entry headroom for OrderedDict/hash-table nodes and allocator slack.
        charge = (
            _deep_size(full_key)
            + sys.getsizeof(packed_ids)
            + sys.getsizeof(special_count)
            + sys.getsizeof((packed_ids, special_count, 0))
            + 1024
        )
        with self.lock:
            previous = self.entries.pop(full_key, None)
            if previous is not None:
                self.bytes -= previous[2]
            if charge <= self.max_bytes:
                while self.entries and self.bytes + charge > self.max_bytes:
                    _, (_, _, old_charge) = self.entries.popitem(last=False)
                    self.bytes -= old_charge
                    self.evictions += 1
                    self.eviction_metric.inc()
                self.entries[full_key] = (packed_ids, special_count, charge)
                self.bytes += charge
            self._update_metrics()

    def stats(self):
        with self.lock:
            return dict(
                bytes=self.bytes,
                max_bytes=self.max_bytes,
                entries=len(self.entries),
                evictions=self.evictions,
            )


class _Manager(BaseManager):
    pass


_Manager.register(_CACHE_FACTORY, exposed=_METHODS)


def _private_file(path, flags):
    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    info = os.fstat(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        os.close(fd)
        raise PermissionError(f"Shared cache file must be owner-only (0600): {path}")
    return fd


@contextlib.contextmanager
def _startup_lock(path):
    fd = _private_file(path / "startup.lock", os.O_CREAT | os.O_RDWR)
    try:
        deadline = time.monotonic() + _START_TIMEOUT
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Shared cache startup lock timed out") from None
                time.sleep(0.02)
        yield
    finally:
        os.close(fd)


def _read_auth(path):
    fd = _private_file(path / "auth", os.O_RDONLY)
    with os.fdopen(fd, "rb") as stream:
        auth = stream.read(33)
    if len(auth) != 32:
        raise RuntimeError("Invalid shared cache authentication file")
    return auth


def _start_server(path, max_bytes, lease):
    (path / "cache.sock").unlink(missing_ok=True)
    auth_fd = _private_file(path / "auth", os.O_CREAT | os.O_WRONLY)
    with os.fdopen(auth_fd, "wb") as stream:
        stream.truncate()
        stream.write(os.urandom(32))
    read_fd, pipe_write_fd = os.pipe()
    write_fd: int | None = pipe_write_fd
    process = None
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--serve",
                str(path),
                str(max_bytes),
                str(lease),
                str(pipe_write_fd),
            ],
            pass_fds=(lease, pipe_write_fd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        os.close(pipe_write_fd)
        write_fd = None
        ready, _, _ = select.select([read_fd], [], [], _START_TIMEOUT)
        if not ready or os.read(read_fd, 1) != b"1":
            raise RuntimeError("Shared cache server failed to start within 10 seconds")
        return process
    except BaseException:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)


class SharedTextPieceCache:
    """Thread-safe client of one shared service, reconnecting after fork.

    Startup errors and broken RPCs raise explicitly; calls never silently start
    a replacement service. A new client may restart a dead service under lock.

    Args:
        path: Container-local private directory (0700), shared by all APIs.
        max_bytes: Total L2 budget, identical for every client, at most 16 GiB.

    """

    def __init__(self, path: str, max_bytes: int):
        if type(max_bytes) is not int or not 0 < max_bytes <= _MAX_BYTES:
            raise ValueError("max_bytes must be a positive integer <= 16 GiB")
        self.path = Path(os.path.abspath(path))
        self.max_bytes = max_bytes
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._manager: _Manager | None = None
        self._proxy = None
        self._server_process = None
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.path.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise PermissionError("Shared cache directory must be owner-only (0700)")
        with _startup_lock(self.path):
            lease = _private_file(self.path / "server.lock", os.O_CREAT | os.O_RDWR)
            try:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    self._server_process = _start_server(self.path, max_bytes, lease)
                self._connect()
            finally:
                # Do not LOCK_UN: the child inherited this open-file-description.
                os.close(lease)

    def _connect(self):
        try:
            manager = _Manager(
                address=str(self.path / "cache.sock"), authkey=_read_auth(self.path)
            )
            manager.connect()
            proxy = getattr(manager, _CACHE_FACTORY)()
            # BaseProxy shares thread-local connections by address. Explicitly
            # replace that connection even after a plain os.fork(), which does
            # not run multiprocessing's after-fork hooks.
            proxy._connect()
            if proxy.stats()["max_bytes"] != self.max_bytes:
                raise ValueError("All shared cache clients must use the same max_bytes")
            self._manager, self._proxy = manager, proxy
        except (OSError, EOFError) as exc:
            raise RuntimeError("Could not connect to shared text-piece cache") from exc

    def _call(self, method, *args):
        if self._pid != os.getpid():
            # Never acquire a lock inherited from a possibly running parent thread.
            self._lock = threading.Lock()
            self._manager = self._proxy = None
            self._server_process = None
            self._pid = os.getpid()
        with self._lock:
            if self._proxy is None:
                self._connect()
            try:
                return getattr(self._proxy, method)(*args)
            except (OSError, EOFError) as exc:
                raise RuntimeError("Shared text-piece cache RPC failed") from exc

    def get(self, key, epoch):
        """Return an immutable (packed uint32 bytes, special_count) or None."""
        return self._call("get", key, epoch)

    def put(self, key, packed_ids: bytes, special_count: int, epoch) -> None:
        self._call("put", key, packed_ids, special_count, epoch)

    def stats(self):
        return self._call("stats")


def _serve(path, max_bytes, lease, ready_fd):
    os.umask(0o077)
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR") and (path / "server.pid").exists():
        from prometheus_client.multiprocess import mark_process_dead

        # The inherited lease proves the previous server is gone. Remove its
        # livesum gauge files before publishing a replacement server's budget.
        pid_fd = _private_file(path / "server.pid", os.O_RDONLY)
        with os.fdopen(pid_fd) as stream:
            mark_process_dead(int(stream.read()))
    cache = _LRU(max_bytes)
    _Manager.register(_CACHE_FACTORY, callable=lambda: cache, exposed=_METHODS)
    manager = _Manager(address=str(path / "cache.sock"), authkey=_read_auth(path))
    server = manager.get_server()
    os.chmod(path / "cache.sock", 0o600)
    pid_fd = _private_file(path / "server.pid", os.O_CREAT | os.O_WRONLY)
    with os.fdopen(pid_fd, "w") as stream:
        stream.truncate()
        stream.write(str(os.getpid()))
    os.write(ready_fd, b"1")
    os.close(ready_fd)
    try:
        server.serve_forever()
    finally:
        os.close(lease)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("path", type=Path)
    parser.add_argument("max_bytes", type=int)
    parser.add_argument("lease", type=int)
    parser.add_argument("ready_fd", type=int)
    args = parser.parse_args()
    _serve(args.path, args.max_bytes, args.lease, args.ready_fd)
