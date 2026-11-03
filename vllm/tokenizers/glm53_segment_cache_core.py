# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact native added-token-delimited BPE reuse for the pinned GLM5.3 tokenizer."""

import hashlib
import json
import sqlite3
import sys
import threading
import time
from array import array
from collections import OrderedDict
from pathlib import Path
from typing import Any

import regex as re


class UnsupportedRequest(ValueError):
    pass


def literal_pattern(literals):
    """Factor literal prefixes; preserve leftmost-longest added-token matches."""
    trie: dict[str | None, Any] = {}
    for literal in literals:
        node = trie
        for char in literal:
            node = node.setdefault(char, {})
        node[None] = {}

    def emit(node):
        choices = [
            re.escape(char) + emit(child)
            for char, child in sorted((c, n) for c, n in node.items() if c is not None)
        ]
        body = "|".join(choices)
        if len(choices) > 1:
            body = "(?:" + body + ")"
        return "(?:" + body + ")?" if None in node and body else body

    return re.compile(emit(trie))


class SegmentCache:
    def __init__(
        self,
        tokenizer,
        tokenizer_path,
        max_bytes=32 * 1024**3,
        shared_store=None,
        generation="",
    ):
        self.tokenizer = tokenizer
        self.backend = tokenizer.backend_tokenizer
        raw = (
            Path(tokenizer_path).read_bytes()
            if tokenizer_path
            else self.backend.to_str().encode()
        )
        config = json.loads(raw)
        normalizer = config.get("normalizer")
        if normalizer not in (None, {"type": "Sequence", "normalizers": []}):
            raise ValueError(
                "Only the pinned identity-normalized tokenizer is supported"
            )
        if config["model"]["type"] != "BPE":
            raise ValueError("Unsupported BPE configuration")
        # Every admitted request has add_special_tokens=False, so the native
        # tokenizer post-processor is disabled for both full and segment encode.
        added = config["added_tokens"]
        if any(
            any(t.get(k, False) for k in ("lstrip", "rstrip", "single_word"))
            for t in added
        ):
            raise ValueError(
                "Added-token whitespace/word-boundary behavior is unsupported"
            )
        self.specials = {
            t["content"]: t["id"]
            for t in added
            if t.get("special") and not t.get("normalized", False)
        }
        if not self.specials:
            raise ValueError("No proven added-token boundaries")
        # Match ALL added tokens first: a longer ordinary added token must shield
        # special-token substrings inside it, just as in the native tokenizer.
        literals = sorted({t["content"] for t in added}, key=lambda x: (-len(x), x))
        self.pattern = literal_pattern(literals)
        self.backend.no_padding()
        self.backend.no_truncation()
        self.backend.encode_special_tokens = False
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        for literal, token_id in self.specials.items():
            if self.backend.encode(literal, add_special_tokens=False).ids != [token_id]:
                raise ValueError("Added-token boundary is not token-exact")
        self.max_bytes = max_bytes
        self.lock = threading.Lock()
        self.entries = OrderedDict()
        self.bytes = 0
        self.stats = {}
        self.shared_store = shared_store
        self.generation = generation
        self.reset_version = 0

    def _segment(self, namespace, text):
        key = (namespace, text)
        hash(key)  # Hash the fresh long string before entering the shared lock.
        with self.lock:
            generation = self.generation
            reset_version = self.reset_version
            entry = self.entries.get(key)
            if entry is not None:
                self.entries.move_to_end(key)
                return entry[0], True
        ids = None
        shared_hit = False
        if self.shared_store is not None:
            try:
                packed = self.shared_store.get(generation, namespace, text)
                if packed is not None:
                    ids = array("I")
                    ids.frombytes(packed)
                    shared_hit = True
                self._record_shared("shared_hits" if shared_hit else "shared_misses")
            except (OSError, ValueError, sqlite3.Error):
                self._record_shared("shared_errors")
        if ids is None:
            ids = array("I", self.backend.encode(text, add_special_tokens=False).ids)
            if self.shared_store is not None:
                try:
                    self.shared_store.put(generation, namespace, text, ids.tobytes())
                except (OSError, ValueError, sqlite3.Error):
                    self._record_shared("shared_errors")
        size = (
            sys.getsizeof(text)
            + sys.getsizeof(ids)
            + sys.getsizeof(key)
            + sys.getsizeof(namespace)
            + 256
        )
        if size <= self.max_bytes:
            with self.lock:
                if reset_version == self.reset_version and key not in self.entries:
                    while self.entries and self.bytes + size > self.max_bytes:
                        _, (_, old_size) = self.entries.popitem(last=False)
                        self.bytes -= old_size
                        self.stats["evictions"] = self.stats.get("evictions", 0) + 1
                    self.entries[key] = (ids, size)
                    self.bytes += size
        return ids, shared_hit

    def _record_shared(self, name):
        with self.lock:
            self.stats[name] = self.stats.get(name, 0) + 1

    def encode(self, text, namespace=""):
        start = time.monotonic()
        result = []
        counts: dict[str, int | float] = {
            "reused_tokens": 0,
            "encoded_tokens": 0,
            "special_tokens": 0,
            "hit_segments": 0,
            "miss_segments": 0,
        }
        cursor = 0
        for match in self.pattern.finditer(text):
            literal = match.group()
            if literal not in self.specials:
                continue
            if match.start() > cursor:
                ids, hit = self._segment(namespace, text[cursor : match.start()])
                result.extend(ids)
                counts["reused_tokens" if hit else "encoded_tokens"] += len(ids)
                counts["hit_segments" if hit else "miss_segments"] += 1
            result.append(self.specials[literal])
            counts["special_tokens"] += 1
            cursor = match.end()
        if cursor < len(text):
            ids, hit = self._segment(namespace, text[cursor:])
            result.extend(ids)
            counts["reused_tokens" if hit else "encoded_tokens"] += len(ids)
            counts["hit_segments" if hit else "miss_segments"] += 1
        counts["encode_seconds"] = time.monotonic() - start
        with self.lock:
            for k, v in counts.items():
                self.stats[k] = self.stats.get(k, 0) + v
            self.stats["encoded_requests"] = self.stats.get("encoded_requests", 0) + 1
        return result, counts

    def snapshot(self):
        with self.lock:
            return {
                **self.stats,
                "cache_bytes": self.bytes,
                "cache_max_bytes": self.max_bytes,
                "cache_entries": len(self.entries),
            }

    def reset(self, generation=None):
        with self.lock:
            self.entries.clear()
            self.bytes = 0
            self.stats.clear()
            self.reset_version += 1
            if generation is not None:
                self.generation = generation
