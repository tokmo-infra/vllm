# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coordinate Kimi prompt-cache reuse and billing across prefill engines."""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import regex as re

from vllm.logger import init_logger

logger = init_logger(__name__)

# Keys are cumulative native block hashes, including the original cache salt,
# multimodal identifiers and lookahead. Redis owns the clock for all workers.
# A competing or retried writer can publish each live logical block only once.
_COMMIT = """
local hits, unit, requested = tonumber(ARGV[1]), tonumber(ARGV[2]), ARGV[3]
local tier, written, short, long = nil, 0, 0, 0
for i, key in ipairs(KEYS) do
    local value = redis.call('GET', key)
    if i <= hits and (not value or value ~= ARGV[4+i]) then break end
    if value then
        tier = string.match(value, '^(%d+):')
        redis.call('EXPIRE', key, tonumber(tier))
    else
        tier = tier or requested
        redis.call('SET', key, tier .. ':' .. ARGV[4], 'EX', tonumber(tier))
        written = written + unit
        if tier == '300' then short = short + unit end
        if tier == '3600' then long = long + unit end
    end
end
return {written, short, long}
"""


@lru_cache(maxsize=1)
def _settings() -> tuple[str, bytes] | None:
    path = os.getenv("VLLM_K31_CACHE_LEASE_URL_FILE")
    if not path:
        return None
    namespace = os.environ["VLLM_K31_CACHE_LEASE_NAMESPACE"]
    if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,128}", namespace):
        raise ValueError("Invalid K31 cache lease namespace")
    return Path(path).read_text().strip(), ("k31:{" + namespace + "}:").encode()


def leases_enabled() -> bool:
    return _settings() is not None


@lru_cache(maxsize=1)
def _reader():
    from redis import Redis

    settings = _settings()
    assert settings is not None
    return Redis.from_url(settings[0], socket_connect_timeout=0.3, socket_timeout=0.3)


@lru_cache(maxsize=1)
def _writer():
    from redis.asyncio import Redis

    settings = _settings()
    assert settings is not None
    return Redis.from_url(settings[0], socket_connect_timeout=0.3, socket_timeout=0.3)


def ttl_seconds(value: object) -> int:
    if isinstance(value, str) and (
        m := re.fullmatch(r"([1-9][0-9]{0,5})([smhd])", value)
    ):
        return int(m[1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m[2]]
    return 300


@dataclass
class K31CachePlan:
    keys: list[bytes]
    observed: list[bytes]
    unit: int
    seconds: int

    @property
    def read_limit(self) -> int:
        return len(self.observed) * self.unit

    async def publish(self, read: int, written: int) -> tuple[int, int, int]:
        # The prefill scheduler supplies actual readable and retained token
        # counts. A short tail never creates a logical full-block lease.
        blocks = min(len(self.keys), (read + written) // self.unit)
        if not blocks:
            return 0, 0, 0
        hits = min(read // self.unit, blocks, len(self.observed))
        from redis.exceptions import RedisError

        try:
            conn = _writer()
            result = await conn.register_script(_COMMIT)(
                keys=self.keys[:blocks],
                args=[
                    hits,
                    self.unit,
                    self.seconds,
                    uuid4().hex,
                    *self.observed[:hits],
                ],
            )
        except (RedisError, OSError):
            logger.warning_once("K31 cache leases unavailable during publication")
            return 0, 0, 0
        return int(result[0]), int(result[1]), int(result[2])


def prepare_k31_cache(hashes: Sequence[bytes], unit: int, seconds: int) -> K31CachePlan:
    settings = _settings()
    assert settings is not None
    prefix = settings[1] + str(unit).encode() + b":"
    keys = [prefix + h for h in hashes]
    active: list[bytes] = []
    if keys:
        from redis.exceptions import RedisError

        try:
            for value in _reader().mget(keys):
                if value is None:
                    break
                active.append(value)
        except (RedisError, OSError):
            logger.warning_once("K31 cache leases unavailable during lookup")
            # No historical reads or billable writes until shared state returns.
            keys = []
    return K31CachePlan(keys, active, unit, seconds)
