# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import uuid
from dataclasses import field
from typing import Any, Literal, get_args

from vllm.config.utils import config, replace
from vllm.utils.hashing import safe_hash

KVProducer = Literal["kv_producer", "kv_both"]
KVConsumer = Literal["kv_consumer", "kv_both"]
KVRole = Literal[KVProducer, KVConsumer]

QWEN38_MTP_CONTEXT_STORE_NAMESPACE = "qwen38-mtp-context-v1/"
QWEN38_NVFP4_MTP_STORE_NAMESPACE = QWEN38_MTP_CONTEXT_STORE_NAMESPACE + "nvfp4-main-v1/"


def hisparse_host_pool_gib(
    kv_transfer_config: "KVTransferConfig | None",
) -> float | None:
    if kv_transfer_config is None:
        return None
    if kv_transfer_config.kv_connector == "MultiConnector":
        connectors = kv_transfer_config.kv_connector_extra_config.get("connectors", [])
    else:
        connectors = [
            {
                "kv_connector": kv_transfer_config.kv_connector,
                "kv_connector_extra_config": (
                    kv_transfer_config.kv_connector_extra_config
                ),
            }
        ]
    entries = [
        connector.get("kv_connector_extra_config", {})
        for connector in connectors
        if connector.get("kv_connector") == "HiSparseConnector"
    ]
    if len(entries) > 1:
        raise ValueError("Only one HiSparseConnector may be configured")
    if not entries:
        return None
    host_pool_gib = entries[0].get("host_pool_gib")
    if host_pool_gib is None:
        raise ValueError("HiSparseConnector requires host_pool_gib")
    host_pool_gib = float(host_pool_gib)
    if host_pool_gib <= 0:
        raise ValueError("HiSparseConnector host_pool_gib must be positive")
    return host_pool_gib


def kv_buffer_device_default_factory() -> str:
    from vllm.platforms import current_platform

    return current_platform.device_type


@config
class KVTransferConfig:
    """Configuration for distributed KV cache transfer."""

    kv_connector: str | None = None
    """The KV connector for vLLM to transmit KV caches between vLLM instances.
    """

    engine_id: str | None = None
    """The engine id for KV transfers."""

    kv_buffer_device: str = field(default_factory=kv_buffer_device_default_factory)
    """The device used by kv connector to buffer the KV cache. Choices are
    'cuda', 'cpu' and 'xpu'."""

    kv_role: KVRole | None = None
    """Whether this vLLM instance produces, consumes KV cache, or both. Choices
    are 'kv_producer', 'kv_consumer', and 'kv_both'."""

    kv_port: int = 14579
    """The KV connector port, used to build distributed connection."""

    kv_connector_extra_config: dict[str, Any] = field(default_factory=dict)
    """any extra config that the connector may need."""

    kv_connector_module_path: str | None = None
    """The Python module path to dynamically load the KV connector from.
    Only supported in V1."""

    enable_permute_local_kv: bool = False
    """Experiment feature flag to enable HND to NHD KV Transfer"""

    kv_load_failure_policy: Literal["recompute", "fail"] = "fail"
    """Policy for handling KV cache load failures.
    'recompute': reschedule the request to recompute failed blocks
    'fail': immediately fail the request with an error finish reason (default)"""

    def compute_hash(self) -> str:
        """WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self) -> None:
        if self.engine_id is None:
            self.engine_id = str(uuid.uuid4())

        if self.kv_role is not None and self.kv_role not in get_args(KVRole):
            raise ValueError(
                f"Unsupported kv_role: {self.kv_role}. "
                f"Supported roles are {get_args(KVRole)}"
            )

        if self.kv_connector is not None and self.kv_role is None:
            raise ValueError(
                "Please specify kv_role when kv_connector "
                f"is set, supported roles are {get_args(KVRole)}"
            )

    @property
    def is_kv_transfer_instance(self) -> bool:
        return self.kv_connector is not None and self.kv_role in get_args(KVRole)

    @property
    def is_kv_producer(self) -> bool:
        return self.kv_connector is not None and self.kv_role in get_args(KVProducer)

    @property
    def is_kv_consumer(self) -> bool:
        return self.kv_connector is not None and self.kv_role in get_args(KVConsumer)

    def get_from_extra_config(self, key, default) -> Any:
        return self.kv_connector_extra_config.get(key, default)

    def has_connector(self, connector_name: str) -> bool:
        """Whether ``connector_name`` is configured, directly or in MultiConnector."""
        if self.kv_connector == connector_name:
            return True
        return self.kv_connector == "MultiConnector" and any(
            child.get("kv_connector") == connector_name
            for child in self.kv_connector_extra_config.get("connectors", [])
        )

    def is_dedicated_mooncake_producer(self) -> bool:
        """Recognize a P2P-only producer, optionally backed by shared Store."""
        if self.kv_connector == "MooncakeConnector":
            return self.kv_role == "kv_producer" and not self.kv_connector_module_path
        if self.kv_connector != "MultiConnector" or self.kv_connector_module_path:
            return False
        children = self.get_from_extra_config("connectors", [])
        if any(child.get("kv_connector_module_path") for child in children):
            return False
        p2p = [c for c in children if c.get("kv_connector") == "MooncakeConnector"]
        store = [
            c for c in children if c.get("kv_connector") == "MooncakeStoreConnector"
        ]
        return (
            len(p2p) == 1
            and p2p[0].get("kv_role") == "kv_producer"
            and len(store) <= 1
            and len(children) == len(p2p) + len(store)
            and self.kv_role in ("kv_producer", "kv_both")
        )

    def uses_qwen38_mtp_context_store_prefix(
        self, namespace: str = QWEN38_MTP_CONTEXT_STORE_NAMESPACE
    ) -> bool:
        """Whether a Store uses the reserved prompt-tail provenance namespace."""
        store_configs = [self.kv_connector_extra_config]
        if self.kv_connector == "MultiConnector":
            store_configs = [
                child.get("kv_connector_extra_config", {})
                for child in self.get_from_extra_config("connectors", [])
                if child.get("kv_connector") == "MooncakeStoreConnector"
            ]
        elif self.kv_connector != "MooncakeStoreConnector":
            return False
        return any(
            isinstance(prefix := config.get("cache_prefix"), str)
            and prefix.startswith(namespace)
            for config in store_configs
        )

    def with_qwen38_nvfp4_mtp_store_prefix(
        self, draft_dtype: str
    ) -> "KVTransferConfig":
        """Isolate mixed-KV MTP provenance without changing the supplied prefix."""
        namespace = f"{QWEN38_NVFP4_MTP_STORE_NAMESPACE}{draft_dtype}/"
        if not self.has_connector("MooncakeStoreConnector"):
            return self
        if self.kv_connector != "MultiConnector":
            raise ValueError("Qwen3.8 NVFP4 MTP Store requires a Mooncake P/D pair")
        children = self.get_from_extra_config("connectors", [])
        updated = []
        for child in children:
            if child.get("kv_connector") == "MooncakeStoreConnector":
                extra = child.get("kv_connector_extra_config", {})
                prefix = extra.get("cache_prefix", "")
                if not isinstance(prefix, str):
                    raise ValueError("Mooncake Store cache_prefix must be a string")
                if prefix.startswith(QWEN38_NVFP4_MTP_STORE_NAMESPACE) and (
                    not prefix.startswith(namespace) or prefix == namespace
                ):
                    raise ValueError("Incompatible Qwen3.8 NVFP4 MTP Store namespace")
                if not prefix.startswith(namespace):
                    suffix = f"named/{prefix}" if prefix else "default"
                    child = {
                        **child,
                        "kv_connector_extra_config": {
                            **extra,
                            "cache_prefix": namespace + suffix,
                        },
                    }
            updated.append(child)
        if children == updated:
            return self
        return replace(
            self,
            kv_connector_extra_config={
                **self.kv_connector_extra_config,
                "connectors": updated,
            },
        )
