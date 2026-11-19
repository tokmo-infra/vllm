# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport contract for GLM's mixed-format NVFP4 MLA cache."""

import json
from collections.abc import Sequence

import torch

from vllm.config.glm_mla_kv import GLM_NVFP4_MLA, GLM_NVFP4_ROW_BYTES
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
    group_kernel_blocks,
)

def is_glm_nvfp4_mla(spec: KVCacheSpec) -> bool:
    return getattr(spec, "cache_dtype_str", None) == GLM_NVFP4_MLA


def group_layer_specs(group: KVCacheGroupSpec) -> dict[str, KVCacheSpec]:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return {name: spec.kv_cache_specs.get(name, spec) for name in group.layer_names}
    return dict.fromkeys(group.layer_names, spec)


def has_glm_nvfp4_mla(groups: Sequence[KVCacheGroupSpec]) -> bool:
    return any(
        is_glm_nvfp4_mla(spec)
        for group in groups
        for spec in group_layer_specs(group).values()
    )


def glm_layout_manifest(groups: Sequence[KVCacheGroupSpec]) -> str:
    """Ordered block-table groups with a format signature for every layer."""

    def layer_format(spec: KVCacheSpec) -> tuple:
        common = (type(spec).__name__, spec.block_size, spec.dcp_sharded)
        if isinstance(spec, AttentionSpec):
            return common + (
                getattr(spec, "cache_dtype_str", None),
                str(spec.dtype),
                spec.kv_quant_mode.name,
                spec.num_heads,
                spec.head_size,
                spec.head_size_v,
                str(spec.tokens_per_state),
                spec.state_content_size_bytes,
                getattr(spec, "sliding_window", None),
            )
        if isinstance(spec, MambaSpec):
            return common + (
                spec.mamba_type.name,
                spec.mamba_cache_mode,
                spec.shapes,
                tuple(map(str, spec.dtypes)),
                spec.tp_replicated,
            )
        raise ValueError(f"{GLM_NVFP4_MLA} does not support {type(spec).__name__}")

    return json.dumps(
        [
            (
                group.role.value,
                group.is_eagle_group,
                {
                    name: layer_format(spec)
                    for name, spec in group_layer_specs(group).items()
                },
            )
            for group in groups
        ],
        sort_keys=True,
        separators=(",", ":"),
    )


def glm_nvfp4_page_view(
    cache: torch.Tensor, spec: KVCacheSpec, num_blocks: int
) -> torch.Tensor:
    """Validate the runtime byte layout and expose each manager page as a view."""
    if (
        not isinstance(spec, MLAAttentionSpec)
        or spec.num_heads != 1
        or spec.head_size != 512
        or spec.head_size_v != 0
        or spec.tokens_per_state != 1
        or spec.state_content_size_bytes != GLM_NVFP4_ROW_BYTES
        or spec.dtype not in (torch.uint8, torch.int8)
        or cache.dtype != spec.dtype
        or cache.ndim != 4
        or cache.shape[1] != 1
        or cache.shape[0] <= 0
        or cache.shape[2] <= 0
        or cache.shape[-1] != GLM_NVFP4_ROW_BYTES
        or num_blocks <= 0
        or cache.shape[0] % num_blocks != 0
        or cache.shape[0] * cache.shape[2] != num_blocks * spec.block_size
    ):
        raise ValueError(
            f"{GLM_NVFP4_MLA} requires [blocks, 1, tokens, 304] byte views: "
            f"got {tuple(cache.shape)}, {cache.dtype}"
        )
    grouped = group_kernel_blocks(cache, num_blocks)
    page_bytes = spec.block_size * GLM_NVFP4_ROW_BYTES
    if not grouped[0].is_contiguous() or grouped.stride(0) < page_bytes:
        raise ValueError(f"{GLM_NVFP4_MLA} requires contiguous, non-overlapping pages")
    return grouped.view(num_blocks, spec.block_size, GLM_NVFP4_ROW_BYTES)


def glm_transfer_page_view(
    cache: torch.Tensor, spec: KVCacheSpec, num_blocks: int
) -> torch.Tensor:
    """Expose one contiguous logical payload per manager block, excluding padding."""
    if is_glm_nvfp4_mla(spec):
        return glm_nvfp4_page_view(cache, spec, num_blocks)

    if isinstance(spec, AttentionSpec):
        page_bytes = spec.unpadded_page_size_bytes
        if (
            num_blocks <= 0
            or cache.ndim != 4
            or cache.dtype != spec.dtype
            or cache.shape[1] != spec.num_heads
            or cache.shape[-1] * cache.element_size() != spec.state_content_size_bytes
            or cache.shape[0] * cache.shape[2] != num_blocks * spec.num_states
            or cache.shape[0] % num_blocks != 0
            or (
                cache.shape[0] != num_blocks
                and (spec.tokens_per_state != 1 or spec.num_heads != 1)
            )
        ):
            raise ValueError(f"Mooncake {GLM_NVFP4_MLA}: unsupported attention view")
        cache = group_kernel_blocks(cache, num_blocks)
    elif isinstance(spec, MambaSpec):
        # The runner exposes one raw byte view for all recurrent states.
        page_bytes = spec.real_page_size_bytes
        if cache.dtype not in (torch.int8, torch.uint8):
            raise ValueError(f"Mooncake {GLM_NVFP4_MLA} requires a raw Mamba byte view")
    else:
        raise ValueError(f"Mooncake {GLM_NVFP4_MLA}: unsupported KV spec {spec}")

    if (
        num_blocks <= 0
        or page_bytes <= 0
        or cache.ndim < 2
        or cache.shape[0] != num_blocks
        or not cache[0].is_contiguous()
        or cache[0].nbytes != page_bytes
        or cache.stride(0) * cache.element_size() < page_bytes
    ):
        raise ValueError(
            f"Mooncake {GLM_NVFP4_MLA} requires contiguous per-layer pages"
        )
    return cache
