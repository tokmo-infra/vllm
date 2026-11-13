# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from dataclasses import dataclass
from typing import cast

from vllm.config import VllmConfig
from vllm.config.kv_transfer import hisparse_host_pool_gib
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.hisparse.runtime import (
    ResolvedHiSparseConfig,
    get_hisparse_host_block_stride,
    use_shared_hisparse_host_pool,
)
from vllm.v1.hisparse.types import ACTIVE_TAIL_PAGES
from vllm.v1.kv_cache_interface import (
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

logger = init_logger(__name__)

HISPARSE_HOT_SUFFIX = ".hisparse_hot"
HISPARSE_RESIDENT_SUFFIX = ".hisparse_resident"


def is_hisparse_mla_spec(spec: KVCacheSpec) -> bool:
    """Dense DSpark layers retain their ordinary device cache."""
    return isinstance(spec, MLAAttentionSpec) and not (
        spec.is_draft
        and spec.topk_buffer_width is None
        and spec.cache_role is SparseCacheRole.SPARSE
    )


@dataclass(frozen=True)
class HiSparseLayout:
    source_group: KVCacheGroupSpec
    device_groups: list[KVCacheGroupSpec]
    host_num_blocks: int
    host_block_stride: int
    shared_host_pool: bool


def get_hisparse_kv_cache_groups(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
) -> list[KVCacheGroupSpec] | None:
    """The groups HiSparse allocates: the host source group, then the GPU
    groups sharing one device pool. Derived resident/hot specs in
    `kv_cache_spec` (see `resolve_hisparse_specs`) are laid out again
    from the attention specs they belong to."""
    attention_config = getattr(vllm_config, "attention_config", None)
    if attention_config is None or attention_config.hisparse_config is None:
        return None

    mla_specs: dict[str, KVCacheSpec] = {
        name: spec for name, spec in kv_cache_spec.items() if is_hisparse_mla_spec(spec)
    }
    other_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if not is_hisparse_mla_spec(spec)
        and not isinstance(spec, (HiSparseResidentSpec, HiSparseHotSpec))
    }
    if not mla_specs:
        return None

    from vllm.v1.core.kv_cache_planning import get_kv_cache_groups

    regular_groups = (
        get_kv_cache_groups(vllm_config, other_specs) if other_specs else []
    )
    return _lay_out_hisparse_groups(vllm_config, mla_specs, regular_groups)


def get_hisparse_host_pool_bytes(vllm_config: VllmConfig) -> int:
    host_pool_gib = hisparse_host_pool_gib(vllm_config.kv_transfer_config)
    if host_pool_gib is None:
        raise ValueError("HiSparse requires HiSparseConnector with host_pool_gib")
    return int(host_pool_gib * 2**30)


def _partition_hisparse_specs(
    specs: dict[str, KVCacheSpec],
) -> tuple[dict[str, MLAAttentionSpec], dict[str, MLAAttentionSpec]]:
    if any(
        isinstance(spec, MLAAttentionSpec) and spec.model_version == "deepseek_v4"
        for spec in specs.values()
    ):
        raise ValueError("HiSparse does not support DeepSeek V4.")
    if not all(isinstance(spec, MLAAttentionSpec) for spec in specs.values()):
        raise ValueError("HiSparse requires its first cache group to contain MLA only.")

    source_specs = {
        name: spec
        for name, spec in specs.items()
        if isinstance(spec, MLAAttentionSpec)
        and spec.cache_role is SparseCacheRole.SPARSE
    }
    indexer_specs = {
        name: spec
        for name, spec in specs.items()
        if isinstance(spec, MLAAttentionSpec)
        and spec.cache_role is SparseCacheRole.INDEXER
    }
    if not source_specs or not indexer_specs:
        raise ValueError("HiSparse requires sparse-MLA and indexer cache specs.")
    return source_specs, indexer_specs


def _lay_out_hisparse_groups(
    vllm_config: VllmConfig,
    mla_specs: dict[str, KVCacheSpec],
    regular_groups: list[KVCacheGroupSpec],
) -> list[KVCacheGroupSpec]:
    source_specs, indexer_specs = _partition_hisparse_specs(mla_specs)
    block_sizes = {spec.block_size for spec in source_specs.values()}
    if len(block_sizes) != 1:
        raise ValueError("HiSparse requires one resolved GPU block size.")
    gpu_block_size = block_sizes.pop()

    config = ResolvedHiSparseConfig.from_vllm_config(
        vllm_config,
        max(
            spec.topk_buffer_width
            if spec.topk_buffer_width is not None
            else vllm_config.model_config.hf_config.index_topk
            for spec in source_specs.values()
        ),
        gpu_block_size,
    )
    assert config is not None
    if vllm_config.parallel_config.decode_context_parallel_size > 1:
        return _lay_out_dcp_groups(source_specs, indexer_specs, regular_groups, config)
    indexer_group_spec = UniformTypeKVCacheSpecs.from_specs(
        cast(dict[str, KVCacheSpec], indexer_specs)
    )
    assert indexer_group_spec is not None
    indexer_group = KVCacheGroupSpec(
        list(indexer_specs),
        indexer_group_spec,
        enable_kv_transfer=True,
        role=KVCacheGroupRole.HISPARSE_INDEXER,
    )

    indexer_page = sum(spec.page_size_bytes for spec in indexer_specs.values())
    hot_blocks_per_request = cdiv(config.device_buffer_size, gpu_block_size)
    hot_units: list[list[tuple[str, MLAAttentionSpec]]] = []
    for layer_name, layer_spec in source_specs.items():
        if layer_spec.is_index_group_leader or not hot_units:
            hot_units.append([])
        hot_units[-1].append((f"{layer_name}{HISPARSE_HOT_SUFFIX}", layer_spec))

    resident_groups: list[KVCacheGroupSpec] = []
    hot_groups: list[KVCacheGroupSpec] = []

    def append_hot_group(layers: list[tuple[str, MLAAttentionSpec]]) -> None:
        page_sizes = {spec.page_size_bytes for _, spec in layers}
        if len(page_sizes) != 1:
            raise ValueError(
                "HiSparse hot-cache groups require one page size, got "
                f"{sorted(page_sizes)}."
            )
        page_size = page_sizes.pop()
        names = [name for name, _ in layers]
        resident_groups.append(
            KVCacheGroupSpec(
                [
                    name[: -len(HISPARSE_HOT_SUFFIX)] + HISPARSE_RESIDENT_SUFFIX
                    for name in names
                ],
                HiSparseResidentSpec(
                    block_size=gpu_block_size,
                    page_size=page_size,
                    dcp_sharded=layers[0][1].dcp_sharded,
                ),
                enable_kv_transfer=False,
            )
        )
        hot_groups.append(
            KVCacheGroupSpec(
                names,
                HiSparseHotSpec(
                    block_size=gpu_block_size,
                    page_size=page_size,
                    blocks_per_request=hot_blocks_per_request,
                ),
                enable_kv_transfer=False,
            )
        )

    current: list[tuple[str, MLAAttentionSpec]] = []
    current_page = 0
    for unit in hot_units:
        unit_page = sum(spec.page_size_bytes for _, spec in unit)
        if current and current_page + unit_page > indexer_page:
            append_hot_group(current)
            current = []
            current_page = 0
        current.extend(unit)
        current_page += unit_page
    if current:
        append_hot_group(current)

    source_group_spec = UniformTypeKVCacheSpecs.from_specs(
        cast(dict[str, KVCacheSpec], source_specs)
    )
    assert source_group_spec is not None
    source_group = KVCacheGroupSpec(
        list(source_specs),
        source_group_spec,
        role=KVCacheGroupRole.HISPARSE_SOURCE,
        host_resident=True,
        enable_kv_transfer=True,
    )
    return [
        source_group,
        indexer_group,
        *resident_groups,
        *hot_groups,
        *regular_groups,
    ]


def _lay_out_dcp_groups(
    sources: dict[str, MLAAttentionSpec],
    indexers: dict[str, MLAAttentionSpec],
    regular: list[KVCacheGroupSpec],
    config: ResolvedHiSparseConfig,
) -> list[KVCacheGroupSpec]:
    """Pack hot rows into shared-pool pages sized for the hybrid state cache.

    Source pages stay small for tail writes. Indexer and hot pages amortize a
    GPU lease over many rows, avoiding one large SSM-sized lease per 64 rows.
    Resident and hot views retain identical per-layer offsets in that lease.
    """
    source = next(iter(sources.values()))
    block = source.block_size
    if len({s.page_size_bytes for s in sources.values()}) != 1:
        raise ValueError("DCP HiSparse requires uniform sparse MLA row formats")
    index_page = sum(s.page_size_bytes for s in indexers.values())
    anchor = max((g.kv_cache_spec.page_size_bytes for g in regular), default=index_page)
    factor = 1 << max(cdiv(anchor, index_page) - 1, 0).bit_length()
    index_block = next(iter(indexers.values())).block_size
    indexers = {
        name: s.copy_with_new_block_size(index_block * factor)
        for name, s in indexers.items()
    }
    index_spec = UniformTypeKVCacheSpecs.from_specs(
        cast(dict[str, KVCacheSpec], indexers)
    )
    source_spec = UniformTypeKVCacheSpecs.from_specs(
        cast(dict[str, KVCacheSpec], sources)
    )
    assert index_spec is not None and source_spec is not None
    hot_factor = max(index_spec.page_size_bytes // source_spec.page_size_bytes, 1)
    hot_block = block * hot_factor
    page = source.page_size_bytes * hot_factor
    # Flat attention addresses need whole source-kernel rows between leases,
    # not an entire hot page (which needlessly enlarges the pool's LCM).
    alignment = source.page_size_bytes
    return [
        KVCacheGroupSpec(
            list(sources),
            source_spec,
            host_resident=True,
            role=KVCacheGroupRole.HISPARSE_SOURCE,
        ),
        KVCacheGroupSpec(
            list(indexers), index_spec, role=KVCacheGroupRole.HISPARSE_INDEXER
        ),
        KVCacheGroupSpec(
            [name + HISPARSE_RESIDENT_SUFFIX for name in sources],
            HiSparseResidentSpec(
                block_size=block,
                page_size=page,
                dcp_sharded=True,
                block_stride_alignment=alignment,
            ),
            enable_kv_transfer=False,
        ),
        KVCacheGroupSpec(
            [name + HISPARSE_HOT_SUFFIX for name in sources],
            HiSparseHotSpec(
                block_size=hot_block,
                page_size=page,
                blocks_per_request=cdiv(config.device_buffer_size, hot_block),
                block_stride_alignment=alignment,
            ),
            enable_kv_transfer=False,
        ),
        *regular,
    ]


def create_hisparse_layout(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroupSpec],
    host_budget: int,
) -> HiSparseLayout:
    """Size the host pool for groups laid out by `get_hisparse_kv_cache_groups`."""
    (source_group,) = [group for group in groups if group.host_resident]
    shared_host_pool = use_shared_hisparse_host_pool(vllm_config)
    host_block_stride = get_hisparse_host_block_stride(
        source_group.kv_cache_spec.page_size_bytes,
        use_shared_host_pool=shared_host_pool,
    )
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    # Each DCP worker owns one disjoint shard of the logical host budget.
    host_num_blocks = host_budget // dcp_size // host_block_stride
    if host_num_blocks <= 0:
        raise ValueError("HiSparse has no allocatable host blocks.")
    # Every computed page needs a host block, so one request at max_model_len
    # must fit alongside the pool's null block and a copy-on-write tail.
    gpu_block_size = source_group.kv_cache_spec.block_size
    min_host_blocks = (
        cdiv(vllm_config.model_config.max_model_len, gpu_block_size * dcp_size) + 2
    )
    if host_num_blocks < min_host_blocks:
        raise ValueError(
            f"HiSparse host pool has {host_num_blocks} blocks but max_model_len "
            f"needs {min_host_blocks}; increase host_pool_gib."
        )

    return HiSparseLayout(
        source_group=source_group,
        device_groups=[group for group in groups if not group.host_resident],
        host_num_blocks=host_num_blocks,
        host_block_stride=host_block_stride,
        shared_host_pool=shared_host_pool,
    )


def _build_hisparse_kv_cache_tensors(
    kv_cache_groups: list[KVCacheGroupSpec],
    num_blocks: int,
    size: int,
    layout: KVCacheLayout,
    bytes_per_block: int,
    *,
    host_resident: bool = False,
) -> list[KVCacheTensor]:
    interleaved_block_stride = bytes_per_block if layout.is_block_outermost else None
    tensors: list[KVCacheTensor] = []
    for group in kv_cache_groups:
        group_spec = group.kv_cache_spec
        layers_by_spec: defaultdict[KVCacheSpec, list[str]] = defaultdict(list)
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            for layer_name, spec in group_spec.kv_cache_specs.items():
                layers_by_spec[spec].append(layer_name)
        elif group.layer_names:
            layers_by_spec[group_spec].extend(group.layer_names)

        byte_offset = 0
        for spec, layer_names in layers_by_spec.items():
            if isinstance(spec, (HiSparseHotSpec, HiSparseResidentSpec)):
                if not layout.is_block_outermost:
                    raise ValueError("HiSparse requires a block-outermost KV layout.")
                layer_stride = spec.page_size_bytes
                block_stride = bytes_per_block
            else:
                layer_stride, block_stride, _, _, _ = compute_layout_strides(
                    spec,
                    num_blocks,
                    len(layer_names),
                    layout,
                    fixed_strides=(None, interleaved_block_stride, None, None, None),
                )
            offset = (
                byte_offset
                * max(layer_stride, spec.page_size_bytes)
                // spec.page_size_bytes
            )
            tensors.append(
                KVCacheTensor(
                    size=size,
                    layers=layer_names,
                    layer_stride=layer_stride,
                    block_stride=block_stride,
                    offset=offset,
                    host_resident=host_resident,
                )
            )
            byte_offset += len(layer_names) * spec.page_size_bytes
    return tensors


def get_hisparse_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
    host_budget: int,
) -> KVCacheConfig:
    from vllm.v1.core.kv_cache_planning import (
        _get_kv_cache_bytes_per_block,
        may_override_num_blocks,
        validate_kv_cache_layout,
    )

    hisparse_layout = create_hisparse_layout(vllm_config, kv_cache_groups, host_budget)
    device_groups = hisparse_layout.device_groups
    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    validate_kv_cache_layout(layout, device_groups)
    bytes_per_block = _get_kv_cache_bytes_per_block(device_groups, layout)
    num_blocks = may_override_num_blocks(
        vllm_config, available_memory // bytes_per_block
    )
    size = bytes_per_block * num_blocks
    kv_cache_tensors = _build_hisparse_kv_cache_tensors(
        device_groups, num_blocks, size, layout, bytes_per_block
    )

    host_groups = [hisparse_layout.source_group]
    host_layout = KVCacheLayout.LBNHC
    validate_kv_cache_layout(host_layout, host_groups)
    host_bytes_per_block = _get_kv_cache_bytes_per_block(host_groups, host_layout)
    host_size = host_bytes_per_block * hisparse_layout.host_num_blocks
    kv_cache_tensors[:0] = _build_hisparse_kv_cache_tensors(
        host_groups,
        hisparse_layout.host_num_blocks,
        host_size,
        host_layout,
        host_bytes_per_block,
        host_resident=True,
    )
    logger.info_once(
        "HiSparse HMA: %.1f GiB host source (%d blocks), %.1f GiB shared "
        "GPU indexer/resident/hot pool (%d blocks).",
        host_size / 2**30,
        hisparse_layout.host_num_blocks,
        size / 2**30,
        num_blocks,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=[*host_groups, *device_groups],
        hisparse_host_num_blocks=hisparse_layout.host_num_blocks,
        hisparse_host_block_stride=hisparse_layout.host_block_stride,
        hisparse_shared_host_pool=hisparse_layout.shared_host_pool,
        prefix_cache_retention_interval=(
            vllm_config.cache_config.prefix_cache_retention_interval
        ),
    )


def get_hisparse_steady_state_concurrency(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> float:
    """Max concurrency at max_model_len once running requests read from host.

    A request reading from host pins only its active tail of each resident
    group, but admitting one still takes its full in-flight window, so the
    bound is all-but-one requests at steady state plus one being admitted.
    """
    admission_blocks = 0
    steady_blocks = 0
    host_blocks = 0
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        required = cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
        if group.host_resident:
            host_blocks += required
            continue
        admission_blocks += required
        if isinstance(spec, HiSparseResidentSpec):
            required = min(required, ACTIVE_TAIL_PAGES)
        steady_blocks += required
    assert kv_cache_config.hisparse_host_num_blocks is not None
    num_blocks = kv_cache_config.num_blocks
    if num_blocks < admission_blocks:
        gpu_concurrency = num_blocks / admission_blocks
    else:
        gpu_concurrency = 1 + (num_blocks - admission_blocks) / steady_blocks
    return min(gpu_concurrency, kv_cache_config.hisparse_host_num_blocks / host_blocks)
