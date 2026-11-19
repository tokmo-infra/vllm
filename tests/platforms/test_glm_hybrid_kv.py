# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import copy
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig, ParallelConfig
from vllm.model_executor.models import ModelRegistry
from vllm.models.glm5next.common.attention import Glm5NextIndexerCache
from vllm.models.glm5next.common.model import Glm5NextForCausalLM
from vllm.platforms import current_platform
from vllm.platforms.interface import Platform
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.mla.indexer import Glm5NextIndexerBackend
from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    resolve_cache_hit_alignment_tokens,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import KVCacheLayout, MambaSpec, MLAAttentionSpec
from vllm.v1.worker.utils import (
    AttentionGroup,
    allocate_kv_cache,
    customize_attention_spec,
)


@pytest.mark.parametrize(
    ("dtype", "cli_block", "block", "pad", "skip"),
    [
        ("nvfp4", None, 14848, 4513792, []),
        ("nvfp4", 64, 14848, 4513792, []),
        ("nvfp4", 8960, 14848, 4513792, []),
        ("nvfp4", 64, 14848, 4513792, ["7"]),
        ("fp8_ds_mla", 64, 8960, 4587520, []),
    ],
)
def test_glm_hybrid_platform_alignment_and_packed_cpu_abi(
    monkeypatch, dtype, cli_block, block, pad, skip
):
    # The real GLM-5.3-Flash TP1 KDA geometry with MTP3, without weights/CUDA.
    cache = CacheConfig(
        cache_dtype=dtype,
        block_size=cli_block,
        prefix_match_unit=256,
        mamba_cache_mode="align",
        kv_cache_dtype_skip_layers=skip,
    )
    cache.kv_cache_layout = "BLHNC"
    snapshot = copy(cache)
    config = SimpleNamespace(
        cache_config=cache,
        model_config=SimpleNamespace(
            use_mla=True,
            is_hybrid=True,
            dtype=torch.bfloat16,
            architecture="GLM",
            hf_text_config=SimpleNamespace(
                model_type="glm5_next_text",
                kv_lora_rank=512,
                qk_rope_head_dim=0,
                index_topk=2048,
                index_kpool=4,
                linear_num_heads=64,
                linear_head_dim=128,
                linear_conv_kernel_dim=4,
            ),
            get_head_size=lambda: 512,
            get_num_kv_heads=lambda _: 1,
        ),
        parallel_config=ParallelConfig(),
        attention_config=SimpleNamespace(hisparse_config=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=SimpleNamespace(
            method="mtp",
            num_speculative_tokens=3,
            use_eagle=lambda: True,
            use_eagle_block_drop=lambda: True,
        ),
        kv_transfer_config=None,
    )
    mla = FlashInferMLASparseSM120Backend
    monkeypatch.setattr(
        Platform, "_find_non_ssm_backends", lambda _: [mla, Glm5NextIndexerBackend]
    )
    monkeypatch.setattr(
        ModelRegistry,
        "resolve_model_cls",
        lambda *a, **kw: (Glm5NextForCausalLM, "GLM"),
    )
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: family == 120
    )
    Platform.update_block_size_for_backend(config)
    assert cache.block_size == cache.mamba_block_size == block
    assert cache.mamba_page_size_padded == pad
    assert cache.cache_dtype == dtype
    assert cache.skip_page_size_padded is None

    indexer = Glm5NextIndexerCache.__new__(Glm5NextIndexerCache)
    indexer.cache_config = snapshot
    indexer.dtype, indexer.head_dim, indexer._index_kpool = torch.uint8, 132, 4
    index_spec = customize_attention_spec(
        Glm5NextIndexerBackend, indexer.get_kv_cache_spec(config)
    )
    assert index_spec.block_size == block

    def mla_spec(layout, row_bytes):
        return customize_attention_spec(
            mla,
            MLAAttentionSpec(
                block_size=block,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.uint8,
                cache_dtype_str=layout,
                state_content_bytes=row_bytes,
            ),
        )

    width = 304 if dtype == "nvfp4" else 528
    model = Glm5NextForCausalLM
    specs = {
        "kda": MambaSpec(
            block_size=block,
            shapes=model.get_mamba_state_shape_from_config(config),
            dtypes=model.get_mamba_state_dtype_from_config(config),
            mamba_cache_mode="align",
            page_size_padded=pad,
        ),
        "target": mla_spec("glm_nvfp4_mla" if width == 304 else dtype, width),
        "draft_index": index_spec,
        "draft": mla_spec("fp8_ds_mla", 528),
    }
    if skip:
        specs["target_skip"] = mla_spec("fp8_ds_mla", 528)
    assert specs["kda"].real_page_size_bytes == 4489216
    groups = get_kv_cache_groups(config, specs)
    planned = get_kv_cache_config_from_groups(config, groups, 64 << 20)
    scheduler_cache = generate_scheduler_kv_cache_config([planned])
    assert resolve_kv_cache_block_sizes(scheduler_cache, config) == (block, 256)
    assert (
        resolve_cache_hit_alignment_tokens(scheduler_cache, config, block, 256) == 256
    )
    views = allocate_kv_cache(planned, torch.device("cpu"), KVCacheLayout.BLHNC)
    assert views["target"].shape[-1] == width
    assert views["draft"].shape[-1] == 528
    views["target"][0].fill_(17)
    views["draft"][0].fill_(29)
    assert torch.all(views["target"][0] == 17)
    assert torch.all(views["draft"][0] == 29)
    assert not torch.any(views["target"][1])
    rows, _ = flat_kv_row_view(views["draft"].squeeze(1), block)
    assert rows.view(-1, 64, 528).shape[-2:] == (64, 528)
    group = AttentionGroup(
        Glm5NextIndexerBackend, ["draft_index"], index_spec, 0, kernel_block_size=256
    )
    packed = group.map_kv_cache_to_kernel_blocks(views["draft_index"])
    assert packed.shape[-2:] == (64, 132)
    assert (
        packed[group.kernel_block_stride].data_ptr()
        == views["draft_index"][1].data_ptr()
    )
