# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The draft must honour ``attention_backend`` from --speculative-config.

``init_attn_backend`` reads the backend off each constructed layer, so these
assert on the config ``load_eagle_model`` hands to ``get_model``.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config import LoadConfig, SpeculativeConfig
from vllm.models.glm5next.common.mtp import Glm5NextMTP
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


@dataclass
class _AttentionConfig:
    backend: str | None = None


@dataclass
class _KernelConfig:
    moe_backend: str | None = None


@dataclass
class _CacheConfig:
    cache_dtype: str = "auto"
    block_size: int = 8960
    user_specified_block_size: bool = False
    kv_cache_dtype_skip_layers: list[str] = field(default_factory=list)


@dataclass
class _SpeculativeConfig:
    method: str = "eagle"
    attention_backend: str | None = None
    moe_backend: str | None = None
    kv_cache_dtype: str | None = None
    draft_model_config: object = None
    draft_load_config: object = None

    apply_draft_overrides = SpeculativeConfig.apply_draft_overrides
    glm_mtp_needs_fp8_kv = SpeculativeConfig.glm_mtp_needs_fp8_kv
    glm_mtp_has_native_attn_overrides = (
        SpeculativeConfig.glm_mtp_has_native_attn_overrides
    )


@dataclass
class _VllmConfig:
    attention_config: _AttentionConfig
    kernel_config: _KernelConfig
    cache_config: _CacheConfig
    speculative_config: _SpeculativeConfig
    load_config: LoadConfig = field(default_factory=LoadConfig)
    compilation_config: object = field(default_factory=object)


def _config(target_backend: str, draft_backend: str | None) -> _VllmConfig:
    return _VllmConfig(
        attention_config=_AttentionConfig(backend=target_backend),
        kernel_config=_KernelConfig(),
        cache_config=_CacheConfig(),
        speculative_config=_SpeculativeConfig(attention_backend=draft_backend),
    )


class _Captured(Exception):
    def __init__(self, vllm_config):
        self.vllm_config = vllm_config


def _capture_draft_config(cfg):
    def _fake_get_model(*, vllm_config, model_config):
        raise _Captured(vllm_config)

    with (
        patch("vllm.v1.worker.gpu.spec_decode.eagle.utils.get_model", _fake_get_model),
        patch(
            "vllm.v1.worker.gpu.spec_decode.utils.get_pp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
        pytest.raises(_Captured) as exc,
    ):
        load_eagle_model(object(), cfg)
    return exc.value.vllm_config


def test_draft_attention_backend_overrides_the_target():
    used = _capture_draft_config(_config("FLASHINFER", "TRITON_ATTN"))
    assert used.attention_config.backend == "TRITON_ATTN"


def test_unset_leaves_the_target_backend_in_place():
    """Clearing it lets the draft autoselect a KV layout the target lacks."""
    cfg = _config("FLEX_ATTENTION", None)
    used = _capture_draft_config(cfg)
    assert used is cfg
    assert used.attention_config.backend == "FLEX_ATTENTION"


def test_override_does_not_mutate_the_target_config():
    cfg = _config("FLASHINFER", "TRITON_ATTN")
    _capture_draft_config(cfg)
    assert cfg.attention_config.backend == "FLASHINFER"


@pytest.mark.parametrize("target_dtype", ["nvfp4", "glm_nvfp4_mla"])
@pytest.mark.parametrize("draft_dtype", [None, "fp8", "nvfp4"])
def test_glm_nvfp4_target_constructs_a_separate_fp8_draft(target_dtype, draft_dtype):
    cfg = _config("target_backend", None)
    cfg.cache_config.cache_dtype = target_dtype
    cfg.cache_config.kv_cache_dtype_skip_layers = ["45"]
    spec = cfg.speculative_config
    spec.method = "mtp"
    spec.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="glm5_next_mtp")
    )
    spec.kv_cache_dtype = draft_dtype
    spec.draft_load_config = LoadConfig(load_format="safetensors")
    if draft_dtype == "nvfp4":
        spec.attention_backend = "other_backend"

    used = _capture_draft_config(cfg)

    assert used.cache_config == _CacheConfig(cache_dtype="fp8_ds_mla")
    assert (
        used.attention_config.backend
        == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120
    )
    assert used.compilation_config is cfg.compilation_config
    assert used.load_config is spec.draft_load_config
    assert cfg.cache_config.cache_dtype == target_dtype
    assert cfg.cache_config.kv_cache_dtype_skip_layers == ["45"]
    assert cfg.attention_config.backend == "target_backend"


@pytest.mark.parametrize("valid", [True, False])
def test_glm_mixed_draft_loader_checks_real_layout_before_sharing(valid):
    cfg = _config("target_backend", None)
    cfg.cache_config.cache_dtype = "nvfp4"
    spec = cfg.speculative_config
    spec.method = "mtp"
    spec.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="glm5_next_mtp")
    )
    mla = SimpleNamespace(
        layer_name="draft.attn",
        get_kv_cache_spec=Mock(
            return_value=MLAAttentionSpec(
                block_size=256,
                num_kv_heads=1,
                head_size=512,
                cache_dtype_str="fp8_ds_mla",
                dtype=torch.uint8,
                state_content_bytes=528 if valid else 304,
            )
        ),
    )
    eagle_model = SimpleNamespace(
        model=SimpleNamespace(_mtp_mla_attns=[SimpleNamespace(mla_attn=mla)]),
    )
    eagle_model.validate_fp8_kv_cache = lambda config: (
        Glm5NextMTP.validate_fp8_kv_cache(eagle_model, config)
    )
    with (
        patch(
            "vllm.v1.worker.gpu.spec_decode.eagle.utils.get_model",
            return_value=eagle_model,
        ) as load,
        patch(
            "vllm.v1.worker.gpu.spec_decode.eagle.utils.maybe_share_target_embed"
        ) as share,
        patch(
            "vllm.v1.worker.gpu.spec_decode.utils.get_pp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
    ):
        if valid:
            assert load_eagle_model(SimpleNamespace(model=object()), cfg) is eagle_model
            share.assert_called_once()
        else:
            with pytest.raises(ValueError, match="528-byte packed FP8"):
                load_eagle_model(SimpleNamespace(model=object()), cfg)
            share.assert_not_called()
        mla.get_kv_cache_spec.assert_called_once_with(
            load.call_args.kwargs["vllm_config"]
        )


def test_legacy_runner_rejects_glm_mixed_kv_before_model_load():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    cfg = _config("target_backend", None)
    cfg.cache_config.cache_dtype = "nvfp4"
    spec = cfg.speculative_config
    spec.method = "mtp"
    spec.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="glm5_next_mtp")
    )
    for name in ("model", "offload", "lora", "parallel", "scheduler"):
        setattr(cfg, f"{name}_config", None)
    with pytest.raises(ValueError, match="requires Model Runner V2"):
        GPUModelRunner(cfg, torch.device("cpu"))
