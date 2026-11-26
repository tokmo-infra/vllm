# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
    maybe_share_target_embed,
)
from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config


def dflash_attention_view(vllm_config: VllmConfig, non_causal: bool) -> VllmConfig:
    """One cache ABI for draft layer creation and its metadata builders.

    Keep the resolved physical page size and the shared layer registry. Avoid
    revalidating the outer target config (e.g. hybrid MLA + GQA), or inheriting
    target-only KV precision exceptions into the external draft model.
    """
    spec = vllm_config.speculative_config
    assert spec is not None
    view = copy.copy(vllm_config)
    view.attention_config = replace(
        vllm_config.attention_config,
        use_non_causal=non_causal,
        backend=spec.attention_backend,
    )
    if spec.kv_cache_dtype is not None:
        view.cache_config = copy.copy(vllm_config.cache_config)
        view.cache_config.cache_dtype = spec.kv_cache_dtype
        view.cache_config.kv_cache_dtype_skip_layers = []
    return view


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    draft_vllm_config = dflash_attention_view(
        vllm_config, dflash_has_any_non_causal(draft_model_config.hf_config)
    )
    draft_vllm_config.load_config = get_pp_safe_draft_load_config(
        get_draft_load_config(vllm_config)
    )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    # MuseGlimmerForCausalLM marks its inner MuseGlimmerModel as the language
    # model, so get_language_model() already returns the inner module and has
    # no .model of its own.
    target_inner = getattr(target_language_model, "model", target_language_model)
    draft_inner = dflash_model.model

    maybe_share_target_embed(dflash_model, draft_inner, target_inner)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if (
        target_lm_head is not None
        and not getattr(dflash_model.config, "has_own_lm_head", False)
        and _should_share(
            dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
        )
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    return dflash_model
