# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in small NVFP4 routed MoE; native attention/shared experts/fallback remain.

All collective allocation, calibration checks and JIT happen at weight loading.
Forward consumes existing CPU DP metadata and launches on the current CUDA stream.
"""

import os

import torch

from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from .glm53_nvfp4_policy import (
    CAPACITY,
    supported_geometry,
    tokens_per_item,
    uniform_bound,
)

logger = init_logger(__name__)
_WORKSPACES = {}
_LOGGED: set[tuple[int, int, int, bool, str, int]] = set()


def _eligible(experts, layer):
    if os.environ.get("VLLM_GLM53_NVFP4_FUSED_MOE") != "1":
        return False
    mc = experts.moe_config
    pc = mc.moe_parallel_config
    if not supported_geometry(
        mc.hidden_dim,
        mc.intermediate_size,
        mc.num_experts,
        mc.experts_per_token,
        mc.num_local_experts,
        pc.ep_size,
    ):
        return False
    if not (
        pc.use_ep
        and pc.dp_size == pc.ep_size
        and pc.tp_size == 1
        and pc.pcp_size == 1
        and pc.sp_size == 1
        and not pc.enable_eplb
        and pc.all2all_backend == "allgather_reducescatter"
    ):
        return False
    if (
        mc.in_dtype != torch.bfloat16
        or mc.activation != MoEActivation.SILU
        or mc.swiglu_limit != 10.0
        or mc.has_bias
        or mc.is_lora_enabled
        or mc.skip_final_all_reduce
        or mc.swiglu_alpha is not None
        or mc.swiglu_beta is not None
    ):
        return False
    if not experts.quant_config.use_nvfp4_w4a4:
        return False
    if (
        getattr(getattr(layer, "expert_map_manager", None), "placement_strategy", None)
        != "linear"
    ):
        return False
    if not layer.w13_weight.is_cuda:
        return False
    if torch.cuda.get_device_capability(layer.w13_weight.device) != (12, 0):
        return False
    from vllm.config import get_current_vllm_config_or_none

    cfg = get_current_vllm_config_or_none()
    return not (
        cfg is not None
        and (
            cfg.parallel_config.enable_dbo
            or cfg.weight_transfer_config is not None
            or cfg.model_config.hf_text_config.model_type != "glm5_next_text"
        )
    )


def bind_native(experts, layer):
    """After native alpha postprocessing: alias the installed parameters exactly."""
    if not _eligible(experts, layer):
        return False
    from vllm._glm53_nvfp4.nvfp4_ep.contracts import ABI_VERSION
    from vllm._glm53_nvfp4.nvfp4_ep.weights import alias_native
    from vllm._glm53_nvfp4.nvfp4_ep.workspace import Nvfp4Workspace

    qc = experts.quant_config
    mc = experts.moe_config
    # Native loader has already normalized the global-expert calibration to
    # one value repeated over local experts. Validate during INIT, never forward.
    a1 = qc.a1_gscale.reshape(-1)
    if not torch.equal(a1, a1[0].expand_as(a1)):
        raise ValueError("GLM native a1 calibration is not uniform after loading")
    for value in (a1, qc.a2_gscale, qc.g1_alphas, qc.g2_alphas):
        if not bool(torch.all(torch.isfinite(value) & (value > 0))):
            raise ValueError("invalid native NVFP4 calibration")
    pack = alias_native(
        w13=layer.w13_weight,
        w2=layer.w2_weight,
        w13_scale=qc.w1_scale,
        w2_scale=qc.w2_scale,
        g1_alphas=qc.g1_alphas,
        g2_alphas=qc.g2_alphas,
        a1_gscale=qc.a1_gscale,
        a1_quant_scalar=a1[:1],
        a2_gscale=qc.a2_gscale,
        local_experts=mc.num_local_experts,
        w13_layout=1,
        scale_layout=1,
    )
    assert pack.w13.data_ptr() == layer.w13_weight.data_ptr()
    assert pack.w2.data_ptr() == layer.w2_weight.data_ptr()
    assert pack.w13_scale.data_ptr() == qc.w1_scale.data_ptr()
    group = get_ep_group()
    assert group.world_size == mc.moe_parallel_config.ep_size
    assert group.rank_in_group == mc.moe_parallel_config.ep_rank
    device = layer.w13_weight.device.index
    workspace_capacity = (
        CAPACITY if os.environ.get("VLLM_GLM53_P_MTP_CONTEXT_ONLY") == "1" else 128
    )
    stage_value = os.environ.get("VLLM_GLM53_NVFP4_STAGE_FC2_INPUT", "0")
    if stage_value not in ("0", "1"):
        raise ValueError("VLLM_GLM53_NVFP4_STAGE_FC2_INPUT must be 0 or 1")
    stage_fc2_input = stage_value == "1"
    vector_value = os.environ.get("VLLM_GLM53_NVFP4_VECTOR_WEIGHTS", "0")
    if vector_value not in ("0", "1"):
        raise ValueError("VLLM_GLM53_NVFP4_VECTOR_WEIGHTS must be 0 or 1")
    vector_weights = vector_value == "1"
    prefetch_value = os.environ.get("VLLM_GLM53_NVFP4_EXTENDED_PREFETCH", "0")
    if prefetch_value not in ("0", "1"):
        raise ValueError("VLLM_GLM53_NVFP4_EXTENDED_PREFETCH must be 0 or 1")
    extended_prefetch = prefetch_value == "1"
    key = (
        ABI_VERSION,
        id(group.device_group),
        device,
        group.world_size,
        mc.hidden_dim,
        mc.intermediate_size,
        mc.num_experts,
        mc.experts_per_token,
        workspace_capacity,
        "p2p",
        stage_fc2_input,
        vector_weights,
        extended_prefetch,
    )
    if key not in _WORKSPACES:
        _WORKSPACES[key] = Nvfp4Workspace(
            group=group.device_group,
            rank=group.rank_in_group,
            world_size=group.world_size,
            num_experts=288,
            capacity=workspace_capacity,
            top_k=8,
            hidden=4096,
            intermediate=2048,
            transport="p2p",
            fast_math=False,
            stage_fc2_input=stage_fc2_input,
            vector_weights=vector_weights,
            extended_prefetch=extended_prefetch,
        )
    experts._glm53_nvfp4_small = SmallMoE(pack, _WORKSPACES[key], group.world_size)
    logger.info(
        "GLM NVFP4 fused MoE bound: EP%s, native W13/W2/scales aliased, "
        "shared workspace=%s bytes, native shared experts and large fallback retained; "
        "FC2 input staging=%s capacity=%s vector_weights=%s extended_prefetch=%s",
        group.world_size,
        _WORKSPACES[key].buffer.numel(),
        stage_fc2_input,
        workspace_capacity,
        vector_weights,
        extended_prefetch,
    )
    return True


class SmallMoE:
    def __init__(self, weights, workspace, ep_size):
        self.weights = weights
        self.workspace = workspace
        self.ep_size = ep_size

    def try_forward(
        self,
        hidden_states,
        topk_ids,
        topk_weights,
        *,
        activation,
        apply_router_weight_on_input,
        shared_experts=None,
        shared_experts_input=None,
    ):
        if activation != MoEActivation.SILU or apply_router_weight_on_input:
            return None
        context = get_forward_context()
        meta = getattr(context, "dp_metadata", None)
        sizes = getattr(meta, "local_sizes", None)
        rows = hidden_states.shape[0]
        bound = uniform_bound(
            sizes, rows, self.ep_size, allow_extended=self.workspace.capacity > 128
        )
        if bound is None:
            return None
        tile = tokens_per_item(sizes, self.workspace.capacity)
        sample = (
            self.ep_size,
            rows,
            bound,
            shared_experts is not None,
            str(getattr(context, "cudagraph_runtime_mode", "unknown")),
            tile,
        )
        if sample not in _LOGGED and len(_LOGGED) < 64:
            _LOGGED.add(sample)
            logger.info(
                "GLM NVFP4 fused dispatch EP%s rows=%s uniform_bound=%s "
                "shared=%s mode=%s tile=%s",
                *sample,
            )
        self.workspace.stage(
            hidden_states,
            topk_ids.to(torch.int64),
            topk_weights,
            self.weights,
            uniform_bound=bound,
        )
        if shared_experts is not None:
            assert shared_experts_input is not None
            from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
                SharedExpertsOrder,
            )

            # NO_OVERLAP and AUX modes were scheduled by MoERunner; this call
            # is a no-op for those modes and executes MK_INTERNAL exactly once.
            shared_experts(
                shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
            )
        output = torch.empty(
            (max(rows, 1), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return self.workspace.compute(
            self.weights, output=output, swiglu_limit=10.0, tokens_per_item=tile
        )
