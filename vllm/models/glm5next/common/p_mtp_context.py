# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Narrow GLM53 producer-only context pass and checkpoint contract."""

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)

OPT_IN = "VLLM_GLM53_P_MTP_CONTEXT_ONLY"
PREFIX = "model.layers.45."
REQUIRED_KEYS = frozenset(
    PREFIX + suffix
    for suffix in (
        "eh_proj.weight",
        "enorm.weight",
        "hnorm.weight",
        "input_layernorm.weight",
        "self_attn.indexer.index_kpool_compress_ape",
        "self_attn.indexer.index_kpool_compress_gate",
        "self_attn.indexer.k_norm.bias",
        "self_attn.indexer.k_norm.weight",
        "self_attn.indexer.weights_proj.weight",
        "self_attn.indexer.wk.weight",
        "self_attn.indexer.wq_b.weight",
        "self_attn.kv_a_layernorm.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.kv_b_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_a_layernorm.weight",
        "self_attn.q_a_proj.weight",
        "self_attn.q_b_proj.weight",
    )
)
DROPPED_KEYS = frozenset(
    [
        f"{PREFIX}mlp.experts.{expert}.{proj}.weight"
        for expert in range(288)
        for proj in ("gate_proj", "up_proj", "down_proj")
    ]
    + [
        f"{PREFIX}mlp.shared_experts.{proj}.weight"
        for proj in ("gate_proj", "up_proj", "down_proj")
    ]
    + [
        PREFIX + "mlp.gate.weight",
        PREFIX + "mlp.gate.e_score_correction_bias",
        PREFIX + "post_attention_layernorm.weight",
        PREFIX + "shared_head.norm.weight",
    ]
)


def p_context_enabled(vllm_config: "VllmConfig") -> bool:
    """Decide before allocating any weights; unsupported runs keep full MTP."""
    flag = os.environ.get(OPT_IN, "0")
    if flag == "0":
        return False
    if flag != "1":
        raise ValueError(f"{OPT_IN} must be 0 or 1")

    if reason := _incompatible_config(vllm_config):
        logger.warning("GLM53 P context-only disabled (%s); using full MTP", reason)
        return False
    return True


def _incompatible_config(vllm_config: "VllmConfig") -> str | None:
    if not vllm_config.use_v2_model_runner:
        return "requires model runner V2"
    spec = vllm_config.speculative_config
    if spec is None or spec.method != "mtp" or not spec.num_speculative_tokens:
        return "requires MTP"
    if (
        spec.uses_dynamic_speculative_decoding()
        or spec.parallel_drafting
        or spec.enable_adaptive_verification
    ):
        return "requires fixed, serial MTP"
    if not spec.glm_mtp_has_native_attn_overrides():
        return "incompatible draft attention overrides"
    if spec.draft_load_config is not None:
        return "independent draft weight loader"
    draft = spec.draft_model_config
    target = vllm_config.model_config
    if (
        target is None
        or draft is None
        or draft.model != (target.model_weights or target.model)
        or draft.revision != target.revision
        or draft.code_revision != target.code_revision
        or draft.quantization != target.quantization
        or spec.max_model_len is not None
    ):
        return "requires the embedded MTP checkpoint"
    geometry = {
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 288,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "num_hidden_layers": 45,
        "num_nextn_predict_layers": 1,
        "first_k_dense_replace": 3,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "mla_use_nope": True,
        "index_kpool": 4,
        "index_head_dim": 128,
        "index_n_heads": 32,
    }
    for config in (target.hf_text_config, draft.hf_text_config):
        if config.model_type not in ("glm5_next_text", "glm5_next", "glm5_next_mtp"):
            return "wrong model type"
        if getattr(config, "index_topk", None) is None:
            return "requires a sparse MLA indexer"
        for field, expected in geometry.items():
            if getattr(config, field, None) != expected:
                return f"unsupported {field}"
    pc = vllm_config.parallel_config
    if spec.draft_tensor_parallel_size not in (None, pc.tensor_parallel_size):
        return "draft TP differs from target TP"
    for field in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if getattr(pc, field) != 1:
            return f"requires {field}=1"
    if pc.enable_eplb or pc.use_sequence_parallel_moe:
        return "EPLB and sequence parallelism are unsupported"
    # EP/DP and draft depth do not change the retained weights. PD still
    # requires the peer to use compatible KDA and indexer-tail cache geometry.
    kt = vllm_config.kv_transfer_config
    if kt is None or kt.kv_connector != "MultiConnector" or kt.kv_role != "kv_both":
        return "requires dedicated producer MultiConnector"
    children = kt.kv_connector_extra_config.get("connectors", [])
    if not isinstance(children, list) or not all(isinstance(x, dict) for x in children):
        return "invalid producer connector list"
    roles = [(x.get("kv_connector"), x.get("kv_role")) for x in children]
    if len(roles) != 2 or not all(
        role in roles
        for role in (
            ("MooncakeConnector", "kv_producer"),
            ("MooncakeStoreConnector", "kv_both"),
        )
    ):
        return "requires P2P producer + Store, never D or mixed roles"
    return None


def normalize_name(name: str) -> str:
    if name.startswith("model.language_model."):
        return name.replace("model.language_model.", "model.", 1)
    return name


class ContextWeightCoverage:
    """Exact source allowlist plus constructed-parameter loading coverage."""

    def __init__(self):
        self.retained = set()
        self.dropped = set()

    def discard(self, name: str) -> bool:
        name = normalize_name(name)
        if name in DROPPED_KEYS:
            self.dropped.add(name)
            return True
        if name.startswith(PREFIX):
            if name not in REQUIRED_KEYS:
                raise ValueError(f"Unrecognized context-MTP weight: {name}")
            if name in self.retained:
                raise ValueError(f"Duplicate context-MTP weight: {name}")
            self.retained.add(name)
        return False

    def finish(self, params, loaded, pending):
        missing = REQUIRED_KEYS - self.retained
        if missing:
            raise ValueError(f"Missing context-MTP source weights: {sorted(missing)}")
        missing_params = set(params) - loaded
        if missing_params or pending:
            raise ValueError(
                f"Unloaded context-MTP params: {sorted(missing_params)}; "
                f"pending conversions: {sorted(pending)}"
            )
