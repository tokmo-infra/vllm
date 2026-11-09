# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Narrow GLM53 producer-only context pass and checkpoint contract."""

import os

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


def p_context_enabled(vllm_config) -> bool:
    """Validate once at construction; never inspect tensors or synchronize."""
    flag = os.environ.get(OPT_IN, "0")
    if flag == "0":
        return False
    if flag != "1":
        raise ValueError(f"{OPT_IN} must be 0 or 1")

    def require(condition, message):
        if not condition:
            raise ValueError(f"GLM53 P context-only: {message}")

    require(vllm_config.use_v2_model_runner, "requires model runner V2")
    spec = vllm_config.speculative_config
    require(spec is not None, "MTP3 must remain enabled")
    require(spec.method == "mtp" and spec.num_speculative_tokens == 3, "requires MTP3")
    require(not spec.uses_dynamic_speculative_decoding(), "no dynamic drafting")
    config = spec.draft_model_config.hf_text_config
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
    require(
        config.model_type in ("glm5_next_text", "glm5_next", "glm5_next_mtp"),
        "wrong model type",
    )
    for field, expected in geometry.items():
        require(getattr(config, field, None) == expected, f"unsupported {field}")
    pc = vllm_config.parallel_config
    for field in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        require(getattr(pc, field, None) == 1, f"requires {field}=1")
    require(pc.enable_expert_parallel and pc.data_parallel_size == 4, "requires EP4")
    require(not pc.enable_eplb, "EPLB is unsupported")
    require(not pc.use_sequence_parallel_moe, "sequence parallel unsupported")
    kt = vllm_config.kv_transfer_config
    require(kt is not None, "requires dedicated producer connector")
    require(
        kt.kv_connector == "MultiConnector" and kt.kv_role == "kv_both",
        "requires producer MultiConnector",
    )
    children = kt.kv_connector_extra_config.get("connectors", [])
    roles = sorted((x.get("kv_connector"), x.get("kv_role")) for x in children)
    require(
        roles
        == [
            ("MooncakeConnector", "kv_producer"),
            ("MooncakeStoreConnector", "kv_both"),
        ],
        "requires P2P producer + Store, never D or mixed roles",
    )
    return True


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
