# SPDX-License-Identifier: Apache-2.0
"""Layer-local sparse MLA record ABI. This is not a generic K/V head format."""

GLM_NVFP4_MLA = "glm_nvfp4_mla"
GLM_NVFP4_ROW_BYTES = 304


def is_glm_next_nope(text_config) -> bool:
    return (
        getattr(text_config, "model_type", None) in ("glm5_next_text", "glm5_next")
        and getattr(text_config, "kv_lora_rank", None) == 512
        and getattr(text_config, "qk_rope_head_dim", None) == 0
        and bool(getattr(text_config, "index_topk", None))
    )
