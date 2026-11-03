"""Frozen-image deployment bridge for the isolated experimental MegaMoE."""

from functools import cache

@cache
def _import_deep_gemm():
    import deep_gemm_overlap
    from vllm.platforms import current_platform
    if current_platform.is_arch_support_pdl():
        deep_gemm_overlap.set_pdl(True)
    return deep_gemm_overlap
