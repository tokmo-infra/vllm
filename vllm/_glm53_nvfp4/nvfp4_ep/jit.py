# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initialization-only JIT; fresh ABI/content key prevents mixing MXFP4/NVFP4.

No build/communicator discovery is permitted inside graph capture or compute().
"""

import functools
import hashlib
import os
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_NCCL_BUILD = None
# Do NOT use nvcc -arch=sm_120a: CUDA13 shorthand also emits generic compute_120
# PTX, where NVFP4 block-scale MMA and E2M1 conversion are not legal.
SM120A_ONLY_FLAGS = ("-gencode=arch=compute_120a,code=sm_120a",)


def require_specific_arch_flags(flags):
    """Keep actual FlashInfer common flags, reject generic/shorthand arch additions."""
    arch = [
        f
        for f in flags
        if f.startswith(("-gencode", "--generate-code", "-arch", "--gpu-architecture"))
    ]
    if arch != list(SM120A_ONLY_FLAGS):
        raise ValueError(f"need only explicit SM120A target, got {arch}")
    return list(flags)


def reject_env_arch_overrides(env):
    # Actual FlashInfer cpp_ext appends FLASHINFER_EXTRA_CUDAFLAGS after dropping
    # global gencode flags. NVCC itself can append/prepend flags as well.
    for name in (
        "FLASHINFER_EXTRA_CUDAFLAGS",
        "NVCC_APPEND_FLAGS",
        "NVCC_PREPEND_FLAGS",
    ):
        flags = shlex.split(env.get(name, ""))
        if any(
            f.startswith(
                (
                    "-gencode",
                    "--generate-code",
                    "-arch",
                    "--gpu-architecture",
                    "-code",
                    "--gpu-code",
                )
            )
            for f in flags
        ):
            raise ValueError(
                f"{name} overrides NVFP4 architecture; use module-specific SM120A flags"
            )


def configure_nccl_build(include, library, gin_backends):
    """Parent can provide the *actual* pinned NCCL paths/backend definitions.

    gin_backends is the pinned FlashInfer GIN_BACKENDS dictionary:
    {name: (integer_gin_type, [nvcc_backend_defines...])}. No guessed flags.
    Call once before creating any workspace, on each rank.
    """
    global _NCCL_BUILD
    if _NCCL_BUILD is not None:
        raise RuntimeError("NCCL build already configured; cannot mutate live modules")
    include, library = Path(include), Path(library)
    if not (include / "nccl_device.h").is_file() or not library.is_file():
        raise ValueError("need actual NCCL Device API headers and matching library")
    _NCCL_BUILD = include, library, gin_backends


def _nccl():
    if _NCCL_BUILD is not None:
        return _NCCL_BUILD
    # Optional GIN resolver; the GLM adapter uses only P2P.
    try:
        from flashinfer.moe_ep.kernel_src.sm120.cake_mxfp4_megamoe.shim import (
            cake_jit,
        )
    except ImportError as exc:
        raise RuntimeError(
            "provide configure_nccl_build() from the pinned runtime; "
            "no guessed GIN flags"
        ) from exc
    include, library = cake_jit.nccl_installation()
    return Path(include), Path(library), cake_jit.GIN_BACKENDS


def _identity(*extra):
    h = hashlib.sha256()
    for p in (
        ROOT / "csrc/megamoe_sm120_nvfp4.cu",
        ROOT / "include/flashinfer/megamoe/sm120_nvfp4_mega_moe.cuh",
        ROOT / "csrc/megamoe_sm120_nvfp4_gin_session.cu",
    ):
        h.update(p.read_bytes())
    h.update(repr(extra).encode())
    return h.hexdigest()[:16]


def _build(uri, source, *, fast_math=False, flags=(), include=(), ldflags=()):
    reject_env_arch_overrides(os.environ)
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import gen_jit_spec, sm120a_nvcc_flags

    actual_flags = require_specific_arch_flags([*sm120a_nvcc_flags, *flags])
    return gen_jit_spec(
        uri,
        [ROOT / "csrc" / source],
        extra_cuda_cflags=["-std=c++17", *actual_flags],
        extra_include_paths=[ROOT / "include", jit_env.FLASHINFER_CSRC_DIR, *include],
        extra_ldflags=list(ldflags),
        use_fast_math=fast_math,
    ).build_and_load()


@functools.lru_cache(maxsize=None, typed=True)
def kernel_module(
    fast_math=False,
    gin_type=None,
    stage_fc2_input=False,
    vector_weights=False,
    extended_prefetch=False,
):
    if type(stage_fc2_input) is not bool:
        raise ValueError("stage_fc2_input must be a fixed initialization-time bool")
    if type(vector_weights) is not bool:
        raise ValueError("vector_weights must be a fixed initialization-time bool")
    if vector_weights and gin_type is not None:
        raise ValueError("vector weights initially validated for P2P only")
    if type(extended_prefetch) is not bool:
        raise ValueError("extended_prefetch must be a fixed initialization-time bool")
    if extended_prefetch and gin_type is not None:
        raise ValueError("extended prefetch initially validated for P2P only")
    prefetch_flags = (
        ["-DFLASHINFER_NVFP4_EXTENDED_PREFETCH=1"] if extended_prefetch else []
    )
    vector_flags = ["-DFLASHINFER_NVFP4_VECTOR_WEIGHTS=1"] if vector_weights else []
    stage_flags = ["-DFLASHINFER_NVFP4_STAGE_FC2_INPUT=1"] if stage_fc2_input else []
    if stage_fc2_input and gin_type is not None:
        raise ValueError("FC2 input staging is initially validated for P2P only")
    if gin_type is None:
        uri = "megamoe_sm120_nvfp4_g16_" + _identity(
            fast_math, stage_fc2_input, vector_weights, extended_prefetch
        )
        return _build(
            uri,
            "megamoe_sm120_nvfp4.cu",
            fast_math=fast_math,
            flags=[*stage_flags, *vector_flags, *prefetch_flags],
        )
    include, library, backends = _nccl()
    names = {kind: (name, flags) for name, (kind, flags) in backends.items()}
    if gin_type not in names:
        raise ValueError(
            f"actual NCCL GIN type{gin_type} absent from pinned build definitions"
        )
    name, flags = names[gin_type]
    uri = (
        "megamoe_sm120_nvfp4_g16_gin_"
        + name
        + "_"
        + _identity(fast_math, gin_type, flags, str(library))
    )
    return _build(
        uri,
        "megamoe_sm120_nvfp4.cu",
        fast_math=fast_math,
        flags=[
            "-DFLASHINFER_MEGAMOE_SM120_GIN=1",
            "-D_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE",
            *flags,
        ],
        include=[include],
        ldflags=[str(library), f"-Wl,-rpath,{library.parent}"],
    )


@functools.cache
def gin_session_module():
    include, library, _ = _nccl()
    return _build(
        "megamoe_sm120_nvfp4_gin_session_" + _identity(str(library)),
        "megamoe_sm120_nvfp4_gin_session.cu",
        include=[include],
        ldflags=[str(library), f"-Wl,-rpath,{library.parent}"],
    )
