ARG BASE_IMAGE=local/glm53flash-1103:router-policy-fi071-20261107
FROM ${BASE_IMAGE}
COPY integrations/glm53flash/release.json /opt/glm53flash-1103-release.json
RUN python3 -c "import hashlib,json,pathlib; m=json.loads(pathlib.Path('/opt/glm53flash-1103-release.json').read_text()); r=pathlib.Path('/usr/local/lib/python3.12/dist-packages'); bad=[x['path'] for x in m['base_validation_files'] if hashlib.sha256((r/x['path']).read_bytes()).hexdigest() not in [x['sha256'],*x.get('compatible_sha256',[])]]; assert not bad, bad"
COPY vllm/_glm53_nvfp4/__init__.py /usr/local/lib/python3.12/dist-packages/vllm/_glm53_nvfp4/__init__.py
COPY vllm/_glm53_nvfp4/csrc/megamoe_sm120_nvfp4.cu /usr/local/lib/python3.12/dist-packages/vllm/_glm53_nvfp4/csrc/megamoe_sm120_nvfp4.cu
COPY vllm/_glm53_nvfp4/csrc/megamoe_sm120_nvfp4_gin_session.cu /usr/local/lib/python3.12/dist-packages/vllm/_glm53_nvfp4/csrc/megamoe_sm120_nvfp4_gin_session.cu
COPY vllm/_glm53_nvfp4/include/flashinfer/megamoe/sm120_nvfp4_mega_moe.cuh /usr/local/lib/python3.12/dist-packages/vllm/_glm53_nvfp4/include/flashinfer/megamoe/sm120_nvfp4_mega_moe.cuh
COPY vllm/_glm53_nvfp4/nvfp4_ep/__init__.py /usr/local/lib/python3.12/dist-packages/vllm/_glm53_nvfp4/nvfp4_ep/__init__.py
COPY vllm/_glm53_nvfp4/nvfp4_ep/combine_reference.py /usr/local/lib/python3.12/dist-packages/vllm/_glm53_nvfp4/nvfp4_ep/combine_reference.py
COPY vllm/_glm53_nvfp4/nvfp4_ep/contracts.py /usr/local/lib/python3.12/dist-packages/vllm/_glm53_nvfp4/nvfp4_ep/contracts.py
COPY vllm/_glm53_nvfp4/nvfp4_ep/jit.py /usr/local/lib/python3.12/dist-packages/vllm/_glm53_nvfp4/nvfp4_ep/jit.py
COPY vllm/_glm53_nvfp4/nvfp4_ep/reference_cpu.py /usr/local/lib/python3.12/dist-packages/vllm/_glm53_nvfp4/nvfp4_ep/reference_cpu.py
COPY vllm/_glm53_nvfp4/nvfp4_ep/verification.py /usr/local/lib/python3.12/dist-packages/vllm/_glm53_nvfp4/nvfp4_ep/verification.py
COPY vllm/_glm53_nvfp4/nvfp4_ep/weights.py /usr/local/lib/python3.12/dist-packages/vllm/_glm53_nvfp4/nvfp4_ep/weights.py
COPY vllm/_glm53_nvfp4/nvfp4_ep/workspace.py /usr/local/lib/python3.12/dist-packages/vllm/_glm53_nvfp4/nvfp4_ep/workspace.py
COPY vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py
COPY vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py
COPY vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/retire_checkpoints.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/retire_checkpoints.py
COPY vllm/model_executor/warmup/kernel_warmup.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/warmup/kernel_warmup.py
COPY vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py
COPY vllm/envs.py /usr/local/lib/python3.12/dist-packages/vllm/envs.py
COPY vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py
COPY vllm/model_executor/layers/fused_moe/glm53_nvfp4.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/glm53_nvfp4.py
COPY vllm/model_executor/layers/fused_moe/glm53_nvfp4_policy.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/glm53_nvfp4_policy.py
COPY vllm/model_executor/layers/fused_moe/modular_kernel.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/modular_kernel.py
COPY vllm/models/glm5next/common/mtp.py /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/common/mtp.py
COPY vllm/models/glm5next/common/p_mtp_context.py /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/common/p_mtp_context.py
COPY vllm/v1/core/kv_cache_manager.py /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_manager.py
COPY vllm/v1/core/sched/hot_tail_budget.py /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/hot_tail_budget.py
COPY vllm/v1/core/sched/scheduler.py /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
COPY vllm/v1/worker/gpu/model_runner.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py
COPY vllm/v1/worker/gpu/spec_decode/mtp/speculator.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/mtp/speculator.py
COPY vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py
COPY vllm/v1/worker/gpu/warmup.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/warmup.py
ENV VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION=0 \
    VLLM_HOT_TAIL_ADMISSION_DIAGNOSTICS=0 \
    VLLM_GLM53_NVFP4_FUSED_MOE=0 \
    VLLM_GLM53_NVFP4_STAGE_FC2_INPUT=0 \
    VLLM_GLM53_NVFP4_VECTOR_WEIGHTS=0 \
    VLLM_GLM53_NVFP4_EXTENDED_PREFETCH=0 \
    VLLM_GLM53_P_MTP_CONTEXT_ONLY=0
RUN test -x /usr/local/cuda/bin/nvcc && python3 -c "from flashinfer.jit.core import gen_jit_spec, sm120a_nvcc_flags; from flashinfer.jit.env import FLASHINFER_CSRC_DIR; assert (FLASHINFER_CSRC_DIR / 'tvm_ffi_utils.h').is_file(); assert '-gencode=arch=compute_120a,code=sm_120a' in sm120a_nvcc_flags"
RUN python3 -c "import compileall,hashlib,json,pathlib; m=json.loads(pathlib.Path('/opt/glm53flash-1103-release.json').read_text()); r=pathlib.Path('/usr/local/lib/python3.12/dist-packages'); bad=[x['path'] for x in m['runtime_source_files'] if hashlib.sha256((r/x['path']).read_bytes()).hexdigest()!=x['sha256']]; assert not bad, bad; assert all(compileall.compile_file(str(r/x['path']),quiet=1) for x in m['image_overlay_files'] if x['path'].endswith('.py'))"
LABEL glm.release="glm5.3-flash-1103"
