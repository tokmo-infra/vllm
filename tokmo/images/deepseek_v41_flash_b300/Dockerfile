# Build from the repository root with the verified v0.32.0 base image.
ARG BASE_IMAGE=vllm/vllm-openai:v0.32.0
FROM ${BASE_IMAGE}

ARG SOURCE_REVISION=unknown
LABEL ai-infra.base="vllm/vllm-openai:v0.32.0" \
      ai-infra.change="exact DSV41 native CPU segment cache and renderer timing; no GPU/model changes" \
      cn.tokmo.source.revision="${SOURCE_REVISION}"

# Refuse a different renderer before installing the extracted image sources.
RUN echo 'ec2fe84538bb30ad7ed1e10833081c8c389e6a2f8fda1dcef50005b9db75a1a7  /usr/local/lib/python3.12/dist-packages/vllm/renderers/base.py' | sha256sum -c -

COPY vllm/renderers/base.py /usr/local/lib/python3.12/dist-packages/vllm/renderers/base.py
COPY vllm/tokenizers/dsv41_native_cache.py vllm/tokenizers/dsv41_segment_cache_core.py /usr/local/lib/python3.12/dist-packages/vllm/tokenizers/
COPY tokmo/images/deepseek_v41_flash_b300/source.sha256 /opt/tokmo/source.sha256
RUN cd /usr/local/lib/python3.12/dist-packages \
    && sha256sum -c /opt/tokmo/source.sha256 \
    && uv venv --python /usr/bin/python3.12 --system-site-packages /tmp/.venv \
    && /tmp/.venv/bin/python -m compileall -q \
        vllm/renderers/base.py \
        vllm/tokenizers/dsv41_native_cache.py \
        vllm/tokenizers/dsv41_segment_cache_core.py \
    && rm -rf /tmp/.venv
