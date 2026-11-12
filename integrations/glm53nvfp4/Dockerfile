ARG VLLM_BASE=vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e
FROM ${VLLM_BASE}

ARG SOURCE_REVISION=uncommitted
LABEL org.opencontainers.image.source="https://github.com/tokmo-infra/vllm" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      ai.tokmo.engine.base="vllm-0.33.0-4ba787ec" \
      ai.tokmo.engine.branch="glm5.3-nvfp4-1113"

COPY vllm /tmp/glm53-overlay/vllm
COPY integrations/glm53nvfp4 /tmp/glm53-overlay/integrations/glm53nvfp4
RUN uv venv /opt/glm53-venv --system-site-packages \
    && cd / \
    && /opt/glm53-venv/bin/python \
        /tmp/glm53-overlay/integrations/glm53nvfp4/install_overlay.py \
        --source /tmp/glm53-overlay \
    && rm -rf /tmp/glm53-overlay
ENV PATH="/opt/glm53-venv/bin:${PATH}"
