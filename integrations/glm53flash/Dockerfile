# Rebuild the same native target; retain the base image's other extensions.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS native
ARG BUILD_JOBS=24
COPY sources/vllm /build/source
RUN cmake -S /build/source -B /build/cmake -G Ninja \
    -DVLLM_PYTHON_EXECUTABLE=/usr/bin/python3 \
    -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=cuda \
    '-DCMAKE_CUDA_ARCHITECTURES=100;120' \
    '-DVLLM_CUDA_ARCHS=10.0;12.0' \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DVLLM_CUTLASS_SRC_DIR=/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON && \
    cmake --build /build/cmake --target _C_stable_libtorch -j${BUILD_JOBS}

FROM ${BASE_IMAGE}
COPY --from=native /build/cmake/_C_stable_libtorch.abi3.so /usr/local/lib/python3.12/dist-packages/vllm/
COPY sources/vllm/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY sources/flashinfer/flashinfer/ /usr/local/lib/python3.12/dist-packages/flashinfer/
# Drop the removed experimental indexer even when rebuilding on an older image.
RUN rm -f /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/sm120_dsa_adapter.py /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/sm120_dsa_prefill.py && rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer/experimental/sm120_dsa_indexer /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/__pycache__ /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/__pycache__ /usr/local/lib/python3.12/dist-packages/flashinfer/__pycache__
COPY integrations/glm53flash/manifest.json /opt/glm53flash-source-manifest.json
