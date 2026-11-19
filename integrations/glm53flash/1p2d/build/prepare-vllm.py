#!/usr/bin/env python3
"""Generate a full-build Dockerfile from the archived vLLM upstream recipe."""
import pathlib
import sys

source, dest = map(pathlib.Path, sys.argv[1:])
text = (source / "docker/Dockerfile").read_text()

def replace_once(old, new):
    global text
    if text.count(old) != 1:
        raise SystemExit(f"unexpected Dockerfile revision; nonunique anchor {old!r}")
    text = text.replace(old, new, 1)

# git archive has no .git. Use a fixed version for both Rust and Python metadata.
version = "0.32.0+glm53flash.67ca8f23"
replace_once(
    '    --mount=type=bind,source=.git,target=.git \\\n'
    '    SETUPTOOLS_SCM_PRETEND_METADATA="{dirty=false}" bash tools/build_rust.sh',
    f'    SETUPTOOLS_SCM_PRETEND_VERSION="{version}" bash tools/build_rust.sh',
)
replace_once(
    'ARG GIT_REPO_CHECK=0\nRUN --mount=type=bind,source=.git,target=.git \\\n'
    '    if [ "$GIT_REPO_CHECK" != "0" ]; then bash tools/check_repo.sh ; fi',
    f'ENV VLLM_VERSION_OVERRIDE="{version}"',
)
replace_once('    --mount=type=bind,source=.git,target=.git \\\n'
             '    if [ "${vllm_target_device}" = "cuda" ]; then \\',
             '    if [ "${vllm_target_device}" = "cuda" ]; then \\')
assert 'source=.git,target=.git' not in text

# Resolve the exact freshly built wheel alongside the CUDA requirements; never
# fetch the same-named FlashInfer wheel or its precompiled kernel packs.
replace_once(
    'COPY requirements/cuda.txt /tmp/requirements-cuda.txt',
    'COPY requirements/cuda.txt /tmp/requirements-cuda.txt\n'
    'COPY --from=flashinfer_wheels / /tmp/glm53-flashinfer-wheels/\n'
    'RUN set -- /tmp/glm53-flashinfer-wheels/flashinfer_python-0.7.1-*.whl; \\\n'
    '    test "$#" = 1 && test -f "$1" && \\\n'
    '    grep -qx "flashinfer-python==0.7.1" /tmp/requirements-cuda.txt && \\\n'
    '    sed -i "s|^flashinfer-python==0.7.1$|flashinfer-python @ file://$1|" /tmp/requirements-cuda.txt',
)
replace_once(
    'RUN --mount=type=cache,target=/opt/pip/cache \\\n'
    '    PIP_CACHE_DIR=/opt/pip/cache flashinfer download-kernels',
    '# FlashInfer kernels use the pinned source wheel and the local CUDA JIT.',
)
text += r'''

# The vLLM parent below is compiled in this same build (csrc-build and rust-build).
FROM vllm-openai AS glm53flash-source
COPY --from=build_helpers /verify-engine.py /opt/glm53/verify-engine.py
COPY integrations/glm53flash/release.json /opt/glm53/release.json
RUN apt-get update && apt-get install -y --no-install-recommends \
      ibverbs-providers librdmacm1 liburing2 libcurl4t64 build-essential && apt-get clean
RUN --mount=type=bind,from=mooncake_wheels,source=/,target=/tmp/mooncake-wheels,ro \
    uv pip install --system /tmp/mooncake-wheels/mooncake_transfer_engine_cuda13-0.3.13.post1-*.whl
RUN python3 /opt/glm53/verify-engine.py
LABEL org.opencontainers.image.revision="67ca8f23ae3192992af6b099b3cc8041316fe787"
'''

dest.write_text(text)
