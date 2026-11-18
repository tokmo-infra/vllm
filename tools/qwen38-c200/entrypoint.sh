#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -eu
case "${VLLM_QWEN38_PD_ROLE:-}" in
    prefill) export PYTHONPATH="/opt/qwen38-prefill${PYTHONPATH:+:$PYTHONPATH}" ;;
    decode) ;;
    *) echo 'Set VLLM_QWEN38_PD_ROLE to prefill or decode' >&2; exit 2 ;;
esac
exec vllm serve "$@"
