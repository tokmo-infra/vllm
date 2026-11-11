# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check gathered-M tuning bounds without changing local dummy allocations."""

import ast
import builtins
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, call, patch

import pytest


@pytest.mark.parametrize("dp", [1, 4, 8])
@pytest.mark.parametrize("draft_queries", [None, 6])
@pytest.mark.parametrize("skip_attn", [False, True])
def test_dp_tuning_covers_gathered_drafts_but_dummy_stays_local(
    source_tree, execute_ast, dp, draft_queries, skip_attn
):
    path = "vllm/model_executor/warmup/kernel_warmup.py"
    fn = next(
        n
        for n in source_tree(path).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_run_flashinfer_autotune_dummy_runs"
    )
    fi = NS(
        flashinfer_get_hybrid_num_tokens_buckets=MagicMock(side_effect=lambda n: (n,)),
        autotune=MagicMock(),
    )
    run = execute_ast(
        [fn],
        {
            "logger": MagicMock(),
            "_flashinfer_autotune_token_counts": lambda _: (2048, 128),
        },
    )[fn.name]
    runner = NS(
        vllm_config=NS(parallel_config=NS(data_parallel_size=dp)),
        max_num_reqs=24,
        max_num_tokens=2048,
        speculator=NS(num_query_per_req=draft_queries) if draft_queries else None,
        _dummy_run=MagicMock(),
    )
    native_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "vllm.utils.flashinfer":
            return NS(utils=NS(flashinfer=fi))
        return native_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        run(runner, skip_attn=skip_attn)
    maxima = [2048, max(128, 24 * (draft_queries or 1))]
    assert fi.flashinfer_get_hybrid_num_tokens_buckets.call_args_list == [
        call(n * dp) for n in maxima
    ]
    assert runner._dummy_run.call_args_list == [
        call(
            num_tokens=n,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
            **({"skip_attn": True} if skip_attn else {}),
        )
        for n in (2048, 128)
    ]
    assert fi.autotune.call_args_list == [
        call(tuning_buckets=(n * dp,), round_up=True) for n in maxima
    ]
