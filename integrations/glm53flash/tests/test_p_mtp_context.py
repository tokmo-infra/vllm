# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer-only MTP loading and default decode behavior, without vLLM imports."""

import ast
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

HELPER = "vllm/models/glm5next/common/p_mtp_context.py"
MTP = "vllm/models/glm5next/common/mtp.py"
SPEC = "vllm/v1/worker/gpu/spec_decode/mtp/speculator.py"
RUNNER = "vllm/v1/worker/gpu/model_runner.py"
TARGET_DEPENDENT_AR = "vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py"


@pytest.fixture
def helper(source_module, monkeypatch):
    module = source_module(HELPER)
    monkeypatch.delenv(module.OPT_IN, raising=False)
    return module


@pytest.fixture
def p_config():
    hf = NS(
        model_type="glm5_next_text",
        hidden_size=4096,
        moe_intermediate_size=2048,
        n_routed_experts=288,
        num_experts_per_tok=8,
        n_shared_experts=1,
        num_hidden_layers=45,
        num_nextn_predict_layers=1,
        first_k_dense_replace=3,
        q_lora_rank=1536,
        kv_lora_rank=512,
        mla_use_nope=True,
        index_kpool=4,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=2048,
    )
    checkpoint = dict(model="glm", revision=None, code_revision=None, quantization=None)
    return NS(
        use_v2_model_runner=True,
        model_config=NS(hf_text_config=hf, model_weights=None, **checkpoint),
        speculative_config=NS(
            method="mtp",
            num_speculative_tokens=3,
            parallel_drafting=False,
            enable_adaptive_verification=False,
            draft_tensor_parallel_size=1,
            draft_load_config=None,
            max_model_len=None,
            draft_model_config=NS(hf_text_config=hf, **checkpoint),
            glm_mtp_has_native_attn_overrides=lambda: True,
            uses_dynamic_speculative_decoding=lambda: False,
        ),
        parallel_config=NS(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=True,
            data_parallel_size=4,
            enable_eplb=False,
            use_sequence_parallel_moe=False,
        ),
        kv_transfer_config=NS(
            kv_connector="MultiConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "MooncakeConnector", "kv_role": "kv_producer"},
                    {"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both"},
                ]
            },
        ),
    )


@pytest.mark.parametrize("flag", [None, "0"])
def test_default_off_never_inspects_configuration(helper, monkeypatch, flag):
    if flag is not None:
        monkeypatch.setenv(helper.OPT_IN, flag)
    assert helper.p_context_enabled(object()) is False


@pytest.mark.parametrize("flag", ["", "true", "2", " 1"])
def test_opt_in_requires_an_exact_boolean_flag(helper, monkeypatch, flag):
    monkeypatch.setenv(helper.OPT_IN, flag)
    with pytest.raises(ValueError, match="0 or 1"):
        helper.p_context_enabled(object())


@pytest.mark.parametrize(
    ("dp_size", "ep", "depth"),
    [(1, False, 1), (2, True, 2), (4, True, 3), (8, True, 5)],
)
def test_dedicated_glm_producer_keeps_independent_ep_and_mtp_settings(
    helper, monkeypatch, p_config, dp_size, ep, depth
):
    monkeypatch.setenv(helper.OPT_IN, "1")
    p_config.parallel_config.data_parallel_size = dp_size
    p_config.parallel_config.enable_expert_parallel = ep
    p_config.speculative_config.num_speculative_tokens = depth
    assert helper.p_context_enabled(p_config)
    assert p_config.speculative_config.num_speculative_tokens == depth


@pytest.mark.parametrize(
    "scope,field,value",
    [
        ("root", "use_v2_model_runner", False),
        ("hf", "hidden_size", 5120),
        ("hf", "model_type", "llama"),
        ("hf", "num_nextn_predict_layers", 2),
        ("parallel", "tensor_parallel_size", 2),
        ("parallel", "enable_eplb", True),
        ("parallel", "use_sequence_parallel_moe", True),
        ("spec", "draft_tensor_parallel_size", 2),
        ("spec", "draft_load_config", object()),
        ("spec", "parallel_drafting", True),
        ("spec", "max_model_len", 2048),
        ("spec", "enable_adaptive_verification", True),
        ("spec", "glm_mtp_has_native_attn_overrides", lambda: False),
        ("spec", "uses_dynamic_speculative_decoding", lambda: True),
        ("draft", "model", "different-checkpoint"),
        ("draft", "revision", "different-revision"),
        ("draft", "quantization", "fp8"),
        ("connector", "kv_connector", "MooncakeConnector"),
        ("connector", "kv_role", "kv_consumer"),
    ],
)
def test_context_mode_falls_back_to_full_mtp_for_unsupported_or_decode_configs(
    helper, monkeypatch, p_config, scope, field, value, caplog
):
    monkeypatch.setenv(helper.OPT_IN, "1")
    target = {
        "root": p_config,
        "hf": p_config.model_config.hf_text_config,
        "parallel": p_config.parallel_config,
        "spec": p_config.speculative_config,
        "draft": p_config.speculative_config.draft_model_config,
        "connector": p_config.kv_transfer_config,
    }[scope]
    setattr(target, field, value)
    assert helper.p_context_enabled(p_config) is False
    assert "using full MTP" in caplog.text
    assert p_config.speculative_config.num_speculative_tokens == 3


@pytest.mark.parametrize("extra_child", [False, True])
def test_consumer_or_mixed_connector_cannot_enable_p_pruning(
    helper, monkeypatch, p_config, extra_child, caplog
):
    monkeypatch.setenv(helper.OPT_IN, "1")
    children = p_config.kv_transfer_config.kv_connector_extra_config["connectors"]
    if extra_child:
        children.append({"kv_connector": "MooncakeConnector", "kv_role": "kv_consumer"})
    else:
        children[0]["kv_role"] = "kv_consumer"
    assert helper.p_context_enabled(p_config) is False
    assert "never D or mixed roles" in caplog.text


def test_exact_mtp45_partition_keeps_attention_and_never_drops_base_experts(helper):
    prefix = "model.layers.45."
    required = {
        prefix + suffix
        for suffix in (
            "eh_proj.weight",
            "enorm.weight",
            "hnorm.weight",
            "input_layernorm.weight",
            "self_attn.indexer.index_kpool_compress_ape",
            "self_attn.indexer.index_kpool_compress_gate",
            "self_attn.indexer.k_norm.bias",
            "self_attn.indexer.k_norm.weight",
            "self_attn.indexer.weights_proj.weight",
            "self_attn.indexer.wk.weight",
            "self_attn.indexer.wq_b.weight",
            "self_attn.kv_a_layernorm.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.q_a_proj.weight",
            "self_attn.q_b_proj.weight",
        )
    }
    dropped = {
        f"{prefix}mlp.experts.{e}.{proj}.weight"
        for e in range(288)
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    dropped |= {
        f"{prefix}mlp.shared_experts.{proj}.weight"
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    dropped |= {
        prefix + suffix
        for suffix in (
            "mlp.gate.weight",
            "mlp.gate.e_score_correction_bias",
            "post_attention_layernorm.weight",
            "shared_head.norm.weight",
        )
    }
    assert required == helper.REQUIRED_KEYS and len(required) == 18
    assert dropped == helper.DROPPED_KEYS and len(dropped) == 871
    assert not required & dropped
    coverage = helper.ContextWeightCoverage()
    for key in required:
        assert not coverage.discard(key.replace("model.", "model.language_model.", 1))
    assert all(coverage.discard(key) for key in dropped)
    assert not coverage.discard("model.layers.44.mlp.experts.0.gate_proj.weight")
    coverage.finish({"loaded_attention": None}, {"loaded_attention"}, {})


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "unloaded", "pending", "unknown"]
)
def test_loading_coverage_rejects_missing_duplicate_or_unhandled_weights(
    helper, failure
):
    coverage = helper.ContextWeightCoverage()
    for key in sorted(helper.REQUIRED_KEYS)[failure == "missing" :]:
        coverage.discard(key)
    with pytest.raises(ValueError):
        if failure == "duplicate":
            coverage.discard(next(iter(helper.REQUIRED_KEYS)))
        elif failure == "unknown":
            coverage.discard("model.layers.45.mlp.experts.288.up_proj.weight")
        else:
            coverage.finish(
                {"p": None} if failure == "unloaded" else {},
                set(),
                {"partial_fp8": None} if failure == "pending" else {},
            )


@pytest.mark.parametrize("producer", [False, True])
@pytest.mark.parametrize("depth", [1, 3, 5])
def test_prefill_only_skips_sampling_on_p_and_keeps_d_native_path(
    source_tree, execute_ast, producer, depth
):
    base_prefill = Mock(return_value="native_decode_path")
    base = type("NativeSpeculator", (), {"_prefill": base_prefill})
    cls = next(n for n in source_tree(SPEC).body if isinstance(n, ast.ClassDef))
    speculator = execute_ast(
        [cls],
        {
            "TargetDependentARSpeculator": base,
            "CUDAGraphMode": NS(NONE="none"),
        },
    )["MTPSpeculator"]()
    speculator.model = NS(p_mtp_context_only=producer)
    speculator._run_model = Mock()
    speculator.draft_tokens = torch.full((3, depth), 7, dtype=torch.int64)
    args = (2, 5, {"attention": "unchanged"}, {"slots": "unchanged"}, None)
    result = speculator._prefill(*args)
    if producer:
        speculator._run_model.assert_called_once_with(*args[1:], "none", None)
        base_prefill.assert_not_called()
        assert result is None and torch.equal(
            speculator.draft_tokens[:2], torch.full((2, depth), -1)
        )
        assert (speculator.draft_tokens[2] == 7).all()
    else:
        base_prefill.assert_called_once_with(*args, "none", None)
        speculator._run_model.assert_not_called()
        assert result == "native_decode_path" and (speculator.draft_tokens == 7).all()


@pytest.mark.parametrize("producer,depth", [(True, 5), (False, 1), (False, 5)])
def test_context_only_captures_prefill_but_never_the_recycle_graph(
    source_tree, execute_ast, producer, depth
):
    cls = next(
        n
        for n in source_tree(TARGET_DEPENDENT_AR).body
        if isinstance(n, ast.ClassDef) and n.name == "TargetDependentARSpeculator"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "capture"
    )
    capture = execute_ast([method], {"logger": Mock()})["capture"]
    prefill, decode = Mock(use_breakable_cg=False), Mock()
    runner = Mock(
        num_speculative_steps=depth,
        p_mtp_context_only=producer,
        supports_decode_metadata_update=True,
        prefill_cudagraph_manager=prefill,
        decode_cudagraph_manager=decode,
    )
    capture(runner)
    prefill.capture.assert_called_once()
    assert prefill.capture.call_args.args[4] is runner.target_attn_groups
    if producer or depth == 1:
        decode.capture.assert_not_called()
    else:
        decode.capture.assert_called_once()
        assert decode.capture.call_args.args[4] is runner.attn_groups


@pytest.mark.parametrize("method_name", ["compute_logits", "get_top_tokens"])
def test_p_sampling_is_forbidden_but_d_shared_head_is_preserved(
    source_tree, execute_ast, method_name
):
    cls = next(
        n
        for n in source_tree(MTP).body
        if isinstance(n, ast.ClassDef) and n.name == "Glm5NextMultiTokenPredictor"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    fn = execute_ast([method], {})[method_name]
    head, hidden, answer = object(), object(), object()
    processor = Mock(return_value=answer)
    processor.get_top_tokens.return_value = answer
    model = NS(
        p_mtp_context_only=False,
        num_mtp_layers=1,
        _mtp_layers=[NS(shared_head=NS(head=head))],
        logits_processor=processor,
    )
    for step in range(3):
        assert fn(model, hidden, step) is answer
    called = processor if method_name == "compute_logits" else processor.get_top_tokens
    assert called.call_count == 3 and all(
        c.args == (head, hidden) for c in called.call_args_list
    )
    model.p_mtp_context_only = True
    with pytest.raises(RuntimeError, match="never be sampled"):
        fn(model, hidden)


@pytest.mark.parametrize("producer", [False, True])
def test_p_completion_uses_device_event_order_not_host_synchronization(
    source_tree, execute_ast, producer
):
    tree = source_tree(RUNNER)
    guards = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and "p_mtp_context_only" in ast.unparse(n.test)
        and any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "wait_stream"
            for c in ast.walk(n)
        )
    ]
    assert len(guards) == 1
    guard = guards[0]
    calls = []
    stream = NS(wait_stream=lambda value: calls.append(("wait", value)))
    runner = NS(
        speculator=NS(p_mtp_context_only=producer),
        output_copy_stream=stream,
        main_stream="main",
    )
    event = NS(record=lambda value: calls.append(("record", value)))
    execute_ast([guard], {"self": runner, "async_output": NS(copy_event=event)})
    assert calls == ([("wait", "main"), ("record", stream)] if producer else [])
    enclosing = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and guard in n.body
    )
    assert "kv_connector.post_forward" in ast.unparse(
        enclosing.body[enclosing.body.index(guard) + 1]
    )
