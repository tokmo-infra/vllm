# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen4ExpConfig, Qwen4ExpTextConfig

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.config import (
    Qwen3_5ForConditionalGenerationConfig,
    Qwen4ExpForConditionalGenerationConfig,
)
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

from ...utils import spawn_new_process_for_each_test


def _text_config(**kwargs) -> Qwen4ExpTextConfig:
    values = {
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        # `Qwen4ExpTextConfig` requires an EOS token whenever PLE is enabled.
        "eos_token_id": 1,
        "hc_count": 2,
        "hc_lowrank": 4,
        "ple_layer_ids": [1],
        "mtp_num_hidden_layers": 1,
        "mtp": {"hybrid": True},
    }
    values.update(kwargs)
    return Qwen4ExpTextConfig(**values)


@pytest.fixture
def qwen_mtp_config(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    _text_config(
        architectures=["Qwen4ExpForCausalLM"],
        ple_layer_ids=[],
    ).save_pretrained(tmp_path)

    def build(steps, opt_in=False, kv=None, **spec_options):
        model = ModelConfig(
            model=str(tmp_path),
            dtype="bfloat16",
            max_model_len=128,
            skip_tokenizer_init=True,
        )
        parallel = ParallelConfig()
        return VllmConfig(
            model_config=model,
            parallel_config=parallel,
            device_config=DeviceConfig("cpu"),
            cache_config=CacheConfig(cache_dtype="fp8_e4m3"),
            kv_transfer_config=kv,
            speculative_config=SpeculativeConfig(
                method="mtp",
                num_speculative_tokens=steps,
                qwen38_p_mtp_context_only=opt_in,
                target_model_config=model,
                target_parallel_config=parallel,
                **spec_options,
            ),
        )

    return build


@pytest.mark.parametrize("steps", [1, 2, 5])
@pytest.mark.parametrize("role", [None, "kv_producer", "kv_consumer"])
def test_qwen_mtp_native_depth_and_role(qwen_mtp_config, role, steps):
    kv = (
        KVTransferConfig(kv_connector="MooncakeConnector", kv_role=role)
        if role
        else None
    )
    native = qwen_mtp_config(steps, kv=kv)
    assert not native.qwen38_p_mtp_context_only
    assert native.num_speculative_tokens == steps
    assert native.uniform_decode_query_len == steps + 1
    assert native.num_prefill_lookahead_tokens == 1
    if role != "kv_producer":
        with pytest.raises(ValueError, match="dedicated Mooncake P2P producer"):
            qwen_mtp_config(steps, opt_in=True, kv=kv)


def test_qwen_mtp_context_store_namespace_is_not_available_to_baseline_p(
    qwen_mtp_config,
):
    store = {
        "kv_connector": "MooncakeStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"cache_prefix": "qwen38-mtp-context-v1/test"},
    }

    def kv(role):
        return KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "MooncakeConnector", "kv_role": role},
                    store,
                ]
            },
        )

    with pytest.raises(ValueError, match="reserved Store cache_prefix"):
        qwen_mtp_config(2, kv=kv("kv_producer"))
    d = qwen_mtp_config(2, kv=kv("kv_consumer"))
    assert not d.qwen38_p_mtp_context_only and d.num_hash_lookahead_tokens == 1
    with pytest.raises(ValueError, match="requires one hash lookahead token"):
        qwen_mtp_config(2, kv=kv("kv_consumer"), disable_lookahead_block_hashes=True)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA context prefill")
@pytest.mark.parametrize("steps", [1, 2, 5])
def test_qwen_mtp_context_prefill_keeps_every_row(qwen_mtp_config, steps):
    from dataclasses import replace

    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
        prepare_prefill_inputs,
    )

    kv = KVTransferConfig(kv_connector="MooncakeConnector", kv_role="kv_producer")
    config = qwen_mtp_config(steps, opt_in=True, kv=kv)
    assert config.qwen38_p_mtp_context_only
    with pytest.raises(ValueError, match="one hash lookahead token"):
        qwen_mtp_config(steps, opt_in=True, kv=kv, disable_lookahead_block_hashes=True)
    # Head quantization needs the real native head; don't apply context slimming.
    full = qwen_mtp_config(steps, opt_in=True, kv=kv, draft_lm_head_quantization="fp8")
    assert not full.qwen38_p_mtp_context_only
    assert full.num_speculative_tokens == steps

    dev = torch.device("cuda")
    speculator = MTPSpeculator(config, dev)
    assert speculator.draft_tokens.shape[1] == steps

    target = InputBuffers(3, 7, dev)
    batch = InputBatch.make_dummy(3, 7, target)
    # One one-token final prefill (no truncation), a middle chunk, and the
    # final chunk of a GDN P request. Each has a different next-token source.
    batch = replace(
        batch, mtp_context_next_token_ids=torch.tensor([-1, 41, 42], device=dev)
    )
    batch.input_ids.copy_(torch.tensor([5, 6, 7, 8, 9, 10, 11], device=dev))
    batch.positions.copy_(torch.tensor([0, 5, 6, 13, 14, 15, 16], device=dev))
    batch.query_start_loc.copy_(torch.tensor([0, 1, 3, 7], device=dev))
    batch.seq_lens.copy_(torch.tensor([1, 7, 17], device=dev))
    last_indices = prepare_prefill_inputs(
        speculator.last_token_indices,
        speculator.current_draft_step,
        speculator.input_buffers,
        batch,
        torch.tensor([1, 0, 1], device=dev),
        torch.zeros(3, dtype=torch.int32, device=dev),
        torch.tensor([51, 52, 53], device=dev),
        torch.tensor([[31, 32, 33]], device=dev),
        speculator.max_num_reqs,
    )
    assert last_indices[:3].tolist() == [0, 2, 6]
    assert speculator.input_buffers.input_ids[:7].tolist() == [51, 7, 32, 9, 10, 11, 42]
    assert speculator.input_buffers.positions[:7].tolist() == batch.positions.tolist()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA output fence")
def test_qwen_mtp_context_output_waits_for_late_compute():
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    main = torch.cuda.current_stream()
    copy = torch.cuda.Stream()
    ids = torch.tensor([[7]], device="cuda")
    count = torch.ones(1, dtype=torch.int32, device="cuda")
    ready = torch.cuda.Event()
    output = ModelRunnerOutput(["p"], {"p": 0})
    pending = AsyncOutput(
        output,
        SamplerOutput(ids, None, None, count, torch.zeros_like(count)),
        count,
        main,
        copy,
        check_ep_fault=False,
        pending_aux_output=None,
    )
    # Enqueue the late context work separately from the early sampler copy.
    context = torch.empty(128 * 1024, device="cuda")
    context.fill_(9)
    ready.record(main)
    pending.wait_for_compute(main, copy)
    output.kv_connector_output = KVConnectorOutput(finished_sending={"old"})
    result = pending.get_output()
    assert ready.query()
    assert result.sampled_token_ids == [[7]]
    assert result.kv_connector_output.finished_sending == {"old"}


def test_moe_sp_rejects_pipeline_parallel() -> None:
    """Automatic MoE SP must validate HC topology even without --enable-hc-sp."""
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=_text_config(num_experts=4, ple_layer_ids=[]),
        ),
        cache_config=CacheConfig(),
        parallel_config=ParallelConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2,
            enable_expert_parallel=True,
            all2all_backend="allgather_reducescatter",
        ),
    )
    with (
        patch("vllm.platforms.current_platform.is_cuda_alike", return_value=True),
        pytest.raises(ValueError, match="MoE SP requires PP=1"),
    ):
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(config)


@spawn_new_process_for_each_test
@pytest.mark.parametrize("backend", ["amd", "nvidia"])
@pytest.mark.parametrize("use_sp", [False, True])
def test_qwen4_exp_mtp_returns_sample_and_multi_streams(
    backend: str, use_sp: bool
) -> None:
    """MTP returns complete, contiguous outputs after gathering padded SP shards."""
    mtp_module = f"vllm.models.qwen4_exp.{backend}.mtp"
    model = object.__new__(import_module(mtp_module).Qwen4ExpMultiTokenPredictor)
    model.use_hc_sequence_parallel = use_sp
    model.use_sequence_parallel = False
    torch.nn.Module.__init__(model)
    model.hc_count = 2
    model.hidden_size = 4
    model.num_mtp_layers = 1
    model.pre_fc_norm_embedding = torch.nn.Identity()
    model.pre_fc_norm_hidden = torch.nn.Identity()
    model.fc_embedding = torch.nn.Identity()
    model.fc_hidden = torch.nn.Identity()

    def decoder(**kwargs):
        """Check that the embedding residual shares the hidden state's token slice."""
        hidden = kwargs["hidden_states"]
        if backend == "nvidia":
            prev_block_output = kwargs["prev_block_output"]
            assert prev_block_output.shape == (hidden.shape[0], model.hidden_size)
        return hidden, hidden, torch.zeros(hidden.shape[0], model.hc_count)

    model.layers = [decoder]
    model.hyper_connection_mixer = SimpleNamespace(
        combine_and_mix=lambda hidden_states, block_output, injection: (
            hidden_states,
            hidden_states.unflatten(-1, (2, 4)).mean(-2),
            None,
        ),
    )
    multi_hidden = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    expected_sample = multi_hidden.unflatten(-1, (2, 4)).mean(-2)
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)

    def gather(local):
        """Supply rank 1's last token and padding for a simulated two-rank gather."""
        assert local.shape[0] == 2
        remote = torch.cat([expected_sample[2:], multi_hidden[2:]], dim=-1)
        return torch.cat([local, remote, torch.full_like(remote, float("nan"))])

    with (
        patch(f"{mtp_module}.get_pp_group", return_value=pp_group),
        patch(f"{mtp_module}.sp_shard", side_effect=lambda x: x[:2]),
        patch(f"{mtp_module}.sp_all_gather", side_effect=gather),
    ):
        sample_hidden, returned_multi_hidden = model.forward(
            input_ids=None,
            positions=torch.arange(3),
            hidden_states=multi_hidden,
            inputs_embeds=torch.zeros(3, 4),
        )

    torch.testing.assert_close(sample_hidden, expected_sample)
    torch.testing.assert_close(returned_multi_hidden, multi_hidden)
    assert sample_hidden.is_contiguous()
    assert returned_multi_hidden.is_contiguous()


@spawn_new_process_for_each_test
@pytest.mark.parametrize("backend", ["amd", "nvidia"])
def test_qwen4_exp_mtp_remaps_mixed_precision_layer_indices(backend: str) -> None:
    mtp_module = import_module(f"vllm.models.qwen4_exp.{backend}.mtp")

    quantized_layers = {
        "model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"},
        "mtp.layers.0.mlp.experts": {
            "quant_algo": "FP8_BLOCK_SCALES",
            "group_size": 128,
        },
    }

    assert mtp_module._remap_quantized_layers(quantized_layers, 48) == {
        "model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"},
        "mtp.layers.48.mlp.experts": {
            "quant_algo": "FP8_BLOCK_SCALES",
            "group_size": 128,
        },
    }


@pytest.mark.parametrize("wrapped_config", [False, True])
def test_qwen4_exp_mtp_override_sets_draft_config(
    wrapped_config: bool,
) -> None:
    text_config = _text_config(
        architectures=["Qwen4ExpForCausalLM"],
        index_share_for_mtp_iteration=True,
    )
    config = (
        Qwen4ExpConfig(
            architectures=["Qwen4ExpForConditionalGeneration"],
            text_config=text_config,
        )
        if wrapped_config
        else text_config
    )

    draft_config = SpeculativeConfig.hf_config_override(config)

    assert draft_config.index_share_for_mtp_iteration is True
    assert draft_config.model_type == "qwen4_exp_mtp"
    assert draft_config.architectures == ["Qwen4ExpMTP"]
    assert draft_config.hc_mult == 2
    assert draft_config.n_predict == 1


@pytest.mark.parametrize("ple_layer_ids", [[1], []])
def test_qwen4_exp_rejects_pipeline_parallel_only_with_ple(ple_layer_ids) -> None:
    """PLE needs raw input_ids, which non-first pipeline ranks never see. The
    rest of the architecture is PP-capable, so the refusal must be conditional
    -- and must land before the engine spends time loading weights."""
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=_text_config(ple_layer_ids=ple_layer_ids),
            multimodal_config=None,
        ),
        parallel_config=ParallelConfig(pipeline_parallel_size=2),
        speculative_config=None,
    )
    with patch.object(
        Qwen3_5ForConditionalGenerationConfig, "verify_and_update_config"
    ):
        if ple_layer_ids:
            with pytest.raises(NotImplementedError, match="pipeline_parallel_size=1"):
                Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(
                    vllm_config
                )
        else:
            Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(vllm_config)


def test_qwen4_exp_model_state_prepares_ngram_context() -> None:
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_context_len = 3
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((8, 3), dtype=torch.int32)
    model_state.ngram_context_offsets = torch.arange(-3, 0, dtype=torch.int64)
    model_state.ple_query_start_loc = torch.empty(9, dtype=torch.int32)

    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=3,
        idx_mapping=torch.tensor([1, 0]),
        query_start_loc=torch.tensor([0, 2, 3, 3], dtype=torch.int32),
    )
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([3, 1])),
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor([[1, 2, 3, 4], [20, 21, 22, 23]], dtype=torch.int32)
        ),
    )

    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    expected_query_start_loc = torch.full((9,), 3, dtype=torch.int32)
    expected_query_start_loc[0] = 0
    expected_query_start_loc[1] = 2
    torch.testing.assert_close(
        model_inputs["query_start_loc"], expected_query_start_loc
    )
    expected_context = torch.full((8, 3), 99, dtype=torch.int32)
    expected_context[:2] = torch.tensor([[99, 99, 20], [1, 2, 3]])
    torch.testing.assert_close(model_inputs["ngram_context"], expected_context)

    # Retain the views to detect reallocations as the request layout changes.
    query_start_loc = model_inputs["query_start_loc"]
    ngram_context = model_inputs["ngram_context"]
    input_batch.num_reqs = 1
    input_batch.num_reqs_after_padding = 1
    input_batch.idx_mapping = torch.tensor([0])
    input_batch.query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    expected_query_start_loc.fill_(3)
    expected_query_start_loc[0] = 0
    torch.testing.assert_close(
        model_inputs["query_start_loc"], expected_query_start_loc
    )
    expected_context.fill_(99)
    expected_context[0] = torch.tensor([1, 2, 3])
    torch.testing.assert_close(model_inputs["ngram_context"], expected_context)
    assert model_inputs["query_start_loc"].data_ptr() == query_start_loc.data_ptr()
    assert model_inputs["ngram_context"].data_ptr() == ngram_context.data_ptr()


def test_qwen4_exp_model_state_prepares_stable_dummy_ngram_inputs() -> None:
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((8, 3), dtype=torch.int32)
    model_state.ple_query_start_loc = torch.empty(9, dtype=torch.int32)

    with patch.object(MambaHybridModelState, "prepare_dummy_inputs", return_value={}):
        first = model_state.prepare_dummy_inputs(num_reqs=3, num_tokens=4)
        # Dummy runs establish the addresses used during CUDA graph capture.
        query_start_loc_ptr = first["query_start_loc"].data_ptr()
        ngram_context_ptr = first["ngram_context"].data_ptr()
        second = model_state.prepare_dummy_inputs(num_reqs=3, num_tokens=4)

    expected_query_start_loc = torch.full((9,), 4, dtype=torch.int32)
    expected_query_start_loc[:4] = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
    torch.testing.assert_close(second["query_start_loc"], expected_query_start_loc)
    torch.testing.assert_close(
        second["ngram_context"], torch.full((8, 3), 99, dtype=torch.int32)
    )
    assert second["query_start_loc"].data_ptr() == query_start_loc_ptr
    assert second["ngram_context"].data_ptr() == ngram_context_ptr
