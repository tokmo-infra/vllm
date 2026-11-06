# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.worker.gpu.async_utils import StepTimingSample
from vllm.v1.worker.gpu.spec_decode import adaptive_verification as adaptive_module
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    assign_draft_token_budget,
    maybe_create_adaptive_verification_manager,
    resolve_adaptive_cudagraph_mode,
)


def make_manager(
    confidences: np.ndarray, verify_cost_ms: np.ndarray
) -> AdaptiveVerificationManager:
    num_reqs, num_steps = confidences.shape
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.num_speculative_steps = num_steps
    manager._stale_confidences = [SimpleNamespace(np=confidences)]  # type: ignore[list-item]  # CPU-only confidence buffer.
    manager._stale_idx = 0
    manager.req_states = SimpleNamespace(
        req_id_to_index={"low": 0, "high": 1},
        num_computed_tokens_np=np.ones(num_reqs, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.ones(num_reqs, dtype=np.int32)),
    )
    manager.cost_tables = (np.zeros(num_reqs + 1), verify_cost_ms)
    manager._max_total_logits = 1 << 30
    manager.num_bonus_tokens = 1
    return manager


@pytest.mark.parametrize(
    ("mode", "piecewise_capture_available", "expected"),
    [
        ("FULL_DECODE_ONLY", True, "FULL_DECODE_ONLY"),
        ("FULL_DECODE_ONLY", False, "FULL_DECODE_ONLY"),
        ("FULL", True, "FULL_AND_PIECEWISE"),
        ("FULL", False, "FULL_DECODE_ONLY"),
        ("FULL_AND_PIECEWISE", True, "FULL_AND_PIECEWISE"),
        ("FULL_AND_PIECEWISE", False, "FULL_DECODE_ONLY"),
    ],
)
def test_resolve_adaptive_cudagraph_mode(mode, piecewise_capture_available, expected):
    assert (
        resolve_adaptive_cudagraph_mode(
            CUDAGraphMode[mode], piecewise_capture_available=piecewise_capture_available
        )
        == CUDAGraphMode[expected]
    )


@pytest.mark.parametrize(
    "target_support,target_bound,device_offsets,additional,error",
    [
        pytest.param(AttentionCGSupport.ALWAYS, None, True, None, None, id="always"),
        pytest.param(
            AttentionCGSupport.UNIFORM_BATCH, 8, True, None, None, id="bounded"
        ),
        pytest.param(
            AttentionCGSupport.UNIFORM_BATCH,
            None,
            True,
            None,
            "TargetBackend allows none",
            id="unsupported",
        ),
        pytest.param(
            AttentionCGSupport.UNIFORM_BATCH,
            7,
            True,
            None,
            "up to 8, but TargetBackend allows at most 7",
            id="too-narrow",
        ),
        pytest.param(
            AttentionCGSupport.UNIFORM_BATCH,
            8,
            False,
            None,
            "trims verification requests",
            id="host-query-lens",
        ),
        pytest.param(
            AttentionCGSupport.ALWAYS,
            None,
            True,
            (AttentionCGSupport.UNIFORM_BATCH, "EncoderBackend"),
            "EncoderBackend allows none",
            id="additional-group",
        ),
    ],
)
def test_manager_checks_target_varlen_cudagraph_bound(
    monkeypatch, target_support, target_bound, device_offsets, additional, error
):
    """Target builders must replay varlen decode graphs of the verification
    width; draft-only groups are not checked."""

    def group(backend_name, layer_name, support, bound=None, mismatch=True):
        class Builder(AttentionMetadataBuilder):
            _cudagraph_support = support

            def __init__(self):
                pass

            def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
                raise NotImplementedError

            if bound is not None:

                @classmethod
                def get_varlen_cudagraph_max_query_len(cls, *_args):
                    return bound

        backend = type(
            backend_name,
            (),
            {"supports_device_cpu_query_lens_mismatch": staticmethod(lambda: mismatch)},
        )
        builder = Builder()
        return SimpleNamespace(
            layer_names=[layer_name],
            backend=backend,
            kv_cache_spec=None,
            get_metadata_builder=lambda _index: builder,
        )

    groups = [
        [
            group(
                "TargetBackend", "target", target_support, target_bound, device_offsets
            ),
            group("DraftBackend", "draft", AttentionCGSupport.UNIFORM_BATCH),
        ]
    ]
    created = object()
    monkeypatch.setattr(
        adaptive_module,
        "AdaptiveVerificationManager",
        lambda *_args, **_kwargs: created,
    )

    with pytest.raises(ValueError, match=error) if error else nullcontext():
        manager = maybe_create_adaptive_verification_manager(
            enable_adaptive_verification=True,
            attn_groups=groups,
            req_states=SimpleNamespace(num_speculative_steps=7),
            query_start_loc=object(),
            num_bonus_tokens=1,
            max_total_logits=1,
            vllm_config=None,
            target_layer_names={"target"},
            additional_attn_cg_support=additional,
        )
        assert manager is created


def test_budget_stops_where_marginal_drafts_stop_paying_for_themselves():
    # Verification is cheap up to two extra tokens, then jumps 100x; only the
    # highest-confidence draft is worth the cheap slot.
    manager = make_manager(
        np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32),
        np.array([1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0]),
    )

    manager.get_num_tokens(
        {"low": 3, "high": 3},
        {"low": [1, 2], "high": [3, 4]},
    )
    assert manager._batch_budget is not None
    valid_drafts, num_non_draft_tokens, draft_budget = manager._batch_budget

    assert draft_budget == 1
    assert valid_drafts == {"low": 2, "high": 2}
    assert num_non_draft_tokens == {"low": 1, "high": 1}


def test_profiled_batches_seed_cost_curves_via_consumer(monkeypatch):
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.req_states = SimpleNamespace(max_num_batched_tokens=4096, max_num_reqs=64)
    manager.num_speculative_steps = 7
    manager.num_bonus_tokens = 1
    curves: dict[str, list[tuple[int, float]]] = {}
    monkeypatch.setattr(
        manager,
        "set_cost_curves",
        lambda draft, verify: curves.update(draft=draft, verify=verify),
    )

    timings = [
        StepTimingSample(
            forward_ms=float(batch["num_tokens"]),
            drafter_ms=1.0,
            num_target_tokens=batch["num_tokens"],
            num_reqs=batch["num_tokens"] // 8,
            # Only the captured sizes replay a graph; the tail sizes run eager.
            full_cudagraph=batch["num_tokens"] <= 1024,
        )
        for batch in manager.batches_to_profile([8, 1024])
    ]
    manager.set_initial_cost_curves(timings)

    # Tail beyond the last capture size: 1.5x then doubling to the max.
    assert curves["verify"] == [
        (8, 8.0),
        (1024, 1024.0),
        (1536, 1536.0),
        (2048, 2048.0),
        (4096, 4096.0),
    ]
    # Eager batches must not contribute to the draft curve: keyed by request
    # count they would land inside the captured range and, once made monotonic,
    # smear that eager cost across every larger request count.
    assert curves["draft"] == [(1, 1.0), (128, 1.0)]


def test_compact_batch_preserves_totals_and_bounds():
    # The CPU placeholder layout must keep the batch total equal to the GPU
    # total and every verification row within decode_query_len, or downstream
    # CPU metadata desyncs from the reallocated GPU boundaries.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9], [1.0, 1.0]], dtype=np.float32),
        np.array([1.0] * 44 + [100.0] * 3),
    )
    manager.req_states.req_id_to_index["prefill"] = 2
    manager.req_states.num_computed_tokens_np = np.zeros(3, dtype=np.int32)
    manager.req_states.prefill_len.np = np.array([0, 0, 60], dtype=np.int32)
    num_tokens = manager.get_num_tokens(
        {"low": 3, "high": 3, "prefill": 40},
        {"low": [1, 2], "high": [3, 4]},
    )
    scheduled = np.array([3, 3, 40], dtype=np.int32)
    drafts = np.array([2, 2, 0], dtype=np.int32)
    cu_num_logits_np = np.array([0, 3, 6, 7], dtype=np.int32)
    compacted, _ = manager.compact_batch(drafts, scheduled, cu_num_logits_np)

    assert int(compacted.sum()) == num_tokens
    num_steps = manager.num_speculative_steps
    assert (compacted[:2] <= 1 + num_steps).all()
    assert compacted[2] == 40


def test_budget_caps_at_one_rejection_sampler_chunk():
    # The chunked verification path cannot address the compacted logits
    # layout, so the budget must keep total logits within a single chunk.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9]], dtype=np.float32),
        np.ones(7),
    )
    manager._max_total_logits = 3  # 2 bonus logits + at most 1 draft
    manager.get_num_tokens(
        {"low": 3, "high": 3},
        {"low": [1, 2], "high": [3, 4]},
    )
    assert manager._batch_budget is not None
    _, _, draft_budget = manager._batch_budget
    assert draft_budget <= 1


def test_zero_budget_rebuilds_cpu_cu_num_logits():
    # When one bonus row per request already overflows a verification chunk, the
    # budget clamps to zero but the batch still needs chunking. Every capacity is
    # zeroed on device, so the CPU can name that layout exactly -- and must, since
    # _iter_request_chunks slices the compacted logits with these offsets.
    #
    # The third request is a chunked prefill (no drafts, still mid-prompt). The
    # runner gives *every* request num_bonus_tokens logits rows regardless
    # (num_logits = num_draft_tokens_per_req + num_bonus_tokens), so the rebuilt
    # offsets stay uniform rather than skipping non-verification rows.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9], [1.0, 1.0]], dtype=np.float32),
        np.ones(64),
    )
    manager.req_states.req_id_to_index["prefill"] = 2
    manager.req_states.num_computed_tokens_np = np.zeros(3, dtype=np.int32)
    manager.req_states.prefill_len.np = np.array([0, 0, 60], dtype=np.int32)
    manager._max_total_logits = 2  # < 3 requests * 1 bonus token

    manager.get_num_tokens(
        {"low": 3, "high": 3, "prefill": 40},
        {"low": [1, 2], "high": [3, 4]},
    )
    assert manager._batch_budget is not None
    _, _, draft_budget = manager._batch_budget
    assert draft_budget == 0

    scheduled = np.array([3, 3, 40], dtype=np.int32)
    drafts = np.array([2, 2, 0], dtype=np.int32)
    scheduled_cu_num_logits = np.array([0, 3, 6, 7], dtype=np.int32)
    compacted, cu_num_logits_np = manager.compact_batch(
        drafts, scheduled, scheduled_cu_num_logits
    )

    # One bonus row per request, matching cumsum(capacities + num_bonus_tokens)
    # with every capacity zeroed -- the prefill row included.
    expected = np.arange(4, dtype=np.int32) * manager.num_bonus_tokens
    assert np.array_equal(cu_num_logits_np, expected)
    assert cu_num_logits_np.dtype == scheduled_cu_num_logits.dtype
    # The prefill keeps its scheduled tokens; only drafts are dropped.
    assert np.array_equal(compacted, np.array([1, 1, 40], dtype=np.int32))


def global_topk_budget(
    confidence_probs: torch.Tensor,
    idx_mapping: torch.Tensor,
    capacities: torch.Tensor,
    draft_budget: int,
) -> torch.Tensor:
    survival = confidence_probs[idx_mapping].cumprod(dim=1)
    steps = torch.arange(survival.shape[1], device=survival.device)
    survival = survival.masked_fill(steps >= capacities[:, None], -float("inf"))
    admitted = torch.zeros(survival.numel(), dtype=torch.bool, device=survival.device)
    admitted[survival.flatten().topk(draft_budget).indices] = True
    return admitted.view_as(survival).sum(dim=1, dtype=torch.int32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_steps", [3, 8])
@pytest.mark.parametrize("num_reqs", [1, 2, 300])
def test_draft_budget_kernel_matches_global_topk(num_steps, num_reqs):
    """With distinct survivals the per-request admits equal a global top-k."""
    max_num_reqs = 512
    gen = torch.Generator(device="cuda").manual_seed(num_reqs * num_steps)
    confidence_probs = torch.rand(max_num_reqs, num_steps, device="cuda", generator=gen)
    idx_mapping = torch.randperm(max_num_reqs, device="cuda", generator=gen)
    idx_mapping = idx_mapping[:num_reqs]
    scheduled = torch.randint(
        1, num_steps + 1, (num_reqs,), dtype=torch.int32, device="cuda", generator=gen
    )
    scheduled[0] = num_steps
    survival = torch.empty(max_num_reqs * num_steps, dtype=torch.int32, device="cuda")
    num_scheduled = int(scheduled.sum())
    # The caller only invokes the kernel when the budget binds.
    budgets = {1, num_scheduled // 2, num_scheduled - 1}
    for draft_budget in sorted(b for b in budgets if 0 < b < num_scheduled):
        capacities = scheduled.clone()
        assign_draft_token_budget(
            confidence_probs, idx_mapping, capacities, draft_budget, survival
        )
        expected = global_topk_budget(
            confidence_probs, idx_mapping, scheduled, draft_budget
        )
        assert torch.equal(capacities, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_draft_budget_kernel_admits_exact_budget_on_ties():
    """Tied survivals are handed out in request order, across request blocks."""
    num_reqs, num_steps = 300, 3
    confidence_probs = torch.ones(num_reqs, num_steps, device="cuda")
    idx_mapping = torch.arange(num_reqs, device="cuda")
    capacities = torch.full((num_reqs,), num_steps, dtype=torch.int32, device="cuda")
    survival = torch.empty(num_reqs * num_steps, dtype=torch.int32, device="cuda")
    assign_draft_token_budget(confidence_probs, idx_mapping, capacities, 800, survival)
    assert capacities.tolist() == [3] * 266 + [2] + [0] * 33


@pytest.mark.parametrize("expected", [1, 3, 5, 7])
def test_candidate_steps_choose_best_profiled_prefix(expected):
    """A compute-cost cliff changes the selected prefix, not the draft width."""
    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
        select_verification_step,
    )

    costs = np.ones(33)
    costs[2 * (expected + 1) + 1 :] = 1000.0
    selected = select_verification_step(
        np.ones((2, 7)), np.array([7, 7]), (1, 3, 5, 7), 2, 2, 1.0, costs, 14
    )
    assert selected == expected


def test_candidate_steps_account_for_acceptance_and_sampler_capacity():
    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
        select_verification_step,
    )

    survival = np.array([[0.9, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert (
        select_verification_step(
            survival, np.array([7]), (1, 3, 5, 7), 1, 1, 1.0, np.arange(16) + 1.0, 7
        )
        == 1
    )
    # All candidates equally cheap: take the longest eligible prefix.
    assert (
        select_verification_step(
            np.ones((2, 7)), np.array([7, 2]), (1, 3, 5, 7), 2, 2, 1.0, np.ones(32), 7
        )
        == 5
    )
    # Never silently select a partial, non-candidate length to fit the sampler.
    assert (
        select_verification_step(
            np.ones((2, 7)), np.array([7, 7]), (1, 3, 5, 7), 2, 2, 1.0, np.ones(32), 1
        )
        == 0
    )


def test_candidate_steps_compact_real_manager_across_switches(
    tmp_path, default_vllm_config
):
    """Real CUDA buffers preserve CPU/GPU offsets for tails and mixed prefill."""
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.worker.gpu.states import RequestState

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"file://{tmp_path / 'dist'}",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        device = torch.device("cuda:0")
        states = RequestState(4, 64, 64, 7, 128, device)
        for req in ("a", "b", "prefill"):
            states.add_request(req, 16, [1] * 16, 0 if req == "prefill" else 16, 32)
        states.apply_staged_writes()
        manager = AdaptiveVerificationManager(
            states,
            torch.empty(5, dtype=torch.int32, device=device),
            1,
            64,
            copy_admitted_drafts_to_host=True,
            candidate_steps=(1, 3, 5, 7),
        )
        for req in states.req_id_to_index:
            manager.add_request(states.req_id_to_index[req])
        for k in (5, 1, 7, 3, 5):
            boundary = 11 + k + min(k, 2)
            manager.set_cost_curves(
                [(i, 1.0) for i in range(5)],
                [(i, 1.0 if i <= boundary else 1000.0) for i in range(65)],
            )
            total = manager.get_num_tokens(
                {"a": 8, "b": 3, "prefill": 9},
                {"a": [1] * 7, "b": [1] * 2},
            )
            assert manager.last_candidate_step == k
            # Exercise a different request order from the scheduling dictionaries.
            drafts = np.array([2, 7, 0], dtype=np.int32)
            scheduled = np.array([3, 8, 9], dtype=np.int32)
            compact, cpu_logits = manager.compact_batch(
                drafts, scheduled, np.array([0, 3, 11, 12], dtype=np.int32)
            )
            admitted = np.array([min(k, 2), k, 0], dtype=np.int32)
            expected_queries = admitted + np.array([1, 1, 9], dtype=np.int32)
            np.testing.assert_array_equal(compact, expected_queries)
            np.testing.assert_array_equal(cpu_logits, np.r_[0, np.cumsum(admitted + 1)])
            assert int(compact.sum()) == total
            ids = ["b", "a", "prefill"]
            mapping = torch.tensor(
                [states.req_id_to_index[x] for x in ids], device=device
            )
            gpu_logits, gpu_query, budget, copied = manager.reallocate_drafts(
                ids, mapping
            )
            np.testing.assert_array_equal(gpu_logits.cpu().numpy(), cpu_logits)
            np.testing.assert_array_equal(
                gpu_query[:4].cpu().numpy(), np.r_[0, np.cumsum(compact)]
            )
            assert budget == int(admitted.sum())
            if k < 7:
                assert copied is not None
                np.testing.assert_array_equal(copied.wait(), admitted)
            else:
                assert copied is None
            assert states.draft_tokens.shape == (4, 7)
    finally:
        cleanup_dist_env_and_memory()
