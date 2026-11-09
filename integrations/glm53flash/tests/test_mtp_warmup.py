# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute native warmup scheduling with CPU token buffers and embedding.

Scheduler/worker doubles isolate the sentinel contract; CUDA and vLLM are not
imported. The no-seed counterexample is made from the same current function,
not an archived baseline.
"""

import ast
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

WARMUP = "vllm/v1/worker/gpu/warmup.py"


def empty_output():
    return NS(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=NS(req_ids=[]),
        scheduled_spec_decode_tokens={},
        num_scheduled_tokens={},
        finished_req_ids=set(),
        total_num_scheduled_tokens=0,
    )


class Request:
    def __init__(self, req_id, *_args, **_kwargs):
        self.req_id = req_id


class NewRequest:
    @staticmethod
    def from_request(request, **kwargs):
        return NS(req_id=request.req_id, **kwargs)


@pytest.fixture
def warmup(source_tree, execute_ast):
    def load(seed=True):
        fn = next(
            n
            for n in source_tree(WARMUP).body
            if isinstance(n, ast.FunctionDef) and n.name == "_warmup_kernels"
        )
        guards = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.If) and "p_mtp_context_only" in ast.unparse(n.test)
        ]
        assert len(guards) == 1
        guard = guards[0]
        assert "any(spec_flags)" in ast.unparse(guard.test)
        assert [
            c.func.attr
            for c in ast.walk(guard)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
        ] == ["zero_"]
        if not seed:

            class RemoveSeed(ast.NodeTransformer):
                def visit_If(self, node):
                    return None if node is guard else self.generic_visit(node)

            fn = RemoveSeed().visit(fn)
        return execute_ast(
            [fn],
            {
                "torch": NS(accelerator=NS(synchronize=Mock())),
                "SchedulerOutput": NS(make_empty=empty_output),
                "CachedRequestData": NS(make_empty=lambda: NS()),
                "NewRequestData": NewRequest,
                "Request": Request,
                "SamplingParams": NS(for_sampler_warmup=lambda: NS()),
                "GrammarOutput": NS,
                "_warmup_block_counter": lambda _: lambda n, _spec: (n + 15) // 16,
                "np": NS(full=lambda *_args, **_kwargs: None, int32=int),
            },
        )["_warmup_kernels"]

    return load


class CPUWarmup:
    def __init__(self, producer, num_reqs=4, marker_present=True):
        self.producer = producer
        self.executions, self.mapping, self.current_ids = [], {}, []
        self.samples = 0
        self.weights = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        self.proposed = torch.full((4, 3), -1 if producer else 3, dtype=torch.int64)
        self.runner = NS(
            vllm_config=NS(is_mm_encoder_only=False),
            num_speculative_steps=3,
            decode_query_len=4,
            is_pooling_model=False,
            kv_cache_config=NS(
                kv_cache_groups=[NS(kv_cache_spec=NS())], num_blocks=128
            ),
            model_state=NS(max_encoder_len=0),
            scheduler_config=NS(max_num_seqs=num_reqs, max_num_batched_tokens=2048),
            block_tables=None,
            kv_block_zeroer=None,
            is_encoder_decoder=False,
            is_last_pp_rank=True,
            model_config=NS(get_vocab_size=lambda: 8),
            kv_connector=NS(set_disabled=Mock()),
            speculator=NS(p_mtp_context_only=producer) if marker_present else NS(),
            req_states=NS(draft_tokens=torch.zeros((4, 3), dtype=torch.int64)),
        )

    def execute(self, output):
        if output.finished_req_ids:
            return
        for request in output.scheduled_new_reqs:
            slot = len(self.mapping)
            self.mapping[request.req_id] = slot
            self.runner.req_states.draft_tokens[slot].zero_()
        self.current_ids = list(output.num_scheduled_tokens)
        for req_id, count in output.num_scheduled_tokens.items():
            drafts = output.scheduled_spec_decode_tokens.get(req_id, [])
            if drafts:
                # Native combine reads the request buffer, not scheduler zeros.
                ids = torch.cat(
                    (
                        torch.tensor([2]),
                        self.runner.req_states.draft_tokens[
                            self.mapping[req_id], : len(drafts)
                        ],
                    )
                )
            else:
                ids = torch.arange(count, dtype=torch.int64) % 8
            self.executions.append((req_id, count, tuple(drafts), tuple(ids.tolist())))
            F.embedding(ids, self.weights)  # Real CPU indexing rejects -1.

    def sample(self, _grammar):
        self.samples += 1
        for req_id in self.current_ids:
            slot = self.mapping[req_id]
            self.runner.req_states.draft_tokens[slot] = self.proposed[slot]

    def run(self, fn):
        fn(self.runner, self.execute, self.sample)
        return self


@pytest.mark.parametrize("num_reqs", [1, 2, 4])
def test_p_synthetic_warmup_seeds_legal_ids_without_changing_proposed_sentinel(
    warmup, num_reqs
):
    trial = CPUWarmup(True, num_reqs).run(warmup())
    speculative = [row for row in trial.executions if row[2]]
    assert speculative and all(row[3] == (2, 0, 0, 0) for row in speculative)
    assert all(min(row[3]) >= 0 for row in trial.executions)
    assert (trial.proposed == -1).all()
    # Seeding covers the entire warmup buffer, but the last sample only writes
    # the requests scheduled in that step. Check their serving sentinel, not
    # inactive rows left at the legal synthetic zero IDs.
    active = [trial.mapping[req_id] for req_id in trial.current_ids]
    assert (trial.runner.req_states.draft_tokens[active] == -1).all()
    assert trial.runner.num_speculative_steps == 3
    trial.runner.kv_connector.set_disabled.assert_any_call(True)
    trial.runner.kv_connector.set_disabled.assert_any_call(False)


def test_removing_only_p_seed_reproduces_the_invalid_embedding_input(warmup):
    trial = CPUWarmup(True)
    with pytest.raises((IndexError, RuntimeError), match="index out of range"):
        trial.run(warmup(seed=False))
    assert trial.samples == 1
    assert trial.executions[-1][2:] == ((0, 0, 0), (2, -1, -1, -1))


@pytest.mark.parametrize("marker_present", [False, True])
def test_d_warmup_keeps_all_three_drafts_with_or_without_p_marker(
    warmup, marker_present
):
    current = CPUWarmup(False, marker_present=marker_present).run(warmup())
    no_seed = CPUWarmup(False, marker_present=marker_present).run(warmup(seed=False))
    assert (
        current.executions == no_seed.executions and current.samples == no_seed.samples
    )
    speculative = [row for row in current.executions if row[2]]
    assert speculative and all(row[3] == (2, 3, 3, 3) for row in speculative)
    assert torch.equal(
        current.runner.req_states.draft_tokens, no_seed.runner.req_states.draft_tokens
    )
