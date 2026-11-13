# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode in the hybrid decomposition (TP-G attention groups
with optional DCP G + routed experts EP over every rank, MegaMoE) on real ranks:
admission
(TP > 1 + EP, FP32 KDA conv and recurrent state, RecoverSSM), the hybrid plan
(``runtime.hybrid.HybridStep``: uniform q-token verify steps with TP-G KDA / DSA
providers, the stock SP MoE on the rank's row shard), and the step's output
against the stock decode of the same step (stock: sequence parallel over the TP
group, the MoE on the rank's row shard).

Default: TP4 x DCP4 x DP1 (EP4) on one 4-GPU node; ``KIMI_K31_HYBRID_WORLD`` /
``KIMI_K31_HYBRID_TP`` / ``KIMI_K31_HYBRID_DCP`` select other shapes (e.g. 8 / 4 / 4:
two groups, EP8, launched one process per GPU by ``KIMI_K31_HYBRID_SRUN=1``).

Each rank runs ``vllm bench forward`` in-process on a layer-truncated K3.1
(``KIMI_K31_DEP_MODEL``: a config-only K3.1 directory with a few layers, e.g. 3
KDA + 1 DSA + MegaMoE layers; dummy weights, balanced EP routing). The bench
compares the step's CUDA graph replay (captured: the specialized decode) with the
same step run eagerly (outside a capture the specialized decode never plans: the
stock forward), per rank. Contexts stay below the DSA top-k (2048), so both paths
attend over every token and the routing is the bench's fixture on both: the
outputs differ only by accumulation order and FP8 / BF16 rounding.
Shared-provider assertions additionally check actual capture metadata and opt-out.
Shared reuse/overlap follows each admitted TP/EP layout, q and capture size;
only the optional routed tile override remains specific to EP8/local40.
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

MODEL = os.environ.get("KIMI_K31_DEP_MODEL")
# Its DSpark draft config (target layers inside the truncated model).
DRAFT = os.environ.get("KIMI_K31_DEP_DRAFT")
WORLD = int(os.environ.get("KIMI_K31_HYBRID_WORLD", "4"))
TP = int(os.environ.get("KIMI_K31_HYBRID_TP", "4"))
DCP = int(os.environ.get("KIMI_K31_HYBRID_DCP", str(TP)))
SHARED_TP = os.environ.get("KIMI_K31_HYBRID_SHARED_TP", "1") == "1"
# Per TP group: uniform DSpark verify steps. (Under DSpark the KDA metadata
# counts draft-less decodes as spec decodes, so a q = 1 step is not uniform and
# runs stock, at TP8 as at DEP.)
# At the default q5, R1 has fully padded ranks and R33/R62 overhang at TP8;
# changing the draft count derives new rows without changing runtime guards.
DRAFT_TOKENS = int(os.environ.get("KIMI_K31_HYBRID_DRAFT_TOKENS", "4"))
Q = 1 + DRAFT_TOKENS
REQUESTS = (1, 4, 16, 33, 62, 64)
CASES = tuple(
    f"v{requests}={requests}*{Q}@{kv}:{DRAFT_TOKENS}"
    for requests, kv in zip(REQUESTS, (1900, 1500, 1100, 900, 800, 700))
)
# The deployment's capture sizes (rows).
CAPTURE = tuple(requests * Q for requests in REQUESTS)

pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_device_capability_family(100),
        reason="the specialized decode kernels need SM100/SM103",
    ),
    pytest.mark.skipif(
        not all(os.path.isdir(p or "") for p in (MODEL, DRAFT)),
        reason="needs a layer-truncated K3.1 config (KIMI_K31_DEP_MODEL) and its "
        "DSpark draft config (KIMI_K31_DEP_DRAFT)",
    ),
    pytest.mark.skipif(
        torch.accelerator.device_count() < min(WORLD, 4),
        reason=f"Need {WORLD} GPUs.",
    ),
]


def admit(capture_sizes, plans, shared_providers):
    """Admit the specialized decode under ``vllm bench forward`` (its graphs are
    one-shot captures of each exact batch; the deployment's capture sizes are
    ``capture_sizes``) and record every captured step's plan."""
    import dataclasses

    import vllm.models.kimi_k3.nvidia.model as model
    import vllm.models.kimi_k3.nvidia.specialized_decode.runtime as runtime
    from vllm.config import CUDAGraphMode, get_current_vllm_config
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import (
        hybrid_shared_moe,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.decode import (
        SpecializedDecode,
    )

    check, build = runtime.unsupported, runtime.specialized_decode_config

    def unsupported(vllm_config):
        compilation = vllm_config.compilation_config
        mode = compilation.cudagraph_mode
        compilation.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
        try:
            return check(vllm_config)
        finally:
            compilation.cudagraph_mode = mode

    def specialized_decode_config(vllm_config):
        config = build(vllm_config)
        assert config is not None, "the specialized decode was not admitted"
        return dataclasses.replace(config, capture_sizes=capture_sizes)

    runtime.unsupported = unsupported
    runtime.specialized_decode_config = specialized_decode_config
    model.specialized_decode_config = specialized_decode_config
    plan = SpecializedDecode.plan

    def recording(self, positions, intermediate_tensors):
        step = plan(self, positions, intermediate_tensors)
        if not torch.cuda.is_current_stream_capturing():
            config = get_current_vllm_config()
            extra = config.additional_config
            assert extra is None or isinstance(extra, dict)
            disabled = dataclasses.replace(
                config,
                additional_config={
                    **(extra or {}),
                    "hybrid_shared_overlap": False,
                },
            )
            assert hybrid_shared_moe.HybridSharedMoE.create(self, disabled) is None
        if torch.cuda.is_current_stream_capturing():
            kinds = sorted({type(a).__name__ for a in self.attention.values()})
            shared = self.hybrid.shared_moe if self.hybrid is not None else None
            if shared is not None:
                shared_providers[id(shared)] = shared
            plans.append(
                dict(
                    rows=positions.shape[0],
                    planned=step is not None,
                    wide=bool(step is not None and step.wide),
                    q=step.q if step is not None else None,
                    hybrid=self.hybrid is not None,
                    attention=kinds,
                    shared_moe=type(shared).__name__ if shared is not None else None,
                    shared_tile32=shared.tile32 if shared is not None else False,
                )
            )
        return step

    setattr(SpecializedDecode, "plan", recording)  # noqa: B010


def worker(rank, port, out_dir):
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(WORLD),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    run(rank, out_dir)


def worker_srun(rank, out_dir):
    """srun: MASTER_ADDR / MASTER_PORT from the launcher's environment."""
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(WORLD))
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    run(rank, out_dir)


def run(rank, out_dir):
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    os.environ["VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT"] = str(int(SHARED_TP))
    plans: list = []
    shared_providers: dict = {}
    admit(CAPTURE, plans, shared_providers)
    import vllm.benchmarks.forward.main as fm
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    spec = {
        "method": "dspark",
        "num_speculative_tokens": DRAFT_TOKENS,
        "model": DRAFT,
    }
    argv = [
        "--model", MODEL, "--load-format", "dummy", "--trust-remote-code",
        "-tp", str(TP), "-dcp", str(DCP), "-dp", str(WORLD // TP),
        "--enable-expert-parallel",
        "--routing", "balanced", "--moe-backend", "deep_gemm_mega_moe",
        "--kv-cache-dtype", "fp8", "--mamba-cache-dtype", "float32",
        "--mamba-ssm-cache-dtype", "float32", "--use-replayssm",
        "--max-model-len", "8192", "--language-model-only",
        "--gpu-memory-utilization", "0.8", "--max-num-seqs", "64",
        "--max-num-batched-tokens", "16384",
        "--speculative-config", json.dumps(spec), "--exclude-drafter",
        "--additional-config", json.dumps({"decode_backend": "latency_specialized"}),
        "--kernel-config", json.dumps({"enable_flashinfer_autotune": False}),
        "--kv-len-jitter", "0", "--iters", "2", "--samples", "1",
        "--output-json", os.path.join(out_dir, "result.json"),
    ]  # fmt: skip
    for case in CASES:
        argv += ["--case", case]
    parser = FlexibleArgumentParser()
    fm.add_cli_args(parser)
    fm.main(parser.parse_args(argv))
    expected = TP in (4, 8, 16) and WORLD % TP == 0 and DCP in (1, TP)
    captured = [
        list(full.shape)
        for provider in shared_providers.values()
        for full, *_ in provider.graph_keepalive
    ]
    assert bool(captured) == expected, (expected, captured)
    if expected:
        assert {shape[0] for shape in captured} == set(CAPTURE), captured
    for provider in shared_providers.values():
        if provider.tile32:
            assert WORLD == TP == 8 and 320 in CAPTURE
        for full, shared_input, local, partial, *_ in provider.graph_keepalive:
            rows = -(-full.shape[0] // TP)
            assert local.shape == (rows, full.shape[1])
            shared_rows = rows * TP if SHARED_TP else rows
            assert shared_input.shape == partial.shape == (shared_rows, full.shape[1])
            # These assertions run after capture/timing, never on the hot path.
            if SHARED_TP:
                assert torch.count_nonzero(shared_input[full.shape[0] :]).item() == 0
    with open(os.path.join(out_dir, f"shared{rank}.json"), "w") as f:
        json.dump(dict(expected=expected, captured_shapes=captured), f)
    with open(os.path.join(out_dir, f"plans{rank}.json"), "w") as f:
        json.dump(plans, f)


def test_hybrid_specialized_decode_matches_stock(tmp_path):
    """Shared reuse/overlap covers TP/DP, q and SP tails; explicit opt-out is stock."""
    if os.environ.get("KIMI_K31_HYBRID_SRUN") == "1":  # one process per GPU (srun)
        rank = int(os.environ["SLURM_PROCID"])
        out = os.environ["KIMI_K31_HYBRID_OUT"]
        os.environ.setdefault("LOCAL_RANK", os.environ["SLURM_LOCALID"])
        worker_srun(rank, out)
        if rank:
            return
        while len([f for f in os.listdir(out) if f.startswith("plans")]) < WORLD:
            time.sleep(5)
        check(Path(out))
        return
    port = get_open_port()
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=worker, args=(r, port, str(tmp_path))) for r in range(WORLD)
    ]
    for p in procs:
        p.start()
    deadline = time.monotonic() + 1500
    for p in procs:
        p.join(timeout=max(0.0, deadline - time.monotonic()))
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join()
    assert [p.exitcode for p in procs] == [0] * WORLD
    check(tmp_path)


def check(tmp_path):
    results = json.loads((tmp_path / "result.json").read_text())
    assert [r["case"] for r in results] == [c.split("=")[0] for c in CASES]
    for rank in range(WORLD):
        plans = json.loads((tmp_path / f"plans{rank}.json").read_text())
        shared = json.loads((tmp_path / f"shared{rank}.json").read_text())
        # Every captured step took the hybrid path, with the TP-G DSA and KDA
        # providers and the default/disabled layout-aware shared provider.
        assert plans, rank
        for plan in plans:
            assert plan["hybrid"] and plan["planned"] and plan["wide"], plan
            assert plan["attention"] == ["HybridAttention"], plan
            assert plan["shared_moe"] == (
                "HybridSharedMoE" if shared["expected"] else None
            ), plan
        assert {p["rows"] for p in plans} >= set(CAPTURE), plans
        assert {p["q"] for p in plans} == {Q}, plans
    for result in results:
        for rank, r in enumerate(result["ranks"]):
            err = r["graph_vs_eager"]["relative_rms"]
            print(f"{result['case']} rank {rank}: spec vs stock rel rms {err:.2e}")
            # Order and BF16 rounding only (measured 3.6e-4 - 6.0e-4 at TP4 / TP8:
            # the TP-G cores read the stock layers' FP8 query and cache; a wrong
            # row in one MoE stage alone gives 3e-2 - 8e-2).
            assert err < 5e-3, (result["case"], rank, err)
