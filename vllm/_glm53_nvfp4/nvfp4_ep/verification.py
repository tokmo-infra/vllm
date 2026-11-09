# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FINITE diagnostic helper, never part of serving dispatch. Parent explicitly calls.

native_call() must be the exact-checkpoint/native-quantization AG4/CUTLASS/RS
routed-only reference, with SAME hidden/ids/route weights and calibration.
Parent owns GPU scheduling, rank-uniform test control, thresholds and aggregation.
No profiler, model launch or SSH is implemented here.
"""


def compare_once(
    workspace,
    pack,
    hidden,
    ids,
    route_weights,
    native_call,
    *,
    uniform_bound,
    relative_l2_limit,
    max_abs_limit,
    graph_replays=3,
):
    import torch

    # Explicit finite-diagnostic synchronization; never move into serving.
    native = native_call().clone()
    workspace.stage(hidden, ids, route_weights, pack, uniform_bound=uniform_bound)
    eager = workspace.compute(pack).clone()
    torch.accelerator.synchronize()
    if (
        not torch.isfinite(eager).all()
        or int(workspace.protocol_error_tensor().item()) != 0
    ):
        raise AssertionError("candidate finite/protocol check failed")
    diff = eager.float() - native.float()
    denominator = torch.linalg.vector_norm(native.float()).clamp_min(1e-12)
    l2 = float(torch.linalg.vector_norm(diff) / denominator)
    max_abs = float(diff.abs().max()) if eager.numel() else 0.0
    cosine = (
        float(
            torch.nn.functional.cosine_similarity(
                eager.float().reshape(1, -1), native.float().reshape(1, -1)
            ).squeeze()
        )
        if eager.numel()
        else 1.0
    )
    if l2 > relative_l2_limit or max_abs > max_abs_limit:
        raise AssertionError(
            f"NVFP4 numerical gate failed: relative_l2={l2}, max_abs={max_abs}"
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        workspace.stage(hidden, ids, route_weights, pack, uniform_bound=uniform_bound)
        output = workspace.compute(pack)
    for _ in range(graph_replays):
        graph.replay()
        torch.accelerator.synchronize()
        if not torch.equal(eager, output):
            raise AssertionError("candidate graph replay differs from candidate eager")
        if int(workspace.protocol_error_tensor().item()) != 0:
            raise AssertionError("candidate graph protocol error")
    return {
        "relative_l2": l2,
        "max_abs": max_abs,
        "cosine_similarity": cosine,
        "graph_equals_candidate_eager": True,
        "protocol_error": 0,
        "formal_serving_score": False,
    }
