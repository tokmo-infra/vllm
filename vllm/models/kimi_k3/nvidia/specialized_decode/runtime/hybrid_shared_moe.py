# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid shared-input reuse and overlap, with shape-specific routed tuning."""

import torch

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.distributed import get_ep_group, get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP4,
    DeepGemmMegaMoEExperts,
)
from vllm.model_executor.layers.fused_moe.prepare_megamoe import prepare_megamoe_inputs
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.common.ops.sequence_parallel import sp_reduce_scatter
from vllm.utils.deep_gemm import _import_deep_gemm
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel

# Native streams are process-owned, not recycled through torch's stream pool.
# Keep both the handle and wrapper alive until every captured graph is gone.
_SHARED_STREAMS: dict = {}


class HybridSharedMoE:
    """A shared-expert provider built before capture; no diagnostic collectives.

    Only Hybrid's replicated pre-SP hidden may enter this provider. Keep it alive
    with its graphs and replay sequentially: MegaMoE shares native scratch.
    """

    @classmethod
    def create(cls, runtime, config):
        extra = config.additional_config
        if extra is not None and (
            not isinstance(extra, dict)
            or extra.get("hybrid_shared_overlap", True) is not True
        ):
            return None
        p = config.parallel_config
        if (
            runtime.tp <= 1
            or runtime.tp != p.tensor_parallel_size
            or not runtime.wide
            or not p.enable_expert_parallel
            or p.enable_dbo
            or config.kernel_config.moe_backend != "deep_gemm_mega_moe"
        ):
            return None
        tp, ep = get_tp_group(), get_ep_group()
        if (
            tp.world_size != runtime.tp
            or ep.world_size % tp.world_size != 0
            or not set(tp.ranks).issubset(ep.ranks)
        ):
            return None
        modules = [layer.mlp for layer in runtime.layers if layer.is_moe_layer]
        if not modules or not all(
            cls._supports(m, tp.world_size, ep.world_size) for m in modules
        ):
            return None
        device = modules[0].gate.weight.device
        if device.type != "cuda" or any(
            m.gate.weight.device != device for m in modules
        ):
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        # Only the native tile override is tied to the measured EP8/local40 case.
        tile32 = (
            tp.world_size == ep.world_size == 8
            and tp.ranks == ep.ranks
            and not p.enable_eplb
            and 320 in runtime.wide
            and torch.cuda.get_device_capability(device) == (10, 3)
            and torch.cuda.get_device_properties(device).multi_processor_count == 148
            and all(
                (
                    m.experts.num_experts,
                    m.experts.top_k,
                    m.experts.hidden_size,
                    m.experts.intermediate_size,
                )
                == (896, 16, 3584, 3072)
                and m.experts.max_num_tokens >= 72
                and m.experts.weight_format == MEGA_MOE_MXFP4
                and m.experts.eplb_state.logical_to_physical_map is None
                for m in modules
            )
        )
        return cls(modules, device, runtime.tp, runtime.q, runtime.wide, tile32)

    @staticmethod
    def _supports(m, tp, ep):
        from ...model import KimiMoE

        if not isinstance(m, KimiMoE):
            return False
        e, s = m.experts, m.shared_experts
        if not (
            m.use_mega_moe
            and m.use_sequence_parallel
            and m.tp_size == tp
            and m.use_latent_moe
            and m.routed_output_transform is not None
            and isinstance(e, DeepGemmMegaMoEExperts)
            and not e.has_fused_shared_experts
            and e.num_local_experts * ep == e.num_experts
            and s is not None
            and s.gemm_rs_ar is None
            and not s.gate_up_proj.gather_output
            and s.down_proj.input_is_parallel
            and not s.down_proj.reduce_results
            and m._down_proj_stream is not None
        ):
            return False
        shared_tp = tp if s.shard_sequence_parallel else 1
        if s.gate_up_proj.tp_size != shared_tp or s.down_proj.tp_size != shared_tp:
            return False
        linears = (s.gate_up_proj, s.down_proj)
        if not all(
            isinstance(x.quant_method, UnquantizedLinearMethod) for x in linears
        ):
            return False
        down, up = m.routed_expert_down_proj, m.routed_expert_up_proj
        if down is None or up is None:
            return False
        hidden, latent = m.gate.weight.shape[1], e.hidden_size
        intermediate = s.down_proj.weight.shape[1]
        shapes = (
            (e.num_logical_experts, hidden),
            (latent, hidden),
            (hidden, latent),
            (2 * intermediate, hidden),
            (hidden, intermediate),
        )
        weights = (
            m.gate.weight,
            down.weight,
            up.weight,
            *[x.weight for x in linears],
        )
        return all(
            w.dtype == torch.bfloat16
            and tuple(w.shape) == shape
            and w.device == m.gate.weight.device
            for w, shape in zip(weights, shapes)
        )

    def __init__(self, modules, device, tp, q, batches, tile32):
        from cuda.bindings import runtime as cuda_runtime

        self.device = device
        self.tp, self.q, self.batches = tp, q, batches
        self.hidden_size = modules[0].gate.weight.shape[1]
        key = int(device.index)
        with torch.accelerator.device_index(key):
            if key not in _SHARED_STREAMS:
                error, native = cuda_runtime.cudaStreamCreateWithPriority(1, 0)
                if int(error) != 0:
                    # Do not silently select a different collective DAG on one rank.
                    raise RuntimeError(f"Cannot create Hybrid shared stream: {error}")
                stream = torch.cuda.ExternalStream(int(native), device=device)
                _SHARED_STREAMS[key] = (native, stream)
            self.stream = _SHARED_STREAMS[key][1]
            self.events = {}
            for m in modules:
                if int(m._down_proj_stream.cuda_stream) == int(self.stream.cuda_stream):
                    raise RuntimeError("Hybrid shared stream aliases stock down stream")
                events = (torch.cuda.Event(), torch.cuda.Event())
                for event in events:
                    event.record()  # Materialize the native events before capture.
                self.events[id(m)] = events
        self.tile32 = tile32 and all(
            m.experts.get_block_m(40) == 16 and m.experts.get_block_m(72) == 32
            for m in modules
        )
        self.expert_buffers = {}
        if self.tile32:
            self.deep_gemm = _import_deep_gemm()
            for m in modules:
                self.expert_buffers[id(m.experts)] = m.experts.get_symm_buffer()
                m.experts._synchronize_first_launch()
        self.graph_keepalive = []

    def run(self, module, full, q):
        from .hybrid import run_stock_moe

        # These are CPU metadata checks; no witness, CPU group or device sync.
        if (
            q != self.q
            or full.ndim != 2
            or full.shape[0] not in self.batches
            or full.shape[1] != self.hidden_size
            or full.dtype != torch.bfloat16
            or full.device != self.device
            or not full.is_contiguous()
            or torch.is_grad_enabled()
            or id(module) not in self.events
            or BreakableCUDAGraphCapture.is_active()
            or not is_forward_context_available()
        ):
            return run_stock_moe(module, full)
        mask = get_forward_context().is_padding
        if (
            mask is None
            or tuple(mask.shape) != (-(-full.shape[0] // self.tp),)
            or mask.dtype != torch.bool
            or mask.device != self.device
        ):
            return run_stock_moe(module, full)
        main, shared, down = (
            torch.cuda.current_stream(self.device),
            self.stream,
            module._down_proj_stream,
        )
        if int(shared.cuda_stream) in (int(main.cuda_stream), int(down.cuda_stream)):
            raise RuntimeError(
                "Hybrid shared stream aliases main or stock-down; abort on all ranks"
            )
        return run_stock_moe(lambda local: self._local(module, local, full), full)

    def _local(self, m, local, full):
        latent, weights, ids = m._maybe_overlap_router_and_down_proj(local)
        capturing = torch.cuda.is_current_stream_capturing()
        shared = m.shared_experts
        shared_input = local
        if shared.shard_sequence_parallel:
            # Match AG(shard(full)), including synthetic zero rows at the SP tail.
            pad = local.shape[0] * self.tp - full.shape[0]
            shared_input = (
                torch.nn.functional.pad(full, (0, 0, 0, pad)) if pad else full
            )
        if not capturing:
            shared_input.record_stream(self.stream)

        def shared_compute():
            gate_up, _ = shared.gate_up_proj(shared_input)
            partial, _ = shared.down_proj(shared.act_fn(gate_up))
            return partial

        events = self.events[id(m)]
        routed, partial = maybe_execute_in_parallel(
            lambda: self._routed(
                m.experts,
                latent,
                weights,
                ids,
                tile32=self.tile32 and full.shape[0] == 320,
            ),
            shared_compute,
            events[0],
            events[1],
            self.stream,
        )
        if not capturing:
            partial.record_stream(torch.cuda.current_stream(self.device))
        shared_local = (
            sp_reduce_scatter(partial) if shared.shard_sequence_parallel else partial
        )
        out = m.routed_output_transform(routed, residual=shared_local)
        if capturing:
            self.graph_keepalive.append(
                (full, shared_input, local, partial, shared_local, out)
            )
        return out

    def _routed(self, expert, hidden, weights, ids, *, tile32):
        if not tile32:
            return expert(hidden, weights, ids)
        if expert.capture_fn is not None:
            expert.capture_fn(ids)
        symm = self.expert_buffers[id(expert)]
        prepare_megamoe_inputs(
            hidden,
            weights,
            ids,
            symm.x[:40],
            symm.x_sf[:40],
            symm.topk_idx[:40],
            symm.topk_weights[:40],
            is_padding=get_forward_context().is_padding
            if envs.VLLM_MOE_SKIP_PADDING
            else None,
            hidden_quant=expert.weight_format.hidden_quant,
        )
        # Negative routes never dispatch; every call clears the shared tail.
        symm.topk_idx[40:72].fill_(-1)
        symm.topk_weights[40:72].zero_()
        output = torch.empty((72, 3584), device=hidden.device, dtype=torch.bfloat16)
        self.deep_gemm.fp8_fp4_mega_moe(
            output,
            expert.l1_weights,
            expert.l2_weights,
            symm,
            fast_math=expert.fast_math,
            **expert._kernel_kwargs,
        )
        return output[:40]
