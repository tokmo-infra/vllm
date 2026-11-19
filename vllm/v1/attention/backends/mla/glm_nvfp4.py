# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM_NEXT native NoPE reader/writer. Keep the pool ABI separate from IPC.

Indices presented to B12X are logical-page slots; its strided cache view, not
the index kernel, applies the physical page pitch (including hybrid padding).
"""
from __future__ import annotations

import torch

from vllm.config.glm_mla_kv import GLM_NVFP4_ROW_BYTES
from vllm.v1.worker.workspace import current_workspace_manager


class GlmNvfp4Ops:
    def __init__(self, *, heads, scale, selector_width, max_batch, max_k):
        try:
            from b12x.attention import sparse_mla
            from b12x.attention._shared.mla.traits import ScaleFormat
            from b12x.preparation import PreparedCall, prepare_default
        except ImportError as exc:
            raise RuntimeError("GLM NVFP4 KV requires the GLM_NEXT-capable B12X build") from exc
        self.api = sparse_mla
        self.fp4 = ScaleFormat.NVFP4_E4M3
        self.prepare = prepare_default
        self.call = PreparedCall
        self.heads = heads
        self.scale = scale
        self.width = selector_width
        self.max_batch = max_batch
        self.max_k = max_k
        if self.max_k > self.width:
            raise ValueError("selector storage shorter than GLM's history plus tail")

    def prepare_cache(self, cache):
        if cache.dtype != torch.uint8 or cache.ndim != 3 or cache.size(-1) != GLM_NVFP4_ROW_BYTES:
            raise ValueError("expected [blocks, physical page tokens, 304] byte pages")
        if cache.stride(-1) != 1 or cache.stride(1) != GLM_NVFP4_ROW_BYTES:
            raise ValueError("non-contiguous GLM record")
        # Prepare from small, independent tensors. Never pin the profiling KV pool
        # in a writer plan or write warmup tokens into an application's cache.
        mgr = current_workspace_manager()
        key = ('glm_next_nvfp4', self.heads, self.scale, self.width, self.max_k,
               self.max_batch, cache.size(1), cache.device)
        self.resources = mgr.get_persistent_resource(key, lambda: self._build(cache))
        # Each vLLM (ubatch, lane) owns its scratch. CUDA graph capture can
        # use a different stream from warmup within the same execution lane.
        # Never use a transient CUDA stream handle as an allocation key.
        for plan, tile, spec in self.resources[1].values():
            self._work_buffers(plan, tile, spec)

    def _work_buffers(self, plan, tile, spec):
        sizes = [(tuple(spec.shape), spec.dtype)]
        if plan.query.max_width != self.width:
            sizes.append(((tile, plan.query.max_width), torch.int32))
        return current_workspace_manager().get_simultaneous(*sizes)

    def _build(self, cache):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("missing GLM NVFP4 preparation before CUDA capture")
        device = cache.device
        page = cache.size(1)
        dummy = torch.zeros((1, page, GLM_NVFP4_ROW_BYTES), dtype=torch.uint8, device=device)
        latent = torch.zeros((1, 512), dtype=torch.bfloat16, device=device)
        writers = {}
        for dtype in (torch.int32, torch.int64):
            slots = torch.zeros(1, dtype=dtype, device=device)
            wp = self.api.plan_cache_writer(latent, dummy, slots)
            self.prepare(wp.request(name="glm-next-cache-writer", prepare_call=lambda st: self.call(run=lambda: st.run(latent, dummy, slots))))
            writers[dtype] = wp

        # Split very large invocations into row tiles; these are scratch limits,
        # not restrictions on --max-num-seqs or --max-num-batched-tokens.
        readers = {}
        for mode in ('decode', 'extend'):
            chunks = 64 if mode == 'decode' else 1
            tile = max(1, min(self.max_batch, (64 << 20) // (chunks * self.heads * 512 * 2)))
            # Decode accepts the vLLM selector pitch. Extend takes a native
            # tile-aligned width covering *all* selected IDs including the
            # incomplete trailing pool. This is not a KV page or a hash unit.
            kernel_width = self.width if mode == 'decode' else min(
                self.width, ((self.max_k + 63) // 64) * 64)
            caps = self.api.Caps(device=device, num_q_heads=self.heads, max_q_rows=tile,
                max_batch=tile, max_width=kernel_width, max_kv_rows=self.max_k,
                softmax_scale=self.scale, kv_dtype=torch.uint8, head_dim=512,
                v_head_dim=512, page_size=page, model_type=self.api.ModelType.GLM_NEXT,
                scale_format=self.fp4, cache_record_bytes=GLM_NVFP4_ROW_BYTES,
                # Extend always writes LSE, even when the caller only consumes
                # attention. Bind its planned buffer instead of allocating an
                # unused [rows, heads] temporary at each tile/capture.
                fp8_rope=False, latent_scale_per_token=True, mode=mode, return_lse=(mode == 'extend'))
            rp = self.api.plan(caps)
            def prime(st):
                spec, = st.scratch_specs()
                scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
                q = torch.zeros((tile, self.heads, 512), dtype=torch.bfloat16, device=device)
                idx = torch.zeros((tile, kernel_width), dtype=torch.int32, device=device)
                valid = torch.ones(tile, dtype=torch.int32, device=device)
                raw = st.bind_for_preparation(scratch=scratch, q=q, kv_cache=dummy,
                    selected_indices=idx, cache_seqlens_int32=valid, nsa_cache_seqlens_int32=valid)
                return self.call(run=lambda: st.run(raw, kv_cache=dummy))
            self.prepare(rp.request(name="glm-next-" + mode, prepare_call=prime))
            spec, = rp.scratch_specs()
            readers[mode] = (rp, tile, spec)
        return writers, readers

    def write(self, latent, cache, slots):
        plan = self.resources[0][slots.dtype]
        self.api.concat_and_cache_glm_next_mla_nvfp4(latent, cache, slots, plan=plan)

    def read_into(self, output, q, cache, indices, counts, mode):
        plan, tile, spec = self.resources[1][mode]
        # Conversion on the indexer's side stream touches only its own buffers.
        # Readers on separate execution lanes/ubatches get separate scratch.
        scratch, *packing = self._work_buffers(plan, tile, spec)
        for start in range(0, len(q), tile):
            s = slice(start, start + tile)
            active = counts[s]
            selected = indices[s]
            if packing:
                dst = packing[0][:len(selected)]
                dst.copy_(selected[:, :dst.size(1)])
                selected = dst
            binding = self.api.bind(plan, scratch=scratch, q=q[s].contiguous(), kv_cache=cache,
                selected_indices=selected,
                cache_lengths=active, selected_lengths=active)
            value = self.api.run(binding)
            output[s].copy_(value[0] if isinstance(value, tuple) else value)
