"""Backport only native-overlap integration onto the frozen serving image."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil

repo = Path('/Users/librua/PycharmProjects/ai-infra/repos/vllm-roce-megamoe-deploy')
base = Path('work/image-base')
out = Path('work/serving-overlay')
out.mkdir(exist_ok=True)

def replace(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)

model_path = Path('vllm/models/deepseek_v4/nvidia/model.py')
model = (base / model_path).read_text()
model = replace(model, '        self.hierarchy = None\n', '''        self.hierarchy = None
        self._native_overlap = vllm_config.kernel_config.hierarchical_mega_moe_native_overlap
        self._native_chunk_size = vllm_config.kernel_config.hierarchical_mega_moe_native_chunk_size
        self._native_dispatch_slot = self.layer_id % vllm_config.kernel_config.hierarchical_mega_moe_dispatch_slots if self._native_overlap else 0
        self._native_return_slot = self.layer_id % vllm_config.kernel_config.hierarchical_mega_moe_return_slots if self._native_overlap else 0
''')
model = replace(model, '            config = vllm_config.kernel_config\n', '''            config = vllm_config.kernel_config
            if self._native_overlap and (
                config.hierarchical_mega_moe_transport != "verbs"
                or config.hierarchical_mega_moe_wire_dtype != "mxfp8"
                or config.hierarchical_mega_moe_return_dtype != "bfloat16"
                or config.hierarchical_mega_moe_rdma_host_staging
            ):
                raise ValueError("Native overlap requires direct verbs MXFP8/BF16 storage")
''')
model = replace(model, '                    host_staging=config.hierarchical_mega_moe_rdma_host_staging,\n', '                    host_staging=config.hierarchical_mega_moe_rdma_host_staging,\n                    peer_reads=self._native_overlap,\n')
start = model.index('    def get_symm_buffer(self):')
end = model.index('    def set_eplb_state(', start)
part = model[start:end].replace('def get_symm_buffer(self):', 'def get_symm_buffer(self, *, routing_groups=1, token_capacity=None):')
part = replace(part, '        backend = self._ensure_backend()\n', '        backend = self._ensure_backend()\n        token_capacity = token_capacity or self.max_num_tokens\n')
part = part.replace('self.max_num_tokens,', 'token_capacity,')
part = replace(part, '            token_capacity,\n', '            token_capacity,\n            routing_groups,\n') if part.count('            token_capacity,\n') == 1 else part.replace('            token_capacity,\n            self.top_k,', '            token_capacity,\n            routing_groups,\n            self.top_k,', 1)
part = replace(part, '                mma_type=backend.mma_type,\n', '                mma_type=backend.mma_type,\n                **({"routing_groups": routing_groups} if routing_groups != 1 else {}),\n')
model = model[:start] + part + model[end:]
model = replace(model, '                ) if self._hierarchical_direct_mxfp8 else None,\n', '''                ) if self._hierarchical_direct_mxfp8 else None,
                native_moe=(
                    lambda plan, hidden, gates, ids: self._forward_streaming(
                        plan, hidden, gates, ids,
                        activation_clamp=activation_clamp, fast_math=fast_math,
                    )
                ) if self._native_overlap else None,
                native_chunk_size=max(1, self._native_chunk_size // (2 if self.top_k <= 3 else 1)),
                native_dispatch_slot=self._native_dispatch_slot,
                native_return_slot=self._native_return_slot,
''')
method = '''    def _forward_streaming(self, plan, hidden, weights, ids, *, activation_clamp, fast_math):
        from vllm.utils.deep_gemm import _import_deep_gemm

        dg = _import_deep_gemm()
        backend = self._ensure_backend()
        buffer = self.get_symm_buffer(
            routing_groups=plan.nodes * plan.chunks,
            token_capacity=cdiv(self.max_num_tokens, 1024) * 1024 * plan.nodes,
        )
        kwargs = {}
        shared = None
        if self.has_fused_shared_experts:
            block_m = dg.get_block_m_for_mega_moe(
                plan.groups.topology.local_size, buffer.num_experts,
                buffer.num_max_tokens_per_rank, plan.nodes * plan.tokens,
                self.top_k, backend.mma_type,
            )
            prepare_megamoe_inputs(
                hidden, weights, ids,
                buffer.x[:plan.tokens], buffer.x_sf[:plan.tokens],
                buffer.topk_idx[:plan.tokens], buffer.topk_weights[:plan.tokens],
                shared_x_sf=buffer.shared_l1_acts_sf,
                shared_block_m=block_m, hidden_quant=backend.hidden_quant,
            )
            shared = plan.shared_output
            kwargs.update(
                shared_l1_weights=self._transformed_shared_l1_weights,
                shared_l2_weights=self._transformed_shared_l2_weights,
                shared_output=shared,
            )
        dg.fp8_fp4_mega_moe(
            plan.partial, self._transformed_l1_weights,
            self._transformed_l2_weights, buffer,
            activation_clamp=activation_clamp, fast_math=fast_math,
            scheduler_workspace=plan.scheduler_workspace(buffer),
            early_combine=True, input_descriptors=plan.input_descriptors,
            input_chunk_size=plan.capacity,
            return_descriptors=plan.return_descriptors,
            return_chunk_size=plan.return_chunk, **kwargs,
        )
        return shared

'''
model = replace(model, '    def _forward_hierarchical_mxfp8(\n', method + '    def _forward_hierarchical_mxfp8(\n')
model = replace(model, '''            and vllm_config.kernel_config.moe_backend
            != "deep_gemm_hierarchical_mega_moe"
''', '''            and (
                vllm_config.kernel_config.moe_backend != "deep_gemm_hierarchical_mega_moe"
                or vllm_config.kernel_config.hierarchical_mega_moe_native_overlap
            )
''')
(out / model_path).parent.mkdir(parents=True, exist_ok=True)
(out / model_path).write_text(model)

config_path = Path('vllm/config/kernel.py')
config = (base / config_path).read_text()
source = (repo / config_path).read_text()
new_fields = source[source.index('    hierarchical_mega_moe_native_overlap:'):source.index('    hierarchical_mega_moe_rdma_hcas:')]
config = replace(config, '    hierarchical_mega_moe_rdma_hcas:', new_fields + '    hierarchical_mega_moe_rdma_hcas:')
(out / config_path).parent.mkdir(parents=True, exist_ok=True)
(out / config_path).write_text(config)

for name in ('hierarchical_mega_moe.py', 'megamoe_rdma_ops.py', 'megamoe_native_overlap.py', 'roce/transport.py', 'roce/verbs.cpp', 'roce/protocol.h', 'roce/device.cuh'):
    path = Path('vllm/distributed') / name
    (out / path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / path, out / path)

diff = []
for path in (model_path, config_path):
    diff.extend(difflib.unified_diff((base / path).read_text().splitlines(True), (out / path).read_text().splitlines(True), fromfile='frozen-image/' + str(path), tofile='optimized-image/' + str(path)))
Path('work/serving-backport.patch').write_text(''.join(diff))
manifest = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'vllm').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
