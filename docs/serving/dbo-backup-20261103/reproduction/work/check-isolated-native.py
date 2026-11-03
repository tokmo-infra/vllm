"""Check both import orders and independent runtime state before serving."""
import inspect
import json
import sys

import torch

from vllm.distributed.megamoe_serving_backend import _import_deep_gemm as overlap_import
from vllm.utils.deep_gemm import _import_deep_gemm as original_import

torch.cuda.set_device(0)
if sys.argv[1] == 'original-first':
    old, new = original_import(), overlap_import()
else:
    new, old = overlap_import(), original_import()
assert old.__name__ == 'vllm.third_party.deep_gemm'
assert new.__name__ == 'deep_gemm_overlap'
assert 'clean_logits' in inspect.signature(old.fp8_fp4_paged_mqa_logits).parameters
assert 'input_descriptors' in inspect.signature(new.fp8_fp4_mega_moe).parameters
old_sms, new_sms = old.get_num_sms(), new.get_num_sms()
new.set_num_sms(new_sms - 2)
assert old.get_num_sms() == old_sms, 'New runtime modified original runtime'
new.set_num_sms(new_sms)
old.set_num_sms(old_sms - 4)
assert new.get_num_sms() == new_sms, 'Original runtime modified new runtime'
old.set_num_sms(old_sms)
print(json.dumps({
    'order': sys.argv[1], 'original': old.__file__, 'overlap': new.__file__,
    'original_version': old.__version__, 'overlap_version': new.__version__,
    'runtime_state_independent': True,
    'original_attention_schema': str(torch.ops.deep_gemm.fp8_fp4_paged_mqa_logits.default._schema),
    'overlap_namespace': str(torch.ops.deep_gemm_overlap.fp8_fp4_mega_moe.default._schema),
}, indent=2), flush=True)
