"""Use isolated new MegaMoE while preserving frozen attention and other ops."""
import difflib
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

subprocess.run(['work/.venv/bin/python', 'work/prepare-serving-overlay.py'], check=True, stdout=subprocess.DEVNULL)
base = Path('work/image-base')
out = Path('work/serving-overlay')
helper = out / 'vllm/distributed/megamoe_serving_backend.py'
helper.write_text('''"""Frozen-image deployment bridge for the isolated experimental MegaMoE."""

from functools import cache

@cache
def _import_deep_gemm():
    import deep_gemm_overlap
    from vllm.platforms import current_platform
    if current_platform.is_arch_support_pdl():
        deep_gemm_overlap.set_pdl(True)
    return deep_gemm_overlap
''')
paths = [Path('vllm/models/deepseek_v4/nvidia/model.py'), Path('vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe.py')]
for path in paths:
    target = out / path
    target.parent.mkdir(parents=True, exist_ok=True)
    text = target.read_text() if path == paths[0] else (base / path).read_text()
    old = 'from vllm.utils.deep_gemm import _import_deep_gemm'
    assert text.count(old) >= 3
    target.write_text(text.replace(old, 'from vllm.distributed.megamoe_serving_backend import _import_deep_gemm'))
diff = []
for path in [*paths, Path('vllm/config/kernel.py')]:
    diff.extend(difflib.unified_diff((base / path).read_text().splitlines(True), (out / path).read_text().splitlines(True), fromfile='frozen-image/' + str(path), tofile='optimized-image/' + str(path)))
Path('work/serving-backport.patch').write_text(''.join(diff))
manifest = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'vllm').rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('._')}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(out / 'Dockerfile.v7').write_text('''FROM xinhao-vllm-roce-megamoe:4110f55-donly
COPY vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY deep_gemm_overlap/ /opt/dbo-isolated/deep_gemm_overlap/
COPY manifest.json /opt/dbo-isolated/manifest.json
ENV PYTHONPATH=/opt/dbo-isolated
''')
for n in (89, 91):
    c=json.loads(Path(f'work/launch-opt-d16-{n}-v2.argv.json').read_text())
    c[c.index('--name')+1]=f'xinhao-dbo-opt-d16-{n}-v7'
    c[c.index('xinhao-vllm-roce-megamoe:dbo-stream-v2')]='xinhao-vllm-roce-megamoe:dbo-stream-v7'
    args=shlex.split(c[-1])
    assert args[args.index('--served-model-name')+1]=='deepseek-v4.1-flash-1017'
    Path(f'work/launch-opt-d16-{n}-v7.argv.json').write_text(json.dumps(c,indent=2)+'\n')
print(json.dumps({'native_package': 'deep_gemm_overlap', 'preserved_default': 'vllm.third_party.deep_gemm', 'manifest': manifest}, indent=2))
