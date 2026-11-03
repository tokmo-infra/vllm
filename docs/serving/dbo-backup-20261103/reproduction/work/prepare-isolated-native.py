"""Build a private operator namespace for the frozen image's MoE backport.

The deployed attention/GEMM operators retain the exact vendored library.
This transforms a separate copy, never the canonical pinned DeepGEMM source.
"""

import hashlib
import json
import shutil
from pathlib import Path

source = Path('/work/DeepGEMM')
target = Path('/work/DeepGEMM-isolated')
assert not target.exists(), 'Refuse to overwrite an existing build copy'
shutil.copytree(source, target, ignore=shutil.ignore_patterns('.git', 'build', 'dist', '*.egg-info', '__pycache__', '_C_extension*.so', '._*'))
changes = {}
for path in (target / 'csrc').rglob('*'):
    if path.suffix not in ('.hpp', '.cpp', '.h'):
        continue
    old = path.read_text()
    new = old
    for macro in ('STABLE_TORCH_LIBRARY', 'STABLE_TORCH_LIBRARY_FRAGMENT', 'STABLE_TORCH_LIBRARY_IMPL'):
        new = new.replace(f'{macro}(deep_gemm,', f'{macro}(deep_gemm_overlap,')
    if old != new:
        path.write_text(new)
        changes[str(path.relative_to(target))] = hashlib.sha256(new.encode()).hexdigest()
path = target / 'deep_gemm/_C.py'
text = path.read_text()
assert text.count('_torch_ops = torch.ops.deep_gemm') == 1
text = text.replace('_torch_ops = torch.ops.deep_gemm', '_torch_ops = torch.ops.deep_gemm_overlap')
path.write_text(text)
changes['deep_gemm/_C.py'] = hashlib.sha256(text.encode()).hexdigest()
path = target / 'setup.py'
text = path.read_text()
assert text.count("'-O3', '-fPIC'") == 1
text = text.replace("'-O3', '-fPIC'", "'-O3', '-fPIC', '-fvisibility=hidden'")
path.write_text(text)
changes['setup.py'] = hashlib.sha256(text.encode()).hexdigest()
(target / 'isolation-manifest.json').write_text(json.dumps(changes, indent=2) + '\n')
print(json.dumps({'source': str(source), 'target': str(target), 'namespace': 'deep_gemm_overlap', 'changed_files': changes}, indent=2), flush=True)
