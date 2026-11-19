from importlib import metadata
import hashlib
import json
from pathlib import Path
import shutil
import sysconfig

site = Path(sysconfig.get_path("platlib"))
manifest = json.loads(Path("/opt/glm53/release.json").read_text())
for entry in manifest["runtime_source_files"]:
    p = site / entry["path"]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == entry["sha256"], str(p)
for p in manifest["historical_P_native_extensions"]:
    # Full recompilation does not imply the historical ELF hashes are identical.
    assert (site / p).is_file(), str(site / p)
assert metadata.version("vllm") == "0.32.0+glm53flash.67ca8f23"
assert metadata.version("flashinfer-python") == "0.7.1"
assert metadata.version("mooncake-transfer-engine-cuda13") == "0.3.13.post1"
assert shutil.which("nvcc")
assert (site / "flashinfer/data/csrc/tvm_ffi_utils.h").is_file()
assert (site / "mooncake/store.so").is_file()
print("Full-source build fingerprints and native artifact presence: OK")
