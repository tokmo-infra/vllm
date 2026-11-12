# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "glm53_install_overlay",
    _ROOT / "integrations/glm53nvfp4/install_overlay.py",
)
assert _SPEC is not None and _SPEC.loader is not None
overlay = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(overlay)


@pytest.fixture
def release(tmp_path):
    source = tmp_path / "source"
    package = tmp_path / "installed"
    (source / "vllm").mkdir(parents=True)
    package.mkdir()
    (source / "vllm/existing.py").write_text("value = 2\n")
    (source / "vllm/added.py").write_text("enabled = True\n")
    (package / "existing.py").write_text("value = 1\n")
    files = [
        {
            "path": "vllm/" + name,
            "sha256": overlay.sha256(source / "vllm" / name),
            "base_sha256": (
                overlay.sha256(package / name) if (package / name).exists() else None
            ),
        }
        for name in ("existing.py", "added.py")
    ]
    return source, package, {"base_vllm_version": "0.33.0", "files": files}


def test_installs_only_manifest_files_and_can_reapply(release):
    source, package, manifest = release
    expected = [entry["path"] for entry in manifest["files"]]
    for _ in range(2):
        assert overlay.install_overlay(source, package, manifest, "0.33.0") == expected
        assert (package / "existing.py").read_bytes() == (
            source / "vllm/existing.py"
        ).read_bytes()
        assert (package / "added.py").read_bytes() == (
            source / "vllm/added.py"
        ).read_bytes()


@pytest.mark.parametrize("version", ["0.32.0", "0.33.0rc3", "0.34.0"])
def test_rejects_old_or_prerelease_engine_before_writing(release, version):
    source, package, manifest = release
    with pytest.raises(ValueError, match="release mismatch"):
        overlay.install_overlay(source, package, manifest, version)
    assert (package / "existing.py").read_text() == "value = 1\n"
    assert not (package / "added.py").exists()


@pytest.mark.parametrize("side", ["source", "installed"])
def test_rejects_mixed_source_or_engine_without_partial_install(release, side):
    source, package, manifest = release
    if side == "source":
        (source / "vllm/added.py").write_text("wrong = True\n")
    else:
        (package / "added.py").write_text("foreign = True\n")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        overlay.install_overlay(source, package, manifest, "0.33.0")
    assert (package / "existing.py").read_text() == "value = 1\n"


@pytest.mark.parametrize("path", ["../outside.py", "/tmp/outside.py", "vllm/native.so"])
def test_rejects_paths_outside_python_overlay(release, path):
    source, package, manifest = release
    manifest["files"][1]["path"] = path
    with pytest.raises(ValueError, match="Invalid overlay path"):
        overlay.install_overlay(source, package, manifest, "0.33.0")
    assert (package / "existing.py").read_text() == "value = 1\n"
