#!/usr/bin/env python3
"""Install a source-built vLLM wheel into the existing Kimi runtime."""

import email.parser
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urldefrag

from packaging.requirements import Requirement

FEATURE_DATA_DIRS = (
    "vllm_flash_attn/cute",
    "third_party/triton_kernels",
    "third_party/deep_gemm",
    "third_party/fmha_sm100",
    "third_party/tml_fa4",
)
# This image is illustrative documentation, not an installed runtime asset.
NON_RUNTIME_INPUTS = {"distributed/kv_transfer/disagg_prefill_workflow.jpg"}


def fingerprint():
    import torch

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_git": torch.version.git_version,
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_member(archive, name):
    digest = hashlib.sha256()
    with archive.open(name) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_source(directory, output):
    package = Path(directory)
    files = {
        "vllm/" + path.relative_to(package).as_posix(): sha256_file(path)
        for path in package.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo", ".md"}
        and path.name not in {".gitkeep", "README"}
        and not is_native_name(path.name)
        and path.relative_to(package).as_posix() != "_version.py"
        and path.relative_to(package).as_posix() != "vllm-rs"
        and path.relative_to(package).as_posix() not in NON_RUNTIME_INPUTS
    }
    if not files:
        raise RuntimeError("no source files in Docker context")
    Path(output).write_text(json.dumps(files, sort_keys=True) + "\n")


def is_native_name(name):
    return name.endswith(".so") or ".so." in name


def snapshot_native(directory, output):
    package = Path(directory)
    files = {
        "vllm/" + path.relative_to(package).as_posix(): sha256_file(path)
        for path in package.rglob("*")
        if path.is_file()
        and (
            is_native_name(path.name)
            or path.relative_to(package).as_posix() == "vllm-rs"
        )
    }
    Path(output).write_text(json.dumps(files, sort_keys=True) + "\n")


def one_wheel(directory):
    wheels = list(Path(directory).glob("vllm-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one vLLM wheel, found {len(wheels)}")
    return wheels[0]


def verify_wheel_inputs(snapshot, rust_snapshot, native_directory, final_directory):
    sources = json.loads(Path(snapshot).read_text())
    rust = json.loads(Path(rust_snapshot).read_text())
    if not sources:
        raise RuntimeError("source snapshot is empty")
    if "vllm/vllm-rs" not in rust:
        raise RuntimeError("source-built Rust frontend is missing")
    with (
        zipfile.ZipFile(one_wheel(native_directory)) as native,
        zipfile.ZipFile(one_wheel(final_directory)) as final,
    ):
        native_names = {
            name
            for name in native.namelist()
            if name.startswith("vllm/") and is_native_name(name.rsplit("/", 1)[-1])
        }
        if not native_names:
            raise RuntimeError("source-built csrc wheel has no native extensions")
        final_list = final.namelist()
        final_names = set(final_list)
        if len(final_names) != len(final_list):
            raise RuntimeError("final wheel contains duplicate paths")
        missing = sorted((set(sources) | native_names | set(rust)) - final_names)
        if missing:
            raise RuntimeError(
                f"final wheel is missing {len(missing)} source/native files: "
                + ", ".join(missing[:20])
            )
        changed_sources = [
            name
            for name, digest in sources.items()
            if sha256_member(final, name) != digest
        ]
        changed_native = [
            name
            for name in native_names
            if sha256_member(native, name) != sha256_member(final, name)
        ]
        changed_rust = [
            name
            for name, digest in rust.items()
            if sha256_member(final, name) != digest
        ]
        final_native = {
            name
            for name in final_names
            if name.startswith("vllm/")
            and (is_native_name(name.rsplit("/", 1)[-1]) or name == "vllm/vllm-rs")
        }
        extra_native = sorted(final_native - native_names - set(rust))
        if changed_sources or changed_native or changed_rust or extra_native:
            raise RuntimeError(
                "final wheel differs from its source inputs: "
                + ", ".join(
                    (changed_sources + changed_native + changed_rust + extra_native)[
                        :20
                    ]
                )
            )
    print(
        f"Verified {len(sources)} source files, {len(native_names)} native files "
        f"and {len(rust)} Rust artifacts"
    )


def native_inventory(package):
    files = {
        "vllm/" + path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and is_native_name(path.name)
    }
    if (package / "vllm-rs").is_file():
        files.add("vllm/vllm-rs")
    return files


def feature_data_inventory(package):
    return {
        "vllm/" + path.relative_to(package).as_posix()
        for directory in FEATURE_DATA_DIRS
        for path in (package / directory).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def direct_url_matches(requirement):
    try:
        text = importlib.metadata.distribution(requirement.name).read_text(
            "direct_url.json"
        )
        info = json.loads(text) if text else {}
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return False
    url, fragment = urldefrag(requirement.url)
    fragments = parse_qs(fragment)
    subdirectories = fragments.get("subdirectory", [])
    if len(subdirectories) > 1 or info.get("subdirectory") != (
        subdirectories[0] if subdirectories else None
    ):
        return False
    git = re.fullmatch(r"git\+(.+)@([0-9a-fA-F]{40})", url)
    if git:
        vcs = info.get("vcs_info", {})
        return (
            info.get("url") == git.group(1)
            and vcs.get("vcs") == "git"
            and vcs.get("commit_id", "").lower() == git.group(2).lower()
        )
    hashes = fragments.get("sha256", [])
    archive = info.get("archive_info", {})
    if not isinstance(archive, dict):
        return False
    recorded = []
    hashes_info = archive.get("hashes", {})
    if not isinstance(hashes_info, dict):
        return False
    if "sha256" in hashes_info:
        recorded.append(hashes_info["sha256"])
    old_hash = archive.get("hash", "")
    if not isinstance(old_hash, str):
        return False
    algorithm, _, old_digest = old_hash.partition("=")
    if algorithm.lower() == "sha256":
        recorded.append(old_digest)
    return (
        len(hashes) == 1
        and re.fullmatch(r"[0-9a-fA-F]{64}", hashes[0]) is not None
        and info.get("url") == url
        and bool(recorded)
        and all(
            isinstance(value, str) and value.lower() == hashes[0].lower()
            for value in recorded
        )
    )


def missing_requirements(requirements, root_extras=()):
    missing = set()
    queue = [(text, set(root_extras) | {""}) for text in requirements or []]
    expanded = set()
    while queue:
        text, parent_extras = queue.pop()
        requirement = Requirement(text)
        if requirement.marker and not any(
            requirement.marker.evaluate({"extra": extra}) for extra in parent_extras
        ):
            continue
        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            missing.add(str(requirement))
            continue
        else:
            if requirement.specifier and not requirement.specifier.contains(
                version, prereleases=True
            ):
                missing.add(str(requirement))
            if requirement.url and not direct_url_matches(requirement):
                missing.add(str(requirement))
        key = (requirement.name.lower(), tuple(sorted(requirement.extras)))
        if key not in expanded:
            expanded.add(key)
            extras = set(requirement.extras) | {""}
            queue.extend(
                (dependency, extras)
                for dependency in importlib.metadata.requires(requirement.name) or []
            )
    return missing


def new_missing_requirements(old, new, root_extras=()):
    old = old or []
    new = new or []
    old_requirements = {str(Requirement(text)) for text in old}
    added_or_changed = [
        text for text in new if str(Requirement(text)) not in old_requirements
    ]
    return (
        missing_requirements(new, root_extras) - missing_requirements(old, root_extras)
    ) | missing_requirements(added_or_changed, root_extras)


def wheel_metadata(archive):
    names = archive.namelist()
    metadata_files = [name for name in names if name.endswith(".dist-info/METADATA")]
    if len(metadata_files) != 1:
        raise RuntimeError("wheel must contain exactly one METADATA file")
    metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_files[0]))
    if metadata.get("Name", "").lower() != "vllm":
        raise RuntimeError("wheel is not a vLLM distribution")
    return metadata


def pip_check():
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = result.stdout.strip()
    if result.returncode not in (0, 1) or (
        result.returncode == 1
        and (
            not output
            or any(
                not re.fullmatch(
                    r".+ \S+ (?:requires .+, which is not installed\.|"
                    r"has requirement .+, but you have .+\.|"
                    r"is not supported on this platform\.?)",
                    line,
                )
                for line in output.splitlines()
            )
        )
    ):
        raise RuntimeError(f"pip check did not complete successfully: {output}")
    return {"exit_code": result.returncode, "output": output}


def pip_problems(result):
    if result["exit_code"] == 0:
        return set()
    return {
        re.sub(r"^vllm\s+\S+\s+", "vllm <version> ", line)
        for line in result.get("output", "").splitlines()
        if line.strip()
    }


def install(directory, record_path="/opt/kimi-k31-build/source-build.json"):
    directory = Path(directory)
    wheel = one_wheel(directory)
    build_abi = json.loads((directory / "build-abi.json").read_text())
    csrc_abi = json.loads((directory / "csrc-abi.json").read_text())
    runtime_abi = fingerprint()
    if build_abi != runtime_abi or csrc_abi != runtime_abi:
        raise RuntimeError(
            f"build/runtime ABI mismatch: csrc={csrc_abi}, build={build_abi}, "
            f"runtime={runtime_abi}"
        )

    spec = importlib.util.find_spec("vllm")
    if not spec or not spec.origin:
        raise RuntimeError("runtime has no installed vLLM package")
    package = Path(spec.origin).resolve().parent
    allowed_parents = {
        Path(sysconfig.get_paths()[name]).resolve() for name in ("purelib", "platlib")
    }
    if package.name != "vllm" or package.parent not in allowed_parents:
        raise RuntimeError(f"refusing to replace unexpected package path {package}")

    expected = native_inventory(package)
    feature_data = feature_data_inventory(package)
    if not expected or "vllm/vllm-rs" not in expected:
        raise RuntimeError("runtime is missing expected native vLLM artifacts")
    before_requirements = importlib.metadata.requires("vllm")
    before_check = pip_check()
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        missing = sorted((expected | feature_data) - names)
        if missing:
            raise RuntimeError(
                "source-built wheel is missing runtime artifacts "
                f"({len(missing)}): " + ", ".join(missing[:20])
            )
        metadata = wheel_metadata(archive)
        new_missing = new_missing_requirements(
            before_requirements,
            metadata.get_all("Requires-Dist"),
            metadata.get_all("Provides-Extra") or (),
        )
        if new_missing:
            raise RuntimeError(
                "runtime lacks new or changed dependencies: "
                + ", ".join(sorted(new_missing))
            )
        native_hashes = {
            name: hashlib.sha256(archive.read(name)).hexdigest() for name in expected
        }
        wheel_version = metadata["Version"]

    subprocess.run(["uv", "pip", "uninstall", "--system", "vllm"], check=True)
    if package.exists():
        shutil.rmtree(package)
    subprocess.run(
        ["uv", "pip", "install", "--system", "--no-deps", "--no-cache", str(wheel)],
        check=True,
    )
    for name, expected_hash in native_hashes.items():
        installed = package.parent / name
        if not installed.is_file() or sha256_file(installed) != expected_hash:
            raise RuntimeError(
                f"installed artifact differs from source-built wheel: {name}"
            )
    after_check = pip_check()
    new_problems = pip_problems(after_check) - pip_problems(before_check)
    if new_problems:
        raise RuntimeError(
            "source-built wheel introduced dependency conflicts: "
            + "; ".join(sorted(new_problems))
        )

    record = {
        "wheel": wheel.name,
        "wheel_sha256": sha256_file(wheel),
        "version": wheel_version,
        "build_commit": os.environ.get("VLLM_BUILD_COMMIT", "unknown"),
        "build_abi": build_abi,
        "csrc_abi": csrc_abi,
        "runtime_abi": runtime_abi,
        "native_sha256": native_hashes,
        "feature_data_files_checked": len(feature_data),
        "pip_check_before": before_check,
        "pip_check_after": after_check,
    }
    output = Path(record_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"Installed source-built vLLM {wheel_version}; recorded {output}")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "fingerprint":
        Path(sys.argv[2]).write_text(json.dumps(fingerprint(), indent=2) + "\n")
    elif len(sys.argv) == 3 and sys.argv[1] == "install":
        install(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "snapshot":
        snapshot_source(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "snapshot-native":
        snapshot_native(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 6 and sys.argv[1] == "verify":
        verify_wheel_inputs(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        raise SystemExit(
            "usage: source-wheel.py fingerprint OUTPUT | install WHEEL_DIR | "
            "snapshot VLLM_DIR OUTPUT | snapshot-native VLLM_DIR OUTPUT | "
            "verify SNAPSHOT RUST_SNAPSHOT CSRC_DIR FINAL_DIR"
        )


if __name__ == "__main__":
    main()
