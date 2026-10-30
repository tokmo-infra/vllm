#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect explicit Bash commands and check local resources; never start a service."""

import argparse
import hashlib
import json
import socket
import subprocess
import sys
from fnmatch import fnmatchcase
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GIB = 1024**3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scripts_for_node(node, root=ROOT):
    require(81 <= node <= 88, "Node must be 81–88")
    if node <= 84:
        names = [
            f"mooncake-store-{node}-0.sh",
            f"mooncake-store-{node}-1.sh",
            f"prefill-{node}-0.sh",
            f"prefill-{node}-1.sh",
        ]
        if node == 81:
            names += ["mooncake-master.sh", "router.sh"]
    else:
        names = [
            f"mooncake-store-{node}-0.sh",
            f"mooncake-store-{node}-1.sh",
            f"decode-{node}.sh",
        ]
    return [root / name for name in names]


def value(args, flag, default=None):
    if flag in args:
        return args[args.index(flag) + 1]
    prefix = flag + "="
    return next((a[len(prefix) :] for a in args if a.startswith(prefix)), default)


def read_plan(script, inventory):
    """Read actual argv/config through side-effect-free inspection modes."""
    script = Path(script)
    raw = subprocess.check_output(["bash", str(script), "--print-argv"])
    require(raw.endswith(b"\0"), f"Invalid argv output from {script.name}")
    command = raw[:-1].decode().split("\0")
    require(command[:2] == ["docker", "run"], f"{script.name} is not one docker run")
    cut = command.index("--entrypoint")
    docker_args = command[2:cut]
    entrypoint, image_id = command[cut + 1 : cut + 3]
    args = command[cut + 3 :]
    env, mounts = {}, []
    for i, item in enumerate(docker_args):
        if item == "-e":
            key, val = docker_args[i + 1].split("=", 1)
            env[key] = val
        elif item == "-v":
            mounts.append(tuple(docker_args[i + 1].rsplit(":", 2)))
    stem = script.stem
    instance = None
    if fnmatchcase(stem, "prefill-8[1-4]-[01]"):
        service, node, instance = "prefill", int(stem.split("-")[1]), int(stem[-1])
    elif fnmatchcase(stem, "decode-8[5-8]"):
        service, node = "decode", int(stem[-2:])
    elif fnmatchcase(stem, "mooncake-store-8[1-8]-[01]"):
        service, node, instance = "store", int(stem.split("-")[2]), int(stem[-1])
    elif stem in ("mooncake-master", "router"):
        service, node = ("master" if stem == "mooncake-master" else "router"), 81
    else:
        raise ValueError(f"Unrecognized independent command: {script.name}")
    model = service in ("prefill", "decode")
    gpus = [int(v) for v in value(args, "--device-ids").split(",")] if model else []
    files = {}
    if model:
        config = json.loads(
            subprocess.check_output(["bash", str(script), "--print-config"], text=True)
        )
        config_path = next(
            src for src, dst, _ in mounts if dst == "/etc/vllm/mooncake.json"
        )
        files[config_path] = config
    elif service == "store":
        topology = next(
            (src for src, dst, _ in mounts if dst == "/etc/mooncake/topology.json"),
            None,
        )
        if topology:
            files[topology] = json.loads(
                subprocess.check_output(
                    ["bash", str(script), "--print-config"], text=True
                )
            )
    ports = []
    if model:
        base = int(env["VLLM_MOONCAKE_BOOTSTRAP_PORT"])
        start = int(value(args, "--data-parallel-start-rank", "0"))
        local = int(value(args, "--data-parallel-size-local"))
        ports += list(range(base + start, base + start + local))
        ports.append(int(value(args, "--data-parallel-rpc-port")))
        if "--headless" not in args:
            ports.append(int(value(args, "--port")))
    elif service == "store":
        te_host, te_port = value(args, "--host").rsplit(":", 1)
        require(te_host == inventory["nodes"][str(node)]["ip"], "Wrong Store TE IP")
        ports = [
            int(te_port),
            int(value(args, "--port")),
            int(value(args, "--http_port")),
        ]
    elif service == "master":
        ports = [int(value(args, "--port")), int(value(args, "--metrics_port"))]
    else:
        ports = [int(value(args, "--port")), int(value(args, "--prometheus-port"))]
    image_key = (
        "router"
        if service == "router"
        else "mooncake"
        if service in ("store", "master")
        else "model"
    )
    return {
        "script": script.name,
        "service": service,
        "node": node,
        "instance": instance,
        "name": value(docker_args, "--name"),
        "image_id": image_id,
        "image_tag": inventory[f"{image_key}_image"],
        "gpus": gpus,
        "ports": ports,
        "files": files,
        "env": env,
        "mounts": mounts,
        "entrypoint": entrypoint,
        "args": args,
        "command": command,
    }


def all_plans(root=ROOT):
    inventory = json.loads((root / "cluster.json").read_text())
    return [
        read_plan(script, inventory)
        for node in range(81, 89)
        for script in scripts_for_node(node, root)
    ]


def run(args):
    return subprocess.check_output(args, text=True)


def rdma_devices(plan):
    """Include all data stacks, rather than checking the UCX list alone."""
    if plan["service"] == "store":
        return {
            name + ":1" for name in value(plan["args"], "--device_names").split(",")
        }
    if plan["service"] not in ("prefill", "decode"):
        return set()
    devices = set()
    devices.update(plan["env"]["NCCL_IB_HCA"].removeprefix("=").split(","))
    devices.update(name + ":1" for name in plan["env"]["MC_TE_FILTERS"].split(","))
    return devices


def check_numa_memory(inventory, plan):
    if plan["service"] != "store":
        return
    numa = plan["instance"]
    node_root = Path("/sys/devices/system/node") / f"node{numa}"
    actual_cpus = node_root.joinpath("cpulist").read_text().strip()
    require(
        actual_cpus == inventory["nodes"][str(plan["node"])]["numa_cpus"][numa],
        f"NUMA{numa} CPU layout differs from the prepared Store binding",
    )
    free_kib = next(
        (
            int(line.split()[-2])
            for line in node_root.joinpath("meminfo").read_text().splitlines()
            if "MemFree:" in line
        ),
        -1,
    )
    require(free_kib >= 0, f"Cannot read NUMA{numa} free RAM")
    need = (
        int(value(plan["args"], "--global_segment_size"))
        + int(value(plan["args"], "--local_buffer_size"))
    ) / GIB + inventory["preflight"]["numa_reserve_gib"]
    require(
        free_kib * 1024 >= need * GIB,
        f"NUMA{numa} free RAM < {need:g} GiB; includes local reserve",
    )
    require(
        value(plan["args"], "--membind") == str(numa)
        and value(plan["args"], "--cpunodebind") == str(numa),
        "Store CPU and memory policies do not match its NUMA instance",
    )
    if plan["node"] <= 84:
        require(
            plan["env"].get("MC_MS_AUTO_DISC") == "1",
            "P Store must enable the cpuset-scoped topology configuration",
        )
        topology = next(iter(plan["files"].values()))
        require(
            list(topology) == [f"cpu:{numa}"],
            "P Store topology declares memory outside its allowed NUMA",
        )
        preferred, fallback = topology[f"cpu:{numa}"]
        require(
            set(preferred).isdisjoint(fallback)
            and set(preferred + fallback)
            == {d.split(":")[0] for d in rdma_devices(plan)},
            "P Store topology does not retain its complete HCA whitelist",
        )
    for device in rdma_devices(plan):
        hca = device.split(":")[0]
        actual_numa = int(
            (Path("/sys/class/infiniband") / hca / "device/numa_node").read_text()
        )
        expected_numa = inventory["nodes"][str(plan["node"])]["hca_numa"][hca]
        require(actual_numa == expected_numa, f"{hca} NUMA differs from the inventory")
        if plan["node"] >= 85:
            require(actual_numa == numa, f"{hca} is not local to D Store NUMA{numa}")


def check_node_memory(inventory, plans):
    stores = [p for p in plans if p["service"] == "store"]
    if not stores:
        return
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    need = sum(
        int(value(p["args"], "--global_segment_size"))
        + int(value(p["args"], "--local_buffer_size"))
        for p in stores
    )
    need += inventory["preflight"]["host_reserve_gib"] * GIB
    # Node 81 also hosts the task's Master and Router; include their explicit limits.
    for plan in plans:
        if plan["service"] == "master":
            need += inventory["mooncake"]["master_memory_gib"] * GIB
        elif plan["service"] == "router":
            need += inventory["router"]["memory_gib"] * GIB
    require(
        available >= need,
        f"Host RAM cannot cover selected Stores plus reserve: {need / GIB:g} GiB",
    )


def check_local(inventory, plan):
    require(sys.platform == "linux", "Run resource checks on the Linux target node")
    node = inventory["nodes"][str(plan["node"])]
    addresses = json.loads(
        run(["ip", "-j", "-4", "addr", "show", "dev", node["socket_ifname"]])
    )
    require(
        any(
            a["local"] == node["ip"]
            for dev in addresses
            for a in dev.get("addr_info", [])
        ),
        f"This host does not have {node['ip']} on {node['socket_ifname']}",
    )
    image_key = (
        "router"
        if plan["service"] == "router"
        else "mooncake"
        if plan["service"] in ("store", "master")
        else "model"
    )
    require(
        plan["image_id"] == inventory[f"{image_key}_image_id"],
        "Command image differs from the pinned inventory",
    )
    actual = run(
        ["docker", "image", "inspect", plan["image_id"], "--format", "{{.Id}}"]
    ).strip()
    require(actual == plan["image_id"], "Pinned image ID is unavailable")
    exists = subprocess.run(
        ["docker", "container", "inspect", plan["name"]],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(exists.returncode != 0, f"Container already exists: {plan['name']}")
    for port in plan["ports"]:
        with socket.socket() as probe:
            try:
                probe.bind(("0.0.0.0", port))
            except OSError as error:
                raise ValueError(
                    f"Port {port} is unavailable (including TIME_WAIT): {error}"
                ) from error
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    service = plan["service"]
    if service in ("prefill", "decode"):
        require(
            plan["env"]["VLLM_KV_CACHE_LAYOUT"]
            == inventory["model"]["kv_cache_layout"],
            "KV cache layout differs from the deployed inventory",
        )
        require("--enforce-eager" not in plan["args"], "CUDA Graph must remain enabled")
        transfer = json.loads(value(plan["args"], "--kv-transfer-config"))
        require(
            transfer["kv_connector_extra_config"]["connectors"][0]["kv_connector"]
            == "MooncakeConnector",
            "PD must use the reference Mooncake connector",
        )
        require(
            value(plan["args"], "--prefix-match-unit") == "256",
            "P and D must share the explicit 256-token hash unit",
        )
        if service == "decode":
            require(
                plan["env"].get("VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE") == "0",
                "D must use the validated DeepEP PCIe direct mode",
            )
            require(
                "--enable-prefix-caching" in plan["args"]
                and value(plan["args"], "--prefix-match-unit") == "256",
                "D CPU cache requires KDA align and the shared 256-token hash unit",
            )
            runtime = next(
                Path(src)
                for src, dst, _ in plan["mounts"]
                if dst
                == "/usr/local/lib/python3.12/dist-packages/deep_ep/buffers/elastic.py"
            )
            patch = json.loads((ROOT / "runtime/patch-manifest.json").read_text())
            require(
                patch["base_image"] == plan["image_id"]
                and hashlib.sha256(runtime.read_bytes()).hexdigest()
                == patch["patched_sha256"],
                "DeepEP compatibility patch is missing or differs from its manifest",
            )
        runtime_ops = next(
            Path(src)
            for src, dst, _ in plan["mounts"]
            if dst == "/usr/local/lib/python3.12/dist-packages/vllm/_custom_ops.py"
        )
        ops_patch = json.loads(
            (ROOT / "runtime/custom-ops-patch-manifest.json").read_text()
        )
        require(
            ops_patch["base_image"] == plan["image_id"]
            and hashlib.sha256(runtime_ops.read_bytes()).hexdigest()
            == ops_patch["patched_sha256"],
            "Image-specific cache operator patch differs from its manifest",
        )
        token_patch = json.loads(
            (ROOT / "runtime/tokenizer-cache-manifest.json").read_text()
        )
        require(
            token_patch["base_image_id"] == plan["image_id"],
            "Tokenizer cache image differs",
        )
        for filename, digest in token_patch["files_sha256"].items():
            source = ROOT / "runtime" / filename
            require(
                hashlib.sha256(source.read_bytes()).hexdigest() == digest,
                "Tokenizer cache runtime differs from its validated manifest: "
                + filename,
            )
            target = (
                "renderers/base.py"
                if filename == "vllm_renderer_base.py"
                else "tokenizers/" + filename
            )
            require(
                (
                    str(source),
                    "/usr/local/lib/python3.12/dist-packages/vllm/" + target,
                    "ro",
                )
                in plan["mounts"],
                "Tokenizer cache must have exact read-only mounts",
            )
        cache_gib = 0 if "--headless" in plan["args"] else 4
        require(
            plan["env"].get("VLLM_GLM53_TOKENIZER_CACHE_GIB") == str(cache_gib),
            "Unexpected native tokenizer cache budget",
        )
        model_dir = next(Path(src) for src, dst, _ in plan["mounts"] if dst == "/model")
        config = json.loads((model_dir / "config.json").read_text())
        require(
            config["architectures"] == ["Glm5NextForConditionalGeneration"],
            "Wrong model",
        )
        require(
            config["text_config"]["n_routed_experts"] == 288, "Unexpected expert count"
        )
        require(
            hashlib.sha256((model_dir / "tokenizer.json").read_bytes()).hexdigest()
            == token_patch["tokenizer_sha256"],
            "Tokenizer SHA differs from exact-ID validation",
        )
        rows = [
            list(map(str.strip, row.split(",")))
            for row in run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,memory.used",
                    "--format=csv,noheader,nounits",
                ]
            ).splitlines()
        ]
        usage = {int(index): int(used) for index, _, used in rows}
        require(len(usage) == 8, "Expected eight GPUs")
        require(
            all(
                usage[g] <= inventory["preflight"]["gpu_max_used_mib"]
                for g in plan["gpus"]
            ),
            "Selected GPUs are occupied",
        )
        selected = {uuid for index, uuid, _ in rows if int(index) in plan["gpus"]}
        processes = run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]
        )
        require(
            not any(
                row.split(",")[0].strip() in selected for row in processes.splitlines()
            ),
            "Selected GPUs have compute processes",
        )
        api_count = (
            int(value(plan["args"], "--api-server-count", "0")) if cache_gib else 0
        )
        need = (
            inventory["preflight"]["model_min_host_available_gib"]
            + cache_gib * api_count
        )
    elif service == "store":
        driver = next(
            Path(src)
            for src, dst, _ in plan["mounts"]
            if dst == "/opt/host-driver/libcuda.so.1"
        )
        require(driver.is_file(), "CPU Store host libcuda is missing")
        need = (
            int(value(plan["args"], "--global_segment_size"))
            + int(value(plan["args"], "--local_buffer_size"))
        ) / GIB + inventory["preflight"]["host_reserve_gib"]
    else:
        need = (
            inventory["mooncake"]["master_memory_gib"]
            if service == "master"
            else inventory["router"]["memory_gib"]
        )
    require(
        available >= need * GIB,
        f"Available host RAM {available / GIB:.1f} GiB < {need:g} GiB",
    )
    devices = rdma_devices(plan)
    if devices:
        require(Path("/dev/infiniband").is_dir(), "RDMA device nodes are missing")
        for device in sorted(devices):
            hca, port = device.split(":")
            state = Path("/sys/class/infiniband") / hca / "ports" / port / "state"
            require("ACTIVE" in state.read_text(), f"Inactive RDMA port: {device}")
    check_numa_memory(inventory, plan)
    print(f"PASS {plan['script']}: local node, image, ports, GPU/RDMA and host memory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", type=int)
    parser.add_argument("--script", help="Check only one command assigned to this node")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print actual command plans only; no Docker/network/file writes",
    )
    args = parser.parse_args()
    try:
        inventory = json.loads((ROOT / "cluster.json").read_text())
        scripts = scripts_for_node(args.node)
        if args.script:
            selected = [p for p in scripts if p.name == args.script]
            require(selected, "The selected command does not belong to this node")
            scripts = selected
        plans = [read_plan(p, inventory) for p in scripts]
        if args.list:
            print(json.dumps(plans, ensure_ascii=False, indent=2))
            return
        require(sys.platform == "linux", "Run resource checks on the Linux target node")
        check_node_memory(inventory, plans)
        for plan in plans:
            check_local(inventory, plan)
    except (
        ValueError,
        OSError,
        KeyError,
        StopIteration,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
