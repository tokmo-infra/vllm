# Matched RoCE payload probes

These probes isolate transport from MoE compute. They send the same payload
sizes as MXFP8 dispatch and BF16 return in the hierarchical backend, on eight
GPU/NIC lanes simultaneously. They do not implement a GPUNetIO serving backend.

## NCCL reference

Launch on both hosts with the cluster's NCCL/Gloo network environment:

```bash
.venv/bin/python -m torch.distributed.run \
    --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
    --master-addr="$HEAD_IP" --master-port=29720 \
    benchmarks/kernels/roce_megamoe/benchmark_payload.py \
    --tokens 768,936,1024,1536,4096 --iterations 1000 \
    --output /tmp/nccl-payload.json
```

## Experimental GPUNetIO probe

Build against the device PortChannel API in MSCCL++
[PR 910](https://github.com/microsoft/mscclpp/pull/910), pinned to
`c5825fbb69b1116287966a373d61ac0d17b7f87e` of
`qinghuazhou/mscclpp`, branch `gpunetio_submodule`. That revision pins
NVIDIA-DOCA/gpunetio to `c5d97fd52346763bcfeaff593dd1d84215156218`.
Follow its GPUNetIO prerequisites, including a supported GPU, RDMA driver,
GPU-visible queue resources and a NIC with a BlueFlame UAR.

```bash
git clone --branch gpunetio_submodule https://github.com/qinghuazhou/mscclpp.git
git -C mscclpp checkout c5825fbb69b1116287966a373d61ac0d17b7f87e
cmake -S mscclpp -B mscclpp/build \
    -DCMAKE_INSTALL_PREFIX="$PWD/mscclpp/install" \
    -DMSCCLPP_GPU_ARCHS=100 -DMSCCLPP_USE_GDRCOPY=OFF \
    -DMSCCLPP_BUILD_TESTS=OFF -DMSCCLPP_BUILD_PYTHON_BINDINGS=OFF \
    -DMSCCLPP_BUILD_EXT_EP=OFF -DMSCCLPP_BUILD_EXT_NCCL=OFF \
    -DMSCCLPP_BUILD_EXT_COLLECTIVES=OFF
cmake --build mscclpp/build -j8
cmake --install mscclpp/build
cmake -S benchmarks/kernels/roce_megamoe -B /tmp/roce-probe \
    -DCMAKE_PREFIX_PATH="$PWD/mscclpp/install" -DCMAKE_CUDA_ARCHITECTURES=100
cmake --build /tmp/roce-probe -j4
```

Launch on both hosts, substituting one verified NIC name per local GPU:

```bash
export LD_LIBRARY_PATH="$PWD/mscclpp/install/lib:$PWD/mscclpp/install/lib64:${LD_LIBRARY_PATH:-}"
.venv/bin/python -m torch.distributed.run \
    --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
    --master-addr="$HEAD_IP" --master-port=29721 \
    benchmarks/kernels/roce_megamoe/launch_gpunetio.py \
    --binary /tmp/roce-probe/gpunetio_payload --nics "$GPU_ORDERED_NICS" \
    --gid 3 --tokens 768,936,1024,1536,4096 --qps 1,4 \
    --output-directory /tmp/gpunetio-run-001
```

Both probes cover one-way and duplex dispatch, return and round-trip
payloads. GPUNetIO uses one or four QPs per lane, CUDA graph replay (1000
iterations), immutable sends and completion acknowledgements. Each rank
checks received bytes; compare the maximum rank latency for each case.
Use a new output directory per run. Do not report a driver/setup failure as
bandwidth or compare a single lane with eight simultaneously active lanes.

The initial B300 probe compiled successfully but failed during device-QP
creation (`DOCA_ERROR_DRIVER`); there is no valid GPUNetIO throughput result
in this archive. The NIC transport mapping has been made process-local to
avoid the pinned API's eight-device enumeration limit. Driver/QP setup still
needs verification before this transport can be assessed.
