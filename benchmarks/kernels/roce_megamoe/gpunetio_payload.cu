// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Standalone matched-payload probe for MSCCL++ PR #910's device PortChannel.
#include <cuda_runtime.h>
#include <mscclpp/core.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/utils.hpp>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Handle =
    mscclpp::PortChannelDeviceHandleT<mscclpp::PortChannelBackend::Device>;
void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

__global__ void exchange(Handle* handles, size_t bytes, int rank,
                         bool one_way) {
  if (threadIdx.x != 0) return;
  auto channel = handles[blockIdx.x];
  const size_t begin = bytes * blockIdx.x / gridDim.x;
  const size_t end = bytes * (blockIdx.x + 1) / gridDim.x;
  // All send buffers stay immutable. A received completion is followed by
  // an acknowledgement before the next iteration can overwrite the buffer.
  if (!one_way || rank == 0) {
    channel.putWithSignal<mscclpp::PortChannelSharingMode::Exclusive>(
        begin, begin, end - begin);
    channel.flush<mscclpp::PortChannelSharingMode::Exclusive>();
  } else {
    channel.signal<mscclpp::PortChannelSharingMode::Exclusive>();
    channel.flush<mscclpp::PortChannelSharingMode::Exclusive>();
  }
  channel.wait();
  channel.signal<mscclpp::PortChannelSharingMode::Exclusive>();
  channel.flush<mscclpp::PortChannelSharingMode::Exclusive>();
  channel.wait();
}

int main(int argc, char** argv) try {
  if (argc != 10) {
    std::cerr << "usage: probe RANK GPU NIC GID BOOTSTRAP_IP:PORT TOKENS "
                 "HIDDEN TOPK QPS\n";
    return 2;
  }
  const int rank = std::stoi(argv[1]), gpu = std::stoi(argv[2]);
  const int world =
      std::getenv("WORLD_SIZE") ? std::stoi(std::getenv("WORLD_SIZE")) : 2;
  if (world < 2 || world % 2)
    throw std::invalid_argument("expected two equal hosts");
  const int side = rank / (world / 2);
  const int peer = (rank + world / 2) % world;
  const int tokens = std::stoi(argv[6]), hidden = std::stoi(argv[7]);
  const int topk = std::stoi(argv[8]), qps = std::stoi(argv[9]);
  if ((world < 2 || world % 2 || rank < 0 || rank >= world) || tokens <= 0 ||
      hidden <= 0 || hidden % 128 || topk <= 0 || (qps != 1 && qps != 4))
    throw std::invalid_argument("invalid shape/rank/QPs");
  check(cudaSetDevice(gpu));
  // Resolve the selected NIC through a process-local transport slot. The
  // pinned getIBTransportByDeviceName API only handles eight enumerated NICs.
  setenv("MSCCLPP_HCA_DEVICES", argv[3], 1);
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world);
  bootstrap->initialize(argv[5]);
  auto comm = std::make_shared<mscclpp::Communicator>(bootstrap);
  const auto transport = mscclpp::Transport::IB0;
  const size_t dispatch_bytes =
      size_t(tokens) * (hidden + hidden / 32 + 8 * topk);
  const size_t return_bytes = size_t(tokens) * hidden * 2;
  const size_t capacity = std::max(dispatch_bytes, return_bytes);
  void *send, *receive;
  check(cudaMalloc(&send, capacity));
  check(cudaMalloc(&receive, capacity));
  check(cudaMemset(send, side + 17, capacity));
  check(cudaMemset(receive, 0, capacity));
  check(cudaDeviceSynchronize());
  auto source = comm->registerMemory(send, capacity, transport);
  auto destination = comm->registerMemory(receive, capacity, transport);
  std::vector<mscclpp::PortChannel> channels;
  std::vector<Handle> host_handles;
  for (int q = 0; q < qps; ++q) {
    mscclpp::EndpointConfig config(transport);
    config.ib.gidIndex = std::stoi(argv[4]);
    config.ib.mode = mscclpp::EndpointConfig::Ib::Mode::Device;
    auto connection = comm->connect(config, peer, q).get();
    comm->sendMemory(destination, peer, q);
    auto remote = comm->recvMemory(peer, q).get();
    auto sem = comm->buildSemaphore(connection, peer, q).get();
    channels.emplace_back(
        std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(sem), remote,
        source);
    host_handles.push_back(
        channels.back().deviceHandle<mscclpp::PortChannelBackend::Device>());
  }
  Handle* handles;
  check(cudaMalloc(&handles, qps * sizeof(Handle)));
  check(cudaMemcpy(handles, host_handles.data(), qps * sizeof(Handle),
                   cudaMemcpyHostToDevice));
  cudaStream_t stream;
  check(cudaStreamCreate(&stream));
  for (bool one_way : {true, false}) {
    for (int phase = 0; phase < 3; ++phase) {
      auto launch = [&] {
        if (phase != 1)
          exchange<<<qps, 32, 0, stream>>>(handles, dispatch_bytes, side,
                                           one_way);
        if (phase != 0)
          exchange<<<qps, 32, 0, stream>>>(handles, return_bytes, side,
                                           one_way);
        check(cudaGetLastError());
      };
      bootstrap->barrier();
      launch();
      check(cudaStreamSynchronize(stream));
      cudaGraph_t graph;
      cudaGraphExec_t executable;
      check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      launch();
      check(cudaStreamEndCapture(stream, &graph));
      check(cudaGraphInstantiate(&executable, graph, 0));
      cudaEvent_t begin, end;
      check(cudaEventCreate(&begin));
      check(cudaEventCreate(&end));
      bootstrap->barrier();
      check(cudaEventRecord(begin, stream));
      constexpr int iterations = 1000;
      for (int i = 0; i < iterations; ++i)
        check(cudaGraphLaunch(executable, stream));
      check(cudaEventRecord(end, stream));
      check(cudaEventSynchronize(end));
      float milliseconds;
      check(cudaEventElapsedTime(&milliseconds, begin, end));
      if (!one_way || side == 1) {
        const size_t bytes = phase == 0 ? dispatch_bytes : return_bytes;
        std::vector<unsigned char> result(bytes);
        check(
            cudaMemcpy(result.data(), receive, bytes, cudaMemcpyDeviceToHost));
        if (!std::all_of(result.begin(), result.end(),
                         [&](unsigned char c) { return c == 18 - side; }))
          throw std::runtime_error("received payload mismatch");
      }
      const size_t bytes =
          (phase != 1 ? dispatch_bytes : 0) + (phase != 0 ? return_bytes : 0);
      std::cout << "{\"rank\":" << rank << ",\"tokens\":" << tokens
                << ",\"qps\":" << qps << ",\"phase\":\""
                << (phase == 0   ? "dispatch"
                    : phase == 1 ? "return"
                                 : "roundtrip")
                << "\",\"direction\":\"" << (one_way ? "one_way" : "duplex")
                << "\",\"sent_bytes_per_active_lane\":" << bytes
                << ",\"milliseconds\":" << milliseconds / iterations << "}"
                << std::endl;
      check(cudaEventDestroy(begin));
      check(cudaEventDestroy(end));
      check(cudaGraphExecDestroy(executable));
      check(cudaGraphDestroy(graph));
    }
  }
  bootstrap->barrier();
  check(cudaStreamDestroy(stream));
  check(cudaFree(handles));
  // RegisteredMemory owns the registrations until communicator teardown.
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
