// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Device/proxy acceptance probe. No stream completion may stand in for an
// in-kernel publish, acquire or buffer-lifetime acknowledgement.
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

#include "../../vllm/distributed/roce/device.cuh"

using namespace vllm::roce;

__device__ void wait_signal(uint64_t address, uint64_t deadline_cycles) {
  const auto start = clock64();
  while (acquire_signal(address) != 1) {
    if (clock64() - start > deadline_cycles) asm volatile("trap;");
  }
}

__global__ void resident_roundtrip(uint8_t* send, const uint8_t* receive,
                                   const DeviceSignals* signals,
                                   const int* generation, uint32_t rank,
                                   uint32_t peer_bytes, uint32_t row_bytes,
                                   uint64_t deadline_cycles) {
  const auto flags = *signals;
  const auto offset = (1 - rank) * peer_bytes;
  if (threadIdx.x == 0) release_signal(flags.receive_request, 1);
  if (rank == 1 && threadIdx.x == 0)
    wait_signal(flags.receive_done, deadline_cycles);
  __syncthreads();
  for (uint32_t i = threadIdx.x; i < row_bytes; i += blockDim.x)
    send[offset + i] = rank == 0 ? (i * 7 + *generation * 13) % 251
                                : (receive[offset + i] + 1) % 251;
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    release_signal(flags.send_rows, 1);
    release_signal(flags.request, 1);
    if (rank == 0) wait_signal(flags.receive_done, deadline_cycles);
    release_signal(flags.receive_done, 0);
    wait_signal(flags.send_done, deadline_cycles);
    release_signal(flags.send_done, 0);
  }
}

void roundtrip(uint64_t send, uint64_t receive, uint64_t signals,
               uint64_t generation, uint32_t rank, uint32_t peer_bytes,
               uint32_t row_bytes, uint64_t stream, uint64_t deadline_cycles) {
  resident_roundtrip<<<1, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<uint8_t*>(send), reinterpret_cast<const uint8_t*>(receive),
      reinterpret_cast<const DeviceSignals*>(signals),
      reinterpret_cast<const int*>(generation), rank, peer_bytes, row_bytes,
      deadline_cycles);
  const auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("roundtrip", &roundtrip);
}
