// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cstdint>

namespace vllm::roce {

// Address order matches PeerSignals in transport.py. Flags live in mapped host
// memory owned by the endpoint and remain valid until transport teardown.
struct DeviceSignals {
  uint64_t request, send_done, receive_request, receive_done;
  uint64_t send_rows, receive_rows;
};
static_assert(sizeof(DeviceSignals) == 6 * sizeof(uint64_t));

__device__ __forceinline__ uint32_t acquire_signal(uint64_t address) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(value) : "l"(address) : "memory");
  return value;
}

// All data writers must finish and establish system visibility before the
// elected publisher calls this. A release by one thread alone does not join
// stores by other warps or CTAs.
__device__ __forceinline__ void release_signal(uint64_t address, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;"
               : : "l"(address), "r"(value) : "memory");
}
}  // namespace vllm::roce
