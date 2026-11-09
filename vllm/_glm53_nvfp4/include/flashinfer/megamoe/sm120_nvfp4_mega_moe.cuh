/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Experimental NVFP4 W4A4 fork of FlashInfer PR8806
// b73ffcf962540820e98c908688a8a0171a71e176. NVFP4 E2M1 codes + E4M3 group16
// scales; no MXFP4/E4M3 reinterpretation. The persistent scheduling, P2P/GIN
// release/acquire protocol, bounded GPU waits, SwiGLU and combine remain from
// the pinned implementation. Only payload encoding, MMA fragments/weight
// addressing/calibration, and intermediate quantization change. The CUDA 12.8+
// NVFP4 MMA contract is scale_vec::4X (four group16 scales per K64). W13 may
// alias native [up;gate] weights and native swizzled scale storage.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#ifndef FLASHINFER_MEGAMOE_SM120_GIN
  #define FLASHINFER_MEGAMOE_SM120_GIN 0
#endif
#ifndef FLASHINFER_NVFP4_VECTOR_WEIGHTS
  #define FLASHINFER_NVFP4_VECTOR_WEIGHTS 0
#endif
#ifndef FLASHINFER_NVFP4_EXTENDED_PREFETCH
  #define FLASHINFER_NVFP4_EXTENDED_PREFETCH 0
#endif
#ifndef FLASHINFER_NVFP4_STAGE_FC2_INPUT
  #define FLASHINFER_NVFP4_STAGE_FC2_INPUT 0
#endif
// GIN code is device-pass only: NCCL's device headers do not compile in nvcc's
// host pass, and the kernel's signature carries no NCCL type
#if FLASHINFER_MEGAMOE_SM120_GIN && defined(__CUDA_ARCH__)
  #define MEGAMOE_SM120_GIN_DEVICE 1
  #include <nccl_device.h>
#else
  #define MEGAMOE_SM120_GIN_DEVICE 0
#endif

namespace flashinfer::megamoe::sm120_nvfp4 {

constexpr int kWarps = 8;
constexpr int kNbPerWarp = 1;  // 16-row weight blocks per warp per item
constexpr int kRowsPerItem =
    kWarps * kNbPerWarp * 16;  // 128 weight rows per work item
constexpr int kTile = 8;  // tokens per MMA (N); a work item covers kTileN = 8,
                          // 16, 24 or 32 tokens
constexpr int kPfBytes =
    8192;  // per 16-row weight block, prefetched to L2 during dispatch
// Inherited prefetch distance; NVFP4-specific benefit is not yet measured.
constexpr int kPfAhead = 128;
constexpr size_t kPf0Bytes =
    size_t(64) << 20;  // t=0 L2 prefetch budget over the locally known experts
constexpr int kUnroll = 4;    // K64 steps in flight per warp
constexpr int kMaxWorld = 8;  // ranks: one PCIe host
constexpr int kMaxEl = 256;   // local experts per rank
constexpr int kMaxTopk =
    8;  // routes per token (pool metadata packs token * 8 + k)
constexpr uint64_t kSpinNs =
    30000000000ull;  // spin timeout: startup warm-ups can skew ranks by seconds

// an expert-pool row: packed NVFP4 token (H/2 bytes), E4M3 scales (H/16 bytes)
// and the source's (token * 8 + k), padded to 16 bytes, so a dispatched copy is
// one contiguous store / put
__host__ __device__ constexpr size_t pool_row_bytes(int H) {
  return size_t(H) / 2 + ((size_t(H) / 16 + sizeof(int) + 15) & ~size_t(15));
}

// error codes (counter kCtrErr)
constexpr int kErrDispatchWait = 1, kErrL1Wait = 2, kErrCombineWait = 3,
              kErrRouteWait = 4;

// counters (local, reset by the last CTA to exit)
constexpr int kCtrDispatch = 0, kCtrQueue = 1, kCtrL2Done = 2, kCtrErr = 3,
              kCtrExit = 4, kCtrEpoch = 5, kCtrL1 = 8;
// GIN transport state: per-tile FC2 completion counts, this call's expected GIN
// signal counts per source (dispatch, combine) and the cumulative totals (u64,
// persistent across calls)
constexpr int kMaxTiles =
    kMaxEl + kMaxWorld * 256 * kMaxTopk / 8;  // sum_el ceil(n_el / kTile)
constexpr int kCtrTile = kCtrL1 + kMaxEl, kCtrCnt = kCtrTile + kMaxTiles,
              kCtrCum = kCtrCnt + 2 * kMaxWorld;
constexpr int kCtrInts = kCtrCum + 2 * 2 * kMaxWorld;
static_assert(kCtrCum % 2 == 0, "cumulative GIN totals must be 8-byte aligned");
// GIN signal ids at the receiver, per source rank
constexpr int kSigRoute = 0, kSigDispatch = kMaxWorld, kSigComb = 2 * kMaxWorld,
              kGinSignals = 3 * kMaxWorld;

struct Peers {
  uint8_t*
      pool[kMaxWorld];  // [EL][W][TCAP] rows of pool_row_bytes(H) (dest-owned)
  int* recv_cnt[kMaxWorld];        // [W][EL]
  int* recv_flag[kMaxWorld];       // [W] epoch
  __nv_bfloat16* comb[kMaxWorld];  // [TCAP][TOPK][H] (src-owned)
  int* comb_flag[kMaxWorld];       // [W] epoch
  int* route[kMaxWorld];  // [W][1 + TCAP*TOPK]: T, then the source's top-k ids
                          // (dest-owned)
  int* route_flag[kMaxWorld];  // [W] epoch
};

// NIC transport: window offsets of the receive regions (identical on every
// rank) and of this rank's staging regions, plus their local pointers. mask ==
// 0 (always in the P2P build): no GIN pair.
struct Gin {
  const void* comm;  // ncclDevComm const*, device memory
  void* win;         // ncclWindow_t of the whole buffer
  uint32_t mask;     // bit w: rank w is reached over GIN
  int nctx;  // GIN contexts: CTA b posts on context b % nctx; a signal is the
             // sum over them
  size_t off_route, off_inbox, off_comb_gin, off_stage_row, off_stage_comb,
      off_slotmap, off_stage_slot;
  uint8_t* stage_row;  // [TCAP] pool rows: a token's NVFP4 row, sent once per
                       // GIN rank
  uint8_t* inbox;  // [W][TCAP] pool rows: tokens received from each GIN source
  __nv_bfloat16* stage_comb;  // [W][TCAP*TOPK][H]: partials per GIN source,
                              // packed (el, slot)
  __nv_bfloat16* comb_gin;  // [W][TCAP*TOPK][H]: partials from each GIN expert
                            // rank, packed
  int* slotmap;  // [W][TCAP*TOPK]: GIN source's pool slots, packed (el, slot)
                 // -> token
  int* stage_slot;  // [W][TCAP*TOPK]: the slot table sent to each GIN rank
  int* cidx;        // [TCAP*TOPK]: packed row of (token, k) sent to a GIN rank
};

struct Args {
  const uint8_t* x;         // [T][H/2] packed E2M1, staged by the caller
  const uint8_t* x_sf;      // [T][H/16] E4M3 scales per 16 channels
  const int64_t* topk_ids;  // [T][TOPK] global expert ids, -1 = masked route
  const float* topk_w;      // [T][TOPK]
  __nv_bfloat16* out;       // [T][H]
  const uint8_t*
      l1;  // [EL][2I][H/2] packed E2M1 W13, row-major, gate/up rows interleaved
  const uint8_t* l2;   // [EL][H][I/2] packed E2M1 W2, row-major
  const uint8_t* l1s;  // [EL][2I][H/16] raw or native-swizzled E4M3 W13 scales
  const uint8_t* l2s;  // [EL][H][I/16] raw or native-swizzled E4M3 W2 scales
  size_t l1_stride, l2_stride, l1s_stride, l2s_stride;  // bytes per expert
  const float *gs1,
      *gs2;  // required [EL] native g1/g2_alphas, includes input dequant scale
  const float*
      a2_gscale;   // [EL], per-expert inverse input_scale for post-SwiGLU NVFP4
  int w13_layout;  // 0=[gate;up], 1=native [up;gate], 2=interleaved8
  int scale_layout;   // 0=plain [row,K/16], 1=native CUTLASS swizzle128
  int* stats;         // optional [EL] cumulative received-token counts
  uint8_t* inter;     // [EL][W*TCAP][I/2] packed E2M1 (local)
  uint32_t* inter_s;  // [EL][W*TCAP][I/64] four E4M3 scale bytes per word
  int* ctr;           // kCtrInts
  unsigned long long* stamps;  // optional [grid][16] globaltimer stamps (debug)
  Peers peers;
  Gin gin;
  int T, H, I, EL, W, TOPK, TCAP, rank;
  float swiglu_limit;
};

__device__ __forceinline__ const uint8_t* w13_of(const Args& a, int el) {
  return a.l1 + size_t(el) * a.l1_stride;
}
__device__ __forceinline__ const uint32_t* ws13_of(const Args& a, int el) {
  return reinterpret_cast<const uint32_t*>(a.l1s + size_t(el) * a.l1s_stride);
}
__device__ __forceinline__ const uint8_t* w2_of(const Args& a, int el) {
  return a.l2 + size_t(el) * a.l2_stride;
}
__device__ __forceinline__ const uint32_t* ws2_of(const Args& a, int el) {
  return reinterpret_cast<const uint32_t*>(a.l2s + size_t(el) * a.l2s_stride);
}

__device__ __forceinline__ uint64_t gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ int ld_acquire_sys(const int* p) {
  int v;
  asm volatile("ld.acquire.sys.global.b32 %0, [%1];"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(int* p, int v) {
  asm volatile("st.release.sys.global.b32 [%0], %1;" ::"l"(p), "r"(v)
               : "memory");
}
// thread 0 spins until *p >= v (sys or gpu scope); returns false on timeout
__device__ __forceinline__ bool wait_ge(const int* p, int v, bool sys) {
  const uint64_t t0 = gtime();
  while ((sys ? ld_acquire_sys(p) : ld_acquire_gpu(p)) < v)
    if (gtime() - t0 > kSpinNs) return false;
  return true;
}

__device__ __forceinline__ unsigned long long* gin_cum(
    int* ctr) {  // [2][kMaxWorld]
  return reinterpret_cast<unsigned long long*>(ctr + kCtrCum);
}

#if MEGAMOE_SM120_GIN_DEVICE
__device__ __forceinline__ bool is_gin(const Args& a, int w) {
  return (a.gin.mask >> w) & 1u;
}
__device__ __forceinline__ const ncclDevComm& gin_comm(const Args& a) {
  return *static_cast<const ncclDevComm*>(a.gin.comm);
}
__device__ __forceinline__ ncclWindow_t gin_win(const Args& a) {
  return static_cast<ncclWindow_t>(a.gin.win);
}
// a CTA posts every put on one context (its own puts to a peer stay ordered, so
// a strong signal settles the earlier puts of the same thread); CTAs spread
// over the contexts' queue pairs
__device__ __forceinline__ int gin_ctx(const Args& a) {
  return int(blockIdx.x) % a.gin.nctx;
}
template <typename Action>
__device__ __forceinline__ void gin_put(const Args& a, int peer, size_t dst_off,
                                        size_t src_off, size_t bytes,
                                        Action action) {
  const ncclDevComm& c = gin_comm(a);
  ncclGin g(c, gin_ctx(a));
  g.put(ncclTeamWorld(c), peer, gin_win(a), dst_off, gin_win(a), src_off, bytes,
        action);
}
template <typename Action>
__device__ __forceinline__ void gin_signal(const Args& a, int peer,
                                           Action action) {
  const ncclDevComm& c = gin_comm(a);
  ncclGin g(c, gin_ctx(a));
  g.signal(ncclTeamWorld(c), peer, action);
}
// thread 0 spins until GIN signal sig, summed over the contexts, reaches target
// (rolling 64-bit compare); false on timeout
__device__ __forceinline__ bool gin_wait(const Args& a, int sig,
                                         unsigned long long target) {
  const uint64_t t0 = gtime();
  while (true) {
    unsigned long long v = 0;
    for (int x = 0; x < a.gin.nctx; ++x)
      v +=
          ncclGin(gin_comm(a), x).readSignal(static_cast<ncclGinSignal_t>(sig));
    if (static_cast<long long>(v - target) >= 0) return true;
    if (gtime() - t0 > kSpinNs) return false;
  }
}
#else
__device__ __forceinline__ constexpr bool is_gin(const Args&, int) {
  return false;
}
#endif

// SPDX-License-Identifier: Apache-2.0
// NVFP4 quant/pack contract also follows vLLM contributors nvfp4_utils.cuh.
// NVIDIA CUTLASS MMA_Traits SM120_16x8x64_TN_VS specifies the packed register
// layouts below. No K permutation, no 4-bit-in-8-bit padding for mxf4nvf4.
// E4M3 scales per K16 -> FOUR bytes per K64 (scale_vec::4X), not 2X.
__device__ __forceinline__ void mma_nvfp4(float (&d)[4], const uint32_t (&a)[4],
                                          uint32_t b0, uint32_t b1, uint32_t sa,
                                          uint32_t sb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row."
      "col."
      "f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},"
      "{%10},{%11,%12},{%13},{%14,%15};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa),
        "h"(uint16_t{0}), "h"(uint16_t{0}), "r"(sb), "h"(uint16_t{0}),
        "h"(uint16_t{0}));
}

// Internal FC1 tile rows are gate8/up8 so paired accumulators share a lane.
// Map to native [up;gate] or checkpoint [gate;up] WITHOUT copying/requantizing
// weights.
__host__ __device__ __forceinline__ int nvfp4_weight_row(int row, int rows,
                                                         int layout) {
  if (layout < 0 || layout == 2)
    return row;  // FC2 or preinterleaved test input
  int logical = (row / 16) * 8 + row % 8;
  bool up = row % 16 >= 8;
  if (layout == 1) return logical + (up ? 0 : rows / 2);  // native [up;gate]
  return logical + (up ? rows / 2 : 0);  // checkpoint [gate;up]
}

// Byte-to-word addressing of native swizzle_blockscale (128-row x 4-scale
// tiles). Returned pointer addresses four contiguous E4M3 scale bytes for one
// row/K64.
__host__ __device__ __forceinline__ size_t nvfp4_scale_word(int row, int k64,
                                                            int K, int layout) {
  const int words_per_row = K / 64;
  if (layout == 0) return size_t(row) * words_per_row + k64;
  return (size_t(row / 128) * words_per_row + k64) * 128 +
         size_t(row % 32) * 4 + (row % 128) / 32;
}

// 4x4 word transpose across the lanes of a quad: v[q] of lane c becomes v[c] of
// lane q
__device__ __forceinline__ void quad_transpose(uint32_t (&v)[4], int c) {
#pragma unroll
  for (int m = 2; m >= 1; m >>= 1) {
    const bool hi = (c & m) != 0;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      if (q & m) continue;
      const uint32_t r = __shfl_xor_sync(0xffffffffu, hi ? v[q] : v[q | m], m);
      if (hi)
        v[q] = r;
      else
        v[q | m] = r;
    }
  }
}

template <int NH>
__device__ __forceinline__ void item_gemm(float (&acc)[kNbPerWarp][NH][4],
                                          const uint8_t* W, const uint32_t* WS,
                                          int nblocks, int nb0, int K,
                                          const uint8_t* const (&xrow)[NH],
                                          const uint32_t* const (&xsrow)[NH],
                                          int lane, int weight_layout,
                                          int scale_layout) {
  const int g = lane / 4, c = lane % 4;
#pragma unroll
  for (int j = 0; j < kNbPerWarp; ++j)
#pragma unroll
    for (int h = 0; h < NH; ++h)
#pragma unroll
      for (int z = 0; z < 4; ++z) acc[j][h][z] = 0.f;
  // Register mappings: A rows g/g+8, K=8c..8c+7 / 32+8c..39+8c.
  // Scale A row g+8*(c&1), duplicated across c and c+2. B row g, replicated
  // quad. Keep kUnroll chunks in flight and reuse each W fragment across NH
  // token groups.
  for (int k0 = 0; k0 < K / 64; k0 += kUnroll) {
    uint32_t wa[kNbPerWarp][kUnroll][4], sa[kNbPerWarp][kUnroll];
#pragma unroll
    for (int j = 0; j < kNbPerWarp; ++j) {
      const int rg =
          nvfp4_weight_row((nb0 + j) * 16 + g, nblocks * 16, weight_layout);
      const int ru =
          nvfp4_weight_row((nb0 + j) * 16 + g + 8, nblocks * 16, weight_layout);
      const int rs = (c & 1) ? ru : rg;
      const uint8_t* wg = W + size_t(rg) * (K / 2);
      const uint8_t* wu = W + size_t(ru) * (K / 2);
      if (lane < 16 && k0 * 32 + kPfAhead < K / 2) {
        const char* pf = reinterpret_cast<const char*>(W) +
                         size_t(nvfp4_weight_row((nb0 + j) * 16 + lane,
                                                 nblocks * 16, weight_layout)) *
                             (K / 2);
        asm volatile("prefetch.global.L2 [%0];" ::"l"(pf + k0 * 32 + kPfAhead));
      }
#if FLASHINFER_NVFP4_VECTOR_WEIGHTS
      // PR8806's vector load + quad transpose, adapted to NVFP4 K64 MMA.
      // No nibble spreading or MXFP4 UE8M0 scale reinterpretation: register
      // bytes match the original four scalar loads for every lane and K64.
  #pragma unroll
      for (int p = 0; p < kUnroll / 2; ++p) {
  #pragma unroll
        for (int r = 0; r < 2; ++r) {
          const uint8_t* row = r == 0 ? wg : wu;
          const uint4 q = __ldcs(reinterpret_cast<const uint4*>(row) +
                                 (k0 / 2 + p) * 4 + c);
          uint32_t v[4] = {q.x, q.y, q.z, q.w};
          quad_transpose(v, c);
  #pragma unroll
          for (int b = 0; b < 4; ++b)
            wa[j][p * 2 + b / 2][r + 2 * (b % 2)] = v[b];
        }
      }
  #pragma unroll
      for (int u = 0; u < kUnroll; ++u)
        sa[j][u] = __ldg(WS + nvfp4_scale_word(rs, k0 + u, K, scale_layout));
#else
  #pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const int off = (k0 + u) * 32 + c * 4;
        wa[j][u][0] = __ldcs(reinterpret_cast<const uint32_t*>(wg + off));
        wa[j][u][1] = __ldcs(reinterpret_cast<const uint32_t*>(wu + off));
        wa[j][u][2] = __ldcs(reinterpret_cast<const uint32_t*>(wg + off + 16));
        wa[j][u][3] = __ldcs(reinterpret_cast<const uint32_t*>(wu + off + 16));
        sa[j][u] = __ldg(WS + nvfp4_scale_word(rs, k0 + u, K, scale_layout));
      }
#endif
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        const uint8_t* xb = xrow[h] + (k0 + u) * 32 + c * 4;
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(xb);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(xb + 16);
        const uint32_t sb = xsrow[h][k0 + u];
#pragma unroll
        for (int j = 0; j < kNbPerWarp; ++j)
          mma_nvfp4(acc[j][h], wa[j][u], b0, b1, sa[j][u], sb);
      }
    }
  }
}

__device__ __forceinline__ float nvfp4_rcp(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}

__device__ __forceinline__ float native_bf16_round(float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}

// Actual FI CUTLASS NVFP4 TMA path: GEMM1 writes BF16, doActivationKernel
// reloads it into ComputeElem=float, SwigluBiasAdaptor computes FLOAT
// gate*Sigmoid(gate)*up, then quantizePackedFPXValue casts the product to BF16
// before computing its group16 scale/codes. No intermediate BF16 Sigmoid or
// exp/add/div/multiply rounding exists in that source contract.
// FI native module uses --use_fast_math: expf and 1/denominator are explicit
// fast intrinsics here, even when this independent candidate is built without
// global fast-math. Multiplication order matches SwigluBiasAdaptor/SiLu.
__device__ __forceinline__ float native_nvfp4_glu(float gate_acc, float up_acc,
                                                  float alpha, float limit) {
  const float gate =
      fminf(native_bf16_round(__fmul_rn(gate_acc, alpha)), limit);
  const float up =
      fminf(fmaxf(native_bf16_round(__fmul_rn(up_acc, alpha)), -limit), limit);
  const float sigmoid = __fdividef(1.f, 1.f + __expf(-gate));
  const float activated = __fmul_rn(__fmul_rn(gate, sigmoid), up);
  return native_bf16_round(activated);
}

__device__ __forceinline__ uint32_t pack_e2m1_8(const float* v) {
  uint32_t word;
  // PTX converts operand1 to HIGH nibble, operand2 to LOW nibble.
  asm volatile(
      "{ .reg .b8 b0,b1,b2,b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0,%2,%1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1,%4,%3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2,%6,%5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3,%8,%7;\n"
      "mov.b32 %0,{b0,b1,b2,b3}; }"
      : "=r"(word)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]),
        "f"(v[6]), "f"(v[7]));
  return word;
}

// Same NVFP4 scale convention as vLLM nvfp4_utils.cuh, cvt_warp_fp16_to_fp4:
// s=E4M3(a_gscale * max(abs(v))/6), q=E2M1(v/(s/a_gscale)).
// a_gscale is reciprocal calibration input_scale, NOT a weight/global alpha.
__device__ __forceinline__ uint8_t quant16(const float (&v)[16], float a_gscale,
                                           uint32_t (&q)[2]) {
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
  const uint8_t sb = __nv_cvt_float_to_fp8(a_gscale * (amax * nvfp4_rcp(6.f)),
                                           __NV_SATFINITE, __NV_E4M3);
  __nv_fp8_e4m3 sf;
  sf.__x = sb;
  const float s = float(sf);
  const float inv = s != 0.f ? nvfp4_rcp(s * nvfp4_rcp(a_gscale)) : 0.f;
  float scaled[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) scaled[i] = v[i] * inv;
  q[0] = pack_e2m1_8(scaled);
  q[1] = pack_e2m1_8(scaled + 8);
  return sb;
}

struct Map {  // per-expert token bookkeeping, built identically by every CTA
              // after the dispatch flags
  int cnt[kMaxWorld][kMaxEl];
  int ne[kMaxEl];
  int tiles_prefix[kMaxEl + 1];
};

// Top-k id of (token t, slot k) as every phase reads it: -1 for a masked route,
// an id outside [0, W * EL), or a repeat of an earlier slot of the same token,
// so a token reaches each expert at most once and its dispatch slot stays below
// TCAP
__device__ __forceinline__ int route_at(const Args& a, int t, int k) {
  const int64_t* ids = a.topk_ids + size_t(t) * a.TOPK;
  const int64_t id = ids[k];
  if (id < 0 || id >= int64_t(a.W) * a.EL) return -1;
  for (int j = 0; j < k; ++j)
    if (ids[j] == id) return -1;
  return int(id);
}

__device__ __forceinline__ void token_src(const Map& m, int el, int j, int W,
                                          int& w, int& i) {
  w = 0;
  i = j;
  while (w < W - 1 && i >= m.cnt[w][el]) {
    i -= m.cnt[w][el];
    ++w;
  }
}

// kTileN tokens per work item (a multiple of 8): the kTileN / 8 MMAs of a step
// share each weight fragment, so larger items cut the weight traffic when the
// experts hold many rows each, smaller ones keep more of the grid busy when
// they hold few
template <int kTileN>
__global__ void __launch_bounds__(kWarps * 32) sm120_mega_moe_kernel(Args a) {
  constexpr int NH = kTileN / kTile;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int H = a.H, I = a.I, EL = a.EL, W = a.W, TOPK = a.TOPK, TCAP = a.TCAP;
  __shared__ Map m;
  __shared__ int s_item, s_ok, s_epoch;
  // L1: [tok][kRowsPerItem/2 inter] / L2: [tok][kRowsPerItem hidden] /
  // dispatch: one NVFP4 token row (H/2 + H/16 bytes)
#if FLASHINFER_NVFP4_STAGE_FC2_INPUT
  // Opt-in GLM I2048 FC2 only. Inputs alias the later output scratch.
  // Padded row strides avoid bank conflicts in the eight-warp MMA reads.
  // FC1's math/layout/quantization and every inter-CTA/EP protocol stay native.
  static_assert(kRowsPerItem == 128, "audited FC2 scratch geometry only");
  __shared__ __align__(128) uint8_t s_fc2_storage[kTileN * 1184];
  auto s_tile = reinterpret_cast<float (*)[256]>(s_fc2_storage);
#else
  __shared__ __align__(
      16) float s_tile[kTileN][kRowsPerItem < 256 ? 256 : kRowsPerItem];
#endif

  unsigned long long st_[16] = {
      gtime()};  // +8 route-push 9 route-ok 10 map 11 pf 12 flag-sent 13 last
                 // FC1 end 14 first FC2 start 15 comb-flag release  // 0 start
                 // 1 pushed 2 flags 3 work end 4 comb flags 5 end 6 L1-wait ns
                 // 7 items
  if (threadIdx.x == 0)
    s_epoch = *reinterpret_cast<volatile int*>(a.ctr + kCtrEpoch) + 1;
  // the error word reports this call only (a wait records one no earlier than
  // kSpinNs from now)
  if (blockIdx.x == 0 && threadIdx.x == 0) a.ctr[kCtrErr] = 0;
  __syncthreads();
  const int epoch = s_epoch;
  int* ctr = a.ctr;

  // ---------------- routing exchange first (tiny): every rank learns the full
  // expert load before the rows land
  const int RS = 1 + TCAP * TOPK;
  const size_t RX = pool_row_bytes(H);
  if (blockIdx.x == (a.T < int(gridDim.x) ? a.T : int(gridDim.x) - 1)) {
    const int n = 1 + a.T * TOPK;
    for (int u = threadIdx.x; u < W * n; u += blockDim.x) {
      const int dst = u / n, v = u % n;
      if (is_gin(a, dst))
        continue;  // put below from this rank's own copy (dst == rank)
      a.peers.route[dst][a.rank * RS + v] =
          v == 0 ? a.T : route_at(a, (v - 1) / TOPK, (v - 1) % TOPK);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
      for (int dst = 0; dst < W; ++dst) {
#if MEGAMOE_SM120_GIN_DEVICE
        if (is_gin(a, dst)) {
          const size_t off =
              a.gin.off_route + size_t(a.rank) * RS * sizeof(int);
          gin_put(
              a, dst, off, off, size_t(1 + a.T * TOPK) * sizeof(int),
              ncclGin_WeakSignalAdd{ncclGinSignal_t(kSigRoute + a.rank), 1});
          continue;
        }
#endif
        st_release_sys(a.peers.route_flag[dst] + a.rank, epoch);
      }
    st_[8] = gtime();
  }

  // ---------------- t=0: this rank's own tokens already tell which local
  // experts are active; pull the head of their FC1 weights into L2 while the
  // rows and the other ranks' routing are in flight
  if (kPf0Bytes > 0 && EL <= 64) {
    __shared__ unsigned long long s_known;
    if (threadIdx.x == 0) s_known = 0;
    __syncthreads();
    for (int u = threadIdx.x; u < a.T * TOPK; u += blockDim.x) {
      const int e = route_at(a, u / TOPK, u % TOPK);
      if (e >= 0 && e / EL == a.rank) atomicOr(&s_known, 1ull << (e % EL));
    }
    __syncthreads();
    const unsigned long long known = s_known;
    const int nk = __popcll(known);
    if (nk > 0) {
      const size_t per = min(a.l1_stride, kPf0Bytes / nk) & ~size_t(127);
      const size_t lines = per / 128 * nk;
      for (size_t l = size_t(blockIdx.x) * blockDim.x + threadIdx.x; l < lines;
           l += size_t(gridDim.x) * blockDim.x) {
        const int ki = int(l / (per / 128));
        unsigned long long rest = known;
        for (int q = 0; q < ki; ++q) rest &= rest - 1;
        const int el = __ffsll(rest) - 1;
        const char* p = reinterpret_cast<const char*>(w13_of(a, el)) +
                        (l % (per / 128)) * 128;
        asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
      }
    }
  }

#if FLASHINFER_NVFP4_EXTENDED_PREFETCH
  // Same bounded t=0 prefetch as upstream, extended for GLM EP4's 72 experts.
  // Preserve the existing <=64 path and every data/EP synchronization contract.
  if (kPf0Bytes > 0 && EL > 64 && EL <= 128) {
    __shared__ unsigned long long s_known_wide[2];
    if (threadIdx.x < 2) s_known_wide[threadIdx.x] = 0;
    __syncthreads();
    for (int u = threadIdx.x; u < a.T * TOPK; u += blockDim.x) {
      const int e = route_at(a, u / TOPK, u % TOPK);
      if (e >= 0 && e / EL == a.rank) {
        const int el = e % EL;
        atomicOr(&s_known_wide[el / 64], 1ull << (el % 64));
      }
    }
    __syncthreads();
    const unsigned long long known0 = s_known_wide[0], known1 = s_known_wide[1];
    const int nk0 = __popcll(known0), nk = nk0 + __popcll(known1);
    if (nk > 0) {
      const size_t per = min(a.l1_stride, kPf0Bytes / nk) & ~size_t(127);
      const size_t lines = per / 128 * nk;
      for (size_t l = size_t(blockIdx.x) * blockDim.x + threadIdx.x; l < lines;
           l += size_t(gridDim.x) * blockDim.x) {
        const int ki = int(l / (per / 128));
        const bool upper = ki >= nk0;
        unsigned long long rest = upper ? known1 : known0;
        for (int q = 0; q < (upper ? ki - nk0 : ki); ++q) rest &= rest - 1;
        const int el = (upper ? 64 : 0) + __ffsll(rest) - 1;
        const char* ptr = reinterpret_cast<const char*>(w13_of(a, el)) +
                          (l % (per / 128)) * 128;
        asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
      }
    }
  }
#endif

  // ---------------- dispatch: one CTA per token: the staged NVFP4 row into
  // smem, push TOPK copies with every thread
  const int nitems = a.T * TOPK;
  {
    uint8_t* s_q = reinterpret_cast<uint8_t*>(&s_tile[0][0]);
    uint8_t* s_sc = s_q + H / 2;
    __shared__ int s_rowk[8], s_dstk[8];
    for (int t = blockIdx.x; t < a.T; t += gridDim.x) {
      if (threadIdx.x < H / 16) {
        reinterpret_cast<uint2*>(s_q)[threadIdx.x] =
            reinterpret_cast<const uint2*>(a.x +
                                           size_t(t) * (H / 2))[threadIdx.x];
        s_sc[threadIdx.x] = a.x_sf[size_t(t) * (H / 16) + threadIdx.x];
      }
      if (warp < TOPK) {  // warp k: destination row of copy k (slot = tokens t'
                          // < t routed to the same expert)
        const int e =
            route_at(a, t, warp);  // < 0: masked route, nothing to send
        int slot = 0;
        for (int u = lane; u < t * TOPK; u += 32)
          slot += (route_at(a, u / TOPK, u % TOPK) == e);
#pragma unroll
        for (int o = 16; o; o >>= 1)
          slot += __shfl_xor_sync(0xffffffffu, slot, o);
#if MEGAMOE_SM120_GIN_DEVICE
        if (e >= 0 &&
            is_gin(
                a,
                e / EL)) {  // slot table entry: packed (el, slot) -> this token
          int pre = 0;      // this rank's routes to lower experts of that rank
          for (int u = lane; u < a.T * TOPK; u += 32) {
            const int e2 = route_at(a, u / TOPK, u % TOPK);
            pre += (e2 >= 0 && e2 / EL == e / EL && e2 % EL < e % EL);
          }
  #pragma unroll
          for (int o = 16; o; o >>= 1)
            pre += __shfl_xor_sync(0xffffffffu, pre, o);
          if (lane == 0) {  // the same packed index places this copy's partial
                            // on the way back
            a.gin.stage_slot[(e / EL) * TCAP * TOPK + pre + slot] = t;
            a.gin.cidx[t * TOPK + warp] = pre + slot;
          }
        }
#endif
        if (lane == 0) {
          if (e < 0) {
            s_rowk[warp] = -1;
          } else {
            const int dst = e / EL, el = e % EL;
            const int row = (el * W + a.rank) * TCAP + slot;
            s_rowk[warp] = row;
            s_dstk[warp] = dst;
          }
        }
      }
      __syncthreads();
      // per copy: H / 16 uint4 of data, the H / 32 scale bytes as H / 256 uint2
      // (a scale row is only 8-byte aligned when hidden % 512 != 0), then the
      // int metadata. Copy TOPK is the token's GIN staging row (written when
      // some copy goes to a GIN rank, no metadata): GIN ranks get the token
      // once, into their inbox, whatever number of their experts it routes to.
      const int per = H / 32 + H / 128 + 1;
      bool any_gin = false;
      for (int k = 0; k < TOPK; ++k)
        any_gin |= s_rowk[k] >= 0 && is_gin(a, s_dstk[k]);
      for (int u = threadIdx.x; u < (TOPK + 1) * per; u += blockDim.x) {
        const int k = u / per, q = u % per;
        uint8_t* dst;
        if (k == TOPK) {
          if (!any_gin || q == per - 1) continue;
          dst = a.gin.stage_row + size_t(t) * RX;
        } else {
          if (s_rowk[k] < 0 || is_gin(a, s_dstk[k])) continue;
          dst = a.peers.pool[s_dstk[k]] + size_t(s_rowk[k]) * RX;
        }
        if (q < H / 32)
          reinterpret_cast<uint4*>(dst)[q] =
              reinterpret_cast<const uint4*>(s_q)[q];
        else if (q < H / 32 + H / 128)
          reinterpret_cast<uint2*>(dst + H / 2)[q - H / 32] =
              reinterpret_cast<const uint2*>(s_sc)[q - H / 32];
        else
          *reinterpret_cast<int*>(dst + H / 2 + H / 16) = t * 8 + k;
      }
      __syncthreads();
#if MEGAMOE_SM120_GIN_DEVICE
      if (any_gin && lane == 0 && warp < TOPK && s_rowk[warp] >= 0 &&
          is_gin(a, s_dstk[warp])) {
        bool first =
            true;  // one put per GIN rank: the token's first copy to it sends
        for (int k = 0; k < warp; ++k)
          first &= !(s_rowk[k] >= 0 && s_dstk[k] == s_dstk[warp]);
        if (first)
          gin_put(
              a, s_dstk[warp],
              a.gin.off_inbox + (size_t(a.rank) * TCAP + t) * RX,
              a.gin.off_stage_row + size_t(t) * RX, size_t(H / 2 + H / 16),
              ncclGin_WeakSignalAdd{ncclGinSignal_t(kSigDispatch + a.rank), 1});
      }
      __syncthreads();
#endif
    }
  }
  __threadfence_system();
  __syncthreads();
  st_[1] = gtime();
  if (threadIdx.x == 0)
    s_item = (atomicAdd(ctr + kCtrDispatch, 1) == int(gridDim.x) - 1);
  __syncthreads();
#if MEGAMOE_SM120_GIN_DEVICE
  if (s_item && a.gin.mask) {  // last CTA out of the dispatch: every slot table
                               // entry is written
    __shared__ int s_nd[kMaxWorld];  // this rank's routes per GIN rank = its
                                     // slot table length
    if (threadIdx.x < W) s_nd[threadIdx.x] = 0;
    __threadfence();
    __syncthreads();
    for (int u = threadIdx.x; u < a.T * TOPK; u += blockDim.x) {
      const int e = route_at(a, u / TOPK, u % TOPK);
      if (e >= 0) atomicAdd(&s_nd[e / EL], 1);
    }
    __syncthreads();
    if (threadIdx.x < W && is_gin(a, threadIdx.x)) {
      const int d = threadIdx.x;
      gin_put(a, d,
              a.gin.off_slotmap + size_t(a.rank) * TCAP * TOPK * sizeof(int),
              a.gin.off_stage_slot + size_t(d) * TCAP * TOPK * sizeof(int),
              size_t(s_nd[d] > 0 ? s_nd[d] : 1) * sizeof(int),
              ncclGin_WeakSignalAdd{ncclGinSignal_t(kSigDispatch + a.rank), 1});
    }
    __syncthreads();
  }
#endif
  if (s_item && threadIdx.x == 0) {  // last CTA out of the dispatch: every row
                                     // of this rank is visible -> flag
    __threadfence_system();
    for (int dst = 0; dst < W; ++dst)
      if (!is_gin(a, dst))
        st_release_sys(a.peers.recv_flag[dst] + a.rank, epoch);
    st_[12] = gtime();
  }
  __syncthreads();

  // ---------------- per-expert map from the exchanged routing (same slot order
  // as the dispatch)
  if (threadIdx.x == 0) {
    s_ok = 1;
    for (int w = 0; w < W; ++w) {
#if MEGAMOE_SM120_GIN_DEVICE
      if (is_gin(a, w)) {  // one routing put per call: the signal counts calls,
                           // like the epoch
        if (!gin_wait(a, kSigRoute + w, (unsigned long long)epoch)) s_ok = 0;
        continue;
      }
#endif
      if (!wait_ge(a.peers.route_flag[a.rank] + w, epoch, true)) s_ok = 0;
    }
    if (!s_ok) atomicExch(ctr + kCtrErr, kErrRouteWait);
  }
  __syncthreads();
  st_[9] = gtime();
  if (!s_ok) goto exit_kernel;
#if MEGAMOE_SM120_GIN_DEVICE
  __shared__ int s_dn[kMaxWorld];  // GIN source w: tokens it sends here (one
                                   // inbox row each)
#endif
  {  // every source's routing in one parallel pass (all loads in flight
     // together), then a warp scan for the prefix
    __shared__ int s_T[kMaxWorld];
    const int* rb = a.peers.route[a.rank];
    for (int u = threadIdx.x; u < W * EL; u += blockDim.x)
      m.cnt[u / EL][u % EL] = 0;
    if (threadIdx.x < W) s_T[threadIdx.x] = __ldcv(rb + threadIdx.x * RS);
    __syncthreads();
    constexpr int kPer = 8;  // entries per thread per pass
    for (int base = 0; base < W * RS; base += kPer * int(blockDim.x)) {
      int e[kPer];
#pragma unroll
      for (int q = 0; q < kPer; ++q) {
        const int u = base + q * blockDim.x + threadIdx.x;
        const int w = u / RS, v = u % RS;
        e[q] = (u < W * RS && v > 0 && v - 1 < s_T[w] * TOPK)
                   ? __ldcv(rb + u)
                   : -1;  // -1 also = padding id
      }
#pragma unroll
      for (int q = 0; q < kPer; ++q) {
        const int u = base + q * blockDim.x + threadIdx.x;
        if (e[q] >= 0 && e[q] / EL == a.rank)
          atomicAdd(&m.cnt[u / RS][e[q] % EL], 1);  // e < 0: padding
      }
    }
#if MEGAMOE_SM120_GIN_DEVICE
    if (a.gin.mask) {
      if (threadIdx.x < W) s_dn[threadIdx.x] = 0;
      __syncthreads();
      for (int u = threadIdx.x; u < W * TCAP; u += blockDim.x) {
        const int w = u / TCAP, t = u % TCAP;
        if (!is_gin(a, w) || t >= s_T[w]) continue;
        bool here = false;
        for (int k = 0; k < TOPK; ++k) {
          const int id = __ldcv(rb + w * RS + 1 + t * TOPK + k);
          here |= id >= 0 && id / EL == a.rank;
        }
        if (here) atomicAdd(&s_dn[w], 1);
      }
    }
#endif
  }
  __syncthreads();
  if (warp == 0) {
    int tiles_sum = 0;
    for (int base = 0; base < EL; base += 32) {
      const int el = base + lane;
      int n = 0;
      if (el < EL)
        for (int w = 0; w < W; ++w) n += m.cnt[w][el];
      int incl = (n + kTileN - 1) / kTileN;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += v;
      }
      if (el < EL) {
        m.ne[el] = n;
        m.tiles_prefix[el + 1] = tiles_sum + incl;
      }
      tiles_sum += __shfl_sync(0xffffffffu, incl, 31);
    }
    if (lane == 0) m.tiles_prefix[0] = 0;
  }
  __syncthreads();
  if (a.stats && blockIdx.x == 0)
    for (int el = threadIdx.x; el < EL; el += blockDim.x)
      a.stats[el] += m.ne[el];
  st_[10] = gtime();
  {  // this CTA's first FC1 item is static (item = blockIdx.x): start pulling
     // its weights into L2 now
    const int nblk1 = 2 * I / kRowsPerItem;
    const int item = blockIdx.x;
    if (item < m.tiles_prefix[EL] * nblk1) {
      const int tile_g = item / nblk1, rb = item % nblk1;
      int el = 0;
      while (m.tiles_prefix[el + 1] <= tile_g) ++el;
      const int nb0 = (rb * kRowsPerItem) / 16 + warp * kNbPerWarp;
      const int blk_bytes = (H / 64) * 32 * 16;
#pragma unroll
      for (int jb = 0; jb < kNbPerWarp; ++jb) {
        for (int off = lane * 128; off < min(kPfBytes, blk_bytes);
             off += 32 * 128) {
          const int logical_row = (nb0 + jb) * 16 + off / (H / 2);
          const int physical_row =
              nvfp4_weight_row(logical_row, 2 * I, a.w13_layout);
          const char* p = reinterpret_cast<const char*>(w13_of(a, el)) +
                          size_t(physical_row) * (H / 2) + off % (H / 2);
          asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
        }
      }
    }
  }
  st_[11] = gtime();

  // ---------------- wait for every source rank, build the per-expert map
  if (threadIdx.x == 0) {
    s_ok = 1;
    for (int w = 0; w < W; ++w) {
#if MEGAMOE_SM120_GIN_DEVICE
      if (is_gin(a, w)) {  // one signal per token that source w sends here,
                           // plus its slot table
        if (blockIdx.x == 0) ctr[kCtrCnt + w] = s_dn[w] + 1;
        if (!gin_wait(a, kSigDispatch + w, gin_cum(ctr)[w] + s_dn[w] + 1))
          s_ok = 0;
        continue;
      }
#endif
      if (!wait_ge(a.peers.recv_flag[a.rank] + w, epoch, true)) s_ok = 0;
    }
    if (!s_ok) atomicExch(ctr + kCtrErr, kErrDispatchWait);
  }
  __syncthreads();
  st_[2] = gtime();
  if (!s_ok) goto exit_kernel;
  {
    const int nblk1 = 2 * I / kRowsPerItem, nblk2 = H / kRowsPerItem;
    const int ntiles = m.tiles_prefix[EL];
    const int n1 = ntiles * nblk1, n2 = ntiles * nblk2;
    const uint8_t* pool = a.peers.pool[a.rank];
    const int g = lane / 4, c = lane % 4;
    const int TOKE = W * TCAP;

    bool first = true;
    while (true) {
      if (threadIdx.x == 0)
        s_item = first ? int(blockIdx.x)
                       : int(gridDim.x) + atomicAdd(ctr + kCtrQueue, 1);
      first = false;
      __syncthreads();
      const int item = s_item;
      if (item >= n1 + n2) break;
      st_[7] += 1;
      const bool l1 = item < n1;
      const int li = l1 ? item : item - n1;
      const int nblk = l1 ? nblk1 : nblk2;
      const int tile_g = li / nblk, rb = li % nblk;
      int el = 0;
      while (m.tiles_prefix[el + 1] <= tile_g) ++el;
      const int tile = tile_g - m.tiles_prefix[el];
      const int ntok = min(kTileN, m.ne[el] - tile * kTileN);
      int jt[NH];  // this lane's tokens (B column g of each group of 8; a
                   // padding lane repeats one)
#pragma unroll
      for (int h = 0; h < NH; ++h)
        jt[h] = tile * kTileN + (h * 8 + g < ntok ? h * 8 + g : 0);
#if MEGAMOE_SM120_GIN_DEVICE
      __shared__ int s_pw[kMaxWorld];  // GIN source w: its slot table index of
                                       // (el, slot 0)
      if (a.gin.mask) {
        if (threadIdx.x < W && is_gin(a, threadIdx.x)) {
          int p = 0;
          for (int e2 = 0; e2 < el; ++e2) p += m.cnt[threadIdx.x][e2];
          s_pw[threadIdx.x] = p;
        }
        __syncthreads();
      }
#endif

      if (!l1) {  // FC2 needs every FC1 item of this expert
        const unsigned long long tw0 = gtime();
        if (threadIdx.x == 0) {
          const int need = ((m.ne[el] + kTileN - 1) / kTileN) * nblk1;
          if (!wait_ge(ctr + kCtrL1 + el, need, false)) {
            atomicExch(ctr + kCtrErr, kErrL1Wait);
            s_ok = 0;
          }
        }
        __syncthreads();
        st_[6] += gtime() - tw0;
        if (!st_[14]) st_[14] = gtime();
        if (!s_ok) goto exit_kernel;
      }

      float acc[kNbPerWarp][NH][4];
      const int nb0 = (rb * kRowsPerItem) / 16 + warp * kNbPerWarp;
      if (l1) {
        const uint8_t* xr[NH];
        const uint32_t* xs[NH];
#pragma unroll
        for (int h = 0; h < NH; ++h) {
          int w, i;
          token_src(m, el, jt[h], W, w, i);
          const size_t row = (size_t(el) * W + w) * TCAP + i;
          xr[h] = pool + row * RX;
#if MEGAMOE_SM120_GIN_DEVICE
          if (is_gin(a, w))
            xr[h] =
                a.gin.inbox + (size_t(w) * TCAP +
                               a.gin.slotmap[w * TCAP * TOPK + s_pw[w] + i]) *
                                  RX;
#endif
          xs[h] = reinterpret_cast<const uint32_t*>(xr[h] + H / 2);
        }
        item_gemm<NH>(acc, w13_of(a, el), ws13_of(a, el), 2 * I / 16, nb0, H,
                      xr, xs, lane, a.w13_layout, a.scale_layout);
        // SwiGLU: acc rows g (gate) / g+8 (up) of block nb -> inter col nb*8+g;
        // tokens 2c, 2c+1
        constexpr int kItemInter =
            kRowsPerItem / 2;  // inter cols per item (multiple of 32)
        constexpr int kGroups = kItemInter / 16;
        const float g1 = a.gs1 ? a.gs1[el] : 1.f;
#pragma unroll
        for (int jb = 0; jb < kNbPerWarp; ++jb) {
          const int col = (warp * kNbPerWarp + jb) * 8 +
                          g;  // 0..kItemInter-1 within the item
#pragma unroll
          for (int th = 0; th < NH; ++th)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              s_tile[th * 8 + 2 * c + h][col] = native_nvfp4_glu(
                  acc[jb][th][h], acc[jb][th][2 + h], g1, a.swiglu_limit);
            }
        }
        __syncthreads();
        if (threadIdx.x <
            kTileN * kGroups) {  // one thread per (token, NVFP4 group16)
          const int tk = threadIdx.x / kGroups, gi = threadIdx.x % kGroups;
          float v[16];
#pragma unroll
          for (int u = 0; u < 16; ++u) v[u] = s_tile[tk][gi * 16 + u];
          uint32_t q8[2];
          const uint8_t sc = quant16(v, a.a2_gscale[el], q8);
          if (tk < ntok) {
            const size_t r = size_t(el) * TOKE + tile * kTileN + tk;
            const int col0 = rb * kItemInter + gi * 16;
            *reinterpret_cast<uint2*>(a.inter + r * (I / 2) + col0 / 2) =
                make_uint2(q8[0], q8[1]);
            reinterpret_cast<uint8_t*>(a.inter_s + r * (I / 64))[col0 / 16] =
                sc;
          }
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(ctr + kCtrL1 + el, 1);
        st_[13] = gtime();
      } else {
        const uint8_t* xr[NH];
        const uint32_t* xs[NH];
#if FLASHINFER_NVFP4_STAGE_FC2_INPUT
        const int code_stride = I / 2 + 16, sf_stride = I / 16 + 16;
        uint8_t* code_stage = s_fc2_storage;
        uint8_t* sf_stage = s_fc2_storage + kTileN * code_stride;
        for (int v = threadIdx.x; v < kTileN * (I / 32); v += blockDim.x) {
          const int row = v / (I / 32), col = v % (I / 32);
          const size_t r =
              size_t(el) * TOKE + tile * kTileN + (row < ntok ? row : 0);
          *reinterpret_cast<uint4*>(code_stage + row * code_stride + col * 16) =
              *reinterpret_cast<const uint4*>(a.inter + r * (I / 2) + col * 16);
        }
        for (int v = threadIdx.x; v < kTileN * (I / 256); v += blockDim.x) {
          const int row = v / (I / 256), col = v % (I / 256);
          const size_t r =
              size_t(el) * TOKE + tile * kTileN + (row < ntok ? row : 0);
          const uint8_t* src = reinterpret_cast<const uint8_t*>(a.inter_s) +
                               r * (I / 16) + col * 16;
          *reinterpret_cast<uint4*>(sf_stage + row * sf_stride + col * 16) =
              *reinterpret_cast<const uint4*>(src);
        }
        __syncthreads();
#endif
#pragma unroll
        for (int h = 0; h < NH; ++h) {
#if FLASHINFER_NVFP4_STAGE_FC2_INPUT
          xr[h] = code_stage + (h * 8 + g) * code_stride;
          xs[h] = reinterpret_cast<const uint32_t*>(sf_stage +
                                                    (h * 8 + g) * sf_stride);
#else
          const size_t r = size_t(el) * TOKE + jt[h];
          xr[h] = a.inter + r * (I / 2);
          xs[h] = a.inter_s + r * (I / 64);
#endif
        }
        item_gemm<NH>(acc, w2_of(a, el), ws2_of(a, el), H / 16, nb0, I, xr, xs,
                      lane, -1, a.scale_layout);
#if FLASHINFER_NVFP4_STAGE_FC2_INPUT
        // No warp can overwrite the shared input until all MMAs consumed it.
        __syncthreads();
#endif
        const float g2 = a.gs2 ? a.gs2[el] : 1.f;
#pragma unroll
        for (int jb = 0; jb < kNbPerWarp; ++jb) {
          const int col = (warp * kNbPerWarp + jb) * 16 +
                          g;  // 0..kRowsPerItem-1 within the item
#pragma unroll
          for (int th = 0; th < NH; ++th)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              s_tile[th * 8 + 2 * c + h][col] = acc[jb][th][h] * g2;
              s_tile[th * 8 + 2 * c + h][col + 8] = acc[jb][th][2 + h] * g2;
            }
        }
        __syncthreads();
        // push: token tk of this tile -> its source rank's combine slot,
        // kRowsPerItem bf16 in 16 B stores
        constexpr int kVec = kRowsPerItem / 8;
        for (int u = threadIdx.x; u < ntok * kVec; u += blockDim.x) {
          const int tk = u / kVec, q = u % kVec;
          int w, i;
          token_src(m, el, tile * kTileN + tk, W, w, i);
          uint32_t pk[4];
#pragma unroll
          for (int p = 0; p < 4; ++p) {
            const __nv_bfloat162 b2 = __floats2bfloat162_rn(
                s_tile[tk][q * 8 + 2 * p], s_tile[tk][q * 8 + 2 * p + 1]);
            pk[p] = *reinterpret_cast<const uint32_t*>(&b2);
          }
          __nv_bfloat16* dst;
#if MEGAMOE_SM120_GIN_DEVICE
          if (is_gin(a, w))  // staged in the source's packed (el, slot) order
            dst =
                a.gin.stage_comb + (size_t(w) * TCAP * TOPK + s_pw[w] + i) * H;
          else
#endif
          {
            const int meta = *reinterpret_cast<const int*>(
                pool + ((size_t(el) * W + w) * TCAP + i) * RX + H / 2 + H / 16);
            dst = a.peers.comb[w] + (size_t(meta / 8) * TOPK + meta % 8) * H;
          }
          *reinterpret_cast<uint4*>(dst + rb * kRowsPerItem + q * 8) =
              make_uint4(pk[0], pk[1], pk[2], pk[3]);
        }
        __threadfence_system();
        __syncthreads();
#if MEGAMOE_SM120_GIN_DEVICE
        if (a.gin.mask) {  // the tile's last FC2 item puts its finished rows to
                           // GIN sources
          if (threadIdx.x == 0) {
            s_item = atomicAdd(ctr + kCtrTile + tile_g, 1) == nblk2 - 1;
            __threadfence();
          }
          __syncthreads();
          // the tile's tokens are in (source, slot) order: one put per run of a
          // GIN source
          __shared__ int s_tw[kTileN], s_ti[kTileN];
          if (s_item && threadIdx.x < ntok)
            token_src(m, el, tile * kTileN + threadIdx.x, W, s_tw[threadIdx.x],
                      s_ti[threadIdx.x]);
          __syncthreads();
          if (s_item && threadIdx.x < ntok && is_gin(a, s_tw[threadIdx.x]) &&
              (threadIdx.x == 0 ||
               s_tw[threadIdx.x - 1] != s_tw[threadIdx.x])) {
            const int w = s_tw[threadIdx.x];
            int n = 1;
            while (threadIdx.x + n < ntok && s_tw[threadIdx.x + n] == w) ++n;
            const size_t r = size_t(
                s_pw[w] + s_ti[threadIdx.x]);  // packed row, same both sides
            gin_put(
                a, w,
                a.gin.off_comb_gin + (size_t(a.rank) * TCAP * TOPK + r) * H * 2,
                a.gin.off_stage_comb + (size_t(w) * TCAP * TOPK + r) * H * 2,
                size_t(n) * H * 2,
                ncclGin_WeakSignalAdd{ncclGinSignal_t(kSigComb + a.rank),
                                      uint64_t(n)});
          }
          __syncthreads();
        }
#endif
        if (threadIdx.x == 0) {
          const int done = atomicAdd(ctr + kCtrL2Done, 1);
          if (done ==
              n2 - 1) {  // this rank's last FC2 item: tell every source rank
            __threadfence_system();
            for (int w = 0; w < W; ++w) {
#if MEGAMOE_SM120_GIN_DEVICE
              if (is_gin(a,
                         w)) {  // "done": the source counts it after the rows
                gin_signal(a, w,
                           ncclGin_WeakSignalAdd{
                               ncclGinSignal_t(kSigComb + a.rank), 1});
                continue;
              }
#endif
              st_release_sys(a.peers.comb_flag[w] + a.rank, epoch);
            }
            st_[15] = gtime();
          }
        }
      }
      __syncthreads();
    }
    st_[3] = gtime();
    if (n2 == 0 && blockIdx.x == 0 &&
        threadIdx.x == 0) {  // no local work: still release every source
      __threadfence_system();
      for (int w = 0; w < W; ++w) {
#if MEGAMOE_SM120_GIN_DEVICE
        if (is_gin(a, w)) {
          gin_signal(
              a, w,
              ncclGin_WeakSignalAdd{ncclGinSignal_t(kSigComb + a.rank), 1});
          continue;
        }
#endif
        st_release_sys(a.peers.comb_flag[w] + a.rank, epoch);
      }
    }
  }

  // ---------------- combine: wait for every expert rank, out[t] = sum_k
  // w[t][k] * partial[t][k]
#if MEGAMOE_SM120_GIN_DEVICE
  __shared__ int s_rows[kMaxWorld];  // this rank's routes per expert rank: GIN
                                     // rows to expect
  if (a.gin.mask) {
    if (threadIdx.x < W) s_rows[threadIdx.x] = 0;
    __syncthreads();
    for (int u = threadIdx.x; u < a.T * TOPK; u += blockDim.x) {
      const int e = route_at(a, u / TOPK, u % TOPK);
      if (e >= 0) atomicAdd(&s_rows[e / EL], 1);
    }
    __syncthreads();
  }
#endif
  if (threadIdx.x == 0) {
    s_ok = 1;
    for (int w = 0; w < W; ++w) {
#if MEGAMOE_SM120_GIN_DEVICE
      if (is_gin(a, w)) {  // one signal per pushed-back row, plus the expert
                           // rank's "done"
        const int n = s_rows[w] + 1;
        if (blockIdx.x == 0) ctr[kCtrCnt + kMaxWorld + w] = n;
        if (!gin_wait(a, kSigComb + w, gin_cum(ctr)[kMaxWorld + w] + n))
          s_ok = 0;
        continue;
      }
#endif
      if (!wait_ge(a.peers.comb_flag[a.rank] + w, epoch, true)) s_ok = 0;
    }
#if MEGAMOE_SM120_GIN_DEVICE
    // the packed rows (cidx) were written by every CTA's dispatch
    if (a.gin.mask && !wait_ge(ctr + kCtrDispatch, int(gridDim.x), false))
      s_ok = 0;
#endif
    if (!s_ok) atomicExch(ctr + kCtrErr, kErrCombineWait);
  }
  __syncthreads();
  st_[4] = gtime();
  if (s_ok) {
    const __nv_bfloat16* comb = a.peers.comb[a.rank];
    const int nvec = a.T * H / 8;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < nvec;
         v += gridDim.x * blockDim.x) {
      const int t = v / (H / 8), h8 = (v % (H / 8)) * 8;
      float s[8] = {};
      for (int k = 0; k < TOPK; ++k) {
        const int e = route_at(a, t, k);
        if (e < 0) continue;  // masked route: never dispatched
        const float wk = a.topk_w[t * TOPK + k];
        const __nv_bfloat16* row = comb + (size_t(t) * TOPK + k) * H;
#if MEGAMOE_SM120_GIN_DEVICE
        if (is_gin(a, e / EL))
          row = a.gin.comb_gin +
                (size_t(e / EL) * TCAP * TOPK + a.gin.cidx[t * TOPK + k]) * H;
#endif
        const uint4 u = *reinterpret_cast<const uint4*>(row + h8);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const float2 f = __bfloat1622float2(b2[r]);
          s[2 * r] += wk * f.x;
          s[2 * r + 1] += wk * f.y;
        }
      }
      uint32_t pk[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const __nv_bfloat162 b2 = __floats2bfloat162_rn(s[2 * r], s[2 * r + 1]);
        pk[r] = *reinterpret_cast<const uint32_t*>(&b2);
      }
      *reinterpret_cast<uint4*>(a.out + size_t(t) * H + h8) =
          make_uint4(pk[0], pk[1], pk[2], pk[3]);
    }
  }

exit_kernel:
  __syncthreads();
  st_[5] = gtime();
  if (a.stamps && threadIdx.x == 0)
    for (int u = 0; u < 16; ++u) a.stamps[blockIdx.x * 16 + u] = st_[u];
  if (threadIdx.x == 0) {
    __threadfence();
    const int done = atomicAdd(ctr + kCtrExit, 1);
    if (done == int(gridDim.x) - 1) {  // last CTA out: advance the epoch, reset
                                       // the per-call counters
      ctr[kCtrEpoch] = epoch;
      ctr[kCtrDispatch] = 0;
      ctr[kCtrQueue] = 0;
      ctr[kCtrL2Done] = 0;
      for (int el = 0; el < EL; ++el) ctr[kCtrL1 + el] = 0;
      if (a.gin
              .mask) {  // advance the expected GIN totals by this call's counts
        for (int w = 0; w < W; ++w)
          for (int p = 0; p < 2; ++p) {
            gin_cum(ctr)[p * kMaxWorld + w] += ctr[kCtrCnt + p * kMaxWorld + w];
            ctr[kCtrCnt + p * kMaxWorld + w] = 0;
          }
        for (int i = 0; i < kMaxTiles; ++i) ctr[kCtrTile + i] = 0;
      }
      __threadfence();
      ctr[kCtrExit] = 0;
    }
  }
}

}  // namespace flashinfer::megamoe::sm120_nvfp4
