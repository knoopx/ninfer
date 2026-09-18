#pragma once

// Ternary PQ2_0 rotation: the signed normalized Sylvester Walsh-Hadamard transform (FWT) at
// block size 1024, shared by every TERNARY_PQ2_0 route (spec: docs/maintainer/ternary-pq2-0.md).
//
//   R(x) = (1/sqrt(1024)) * H_1024 * (x .* s)
//
// `s` is the per-input-width sign vector (every element exactly +/-1.0) stored in the parent's
// rotation auxiliary: a 16-byte header followed by K binary32 signs (src/core/weight.h). The
// inverse route (embed_tokens) is the same kernel with the same sign vector, because
// R^-1 = (1/sqrt(1024)) * H_1024^T * s and H_1024 is symmetric (spec section 3).
//
// Butterfly convention (pinned, fork reference ggml-cuda/fwht.cu): within a pair (j, j^h) with
// j < j^h, the low element takes x + y and the high element takes x - y.
//
// Two device routes (spec section 6.1):
//
//   1. Shared-memory butterfly (decode / small-T route). One block per transform row, 256
//      threads, N/256 = 4 elements per thread. Stages run in three segments: within-warp stages
//      (h < 32) with __shfl_xor, across-warp stages (32 <= h < 256) with a shared-memory round
//      trip, and above-block stages (h >= 256) in registers. The sign flip and the 1/sqrt(1024)
//      normalization are fused into the load; there is no separate memory pass. The fused
//      gemv/simt GEMM kernels inline ternary_fwt_butterfly() on the A-tile load; the standalone
//      kernel below is the materializing form used where the transform is a separate step.
//
//   2. MMA 16x16-chunk route (large-T / prefill). HadaCore-style tensor-core FWT: the
//      within-window stages (h = 1..8, coupling features within each 16-wide feature window)
//      are dense 16x16 +/-1 coefficient matrices applied with mma.sync.m16n8k16 (bf16
//      operands, FP32 accumulation); the above-window stages (h = 16..512) are +/-1
//      butterflies between 16x16 window planes, staged through shared memory.
//
// Pinned chunk-rounding semantics of the MMA route:
//   - the sign flip and the 1/32 normalization are applied elementwise to the BF16 input
//     (RN) before the first MMA stage;
//   - the four within-window stages accumulate in FP32 within a 16x16 chunk (eight
//     mma.sync.m16n8k16 instructions per chunk);
//   - each 16x16 chunk is converted to BF16 with RN exactly once, at the chunk boundary,
//     before any above-window stage reads it;
//   - every above-window butterfly performs a BF16 add/subtract and stores the RN result.
// The FP64 oracle of the spec qualifies the exact butterfly against this pinned rule.

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kTernaryFwtBlock = 1024;
// 1/sqrt(1024) = 1/32, exact in binary.
inline constexpr float kTernaryFwtScale = 0.03125F;

// In-place 1024-wide Sylvester Walsh-Hadamard butterfly over a shared-memory row.
//
// `row` must be addressed by a 256-thread block as row[i * 256 + threadIdx.x]. The sign vector
// and the 1/sqrt(1024) normalization are applied by the caller on the load; this function is
// the pure transform.
__device__ __forceinline__ void ternary_fwt_butterfly(float* __restrict__ row) {
    constexpr int kThreads           = 256;
    constexpr int kElementsPerThread = kTernaryFwtBlock / kThreads;
    const int tid                    = static_cast<int>(threadIdx.x);
    const int lane                   = tid & 31;
    float reg[kElementsPerThread];
#pragma unroll
    for (int i = 0; i < kElementsPerThread; ++i) {
        reg[i] = row[i * kThreads + tid];
    }

    // Within-warp stages: the partner differs in the lane bits.
#pragma unroll
    for (int h = 1; h < 32; h *= 2) {
#pragma unroll
        for (int j = 0; j < kElementsPerThread; ++j) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xffffffffu, val, h, 32);
            reg[j]           = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Across-warp stages: the partner is another thread of the block.
#pragma unroll
    for (int h = 32; h < kThreads; h *= 2) {
#pragma unroll
        for (int j = 0; j < kElementsPerThread; ++j) {
            row[j * kThreads + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < kElementsPerThread; ++j) {
            const float val  = reg[j];
            const float val2 = row[j * kThreads + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // Above-block stages: the partner is another register of the same thread.
#pragma unroll
    for (int h = kThreads; h < kTernaryFwtBlock; h *= 2) {
        const int step = h / kThreads;
#pragma unroll
        for (int j = 0; j < kElementsPerThread; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]         = x + y;
                reg[j + k + step]  = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < kElementsPerThread; ++i) {
        row[i * kThreads + tid] = reg[i];
    }
}

// Materializing standalone FWT (decode / small-T route): one block per (token, 1024 block).
// The sign flip and the 1/sqrt(1024) scale are fused into the load; the transformed row is
// written back as BF16 (the pinned transform output type of every route).
template <std::int32_t InputRows>
__global__ __launch_bounds__(256) void ternary_fwt_butterfly_kernel(
    const __nv_bfloat16* __restrict__ x, const float* __restrict__ signs,
    __nv_bfloat16* __restrict__ out, std::int32_t tokens) {
    static_assert((InputRows % kTernaryFwtBlock) == 0);
    (void)tokens;
    __shared__ float row[kTernaryFwtBlock];

    const int token  = static_cast<int>(blockIdx.x);
    const int block  = static_cast<int>(blockIdx.y);
    const int k0     = block * kTernaryFwtBlock;
    const int tid    = static_cast<int>(threadIdx.x);
    const auto* xr   = reinterpret_cast<const std::uint16_t*>(
        static_cast<const __nv_bfloat16*>(x) + static_cast<std::int64_t>(token) * InputRows + k0);
    const float* sign_block = reinterpret_cast<const float*>(signs) + k0;

#pragma unroll
    for (int i = 0; i < kTernaryFwtBlock / 256; ++i) {
        const int idx = i * 256 + tid;
        // Fused sign flip + 1/sqrt(1024) into the load.
        row[idx] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(xr + idx)) *
                   sign_block[idx] * kTernaryFwtScale;
    }
    __syncthreads();
    ternary_fwt_butterfly(row);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kTernaryFwtBlock / 256; ++i) {
        const int idx = i * 256 + tid;
        auto* out_row =
            reinterpret_cast<std::uint16_t*>(
                static_cast<__nv_bfloat16*>(out) + static_cast<std::int64_t>(token) * InputRows +
                k0 + idx);
        *out_row = __bfloat16_as_ushort(__float2bfloat16_rn(row[idx]));
    }
}

namespace ternary_fwt_mma {

// One CTA owns a 16-token x 1024-feature slab. Each warp owns 8 of the 64 16x16 feature chunks.

constexpr int kSlabTokens      = 16;
constexpr int kThreads         = 256;
constexpr int kWindows         = kTernaryFwtBlock / 16;  // 64 feature windows per token row.
constexpr int kChunksPerWarp   = kWindows / (kThreads / 32);
constexpr int kSlabElements    = kSlabTokens * kTernaryFwtBlock;
constexpr int kSlabBf16Bytes   = kSlabElements * static_cast<int>(sizeof(std::uint16_t));

// Within-window 16x16 Sylvester H_16 coefficient: H_16[k][n] = +/-1 = (-1)^popcount(k & n)
// (natural-order Walsh; H_16 is symmetric, so = H_16[n][k]). The four within-window butterflies
// (h = 1,2,4,8) compose to this single 16x16 transform, so the mma applies it in one pass.
__device__ __forceinline__ float had16_coeff(int k, int n) {
    return (__popc(k & n) & 1) ? -1.0F : 1.0F;
}

// mma m16n8k16 B fragment (two bf16x2 registers) for the within-window H_16, n-tile 0/1.
// Lane l holds B[k = 2*(l&3) + {0,1}, n] and B[k = 8 + 2*(l&3) + {0,1}, n] with n = tile*8 + l/4,
// where B[k][n] = H_16[n][k].
__device__ __forceinline__ void within_window_b_fragments(int n_tile, int lane, unsigned& b0,
                                                          unsigned& b1) {
    const int n    = n_tile * 8 + (lane >> 2);
    const int k_lo = (lane & 3) * 2;
    const int k_hi = 8 + k_lo;
    const __nv_bfloat162 r0 =
        __floats2bfloat162_rn(had16_coeff(k_lo, n), had16_coeff(k_lo + 1, n));
    const __nv_bfloat162 r1 =
        __floats2bfloat162_rn(had16_coeff(k_hi, n + 8), had16_coeff(k_hi + 1, n + 8));
    b0 = load_vec<unsigned>(&r0);
    b1 = load_vec<unsigned>(&r1);
}

// The full slab transform. `slab` is a 32KB shared BF16 plane (16 tokens x 1024 features,
// row-major). The sign vector covers one 1024 block; `sign_base` indexes into it.
//
// Stages:
//   (a) cp.async loads the 16 x 1024 BF16 slab from global memory;
//   (b) each warp runs its 8 chunks through the four within-window stages (h = 1..8) with
//       mma.sync.m16n8k16, FP32 accumulation, one RN-BF16 conversion at the chunk boundary;
//   (c) six above-window stages (h = 16..512) butterfly the 16x16 chunks in shared memory,
//       one RN-BF16 per butterfly;
//   (d) the transformed slab is written back as BF16.
template <std::int32_t InputRows>
__device__ __forceinline__ void ternary_fwt_mma_slab(__nv_bfloat16* slab,
                                                     const float* __restrict__ signs, int k_block) {
    static_assert((InputRows % kTernaryFwtBlock) == 0);
    constexpr int kChunksTotal = kSlabTokens * kWindows;  // 1024 chunks of 16x16
    constexpr int kCtasPerWarpChunk = kChunksTotal / (kThreads / 32) / 8;  // 4 slabs per pass
    (void)kCtasPerWarpChunk;

    // (a) Stage the slab.
    constexpr int kSixteenByteLoads = kSlabBf16Bytes / 16;
    {
        constexpr int kLoadsPerThread = kSixteenByteLoads / kThreads;
        const int tid                  = static_cast<int>(threadIdx.x);
        const auto* source             = reinterpret_cast<const std::uint16_t*>(
            static_cast<const __nv_bfloat16*>(slab));
        (void)source;
#pragma unroll
        for (int i = 0; i < kLoadsPerThread; ++i) {
            const int task = i * kThreads + tid;
            // The caller filled the slab via cp.async; this branch documents the shape of the
            // staging pass.
            (void)task;
        }
    }
    __syncthreads();

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const float* sign_block = signs + k_block * kTernaryFwtBlock;

    // (b) Within-window stages, per 16x16 chunk. The sign flip and 1/32 scale are fused into
    // the operand load (RN per element, pinned).
#pragma unroll 2
    for (int chunk_slot = 0; chunk_slot < kChunksPerWarp; chunk_slot += 2) {
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int chunk = warp * kChunksPerWarp + chunk_slot + s;
            const int token = chunk / kWindows;
            const int window = chunk - token * kWindows;
            const int feature0 = window * 16;
            const auto* chunk_smem =
                reinterpret_cast<std::uint16_t*>(slab) + token * (kTernaryFwtBlock * 2) +
                feature0 * 2;

            // A fragment: 16 tokens x 16 features, with the fused sign + scale on the load.
            unsigned a0, a1, a2, a3;
            {
                // A fragment layout of mma m16n8k16: lane l holds A[m = l/4, k = 2*(l&3)+{0,1}]
                // (a0), the m+8 row (a1), and the k + 8 pairs (a2, a3).
                const int m0 = lane >> 2;
                const int m1 = m0 + 8;
                const int k0 = (lane & 3) * 2;
                auto value_at = [&](int m, int k) {
                    const float v = __bfloat162float(
                        *reinterpret_cast<const __nv_bfloat16*>(chunk_smem + m * (kTernaryFwtBlock * 2) + k * 2));
                    // Fused sign flip + 1/sqrt(1024) into the load (pinned: RN at the operand).
                    return __bfloat162float(__float2bfloat16_rn(v * sign_block[feature0 + k] * kTernaryFwtScale));
                };
                const __nv_bfloat162 p0 = __floats2bfloat162_rn(value_at(m0, k0), value_at(m0, k0 + 1));
                const __nv_bfloat162 p1 = __floats2bfloat162_rn(value_at(m1, k0), value_at(m1, k0 + 1));
                const __nv_bfloat162 p2 = __floats2bfloat162_rn(value_at(m0, k0 + 8), value_at(m0, k0 + 9));
                const __nv_bfloat162 p3 = __floats2bfloat162_rn(value_at(m1, k0 + 8), value_at(m1, k0 + 9));
                a0 = load_vec<unsigned>(&p0);
                a1 = load_vec<unsigned>(&p1);
                a2 = load_vec<unsigned>(&p2);
                a3 = load_vec<unsigned>(&p3);
            }

            // Within-window 16x16 Sylvester H_16 applied in one pass (two 8-wide n-tiles):
            // c = A * H_16, with the fused sign + 1/32 on the operand load.
            float c[2][4] = {};
#pragma unroll
            for (int n_tile = 0; n_tile < 2; ++n_tile) {
                unsigned b0, b1;
                within_window_b_fragments(n_tile, lane, b0, b1);
                mma_bf16(c[n_tile][0], c[n_tile][1], c[n_tile][2], c[n_tile][3], a0, a1, a2, a3,
                         b0, b1);
            }

            // Pinned chunk boundary: one RN-BF16 conversion of the 16x16 chunk.
            auto* chunk_out = reinterpret_cast<std::uint16_t*>(slab) +
                              token * (kTernaryFwtBlock * 2) + feature0 * 2;
            // C-fragment of mma m16n8k16: lane l holds (m,n),(m,n+1),(m+8,n),(m+8,n+1) with
            // m = l/4 (row) and n = tile*8 + (l&3)*2 (column within the 8-wide n-tile).
            const int m = lane >> 2;
#pragma unroll
            for (int n_tile = 0; n_tile < 2; ++n_tile) {
                const int col = n_tile * 8 + (lane & 3) * 2;
                const __nv_bfloat162 q0 =
                    __floats2bfloat162_rn(c[n_tile][0], c[n_tile][1]);  // row m, cols col..col+1
                const __nv_bfloat162 q1 =
                    __floats2bfloat162_rn(c[n_tile][2], c[n_tile][3]);  // row m+8, cols col..col+1
                *reinterpret_cast<unsigned*>(chunk_out + m * (kTernaryFwtBlock * 2) + col * 2) =
                    load_vec<unsigned>(&q0);
                *reinterpret_cast<unsigned*>(
                    chunk_out + (m + 8) * (kTernaryFwtBlock * 2) + col * 2) =
                    load_vec<unsigned>(&q1);
            }
        }
        __syncthreads();
    }

    // (c) Above-window stages: +/-1 butterflies between 16x16 window planes, one RN-BF16 each.
    // For h = 16..512 the partner of window w is w ^ (h/16); every (token, within-window feature)
    // couples its window with the partner window. Full coverage: thread tid owns slab BF16
    // elements tid, tid+256, ... (kSlabElements/kThreads = 64 per thread, 16384 total); each
    // pair is written exactly once, from its lower-index window (window < partner).
    const int tid_c = static_cast<int>(threadIdx.x);
    constexpr int kPairsPerThread = kSlabElements / kThreads;
#pragma unroll
    for (int h = 16; h <= 512; h *= 2) {
        const int partner_bit = h >> 4;
#pragma unroll
        for (int j = 0; j < kPairsPerThread; ++j) {
            const int g = j * kThreads + tid_c;            // global BF16 index 0..16383
            const int feature = g % kTernaryFwtBlock;
            const int window  = feature / 16;
            const int within  = feature % 16;
            const int partner = window ^ partner_bit;
            if (window >= partner) { continue; }          // the lower-index window writes the pair
            const int pg = (g / kTernaryFwtBlock) * kTernaryFwtBlock + partner * 16 + within;
            const float a = __bfloat162float(slab[g]);
            const float b = __bfloat162float(slab[pg]);
            slab[g]  = __float2bfloat16_rn(a + b);        // low window: x + y
            slab[pg] = __float2bfloat16_rn(a - b);        // high window: x - y
        }
        __syncthreads();
    }

    // (d) Write the slab back.
    {
        const int tid = static_cast<int>(threadIdx.x);
        constexpr int kStores = kSlabElements / kThreads;  // 64 bf16 per thread
#pragma unroll
        for (int i = 0; i < kStores; ++i) {
            const int idx = i * kThreads + tid;
            auto* destination =
                reinterpret_cast<std::uint16_t*>(slab) + idx * 2;
            // Identity copy from the slab; the caller streams this to global memory.
            (void)destination;
        }
    }
}

} // namespace ternary_fwt_mma

// Standalone large-T FWT route: one CTA per 16-token slab of one 1024 block.
// `x` is [tokens, InputRows] BF16, `out` the same shape, `signs` one sign vector of InputRows
// binary32 values.
template <std::int32_t InputRows>
__global__ __launch_bounds__(ternary_fwt_mma::kThreads) void ternary_fwt_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const float* __restrict__ signs,
    __nv_bfloat16* __restrict__ out, std::int32_t tokens) {
    using namespace ternary_fwt_mma;
    static_assert((InputRows % kTernaryFwtBlock) == 0);
    constexpr int kBlocksPerRow = InputRows / kTernaryFwtBlock;

    extern __shared__ std::uint16_t slab_raw[];
    auto* slab = reinterpret_cast<__nv_bfloat16*>(slab_raw);

    const int token0 = static_cast<int>(blockIdx.x) * kSlabTokens;
    const int k_block = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);

    // (a) Stage the slab: 16 tokens x 1024 features of BF16 (32KB), 16-byte cp.async vectors.
    {
        constexpr int kSixteenByteLoads = kSlabBf16Bytes / 16;
        constexpr int kLoadsPerThread   = kSixteenByteLoads / kThreads;
        const auto* source = reinterpret_cast<const std::uint16_t*>(x) +
                            static_cast<std::int64_t>(token0) * (InputRows * 2);
#pragma unroll
        for (int i = 0; i < kLoadsPerThread; ++i) {
            const int task = i * kThreads + tid;
            const int row  = task / (kTernaryFwtBlock / 8);  // 128 16-byte vectors per row
            const int vec  = task - row * (kTernaryFwtBlock / 8);
            const int active = (token0 + row) < tokens ? 16 : 0;
            cp_async_zfill<16, Cache::cg>(slab_raw + task * 16, source + row * (kTernaryFwtBlock * 2) + vec * 16,
                                          active);
        }
        cp_commit();
        cp_wait<0>();
    }
    __syncthreads();

    ternary_fwt_mma_slab<InputRows>(slab, signs, k_block);
    __syncthreads();

    {
        constexpr int kStores = kSlabElements / kThreads;  // 64 bf16 per thread
        auto* destination = reinterpret_cast<std::uint16_t*>(out) +
                            static_cast<std::int64_t>(token0) * (InputRows * 2) +
                            k_block * kTernaryFwtBlock * 2;
#pragma unroll
        for (int i = 0; i < kStores; ++i) {
            const int idx = i * kThreads + tid;
            const int row = idx / kTernaryFwtBlock;
            const int feature = idx - row * kTernaryFwtBlock;
            if (token0 + row < tokens) {
                destination[row * (InputRows * 2) + feature * 2] =
                    load_vec<std::uint16_t>(slab_raw + idx * 2);
            }
        }
    }
}

} // namespace ninfer::ops::detail
