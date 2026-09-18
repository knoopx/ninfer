#pragma once

#include "ops/linear/ternary/ternary_gemv.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Small-T SIMT route: one 256-thread CTA owns Schedule::kRowsPerCta weight rows and up to
// Capacity tokens. The 1024-wide transform row is shared: each token is staged, run through
// ternary_fwt_butterfly(), and multiplied against the exact FP32 trit decode before the next
// token reuses the row. Same numerics as the GEMV route; the weights stay packed.

template <class Geometry, class Schedule>
struct TernarySimtSharedStorage {
    alignas(16) float fwt_row[kTernaryFwtBlock];
};

// Stages one token's 1024 K block into `fwt_row` with the fused sign flip and 1/sqrt(1024)
// scale. Ragged token tiles read from the last live token so the padded lanes are defined.
template <class Geometry, class Schedule>
__device__ __forceinline__ void stage_ternary_simt_token(const __nv_bfloat16* __restrict__ x,
                                                          const float* __restrict__ signs,
                                                          int k_block, int token,
                                                          float* __restrict__ fwt_row) {
    const int tid     = static_cast<int>(threadIdx.x);
    const int local0  = tid * (kTernaryFwtBlock / 256);
    const auto* source = x + static_cast<std::int64_t>(token) * Geometry::kInputRows +
                     k_block * kTernaryFwtBlock + local0;
    const float* sign_block = signs + k_block * kTernaryFwtBlock;
#pragma unroll
    for (int i = 0; i < kTernaryFwtBlock / 512; ++i) {
        const int local = local0 + 2 * i;
        const std::uint32_t bits = load_vec<std::uint32_t>(source + 2 * i);
        const float2 value       = bf16x2_bits_to_float2(bits);
        fwt_row[local]           = value.x * sign_block[local] * kTernaryFwtScale;
        fwt_row[local + 1]       = value.y * sign_block[local + 1] * kTernaryFwtScale;
    }
}

template <class Geometry, class Schedule>
__device__ __forceinline__ void compute_ternary_simt_rows(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ blocks,
    const float* __restrict__ signs, TernarySimtSharedStorage<Geometry, Schedule>& shared,
    int cta_row0, int token0, int live_tokens,
    float (&accumulators)[Schedule::kRowsPerWarp][Schedule::kTokenTile]) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;

#pragma unroll 1
    for (int k_block = 0; k_block < Geometry::kKBlocks; ++k_block) {
        // The weights are token independent: hoist the per-group byte and scale decode out of
        // the token loop.
        std::uint8_t code_bytes[Schedule::kRowsPerWarp][Geometry::kGroupsPerKBlock];
        float group_scale[Schedule::kRowsPerWarp][Geometry::kGroupsPerKBlock];
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const std::uint8_t* row_blocks =
                blocks +
                (cta_row0 + warp * Schedule::kRowsPerWarp + local_row) *
                    Geometry::kGroupsPerRow * kTernaryPq2BlockBytes +
                k_block * Geometry::kGroupsPerKBlock * kTernaryPq2BlockBytes;
#pragma unroll
            for (int group = 0; group < Geometry::kGroupsPerKBlock; ++group) {
                const std::uint8_t* block = row_blocks + group * kTernaryPq2BlockBytes;
                code_bytes[local_row][group] = block[2 + lane];
                const std::uint16_t scale_bits = load_vec<std::uint16_t>(block);
                group_scale[local_row][group] = __half2float(__half(__half_raw{scale_bits}));
            }
        }
#pragma unroll
        for (int token = 0; token < Schedule::kTokenTile; ++token) {
            const int live = token0 + token;
            if (live >= live_tokens) { break; }
            stage_ternary_simt_token<Geometry, Schedule>(x, signs, k_block, live, shared.fwt_row);
            __syncthreads();
            ternary_fwt_butterfly(shared.fwt_row);
            __syncthreads();
            const float* fwt_row = shared.fwt_row;
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
#pragma unroll
                for (int group = 0; group < Geometry::kGroupsPerKBlock; ++group) {
                    const int feature0 = group * kTernaryPq2GroupSize + lane * 4;
                    float partial      = 0.0F;
#pragma unroll
                    for (int slot = 0; slot < 4; ++slot) {
                        const int code = (code_bytes[local_row][group] >> (2 * slot)) & 3;
                        partial = fmaf(static_cast<float>(code - 1) *
                                           group_scale[local_row][group],
                                       fwt_row[feature0 + slot], partial);
                    }
                    accumulators[local_row][token] += partial;
                }
            }
            __syncthreads();
        }
    }
}

template <class Geometry, int Capacity, class Schedule, class Epilogue, class Output,
          bool RuntimeColumns = false>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void
ternary_simt_kernel(const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ blocks,
                    const float* __restrict__ signs, Epilogue epilogue, Output output,
                    int columns = Capacity) {
    static_assert(Capacity == Schedule::kTokenTile);
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    const int live_tokens = RuntimeColumns ? columns : Capacity;

    __shared__ TernarySimtSharedStorage<Geometry, Schedule> shared;
    const int cta_row0 = static_cast<int>(blockIdx.x) * Schedule::kRowsPerCta;
    const int token0   = static_cast<int>(blockIdx.y) * Capacity;

    float accumulators[Schedule::kRowsPerWarp][Schedule::kTokenTile] = {};
    compute_ternary_simt_rows<Geometry, Schedule>(x, blocks, signs, shared, cta_row0, token0,
                                                  live_tokens, accumulators);

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
#pragma unroll
        for (int token = 0; token < Schedule::kTokenTile; ++token) {
            const int live = token0 + token;
            if (live < live_tokens) {
                float total = accumulators[local_row][token];
                total       = warp_reduce_sum(total);
                if (lane == 0) {
                    const int parent_row =
                        cta_row0 + warp * Schedule::kRowsPerWarp + local_row;
                    output.store(parent_row, live, epilogue.apply(parent_row, live, total));
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
