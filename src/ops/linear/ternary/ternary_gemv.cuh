#pragma once

#include "ops/linear/ternary/ternary_config.h"
#include "ops/linear/ternary/ternary_output.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/common/fwt.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// T=1 decode route: one 256-thread CTA owns Schedule::kRowsPerCta weight rows and one input
// token. Each 1024-wide K block is staged as `x * sign * (1/32)` into the shared 1024-float
// transform row (the pinned fused load of the butterfly route), run through
// ternary_fwt_butterfly(), and multiplied against the exact FP32 trit decode (code - 1) *
// binary16 scale, FP32 accumulated per row.
//
// The packed weights are never materialized: each lane owns one code byte of its row and
// decodes its 4 trit slots in registers.

template <class Geometry, class Schedule>
struct TernaryGemvSharedStorage {
    alignas(16) float fwt_row[kTernaryFwtBlock];
};

// Stages one 1024 K block of the single input token into `fwt_row` with the fused sign flip
// and 1/sqrt(1024) scale. All 256 threads participate.
template <class Geometry, class Schedule>
__device__ __forceinline__ void stage_ternary_fwt_row(const __nv_bfloat16* __restrict__ x,
                                                       const float* __restrict__ signs,
                                                       int k_block, float* __restrict__ fwt_row) {
    const int tid = static_cast<int>(threadIdx.x);
    const int local0 = tid * (kTernaryFwtBlock / 256);
    const auto* source = x + k_block * kTernaryFwtBlock + local0;
    const float* sign_block = signs + k_block * kTernaryFwtBlock;
    // Each thread owns 4 consecutive features (two 32-bit loads); 256 threads x 4 covers
    // the 1024-element row.
#pragma unroll
    for (int i = 0; i < kTernaryFwtBlock / 512; ++i) {
        const int local = local0 + 2 * i;
        const std::uint32_t bits = load_vec<std::uint32_t>(source + 2 * i);
        const float2 value       = bf16x2_bits_to_float2(bits);
        fwt_row[local]           = value.x * sign_block[local] * kTernaryFwtScale;
        fwt_row[local + 1]       = value.y * sign_block[local + 1] * kTernaryFwtScale;
    }
}

// Dot product of one weight row against the transformed row: 8 groups of 128 trit slots.
// `row_blocks` points at the row's 34-byte blocks for this K block (256 bytes). Lane l owns
// code byte l (4 trit slots 4l..4l+3) of every group; the 4 slot products fold into
// Schedule::kAccumulatorChains chains (slot index mod chain count, a power of two), reduced
// by the caller.
template <class Geometry, class Schedule>
__device__ __forceinline__ void accumulate_ternary_row(const std::uint8_t* __restrict__ row_blocks,
                                                        float* __restrict__ fwt_row, int lane,
                                                        float* accumulators) {
    static_assert((Schedule::kAccumulatorChains & (Schedule::kAccumulatorChains - 1)) == 0);
#pragma unroll
    for (int group = 0; group < Geometry::kGroupsPerKBlock; ++group) {
        const std::uint8_t* block = row_blocks + group * kTernaryPq2BlockBytes;
        const std::uint8_t byte   = block[2 + lane];
        const std::uint16_t scale_bits = load_vec<std::uint16_t>(block);
        const float scale              = __half2float(__half(__half_raw{scale_bits}));
        const int feature0             = group * kTernaryPq2GroupSize + lane * 4;
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            const int code   = (byte >> (2 * slot)) & 3;
            const float weight = static_cast<float>(code - 1) * scale;
            const int chain = slot & (Schedule::kAccumulatorChains - 1);
            accumulators[chain] = fmaf(weight, fwt_row[feature0 + slot], accumulators[chain]);
        }
    }
}

template <class Geometry, class Schedule, class Epilogue, class Output>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void
ternary_gemv_kernel(const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ blocks,
                    const float* __restrict__ signs, Epilogue epilogue, Output output) {
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    static_assert((Schedule::kRowsPerCta % 4) == 0);

    __shared__ TernaryGemvSharedStorage<Geometry, Schedule> shared;
    const int cta_row0 = static_cast<int>(blockIdx.x) * Schedule::kRowsPerCta;
    const int lane     = static_cast<int>(threadIdx.x) & 31;

    float accumulators[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};

#pragma unroll 1
    for (int k_block = 0; k_block < Geometry::kKBlocks; ++k_block) {
        stage_ternary_fwt_row<Geometry, Schedule>(x, signs, k_block, shared.fwt_row);
        __syncthreads();
        ternary_fwt_butterfly(shared.fwt_row);
        __syncthreads();
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const std::int64_t row = cta_row0 + (static_cast<int>(threadIdx.x) >> 5) *
                                           Schedule::kRowsPerWarp +
                                       local_row;
            const std::uint8_t* row_blocks =
                blocks + row * Geometry::kGroupsPerRow * kTernaryPq2BlockBytes +
                k_block * Geometry::kGroupsPerKBlock * kTernaryPq2BlockBytes;
            accumulate_ternary_row<Geometry, Schedule>(row_blocks, shared.fwt_row, lane,
                                                       accumulators[local_row]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
        float total = 0.0F;
#pragma unroll
        for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
            total += accumulators[local_row][chain];
        }
        total = warp_reduce_sum(total);
        if (lane == 0) {
            const int parent_row = cta_row0 + (static_cast<int>(threadIdx.x) >> 5) *
                                        Schedule::kRowsPerWarp +
                                    local_row;
            output.store(parent_row, 0, epilogue.apply(parent_row, 0, total));
        }
    }
}

} // namespace ninfer::ops::detail
