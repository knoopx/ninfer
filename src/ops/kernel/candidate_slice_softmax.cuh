#pragma once

// Implements: include/ninfer/ops/candidate_slice_softmax.h
// Match: contiguous FP32 [N,C] logits, I32 [N,C] groups, and FP32 [N,C] out.
// Algorithm assumptions: one 256-thread CTA per row; the 256-slot shared pool covers C<=256.

#include "ops/common/warp.cuh"

#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCandidateSliceBlock = 256;

template <int BlockSize>
__device__ __forceinline__ float candidate_slice_block_max(float value, float* warp_maxima,
                                                            float* result) {
    static_assert(BlockSize >= kWarpSize && BlockSize <= 1024);
    static_assert((BlockSize & (BlockSize - 1)) == 0);
    constexpr int kWarps = BlockSize / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    value          = warp_max(value);
    if (lane == 0) { warp_maxima[warp] = value; }
    __syncthreads();

    if (warp == 0) {
        value = lane < kWarps ? warp_maxima[lane] : -CUDART_INF_F;
        value = warp_max(value);
        if (lane == 0) { *result = value; }
    }
    __syncthreads();
    return *result;
}

__device__ __forceinline__ void atomic_max_float(float* address, float value) {
    auto* bits              = reinterpret_cast<std::uint32_t*>(address);
    std::uint32_t observed  = *bits;
    while (true) {
        if (__uint_as_float(observed) >= value) { break; }
        observed = atomicCAS(bits, observed, __float_as_uint(value));
    }
}

__launch_bounds__(kCandidateSliceBlock) __global__
void candidate_slice_softmax_kernel(const float* logits, const std::int32_t* groups, float* out,
                                    std::int32_t columns, float temperature) {
    constexpr int BlockSize = kCandidateSliceBlock;
    const std::int32_t row          = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t column       = static_cast<std::int32_t>(threadIdx.x);
    const bool active               = column < columns;
    const float* row_logits         = logits + static_cast<std::int64_t>(row) * columns;
    const std::int32_t* row_groups  = groups + static_cast<std::int64_t>(row) * columns;

    __shared__ float pool_slots[BlockSize];
    __shared__ float warp_maxima[BlockSize / kWarpSize];
    __shared__ float warp_sums[BlockSize / kWarpSize];
    __shared__ float result;
    __shared__ float sum_result;

    const float value         = active ? row_logits[column] : 0.0f;
    const std::int32_t group  = active ? row_groups[column] : -1;
    pool_slots[column]        = -CUDART_INF_F;
    __syncthreads();

    // Detect row pooling: block-max over the group indicator (idle threads contribute -INF).
    const float indicator   = active ? (group >= 0 ? 1.0f : 0.0f) : -CUDART_INF_F;
    const float pooling_max = candidate_slice_block_max<BlockSize>(indicator, warp_maxima,
                                                                   &result);
    const bool pooling_active = pooling_max > 0.0f;

    float score;
    if (pooling_active) {
        if (group >= 0) { atomic_max_float(&pool_slots[group], value); }
        __syncthreads();
        score = (group >= 0) ? pool_slots[group] : value;
    } else {
        score = value;
    }

    const float T         = fmaxf(temperature, 1e-6f);
    const float z         = score / T;
    const float z_value   = active ? z : -CUDART_INF_F;
    const float row_max   = candidate_slice_block_max<BlockSize>(z_value, warp_maxima, &result);

    // When row_max is -inf every active candidate's logit was -inf; the
    // subtraction -inf - (-inf) would yield NaN.  Emit uniform 1.0 so that
    // after the sum-division each active column receives probability 1/C.
    float exp_value = 0.0f;
    if (active) { exp_value = (row_max == -CUDART_INF_F) ? 1.0f : expf(z - row_max); }
    const float row_sum_partial = block_reduce_sum<BlockSize>(exp_value, warp_sums);
    // block_reduce_sum returns the full block sum only to lane 0 (the other threads hold a
    // partial sum or zero); broadcast it so every active thread divides by the same row_sum.
    if (threadIdx.x == 0) { sum_result = row_sum_partial; }
    __syncthreads();
    const float row_sum = sum_result;

    if (active) { out[static_cast<std::int64_t>(row) * columns + column] = exp_value / row_sum; }
}

} // namespace ninfer::ops
